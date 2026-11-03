package main

// upstream.go - pooled keep-alive HTTP client with hedged requests,
// live model catalog caching, and the effort-injection policy.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// UpstreamError carries an HTTP >= 400 answer from an upstream endpoint.
type UpstreamError struct {
	Status int
	Body   string
	URL    string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("HTTP %d from %s: %s", e.Status, e.URL, truncate(e.Body, 300))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// Fields that count as explicit reasoning control by the client.
var reasoningControlFields = []string{"reasoning_effort", "max_thinking_tokens", "thinking", "reasoning"}

type Upstream struct {
	Gateway       string
	APIKey        string
	GeminiKey     string
	HedgeMS       int
	DefaultEffort string
	Timeout       time.Duration
	HTTP          *http.Client
	ModelsFilter  []string // e.g. ["ais","ag"]; nil = show everything

	Requests     int64
	HedgeFired   int64
	UpstreamErrs int64

	mu       sync.RWMutex
	models   []map[string]any
	modelsAt time.Time
}

func NewUpstream(cfg *Config) *Upstream {
	transport := &http.Transport{
		MaxIdleConns:          32,
		MaxIdleConnsPerHost:   8,
		IdleConnTimeout:       120 * time.Second,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: time.Duration(cfg.TimeoutSec) * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ForceAttemptHTTP2:     true,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	}
	return &Upstream{
		Gateway:       cfg.Gateway,
		APIKey:        cfg.APIKey,
		GeminiKey:     cfg.GeminiKey,
		HedgeMS:       cfg.HedgeMS,
		DefaultEffort: effortOrDefault(cfg.DefaultEffort),
		Timeout:       time.Duration(cfg.TimeoutSec) * time.Second,
		ModelsFilter:  cfg.ModelsFilter,
		HTTP:          &http.Client{Transport: transport},
	}
}

func effortOrDefault(e string) string {
	if e == "" || e == "auto" {
		return ""
	}
	return e
}

// ---------------------------------------------------------------------------
// Core request path (keep-alive pooled, TCP_NODELAY is Go default)
// ---------------------------------------------------------------------------

func (u *Upstream) doRequest(ctx context.Context, method, url string, headers map[string]string, body []byte) (*http.Response, error) {
	atomic.AddInt64(&u.Requests, 1)
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	// identity: keep SSE uncompressed (Go's transparent gzip buffers streams)
	req.Header.Set("Accept-Encoding", "identity")

	resp, err := u.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
		resp.Body.Close()
		return nil, &UpstreamError{Status: resp.StatusCode, Body: string(b), URL: url}
	}
	return resp, nil
}

// doHedged: fire the request; if hedge_ms elapses without response headers,
// fire a duplicate and take whichever answers first (the loser gets closed
// by a janitor goroutine when it lands).
func (u *Upstream) doHedged(ctx context.Context, method, url string, headers map[string]string, body []byte) (*http.Response, error) {
	type result struct {
		resp *http.Response
		err  error
	}
	ch := make(chan result, 2)
	attempt := func() {
		resp, err := u.doRequest(ctx, method, url, headers, body)
		ch <- result{resp, err}
	}
	go attempt()
	timer := time.NewTimer(time.Duration(u.HedgeMS) * time.Millisecond)
	defer timer.Stop()
	select {
	case r := <-ch:
		return r.resp, r.err
	case <-timer.C:
		atomic.AddInt64(&u.HedgeFired, 1)
		go attempt()
		r := <-ch // first to land wins
		go func() { // janitor: drain & close the loser whenever it lands
			if lr := <-ch; lr.resp != nil {
				_, _ = io.Copy(io.Discard, io.LimitReader(lr.resp.Body, 1<<20))
				lr.resp.Body.Close()
			}
		}()
		return r.resp, r.err
	}
}

func (u *Upstream) forward(ctx context.Context, method, url string, headers map[string]string, body []byte) (*http.Response, error) {
	if u.HedgeMS > 0 {
		return u.doHedged(ctx, method, url, headers, body)
	}
	return u.doRequest(ctx, method, url, headers, body)
}

// ChatCompletions routes an OpenAI-format request: cloud gateway first
// (with transparent retries for transient edge blocks), official Google API
// (native AI Studio payloads) as final fallback. The relay is the primary
// path on purpose: AI Studio web-session quota is far more generous than
// API-key quota.
func (u *Upstream) ChatCompletions(ctx context.Context, body []byte, stream bool) (*http.Response, error) {
	headers := map[string]string{
		"Authorization": "Bearer " + u.APIKey,
		"Content-Type":  "application/json",
	}
	if stream {
		headers["Accept"] = "text/event-stream"
	}
	var resp *http.Response
	var err error
	// The gateway relays to the real AI Studio backend, which INTERMITTENTLY
	// answers 403 "The caller does not have permission" (transient session /
	// permission rejections that clear within seconds). IDEs classify 403 as
	// fatal auth_failed, so GAI rides these out with patient retries.
	const maxAttempts = 5
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		resp, err = u.forward(ctx, "POST", u.Gateway+"/chat/completions", headers, body)
		if err == nil {
			return resp, nil
		}
		var ue *UpstreamError
		if !errors.As(err, &ue) || !retryableStatus(ue.Status) {
			break // network error or permanent 4xx: do not retry
		}
		if attempt == maxAttempts {
			break
		}
		backoff := time.Duration(400*(1<<(attempt-1))) * time.Millisecond // 0.4s, 0.8s, 1.6s, 3.2s
		log.Printf("[gai] upstream HTTP %d (attempt %d/%d) - retrying in %v",
			ue.Status, attempt, maxAttempts, backoff)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
		}
	}
	if u.GeminiKey != "" {
		if resp2, err2 := u.googleGenerate(ctx, body, stream); err2 == nil {
			return resp2, nil
		}
	}
	return nil, err
}

// retryableStatus: transient upstream conditions worth a transparent retry.
func retryableStatus(status int) bool {
	return status == 403 || status == 408 || status == 425 || status == 429 || status >= 500
}

// ---------------------------------------------------------------------------
// Effort injection policy
// ---------------------------------------------------------------------------

// InjectDefaultEffort returns the (possibly modified) request body bytes.
// IDE clients (Zed, Cursor, Claude Code) rarely send reasoning fields, so
// without this every request burns thinking tokens. Explicit fields win.
func (u *Upstream) InjectDefaultEffort(body []byte) []byte {
	if u.DefaultEffort == "" {
		return body
	}
	for _, f := range reasoningControlFields {
		if bytes.Contains(body, []byte(`"`+f+`"`)) {
			return body
		}
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}
	m["reasoning_effort"] = u.DefaultEffort
	out, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return out
}

// ---------------------------------------------------------------------------
// Live model catalog (fetched from upstream, augmented with efforts)
// ---------------------------------------------------------------------------

func supportedEfforts(m map[string]any) []string {
	caps, _ := m["capabilities"].(map[string]any)
	// Gateway owners expose reasoning under either "thinking" or "reasoning".
	thinking := asBool(caps["thinking"]) || asBool(caps["reasoning"])
	canDisable := asBool(caps["thinkingCanDisable"])
	effortSup := asBool(caps["thinkingEffortSupported"])
	efforts := []string{"none"}
	if thinking || effortSup {
		if !canDisable {
			efforts = []string{"low", "medium", "high"}
		} else {
			efforts = append(efforts, "low", "medium", "high")
		}
	}
	return efforts
}

func (u *Upstream) FetchModels(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", u.Gateway+"/models", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+u.APIKey)
	req.Header.Set("Accept-Encoding", "identity")
	resp, err := u.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("models: HTTP %d: %s", resp.StatusCode, string(b))
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return err
	}
	var cat struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(data, &cat); err != nil {
		return err
	}
	if len(cat.Data) == 0 {
		return errors.New("empty model catalog")
	}
	// Keep only the clusters we host (prefix filter), augment with efforts.
	filtered := make([]map[string]any, 0, len(cat.Data))
	for _, m := range cat.Data {
		if !u.modelAllowed(asString(m["id"])) {
			continue
		}
		m["supported_efforts"] = supportedEfforts(m)
		filtered = append(filtered, m)
	}
	if len(filtered) == 0 {
		return errors.New("no models left after filter (check --models-filter)")
	}
	u.mu.Lock()
	u.models = filtered
	u.modelsAt = time.Now()
	u.mu.Unlock()
	return nil
}

// modelAllowed reports whether a model id passes the prefix filter.
func (u *Upstream) modelAllowed(id string) bool {
	if len(u.ModelsFilter) == 0 {
		return true
	}
	for _, pre := range u.ModelsFilter {
		if pre == "*" || pre == "all" || id == pre || strings.HasPrefix(id, pre+"/") {
			return true
		}
	}
	return false
}

func (u *Upstream) ModelsCached() ([]map[string]any, time.Time) {
	u.mu.RLock()
	defer u.mu.RUnlock()
	return u.models, u.modelsAt
}

// StartKeepalive keeps one upstream connection permanently hot and refreshes
// the model catalog at the same time. Self-heals when the upstream closes it.
func (u *Upstream) StartKeepalive(interval time.Duration) {
	go func() {
		for {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			_ = u.FetchModels(ctx)
			cancel()
			time.Sleep(interval)
		}
	}()
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

// forEachSSEEvent calls fn with each complete SSE event (raw bytes, ending
// with the blank line) the moment it arrives from upstream.
func forEachSSEEvent(body io.Reader, fn func(event []byte)) {
	reader := newLineReader(body)
	var pending []byte
	for {
		line, err := reader()
		if len(line) > 0 {
			pending = append(pending, line...)
			if isBlankLine(line) {
				fn(pending)
				pending = pending[:0]
			}
		}
		if err != nil {
			if len(pending) > 0 {
				fn(pending)
			}
			return
		}
	}
}

func isBlankLine(line []byte) bool {
	return len(bytes.TrimSpace(line)) == 0
}

// newLineReader returns a closure that reads one line (incl. '\n') per call.
func newLineReader(body io.Reader) func() ([]byte, error) {
	buf := make([]byte, 0, 64*1024)
	chunk := make([]byte, 32*1024)
	eof := false
	return func() ([]byte, error) {
		for {
			if i := bytes.IndexByte(buf, '\n'); i >= 0 {
				line := append([]byte(nil), buf[:i+1]...)
				buf = buf[i+1:]
				return line, nil
			}
			if eof {
				if len(buf) > 0 {
					line := buf
					buf = nil
					return line, nil
				}
				return nil, io.EOF
			}
			n, err := body.Read(chunk)
			if n > 0 {
				buf = append(buf, chunk[:n]...)
			}
			if err != nil {
				eof = true
			}
		}
	}
}
