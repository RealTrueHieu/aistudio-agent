package main

// handlers.go - OpenAI + Anthropic compatible HTTP server with
// low-latency chunked SSE relaying.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"
)

type Server struct {
	Up  *Upstream
	Cfg *Config
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/chat/completions", s.handleChat)
	mux.HandleFunc("/v1/v1/chat/completions", s.handleChat) // clients that append /v1 themselves
	mux.HandleFunc("/v1/messages", s.handleAnthropic)
	mux.HandleFunc("/messages", s.handleAnthropic)
	mux.HandleFunc("/v1/v1/messages", s.handleAnthropic)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/models", s.handleModels)
	mux.HandleFunc("/v1/v1/models", s.handleModels)
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/status", s.handleHealth)
	mux.HandleFunc("/", s.handleRoot)
	return mux
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", "*")
		h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		h.Set("Access-Control-Allow-Headers", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// statusRecorder captures the response status and keeps Flush working for SSE.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// withLogging logs every request: method, path, status, duration.
func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: 200}
		next.ServeHTTP(rec, r)
		log.Printf("[gai] %s %s -> %d (%.0fms)", r.Method, r.URL.Path, rec.status,
			float64(time.Since(start).Microseconds())/1000)
	})
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

func errBody(message, typ string) map[string]any {
	return map[string]any{"error": map[string]any{"message": message, "type": typ}}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, _ := json.Marshal(v)
	h := w.Header()
	h.Set("Content-Type", "application/json; charset=utf-8")
	h.Set("Content-Length", strconv.Itoa(len(b)))
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

// writeUpstreamError forwards an upstream error answer to the client.
func writeUpstreamError(w http.ResponseWriter, err error) {
	var ue *UpstreamError
	if errors.As(err, &ue) {
		ct := "text/plain; charset=utf-8"
		if json.Valid([]byte(ue.Body)) {
			ct = "application/json; charset=utf-8"
		}
		h := w.Header()
		h.Set("Content-Type", ct)
		h.Set("Content-Length", strconv.Itoa(len(ue.Body)))
		status := ue.Status
		if status >= 500 || status == 429 {
			status = http.StatusBadGateway
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(ue.Body))
		return
	}
	writeJSON(w, http.StatusBadGateway, errBody("All upstream endpoints failed: "+err.Error(), "server_error"))
}

// ---------------------------------------------------------------------------
// GET /
// ---------------------------------------------------------------------------

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		writeJSON(w, http.StatusNotFound, errBody("Not found: "+r.URL.Path, "invalid_request_error"))
		return
	}
	writeJSON(w, 200, map[string]any{
		"name":     "GAI - Google AI Studio Gateway",
		"version":  Version,
		"endpoints": []string{
			"/v1/chat/completions", "/v1/messages", "/v1/models", "/healthz",
		},
		"docs": "gai --help",
	})
}

// ---------------------------------------------------------------------------
// GET /v1/models  (live catalog from upstream, augmented with efforts)
// ---------------------------------------------------------------------------

// fallback catalog served only if the upstream is unreachable and never cached.
var fallbackModels = []map[string]any{
	{"id": "ais/gemini-4-argon", "object": "model", "owned_by": "ais", "context_length": 1048576, "max_completion_tokens": 65536,
		"supported_efforts": []string{"low", "medium", "high"}, "capabilities": map[string]any{"thinking": true, "tools": true, "vision": true}},
	{"id": "ais/gemini-3.8-flash", "object": "model", "owned_by": "ais", "context_length": 1048576, "max_completion_tokens": 65536,
		"supported_efforts": []string{"low", "medium", "high"}, "capabilities": map[string]any{"thinking": true, "tools": true, "vision": true}},
	{"id": "ais/gemini-2.5-pro", "object": "model", "owned_by": "ais", "context_length": 1048576, "max_completion_tokens": 65536,
		"supported_efforts": []string{"none", "low", "medium", "high"}, "capabilities": map[string]any{"thinking": true, "tools": true, "vision": true}},
	{"id": "ais/gemini-2.5-flash", "object": "model", "owned_by": "ais", "context_length": 1048576, "max_completion_tokens": 65536,
		"supported_efforts": []string{"none", "low", "medium", "high"}, "capabilities": map[string]any{"thinking": true, "tools": true, "vision": true}},
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	models, fetchedAt := s.Up.ModelsCached()
	if time.Since(fetchedAt) > 30*time.Second || len(models) == 0 {
		ctx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
		_ = s.Up.FetchModels(ctx)
		cancel()
		models, _ = s.Up.ModelsCached()
	}
	if len(models) == 0 {
		models = fallbackModels
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": models})
}

// ---------------------------------------------------------------------------
// GET /healthz
// ---------------------------------------------------------------------------

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	models, _ := s.Up.ModelsCached()
	gemini := "off"
	if s.Up.GeminiKey != "" {
		gemini = "on"
	}
	writeJSON(w, 200, map[string]any{
		"status":         "healthy",
		"version":        Version,
		"uptime_seconds": int(time.Since(startedAt).Seconds()),
		"requests":       atomic.LoadInt64(&s.Up.Requests),
		"hedged_fired":   atomic.LoadInt64(&s.Up.HedgeFired),
		"models_cached":  len(models),
		"upstream":       map[string]any{"gateway": s.Up.Gateway, "gemini_fallback": gemini},
		"config": map[string]any{
			"default_effort": s.Up.DefaultEffort,
			"hedge_ms":        s.Up.HedgeMS,
			"keepalive_sec":   s.Cfg.KeepaliveSec,
			"models_filter":   s.Up.ModelsFilter,
		},
	})
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions  (OpenAI)
// ---------------------------------------------------------------------------

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errBody("POST required", "invalid_request_error"))
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, errBody("Invalid body", "invalid_request_error"))
		return
	}
	var req map[string]any
	if json.Unmarshal(body, &req) != nil {
		writeJSON(w, http.StatusBadRequest, errBody("Invalid JSON payload", "invalid_request_error"))
		return
	}
	stream, _ := req["stream"].(bool)
	body = s.Up.InjectDefaultEffort(body)

	resp, err := s.Up.ChatCompletions(r.Context(), body, stream)
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	defer resp.Body.Close()

	if stream {
		relaySSE(w, resp.Body)
		return
	}
	// Non-stream: pipe the upstream JSON straight through (zero parsing).
	h := w.Header()
	h.Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = io.Copy(w, resp.Body)
}

// relaySSE streams upstream SSE events to the client with immediate flushes.
// Go's net/http writes HTTP/1.1 chunked automatically and sets TCP_NODELAY.
func relaySSE(w http.ResponseWriter, body io.Reader) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, canFlush := w.(http.Flusher)

	done := false
	forEachSSEEvent(body, func(ev []byte) {
		if bytes.Contains(ev, []byte("[DONE]")) {
			done = true
		}
		_, _ = w.Write(ev)
		if canFlush {
			flusher.Flush()
		}
	})
	if !done {
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if canFlush {
			flusher.Flush()
		}
	}
}

// ---------------------------------------------------------------------------
// POST /v1/messages  (Anthropic -> translated to OpenAI -> upstream)
// ---------------------------------------------------------------------------

func (s *Server) handleAnthropic(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errBody("POST required", "invalid_request_error"))
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, errBody("Invalid body", "invalid_request_error"))
		return
	}
	var req map[string]any
	if json.Unmarshal(body, &req) != nil {
		writeJSON(w, http.StatusBadRequest, errBody("Invalid JSON payload", "invalid_request_error"))
		return
	}
	stream, _ := req["stream"].(bool)
	modelName := asString(req["model"])

	openaiReq := anthropicToOpenai(req)
	if stream {
		openaiReq["stream"] = true
	}
	ob, _ := json.Marshal(openaiReq)
	ob = s.Up.InjectDefaultEffort(ob)

	resp, err := s.Up.ChatCompletions(r.Context(), ob, stream)
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	defer resp.Body.Close()

	if stream {
		s.relayAnthropicStream(w, resp.Body, modelName)
		return
	}
	data, _ := io.ReadAll(resp.Body)
	var obj map[string]any
	_ = json.Unmarshal(data, &obj)
	writeJSON(w, 200, openaiToAnthropicResponse(obj, modelName))
}

func (s *Server) relayAnthropicStream(w http.ResponseWriter, body io.Reader, modelName string) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, canFlush := w.(http.Flusher)

	tr := newAnthropicStreamTranslator(modelName)
	forEachSSEEvent(body, func(ev []byte) {
		for _, li := range bytes.Split(ev, []byte("\n")) {
			line := bytes.TrimSpace(li)
			if !bytes.HasPrefix(line, []byte("data:")) {
				continue
			}
			payload := bytes.TrimSpace(line[5:])
			if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
				continue
			}
			var obj map[string]any
			if json.Unmarshal(payload, &obj) != nil {
				continue
			}
			if out := tr.HandleChunk(obj); len(out) > 0 {
				_, _ = w.Write(out)
				if canFlush {
					flusher.Flush()
				}
			}
		}
	})
	if tail := tr.Finish(); len(tail) > 0 {
		_, _ = w.Write(tail)
		if canFlush {
			flusher.Flush()
		}
	}
}
