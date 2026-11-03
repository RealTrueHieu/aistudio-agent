package main

// GAI - Google AI Studio Gateway (Go binary edition)
//
// Modes:
//   gai                  -> start the OpenAI/Anthropic compatible gateway server
//   gai "prompt"         -> one-shot CLI query (streaming)
//   gai chat             -> interactive multi-turn REPL (warm connection + context)
//   gai models           -> print the live model catalog from the upstream gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const Version = "3.0.0"

const DefaultGatewayURL = "http://localhost:8080/v1"
const DefaultAPIKey = ""
const DefaultModel = "ais/gemini-4-argon"

// ---------------------------------------------------------------------------
// Config & flag parsing
// ---------------------------------------------------------------------------

type Config struct {
	Mode          string // serve | oneshot | chat | models
	Prompt        string
	Host          string
	Port          int
	Model         string
	Effort        string // CLI query effort (none = fastest)
	Budget        int    // thinking budget in tokens (overrides effort)
	DefaultEffort string // server: injected when client sends no reasoning control
	HedgeMS       int
	KeepaliveSec  int
	TimeoutSec    int
	NoStream      bool
	Gateway       string
	APIKey        string
	GeminiKey     string

	ModelsFilterStr string       // raw flag value, e.g. "ais,ag"
	ModelsFilter    []string     // parsed prefixes; nil = show everything
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func parseFlags() *Config {
	c := &Config{
		Host:            "127.0.0.1",
		Port:            8417,
		Model:           DefaultModel,
		Effort:          "auto",
		DefaultEffort:   "auto",
		HedgeMS:         3000,
		KeepaliveSec:    45,
		TimeoutSec:      120,
		Gateway:         strings.TrimRight(getenv("AISTUDIO_GATEWAY", DefaultGatewayURL), "/"),
		APIKey:          getenv("AISTUDIO_API_KEY", DefaultAPIKey),
		GeminiKey:       os.Getenv("GEMINI_API_KEY"),
		ModelsFilterStr: "ais,ag",
	}
	var positional []string
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "-") {
			positional = append(positional, arg)
			continue
		}
		var key, val string
		hasVal := false
		if eq := strings.IndexByte(arg, '='); eq > 0 {
			key, val, hasVal = arg[:eq], arg[eq+1:], true
		} else {
			key = arg
		}
		needVal := func() {
			if !hasVal {
				i++
				if i < len(args) {
					val = args[i]
				}
			}
		}
		switch key {
		case "--help", "-h":
			printHelp()
			os.Exit(0)
		case "--version", "-v":
			fmt.Println("gai v" + Version)
			os.Exit(0)
		case "--serve":
			c.Mode = "serve"
		case "--chat":
			c.Mode = "chat"
		case "--models":
			c.Mode = "models"
		case "--no-stream":
			c.NoStream = true
		case "--host":
			needVal()
			c.Host = val
		case "--port", "-p":
			needVal()
			c.Port, _ = strconv.Atoi(val)
		case "--model", "-m":
			needVal()
			c.Model = val
		case "--effort", "-e":
			needVal()
			c.Effort = val
		case "--budget", "-b":
			needVal()
			c.Budget, _ = strconv.Atoi(val)
		case "--default-effort":
			needVal()
			c.DefaultEffort = val
		case "--hedge-ms":
			needVal()
			c.HedgeMS, _ = strconv.Atoi(val)
		case "--keepalive":
			needVal()
			c.KeepaliveSec, _ = strconv.Atoi(val)
		case "--timeout":
			needVal()
			c.TimeoutSec, _ = strconv.Atoi(val)
		case "--gateway":
			needVal()
			c.Gateway = strings.TrimRight(val, "/")
		case "--key":
			needVal()
			c.APIKey = val
		case "--models-filter":
			needVal()
			c.ModelsFilterStr = val
		default:
			fmt.Println("unknown flag:", arg)
			printHelp()
			os.Exit(2)
		}
	}
	if c.Mode == "" {
		if len(positional) == 0 {
			c.Mode = "serve"
		} else if positional[0] == "chat" {
			c.Mode = "chat"
		} else if positional[0] == "models" {
			c.Mode = "models"
		} else {
			c.Mode = "oneshot"
			c.Prompt = strings.Join(positional, " ")
		}
	}
	c.ModelsFilter = parseModelsFilter(c.ModelsFilterStr)
	return c
}

// parseModelsFilter turns "ais,ag" into ["ais","ag"]; "all"/"*"/"" -> nil.
func parseModelsFilter(s string) []string {
	if s == "" || s == "all" || s == "*" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(p), "/"))
		if p != "" && p != "*" && p != "all" {
			out = append(out, p)
		}
	}
	return out
}

func printHelp() {
	fmt.Print(`GAI - Google AI Studio Gateway v` + Version + ` (Go binary edition)

Usage:
  gai                        start the gateway server (OpenAI + Anthropic APIs)
  gai "your prompt"          one-shot CLI query with streaming
  gai chat                   interactive multi-turn REPL
  gai models                 print the live model catalog

Server flags:
  --host <addr>              bind address       (default 127.0.0.1, 0.0.0.0 = LAN)
  --port <n>                 port               (default 8417, chosen because 8080
                             is commonly stolen by Steam and other apps; if busy
                             the next free port is used automatically)
  --models-filter <list>     only list models you host, by prefix
                             (default "ais,ag" = AI Studio + Antigravity clusters;
                              use "all" to show everything the gateway serves)
  --default-effort <x>       effort injected for client requests without
                             reasoning control: none|low|medium|high|auto
                             (default none = fastest; auto = never inject)
  --hedge-ms <n>             fire a duplicate request if upstream hasn't
                             answered within n ms; first reply wins (default 0=off)
  --keepalive <sec>          upstream keep-alive ping + catalog refresh
                             interval, 0 disables (default 45)

Query flags:
  -m, --model <id>           model id           (default ` + DefaultModel + `)
  -e, --effort <x>           reasoning effort: none|low|medium|high|auto
                             (default none = skip thinking tokens = fastest)
  -b, --budget <tokens>      explicit thinking budget (overrides --effort)
  --no-stream                wait for the full answer instead of streaming

Common flags:
  --gateway <url>           upstream gateway URL
  --key <key>               gateway API key
  --timeout <sec>           upstream timeout (default 120)

Environment:
  AISTUDIO_GATEWAY, AISTUDIO_API_KEY, GEMINI_API_KEY (optional direct fallback)

Endpoints (when serving):
  http://127.0.0.1:8080/v1/chat/completions   OpenAI Chat API
  http://127.0.0.1:8080/v1/messages           Anthropic Messages API
  http://127.0.0.1:8080/v1/models             live model catalog (with efforts)
  http://127.0.0.1:8080/healthz               health & stats
`)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asFloat(v any) float64 {
	f, _ := v.(float64)
	return f
}

func asBool(v any) bool {
	b, _ := v.(bool)
	return b
}

var startedAt = time.Now()

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	cfg := parseFlags()
	up := NewUpstream(cfg)
	switch cfg.Mode {
	case "serve":
		runServer(cfg, up)
	case "models":
		runListModels(up)
	case "chat":
		runREPL(cfg, up)
	default:
		runOneshot(cfg, up)
	}
}

// ---------------------------------------------------------------------------
// CLI: one-shot query
// ---------------------------------------------------------------------------

func runOneshot(cfg *Config, up *Upstream) {
	messages := []map[string]any{{"role": "user", "content": cfg.Prompt}}
	fmt.Printf("\n[gai] Model: %s | Effort: %s\n%s\n", cfg.Model, cfg.Effort, strings.Repeat("=", 60))
	if cfg.NoStream {
		req := map[string]any{
			"model": cfg.Model, "messages": messages,
			"max_tokens": 4096,
		}
		if cfg.Budget > 0 {
			req["max_thinking_tokens"] = cfg.Budget
		} else if cfg.Effort != "" && cfg.Effort != "auto" {
			req["reasoning_effort"] = cfg.Effort
		}
		body, _ := json.Marshal(req)
		resp, err := up.ChatCompletions(context.Background(), body, false)
		if err != nil {
			fmt.Println("[Error]", err)
			os.Exit(1)
		}
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		var obj map[string]any
		_ = json.Unmarshal(data, &obj)
		if e, ok := obj["error"].(map[string]any); ok {
			fmt.Println("[Error]", asString(e["message"]))
			os.Exit(1)
		}
		choices, _ := obj["choices"].([]any)
		if len(choices) > 0 {
			ch0, _ := choices[0].(map[string]any)
			msg, _ := ch0["message"].(map[string]any)
			fmt.Println(asString(msg["content"]))
		}
		fmt.Println(strings.Repeat("=", 60))
		return
	}
	_, _ = streamQuery(up, messages, cfg.Model, cfg.Effort, cfg.Budget)
	fmt.Println(strings.Repeat("=", 60))
}

// ---------------------------------------------------------------------------
// CLI: interactive REPL
// ---------------------------------------------------------------------------

func runREPL(cfg *Config, up *Upstream) {
	reader := bufio.NewReader(os.Stdin)
	model, effort, budget := cfg.Model, cfg.Effort, cfg.Budget
	var history []map[string]any

	fmt.Println("\n[+] GAI interactive chat - warm connection, multi-turn context.")
	fmt.Println("    Commands: /exit  /new  /model <id>  /effort <x>  /budget <tokens>")
	for {
		fmt.Print("\nYou> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("\n[+] Bye!")
			return
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		switch {
		case line == "/exit" || line == "/quit":
			fmt.Println("[+] Bye!")
			return
		case line == "/new":
			history = nil
			fmt.Println("[+] Context cleared.")
			continue
		case strings.HasPrefix(line, "/model "):
			model = strings.TrimSpace(line[7:])
			fmt.Printf("[+] Model set to %s\n", model)
			continue
		case strings.HasPrefix(line, "/effort "):
			effort = strings.TrimSpace(line[8:])
			budget = 0
			fmt.Printf("[+] Effort set to %s\n", effort)
			continue
		case strings.HasPrefix(line, "/budget "):
			budget, _ = strconv.Atoi(strings.TrimSpace(line[8:]))
			effort = ""
			fmt.Printf("[+] Thinking budget set to %d\n", budget)
			continue
		}

		history = append(history, map[string]any{"role": "user", "content": line})
		fmt.Print("Assistant> ")
		content, err := streamQuery(up, history, model, effort, budget)
		if err != nil {
			fmt.Println("\n[Error]", err)
			continue
		}
		fmt.Println()
		history = append(history, map[string]any{"role": "assistant", "content": content})
	}
}

// streamQuery runs one streaming query, prints tokens live, returns the content.
func streamQuery(up *Upstream, messages []map[string]any, model, effort string, budget int) (string, error) {
	req := map[string]any{
		"model":      model,
		"messages":   messages,
		"stream":     true,
		"max_tokens": 4096,
	}
	if budget > 0 {
		req["max_thinking_tokens"] = budget
	} else if effort != "" && effort != "auto" {
		req["reasoning_effort"] = effort
	}
	body, _ := json.Marshal(req)
	resp, err := up.ChatCompletions(context.Background(), body, true)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var sb strings.Builder
	forEachSSEEvent(resp.Body, func(ev []byte) {
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
			if e, ok := obj["error"].(map[string]any); ok {
				fmt.Printf("\n[Error] %s\n", asString(e["message"]))
				continue
			}
			choices, _ := obj["choices"].([]any)
			if len(choices) == 0 {
				continue
			}
			ch0, _ := choices[0].(map[string]any)
			delta, _ := ch0["delta"].(map[string]any)
			if rc, ok := delta["reasoning_content"].(string); ok && rc != "" {
				fmt.Printf("\n[Thinking] %s", rc)
			}
			if c, ok := delta["content"].(string); ok && c != "" {
				fmt.Print(c)
				sb.WriteString(c)
			}
		}
	})
	fmt.Println()
	return sb.String(), nil
}

// ---------------------------------------------------------------------------
// CLI: model catalog listing
// ---------------------------------------------------------------------------

func runListModels(up *Upstream) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := up.FetchModels(ctx); err != nil {
		fmt.Println("[!] Cannot fetch model catalog:", err)
		os.Exit(1)
	}
	models, _ := up.ModelsCached()

	fmt.Printf("\n GAI LIVE MODEL CATALOG  (%d models", len(models))
	if len(up.ModelsFilter) > 0 {
		fmt.Printf(", filter: %s", strings.Join(up.ModelsFilter, "/, ")+"/")
	}
	fmt.Printf(")\n %s\n\n", strings.Repeat("-", 78))
	fmt.Printf(" %-44s %9s %8s  %-23s %s\n", "MODEL", "CTX", "MAX-OUT", "EFFORTS", "CAPABILITIES")
	fmt.Printf(" %s\n", strings.Repeat("-", 78))
	for _, m := range models {
		caps, _ := m["capabilities"].(map[string]any)
		var notes []string
		if asBool(caps["vision"]) {
			notes = append(notes, "vision")
		}
		if asBool(caps["tools"]) {
			notes = append(notes, "tools")
		}
		if asBool(caps["search"]) {
			notes = append(notes, "search")
		}
		if asBool(caps["pdf"]) {
			notes = append(notes, "pdf")
		}
		if asBool(caps["thinking"]) || asBool(caps["reasoning"]) {
			notes = append(notes, "thinking")
		}
		efforts := strings.Join(toStrings(m["supported_efforts"]), ",")
		fmt.Printf(" %-44s %9d %8d  %-23s %s\n",
			asString(m["id"]), int(asFloat(m["context_length"])),
			int(asFloat(m["max_completion_tokens"])), efforts, strings.Join(notes, ","))
	}
	fmt.Printf("\n %s\n Effort levels: none | low | medium | high\n", strings.Repeat("-", 78))
	fmt.Println(" API: GET " + up.Gateway + "/models  (or via local server: /v1/models)")
}

func toStrings(v any) []string {
	arr, _ := v.([]string)
	if arr != nil {
		return arr
	}
	list, _ := v.([]any)
	out := make([]string, 0, len(list))
	for _, x := range list {
		out = append(out, asString(x))
	}
	return out
}

// ---------------------------------------------------------------------------
// Server mode
// ---------------------------------------------------------------------------

func runServer(cfg *Config, up *Upstream) {
	// Bind the port first (before the banner) so the banner shows the REAL port.
	listener := bindListener(cfg)
	if listener == nil {
		return // GAI is already running elsewhere - nothing to start
	}

	// Serve IMMEDIATELY after binding so a second `gai` instance can detect
	// this one (via /healthz) instead of treating the port as "busy".
	srv := &Server{Up: up, Cfg: cfg}
	httpServer := &http.Server{Handler: withLogging(corsMiddleware(srv.Handler()))}
	serveErr := make(chan error, 1)
	go func() { serveErr <- httpServer.Serve(listener) }()

	// Warm-up: fetch the live catalog (also primes the TLS connection so the
	// FIRST real request pays zero handshake cost).
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	modelCount := 0
	if err := up.FetchModels(ctx); err == nil {
		models, _ := up.ModelsCached()
		modelCount = len(models)
	}
	cancel()

	if cfg.KeepaliveSec > 0 {
		up.StartKeepalive(time.Duration(cfg.KeepaliveSec) * time.Second)
	}

	featured := featuredModels(up)
	geminiNote := "off"
	if up.GeminiKey != "" {
		geminiNote = "on (direct GenerateContent fallback)"
	}
	filterNote := "all (no filter)"
	if len(up.ModelsFilter) > 0 {
		filterNote = strings.Join(up.ModelsFilter, "/, ") + "/ (own clusters only)"
	}

	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("  GAI - Google AI Studio Gateway v" + Version + "  (Go binary edition)")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("  OpenAI API     =>  http://127.0.0.1:%d/v1/chat/completions\n", cfg.Port)
	fmt.Printf("  Anthropic API  =>  http://127.0.0.1:%d/v1/messages\n", cfg.Port)
	if modelCount > 0 {
		fmt.Printf("  Models         =>  http://127.0.0.1:%d/v1/models  (%d models, filter: %s)\n", cfg.Port, modelCount, filterNote)
	} else {
		fmt.Printf("  Models         =>  http://127.0.0.1:%d/v1/models  (catalog offline - retries automatically)\n", cfg.Port)
	}
	fmt.Printf("  Health         =>  http://127.0.0.1:%d/healthz\n", cfg.Port)
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("  Upstream       =>  %s\n", up.Gateway)
	fmt.Printf("  Default effort =>  %s   (options: none low medium high | auto = client decides)\n", cfg.DefaultEffort)
	fmt.Printf("  Hedge          =>  %d ms | Keepalive: %ds | Gemini fallback: %s\n", cfg.HedgeMS, cfg.KeepaliveSec, geminiNote)
	fmt.Println(strings.Repeat("-", 70))
	if len(featured) > 0 {
		fmt.Println("  Featured models (full catalog: 'gai models' or GET /v1/models):")
		for _, f := range featured {
			fmt.Printf("    %s\n", f)
		}
		fmt.Println(strings.Repeat("-", 70))
	}
	fmt.Printf("  Zed/Cursor:    api_url = \"http://127.0.0.1:%d/v1\"\n", cfg.Port)
	fmt.Println("  Press Ctrl+C to stop.")

	if err := <-serveErr; err != nil {
		fmt.Println("[!] Server error:", err)
		os.Exit(1)
	}
}

// bindListener binds cfg.Host:cfg.Port; if busy it detects an already-running
// GAI (returns nil) or automatically falls back to the next free port.
func bindListener(cfg *Config) net.Listener {
	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	listener, err := net.Listen("tcp", addr)
	if err == nil {
		return listener
	}

	// A GAI gateway already running on that port? Then nothing to start.
	if isGAIAlreadyRunning(cfg.Port) {
		fmt.Printf("[i] GAI is already running on port %d - nothing to start.\n", cfg.Port)
		fmt.Printf("    API: http://127.0.0.1:%d/v1  |  models: http://127.0.0.1:%d/v1/models\n", cfg.Port, cfg.Port)
		fmt.Println("    To restart it: close its terminal window, or run  taskkill /IM gai.exe /F")
		return nil
	}

	// Port taken by another app: try the next 10 ports.
	for offset := 1; offset <= 10; offset++ {
		p := cfg.Port + offset
		// A GAI gateway already serving on a fallback port? Nothing to start.
		if isGAIAlreadyRunning(p) {
			fmt.Printf("[i] GAI is already running on port %d - nothing to start.\n", p)
			fmt.Printf("    API: http://127.0.0.1:%d/v1  |  models: http://127.0.0.1:%d/v1/models\n", p, p)
			fmt.Println("    To restart it: close its terminal window, or run  taskkill /IM gai.exe /F")
			return nil
		}
		l, e := net.Listen("tcp", fmt.Sprintf("%s:%d", cfg.Host, p))
		if e == nil {
			fmt.Printf("[!] Port %d is busy (used by another app) - serving on port %d instead.\n", cfg.Port, p)
			fmt.Printf("    Tip: pick a fixed port with  gai --port %d  if your IDE points there.\n", p)
			cfg.Port = p
			return l
		}
	}

	fmt.Printf("[!] Cannot listen on %s and the next 10 ports are busy too: %v\n", addr, err)
	os.Exit(1)
	return nil
}

// isGAIAlreadyRunning probes localhost:port/healthz for our health signature.
func isGAIAlreadyRunning(port int) bool {
	client := &http.Client{Timeout: 2500 * time.Millisecond}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/healthz", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 8192))
	if err != nil {
		return false
	}
	return bytes.Contains(b, []byte(`"status":"healthy"`)) && bytes.Contains(b, []byte(`"version"`))
}

func featuredModels(up *Upstream) []string {
	models, _ := up.ModelsCached()
	priority := []string{
		"ais/gemini-4-argon", "ais/gemini-3.8-flash", "ais/gemini-2.5-pro", "ais/gemini-2.5-flash",
		"ag/claude-sonnet-5-5", "ag/claude-opus-5-5",
	}
	seen := map[string]bool{}
	out := []string{}
	add := func(m map[string]any) {
		id := asString(m["id"])
		if id == "" || seen[id] {
			return
		}
		seen[id] = true
		efforts := strings.Join(toStrings(m["supported_efforts"]), "/")
		out = append(out, fmt.Sprintf("%-42s ctx=%-8d efforts=%s", id, int(asFloat(m["context_length"])), efforts))
	}
	for _, want := range priority {
		for _, m := range models {
			if asString(m["id"]) == want {
				add(m)
				break
			}
		}
	}
	for _, m := range models { // fill remaining slots from the filtered catalog
		if len(out) >= 10 {
			break
		}
		add(m)
	}
	return out
}
