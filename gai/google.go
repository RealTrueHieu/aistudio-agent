package main

// google.go - native Google AI Studio payload path (the "web exploitation"
// engine): converts OpenAI-format requests into GenerateContent payloads
// and calls the Google API directly. Used as fallback when the cloud
// gateway is unreachable (requires GEMINI_API_KEY).

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
)

// Permissive safety settings to avoid upstream false positives.
var safetySettings = []any{
	map[string]any{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
	map[string]any{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
	map[string]any{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
	map[string]any{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
	map[string]any{"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
}

// Model alias & normalization table (cluster-internal routing).
var modelAliases = map[string]string{
	"gemini-4-argon":              "gemini-2.5-pro",
	"ais/gemini-4-argon":          "gemini-2.5-pro",
	"gemini-4-argon-high":         "gemini-2.5-pro",
	"ais/gemini-4-argon-high":     "gemini-2.5-pro",
	"gemini-3.8-flash":            "gemini-3.8-flash",
	"ais/gemini-3.8-flash":        "gemini-3.8-flash",
	"gemini-2.5-pro":              "gemini-2.5-pro",
	"ais/gemini-2.5-pro":          "gemini-2.5-pro",
	"gemini-2.5-flash":            "gemini-2.5-flash",
	"ais/gemini-2.5-flash":        "gemini-2.5-flash",
	"gemini-3.1-pro-preview":      "gemini-3.1-pro-preview",
	"ais/gemini-3.1-pro-preview":  "gemini-3.1-pro-preview",
}

func normalizeModel(m string) string {
	if m == "" {
		return "gemini-2.5-pro"
	}
	if v, ok := modelAliases[m]; ok {
		return v
	}
	return strings.TrimPrefix(m, "ais/")
}

// ---------------------------------------------------------------------------
// OpenAI request -> AI Studio GenerateContent payload
// ---------------------------------------------------------------------------

func openaiToAIStudio(req map[string]any) (string, map[string]any) {
	target := normalizeModel(asString(req["model"]))

	contents := []any{}
	sysParts := []any{}
	toolCallNames := map[string]string{}

	messages, _ := req["messages"].([]any)
	for _, mi := range messages {
		msg, _ := mi.(map[string]any)
		if msg == nil {
			continue
		}
		role := asString(msg["role"])
		content := msg["content"]

		if role == "system" {
			switch c := content.(type) {
			case string:
				if strings.TrimSpace(c) != "" {
					sysParts = append(sysParts, map[string]any{"text": c})
				}
			case []any:
				for _, pi := range c {
					p, _ := pi.(map[string]any)
					if p != nil && asString(p["type"]) == "text" {
						if t := asString(p["text"]); t != "" {
							sysParts = append(sysParts, map[string]any{"text": t})
						}
					}
				}
			}
			continue
		}

		aiRole := "user"
		if role == "assistant" || role == "model" {
			aiRole = "model"
		}
		var parts []any

		switch {
		case role == "assistant" && msg["tool_calls"] != nil:
			tcs, _ := msg["tool_calls"].([]any)
			for _, tci := range tcs {
				tc, _ := tci.(map[string]any)
				fn, _ := tc["function"].(map[string]any)
				if fn == nil {
					continue
				}
				fnName := asString(fn["name"])
				var fnArgs map[string]any
				switch a := fn["arguments"].(type) {
				case string:
					_ = json.Unmarshal([]byte(a), &fnArgs)
				case map[string]any:
					fnArgs = a
				}
				if fnArgs == nil {
					fnArgs = map[string]any{}
				}
				toolCallNames[asString(tc["id"])] = fnName
				parts = append(parts, map[string]any{
					"functionCall": map[string]any{"name": fnName, "args": fnArgs},
				})
			}
		case role == "tool":
			callID := asString(msg["tool_call_id"])
			toolContent := ""
			switch c := content.(type) {
			case string:
				toolContent = c
			default:
				if c != nil {
					b, _ := json.Marshal(c)
					toolContent = string(b)
				} else {
					toolContent = callID
				}
			}
			fnName := toolCallNames[callID]
			if fnName == "" {
				fnName = callID
			}
			parts = append(parts, map[string]any{
				"functionResponse": map[string]any{
					"name":     fnName,
					"response": map[string]any{"name": fnName, "content": toolContent},
				},
			})
		default:
			switch c := content.(type) {
			case string:
				if strings.TrimSpace(c) != "" {
					parts = append(parts, map[string]any{"text": c})
				}
			case []any:
				for _, ii := range c {
					item, _ := ii.(map[string]any)
					if item == nil {
						continue
					}
					switch asString(item["type"]) {
					case "text":
						if t := asString(item["text"]); t != "" {
							parts = append(parts, map[string]any{"text": t})
						}
					case "image_url":
						iu, _ := item["image_url"].(map[string]any)
						url := asString(iu["url"])
						if strings.HasPrefix(url, "data:") {
							mime, b64 := parseDataURI(url)
							parts = append(parts, map[string]any{
								"inlineData": map[string]any{"mimeType": mime, "data": b64},
							})
						} else if url != "" {
							parts = append(parts, map[string]any{"text": "[Image Link: " + url + "]"})
						}
					}
				}
			}
		}
		if len(parts) > 0 {
			contents = append(contents, map[string]any{"role": aiRole, "parts": parts})
		}
	}
	if len(contents) == 0 {
		contents = append(contents, map[string]any{"role": "user", "parts": []any{map[string]any{"text": "Hello"}}})
	}

	// Generation config
	gen := map[string]any{}
	if v := req["temperature"]; v != nil {
		gen["temperature"] = v
	}
	if v := req["top_p"]; v != nil {
		gen["topP"] = v
	}
	if v := asFloat(req["max_tokens"]); v > 0 {
		gen["maxOutputTokens"] = int(v)
	}
	switch st := req["stop"].(type) {
	case string:
		gen["stopSequences"] = []string{st}
	case []any:
		seqs := []string{}
		for _, s := range st {
			seqs = append(seqs, asString(s))
		}
		gen["stopSequences"] = seqs
	}

	// Thinking budget / reasoning effort
	budget := -1
	if th, ok := req["thinking"].(map[string]any); ok {
		budget = int(asFloat(th["budget_tokens"]))
	} else if v, ok := req["max_thinking_tokens"].(float64); ok {
		budget = int(v)
	}
	if budget >= 0 {
		gen["thinkingConfig"] = map[string]any{"thinkingBudget": budget, "includeThoughts": true}
	} else if e := strings.ToLower(asString(req["reasoning_effort"])); e != "" {
		m := map[string]int{"none": 0, "low": 4096, "medium": 12288, "high": 24576}
		if b, ok := m[e]; ok {
			gen["thinkingConfig"] = map[string]any{"thinkingBudget": b, "includeThoughts": b > 0}
		}
	}

	payload := map[string]any{"contents": contents, "safetySettings": safetySettings}
	if len(sysParts) > 0 {
		payload["systemInstruction"] = map[string]any{"parts": sysParts}
	}
	if len(gen) > 0 {
		payload["generationConfig"] = gen
	}
	if tools, ok := req["tools"].([]any); ok && len(tools) > 0 {
		var decls []any
		for _, ti := range tools {
			t, _ := ti.(map[string]any)
			if t == nil || asString(t["type"]) != "function" {
				continue
			}
			fn, _ := t["function"].(map[string]any)
			if fn == nil {
				continue
			}
			params, _ := fn["parameters"].(map[string]any)
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			decls = append(decls, map[string]any{
				"name":        asString(fn["name"]),
				"description": asString(fn["description"]),
				"parameters":  params,
			})
		}
		if len(decls) > 0 {
			payload["tools"] = []any{map[string]any{"functionDeclarations": decls}}
		}
	}
	return target, payload
}

func parseDataURI(u string) (string, string) {
	if i := strings.Index(u, ";base64,"); i > 5 {
		return u[5:i], u[i+8:]
	}
	return "image/jpeg", u
}

// ---------------------------------------------------------------------------
// AI Studio response -> OpenAI response
// ---------------------------------------------------------------------------

var callSeq int64

func aistudioToOpenaiResponse(resp map[string]any, model, reqID string) map[string]any {
	content := strings.Builder{}
	reasoning := strings.Builder{}
	var toolCalls []any
	finishReason := "stop"

	cands, _ := resp["candidates"].([]any)
	if len(cands) > 0 {
		cand, _ := cands[0].(map[string]any)
		fr := strings.ToLower(asString(cand["finishReason"]))
		if strings.Contains(fr, "max_tokens") || strings.Contains(fr, "length") {
			finishReason = "length"
		}
		contentField, _ := cand["content"].(map[string]any)
		parts, _ := contentField["parts"].([]any)
		for _, pi := range parts {
			p, _ := pi.(map[string]any)
			if p == nil {
				continue
			}
			if asBool(p["thought"]) {
				reasoning.WriteString(asString(p["text"]))
			} else if _, ok := p["text"]; ok {
				content.WriteString(asString(p["text"]))
			} else if fc, ok := p["functionCall"].(map[string]any); ok {
				args, _ := json.Marshal(fc["args"])
				toolCalls = append(toolCalls, map[string]any{
					"id":   fmt.Sprintf("call_%s_%d", asString(fc["name"]), atomic.AddInt64(&callSeq, 1)),
					"type": "function",
					"function": map[string]any{
						"name":      asString(fc["name"]),
						"arguments": string(args),
					},
				})
			}
		}
	}

	msg := map[string]any{"role": "assistant", "content": content.String()}
	if reasoning.Len() > 0 {
		msg["reasoning_content"] = reasoning.String()
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}

	usage, _ := resp["usageMetadata"].(map[string]any)
	if usage == nil {
		usage = map[string]any{}
	}
	promptTokens := asFloat(usage["promptTokenCount"])
	candTokens := asFloat(usage["candidatesTokenCount"])
	return map[string]any{
		"id": reqID, "object": "chat.completion", "created": 1793620000, "model": model,
		"choices": []any{map[string]any{
			"index": 0, "message": msg, "logprobs": nil, "finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": promptTokens, "completion_tokens": candTokens,
			"total_tokens": promptTokens + candTokens,
		},
		"provider_model": model,
	}
}

// ---------------------------------------------------------------------------
// Google SSE -> OpenAI SSE (streaming conversion)
// ---------------------------------------------------------------------------

// convertGoogleChunk converts one Google SSE line into an OpenAI SSE event.
func convertGoogleChunk(line []byte, model, reqID string) []byte {
	s := strings.TrimSpace(string(line))
	if s == "" {
		return nil
	}
	s = strings.TrimSpace(strings.TrimPrefix(s, "data:"))
	if s == "[DONE]" {
		return []byte("data: [DONE]\n\n")
	}
	var obj map[string]any
	if json.Unmarshal([]byte(s), &obj) != nil {
		return nil
	}
	cands, _ := obj["candidates"].([]any)
	if len(cands) == 0 {
		return nil
	}
	cand, _ := cands[0].(map[string]any)
	contentField, _ := cand["content"].(map[string]any)
	parts, _ := contentField["parts"].([]any)

	var contentParts, reasoningParts []string
	var toolCalls []any
	for _, pi := range parts {
		p, _ := pi.(map[string]any)
		if p == nil {
			continue
		}
		if asBool(p["thought"]) {
			reasoningParts = append(reasoningParts, asString(p["text"]))
		} else if _, ok := p["text"]; ok {
			contentParts = append(contentParts, asString(p["text"]))
		} else if fc, ok := p["functionCall"].(map[string]any); ok {
			args, _ := json.Marshal(fc["args"])
			toolCalls = append(toolCalls, map[string]any{
				"index": len(toolCalls),
				"id":    fmt.Sprintf("call_%s_%d", asString(fc["name"]), atomic.AddInt64(&callSeq, 1)),
				"type":  "function",
				"function": map[string]any{
					"name":      asString(fc["name"]),
					"arguments": string(args),
				},
			})
		}
	}

	delta := map[string]any{}
	if len(contentParts) > 0 {
		delta["content"] = strings.Join(contentParts, "")
	}
	if len(reasoningParts) > 0 {
		delta["reasoning_content"] = strings.Join(reasoningParts, "")
	}
	if len(toolCalls) > 0 {
		delta["tool_calls"] = toolCalls
	}

	openaiFinish := ""
	switch fr := strings.ToLower(asString(cand["finishReason"])); {
	case fr == "":
	case strings.Contains(fr, "stop"):
		openaiFinish = "stop"
	case strings.Contains(fr, "max_tokens") || strings.Contains(fr, "length"):
		openaiFinish = "length"
	case strings.Contains(fr, "safety") || strings.Contains(fr, "recitation"):
		openaiFinish = "content_filter"
	default:
		openaiFinish = "stop"
	}
	if len(delta) == 0 && openaiFinish == "" {
		return nil
	}

	chunk := map[string]any{
		"id": reqID, "object": "chat.completion.chunk", "created": 1793620000, "model": model,
		"choices": []any{map[string]any{
			"index": 0, "delta": delta, "finish_reason": nilOrString(openaiFinish),
		}},
	}
	b, _ := json.Marshal(chunk)
	return append(append([]byte("data: "), b...), '\n', '\n')
}

func nilOrString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// googleSSEConverter lazily converts a Google SSE body into OpenAI SSE bytes.
type googleSSEConverter struct {
	src        *bufio.Reader
	model      string
	reqID      string
	buf        bytes.Buffer
	done       bool
	sentDone   bool
}

func (g *googleSSEConverter) Read(p []byte) (int, error) {
	for g.buf.Len() == 0 && !g.done {
		line, err := g.src.ReadBytes('\n')
		if len(line) > 0 {
			if out := convertGoogleChunk(line, g.model, g.reqID); out != nil {
				g.buf.Write(out)
			}
		}
		if err != nil {
			if !g.sentDone {
				g.buf.WriteString("data: [DONE]\n\n")
				g.sentDone = true
			}
			g.done = true
			break
		}
	}
	if g.buf.Len() == 0 {
		return 0, io.EOF
	}
	return g.buf.Read(p)
}

type convertedBody struct {
	conv *googleSSEConverter
	orig io.ReadCloser
}

func (c *convertedBody) Read(p []byte) (int, error) { return c.conv.Read(p) }
func (c *convertedBody) Close() error               { return c.orig.Close() }

// ---------------------------------------------------------------------------
// Direct Google call (fallback path)
// ---------------------------------------------------------------------------

func (u *Upstream) googleGenerate(ctx context.Context, body []byte, stream bool) (*http.Response, error) {
	var req map[string]any
	if json.Unmarshal(body, &req) != nil {
		return nil, fmt.Errorf("google fallback: invalid request body")
	}
	target, payload := openaiToAIStudio(req)
	pb, _ := json.Marshal(payload)

	method := "generateContent"
	if stream {
		method = "streamGenerateContent?alt=sse"
	}
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + target + ":" + method
	headers := map[string]string{
		"Content-Type":   "application/json",
		"x-goog-api-key": u.GeminiKey,
	}
	resp, err := u.doRequest(ctx, "POST", url, headers, pb)
	if err != nil {
		return nil, err
	}

	reqID := fmt.Sprintf("chatcmpl_%d", atomic.AddInt64(&callSeq, 1))
	if stream {
		resp.Body = &convertedBody{
			conv: &googleSSEConverter{
				src:   bufio.NewReader(resp.Body),
				model: target, reqID: reqID,
			},
			orig: resp.Body,
		}
		return resp, nil
	}
	data, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, err
	}
	var g map[string]any
	if json.Unmarshal(data, &g) != nil {
		return nil, fmt.Errorf("google fallback: unparseable response")
	}
	out, _ := json.Marshal(aistudioToOpenaiResponse(g, target, reqID))
	resp.Body = io.NopCloser(bytes.NewReader(out))
	resp.ContentLength = int64(len(out))
	return resp, nil
}
