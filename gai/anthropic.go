package main

// anthropic.go - real Anthropic <-> OpenAI translation: system prompts,
// multimodal blocks, tool_use / tool_result, thinking budgets, and a
// stateful OpenAI-SSE -> Anthropic-SSE stream translator.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// Anthropic request -> OpenAI request
// ---------------------------------------------------------------------------

func anthropicToOpenai(req map[string]any) map[string]any {
	out := map[string]any{"model": asString(req["model"])}
	if out["model"] == "" {
		out["model"] = "gemini-2.5-pro"
	}
	messages := []any{}

	// 1. System prompt
	switch sys := req["system"].(type) {
	case string:
		if strings.TrimSpace(sys) != "" {
			messages = append(messages, map[string]any{"role": "system", "content": sys})
		}
	case []any:
		parts := []string{}
		for _, b := range sys {
			bm, _ := b.(map[string]any)
			if t := asString(bm["text"]); t != "" {
				parts = append(parts, t)
			}
		}
		if len(parts) > 0 {
			messages = append(messages, map[string]any{"role": "system", "content": strings.Join(parts, "\n")})
		}
	}

	// 2. Messages
	msgs, _ := req["messages"].([]any)
	for _, mi := range msgs {
		msg, _ := mi.(map[string]any)
		if msg == nil {
			continue
		}
		role := "user"
		if asString(msg["role"]) == "assistant" {
			role = "assistant"
		}

		if s, ok := msg["content"].(string); ok {
			messages = append(messages, map[string]any{"role": role, "content": s})
			continue
		}

		blocks, _ := msg["content"].([]any)
		var textParts []string
		var imageParts []any
		var toolCalls []any

		for _, bi := range blocks {
			block, _ := bi.(map[string]any)
			if block == nil {
				continue
			}
			switch asString(block["type"]) {
			case "text":
				if t := asString(block["text"]); t != "" {
					textParts = append(textParts, t)
				}
			case "image":
				src, _ := block["source"].(map[string]any)
				if asString(src["type"]) == "base64" {
					uri := "data:" + asString(src["media_type"]) + ";base64," + asString(src["data"])
					imageParts = append(imageParts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": uri}})
				}
			case "tool_use":
				input, _ := json.Marshal(block["input"])
				toolCalls = append(toolCalls, map[string]any{
					"id":   orDefault(asString(block["id"]), "call_default"),
					"type": "function",
					"function": map[string]any{
						"name":      asString(block["name"]),
						"arguments": string(input),
					},
				})
			case "tool_result":
				rc := block["content"]
				rcStr, ok := rc.(string)
				if !ok {
					b, _ := json.Marshal(rc)
					rcStr = string(b)
				}
				messages = append(messages, map[string]any{
					"role":         "tool",
					"tool_call_id": orDefault(asString(block["tool_use_id"]), "call_default"),
					"content":      rcStr,
				})
			}
		}

		if role == "assistant" {
			m := map[string]any{"role": "assistant", "content": nil}
			if joined := strings.Join(textParts, "\n"); joined != "" {
				m["content"] = joined
			}
			if len(toolCalls) > 0 {
				m["tool_calls"] = toolCalls
			}
			messages = append(messages, m)
		} else if len(textParts) > 0 || len(imageParts) > 0 {
			combined := []any{}
			for _, t := range textParts {
				combined = append(combined, map[string]any{"type": "text", "text": t})
			}
			combined = append(combined, imageParts...)
			if len(combined) == 1 {
				if tm, _ := combined[0].(map[string]any); asString(tm["type"]) == "text" {
					messages = append(messages, map[string]any{"role": "user", "content": asString(tm["text"])})
					continue
				}
			}
			messages = append(messages, map[string]any{"role": "user", "content": combined})
		}
	}

	if len(messages) == 0 {
		messages = append(messages, map[string]any{"role": "user", "content": "Hello"})
	}
	out["messages"] = messages

	// 3. Sampling params
	if v := req["max_tokens"]; v != nil {
		out["max_tokens"] = v
	}
	if v := req["temperature"]; v != nil {
		out["temperature"] = v
	}
	if v := req["top_p"]; v != nil {
		out["top_p"] = v
	}
	if ss, ok := req["stop_sequences"].([]any); ok && len(ss) > 0 {
		out["stop"] = ss
	}

	// 4. Thinking budget
	if th, ok := req["thinking"].(map[string]any); ok && asString(th["type"]) == "enabled" {
		if b := asFloat(th["budget_tokens"]); b > 0 {
			out["max_thinking_tokens"] = int(b)
		}
	}

	// 5. Tools
	if tools, ok := req["tools"].([]any); ok && len(tools) > 0 {
		var outTools []any
		for _, ti := range tools {
			t, _ := ti.(map[string]any)
			if t == nil {
				continue
			}
			schema, _ := t["input_schema"].(map[string]any)
			if schema == nil {
				schema = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			outTools = append(outTools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        asString(t["name"]),
					"description": orDefault(asString(t["description"]), ""),
					"parameters":  schema,
				},
			})
		}
		if len(outTools) > 0 {
			out["tools"] = outTools
		}
	}
	return out
}

func orDefault(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

func maybeMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

// ---------------------------------------------------------------------------
// OpenAI response -> Anthropic response
// ---------------------------------------------------------------------------

func openaiToAnthropicResponse(openaiResp map[string]any, model string) map[string]any {
	if e, ok := openaiResp["error"].(map[string]any); ok {
		return map[string]any{
			"type":  "error",
			"error": map[string]any{"type": orDefault(asString(e["type"]), "api_error"), "message": asString(e["message"])},
		}
	}
	var content []any
	choices, _ := openaiResp["choices"].([]any)
	var msg map[string]any
	finish := ""
	if len(choices) > 0 {
		ch0, _ := choices[0].(map[string]any)
		msg, _ = ch0["message"].(map[string]any)
		finish = asString(ch0["finish_reason"])
	}
	if msg == nil {
		msg = map[string]any{}
	}
	if reasoning := asString(msg["reasoning_content"]); reasoning != "" {
		content = append(content, map[string]any{"type": "thinking", "thinking": reasoning})
	}
	if text := asString(msg["content"]); text != "" {
		content = append(content, map[string]any{"type": "text", "text": text})
	}
	hasTools := false
	if tcs, ok := msg["tool_calls"].([]any); ok {
		for _, tci := range tcs {
			tc, _ := tci.(map[string]any)
			fn, _ := tc["function"].(map[string]any)
			var input map[string]any
			_ = json.Unmarshal([]byte(asString(fn["arguments"])), &input)
			content = append(content, map[string]any{
				"type": "tool_use", "id": orDefault(asString(tc["id"]), "call_default"),
				"name": asString(fn["name"]), "input": input,
			})
			hasTools = true
		}
	}
	if len(content) == 0 {
		content = append(content, map[string]any{"type": "text", "text": ""})
	}

	stopReason := "end_turn"
	switch {
	case hasTools:
		stopReason = "tool_use"
	case finish == "length":
		stopReason = "max_tokens"
	}

	usage, _ := openaiResp["usage"].(map[string]any)
	if usage == nil {
		usage = map[string]any{}
	}
	return map[string]any{
		"id":            orDefault(asString(openaiResp["id"]), "msg_gai"),
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  asFloat(usage["prompt_tokens"]),
			"output_tokens": asFloat(usage["completion_tokens"]),
		},
	}
}

// ---------------------------------------------------------------------------
// OpenAI SSE chunks -> Anthropic Messages SSE events (stateful translator)
// ---------------------------------------------------------------------------

type anthropicStreamTranslator struct {
	model        string
	started      bool
	blockIndex   int
	blockType    string // "", "thinking", "text", "tool_use"
	toolName     string
	inputTokens  float64
	outputTokens float64
	stopReason   string
}

func newAnthropicStreamTranslator(model string) *anthropicStreamTranslator {
	return &anthropicStreamTranslator{model: model, blockIndex: -1}
}

func (t *anthropicStreamTranslator) event(eventType string, data any) []byte {
	b, _ := json.Marshal(data)
	out := make([]byte, 0, len(b)+len(eventType)+16)
	out = append(out, "event: "...)
	out = append(out, eventType...)
	out = append(out, "\ndata: "...)
	out = append(out, b...)
	out = append(out, '\n', '\n')
	return out
}

func (t *anthropicStreamTranslator) messageStart() []byte {
	t.started = true
	return t.event("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_gai", "type": "message", "role": "assistant", "model": t.model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	})
}

func (t *anthropicStreamTranslator) openBlock(btype string) []byte {
	var out []byte
	if t.blockType != "" {
		out = append(out, t.closeBlock()...)
	}
	t.blockIndex++
	t.blockType = btype
	block := map[string]any{"type": btype, "index": t.blockIndex}
	switch btype {
	case "text":
		block["text"] = ""
	case "thinking":
		block["thinking"] = ""
	case "tool_use":
		block["id"] = fmt.Sprintf("call_%d", t.blockIndex)
		block["name"] = t.toolName
		block["input"] = map[string]any{}
	}
	return append(out, t.event("content_block_start", map[string]any{
		"type": "content_block_start", "index": t.blockIndex, "content_block": block,
	})...)
}

func (t *anthropicStreamTranslator) closeBlock() []byte {
	out := t.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": t.blockIndex})
	t.blockType = ""
	return out
}

func (t *anthropicStreamTranslator) absorbUsage(usage map[string]any) {
	if usage == nil {
		return
	}
	if v := asFloat(usage["prompt_tokens"]); v > 0 {
		t.inputTokens = v
	}
	if v := asFloat(usage["completion_tokens"]); v > 0 {
		t.outputTokens = v
	}
}

// HandleChunk feeds one parsed OpenAI chunk, returns Anthropic SSE bytes.
func (t *anthropicStreamTranslator) HandleChunk(chunk map[string]any) []byte {
	if e, ok := chunk["error"].(map[string]any); ok {
		return t.event("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": "api_error", "message": orDefault(asString(e["message"]), "upstream error")},
		})
	}
	t.absorbUsage(maybeMap(chunk["usage"]))

	choices, _ := chunk["choices"].([]any)
	if len(choices) == 0 {
		return nil
	}
	var out []byte
	if !t.started {
		out = append(out, t.messageStart()...)
	}
	choice, _ := choices[0].(map[string]any)
	delta, _ := choice["delta"].(map[string]any)
	if delta == nil {
		delta = map[string]any{}
	}

	if rc := asString(delta["reasoning_content"]); rc != "" {
		if t.blockType != "thinking" {
			out = append(out, t.openBlock("thinking")...)
		}
		out = append(out, t.event("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": t.blockIndex,
			"delta": map[string]any{"type": "thinking_delta", "thinking": rc},
		})...)
	}
	if c := asString(delta["content"]); c != "" {
		if t.blockType != "text" {
			out = append(out, t.openBlock("text")...)
		}
		out = append(out, t.event("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": t.blockIndex,
			"delta": map[string]any{"type": "text_delta", "text": c},
		})...)
	}
	if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
		tc, _ := tcs[0].(map[string]any)
		fn, _ := tc["function"].(map[string]any)
		if fn == nil {
			fn = map[string]any{}
		}
		if name := asString(fn["name"]); name != "" && t.blockType != "tool_use" {
			t.toolName = name
			out = append(out, t.openBlock("tool_use")...)
		}
		if t.blockType == "tool_use" {
			if args := asString(fn["arguments"]); args != "" {
				out = append(out, t.event("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": t.blockIndex,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
				})...)
			}
		}
	}

	switch fr := asString(choice["finish_reason"]); {
	case fr == "length":
		t.stopReason = "max_tokens"
	case fr == "tool_calls" || t.blockType == "tool_use":
		t.stopReason = "tool_use"
	case fr != "":
		t.stopReason = "end_turn"
	}
	return out
}

// Finish closes any open block and emits the terminating events.
func (t *anthropicStreamTranslator) Finish() []byte {
	if !t.started {
		return nil
	}
	var out []byte
	if t.blockType != "" {
		out = append(out, t.closeBlock()...)
	}
	out = append(out, t.event("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": orDefault(t.stopReason, "end_turn"), "stop_sequence": nil},
		"usage": map[string]any{"input_tokens": t.inputTokens, "output_tokens": t.outputTokens},
	})...)
	out = append(out, t.event("message_stop", map[string]any{"type": "message_stop"})...)
	return out
}
