---
name: aistudio-payload-agent
description: >-
  Autonomous Google AI Studio Universal Agent & Payload Engine.
  Translates OpenAI / Anthropic requests into Google AI Studio native Protobuf/JSON payloads,
  manages multi-account pooling, reasoning/thinking budget control, streaming SSE, and tool calling.
---

# AI Studio Autonomous Agent & Payload Specification

This agent provides a complete translation and execution engine between client LLM protocols (OpenAI `/v1/chat/completions` and Anthropic Claude `/v1/messages`) and the native Google AI Studio backend (`GenerateContent` & `streamGenerateContent`).

## 1. Supported Model Topology

| Client Requested Model | Internal Route Target | Engine Characteristic |
| :--- | :--- | :--- |
| `ais/gemini-4-argon` | `gemini-2.5-pro` | Flagship reasoning & heavy coding synthesis |
| `gemini-4-argon` | `gemini-2.5-pro` | Fallback alias without prefix |
| `gemini-3.8-flash` | `gemini-3.8-flash` | Ultra-low latency conversational & general tasks |
| `gemini-2.5-pro` | `gemini-2.5-pro` | Native complex reasoning & deep context |
| `gemini-2.5-flash` | `gemini-2.5-flash` | Lightweight fast tasks |

## 2. Core Execution Capabilities

- **Zero-Dependency Gateway Server**: Built purely on Python standard library (`http.server`, `urllib`).
- **Thinking / Extended Reasoning Support**: Extracts `thinking: {budget_tokens: ...}` or `max_thinking_tokens` and routes them into Google's `thinkingConfig`.
- **Bidirectional Streaming Adapter**: Translates Google's chunk structure (`candidates[0].content.parts[0].text` + `.thought`) into standard OpenAI SSE chunks with `delta.reasoning_content`.
- **Multimodal Ingestion**: Supports `inlineData` with Base64 encoding for image, audio, and documents.
- **Failover & Session Resilience**: Supports multi-account Round-Robin rotation and transparent fallback.

## 3. Quick Run Commands

```bash
# 1. Direct CLI Query with streaming
python aistudio_agent.py "Giải thích định lý Bayes trong 3 gạch đầu dòng"

# 2. Launch Local Gateway Server on port 8080 (serves Zed, Cursor, Cline)
python aistudio_agent.py --serve --port 8080

# 3. Run Self-Test Suite
python test_agent.py
```
