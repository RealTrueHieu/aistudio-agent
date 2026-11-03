#!/usr/bin/env bash
# ==============================================================================
# AI Studio Agent Ready-to-Run cURL Examples
# ==============================================================================

GATEWAY="http://localhost:8080/v1"
API_KEY="sk-your-agent-secret-key"

echo "=== 1. Non-streaming Chat Completion ==="
curl -s -X POST "${GATEWAY}/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ais/gemini-4-argon",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 25 * 25?"}
    ],
    "max_tokens": 100
  }' | jq .

echo ""
echo "=== 2. Streaming SSE Chat Completion ==="
curl -N -s -X POST "${GATEWAY}/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "gemini-3.8-flash",
    "messages": [
      {"role": "user", "content": "Write a 4-line poem about coding at night."}
    ],
    "stream": true
  }'

echo ""
echo "=== 3. Anthropic Messages Compatibility (/v1/messages) ==="
curl -s -X POST "${GATEWAY}/messages" \
  -H "x-api-key: ${API_KEY}" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-pro",
    "max_tokens": 150,
    "messages": [
      {"role": "user", "content": "Explain quicksort in 2 sentences."}
    ]
  }' | jq .
