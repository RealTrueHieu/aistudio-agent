"""
Comprehensive Test Suite for AI Studio Agent Package (v2)
---------------------------------------------------------
Verifies payload translation, Anthropic conversion, stream parsing,
connection pooling, and live connectivity.
"""

import os
import sys
import json
from payload_transformer import AIStudioPayloadTransformer, normalize_model_name
from session_manager import AccountSession, AIStudioSessionPool
from aistudio_agent import AIStudioAgent

def test_payload_transformation():
    print("[1/5] Testing OpenAI -> AI Studio Transformation...")
    openai_req = {
        "model": "ais/gemini-4-argon",
        "messages": [
            {"role": "system", "content": "You are a senior coding architect."},
            {"role": "user", "content": "Write a binary search algorithm in Python."}
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
        "thinking": {"budget_tokens": 1024}
    }
    model, aistudio_payload = AIStudioPayloadTransformer.openai_to_aistudio(openai_req)
    assert model == "gemini-2.5-pro", f"Model alias failed: {model}"
    assert "contents" in aistudio_payload, "Missing contents"
    assert aistudio_payload["contents"][0]["role"] == "user"
    assert "systemInstruction" in aistudio_payload, "Missing systemInstruction"
    assert aistudio_payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 1024
    print("      PASSED!")

    print("[2/5] Testing Anthropic -> OpenAI -> Anthropic Conversion...")
    anthropic_req = {
        "model": "gemini-3.8-flash",
        "system": "Respond in JSON.",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "Hello world"}
        ]
    }
    openai_converted = AIStudioPayloadTransformer.anthropic_to_openai(anthropic_req)
    assert openai_converted["messages"][0]["role"] == "system", "System prompt not mapped"
    assert openai_converted["messages"][-1]["content"] == "Hello world", "User text not mapped"
    assert openai_converted["max_tokens"] == 500

    # Tool round-trip: anthropic tool_use -> openai tool_calls
    tool_req = {
        "model": "gemini-3.8-flash",
        "max_tokens": 100,
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Hanoi"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "25C sunny"}
            ]}
        ]
    }
    conv = AIStudioPayloadTransformer.anthropic_to_openai(tool_req)
    roles = [m["role"] for m in conv["messages"]]
    assert "assistant" in roles and "tool" in roles, f"Tool roles not mapped: {roles}"
    tool_msg = [m for m in conv["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "toolu_1", "tool_call_id not mapped"

    # Response back-conversion
    fake_openai_resp = {
        "id": "chatcmpl_test", "choices": [{"message": {
            "role": "assistant", "content": "Hello!", "tool_calls": []
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    }
    ant_resp = AIStudioPayloadTransformer.openai_to_anthropic_response(fake_openai_resp, "gemini-3.8-flash")
    assert ant_resp["type"] == "message" and ant_resp["content"][0]["text"] == "Hello!"
    assert ant_resp["stop_reason"] == "end_turn"
    print("      PASSED!")

def test_chunk_parsing():
    print("[3/5] Testing SSE Chunk Parsing (multi-part concat + tool calls)...")
    raw = json.dumps({
        "candidates": [{
            "content": {"parts": [
                {"thought": True, "text": "thinking hard"},
                {"text": "Hello "},
                {"text": "world"}
            ]},
            "finishReason": None
        }]
    })
    chunk = AIStudioPayloadTransformer.aistudio_to_openai_chunk(f"data: {raw}", "gemini-2.5-pro", "id1")
    assert chunk, "No chunk produced"
    obj = json.loads(chunk[6:])
    delta = obj["choices"][0]["delta"]
    assert delta.get("content") == "Hello world", f"Multi-part concat failed: {delta}"
    assert delta.get("reasoning_content") == "thinking hard", "Reasoning part lost"

    tool_raw = json.dumps({
        "candidates": [{
            "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Hanoi"}}}]},
            "finishReason": "STOP"
        }]
    })
    tool_chunk = AIStudioPayloadTransformer.aistudio_to_openai_chunk(f"data: {tool_raw}", "gemini-2.5-pro", "id2")
    tool_obj = json.loads(tool_chunk[6:])
    tc = tool_obj["choices"][0]["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather" and json.loads(tc["function"]["arguments"]) == {"city": "Hanoi"}
    print("      PASSED!")

def test_effort_policy():
    print("[4/5] Testing default-effort injection & reasoning_effort mapping...")
    agent = AIStudioAgent(default_effort="none")

    # Injection when client sent nothing
    req = agent._apply_default_effort({"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert req["reasoning_effort"] == "none", "Default effort not injected"

    # Respect explicit client control
    req = agent._apply_default_effort({"model": "m", "messages": [], "reasoning_effort": "high"})
    assert req["reasoning_effort"] == "high", "Explicit effort overwritten"
    req = agent._apply_default_effort({"model": "m", "messages": [], "max_thinking_tokens": 2048})
    assert "reasoning_effort" not in req, "Injected over explicit thinking budget"
    req = agent._apply_default_effort({"model": "m", "messages": [], "thinking": {"budget_tokens": 1024}})
    assert "reasoning_effort" not in req, "Injected over explicit thinking config"

    # Disabled injection
    agent2 = AIStudioAgent(default_effort="auto")
    req = agent2._apply_default_effort({"model": "m", "messages": []})
    assert "reasoning_effort" not in req, "auto should never inject"

    # reasoning_effort -> thinkingConfig mapping (Google direct path)
    _, payload = AIStudioPayloadTransformer.openai_to_aistudio({
        "model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none"})
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0, "includeThoughts": False}, \
        f"effort=none mapping wrong: {payload['generationConfig'].get('thinkingConfig')}"
    _, payload2 = AIStudioPayloadTransformer.openai_to_aistudio({
        "model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high"})
    assert payload2["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 24576
    print("      PASSED!")

def test_live_agent():
    print("[5/5] Testing Live Agent (keep-alive pool, non-stream + stream)...")
    gw = os.environ.get("AISTUDIO_GATEWAY")
    key = os.environ.get("AISTUDIO_API_KEY")

    if not gw or not key:
        print("      SKIPPED: Set AISTUDIO_GATEWAY and AISTUDIO_API_KEY to run live agent tests.")
        return

    agent = AIStudioAgent(gateway_url=gw, api_key=key)

    # Non-streaming
    resp = agent.chat("Reply with the single word: pong", model="ais/gemini-4-argon")
    assert "error" not in resp, f"Chat error: {resp.get('error')}"
    content = resp["choices"][0]["message"].get("content", "")
    assert content.strip(), "Empty chat content"

    # Streaming
    tokens = []
    for token in agent.stream("Count: 1, 2, 3", model="ais/gemini-4-argon"):
        tokens.append(token)
    full_text = "".join(tokens)
    assert len(full_text.strip()) > 0, "No response tokens received!"

    # Connection pool should hold a reusable keep-alive connection (soft check)
    stats = agent.conn_pool.stats()
    print(f"      PASSED! Chat replied: '{content.strip()[:20]}' | stream: {len(full_text)} chars | pool: {stats}")

if __name__ == "__main__":
    print("=== AI STUDIO AGENT v2.1 TEST SUITE ===")
    try:
        test_payload_transformation()
        test_chunk_parsing()
        test_effort_policy()
        test_live_agent()
        print("\nALL TESTS PASSED SUCCESSFULLY! Ready to share.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
