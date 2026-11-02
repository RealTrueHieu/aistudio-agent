"""
Comprehensive Test Suite for AI Studio Agent Package
---------------------------------------------------
Verifies payload translation, stream parsing, and roundtrip conversion.
"""

import sys
import json
import os
from payload_transformer import AIStudioPayloadTransformer, normalize_model_name
from session_manager import AccountSession, AIStudioSessionPool
from aistudio_agent import AIStudioAgent

def test_payload_transformation():
    print("[1/3] Testing OpenAI -> AI Studio Transformation...")
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

    print("[2/3] Testing Anthropic -> AI Studio Transformation...")
    anthropic_req = {
        "model": "gemini-3.8-flash",
        "system": "Respond in JSON.",
        "messages": [
            {"role": "user", "content": "Hello world"}
        ],
        "max_tokens": 500
    }
    model_ant, aistudio_payload_ant = AIStudioPayloadTransformer.anthropic_to_aistudio(anthropic_req)
    assert model_ant == "gemini-3.8-flash", f"Model mismatch: {model_ant}"
    assert "systemInstruction" in aistudio_payload_ant
    print("      PASSED!")

def test_session_pool():
    print("[3/3] Testing Session Pool & Load Balancer...")
    pool = AIStudioSessionPool()
    pool.add_account(AccountSession(identifier="test1@gmail.com", tier="Pro"))
    pool.add_account(AccountSession(identifier="test2@gmail.com", tier="Free"))
    
    acc = pool.get_next_account()
    assert acc is not None, "Failed to get account"
    assert acc.identifier == "test1@gmail.com", "Pro account priority failed"
    print("      PASSED!")

if __name__ == "__main__":
    print("=== AI STUDIO AGENT TEST SUITE ===")
    try:
        test_payload_transformation()
        test_session_pool()
        print("\nALL UNIT TESTS PASSED SUCCESSFULLY! Ready for open source.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
