"""
Example: Using AI Studio Agent with the official OpenAI Python SDK
------------------------------------------------------------------
Install: pip install openai
"""

from openai import OpenAI

# 1. Connect either to the local Agent Server (localhost:8080) or the Cloud Gateway
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-your-agent-secret-key"
)

# 2. Non-streaming Call with Reasoning Model
print("=== 1. Non-Streaming Call ===")
response = client.chat.completions.create(
    model="ais/gemini-4-argon",  # Automatically mapped internally to gemini-2.5-pro in AI Studio
    messages=[
        {"role": "system", "content": "You are a concise engineering assistant."},
        {"role": "user", "content": "What are the 3 primary ACID principles in databases?"}
    ],
    temperature=0.3,
    max_tokens=200
)

print("Model:", response.model)
print("Answer:\n", response.choices[0].message.content)

# 3. Streaming Call with Realtime Thinking / Reasoning Chunks
print("\n=== 2. Streaming Call ===")
stream = client.chat.completions.create(
    model="gemini-3.8-flash",
    messages=[
        {"role": "user", "content": "Count from 1 to 5 with short cheerful greetings."}
    ],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(f"[Thinking] {delta.reasoning_content}", end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)

print("\n\nDone!")
