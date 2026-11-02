"""
AI Studio Autonomous Agent & Gateway Server
============================================
A fully self-contained AI Studio proxy server, CLI agent, and Python SDK.
Zero external dependencies - 100% Python standard library compatible.

Author: Antigravity & RealTrueHieu
Version: 1.0.0
"""

import os
import sys
import time
import json
import uuid
import argparse
import urllib.request
import urllib.error
import http.server
from typing import Dict, Any, List, Optional, Generator, Union

from payload_transformer import AIStudioPayloadTransformer, normalize_model_name
from session_manager import AccountSession, AIStudioSessionPool

# Default Relay Endpoint (Online Cloud Gateway)
DEFAULT_GATEWAY_URL = os.environ.get("AISTUDIO_GATEWAY", "https://truehieu.eu.cc/v1")
DEFAULT_API_KEY = os.environ.get("AISTUDIO_API_KEY", "sk-your-agent-secret-key")
DEFAULT_MODEL = "ais/gemini-4-argon"

SUPPORTED_MODELS = [
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro (Flagship Reasoning & Coding)"},
    {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash (Ultra Fast Reasoning)"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash (Standard Speed)"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro (Preview Tier)"},
    {"id": "ais/gemini-4-argon", "name": "Gemini 4 Argon (Reasoning Preview)"},
    {"id": "ais/gemini-4-argon-high", "name": "Gemini 4 Argon High Budget"},
    {"id": "antigravity-preview-09-2026", "name": "Antigravity Preview Engine"}
]


class AIStudioAgent:
    """Core Agent engine for executing queries against AI Studio backends."""

    def __init__(
        self,
        gateway_url: str = DEFAULT_GATEWAY_URL,
        api_key: str = DEFAULT_API_KEY,
        official_gemini_key: Optional[str] = None
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.api_key = api_key
        self.official_gemini_key = official_gemini_key or os.environ.get("GEMINI_API_KEY")
        self.pool = AIStudioSessionPool()

        # Initialize primary gateway connection
        self.pool.add_account(
            AccountSession(
                identifier="primary-cloud-gateway",
                auth_type="gateway",
                credential=self.api_key,
                endpoint=self.gateway_url
            )
        )

        # If official Gemini API Key provided, add direct Google endpoint
        if self.official_gemini_key:
            self.pool.add_account(
                AccountSession(
                    identifier="official-google-api",
                    auth_type="api_key",
                    credential=self.official_gemini_key,
                    endpoint="https://generativelanguage.googleapis.com"
                )
            )

    def chat(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        thinking_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """Synchronous non-streaming query returning OpenAI format response."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        req_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if thinking_budget:
            req_body["max_thinking_tokens"] = thinking_budget

        return self.execute_openai_chat(req_body, stream=False)

    def stream(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        thinking_budget: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Streaming generator yielding text and reasoning tokens."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        req_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        if thinking_budget:
            req_body["max_thinking_tokens"] = thinking_budget

        for chunk_line in self.execute_openai_chat(req_body, stream=True):
            if chunk_line.startswith("data: ") and chunk_line.strip() != "data: [DONE]":
                try:
                    obj = json.loads(chunk_line[6:])
                    delta = obj["choices"][0]["delta"]
                    # Yield content or reasoning
                    if delta.get("content"):
                        yield delta["content"]
                    elif delta.get("reasoning_content"):
                        yield f"\n[Thinking] {delta['reasoning_content']}\n"
                except Exception:
                    pass

    def execute_openai_chat(self, req_data: Dict[str, Any], stream: bool = False) -> Any:
        """Main dispatcher for /v1/chat/completions."""
        req_id = f"chatcmpl_{uuid.uuid4().hex[:12]}"
        model = req_data.get("model", DEFAULT_MODEL)

        # Route through Cloud Gateway
        url = f"{self.gateway_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AIStudio-Autonomous-Agent/1.0"
        }
        if stream:
            headers["Accept"] = "text/event-stream"

        body_bytes = json.dumps(req_data).encode("utf-8")
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")

        try:
            resp = urllib.request.urlopen(req, timeout=60)
            if stream:
                def event_stream_generator():
                    try:
                        for line in resp:
                            l = line.decode("utf-8")
                            yield l
                    finally:
                        resp.close()
                return event_stream_generator()
            else:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            # If upstream returns error, fallback to native transformation
            return self._fallback_direct_google_call(req_data, stream, err_msg)
        except Exception as e:
            return self._fallback_direct_google_call(req_data, stream, str(e))

    def _fallback_direct_google_call(self, req_data: Dict[str, Any], stream: bool, prev_err: str):
        """Direct fallback executing raw AI Studio transformation if gateway is unreachable."""
        target_model, aistudio_payload = AIStudioPayloadTransformer.openai_to_aistudio(req_data)
        req_id = f"chatcmpl_{uuid.uuid4().hex[:12]}"

        # If official Gemini API Key is available, query Google directly
        if self.official_gemini_key:
            endpoint_method = "streamGenerateContent?alt=sse" if stream else "generateContent"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:{endpoint_method}&key={self.official_gemini_key}"
            body_bytes = json.dumps(aistudio_payload).encode("utf-8")
            req = urllib.request.Request(url, data=body_bytes, headers={"Content-Type": "application/json"}, method="POST")
            try:
                resp = urllib.request.urlopen(req, timeout=60)
                if stream:
                    def stream_gen():
                        for line in resp:
                            l = line.decode("utf-8")
                            chunk_out = AIStudioPayloadTransformer.aistudio_to_openai_chunk(l, target_model, req_id)
                            if chunk_out:
                                yield chunk_out
                        yield "data: [DONE]\n\n"
                    return stream_gen()
                else:
                    raw_data = json.loads(resp.read().decode("utf-8"))
                    return AIStudioPayloadTransformer.aistudio_to_openai_response(raw_data, target_model, req_id)
            except Exception as e:
                pass

        # Return structured error
        err_response = {
            "error": {
                "message": f"Gateway unreachable and no local Gemini API key available. Details: {prev_err}",
                "type": "server_error",
                "code": 503
            }
        }
        if stream:
            return iter([f"data: {json.dumps(err_response)}\n\n", "data: [DONE]\n\n"])
        return err_response


class GatewayHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Server providing OpenAI /v1/chat/completions and Anthropic /v1/messages compatibility."""

    agent: AIStudioAgent = None

    def log_message(self, format, *args):
        # Clean custom logging
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]} -> {args[2]}\n")

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ["/v1/models", "/models"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            model_list = {
                "object": "list",
                "data": [{"id": m["id"], "object": "model", "created": 1793620000, "owned_by": "google"} for m in SUPPORTED_MODELS]
            }
            self.wfile.write(json.dumps(model_list, indent=2).encode("utf-8"))
        elif self.path in ["/healthz", "/status"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            status_obj = {
                "status": "healthy",
                "version": "1.0.0",
                "pool": self.agent.pool.get_status_summary()
            }
            self.wfile.write(json.dumps(status_obj, indent=2).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)

        try:
            req_json = json.loads(post_data.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid JSON payload","type":"invalid_request_error"}}')
            return

        is_stream = req_json.get("stream", False)

        # 1. Handle OpenAI /v1/chat/completions
        if self.path in ["/v1/chat/completions", "/chat/completions"]:
            if is_stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_cors_headers()
                self.end_headers()

                gen = self.agent.execute_openai_chat(req_json, stream=True)
                for chunk in gen:
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            else:
                resp = self.agent.execute_openai_chat(req_json, stream=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

        # 2. Handle Anthropic /v1/messages
        elif self.path in ["/v1/messages", "/messages"]:
            # Convert Anthropic request to OpenAI format internally
            target_model, aistudio_payload = AIStudioPayloadTransformer.anthropic_to_aistudio(req_json)
            fake_openai_req = {
                "model": target_model,
                "messages": [{"role": "user", "content": json.dumps(aistudio_payload["contents"])}],
                "stream": is_stream
            }
            if is_stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_cors_headers()
                self.end_headers()
                for chunk in self.agent.execute_openai_chat(fake_openai_req, stream=True):
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            else:
                resp = self.agent.execute_openai_chat(fake_openai_req, stream=False)
                # Format into Anthropic response
                anthropic_resp = {
                    "id": resp.get("id", "msg_default"),
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": resp["choices"][0]["message"].get("content", "")}],
                    "model": target_model,
                    "stop_reason": "end_turn",
                    "usage": resp.get("usage", {"input_tokens": 10, "output_tokens": 10})
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(anthropic_resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8080, gateway: str = DEFAULT_GATEWAY_URL, key: str = DEFAULT_API_KEY):
    agent = AIStudioAgent(gateway_url=gateway, api_key=key)
    GatewayHTTPRequestHandler.agent = agent
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), GatewayHTTPRequestHandler)
    print("=" * 65)
    print(" 🚀 AI STUDIO UNIVERSAL AGENT & GATEWAY SERVER")
    print("=" * 65)
    print(f" ▶ Local Endpoint:   http://127.0.0.1:{port}/v1/chat/completions")
    print(f" ▶ Anthropic Compat: http://127.0.0.1:{port}/v1/messages")
    print(f" ▶ Models Catalog:   http://127.0.0.1:{port}/v1/models")
    print(f" ▶ Health Endpoint:  http://127.0.0.1:{port}/healthz")
    print(f" ▶ Upstream Gateway: {gateway}")
    print("-" * 65)
    print(" Ready to serve IDEs (Zed, Cursor, Cline, Claude Code) and SDKs!")
    print(" Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Shutting down Gateway Server.")


def main():
    parser = argparse.ArgumentParser(description="AI Studio Universal Autonomous Agent")
    parser.add_argument("prompt", nargs="?", help="Direct prompt to query the Agent")
    parser.add_argument("--serve", action="store_true", help="Launch OpenAI/Anthropic compatible Gateway Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default: 8080)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY_URL, help="Upstream AI Studio Gateway URL")
    parser.add_argument("--key", default=DEFAULT_API_KEY, help="Master Gateway API Key")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable streaming SSE response")

    args = parser.parse_args()

    if args.serve:
        run_server(port=args.port, gateway=args.gateway, key=args.key)
    elif args.prompt:
        agent = AIStudioAgent(gateway_url=args.gateway, api_key=args.key)
        print(f"\n[AI Studio Agent] Model: {args.model}\n" + "=" * 50)
        if args.stream:
            for chunk in agent.stream(args.prompt, model=args.model):
                print(chunk, end="", flush=True)
            print("\n" + "=" * 50 + "\n")
        else:
            resp = agent.chat(args.prompt, model=args.model)
            content = resp["choices"][0]["message"].get("content", "")
            print(content)
            print("=" * 50 + "\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
