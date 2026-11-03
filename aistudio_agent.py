"""
AI Studio Autonomous Agent & Gateway Server (Fast Edition v2)
=============================================================
A fully self-contained AI Studio proxy server, CLI agent, and Python SDK.
Zero external dependencies - 100% Python standard library compatible.

v2 performance highlights:
  * Keep-alive HTTP/1.1 connection pool  -> no TCP/TLS handshake per request
  * HTTP/1.1 chunked SSE + TCP_NODELAY    -> smooth, low-latency token streaming
  * Upstream failover with priority      -> gateway first, Google API fallback
  * reasoning_effort control             -> skip thinking tokens for fast replies

Author: Antigravity & RealTrueHieu
Version: 2.0.0
"""

import os
import sys
import time
import json
import uuid
import argparse
import threading
import http.server
from typing import Dict, Any, List, Optional, Generator, Union, Iterator

from payload_transformer import AIStudioPayloadTransformer, normalize_model_name, AnthropicStreamTranslator
from session_manager import AccountSession, AIStudioSessionPool
from http_pool import (
    ConnectionPool,
    UpstreamHTTPError,
    open_response,
    open_response_hedged,
    read_json,
    iter_sse_events,
)

# Make Windows cp1252 consoles safe for UTF-8 output (emoji, Vietnamese, ...)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Default Relay Endpoint (Online Cloud Gateway)
DEFAULT_GATEWAY_URL = os.environ.get("AISTUDIO_GATEWAY", "http://localhost:8080/v1")
DEFAULT_API_KEY = os.environ.get("AISTUDIO_API_KEY", "")
DEFAULT_MODEL = "ais/gemini-4-argon"
DONE_EVENT = b"data: [DONE]\n\n"

SUPPORTED_MODELS = [
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro (Flagship Reasoning & Coding)"},
    {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash (Ultra Fast Reasoning)"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash (Standard Speed)"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro (Preview Tier)"},
    {"id": "ais/gemini-4-argon", "name": "Gemini 4 Argon (Reasoning Preview)"},
    {"id": "ais/gemini-4-argon-high", "name": "Gemini 4 Argon High Budget"},
    {"id": "antigravity-preview-09-2026", "name": "Antigravity Preview Engine"}
]


class _ErrorEventStream:
    """Marker iterator for upstream failures on streaming requests."""

    def __init__(self, events: List[bytes], error_obj: Dict[str, Any]):
        self._it = iter(events)
        self.error_obj = error_obj

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        return next(self._it)


class AIStudioAgent:
    """Core Agent engine for executing queries against AI Studio backends.

    Requests are routed through the account pool with priority failover
    (cloud gateway first, official Google API as fallback) over pooled
    keep-alive HTTP/1.1 connections.
    """

    def __init__(
        self,
        gateway_url: str = DEFAULT_GATEWAY_URL,
        api_key: str = DEFAULT_API_KEY,
        official_gemini_key: Optional[str] = None,
        timeout: float = 120.0,
        default_effort: Optional[str] = "auto",
        hedge_ms: int = 0
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.api_key = api_key
        self.official_gemini_key = official_gemini_key or os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout
        # Auto-injected when the client sends no reasoning control at all
        # (None / "auto" = never inject; "none" = skip thinking tokens = fastest)
        self.default_effort = None if default_effort in ("auto", "", None) else default_effort
        self.hedge_ms = max(0, int(hedge_ms or 0))

        self.conn_pool = ConnectionPool(max_per_host=8, idle_timeout=120.0)
        self.pool = AIStudioSessionPool()

        # Primary: cloud gateway (priority 0)
        self.pool.add_account(
            AccountSession(
                identifier="primary-cloud-gateway",
                auth_type="gateway",
                credential=self.api_key,
                endpoint=self.gateway_url,
                priority=0
            )
        )

        # Fallback: official Google Gemini API (priority 10) - native AI Studio
        # payload transformation, used when the gateway is unreachable.
        if self.official_gemini_key:
            self.pool.add_account(
                AccountSession(
                    identifier="official-google-api",
                    auth_type="api_key",
                    credential=self.official_gemini_key,
                    endpoint="https://generativelanguage.googleapis.com",
                    priority=10
                )
            )

    # ------------------------------------------------------------------
    # Public SDK API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        thinking_budget: Optional[int] = None,
        reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronous non-streaming query returning OpenAI format response."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        req_body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if thinking_budget:
            req_body["max_thinking_tokens"] = thinking_budget
        elif reasoning_effort:
            req_body["reasoning_effort"] = reasoning_effort
        return self.execute_openai_chat(req_body, stream=False)

    def stream(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        thinking_budget: Optional[int] = None,
        reasoning_effort: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streaming generator yielding text and reasoning tokens."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        req_body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        if thinking_budget:
            req_body["max_thinking_tokens"] = thinking_budget
        elif reasoning_effort:
            req_body["reasoning_effort"] = reasoning_effort

        for event in self.execute_openai_chat(req_body, stream=True):
            for piece in self._parse_sse_event_text(event):
                yield piece

    @staticmethod
    def _parse_sse_event_text(event: bytes) -> Generator[str, None, None]:
        """Extracts content / reasoning / error text from one raw SSE event (bytes)."""
        for raw in event.splitlines():
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if isinstance(obj.get("error"), dict):
                yield f"\n[Error] {obj['error'].get('message', 'upstream error')}\n"
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("reasoning_content"):
                yield f"\n[Thinking] {delta['reasoning_content']}"
            if delta.get("content"):
                yield delta["content"]

    # ------------------------------------------------------------------
    # Dispatcher with priority failover through the account pool
    # ------------------------------------------------------------------

    def execute_openai_chat(self, req_data: Dict[str, Any], stream: bool = False) -> Any:
        """
        Main dispatcher for /v1/chat/completions.
        Returns a dict (non-stream) or an iterator of raw SSE event bytes (stream).
        """
        req_data = self._apply_default_effort(req_data)
        model = req_data.get("model", DEFAULT_MODEL)
        tried = set()
        last_error: Optional[Exception] = None

        while True:
            account = self.pool.get_next_account(model)
            if account is None or account.identifier in tried:
                break
            tried.add(account.identifier)
            try:
                if account.auth_type == "api_key":
                    result = self._call_google_direct(account, req_data, stream)
                else:
                    result = self._call_gateway(account, req_data, stream)
                account.mark_success()
                return result
            except UpstreamHTTPError as e:
                account.mark_failure(f"HTTP {e.status}: {e.body[:200]}", e.status)
                last_error = e
            except Exception as e:
                account.mark_failure(str(e), 599)
                last_error = e

        return self._error_result(last_error, stream)

    # Fields that count as explicit reasoning control by the client
    _REASONING_FIELDS = ("reasoning_effort", "max_thinking_tokens", "thinking", "reasoning")

    def _apply_default_effort(self, req_data: Dict[str, Any]) -> Dict[str, Any]:
        """Injects default reasoning_effort when the client sent no reasoning control.

        IDE clients (Zed, Cursor, Claude Code) rarely send reasoning fields, so
        without this every request burns thinking tokens. Explicit client fields
        are always respected.
        """
        if not self.default_effort:
            return req_data
        if any(f in req_data for f in self._REASONING_FIELDS):
            return req_data
        out = dict(req_data)
        out["reasoning_effort"] = self.default_effort
        return out

    def _error_result(self, err: Optional[Exception], stream: bool):
        err_obj = {
            "error": {
                "message": f"All upstream accounts failed. Last error: {err}",
                "type": "server_error",
                "code": 503
            }
        }
        if stream:
            events = [f"data: {json.dumps(err_obj)}\n\n".encode("utf-8"), DONE_EVENT]
            return _ErrorEventStream(events, err_obj)
        return err_obj

    # ------------------------------------------------------------------
    # Upstream transports (both use pooled keep-alive connections)
    # ------------------------------------------------------------------

    def _call_gateway(self, account: AccountSession, req_data: Dict[str, Any], stream: bool):
        """Raw OpenAI-protocol passthrough to the cloud gateway (fastest path)."""
        url = f"{account.endpoint}/chat/completions"
        headers = {
            "Authorization": f"Bearer {account.credential}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        body = json.dumps(req_data, separators=(",", ":")).encode("utf-8")
        resp = open_response_hedged(self.conn_pool, url, "POST", headers, body,
                                    timeout=self.timeout, hedge_ms=self.hedge_ms)
        if stream:
            return iter_sse_events(resp)  # pass raw bytes straight to the client
        return read_json(resp)

    def _call_google_direct(self, account: AccountSession, req_data: Dict[str, Any], stream: bool):
        """Native AI Studio payload path: transform to GenerateContent, call Google directly."""
        target_model, aistudio_payload = AIStudioPayloadTransformer.openai_to_aistudio(req_data)
        req_id = f"chatcmpl_{uuid.uuid4().hex[:16]}"
        endpoint_method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        url = f"{account.endpoint}/v1beta/models/{target_model}:{endpoint_method}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": account.credential
        }
        body = json.dumps(aistudio_payload, separators=(",", ":")).encode("utf-8")
        resp = open_response_hedged(self.conn_pool, url, "POST", headers, body,
                                    timeout=self.timeout, hedge_ms=self.hedge_ms)

        if stream:
            def generate() -> Generator[bytes, None, None]:
                saw_done = False
                try:
                    for event in iter_sse_events(resp):
                        out = []
                        for raw in event.splitlines():
                            line = raw.decode("utf-8", errors="replace")
                            chunk = AIStudioPayloadTransformer.aistudio_to_openai_chunk(
                                line, target_model, req_id)
                            if chunk:
                                if "[DONE]" in chunk:
                                    saw_done = True
                                out.append(chunk)
                        if out:
                            yield "".join(out).encode("utf-8")
                    if not saw_done:
                        yield DONE_EVENT
                except Exception:
                    yield DONE_EVENT
                    raise
            return generate()

        raw_data = read_json(resp)
        return AIStudioPayloadTransformer.aistudio_to_openai_response(raw_data, target_model, req_id)


class GatewayHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    OpenAI / Anthropic compatible gateway with low-latency streaming.

    v2 fixes vs v1:
      * protocol_version HTTP/1.1 + chunked transfer for SSE  (v1 hung clients
        forever: HTTP/1.0 + manual "Connection: keep-alive" + no framing)
      * disable_nagle_algorithm -> TCP_NODELAY, no 40-200ms lag between chunks
      * byte-level SSE passthrough (no per-line decode/encode round trips)
    """

    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    agent: AIStudioAgent = None
    server_version = "AIStudioGateway/2.0"

    def log_message(self, format, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), format % args))

    # ---------------- generic helpers ----------------

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def _send_json(self, obj, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    # ---------------- SSE with chunked framing ----------------

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.send_cors_headers()
        self.end_headers()

    def _sse_write(self, data: bytes):
        # HTTP/1.1 chunked framing: <hex-length>\r\n<data>\r\n
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _sse_finish(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # ---------------- GET ----------------

    def do_GET(self):
        if self.path in ["/v1/models", "/models"]:
            model_list = {
                "object": "list",
                "data": [
                    {"id": m["id"], "object": "model", "created": 1793620000, "owned_by": "google"}
                    for m in SUPPORTED_MODELS
                ]
            }
            self._send_json(model_list)
        elif self.path in ["/healthz", "/status"]:
            self._send_json({
                "status": "healthy",
                "version": "2.1.0",
                "pool": self.agent.pool.get_status_summary(),
                "connections": self.agent.conn_pool.stats()
            })
        else:
            self._send_json({"error": {"message": f"Not found: {self.path}",
                                       "type": "invalid_request_error"}}, 404)

    # ---------------- POST ----------------

    def do_POST(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
        except ValueError:
            content_len = 0
        post_data = self.rfile.read(content_len) if content_len else b""

        try:
            req_json = json.loads(post_data.decode("utf-8"))
        except Exception:
            self._send_json({"error": {"message": "Invalid JSON payload",
                                       "type": "invalid_request_error"}}, 400)
            return

        is_stream = bool(req_json.get("stream", False))

        # 1. OpenAI /v1/chat/completions
        if self.path in ["/v1/chat/completions", "/chat/completions"]:
            if is_stream:
                gen = self.agent.execute_openai_chat(req_json, stream=True)
                if isinstance(gen, _ErrorEventStream):
                    self._send_json(gen.error_obj, 502)
                else:
                    self._stream_upstream(gen)
            else:
                resp = self.agent.execute_openai_chat(req_json, stream=False)
                status = 502 if isinstance(resp, dict) and resp.get("error") else 200
                self._send_json(resp, status)

        # 2. Anthropic /v1/messages
        elif self.path in ["/v1/messages", "/messages"]:
            self._handle_anthropic(req_json, is_stream)

        else:
            self._send_json({"error": {"message": f"Not found: {self.path}",
                                       "type": "invalid_request_error"}}, 404)

    # ---------------- streaming bodies ----------------

    def _stream_upstream(self, gen):
        """Streams raw upstream SSE events to the client with chunked framing."""
        self._start_sse()
        done_seen = False
        try:
            for event in gen:
                if b"[DONE]" in event:
                    done_seen = True
                self._sse_write(event)
            if not done_seen:
                self._sse_write(DONE_EVENT)
            self._sse_finish()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True  # client hung up mid-stream
        except Exception as e:
            self.log_message("stream error: %s", e)
            try:
                err = {"error": {"message": str(e), "type": "server_error"}}
                self._sse_write(f"data: {json.dumps(err)}\n\n".encode("utf-8"))
                self._sse_write(DONE_EVENT)
                self._sse_finish()
            except Exception:
                self.close_connection = True

    def _handle_anthropic(self, req_json: Dict[str, Any], is_stream: bool):
        """Real Anthropic <-> OpenAI translation (system, tools, thinking preserved)."""
        openai_req = AIStudioPayloadTransformer.anthropic_to_openai(req_json)
        model_name = req_json.get("model", "gemini-2.5-pro")

        if is_stream:
            openai_req["stream"] = True
            gen = self.agent.execute_openai_chat(openai_req, stream=True)
            if isinstance(gen, _ErrorEventStream):
                msg = gen.error_obj.get("error", {}).get("message", "upstream error")
                self._send_json({"type": "error",
                                 "error": {"type": "api_error", "message": msg}}, 502)
                return
            self._stream_anthropic(gen, model_name)
        else:
            resp = self.agent.execute_openai_chat(openai_req, stream=False)
            status = 502 if isinstance(resp, dict) and resp.get("error") else 200
            self._send_json(
                AIStudioPayloadTransformer.openai_to_anthropic_response(resp, model_name),
                status
            )

    def _stream_anthropic(self, gen, model_name: str):
        """Translates OpenAI SSE chunks into Anthropic Messages SSE events."""
        translator = AnthropicStreamTranslator(model_name)
        self._start_sse()
        try:
            for event in gen:
                for raw in event.splitlines():
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except ValueError:
                        continue
                    out = translator.handle_chunk(obj)
                    if out:
                        self._sse_write(out.encode("utf-8"))
            tail = translator.finish()
            if tail:
                self._sse_write(tail.encode("utf-8"))
            self._sse_finish()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True
        except Exception as e:
            self.log_message("anthropic stream error: %s", e)
            try:
                err_event = f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
                self._sse_write(err_event.encode("utf-8"))
                self._sse_finish()
            except Exception:
                self.close_connection = True


def run_server(
    port: int = 8080,
    host: str = "127.0.0.1",
    gateway: str = DEFAULT_GATEWAY_URL,
    key: str = DEFAULT_API_KEY,
    timeout: float = 120.0,
    default_effort: str = "auto",
    hedge_ms: int = 0,
    keepalive_interval: float = 45.0
):
    agent = AIStudioAgent(gateway_url=gateway, api_key=key, timeout=timeout,
                         default_effort=default_effort, hedge_ms=hedge_ms)
    GatewayHTTPRequestHandler.agent = agent

    # Fire up the server first, then warm the upstream pool in the background
    # so the FIRST real request pays zero TCP/TLS handshake cost.
    server = http.server.ThreadingHTTPServer((host, port), GatewayHTTPRequestHandler)
    server.daemon_threads = True

    warm_headers = {"Authorization": f"Bearer {key}"}
    warm_url = f"{agent.gateway_url}/models"
    if keepalive_interval and keepalive_interval > 0:
        threading.Thread(
            target=agent.conn_pool.warm_up,
            args=(warm_url, warm_headers, 2),
            daemon=True,
            name="pool-warmup"
        ).start()
        agent.conn_pool.start_keepalive(warm_url, warm_headers, interval=keepalive_interval)

    scope = "localhost only" if host in ("127.0.0.1", "localhost") else f"ALL interfaces ({host}) - LAN can access!"
    print("=" * 65)
    print(" AI STUDIO UNIVERSAL AGENT & GATEWAY SERVER v2.1 (Squeezed Edition)")
    print("=" * 65)
    print(f" > OpenAI Endpoint:   http://127.0.0.1:{port}/v1/chat/completions")
    print(f" > Anthropic Compat:  http://127.0.0.1:{port}/v1/messages")
    print(f" > Models Catalog:    http://127.0.0.1:{port}/v1/models")
    print(f" > Health Endpoint:   http://127.0.0.1:{port}/healthz")
    print(f" > Upstream Gateway:  {gateway}")
    print(f" > Binding:           {scope}")
    print(f" > Default effort:    {default_effort} | Hedge: {hedge_ms}ms | Keepalive: {keepalive_interval}s")
    print("-" * 65)
    print(" Warm-pooled upstream + TCP_NODELAY + chunked SSE + effort injection.")
    print(" Ready to serve IDEs (Zed, Cursor, Cline, Claude Code) and SDKs!")
    print(" Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Shutting down Gateway Server.")
    finally:
        agent.conn_pool.stop_keepalive()
        agent.conn_pool.close_all()
        server.server_close()


def run_interactive(agent: AIStudioAgent, args):
    """Multi-turn REPL: connection stays warm between questions, context is kept."""
    model = args.model
    effort = None if args.effort == "auto" else args.effort
    budget = args.thinking_budget
    history: List[Dict[str, Any]] = []

    print("\n[+] Interactive chat - connection stays warm between turns.")
    print("    Commands: /exit  /new (clear context)  /model <id>  /effort <x>  /budget <tokens>")
    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[+] Bye!")
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            print("[+] Bye!")
            return
        if line == "/new":
            history = []
            print("[+] Context cleared.")
            continue
        if line.startswith("/model "):
            model = line[7:].strip()
            print(f"[+] Model set to {model}")
            continue
        if line.startswith("/effort "):
            val = line[8:].strip()
            effort = None if val == "auto" else val
            budget = None
            print(f"[+] Effort set to {val}")
            continue
        if line.startswith("/budget "):
            budget = int(line[8:].strip() or 0)
            effort = None
            print(f"[+] Thinking budget set to {budget}")
            continue

        history.append({"role": "user", "content": line})
        print("Assistant> ", end="", flush=True)
        collected: List[str] = []
        try:
            for token in agent.stream(history, model=model,
                                      thinking_budget=budget, reasoning_effort=effort):
                print(token, end="", flush=True)
                if not token.startswith("\n[Thinking]") and not token.startswith("\n[Error]"):
                    collected.append(token)
        except KeyboardInterrupt:
            print("\n[+] (interrupted)")
        history.append({"role": "assistant", "content": "".join(collected).strip()})


def main():
    parser = argparse.ArgumentParser(description="AI Studio Universal Autonomous Agent (Fast Edition v2.1)")
    parser.add_argument("prompt", nargs="?", help="Direct prompt to query the Agent")
    parser.add_argument("--serve", action="store_true", help="Launch OpenAI/Anthropic compatible Gateway Server")
    parser.add_argument("--chat", action="store_true", help="Interactive multi-turn REPL (warm connection + context)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server bind address (default: 127.0.0.1; use 0.0.0.0 to expose on LAN)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default: 8080)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY_URL, help="Upstream AI Studio Gateway URL")
    parser.add_argument("--key", default=DEFAULT_API_KEY, help="Master Gateway API Key")
    parser.add_argument("--effort", default="none",
                        choices=["none", "low", "medium", "high", "auto"],
                        help="reasoning_effort sent upstream (default: none = fastest, skip thinking tokens)")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Enable thinking budget in tokens (overrides --effort)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming token output")
    parser.add_argument("--timeout", type=float, default=120.0, help="Upstream timeout in seconds (default: 120)")
    parser.add_argument("--hedge-ms", type=int, default=0,
                        help="Tail-tolerant hedged requests: if upstream hasn't answered within N ms, "
                             "fire a duplicate and take the first reply (default: 0 = off)")
    parser.add_argument("--default-effort", default="auto",
                        choices=["none", "low", "medium", "high", "auto"],
                        help="Server: effort injected for client requests that send no reasoning control "
                             "(default: none = fast IDE responses)")
    parser.add_argument("--keepalive-interval", type=float, default=45.0,
                        help="Server: upstream keep-alive ping interval in seconds, 0 disables (default: 45)")
    parser.add_argument("--stream", action="store_true", default=True, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.serve:
        run_server(port=args.port, host=args.host, gateway=args.gateway, key=args.key,
                   timeout=args.timeout, default_effort=args.default_effort,
                   hedge_ms=args.hedge_ms, keepalive_interval=args.keepalive_interval)
    elif args.chat:
        agent = AIStudioAgent(gateway_url=args.gateway, api_key=args.key, timeout=args.timeout)
        run_interactive(agent, args)
    elif args.prompt:
        agent = AIStudioAgent(gateway_url=args.gateway, api_key=args.key, timeout=args.timeout)
        effort = None if args.effort == "auto" else args.effort
        print(f"\n[AI Studio Agent v2] Model: {args.model} | Effort: {args.effort}")
        print("=" * 50)
        try:
            if args.no_stream:
                resp = agent.chat(args.prompt, model=args.model,
                                  thinking_budget=args.thinking_budget, reasoning_effort=effort)
                if isinstance(resp, dict) and resp.get("error"):
                    print(f"[Error] {resp['error'].get('message', 'upstream error')}")
                    sys.exit(1)
                content = resp["choices"][0]["message"].get("content", "")
                print(content)
            else:
                for token in agent.stream(args.prompt, model=args.model,
                                          thinking_budget=args.thinking_budget, reasoning_effort=effort):
                    print(token, end="", flush=True)
            print("\n" + "=" * 50 + "\n")
        except KeyboardInterrupt:
            print("\n[+] Interrupted.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
