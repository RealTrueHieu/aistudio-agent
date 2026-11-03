"""
Fast Keep-Alive HTTP Connection Pool & SSE Reader
-------------------------------------------------
Persistent HTTP/1.1 connections for upstream gateway calls.
Eliminates per-request TCP/TLS handshake cost and SSE read latency.

v2.1 additions:
  * TCP_NODELAY on all upstream sockets (no Nagle + delayed-ACK stalls)
  * warm_up() - pre-establish TLS connections at startup
  * start_keepalive() - background refresher keeps pooled connections hot
  * open_response_hedged() - tail-tolerant hedged requests (race duplicates)
Zero external dependencies - 100% Python standard library compatible.
"""

import time
import json
import socket
import threading
import http.client
from urllib.parse import urlsplit
from typing import Any, Dict, Optional


class UpstreamHTTPError(Exception):
    """Raised when an upstream endpoint answers with HTTP >= 400."""

    def __init__(self, status: int, body: str, url: str = ""):
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")


class _NoDelayHTTPConnection(http.client.HTTPConnection):
    """HTTP connection with TCP_NODELAY (request headers + body leave in one burst)."""

    def connect(self):
        super().connect()
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


class _NoDelayHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection with TCP_NODELAY (request headers + body leave in one burst)."""

    def connect(self):
        super().connect()
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


class ConnectionPool:
    """Thread-safe pool of keep-alive HTTP/1.1 connections, keyed by (scheme, host, port).

    Reusing connections removes the TCP + TLS handshake (~0.3-1s) from every
    request and keeps SSE streams hot.
    """

    def __init__(self, max_per_host: int = 8, idle_timeout: float = 120.0):
        self._pool: Dict[Any, list] = {}
        self._lock = threading.Lock()
        self.max_per_host = max_per_host
        self.idle_timeout = idle_timeout
        self._keepalive_stop = threading.Event()

    @staticmethod
    def _key(url: str):
        p = urlsplit(url)
        scheme = p.scheme or "http"
        host = p.hostname or "127.0.0.1"
        port = p.port or (443 if scheme == "https" else 80)
        return scheme, host, port

    def _new_connection(self, key, timeout: float) -> http.client.HTTPConnection:
        scheme, host, port = key
        cls = _NoDelayHTTPSConnection if scheme == "https" else _NoDelayHTTPConnection
        return cls(host, port, timeout=timeout)

    def acquire(self, url: str, timeout: float = 30.0, fresh: bool = False) -> http.client.HTTPConnection:
        """Returns a live connection for the URL (pooled if possible, else new)."""
        key = self._key(url)
        if not fresh:
            with self._lock:
                stack = self._pool.get(key)
                while stack:
                    conn, ts = stack.pop()
                    if time.time() - ts < self.idle_timeout:
                        return conn
                    self.close(conn)
        return self._new_connection(key, timeout)

    def release(self, conn: Optional[http.client.HTTPConnection]):
        """Returns a healthy, fully-drained connection to the pool."""
        if conn is None:
            return
        key = getattr(conn, "_pool_key", None)
        if key is None:
            self.close(conn)
            return
        with self._lock:
            stack = self._pool.setdefault(key, [])
            if len(stack) < self.max_per_host:
                stack.append((conn, time.time()))
                return
        self.close(conn)

    @staticmethod
    def close(conn):
        try:
            conn.close()
        except Exception:
            pass

    def discard(self, conn: Optional[http.client.HTTPConnection]):
        """Drops a broken connection (never returns it to the pool)."""
        self.close(conn)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"hosts": len(self._pool), "idle": sum(len(s) for s in self._pool.values())}

    # ------------------------------------------------------------------
    # Warm-up & keep-alive refresh
    # ------------------------------------------------------------------

    def warm_up(self, url: str, headers: Optional[Dict[str, str]] = None,
                count: int = 2, timeout: float = 20.0):
        """Pre-establishes `count` TLS connections so the first real request
        pays zero handshake cost. Safe to call at server startup."""
        for _ in range(count):
            try:
                resp = open_response(self, url, "GET", headers, timeout=timeout)
                finish_response(resp)
            except Exception:
                pass

    def start_keepalive(self, url: str, headers: Optional[Dict[str, str]] = None,
                        interval: float = 45.0):
        """Background thread that pings the upstream every `interval` seconds,
        keeping one pooled connection permanently hot (and self-healing when
        the upstream closes it)."""
        if interval <= 0:
            return

        def loop():
            while not self._keepalive_stop.wait(interval):
                try:
                    resp = open_response(self, url, "GET", headers, timeout=15.0)
                    finish_response(resp)
                except Exception:
                    pass  # next cycle opens a fresh connection automatically

        threading.Thread(target=loop, daemon=True, name="pool-keepalive").start()

    def stop_keepalive(self):
        self._keepalive_stop.set()

    def close_all(self):
        self._keepalive_stop.set()
        with self._lock:
            for stack in self._pool.values():
                for conn, _ in stack:
                    self.close(conn)
            self._pool.clear()


def open_response(
    pool: ConnectionPool,
    url: str,
    method: str = "POST",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
    timeout: float = 60.0,
) -> http.client.HTTPResponse:
    """
    Sends a request over a pooled keep-alive connection and returns the raw
    HTTPResponse (status < 400). The caller MUST finish reading it and then
    call finish_response() / discard_response().

    Stale pooled connections are retried once with a fresh connection.
    Raises UpstreamHTTPError when the upstream answers HTTP >= 400.
    """
    p = urlsplit(url)
    path = p.path or "/"
    if p.query:
        path += "?" + p.query

    hdrs = dict(headers or {})
    hdrs.setdefault("Accept-Encoding", "identity")  # avoid compressed SSE buffering
    hdrs.setdefault("Connection", "keep-alive")

    conn = pool.acquire(url, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
    except Exception:
        # Stale keep-alive connection (closed by server between requests):
        # transparently retry once with a brand-new connection.
        pool.discard(conn)
        conn = pool.acquire(url, timeout=timeout, fresh=True)
        try:
            conn.request(method, path, body=body, headers=hdrs)
            resp = conn.getresponse()
        except Exception:
            pool.discard(conn)
            raise

    conn._pool_key = pool._key(url)
    resp._pool = pool
    resp._conn = conn

    if resp.status >= 400:
        try:
            err_body = resp.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        finish_response(resp)
        raise UpstreamHTTPError(resp.status, err_body, url)
    return resp


def open_response_hedged(
    pool: ConnectionPool,
    url: str,
    method: str = "POST",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
    timeout: float = 60.0,
    hedge_ms: int = 0,
) -> http.client.HTTPResponse:
    """
    Tail-tolerant hedged request: fires the request immediately, and if no
    response headers arrive within `hedge_ms`, fires a duplicate on a fresh
    connection. Whichever answers FIRST wins; the loser is closed when it lands.

    Note: a slow upstream means both requests may execute server-side
    (possible double token cost on tail-latency cases only).
    """
    if not hedge_ms or hedge_ms <= 0:
        return open_response(pool, url, method, headers, body, timeout)

    state: Dict[str, Any] = {}
    cv = threading.Condition()

    def attempt():
        try:
            resp = open_response(pool, url, method, headers, body, timeout)
        except Exception as e:
            resp = e
        with cv:
            if "winner" not in state:
                state["winner"] = resp      # first to land claims the race
                cv.notify_all()
            elif not isinstance(resp, Exception):
                discard_response(resp)      # lost the race -> close it

    threading.Thread(target=attempt, daemon=True).start()
    with cv:
        if "winner" not in state:
            if not cv.wait(hedge_ms / 1000.0):
                threading.Thread(target=attempt, daemon=True).start()
                cv.wait(max(timeout, 30.0))  # block until one of them lands

    winner = state.get("winner")
    if winner is None:
        raise UpstreamHTTPError(599, "hedged request timed out with no response", url)
    if isinstance(winner, Exception):
        raise winner
    return winner


def finish_response(resp):
    """Returns the (fully drained) connection to the pool."""
    pool = getattr(resp, "_pool", None)
    conn = getattr(resp, "_conn", None)
    if pool is None or conn is None:
        return
    if getattr(resp, "will_close", True):
        pool.discard(conn)
    else:
        pool.release(conn)


def discard_response(resp):
    """Drops a broken / half-read response's connection."""
    pool = getattr(resp, "_pool", None)
    conn = getattr(resp, "_conn", None)
    if pool is not None and conn is not None:
        pool.discard(conn)


def read_json(resp) -> Any:
    """Fully reads a JSON response, releases the connection, parses the body."""
    try:
        data = resp.read()
        finish_response(resp)
    except Exception:
        discard_response(resp)
        raise
    if not data:
        return {}
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError:
        return {"_raw": data.decode("utf-8", errors="replace")}


def iter_sse_events(resp):
    """
    Yields complete SSE events (raw bytes, blank-line terminated) the moment
    they arrive from upstream. Releases the connection when the stream ends
    cleanly, discards it on error / early close.
    """
    completed = False
    try:
        pending = []
        while True:
            line = resp.readline()
            if not line:
                completed = True
                break
            pending.append(line)
            if line in (b"\n", b"\r\n"):  # blank line terminates an SSE event
                yield b"".join(pending)
                pending = []
        if pending:
            yield b"".join(pending)
    finally:
        if completed:
            finish_response(resp)
        else:
            discard_response(resp)
