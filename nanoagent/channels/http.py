"""HTTP: talk to the agent from anything, including as if it were a model.

    agent.use(http(port=8080))

    POST /message               {"text": "...", "session": "optional"} -> {"answer": "..."}
                                with `Accept: text/event-stream`, the answer streams
    POST /v1/chat/completions   the Chat Completions format, streamed or not: any chat
                                interface or client library can use the agent, with
                                its tools and its memory, as a model

Bound to localhost by default. On any other address a token is required, and
every request must send `Authorization: Bearer <token>`. A conversation is
kept per session: the `session` field, or the `X-Session-Id` header.
Approvals cannot be asked over a single request, so tools that need one are
declined.
"""

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..runtime import Channel

LOCAL = ("127.0.0.1", "localhost", "::1")
DONE = object()


class HTTP(Channel):
    name = "http"
    can_approve = False

    def __init__(self, port=8080, host="127.0.0.1", token=None, timeout=600):
        self.port, self.host, self.token, self.timeout = port, host, token, timeout
        self.waiting = {}  # request id -> queue of streamed pieces, then the answer, then DONE
        self.server = None

    async def start(self, deliver):
        if self.host not in LOCAL and not self.token:
            raise ValueError(f"http(host={self.host!r}) is reachable from other machines: pass token=... to require one.")
        self.server = ThreadingHTTPServer((self.host, self.port), _handler(self))
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, args=(0.1,), daemon=True).start()

    async def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    async def send(self, peer, text):
        if peer in self.waiting:
            self.waiting[peer].put(("answer", text))
            self.waiting[peer].put(DONE)

    def on_text(self, peer, piece):
        if peer in self.waiting:
            self.waiting[peer].put(("text", piece))

    def ask_agent(self, text, session):
        """From a request thread: yield ("text", piece) while the agent writes, then ("answer", text)."""
        request_id = uuid.uuid4().hex
        pieces = self.waiting[request_id] = queue.Queue()
        try:
            self.post(request_id, text, session=f"http:{session}")
            deadline = time.monotonic() + self.timeout
            while True:
                item = pieces.get(timeout=max(0.1, deadline - time.monotonic()))
                if item is DONE:
                    return
                yield item
        finally:
            self.waiting.pop(request_id, None)


def _handler(channel):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            if channel.token and self.headers.get("Authorization") != f"Bearer {channel.token}":
                return self._json(401, {"error": {"message": "missing or wrong bearer token"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "the body must be JSON"}})
            session = body.get("session") or self.headers.get("X-Session-Id") or body.get("user") or "default"
            if self.path == "/message":
                return self._message(body, session)
            if self.path in ("/v1/chat/completions", "/chat/completions"):
                return self._completion(body, session)
            self._json(404, {"error": {"message": f"no route {self.path}"}})

        def _message(self, body, session):
            if not isinstance(body.get("text"), str):
                return self._json(400, {"error": {"message": "send {\"text\": \"...\"}"}})
            replies = channel.ask_agent(body["text"], session)
            if "text/event-stream" not in self.headers.get("Accept", ""):
                answer = next(text for kind, text in replies if kind == "answer")
                return self._json(200, {"answer": answer, "session": session})
            self._start_stream()
            for kind, text in replies:
                name = "text" if kind == "text" else "done"
                key = "text" if kind == "text" else "answer"
                self._event(f"event: {name}\ndata: {json.dumps({key: text})}\n\n")
            self._end_stream()

        def _completion(self, body, session):
            request = _last_user_text(body.get("messages") or [])
            if request is None:
                return self._json(400, {"error": {"message": "messages must end with a user message"}})
            model, created, identifier = body.get("model") or "nanoagent", int(time.time()), f"chatcmpl-{uuid.uuid4().hex[:12]}"
            replies = channel.ask_agent(request, session)
            if not body.get("stream"):
                answer = next(text for kind, text in replies if kind == "answer")
                return self._json(200, {
                    "id": identifier, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })

            def chunk(delta, finish=None):
                return {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            self._start_stream()
            self._event(f"data: {json.dumps(chunk({'role': 'assistant', 'content': ''}))}\n\n")
            streamed = False
            for kind, text in replies:
                if kind == "text":
                    streamed = True
                    self._event(f"data: {json.dumps(chunk({'content': text}))}\n\n")
                elif not streamed:  # the model wrote nothing while working: send the answer whole
                    self._event(f"data: {json.dumps(chunk({'content': text}))}\n\n")
            self._event(f"data: {json.dumps(chunk({}, 'stop'))}\n\n")
            self._event("data: [DONE]\n\n")
            self._end_stream()

        # --- writing -------------------------------------------------------------

        def _json(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _start_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def _event(self, text):
            data = text.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _end_stream(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return Handler


def _last_user_text(messages):
    if not messages or messages[-1].get("role") != "user":
        return None
    content = messages[-1].get("content")
    if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else None


def http(port=8080, host="127.0.0.1", token=None, **options):
    """An HTTP channel. See the module docstring."""
    return HTTP(port, host, token, **options)
