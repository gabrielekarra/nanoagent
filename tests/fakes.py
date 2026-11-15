"""Fake model servers that speak the real wire formats, for testing the adapters.

Each one is a real HTTP server on localhost. It answers with replies scripted
with `nanoagent.testing.reply`, rendered in its provider's format, and keeps
every request it received. A script entry made with `failure` answers with an
HTTP error instead.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def failure(status, message, shape="openai"):
    """A scripted HTTP error. `shape` picks the error body format."""
    return {"failure": status, "message": message, "shape": shape}


class FakeServer:
    def __init__(self, *script):
        self.script = list(script)
        self.requests = []  # {"path", "body", "headers"}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                server.handle(self, self.path, body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, args=(0.01,), daemon=True).start()

    @property
    def bodies(self):
        return [request["body"] for request in self.requests]

    def next(self, handler):
        """The next scripted entry, or None after sending its error."""
        if not self.script:
            self.send_json(handler, 500, {"error": {"message": "fake server: the script has no replies left"}})
            return None
        entry = self.script.pop(0)
        if "failure" in entry:
            if entry["shape"] == "ollama":
                payload = {"error": entry["message"]}
            elif entry["shape"] == "anthropic":
                payload = {"type": "error", "error": {"type": "invalid_request_error", "message": entry["message"]}}
            else:
                payload = {"error": {"message": entry["message"], "type": "invalid_request_error"}}
            self.send_json(handler, entry["failure"], payload)
            return None
        return entry

    @staticmethod
    def send_json(handler, status, payload):
        data = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def send_stream(handler, content_type, lines):
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.end_headers()
        for line in lines:
            handler.wfile.write(line.encode())
        handler.wfile.flush()

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


# --- Anthropic Messages API -----------------------------------------------------

ANTHROPIC_STOPS = {"end": "end_turn", "tool": "tool_use", "length": "max_tokens", "refusal": "refusal"}


class FakeAnthropic(FakeServer):
    def tool_results(self, request=-1):
        return self.bodies[request]["messages"][-1]["content"]

    def handle(self, handler, path, body):
        entry = self.next(handler)
        if entry is None:
            return
        message = {
            "id": f"msg_{len(self.requests)}",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": entry["content"],
            "stop_reason": ANTHROPIC_STOPS[entry["stop"]],
            "stop_sequence": None,
            "usage": {"input_tokens": entry["tokens"], "output_tokens": 1},
        }
        if not body.get("stream"):
            return self.send_json(handler, 200, message)
        events = [("message_start", {"message": {**message, "content": [], "stop_reason": None}})]
        for index, block in enumerate(message["content"]):
            if block["type"] == "text":
                start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
            else:
                start = {**block, "input": {}}
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            events += [
                ("content_block_start", {"index": index, "content_block": start}),
                ("content_block_delta", {"index": index, "delta": delta}),
                ("content_block_stop", {"index": index}),
            ]
        end = {"stop_reason": message["stop_reason"], "stop_sequence": None}
        events += [("message_delta", {"delta": end, "usage": {"output_tokens": 1}}), ("message_stop", {})]
        lines = [f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n" for name, data in events]
        self.send_stream(handler, "text/event-stream", lines)


# --- OpenAI Chat Completions ------------------------------------------------------

OPENAI_FINISH = {"end": "stop", "tool": "tool_calls", "length": "length", "refusal": "content_filter"}


class FakeOpenAI(FakeServer):
    """Extra keys on a scripted reply change how it is sent:

    finish        the finish_reason to report instead of the usual one
    split         send each tool call's arguments in this many fragments
    object_args   send arguments as a JSON object instead of a string
    chunks        send each text block in pieces of this many characters
    usage         False to report no token usage
    """

    def handle(self, handler, path, body):
        entry = self.next(handler)
        if entry is None:
            return
        texts = [b["text"] for b in entry["content"] if b["type"] == "text"]
        calls = [b for b in entry["content"] if b["type"] == "tool_use"]
        finish = entry.get("finish", OPENAI_FINISH[entry["stop"]])
        usage = {"prompt_tokens": entry["tokens"] - 1, "completion_tokens": 1} if entry.get("usage", True) else None

        def arguments(block):
            return block["input"] if entry.get("object_args") else json.dumps(block["input"])

        if not body.get("stream"):
            message = {"role": "assistant", "content": "".join(texts) or None}
            if calls:
                message["tool_calls"] = [
                    {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": arguments(b)}}
                    for b in calls
                ]
            payload = {"id": "c1", "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
            if usage:
                payload["usage"] = usage
            return self.send_json(handler, 200, payload)

        chunks = []
        size = entry.get("chunks")
        for piece in texts:
            for start in range(0, len(piece), size or max(len(piece), 1)):
                chunks.append({"content": piece[start : start + (size or len(piece))]})
        for index, block in enumerate(calls):
            args = arguments(block)
            parts = entry.get("split", 1)
            if isinstance(args, str) and parts > 1:
                step = -(-len(args) // parts)
                fragments = [args[i : i + step] for i in range(0, len(args), step)]
            else:
                fragments = [args]
            for number, fragment in enumerate(fragments):
                call = {"index": index, "function": {"arguments": fragment}}
                if number == 0:
                    call.update(id=block["id"], type="function")
                    call["function"]["name"] = block["name"]
                chunks.append({"tool_calls": [call]})
        events = [{"choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": None}]} for delta in chunks]
        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        if usage and (body.get("stream_options") or {}).get("include_usage"):
            events.append({"choices": [], "usage": usage})
        lines = [f"data: {json.dumps(event)}\n\n" for event in events] + ["data: [DONE]\n\n"]
        self.send_stream(handler, "text/event-stream", lines)


# --- Ollama native API -----------------------------------------------------------


class FakeOllama(FakeServer):
    def __init__(self, *script):
        super().__init__(*script)
        self.pulled = []

    def handle(self, handler, path, body):
        if path == "/api/pull":
            self.pulled.append(body["model"])
            return self.send_json(handler, 200, {"status": "success"})
        entry = self.next(handler)
        if entry is None:
            return
        texts = [b["text"] for b in entry["content"] if b["type"] == "text"]
        calls = [{"function": {"name": b["name"], "arguments": b["input"]}} for b in entry["content"] if b["type"] == "tool_use"]
        end = {
            "done": True,
            "done_reason": "length" if entry["stop"] == "length" else "stop",
            "prompt_eval_count": entry["tokens"] - 1,
            "eval_count": 1,
        }
        if not body.get("stream", True):
            message = {"role": "assistant", "content": "".join(texts)}
            return self.send_json(handler, 200, {"model": body["model"], "message": message, **end})
        lines = [{"message": {"role": "assistant", "content": t}, "done": False} for t in texts]
        if calls:
            lines.append({"message": {"role": "assistant", "content": "", "tool_calls": calls}, "done": False})
        lines.append({"message": {"role": "assistant", "content": ""}, **end})
        self.send_stream(handler, "application/x-ndjson", [json.dumps(line) + "\n" for line in lines])
