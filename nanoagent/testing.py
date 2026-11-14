"""Test an agent without calling a model.

`FakeModel` is a stand-in for the Messages API: a real HTTP server on
localhost that gives the replies you scripted. Your agent talks to it through
the real SDK, so requests are serialized and replies are streamed and parsed
exactly as in production. Only the model is fake.

    from nanoagent import Agent
    from nanoagent.testing import FakeModel, call, reply, text

    def test_weather_agent():
        with FakeModel(
            reply(call("get_weather", city="Rome")),   # the model asks for a tool
            reply(text("Sunny, no umbrella needed.")),  # then answers
        ) as model:
            agent = Agent(tools=[get_weather], client=model.client)
            assert agent.run("Umbrella in Rome?") == "Sunny, no umbrella needed."
            assert model.tool_results()[0]["content"] == "sunny"
"""

import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic

_ids = itertools.count(1)


def text(value):
    """A text block in a scripted reply."""
    return {"type": "text", "text": value}


def call(name, _id=None, **arguments):
    """A tool call in a scripted reply."""
    return {"type": "tool_use", "id": _id or f"toolu_{next(_ids)}", "name": name, "input": arguments}


def reply(*blocks, stop_reason=None, input_tokens=10):
    """One scripted model reply. The stop reason defaults to what the blocks imply."""
    if stop_reason is None:
        used_tool = any(block["type"] == "tool_use" for block in blocks)
        stop_reason = "tool_use" if used_tool else "end_turn"
    return {"content": list(blocks), "stop_reason": stop_reason, "input_tokens": input_tokens}


class FakeModel:
    """Serves scripted replies, in order, and records every request.

    script     the replies still to give; append to it at any time
    requests   the body of every request received
    client     an `anthropic.Anthropic` client pointed at this server
    url        the server address, for code that builds its own client
    """

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self.client = anthropic.Anthropic(api_key="fake", base_url=self.url, max_retries=0)
        threading.Thread(target=self._server.serve_forever, args=(0.01,), daemon=True).start()

    def tool_results(self, request=-1):
        """The tool results the agent sent in a request (the last one by default)."""
        return self.requests[request]["messages"][-1]["content"]

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def _handler(model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model.requests.append(body)
            if not model.script:
                error = {"type": "invalid_request_error", "message": "FakeModel: the script has no replies left"}
                return self._json(400, {"type": "error", "error": error})
            scripted = model.script.pop(0)
            message = {
                "id": f"msg_{len(model.requests)}",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": scripted["content"],
                "stop_reason": scripted["stop_reason"],
                "stop_sequence": None,
                "usage": {"input_tokens": scripted["input_tokens"], "output_tokens": 1},
            }
            if body.get("stream"):
                self._events(message)
            else:
                self._json(200, message)

        def _json(self, status, payload):
            payload = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _events(self, message):
            """Send a reply as the server-sent events of a streamed response."""
            events = [("message_start", {"message": {**message, "content": [], "stop_reason": None}})]
            for index, block in enumerate(message["content"]):
                if block["type"] == "text":
                    start = {"type": "text", "text": ""}
                    delta = {"type": "text_delta", "text": block["text"]}
                else:
                    start = {**block, "input": {}}
                    delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
                events += [
                    ("content_block_start", {"index": index, "content_block": start}),
                    ("content_block_delta", {"index": index, "delta": delta}),
                    ("content_block_stop", {"index": index}),
                ]
            end = {"stop_reason": message["stop_reason"], "stop_sequence": None}
            events += [
                ("message_delta", {"delta": end, "usage": {"output_tokens": 1}}),
                ("message_stop", {}),
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for name, data in events:
                line = json.dumps({"type": name, **data})
                self.wfile.write(f"event: {name}\ndata: {line}\n\n".encode())
            self.wfile.flush()

    return Handler
