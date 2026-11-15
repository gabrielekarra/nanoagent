"""Issue #21: the MCP client."""

import sys
import time
from pathlib import Path

import pytest

from nanoagent import Agent, read_file
from nanoagent.mcp import MCPError, _short_name, mcp
from nanoagent.testing import FakeModel, call, reply, text

SERVER = [sys.executable, str(Path(__file__).parent / "mcp_server.py")]


@pytest.fixture
def agent():
    made = []

    def make(script=(), **options):
        model = FakeModel(*script)
        new = Agent(tools=[read_file], model=model)
        new.use(mcp(SERVER, approve=False, **options))
        made.append(new)
        return new, model

    yield make
    for each in made:
        each.close()


def test_server_tools_are_registered(agent):
    new, _ = agent()
    tools = new.tools
    assert {"echo", "tag", "fail", "crash", "env", "picture"} <= set(tools)
    assert tools["echo"].description == "Echo text back."
    assert tools["echo"].schema["required"] == ["text"]


def test_call_round_trip(agent):
    new, model = agent([reply(call("echo", text="hello")), reply(text("done"))])
    new.run("echo it")
    assert model.tool_results()[0]["content"] == "hello"


def test_server_error_is_an_error_result(agent):
    new, model = agent([reply(call("fail")), reply(text("ok"))])
    new.run("go")
    result = model.tool_results()[0]
    assert result["error"] and "disk is full" in result["content"]


def test_nested_schemas_are_accepted(agent):
    new, model = agent([reply(call("tag", items=["a", "b"], meta={"color": "red"})),
                        reply(call("tag", items=["a", 3], meta={})), reply(text("ok"))])
    new.run("tag")
    assert model.tool_results(1)[0]["content"] == "red: a,b"
    problem = model.tool_results(2)[0]["content"]
    assert "`items[1]` must be a string" in problem and "missing `meta.color`" in problem


def test_crashed_server_is_restarted(agent):
    new, model = agent([reply(call("crash")), reply(call("echo", text="back")), reply(text("ok"))])
    new.run("crash then echo")
    assert model.tool_results(1)[0]["error"]
    assert model.tool_results(2)[0]["content"] == "back"


def test_server_stops_with_the_agent(agent):
    new, _ = agent()
    (server,) = [closer.__self__ for closer in new.closers]
    process = server.process
    new.close()
    assert process.poll() is not None


def test_server_does_not_inherit_secrets(agent, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    new, model = agent([reply(call("env", name="SECRET_TOKEN"), call("env", name="GIVEN")), reply(text("ok"))],
                       env={"GIVEN": "yes"})
    new.run("look")
    assert [r["content"] for r in model.tool_results()] == ["<unset>", "yes"]


def test_approval_default():
    model = FakeModel()
    new = Agent(model=model)
    new.use(mcp(SERVER, trust=["echo"]))
    try:
        assert new.tools["fail"].approve and not new.tools["echo"].approve
    finally:
        new.close()


def test_images_from_a_server(agent):
    new, model = agent([reply(call("picture")), reply(text("ok"))])
    new.run("draw")
    content = model.tool_results()[0]["content"]
    assert content[0] == "a chart" and content[1]["type"] == "image"


def test_clashing_names_are_prefixed(agent):
    new, _ = agent()
    assert new.tools["read_file"] is not None and "mcp_server_read_file" in new.tools


def test_missing_command_is_explained():
    with pytest.raises(MCPError, match="not installed or not on PATH"):
        Agent(model=FakeModel()).use(mcp("definitely-not-a-command --flag"))


@pytest.mark.parametrize("command, name", [
    ("npx -y @modelcontextprotocol/server-filesystem ~/notes", "filesystem"),
    ("uvx mcp-server-git", "git"),
    ("docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server", "github"),
])
def test_short_names(command, name):
    import shlex

    assert _short_name(shlex.split(command)) == name


def test_remote_server_over_http():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((message.get("method"), self.headers.get("Mcp-Session-Id"), self.headers.get("Authorization")))
            if "id" not in message:
                self.send_response(202)
                self.end_headers()
                return
            if message["method"] == "initialize":
                result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            elif message["method"] == "tools/list":
                result = {"tools": [{"name": "hello", "inputSchema": {"type": "object", "properties": {}}}]}
            else:
                result = {"content": [{"type": "text", "text": "hi from far away"}]}
            body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result})}\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Mcp-Session-Id", "s-1")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = FakeModel(reply(call("hello")), reply(text("ok")))
        new = Agent(model=model)
        new.use(mcp(f"http://127.0.0.1:{server.server_port}/mcp", approve=False, headers={"Authorization": "Bearer t"}))
        new.run("say hello")
    finally:
        server.shutdown()
    assert model.tool_results()[0]["content"] == "hi from far away"
    assert seen[0] == ("initialize", None, "Bearer t")
    assert all(session == "s-1" for _, session, _ in seen[1:])
