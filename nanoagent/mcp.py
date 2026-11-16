"""MCP: any Model Context Protocol server becomes tools, in one line.

    agent.use(mcp("npx -y @modelcontextprotocol/server-filesystem ~/notes"))
    agent.use(mcp("uvx mcp-server-git", trust=["git_status", "git_log"]))
    agent.use(mcp("https://example.com/mcp", headers={"Authorization": "Bearer ..."}))

The server is started when it is added, its tools are registered with the
agent, and it is stopped with the agent (and restarted if it dies). Written
directly on JSON-RPC, so it needs nothing installed.

Tools from a server ask for approval by default, since they act outside
nanoagent's control. `approve=False` turns that off; `trust=[...]` lists the
tools that may run without asking. The server sees only the environment
variables you pass in `env`, plus the few it needs to run (PATH, HOME...).
"""

import itertools
import json
import os
import shlex
import subprocess
import threading
import urllib.error
import urllib.request
from collections import deque

from .tools import Image, Tool

PROTOCOL = "2025-06-18"
BASE_ENV = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "APPDATA", "SHELL")


class MCPError(Exception):
    pass


class MCP:
    def __init__(self, server, *, env=None, approve=True, trust=(), headers=None, name=None, timeout=60, cwd=None):
        self.server = server
        self.env = env or {}
        self.approve = approve
        self.trust = set(trust)
        self.headers = headers or {}
        self.timeout = timeout
        self.cwd = cwd
        self.remote = isinstance(server, str) and server.startswith(("http://", "https://"))
        words = server if isinstance(server, list) else shlex.split(server) if not self.remote else [server]
        self.name = name or _short_name(words)
        self.words = words
        self.process = None
        self.session = None  # Mcp-Session-Id for remote servers
        self.pending = {}  # request id -> [event, response]
        self.errors = deque(maxlen=20)  # the server's last stderr lines, for error messages
        self.ids = itertools.count(1)
        self.lock = threading.Lock()

    # --- plugging in -------------------------------------------------------------------

    def attach(self, agent):
        self.start()
        agent.closers.append(self.close)
        tools = []
        for spec in self.list_tools():
            name = spec["name"] if spec["name"] not in agent.tools else f"{self.name}_{spec['name']}"
            needs = self.approve and spec["name"] not in self.trust
            tools.append(Tool(name, spec.get("description") or "", _schema(spec.get("inputSchema")),
                              self._caller(spec["name"]), approve=bool(needs)))
        agent._add_tools(tools)

    def _caller(self, name):
        def call(**arguments):
            return self.call(name, arguments)

        return call

    # --- the protocol ----------------------------------------------------------------------

    def start(self):
        if not self.remote:
            env = {key: os.environ[key] for key in BASE_ENV if key in os.environ}
            env.update(self.env)
            try:
                self.process = subprocess.Popen(self.words, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, env=env, cwd=self.cwd)
            except FileNotFoundError:
                raise MCPError(f"Cannot start the MCP server: `{self.words[0]}` is not installed or not on PATH.") from None
            threading.Thread(target=self._read, args=(self.process,), daemon=True).start()
            threading.Thread(target=self._read_errors, args=(self.process,), daemon=True).start()
        self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                    "clientInfo": {"name": "nanoagent", "version": "0.2"}})
        self.notify("notifications/initialized")

    def list_tools(self):
        tools, cursor = [], None
        while True:
            page = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += page.get("tools", [])
            cursor = page.get("nextCursor")
            if not cursor:
                return tools

    def call(self, name, arguments):
        try:
            result = self.request("tools/call", {"name": name, "arguments": arguments})
        except (BrokenPipeError, ConnectionError, OSError) as error:
            if self.remote:
                raise MCPError(f"{self.name}: {error}") from None
            self.restart()  # the server died: start it again and retry once
            result = self.request("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "image":
                import base64

                parts.append(Image(base64.b64decode(item["data"]), item.get("mimeType", "image/png")))
            elif item.get("type") == "resource":
                parts.append(item.get("resource", {}).get("text") or json.dumps(item.get("resource")))
        if not parts and "structuredContent" in result:
            parts.append(json.dumps(result["structuredContent"]))
        if result.get("isError"):
            raise MCPError("\n".join(p for p in parts if isinstance(p, str)) or "the tool failed")
        return parts if any(isinstance(p, Image) for p in parts) else "\n".join(parts)

    def restart(self):
        self.close()
        self.start()

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    # --- JSON-RPC ------------------------------------------------------------------------

    def request(self, method, params):
        message = {"jsonrpc": "2.0", "id": next(self.ids), "method": method, "params": params}
        response = self._remote(message) if self.remote else self._local(message)
        if "error" in response:
            raise MCPError(f"{self.name} {method}: {response['error'].get('message', response['error'])}")
        return response.get("result") or {}

    def notify(self, method):
        message = {"jsonrpc": "2.0", "method": method}
        if self.remote:
            self._remote(message, expect=False)
        else:
            self._write(message)

    def _local(self, message):
        if not self.process or self.process.poll() is not None:
            raise ConnectionError(f"the MCP server {self.name} is not running{self._stderr()}")
        waiting = self.pending[message["id"]] = [threading.Event(), None]
        self._write(message)
        if not waiting[0].wait(self.timeout):
            self.pending.pop(message["id"], None)
            raise MCPError(f"{self.name} did not answer {message['method']} within {self.timeout}s{self._stderr()}")
        self.pending.pop(message["id"], None)
        if waiting[1] is None:
            raise ConnectionError(f"the MCP server {self.name} stopped{self._stderr()}")
        return waiting[1]

    def _write(self, message):
        with self.lock:
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            self.process.stdin.flush()

    def _read(self, process):
        for line in process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue  # servers sometimes print other things
            if "id" in message and ("result" in message or "error" in message):
                waiting = self.pending.get(message["id"])
                if waiting:
                    waiting[1] = message
                    waiting[0].set()
            elif "id" in message and "method" in message:  # a request from the server
                answer = {"result": {}} if message["method"] == "ping" else {
                    "error": {"code": -32601, "message": f"nanoagent does not support {message['method']}"}}
                self._write({"jsonrpc": "2.0", "id": message["id"], **answer})
        for waiting in list(self.pending.values()):  # the server exited: wake everyone up
            waiting[0].set()

    def _read_errors(self, process):
        for line in process.stderr:
            self.errors.append(line.decode("utf-8", "replace").rstrip())

    def _stderr(self):
        return (": " + " | ".join(self.errors)) if self.errors else ""

    def _remote(self, message, expect=True):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL, **self.headers}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        request = urllib.request.Request(self.server, json.dumps(message).encode(), headers, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise MCPError(f"{self.name}: HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:300]}") from None
        except (urllib.error.URLError, OSError) as error:
            raise ConnectionError(f"cannot reach {self.server}: {getattr(error, 'reason', error)}") from None
        with response:
            self.session = response.headers.get("Mcp-Session-Id") or self.session
            if not expect:
                return None
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                for line in response:
                    line = line.decode().strip()
                    if line.startswith("data:"):
                        data = json.loads(line[5:])
                        if data.get("id") == message["id"]:
                            return data
                raise MCPError(f"{self.name}: the stream ended without an answer")
            return json.load(response)


def _schema(schema):
    schema = dict(schema or {"type": "object", "properties": {}})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


RUNNERS = {"npx", "uvx", "node", "python", "python3", "docker", "run", "deno", "bunx", "pipx"}


def _short_name(words):
    """A short name for the server, used to prefix its tools when their names clash."""
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word in ("-e", "--env", "-v", "--volume", "--name"):
            skip = True  # the next word is this flag's value
            continue
        program = word.rstrip("/").split("/")[-1]
        if word.startswith("-") or program in RUNNERS or program.startswith("python") or "://" in word:
            continue
        base = program.split("@")[0].split(":")[0]
        for extension in (".py", ".js", ".mjs", ".ts"):
            base = base.removesuffix(extension)
        for prefix in ("mcp-server-", "server-", "mcp-"):
            base = base.removeprefix(prefix)
        for suffix in ("-mcp-server", "-mcp"):
            base = base.removesuffix(suffix)
        return base.replace("-", "_") or "mcp"
    return "mcp"


def mcp(server, **options):
    """Tools from an MCP server: a command line (stdio) or an https:// URL. See the module docstring."""
    return MCP(server, **options)

