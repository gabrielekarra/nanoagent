"""A tiny MCP server on stdio, for the tests. Behaviour is chosen by its tools."""

import base64
import json
import os
import sys

TOOLS = [
    {"name": "echo", "description": "Echo text back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "tag", "description": "Tag items.", "inputSchema": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "string"}},
        "meta": {"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]}},
        "required": ["items", "meta"]}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "crash", "description": "Exit the server.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "env", "description": "Read a variable.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "picture", "description": "Return an image.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "read_file", "description": "Clashes with a built-in name.", "inputSchema": {"type": "object", "properties": {}}},
]


def answer(message_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\n")
    sys.stdout.flush()


print("fake server starting", file=sys.stderr, flush=True)
for line in sys.stdin:
    message = json.loads(line)
    method, params = message.get("method"), message.get("params") or {}
    if "id" not in message:
        continue
    if method == "initialize":
        answer(message["id"], {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                               "serverInfo": {"name": "fake", "version": "1"}})
    elif method == "tools/list":  # two pages
        if params.get("cursor"):
            answer(message["id"], {"tools": TOOLS[4:]})
        else:
            answer(message["id"], {"tools": TOOLS[:4], "nextCursor": "page2"})
    elif method == "tools/call":
        name, args = params["name"], params.get("arguments", {})
        if name == "echo":
            answer(message["id"], {"content": [{"type": "text", "text": args["text"]}]})
        elif name == "tag":
            answer(message["id"], {"content": [{"type": "text", "text": f"{args['meta']['color']}: {','.join(args['items'])}"}]})
        elif name == "fail":
            answer(message["id"], {"content": [{"type": "text", "text": "disk is full"}], "isError": True})
        elif name == "crash":
            sys.exit(1)
        elif name == "env":
            answer(message["id"], {"content": [{"type": "text", "text": os.environ.get(args["name"], "<unset>")}]})
        elif name == "picture":
            png = base64.b64encode(b"\x89PNG fake").decode()
            answer(message["id"], {"content": [{"type": "text", "text": "a chart"}, {"type": "image", "data": png, "mimeType": "image/png"}]})
        else:
            answer(message["id"], {"content": [{"type": "text", "text": "ok"}]})
    else:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "no"}}) + "\n")
        sys.stdout.flush()
