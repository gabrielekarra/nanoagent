"""Any server that speaks the OpenAI Chat Completions format.

That covers most of them: Ollama, llama.cpp, LM Studio, vLLM, OpenRouter,
OpenAI itself. Written on the standard library, so it needs nothing installed.
"""

import itertools
import json
import urllib.error
import urllib.request

from ..model import Call, Model, ModelError, Reply
from . import guided

# Phrases servers use to say a model has no tool support.
NO_TOOLS = (
    "does not support tools", "tools param requires", "--jinja", "tool use is not supported",
    "tools are not supported", "function calling is not supported", "does not support function",
)


def rejects_tools(error):
    return error.status in (400, 404, 422, 500) and any(p in error.detail.lower() for p in NO_TOOLS)


class ThinkFilter:
    """Remove <think>...</think> from streamed text, even when a tag is split across chunks."""

    def __init__(self):
        self.buffer, self.inside = "", False

    def feed(self, chunk):
        self.buffer += chunk
        shown = []
        while True:
            tag = "</think>" if self.inside else "<think>"
            at = self.buffer.find(tag)
            if at >= 0:
                if not self.inside:
                    shown.append(self.buffer[:at])
                self.buffer, self.inside = self.buffer[at + len(tag):], not self.inside
                continue
            # Hold back an ending that could be the start of a tag.
            keep = next((k for k in range(len(tag) - 1, 0, -1) if self.buffer.endswith(tag[:k])), 0)
            if not self.inside:
                shown.append(self.buffer[: len(self.buffer) - keep])
            self.buffer = self.buffer[len(self.buffer) - keep:]
            return "".join(shown)

    def flush(self):
        rest = "" if self.inside else self.buffer
        self.buffer = ""
        return rest


class OpenAICompatible(Model):
    """A model behind a Chat Completions endpoint.

        OpenAICompatible("http://localhost:8080/v1", "qwen3-4b")

    tools           "native" uses the server's tool calling, "guided" asks for
                    JSON steps instead (see `guided.py`), "auto" starts native
                    and switches to guided if the server says the model has no
                    tool support
    context_window  the model's context size in tokens
    max_tokens      output limit for one reply; the server's default if None
    options         extra fields for every request, e.g. temperature=0.2
    """

    key = "openai"

    def __init__(self, base_url, model, *, api_key=None, tools="auto", context_window=32_000,
                 max_tokens=None, timeout=600, **options):
        if tools not in ("auto", "native", "guided"):
            raise ValueError("tools must be 'auto', 'native' or 'guided'")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"{self.base_url} {model}"
        self.api_key = api_key
        self.tools = tools
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.options = options
        self.stream_usage = True  # off if the server rejects stream_options
        self.formats = ["json_schema", "json_object", None]  # what guided mode can still try
        self._ids = itertools.count(1)

    # --- the Model interface ----------------------------------------------------

    def reply(self, system, tools, messages, on_text=None):
        if tools and self.tools == "guided":
            return guided.step(self._complete, system, tools, messages, on_text, self._ids)
        try:
            return self._native(system, tools, messages, on_text)
        except ModelError as error:
            if tools and self.tools == "auto" and rejects_tools(error):
                self.tools = "guided"  # remembered for the rest of the session
                return guided.step(self._complete, system, tools, messages, on_text, self._ids)
            raise

    # --- native tool calling ----------------------------------------------------

    def _native(self, system, tools, messages, on_text):
        body = {"model": self.model, "messages": self._messages(system, messages), "stream": True, **self.options}
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
                for t in tools
            ]
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens

        thinking, text, slots, finish, usage = ThinkFilter(), [], {}, None, None

        def show(piece):
            if piece:
                if not text:
                    piece = piece.lstrip()  # models often start with blank lines after thinking
                if piece:
                    text.append(piece)
                    if on_text:
                        on_text(piece)

        for chunk in self._chunks(body):
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or choice.get("message") or {}
                if isinstance(delta.get("content"), str):  # reasoning_content is never shown
                    show(thinking.feed(delta["content"]))
                for position, part in enumerate(delta.get("tool_calls") or []):
                    index = part.get("index")
                    slot_key = index if index is not None else (part.get("id") or f"position-{len(slots) + position}")
                    slot = slots.setdefault(slot_key, {"id": None, "name": "", "arguments": ""})
                    function = part.get("function") or {}
                    slot["id"] = slot["id"] or part.get("id")
                    slot["name"] = slot["name"] or function.get("name") or ""
                    pieces = function.get("arguments")
                    if isinstance(pieces, dict):
                        slot["arguments"] = pieces
                    elif isinstance(pieces, str) and isinstance(slot["arguments"], str):
                        slot["arguments"] += pieces
                finish = choice.get("finish_reason") or finish
        show(thinking.flush())

        calls = [Call(s["id"] or f"call_{next(self._ids)}", s["name"], _arguments(s["arguments"])) for s in slots.values()]
        answer = "".join(text).strip()
        if not calls and tools and answer and answer[0] in "{`":
            # Small models often write the call as JSON text instead of using
            # the tool-calling format. If it names a real tool, run it.
            found = guided.parse(answer, {t.name for t in tools})
            if found and "tool" in found:
                calls, answer = [Call(f"call_{next(self._ids)}", found["tool"], found["arguments"])], ""
        if finish == "length":
            stop = "length"
        elif finish == "content_filter":
            stop = "refusal"
        else:  # several servers say "stop" even when the reply holds tool calls
            stop = "tool" if calls else "end"
        return Reply(answer, calls, stop, self._context(usage, body, answer))

    def _messages(self, system, messages):
        wire = [{"role": "system", "content": system}] if system else []
        for turn in messages:
            if turn["role"] == "user":
                wire.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "assistant":
                calls = turn.get("calls", [])
                if not turn["content"] and not calls:
                    continue
                message = {"role": "assistant", "content": turn["content"] or (None if calls else "")}
                if calls:
                    message["tool_calls"] = [
                        {"id": c["id"], "type": "function", "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"]) if isinstance(c["arguments"], dict) else c["arguments"],
                        }}
                        for c in calls
                    ]
                wire.append(message)
            elif turn["role"] == "tool":
                wire += [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in turn["results"]]
        return guided.merge(wire)

    # --- guided mode transport ------------------------------------------------

    def _complete(self, chat, schema):
        """One non-streamed completion constrained to `schema` as far as the server allows."""
        while True:
            body = {"model": self.model, "messages": chat, "stream": False, **self.options}
            if self.max_tokens:
                body["max_tokens"] = self.max_tokens
            shape = self.formats[0]
            if shape == "json_schema":
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "step", "schema": schema}}
            elif shape == "json_object":
                body["response_format"] = {"type": "json_object"}
            try:
                (chunk,) = self._chunks(body)
            except ModelError as error:
                if shape and error.status in (400, 422, 500):
                    self.formats.pop(0)  # the server rejected this format: try a looser one
                    continue
                raise
            message = (chunk.get("choices") or [{}])[0].get("message") or {}
            text = message.get("content") or ""
            return text, self._context(chunk.get("usage"), body, text)

    # --- HTTP -------------------------------------------------------------------

    def _chunks(self, body):
        """Yield the response's chunks: one per event if streamed, else the whole response."""
        if body.get("stream") and self.stream_usage:
            body = {**body, "stream_options": {"include_usage": True}}
        try:
            response = self._post("/chat/completions", body)
        except ModelError as error:
            mentioned = "stream_options" in error.detail or "stream options" in error.detail.lower()
            if "stream_options" not in body or error.status not in (400, 422) or not mentioned:
                raise
            self.stream_usage = False  # an older server: retry without the usage request
            body = {key: value for key, value in body.items() if key != "stream_options"}
            response = self._post("/chat/completions", body)
        return self._read(response)

    def _read(self, response):
        with response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                yield json.load(response)
                return
            for line in response:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                event = json.loads(data)
                if "error" in event:
                    raise ModelError(f"{self.name}: {event['error']}", detail=str(event["error"]))
                yield event

    def _post(self, path, body):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + path, json.dumps(body).encode(), headers, method="POST")
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            detail = _error_message(error.read().decode("utf-8", "replace"))
            raise ModelError(f"{self.name}: HTTP {error.code}: {detail}", error.code, detail) from None
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise ModelError(f"Cannot reach {self.base_url} ({reason}). Is the server running?") from None

    def _context(self, usage, body, text):
        if usage and usage.get("prompt_tokens") is not None:
            return usage["prompt_tokens"] + (usage.get("completion_tokens") or 0)
        return estimate(json.dumps(body.get("messages", [])) + text)


def estimate(text):
    """A rough token count for servers that report none: about four characters a token."""
    return len(text) // 4


def _arguments(value):
    if isinstance(value, dict):
        return value
    if not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except ValueError:
        return value
    return loaded if isinstance(loaded, dict) else value


def _error_message(body):
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:500]
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500]
    return str(error)[:500]
