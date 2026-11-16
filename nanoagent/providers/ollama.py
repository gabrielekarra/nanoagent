"""Ollama, through its own API.

Ollama also speaks the OpenAI format, but there it runs every model with a
small default context window and silently drops whatever does not fit: an
agent loses its instructions and its tools without any error. The native
API lets nanoagent set the window on every request.
"""

import json

from ..model import Call, ModelError, Reply
from . import guided
from .openai import OpenAICompatible, ThinkFilter, estimate, rejects_tools


class Ollama(OpenAICompatible):
    """A model served by Ollama.

        Ollama("qwen3:4b")
        Ollama("llama3.2:3b", context=16_384, pull=True)

    context   the context window to run the model with, in tokens
    pull      download the model if it is not there yet
    tools     "auto", "native" or "guided", as for `OpenAICompatible`
    options   extra model options, e.g. temperature=0.2
    """

    key = "ollama"

    def __init__(self, model, *, host="http://localhost:11434", context=8192, pull=False, tools="auto",
                 timeout=600, **options):
        super().__init__(host, model, tools=tools, context_window=context, timeout=timeout)
        self.name = f"ollama:{model}"
        self.pull = pull
        self.model_options = {"num_ctx": context, **options}
        self.formats = ["schema", "json", None]

    def reply(self, system, tools, messages, on_text=None):
        try:
            return super().reply(system, tools, messages, on_text)
        except ModelError as error:
            if error.status != 404 or "not found" not in error.detail:
                raise
            if not self.pull:
                raise ModelError(
                    f"Ollama has no model `{self.model}`. Run: ollama pull {self.model}  "
                    "(or pass pull=True)", 404, error.detail) from None
            self._download()
            return super().reply(system, tools, messages, on_text)

    def _download(self):
        print(f"Downloading {self.model} with Ollama...", flush=True)
        with self._post("/api/pull", {"model": self.model, "stream": False}) as response:
            status = json.load(response).get("status")
        if status != "success":
            raise ModelError(f"Ollama could not pull `{self.model}`: {status}")
        self.pull = False

    # --- native tool calling ----------------------------------------------------

    def _native(self, system, tools, messages, on_text):
        body = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "stream": True,
            "options": self.model_options,
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
                for t in tools
            ]
        thinking, text, calls, last = ThinkFilter(), [], [], {}

        def show(piece):
            if not text:
                piece = piece.lstrip()
            if piece:
                text.append(piece)
                if on_text:
                    on_text(piece)

        for chunk in self._lines(body):
            message = chunk.get("message") or {}
            if message.get("content"):  # message.thinking is never shown
                show(thinking.feed(message["content"]))
            for part in message.get("tool_calls") or []:
                function = part.get("function") or {}
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except ValueError:
                        pass
                calls.append(Call(part.get("id") or f"call_{next(self._ids)}", function.get("name", ""), arguments))
            last = chunk
        show(thinking.flush())

        answer = "".join(text).strip()
        if not calls and tools and answer and answer[0] in "{`":
            found = guided.parse(answer, {t.name for t in tools})
            if found and "tool" in found:
                calls, answer = [Call(f"call_{next(self._ids)}", found["tool"], found["arguments"])], ""
        stop = "length" if last.get("done_reason") == "length" else ("tool" if calls else "end")
        return Reply(answer, calls, stop, self._tokens(last, body, answer))

    def _messages(self, system, messages):
        wire = [{"role": "system", "content": system}] if system else []
        for turn in messages:
            if turn["role"] == "user":
                wire.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "assistant":
                calls = turn.get("calls", [])
                if not turn["content"] and not calls:
                    continue
                message = {"role": "assistant", "content": turn["content"]}
                if calls:
                    message["tool_calls"] = [
                        {"function": {"name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], dict) else {}}}
                        for c in calls
                    ]
                wire.append(message)
            elif turn["role"] == "tool":
                wire += [{"role": "tool", "tool_name": r["name"], "content": r["content"]} for r in turn["results"]]
        return guided.merge(wire)

    # --- guided mode transport ------------------------------------------------

    def _complete(self, chat, schema):
        while True:
            body = {"model": self.model, "messages": chat, "stream": False, "options": self.model_options}
            shape = self.formats[0]
            if shape == "schema":
                body["format"] = schema
            elif shape == "json":
                body["format"] = "json"
            try:
                with self._post("/api/chat", body) as response:
                    chunk = json.load(response)
            except ModelError as error:
                if shape and error.status in (400, 500) and not rejects_tools(error) and error.status != 404:
                    self.formats.pop(0)
                    continue
                raise
            text = (chunk.get("message") or {}).get("content") or ""
            return text, self._tokens(chunk, body, text)

    # --- HTTP -------------------------------------------------------------------

    def _lines(self, body):
        response = self._post("/api/chat", body)
        return self._read_lines(response)

    def _read_lines(self, response):
        with response:
            for line in response:
                if line.strip():
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise ModelError(f"{self.name}: {chunk['error']}", detail=str(chunk["error"]))
                    yield chunk

    def _post(self, path, body):
        try:
            return super()._post(path, body)
        except ModelError as error:
            if error.status is None:
                raise ModelError(f"Cannot reach Ollama at {self.base_url}. Start it with: ollama serve") from None
            raise

    def _tokens(self, chunk, body, text):
        if chunk.get("prompt_eval_count") is not None:
            return chunk["prompt_eval_count"] + (chunk.get("eval_count") or 0)
        return estimate(json.dumps(body.get("messages", [])) + text)
