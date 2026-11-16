"""Models: the one interface the agent loop talks to.

A model takes the instructions, the tools and the conversation, and returns
a `Reply`: some text, some tool calls, or both. Anything that can do that is
a model, whether it runs in a data centre or on your laptop.

The conversation is a list of plain dicts that nanoagent owns:

    {"role": "user", "content": "What's the weather in Rome?"}
    {"role": "assistant", "content": "", "calls": [{"id": ..., "name": ..., "arguments": {...}}],
     "raw": ..., "by": "anthropic"}
    {"role": "tool", "results": [{"id": ..., "name": ..., "content": "sunny", "error": False}]}

Each provider converts it to its own wire format. `raw` is the provider's own
copy of a reply, replayed unchanged by the provider that wrote it (`by`) and
ignored by every other one.
"""

import json
import os
from dataclasses import dataclass, field


@dataclass
class Call:
    id: str
    name: str
    arguments: dict | str  # a str is model output that was not a JSON object


@dataclass
class Reply:
    text: str = ""
    calls: list = field(default_factory=list)
    stop: str = "end"  # "end", "tool", "length" (cut off) or "refusal"
    context: int = 0  # tokens the conversation occupies after this reply
    raw: object = None  # provider data to replay unchanged; must be JSON-serializable


class ModelError(Exception):
    """A model could not be reached or refused the request."""

    def __init__(self, message, status=None, detail=""):
        super().__init__(message)
        self.status = status
        self.detail = detail


class Model:
    """Subclass this and implement `reply` to make any model an agent."""

    key = "custom"  # identifies the provider in `raw`
    name = "custom model"
    context_window = 32_000

    def reply(self, system, tools, messages, on_text=None):
        """Return the model's next `Reply`. Call `on_text` with text as it is written."""
        raise NotImplementedError


def estimate(*parts):
    """A rough token count: about four characters a token, over the JSON form of `parts`."""
    return sum(len(part if isinstance(part, str) else json.dumps(part, default=str)) for part in parts) // 4


def _ollama_url():
    host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


# prefix -> (base URL, environment variable holding the API key)
OPENAI_COMPATIBLE = {
    "openai": (lambda: "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": (lambda: "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "lmstudio": (lambda: "http://localhost:1234/v1", None),
    "llamacpp": (lambda: "http://localhost:8080/v1", None),
    "vllm": (lambda: "http://localhost:8000/v1", None),
}
PREFIXES = ["anthropic", "ollama", *OPENAI_COMPATIBLE]


def resolve(spec):
    """Turn `"provider:model"` into a model. A `Model` is returned as it is.

        resolve("ollama:qwen3:4b")
        resolve("anthropic:claude-opus-5-5")   # or just "claude-opus-5-5"
        resolve("openrouter:meta-llama/llama-3.3-70b-instruct")
    """
    if isinstance(spec, Model):
        return spec
    if not spec:
        raise ValueError(
            "No model given. Pass one, e.g. Agent(model='ollama:qwen3:4b') or "
            "Agent(model='anthropic:claude-opus-5-5'), or set NANOAGENT_MODEL."
        )
    if not isinstance(spec, str):
        raise TypeError(f"model must be a string or a Model, not {type(spec).__name__}")
    prefix, _, name = spec.partition(":")
    if spec.startswith("claude-"):
        prefix, name = "anthropic", spec
    if prefix == "anthropic":
        from .providers.anthropic import Anthropic

        return Anthropic(name)
    if prefix == "ollama":
        from .providers.ollama import Ollama

        return Ollama(name, host=_ollama_url())
    if prefix in OPENAI_COMPATIBLE:
        from .providers.openai import OpenAICompatible

        url, key = OPENAI_COMPATIBLE[prefix]
        return OpenAICompatible(url(), name, api_key=os.environ.get(key) if key else None)
    raise ValueError(f"Unknown model provider `{prefix}` in {spec!r}. Known: {', '.join(PREFIXES)}.")
