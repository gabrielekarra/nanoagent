# nanoagent

**Any model becomes an agent, and you can read all of it.** Claude, GPT, or a 1.5B model running on your laptop through Ollama: the same few lines give it tools and a loop. The engine is under 500 lines of Python with no dependencies, and a seven-chapter tutorial builds it from an empty file.

```python
from nanoagent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return "sunny, 24°C"

agent = Agent("You are a travel assistant.", tools=[get_weather], model="ollama:qwen3:4b")
print(agent.run("Do I need an umbrella in Rome?"))
```

Change `model=` and nothing else: `"anthropic:claude-opus-5-5"`, `"openai:gpt-5"`, `"lmstudio:qwen3-4b"`.

Where this is going (voice, Telegram and WhatsApp, schedules, app control, always on, each in one line) is in [ROADMAP.md](ROADMAP.md).

## Why another agent SDK

Most agent frameworks assume a large hosted model and are big enough that you cannot tell what happens between your code and it. nanoagent takes the opposite bet:

- **Small models are first-class.** A model with no tool support at all still becomes an agent (see [guided mode](#models-that-cannot-call-tools)), and the context budget adapts to a 4k window.
- **You can read all of it.** `agent.py` is the loop, `tools.py` the tool decorator, `model.py` the one interface a model implements. Each provider is one more file.
- **You can test your agent offline.** A built-in fake model serves scripted replies, so tests run in milliseconds with no model, no key and no flakiness.
- **You can learn how it works.** `tutorial/` rebuilds the engine one idea per chapter.

It still does what a production loop has to do: streaming, approval before risky actions, input validation, prompt caching, context compaction, persistent memory, subagents, and clean recovery from Ctrl-C.

## Install

```bash
pip install git+https://github.com/gabrielekarra/nanoagent
```

That is enough for local models and every OpenAI-compatible server. For Claude, add the extra:

```bash
pip install "nanoagent[anthropic] @ git+https://github.com/gabrielekarra/nanoagent"
```

Requires Python 3.10+.

## Models

| `model=` | What it talks to | Key |
|---|---|---|
| `"ollama:qwen3:4b"` | Ollama, native API (`OLLAMA_HOST`, default `localhost:11434`) | none |
| `"llamacpp:any"` | llama.cpp server on `localhost:8080` | none |
| `"lmstudio:qwen3-4b"` | LM Studio on `localhost:1234` | none |
| `"vllm:Qwen/Qwen3-8B"` | vLLM on `localhost:8000` | none |
| `"openrouter:meta-llama/llama-3.3-70b-instruct"` | OpenRouter | `OPENROUTER_API_KEY` |
| `"openai:gpt-5"` | OpenAI | `OPENAI_API_KEY` |
| `"anthropic:claude-opus-5-5"` or `"claude-opus-5-5"` | Anthropic | `ANTHROPIC_API_KEY` |

With no `model=`, the `NANOAGENT_MODEL` environment variable is used.

For anything else that speaks the Chat Completions format, or to set options:

```python
from nanoagent import Agent
from nanoagent.providers.ollama import Ollama
from nanoagent.providers.openai import OpenAICompatible

Agent(model=OpenAICompatible("http://gpu-box:8000/v1", "my-model", context_window=16_000, temperature=0.2))
Agent(model=Ollama("llama3.2:3b", context=16_384, pull=True))   # downloads the model if missing
```

Ollama gets its own adapter for one reason: on its OpenAI-compatible endpoint it runs models with a small default window and silently drops the start of a long conversation, which is where the instructions and the tools are. The native adapter sets the window on every request.

A model of your own is one method:

```python
from nanoagent import Call, Model, Reply

class MyModel(Model):
    context_window = 8000

    def reply(self, system, tools, messages, on_text=None):
        ...
        return Reply(text="Hello")      # or Reply(calls=[Call(id, name, arguments)], stop="tool")
```

### Models that cannot call tools

Small models often have no tool-calling template, or use it unreliably. With `tools="guided"`, nanoagent stops relying on it: the model answers each step with one JSON object, either a tool call or the final answer.

```json
{"thought": "I need the weather", "tool": "get_weather", "arguments": {"city": "Rome"}}
{"thought": "I have it", "answer": "Sunny, no umbrella needed."}
```

Where the server supports constrained decoding (llama.cpp, Ollama, vLLM, LM Studio), the object is enforced by a JSON schema built from your tools, so the model cannot write anything else. Whatever comes back is still parsed defensively: code fences, prose around the JSON, trailing commas, single quotes, `<think>` blocks and common alternative key names are all accepted, and an unreadable reply gets one correction.

You rarely need to choose. The default, `tools="auto"`, starts with native tool calling and switches to guided the moment the server says the model has no tool support. In native mode, a call the model wrote as JSON text instead of using the tool format is recognized and run.

```python
Agent(tools=[get_weather], model=Ollama("gemma2:2b", tools="guided"))
```

### Small context windows

The defaults follow the model's window. With an 8k model, the history is summarized at 70% of the window instead of at 150,000 tokens, a tool result is clipped to about a quarter of it, and the request is checked before it is sent, so the server never has to truncate it. Large models keep the old limits.

## The SDK in five minutes

### Tools

A tool is a function. Its name, docstring and type hints become what the model sees.

```python
from typing import Annotated
from nanoagent import tool

@tool
def search_orders(
    customer: Annotated[str, "Customer email"],
    limit: int = 10,                      # has a default, so it is optional
) -> str:
    """Find a customer's most recent orders."""
    ...

@tool(approve=True)                       # ask before running this one
def refund(order_id: str) -> str:
    """Refund an order in full."""
    ...
```

Parameters can be `str`, `int`, `float` or `bool`. If a tool raises, the error is sent to the model, which usually corrects itself. Input written by the model is checked against the schema before your function runs.

### Agents

```python
agent = Agent(
    "You handle support for an online shop.",
    tools=[search_orders, refund],
    model="ollama:qwen3:4b",
    approve=lambda name, args: input(f"{name}({args})? [y/n] ") == "y",
    on_text=lambda chunk: print(chunk, end="", flush=True),   # stream the reply
    on_tool=lambda name, args: print(f"\n[{name}]"),           # see each tool call
)

agent.run("Refund the last order of ada@example.com")
agent.run("And email her a confirmation")   # same conversation continues
```

| Option | What it does |
|---|---|
| `model` | `"provider:model"` or a `Model` (see [Models](#models)). Default: `$NANOAGENT_MODEL`. |
| `approve` | Called before each tool marked `approve=True`. Return `False` to decline; the model is told and adapts. Without a callback, marked tools just run. |
| `on_text`, `on_tool` | Callbacks for streamed text and tool calls. |
| `memory="NOTES.md"` | Loads the file into the instructions and gives the model a `remember` tool that appends to it. |
| `subagents=True` | Gives the model a `task` tool that hands work to a fresh agent and gets back only its report. |
| `compact_at` | Context size in tokens above which history is replaced by a model-written handoff note. Default: 70% of the model's window, at most 150,000. |
| `max_output` | Characters kept from one tool result (head and tail). Default: about a quarter of the window, at most 20,000. |

After a run, `agent.messages` holds the conversation and `agent.stop_reason` says why the last reply ended: `"end"`, `"length"` (cut off) or `"refusal"`.

### Testing without a model

Script what the model should say, then assert on what your agent did.

```python
from nanoagent import Agent
from nanoagent.testing import FakeModel, call, reply, text

def test_refund_needs_approval():
    model = FakeModel(
        reply(call("refund", order_id="A17")),       # the model asks for a refund
        reply(text("I could not refund the order.")),
    )
    agent = Agent(tools=[refund], model=model, approve=lambda *_: False)

    agent.run("Refund order A17")

    result = model.tool_results()[0]                 # what the agent sent back
    assert result["error"] and "declined" in result["content"]
```

`model.requests` records the instructions, tools and messages of every request.

### A coding agent in your terminal

The SDK ships with file and shell tools and a small command-line agent built on them (`nanoagent/cli.py`):

```bash
nanoagent --model ollama:qwen3:4b
```

It asks before running commands or changing files, and keeps project notes in `AGENT.md`.

## The tutorial: build it from scratch

Each chapter is one self-contained file that runs on its own and adds one idea to the previous one. Read them in order, or diff two neighbours to see exactly what an idea costs. The chapters use Claude, so they need the `anthropic` extra and a key.

| Chapter | Lines | The idea |
|---|---|---|
| [`01_loop.py`](tutorial/01_loop.py) | 85 | An agent is a model, a tool and a `while` loop. |
| [`02_tools.py`](tutorial/02_tools.py) | 133 | Tools are functions. A failed tool is information for the model, not a crash. |
| [`03_permissions.py`](tutorial/03_permissions.py) | 171 | The model proposes, the harness decides. Ctrl-C without corrupting the conversation. |
| [`04_streaming.py`](tutorial/04_streaming.py) | 210 | Show text as it is written. Never run a tool call that was cut off or malformed. |
| [`05_context.py`](tutorial/05_context.py) | 272 | Clip big outputs, cache the unchanged prefix, compact when the conversation gets large. |
| [`06_memory.py`](tutorial/06_memory.py) | 285 | A notes file that outlives the session. |
| [`07_subagents.py`](tutorial/07_subagents.py) | 298 | A subagent is the same loop, called from inside a tool. |

```bash
python tutorial/01_loop.py
```

```bash
diff tutorial/02_tools.py tutorial/03_permissions.py
```

## Design rules

- **The history is append-only.** Messages are never edited in place; the history is either extended or replaced whole by a summary. This keeps prompt caches valid, and some models reject edited histories.
- **The instructions and the tool set are fixed for an agent's lifetime**, for the same reason.
- **Every tool call gets a result**, even when it was declined, invalid, cut off or interrupted. A call without a result makes the next request fail.
- **Each provider's reply is stored as received** next to the neutral form, and replayed unchanged to the provider that wrote it.

## Safety

The shell tool runs commands on your machine with your permissions. Until the sandbox in the [roadmap](ROADMAP.md) lands, the approval prompt is the only barrier, so read what you approve.

## Development

```bash
git clone https://github.com/gabrielekarra/nanoagent && cd nanoagent
```

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The suite covers the SDK, every provider (against fake servers that speak each wire format) and every tutorial chapter, and runs in a few seconds with no model and no key. To also run it against a real local model:

```bash
NANOAGENT_LOCAL_MODEL=ollama:qwen2.5:1.5b pytest -m local
```

## License

MIT
