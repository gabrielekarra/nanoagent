# nanoagent

[![tests](https://github.com/gabrielekarra/nanoagent/actions/workflows/test.yml/badge.svg)](https://github.com/gabrielekarra/nanoagent/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**An agent SDK small enough to read.** The engine is under 400 lines of Python in two files, and a seven-chapter tutorial builds it from an empty file so you know why every line is there.

```python
from nanoagent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return "sunny, 24°C"

agent = Agent("You are a travel assistant.", tools=[get_weather])
print(agent.run("Do I need an umbrella in Rome?"))
```

That is the whole API: a function becomes a tool, tools go into an agent, and `run` works until the job is done.

## Try it in 30 seconds

```bash
pip install git+https://github.com/gabrielekarra/nanoagent
export ANTHROPIC_API_KEY=...
nanoagent
```

<!-- demo: record docs/demo.tape, commit docs/demo.gif, then replace this comment with ![demo](docs/demo.gif) -->

## Why another agent SDK

Most agent frameworks are large enough that you cannot tell what happens between your code and the model. When the agent misbehaves, you are debugging someone else's abstractions.

nanoagent takes the opposite bet:

- **You can read all of it.** `agent.py` is the loop, `tools.py` is the tool decorator. There is no plugin system, no graph, no chain.
- **You can test your agent offline.** A built-in fake model serves scripted replies over real HTTP, so your tests run in milliseconds with no API key and no flakiness.
- **You can learn how it works.** `tutorial/` rebuilds the engine one idea per chapter, each a single runnable file.

It still does what a production loop has to do: streaming, approval before risky actions, input validation, prompt caching, context compaction, persistent memory, subagents, and clean recovery from Ctrl-C.

## Install

```bash
pip install git+https://github.com/gabrielekarra/nanoagent
```

```bash
export ANTHROPIC_API_KEY=...
```

Requires Python 3.10+. The only dependency is the official `anthropic` SDK.

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
    approve=lambda name, args: input(f"{name}({args})? [y/n] ") == "y",
    on_text=lambda chunk: print(chunk, end="", flush=True),   # stream the reply
    on_tool=lambda name, args: print(f"\n[{name}]"),           # see each tool call
)

agent.run("Refund the last order of ada@example.com")
agent.run("And email her a confirmation")   # same conversation continues
```

| Option | What it does |
|---|---|
| `model` | Any Claude model id. Default `claude-opus-5-5`. |
| `approve` | Called before each tool marked `approve=True`. Return `False` to decline; the model is told and adapts. Without a callback, marked tools just run. |
| `on_text`, `on_tool` | Callbacks for streamed text and tool calls. |
| `memory="NOTES.md"` | Loads the file into the instructions and gives the model a `remember` tool that appends to it. |
| `subagents=True` | Gives the model a `task` tool that hands work to a fresh agent and gets back only its report. |
| `compact_at` | Context size in tokens above which history is replaced by a model-written handoff note. Default 150,000. |
| `max_output` | Characters kept from one tool result (head and tail). Default 20,000. |
| `client` | Your own `anthropic.Anthropic` client. |

After a run, `agent.messages` holds the conversation and `agent.stop_reason` says why the last reply ended (`"end_turn"`, `"refusal"`, ...).

### Testing without a model

Script what the model should say, then assert on what your agent did.

```python
from nanoagent import Agent
from nanoagent.testing import FakeModel, call, reply, text

def test_refund_needs_approval():
    with FakeModel(
        reply(call("refund", order_id="A17")),      # the model asks for a refund
        reply(text("I could not refund the order.")),
    ) as model:
        agent = Agent(tools=[refund], client=model.client, approve=lambda *_: False)

        agent.run("Refund order A17")

        result = model.tool_results()[0]            # what the agent sent back
        assert result["is_error"] and "declined" in result["content"]
```

`FakeModel` is a real HTTP server on localhost speaking the Messages API, streaming included. Your agent goes through the real client, so serialization and parsing are exercised too. `model.requests` records every request body.

### A coding agent in your terminal

The SDK ships with file and shell tools and a small command-line agent built on them (59 lines, in `nanoagent/cli.py`):

```bash
nanoagent
```

It asks before running commands or changing files, and keeps project notes in `AGENT.md`.

## The tutorial: build it from scratch

Each chapter is one self-contained file that runs on its own and adds one idea to the previous one. Read them in order, or diff two neighbours to see exactly what an idea costs.

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

Chapter 7 is a complete coding agent. `nanoagent/agent.py` is the same logic arranged as a class.

## Design rules

- **The history is append-only.** Messages are never edited in place; the history is either extended or replaced whole by a summary. This keeps the prompt cache valid, and newer models reject edited histories.
- **The instructions and the tool set are fixed for an agent's lifetime**, for the same reason.
- **Every tool call gets a result**, even when it was declined, invalid, cut off or interrupted. A call without a result makes the next request fail.
- **The model's reply is stored exactly as received**, including the blocks you do not display.

## Who this is for

Pick nanoagent if you want to understand how an agent works, build a small one you can fully audit, or test agent behaviour in CI without a network. Pick a larger framework if you need many model providers, async, or ready-made integrations out of the box.

## What is not here

On purpose, so the code stays readable: other model providers, parallel tool execution, MCP, async, saving a session to disk, and sandboxing of the shell tool. Each is a good exercise, and most fit in a few dozen lines on top of what is here.

The shell tool runs commands on your machine with your permissions. The approval prompt is the only barrier, so read what you approve.

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

The suite covers the SDK and every tutorial chapter, and runs in a few seconds without an API key.

## License

MIT
