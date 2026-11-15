# nanoagent

[![tests](https://github.com/gabrielekarra/nanoagent/actions/workflows/test.yml/badge.svg)](https://github.com/gabrielekarra/nanoagent/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**Any model becomes an agent, and every ability is one line.**

```python
from nanoagent import Agent, app, every, sandbox, telegram, voice

agent = Agent("You are Jarvis.", model="ollama:qwen3:4b")      # a model that fits in 16 GB of RAM
agent.use(voice())                                             # it listens and speaks
agent.use(telegram(), notify=True)                             # it is on your phone, and writes first
agent.use(every("1h", "Check my inbox, tell me what matters"))  # it acts on its own
agent.use(app("github"))                                       # it controls an app
agent.use(sandbox(root="~/jarvis"))                            # its files stay in one folder
agent.live()                                                   # it is always on
```

The same lines work with Claude, GPT, or a 1.5B model that cannot call tools at all. The core has no dependencies, every module is small enough to read in one sitting, and every feature is tested without a model or a network.

## Try it in 30 seconds

With [Ollama](https://ollama.com) running:

```bash
pip install git+https://github.com/gabrielekarra/nanoagent
```

```bash
nanoagent --model ollama:qwen3:4b
```

That is a coding agent in your terminal. It asks before running commands or changing files, and keeps notes on your project in `AGENT.md`. With Claude instead: install the `anthropic` extra, set `ANTHROPIC_API_KEY`, and use `--model anthropic:claude-opus-5-5`.

<!-- demo: record docs/demo.tape, commit docs/demo.gif, then replace this comment with ![demo](docs/demo.gif) -->

## Why nanoagent

Most agent frameworks assume a large hosted model and compete on how much they can orchestrate. nanoagent bets on two things they mostly ignore:

- **Small models are first-class.** A model with no tool support still becomes an agent through [guided tool calling](#models-that-cannot-call-tools), the context budget adapts to a 4k window, and a [benchmark](#which-models-work) tells you which local models hold up.
- **An idea is one line away from running.** "I want it to talk", "I want it on WhatsApp", "I want it to check the build every hour": each is an `agent.use(...)`.

And it stays honest about what it is: plain Python you can read, with no dependencies in the core, tested offline with fakes of every model and service it talks to.

## Install

```bash
pip install git+https://github.com/gabrielekarra/nanoagent
```

That covers local models, every OpenAI-compatible server, and every channel. Optional extras, only for what needs them:

| Extra | For |
|---|---|
| `anthropic` | Claude models |
| `voice` | `voice()` defaults: microphone, local Whisper |
| `browser` | `browser()`, through Playwright |
| `computer` | `computer()`, through pyautogui |

```bash
pip install "nanoagent[anthropic,voice] @ git+https://github.com/gabrielekarra/nanoagent"
```

Requires Python 3.10 or later.

## Every ability, one line

| You want it to | Line | Example |
|---|---|---|
| use your functions | `agent.use(get_weather)` or `Agent(tools=[...])` | [hello.py](examples/hello.py) |
| run on a tiny model | `Agent(model=Ollama("smollm2:1.7b", tools="guided"))` | [tiny_model.py](examples/tiny_model.py) |
| be on Telegram | `agent.use(telegram())` | [telegram_bot.py](examples/telegram_bot.py) |
| be on WhatsApp | `agent.use(whatsapp(allow=["39333..."]))` | [whatsapp_bot.py](examples/whatsapp_bot.py) |
| be on Discord, Slack, email | `agent.use(discord())`, `slack()`, `email(...)` | [docs/channels.md](docs/channels.md) |
| listen and speak | `agent.use(voice())` | [voice_assistant.py](examples/voice_assistant.py) |
| be an API, or a model for other apps | `agent.use(http(port=8080))` | [as_an_api.py](examples/as_an_api.py) |
| act on a schedule | `agent.use(every("1h", "..."))`, `at("mon 09:00", "...")` | [scheduled.py](examples/scheduled.py) |
| contact you first | `agent.use(telegram(), notify=True)` | [scheduled.py](examples/scheduled.py) |
| control an app | `agent.use(app("github"))` | [github_agent.py](examples/github_agent.py) |
| use any MCP server | `agent.use(mcp("uvx mcp-server-git"))` | [MCP](#apps-and-mcp-servers) |
| browse the web | `agent.use(browser())` | [browser_agent.py](examples/browser_agent.py) |
| use the mouse and keyboard | `agent.use(computer())` | [computer](#the-browser-and-the-computer) |
| follow your rules | `agent.use(rules(allow=["git status"], deny=["rm *"]))` | [safe_coder.py](examples/safe_coder.py) |
| stay in a folder, or a container | `agent.use(sandbox(root="."))`, `sandbox("docker")` | [safe_coder.py](examples/safe_coder.py) |
| keep a record you can replay | `agent.use(log("runs/"))` | [safe_coder.py](examples/safe_coder.py) |
| remember conversations | `Agent(state="~/.nanoagent/jarvis")` | [jarvis.py](examples/jarvis.py) |
| be always on | `agent.live()`, then `nanoagent service install agent.py` | [always on](#always-on) |

## Models

| `model=` | What it talks to | Key |
|---|---|---|
| `"ollama:qwen3:4b"` | Ollama, native API (`OLLAMA_HOST`, default `localhost:11434`) | none |
| `"llamacpp:any"` | llama.cpp server on `localhost:8080` | none |
| `"lmstudio:qwen3-4b"` | LM Studio on `localhost:1234` | none |
| `"vllm:Qwen/Qwen3-8B"` | vLLM on `localhost:8000` | none |
| `"openrouter:meta-llama/llama-3.3-70b-instruct"` | OpenRouter | `OPENROUTER_API_KEY` |
| `"openai:gpt-5"` | OpenAI | `OPENAI_API_KEY` |
| `"anthropic:claude-opus-5-5"` or `"claude-opus-5-5"` | Anthropic (the `anthropic` extra) | `ANTHROPIC_API_KEY` |

With no `model=`, the `NANOAGENT_MODEL` environment variable is used. For other servers, or to set options:

```python
from nanoagent.providers.ollama import Ollama
from nanoagent.providers.openai import OpenAICompatible

Agent(model=OpenAICompatible("http://gpu-box:8000/v1", "my-model", context_window=16_000, temperature=0.2))
Agent(model=Ollama("llama3.2:3b", context=16_384, pull=True))   # downloads the model if missing
```

Ollama has its own adapter because, on its OpenAI-compatible endpoint, it runs models with a small default window and silently drops the start of a long conversation: the instructions and the tools.

A model of your own is one method, `reply` or its async form `areply`:

```python
from nanoagent import Call, Model, Reply

class MyModel(Model):
    context_window = 8000

    def reply(self, system, tools, messages, on_text=None):
        return Reply(text="Hello")      # or Reply(calls=[Call(id, name, arguments)], stop="tool")
```

### Models that cannot call tools

Small models often have no tool-calling template, or use it unreliably. With `tools="guided"`, the model answers each step with one JSON object, a tool call or the final answer:

```json
{"thought": "I need the weather", "tool": "get_weather", "arguments": {"city": "Rome"}}
{"thought": "I have it", "answer": "Sunny, no umbrella needed."}
```

Where the server supports constrained decoding (llama.cpp, Ollama, vLLM, LM Studio), a JSON schema built from your tools makes anything else impossible. Whatever comes back is still parsed defensively (code fences, prose around the JSON, trailing commas, single quotes, `<think>` blocks, common alternative key names), and an unreadable reply gets one correction.

You rarely choose: the default `tools="auto"` starts native and switches to guided as soon as the server says the model has no tool support. In native mode, a call the model wrote as JSON text is recognized and run.

### Small context windows

Defaults follow the model's window. With an 8k model the history is summarized at 70% of the window, a tool result is clipped to about a quarter of it, the request is checked before it is sent so the server never truncates it, and with more than 8 tools each request carries only the ones relevant to the conversation, plus a `find_tools` tool to ask for others.

### Which models work

```bash
nanoagent bench --model ollama:qwen3:4b
```

```bash
nanoagent bench --model ollama:gemma2:2b --guided
```

Twenty short tasks (file edits, lookups over several steps, recovering from a failing tool, knowing when not to use one), each in an empty folder, checked on the outcome. The output is a Markdown table. Results for models that fit in 16 GB will be published here; run it on yours and open a pull request with the table.

## Tools

A tool is a function. Its name, docstring and type hints are what the model sees.

```python
from typing import Annotated
from nanoagent import Agent, Image, tool

@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return "sunny, 24°C"

@tool
def search_orders(customer: Annotated[str, "Customer email"], limit: int = 10) -> str:
    """Find a customer's most recent orders."""
    return "A17, A18"

@tool(approve=True)                       # asks before running
def refund(order_id: str) -> str:
    """Refund an order in full."""
    return f"refunded {order_id}"

@tool
async def fetch(url: str) -> str:         # async functions work too
    """Fetch a page."""
    return "<html>...</html>"
```

Parameters can be `str`, `int`, `float` or `bool`. Input written by the model is checked against the schema before your function runs, a tool that raises sends its error to the model, and calls in one reply run in parallel (`Agent(parallel=4)`, or `@tool(parallel=False)` to opt out). A tool can return an `Image`, alone or in a list with text, for models that see images.

```python
agent = Agent(
    "You handle support for an online shop.",
    tools=[search_orders, refund],
    model="ollama:qwen3:4b",
    approve=lambda name, args: input(f"{name}({args})? [y/n] ") == "y",
    on_text=lambda chunk: print(chunk, end="", flush=True),   # stream the reply
)
agent.run("Refund the last order of ada@example.com")
agent.run("And tell me what you refunded")    # the conversation continues
```

| Option | What it does |
|---|---|
| `model` | `"provider:model"` or a `Model`. Default: `$NANOAGENT_MODEL`. |
| `approve` | Asked before each tool marked `approve=True` (may be async). Over a channel, the question is asked there instead. |
| `on_text`, `on_tool` | Callbacks for streamed text and tool calls. |
| `memory="NOTES.md"` | Loads the file into the instructions and adds a `remember` tool that appends to it. |
| `subagents=True` | Adds a `task` tool that hands work to a fresh agent and gets back only its report. |
| `state="dir"` | Saves every conversation there, so it survives restarts. |
| `parallel`, `max_steps`, `shortlist` | Concurrent calls (4), a step limit per request, the tool shortlist size (8 for small windows). |
| `compact_at`, `max_output` | When to summarize the history and how much of a tool result to keep. Default: from the model's window. |

`agent.messages` holds the conversation; `agent.stop_reason` says why the last reply ended (`"end"`, `"length"`, `"refusal"`, `"steps"`). `await agent.arun(...)` is the async form of `run`.

## Channels

A channel is a way to talk to the agent. `agent.use(channel)` adds it; `agent.live()` runs them all and answers each message where it came from, keeping one conversation per person per channel.

```python
from nanoagent import discord, email, http, slack, telegram, voice, whatsapp

agent = Agent("You are a helpful assistant.", model="ollama:qwen3:4b", state="~/.nanoagent/bot")
agent.use(telegram(allow=[123456789]))                      # TELEGRAM_BOT_TOKEN
agent.use(whatsapp(allow=["393331234567"]))                 # WHATSAPP_* (Cloud API)
agent.use(discord())                                        # DISCORD_BOT_TOKEN
agent.use(slack())                                          # SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET
agent.use(email("agent@example.com", allow=["me@example.com"]))   # EMAIL_* (IMAP and SMTP)
agent.use(http(port=8080))                                  # POST /message, /v1/chat/completions
agent.use(voice())                                          # microphone and speakers, locally
agent.live()
```

Messaging channels are **private by default**: they answer only their allow list. With an empty list, a pairing code is printed at startup and the first person to send it becomes the owner. Tools that need approval are asked about on the channel the request came from, by the same person, with Yes and No buttons where the service has them.

`http()` also speaks the OpenAI Chat Completions format, so any chat interface or client library can use the agent, with its tools and memory, as if it were a model. `voice()` speaks a sentence at a time while the model is still writing, summarizes code and links instead of reading them, and runs Whisper and Piper locally.

Writing a channel is three methods; [docs/channels.md](docs/channels.md) shows how, with a contract test suite every channel passes.

## Acting on its own

```python
from nanoagent import at, every

agent.use(telegram(), notify=True, quiet=("23:00", "07:00"))       # where unprompted messages go
agent.use(every("1h", "Check the build; tell me only if it is red"))
agent.use(at("weekdays 08:30", "Summarize my calendar for today"))
```

Trigger answers go to the channel added with `notify=True`, and the model gets a `notify` tool to send a message in the middle of a task. A rate limit stops a confused model from sending a hundred messages; quiet hours hold messages until morning. With `state=`, a restart never runs a trigger twice, and a run missed while the agent was down happens once when it is back (`catch_up=False` skips it). Approvals needed by trigger work are declined, or asked on the notify channel with `agent.live(approvals="ask_default")`.

## Always on

```bash
nanoagent run jarvis.py
```

```bash
nanoagent service install jarvis.py --env TELEGRAM_BOT_TOKEN=...
```

The second writes a systemd user unit (Linux) or a launchd agent (macOS) that starts the agent at login and restarts it if it crashes; `status`, `logs` and `uninstall` complete the set, and `--dry-run` prints the file. Secrets go to a file only you can read, never into the service definition.

## Apps and MCP servers

```python
from nanoagent import app, mcp

agent.use(app("github"))                               # a reviewed list: nanoagent apps
agent.use(app("filesystem", "~/notes"))
agent.use(mcp("uvx mcp-server-git", trust=["git_status", "git_log"]))
agent.use(mcp("https://example.com/mcp", headers={"Authorization": "Bearer ..."}))
```

Any [MCP](https://modelcontextprotocol.io) server becomes tools: started when added, stopped with the agent, restarted if it dies, over stdio or HTTP, with no dependency. Its tools ask for approval unless trusted, and the server sees only the environment variables you pass it. `app()` knows the command and the variables for common servers and says where to get a missing token.

## The browser and the computer

```python
from nanoagent import browser, computer

agent.use(browser())       # open_page, read_page, click, type_text, submit, go_back
```

Pages are read as compact text with every link, button and field numbered, sized for small windows. Submitting a form needs approval, and page text is marked as untrusted, so the model treats it as information rather than instructions. `computer()` gives a vision model the screen, mouse and keyboard, every action approved; it is experimental, and small local models are not good at it yet.

## Safety

```python
from nanoagent import log, rules, sandbox

agent.use(rules(allow=["git status", "git diff *", "pytest *"], deny=["rm *", "curl *"]))
agent.use(sandbox(root="."))        # file tools cannot leave this folder
agent.use(log("runs/"))             # everything, with secrets masked
```

Rules decide before approval: allowed commands run without asking, denied ones never run even if approved, the rest is asked. Chained commands are split, so a rule for `git status` does not allow `git status && rm -rf /`. `sandbox("docker")` also runs shell commands in a container that sees only the folder and has no network. Without it, the shell runs with your permissions, so read what you approve.

`nanoagent show runs/<file>.jsonl` prints a recorded session, and `FakeModel.from_log(path)` replays it as a test.

## Testing without a model

```python
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

test_refund_needs_approval()
```

`model.requests` records the instructions, tools and messages of every request. `FakeChannel`, `FakeClock` and `running(agent)` test channels, triggers and approvals; `ChannelContract` tests a channel you wrote.

## The tutorial: build it from scratch

Each chapter is one file that runs on its own and adds one idea to the previous one. Diff two neighbours to see exactly what an idea costs.

| Chapter | Lines | The idea |
|---|---|---|
| [`01_loop.py`](tutorial/01_loop.py) | 85 | An agent is a model, a tool and a `while` loop. |
| [`02_tools.py`](tutorial/02_tools.py) | 133 | Tools are functions. A failed tool is information for the model, not a crash. |
| [`03_permissions.py`](tutorial/03_permissions.py) | 171 | The model proposes, the harness decides. Ctrl-C without corrupting the conversation. |
| [`04_streaming.py`](tutorial/04_streaming.py) | 210 | Show text as it is written. Never run a tool call that was cut off or malformed. |
| [`05_context.py`](tutorial/05_context.py) | 272 | Clip big outputs, cache the unchanged prefix, compact when the conversation gets large. |
| [`06_memory.py`](tutorial/06_memory.py) | 285 | A notes file that outlives the session. |
| [`07_subagents.py`](tutorial/07_subagents.py) | 298 | A subagent is the same loop, called from inside a tool. |
| [`08_any_model.py`](tutorial/08_any_model.py) | 148 | Any model: Chat Completions on the standard library, and guided JSON steps for models without tools. |
| [`09_always_on.py`](tutorial/09_always_on.py) | 117 | Always on: channels and triggers feed one queue of events. |
| [`10_a_channel.py`](tutorial/10_a_channel.py) | 122 | A real channel, Telegram, in a page. |

Chapters 1 to 7 use Claude (the `anthropic` extra and a key); 8 to 10 use any OpenAI-compatible server, such as Ollama.

## Design rules

- **Small models first.** Every feature is designed and tested for a model of a few billion parameters on a laptop.
- **The history is append-only.** Messages are never edited in place; the history is extended, or replaced whole by a summary. This keeps prompt caches valid, and some models reject edited histories.
- **The instructions and the tools are fixed** once a conversation starts, for the same reason. Add tools before the first run.
- **Every tool call gets a result**, even when it was declined, invalid, cut off or interrupted.
- **Each provider's reply is stored as received** next to the neutral form, and replayed unchanged to the provider that wrote it.

The core (`agent.py`, `model.py` and `tools.py`) is 847 lines. The whole package, every provider and channel included, is 5319 lines of Python with no dependencies.

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

The suite covers every provider and channel against fake versions of their services, every tutorial chapter, every example and every Python block in this README, in about half a minute with no model and no key. Tests that need real things are marked and skipped unless available: `local` (set `NANOAGENT_LOCAL_MODEL=ollama:qwen2.5:1.5b`), `browser`, `docker`, `voice`.

The plan is in [ROADMAP.md](ROADMAP.md); [CONTRIBUTING.md](CONTRIBUTING.md) says what fits.

## License

MIT
