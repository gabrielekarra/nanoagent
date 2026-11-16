# Roadmap

**Any model becomes an agent, and every ability is one line.**

```python
from nanoagent import Agent, voice, telegram, every, app, sandbox

agent = Agent("You are Jarvis.", model="ollama:qwen3:4b")   # a model that fits in 16 GB of RAM
agent.use(voice())                                          # it listens and speaks
agent.use(telegram())                                       # it is on your phone
agent.use(every("1h", "Check my inbox, tell me what matters"))  # it acts on its own
agent.use(app("github"))                                    # it controls an app
agent.use(sandbox("docker"))                                # it cannot hurt your machine
agent.live()                                                # it is always on
```

Other agent frameworks compete on how much they can orchestrate. nanoagent competes on two things they mostly ignore: the smallest models, and the shortest path from an idea ("I want it to talk", "I want it on WhatsApp") to a running agent.

## Principles

1. **Small models first.** Every feature is designed and tested for a model of a few billion parameters running on a laptop. If it works there, it works everywhere. Large hosted models are supported, never assumed.
2. **One line per ability.** `agent.use(thing)` plugs in a tool, a channel (a way to talk to the agent) or a trigger (a reason for it to act). If an ability needs more than one line, the design is not finished.
3. **The core has no dependencies.** It runs on the Python standard library. Voice, the browser and each provider SDK are optional extras.
4. **Readable all the way down.** Every module stays small enough to read in one sitting, and the tutorial keeps explaining how it works.
5. **Tested without a model or a network.** Every feature ships with fakes, so its tests run in seconds on any machine. Tests that need a real model or real hardware are marked and skipped by default.

## Milestones

Each issue lists the tests that define it as done.

### M1 - Any model

The foundation: the loop stops depending on one provider, and small models learn to use tools.

- [#1](https://github.com/gabrielekarra/nanoagent/issues/1) Provider-neutral core: a `Model` interface
- [#2](https://github.com/gabrielekarra/nanoagent/issues/2) OpenAI-compatible model: Ollama, llama.cpp, LM Studio, vLLM, OpenRouter, OpenAI
- [#3](https://github.com/gabrielekarra/nanoagent/issues/3) Guided tool calling: models with no tool support become agents
- [#4](https://github.com/gabrielekarra/nanoagent/issues/4) Ollama native adapter: context length and model pull
- [#5](https://github.com/gabrielekarra/nanoagent/issues/5) Context budget for 4k-8k windows
- [#6](https://github.com/gabrielekarra/nanoagent/issues/6) Tool shortlist for small models
- [#7](https://github.com/gabrielekarra/nanoagent/issues/7) `nanoagent bench`: which small models can be agents

### M2 - Always on

- [#8](https://github.com/gabrielekarra/nanoagent/issues/8) Async core: `await agent.arun()` and `async def` tools
- [#9](https://github.com/gabrielekarra/nanoagent/issues/9) Parallel tool execution
- [#10](https://github.com/gabrielekarra/nanoagent/issues/10) Runtime: `agent.use()` and `agent.live()`
- [#11](https://github.com/gabrielekarra/nanoagent/issues/11) Sessions on disk
- [#12](https://github.com/gabrielekarra/nanoagent/issues/12) Triggers: `every()` and `at()`
- [#13](https://github.com/gabrielekarra/nanoagent/issues/13) `notify`: the agent contacts you first
- [#14](https://github.com/gabrielekarra/nanoagent/issues/14) Approvals over any channel
- [#15](https://github.com/gabrielekarra/nanoagent/issues/15) `nanoagent service`: start at boot, restart on crash

### M3 - Channels

- [#16](https://github.com/gabrielekarra/nanoagent/issues/16) Telegram
- [#17](https://github.com/gabrielekarra/nanoagent/issues/17) WhatsApp
- [#18](https://github.com/gabrielekarra/nanoagent/issues/18) HTTP, and the agent as an OpenAI-compatible endpoint
- [#19](https://github.com/gabrielekarra/nanoagent/issues/19) Voice: listens and speaks, locally
- [#20](https://github.com/gabrielekarra/nanoagent/issues/20) Channel authoring guide; Discord, Slack, email

### M4 - Hands

- [#21](https://github.com/gabrielekarra/nanoagent/issues/21) MCP client: any MCP server in one line
- [#22](https://github.com/gabrielekarra/nanoagent/issues/22) `app()`: control an app by name
- [#23](https://github.com/gabrielekarra/nanoagent/issues/23) Browser
- [#24](https://github.com/gabrielekarra/nanoagent/issues/24) Computer control (experimental)

### M5 - Trust and launch

- [#25](https://github.com/gabrielekarra/nanoagent/issues/25) Approval rules and a sandboxed shell
- [#26](https://github.com/gabrielekarra/nanoagent/issues/26) Record and replay
- [#27](https://github.com/gabrielekarra/nanoagent/issues/27) Continuous integration and a PyPI release
- [#28](https://github.com/gabrielekarra/nanoagent/issues/28) Launch: README, examples and demo

## Order of work

M1 comes first because everything else runs on it, and because it is what makes the project different. M2 follows because channels and apps both need the runtime. M3 and M4 can proceed in parallel after that. #27 (continuous integration) can land at any time and should land early.
