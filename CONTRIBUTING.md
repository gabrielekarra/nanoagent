# Contributing

nanoagent is deliberately small. The most useful contributions keep it that way.

## What fits

- Bug fixes, with a test that fails before the fix.
- Clearer wording in the tutorial or README.
- Work on an issue from the [roadmap](ROADMAP.md). Each issue lists the tests that define it as done.
- A new model provider: one file in `nanoagent/providers/`, tested against a fake server in `tests/fakes.py`.
- A new channel: one file in `nanoagent/channels/`, with a fake of its service, passing `ChannelContract` (see [docs/channels.md](docs/channels.md)).
- An entry in the `app()` registry (`nanoagent/apps.py`), pointing to the server's official source.
- Benchmark results: run `nanoagent bench --model ...` and add the table to the README.
- Tutorial chapters or short guides that build on the SDK.

## What probably does not

Anything that grows the core (`agent.py`, `model.py` and `tools.py`) well past 1,000 lines combined, or adds a dependency to it. Optional extras are fine for features that need one (a provider SDK, voice, the browser). A new ability should be one `agent.use(...)` line for the person using it. If you are unsure, open an issue first and describe the idea in a few lines.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

The suite needs no model and no API key. The SDK tests use `FakeModel`, an in-process model with scripted replies; each provider is tested against a fake server in `tests/fakes.py` that speaks its wire format. Tests that need a real local model are marked `local` and skipped unless `NANOAGENT_LOCAL_MODEL` is set.

## Style

Plain Python, no framework, comments only where the reason is not obvious from the code.
