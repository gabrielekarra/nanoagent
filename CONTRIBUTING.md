# Contributing

nanoagent is deliberately small. The most useful contributions keep it that way.

## What fits

- Bug fixes, with a test that fails before the fix.
- Clearer wording in the tutorial or README.
- Tutorial chapters or short guides that build on the SDK, for example "add another model provider" or "save a session to disk".

## What probably does not

Anything that grows `nanoagent/agent.py` and `nanoagent/tools.py` well past 400 lines combined, or adds a dependency. If you are unsure, open an issue first and describe the idea in a few lines.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

The suite needs no API key. It talks to `FakeModel`, a local server that speaks the Messages API.

## Style

Plain Python, no framework, comments only where the reason is not obvious from the code.
