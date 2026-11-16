"""The agent behind an HTTP endpoint, including the OpenAI Chat Completions format.

    python examples/as_an_api.py
    curl localhost:8080/message -d '{"text": "hello"}'

Any OpenAI-compatible chat interface can use http://localhost:8080/v1 as its server.
"""

import os

from nanoagent import Agent, http

agent = Agent("You are a helpful assistant.", model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"), state="~/.nanoagent/api")
agent.use(http(port=8080))
agent.live()
