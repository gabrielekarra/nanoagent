"""An assistant you talk to, entirely on your machine.

    pip install 'nanoagent[voice]' && python examples/voice_assistant.py
"""

import os

from nanoagent import Agent, voice

agent = Agent("You are a voice assistant. Answer in one or two short sentences.",
              model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"))
agent.use(voice())
agent.live()
