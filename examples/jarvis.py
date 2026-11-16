"""Everything at once: a local model that listens, speaks, lives on your phone,
checks things on its own, controls an app and cannot hurt your machine.

    NANOAGENT_MODEL=ollama:qwen3:4b TELEGRAM_BOT_TOKEN=... GITHUB_PERSONAL_ACCESS_TOKEN=... \\
        python examples/jarvis.py
"""

import os

from nanoagent import CODING_TOOLS, Agent, app, every, sandbox, telegram, voice

agent = Agent("You are Jarvis, a helpful assistant.", tools=CODING_TOOLS,
              model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"), state="~/.nanoagent/jarvis")
agent.use(voice())                                              # it listens and speaks
agent.use(telegram(), notify=True)                              # it is on your phone
agent.use(every("1h", "Check my GitHub notifications, tell me what matters"))  # it acts on its own
agent.use(app("github"))                                        # it controls an app
agent.use(sandbox(root="~/jarvis"))                             # its files stay in one folder
agent.live()                                                    # it is always on
