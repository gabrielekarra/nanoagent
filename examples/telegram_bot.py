"""Your agent on Telegram. Get a token from @BotFather, then:

    TELEGRAM_BOT_TOKEN=... python examples/telegram_bot.py

The first time, send the pairing code it prints to your bot.
"""

import os

from nanoagent import Agent, telegram

agent = Agent("You are a helpful assistant.", model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"), state="~/.nanoagent/telegram")
agent.use(telegram(), notify=True)
agent.live()
