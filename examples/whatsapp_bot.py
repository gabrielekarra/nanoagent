"""Your agent on WhatsApp, through the Cloud API (see nanoagent/channels/whatsapp.py for the setup).

    WHATSAPP_TOKEN=... WHATSAPP_PHONE_ID=... WHATSAPP_APP_SECRET=... WHATSAPP_VERIFY_TOKEN=... \\
        python examples/whatsapp_bot.py
"""

import os

from nanoagent import Agent, whatsapp

agent = Agent("You are a helpful assistant. Keep answers short.", model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"))
agent.use(whatsapp(allow=os.environ.get("WHATSAPP_ALLOW", "").split(",")))
agent.live()
