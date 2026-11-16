"""An agent that reads the web.

    pip install 'nanoagent[browser]' && playwright install chromium && python examples/browser_agent.py
"""

import os

from nanoagent import Agent, browser

agent = Agent("You research questions on the web and cite the pages you used.",
              model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:8b"))
agent.use(browser())
print(agent.run("What is the latest Python release? Check python.org."))
agent.close()
