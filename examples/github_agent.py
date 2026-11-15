"""An agent that works on GitHub, through GitHub's own MCP server (needs Docker).

    GITHUB_PERSONAL_ACCESS_TOKEN=... python examples/github_agent.py
"""

import os

from nanoagent import Agent, app

agent = Agent("You help maintain GitHub repositories.", model=os.environ.get("NANOAGENT_MODEL", "anthropic:claude-opus-5-5"),
              approve=lambda name, arguments: input(f"Allow {name}? [y/n] ") == "y")
agent.use(app("github"))
print(agent.run("List my open pull requests and say which one needs attention first."))
