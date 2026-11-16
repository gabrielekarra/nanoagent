"""A coding agent with guard rails: rules, a sandboxed folder, and a log of everything."""

import os

from nanoagent import CODING_TOOLS, Agent, log, rules, sandbox

agent = Agent("You are a coding agent. Work only in this folder.", tools=CODING_TOOLS,
              model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"),
              approve=lambda name, arguments: input(f"Allow {name}({arguments})? [y/n] ") == "y")
agent.use(rules(allow=["ls *", "cat *", "git status", "git diff *", "pytest *"], deny=["rm -rf *", "curl *", "sudo *"]))
agent.use(sandbox(root="."))
agent.use(log("runs"))

while True:
    try:
        print(agent.run(input("> ")))
    except (EOFError, KeyboardInterrupt):
        break
