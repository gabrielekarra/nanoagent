"""A coding agent for your terminal, in one screen of code.

    nanoagent            # or: python -m nanoagent

It is the SDK used the way you would use it: the built-in coding tools,
approval before anything that changes your machine, notes kept in AGENT.md.
"""

import json
import os

from . import CODING_TOOLS, Agent

always_allowed = set()


def ask_in_terminal(name, arguments):
    if name in always_allowed:
        return True
    answer = input("  allow? [y]es / [n]o / [a]lways for this tool: ").strip().lower()
    if answer.startswith("a"):
        always_allowed.add(name)
    return answer.startswith(("y", "a"))


def show_tool(name, arguments):
    print(f"\n  [{name}] {json.dumps(arguments)[:100]}")


def main():
    agent = Agent(
        f"You are a coding agent working in {os.getcwd()}. Use your tools to do what "
        "the user asks, then say briefly what you did. When you learn something that "
        "will matter in later sessions (a build command, a convention, a preference "
        "of the user), save it with `remember`.",
        tools=CODING_TOOLS,
        model=os.environ.get("AGENT_MODEL", "claude-opus-5-5"),
        approve=ask_in_terminal,
        on_text=lambda text: print(text, end="", flush=True),
        on_tool=show_tool,
        memory="AGENT.md",
        subagents=True,
    )
    while True:
        try:
            request = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        try:
            agent.run(request)
            if agent.stop_reason == "refusal":
                print("(the model declined this request)")
        except KeyboardInterrupt:  # Ctrl-C stops the agent, not the program
            print("\n(interrupted)")
        print()


if __name__ == "__main__":
    main()
