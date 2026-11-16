"""A coding agent for your terminal, in one screen of code.

    nanoagent --model ollama:qwen3:4b          # or set NANOAGENT_MODEL
    python -m nanoagent --model anthropic:claude-opus-5-5

It is the SDK used the way you would use it: the built-in coding tools,
approval before anything that changes your machine, notes kept in AGENT.md.
"""

import argparse
import json
import os
import sys

from . import CODING_TOOLS, Agent, ModelError

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


def main(model=None):
    if model is None:
        parser = argparse.ArgumentParser(prog="nanoagent", description="A coding agent for your terminal.")
        parser.add_argument("--model", default=os.environ.get("NANOAGENT_MODEL"),
                            help="e.g. ollama:qwen3:4b or anthropic:claude-opus-5-5 (default: $NANOAGENT_MODEL)")
        model = parser.parse_args().model
        if not model:
            parser.error("choose a model with --model, e.g. --model ollama:qwen3:4b")
    agent = Agent(
        f"You are a coding agent working in {os.getcwd()}. Use your tools to do what "
        "the user asks, then say briefly what you did. When you learn something that "
        "will matter in later sessions (a build command, a convention, a preference "
        "of the user), save it with `remember`.",
        tools=CODING_TOOLS,
        model=model,
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
        except ModelError as error:
            print(f"\n{error}", file=sys.stderr)
        print()


if __name__ == "__main__":
    main()
