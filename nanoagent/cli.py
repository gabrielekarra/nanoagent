"""The `nanoagent` command.

    nanoagent --model ollama:qwen3:4b              a coding agent in your terminal
    nanoagent run jarvis.py                        run a file that calls agent.live()
    nanoagent service install jarvis.py            keep it running (see service.py)
    nanoagent apps                                 the apps app("...") can control
    nanoagent show runs/default-2026-11-15.jsonl   a recorded session, as a transcript
    nanoagent bench --model ollama:qwen3:4b        how well a model works as an agent

The coding agent is the SDK used the way you would use it: the built-in
coding tools, approval before anything that changes your machine, notes
kept in AGENT.md.
"""

import argparse
import json
import os
import subprocess
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


COMMANDS = ("run", "service", "apps", "show", "bench")


def main(model=None, argv=None):
    """Entry point. `main(model)` starts the terminal agent directly (used by tests)."""
    if model is not None:
        return chat(model)
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        return command(argv)
    parser = argparse.ArgumentParser(prog="nanoagent", description="A coding agent for your terminal.",
                                     epilog=f"Other commands: {', '.join(COMMANDS)} (nanoagent <command> --help)")
    parser.add_argument("--model", default=os.environ.get("NANOAGENT_MODEL"),
                        help="e.g. ollama:qwen3:4b or anthropic:claude-opus-5-5 (default: $NANOAGENT_MODEL)")
    model = parser.parse_args(argv).model
    if not model:
        parser.error("choose a model with --model, e.g. --model ollama:qwen3:4b")
    return chat(model)


def command(argv):
    parser = argparse.ArgumentParser(prog="nanoagent")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a file that defines an agent and calls agent.live()")
    run.add_argument("script")
    run.add_argument("--env-file", help="load KEY=VALUE lines into the environment first")
    service = commands.add_parser("service", help="start an agent at login and restart it if it crashes")
    service.add_argument("action", choices=["install", "status", "logs", "uninstall"])
    service.add_argument("script")
    service.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                         help="a variable for the agent, e.g. a token; stored in a private file")
    service.add_argument("--dry-run", action="store_true", help="print the service file instead of installing it")
    service.add_argument("--system", choices=["linux", "macos"], help="default: this machine")
    commands.add_parser("apps", help="list the apps app() can control")
    show = commands.add_parser("show", help="print a log recorded with log() as a transcript")
    show.add_argument("file")
    bench = commands.add_parser("bench", help="measure how well a model works as an agent")
    bench.add_argument("--model", default=os.environ.get("NANOAGENT_MODEL"), required=not os.environ.get("NANOAGENT_MODEL"))
    bench.add_argument("--guided", action="store_true", help="use guided tool calling (JSON steps)")
    bench.add_argument("--tasks", help="comma-separated task names (default: all)")
    bench.add_argument("--steps", type=int, default=12, help="step limit per task")
    options = parser.parse_args(argv)

    if options.command == "bench":
        from . import bench as benchmark
        from .model import resolve

        model = resolve(options.model)
        if options.guided:
            if not hasattr(model, "tools") or isinstance(model.tools, dict):
                parser.exit(1, "nanoagent: --guided works with Ollama and OpenAI-compatible models\n")
            model.tools = "guided"
        chosen = [t for t in benchmark.TASKS if not options.tasks or t.name in options.tasks.split(",")]
        title = f"{options.model}{' (guided)' if options.guided else ''}"
        return print(benchmark.table(benchmark.run(model, chosen, options.steps), title))

    if options.command == "show":
        from .log import show as transcript

        return print(transcript(options.file))

    if options.command == "apps":
        from .apps import listing

        return print(listing())
    if options.command == "run":
        from .service import run_file

        return run_file(options.script, options.env_file)

    from .service import Service

    target = Service(options.script, system=options.system)
    if options.action == "install":
        env = dict(item.split("=", 1) for item in options.env if "=" in item)
        try:
            print(target.install(env, dry_run=options.dry_run), end="" if options.dry_run else "\n")
        except FileNotFoundError as error:
            parser.exit(1, f"nanoagent: {error}\n")
    elif options.action == "uninstall":
        removed = target.uninstall()
        print("\n".join(f"Removed {path}" for path in removed) or "Nothing to remove.")
    else:
        command = target.status_command() if options.action == "status" else target.logs_command()
        return subprocess.run(command, check=False).returncode


def chat(model):
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
