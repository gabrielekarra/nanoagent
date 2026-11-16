"""Which models can be agents? A small, fixed benchmark.

    nanoagent bench --model ollama:qwen3:4b
    nanoagent bench --model ollama:gemma2:2b --guided
    nanoagent bench --model ollama:qwen3:4b --tasks sum_numbers,chain_lookup

Twenty short tasks, each in an empty folder of its own: file edits, lookups
over several steps, recovering from a failing tool, and knowing when not to
use a tool. A task passes when its outcome is right (the file, the answer),
whatever the transcript looks like. Each task also has a scripted solution,
which is how the benchmark itself is tested.
"""

import itertools
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .agent import Agent
from .tools import edit_file, read_file, tool, write_file

FILE_TOOLS = [read_file, write_file, edit_file]


@dataclass
class Task:
    name: str
    prompt: str
    check: Callable  # check(folder, answer, calls) -> bool
    files: dict = field(default_factory=dict)
    tools: list = field(default_factory=lambda: list(FILE_TOOLS))
    solution: list = field(default_factory=list)  # scripted replies that solve it
    fresh: Callable = None  # makes new tools for each run, for tools that keep state


# --- tools some tasks use ---------------------------------------------------------------

CAPITALS = {"zubrowka": "Lutz", "freedonia": "Fredville", "genovia": "Pyrus"}
POPULATIONS = {"lutz": 41_200, "fredville": 38_900}
MANAGERS = {"ada": "Ben", "ben": "Cleo", "cleo": "Dara"}


@tool
def capital(country: str) -> str:
    """The capital city of a country."""
    return CAPITALS.get(country.strip().lower(), f"Unknown country: {country}")


@tool
def population(city: str) -> str:
    """The population of a city."""
    found = POPULATIONS.get(city.strip().lower())
    return f"{found}" if found else f"Unknown city: {city}"


@tool
def manager_of(person: str) -> str:
    """The name of a person's manager."""
    return MANAGERS.get(person.strip().lower(), f"Unknown person: {person}")


def flaky():
    failures = [1]

    @tool
    def build_status() -> str:
        """The status of the build."""
        if failures:
            failures.pop()
            raise ConnectionError("temporary network failure, try again")
        return "green: all 214 tests pass"

    return build_status


@tool
def convert(amount: float, currency: str) -> str:
    """Convert an amount of euros to another currency, given its ISO code (e.g. USD)."""
    rates = {"USD": 1.10, "GBP": 0.85}
    if currency not in rates:
        raise ValueError(f"Unknown currency {currency!r}: use an ISO code in capitals, one of {', '.join(rates)}.")
    return f"{amount * rates[currency]:.2f} {currency}"


# --- the tasks -----------------------------------------------------------------------------


def _file(name, expected):
    return lambda folder, answer, calls: (folder / name).exists() and (folder / name).read_text().strip() == expected


def _says(*words):
    return lambda folder, answer, calls: all(word.lower() in (answer or "").lower() for word in words)


_ids = itertools.count(1)


def _call(name, **arguments):
    return {"type": "tool_use", "id": f"bench_{next(_ids)}", "name": name, "input": arguments}


def _say(text):
    return {"type": "text", "text": text}


def _step(*blocks):
    used = any(block["type"] == "tool_use" for block in blocks)
    return {"content": list(blocks), "stop": "tool" if used else "end", "tokens": 100}


def _json_age(folder, answer, calls):
    try:
        return json.loads((folder / "data.json").read_text()) == {"name": "Ada", "age": 37}
    except (OSError, ValueError):
        return False


TASKS = [
    Task("write_greeting", "Create a file hello.txt containing exactly: Hello, world", _file("hello.txt", "Hello, world"),
         solution=[_step(_call("write_file", path="hello.txt", content="Hello, world")), _step(_say("Created hello.txt."))]),
    Task("read_and_answer", "Read notes.txt. On which day is the meeting?", _says("thursday"),
         files={"notes.txt": "Reminder: the team meeting moved to Thursday at 10.\n"},
         solution=[_step(_call("read_file", path="notes.txt")), _step(_say("The meeting is on Thursday."))]),
    Task("sum_numbers", "Read prices.txt, add up the numbers in it, and write only the total to total.txt.",
         _file("total.txt", "12"), files={"prices.txt": "apples 3\npears 4\nfigs 5\n"},
         solution=[_step(_call("read_file", path="prices.txt")), _step(_call("write_file", path="total.txt", content="12")),
                   _step(_say("The total is 12."))]),
    Task("edit_setting", "In config.ini, turn debug on (debug = true). Change nothing else.",
         _file("config.ini", "name = shop\ndebug = true\nport = 8000"), files={"config.ini": "name = shop\ndebug = false\nport = 8000\n"},
         solution=[_step(_call("read_file", path="config.ini")),
                   _step(_call("edit_file", path="config.ini", old="debug = false", new="debug = true")), _step(_say("Done."))]),
    Task("count_lines", "How many lines does list.txt have? Answer with the number.", _says("7"),
         files={"list.txt": "".join(f"item {n}\n" for n in range(7))},
         solution=[_step(_call("read_file", path="list.txt")), _step(_say("It has 7 lines."))]),
    Task("copy_file", "Copy the contents of a.txt into a new file b.txt.", _file("b.txt", "The quick brown fox."),
         files={"a.txt": "The quick brown fox.\n"},
         solution=[_step(_call("read_file", path="a.txt")), _step(_call("write_file", path="b.txt", content="The quick brown fox.\n")),
                   _step(_say("Copied."))]),
    Task("uppercase", "Write the contents of quiet.txt in capital letters to loud.txt.", _file("loud.txt", "PLEASE BE QUIET"),
         files={"quiet.txt": "please be quiet\n"},
         solution=[_step(_call("read_file", path="quiet.txt")), _step(_call("write_file", path="loud.txt", content="PLEASE BE QUIET")),
                   _step(_say("Done."))]),
    Task("lookup_capital", "What is the capital of Zubrowka? Use the tool.", _says("lutz"), tools=[capital],
         solution=[_step(_call("capital", country="Zubrowka")), _step(_say("The capital of Zubrowka is Lutz."))]),
    Task("two_lookups", "Which has more people, the capital of Zubrowka or the capital of Freedonia? Use the tools.",
         _says("lutz"), tools=[capital, population],
         solution=[_step(_call("capital", country="Zubrowka")), _step(_call("capital", country="Freedonia")),
                   _step(_call("population", city="Lutz")), _step(_call("population", city="Fredville")),
                   _step(_say("Lutz, the capital of Zubrowka, has more people."))]),
    Task("chain_lookup", "Who is the manager of Ada's manager? Use the tool.", _says("cleo"), tools=[manager_of],
         solution=[_step(_call("manager_of", person="Ada")), _step(_call("manager_of", person="Ben")),
                   _step(_say("Cleo is the manager of Ada's manager."))]),
    Task("recover_from_error", "Is the build green? Check with the tool.", _says("green"), fresh=lambda: [flaky()],
         solution=[_step(_call("build_status")), _step(_call("build_status")), _step(_say("Yes, the build is green."))]),
    Task("fix_arguments", "How much is 20 euros in US dollars? Use the tool.", _says("22"), tools=[convert],
         solution=[_step(_call("convert", amount=20, currency="dollars")), _step(_call("convert", amount=20, currency="USD")),
                   _step(_say("20 euros is 22.00 USD."))]),
    Task("no_tool_needed", "What is 2 + 2? Answer directly, without using any tool.",
         lambda folder, answer, calls: "4" in (answer or "") and not calls,
         solution=[_step(_say("4"))]),
    Task("missing_file", "Read report.txt and tell me what it says.",
         lambda folder, answer, calls: any(word in (answer or "").lower() for word in
                                           ("not exist", "doesn't exist", "does not exist", "not found", "no such", "missing", "couldn't", "could not", "cannot")),
         solution=[_step(_call("read_file", path="report.txt")), _step(_say("report.txt does not exist."))]),
    Task("json_edit", "In data.json, change Ada's age to 37. Keep it valid JSON.", _json_age,
         files={"data.json": '{"name": "Ada", "age": 36}\n'},
         solution=[_step(_call("read_file", path="data.json")),
                   _step(_call("edit_file", path="data.json", old='"age": 36', new='"age": 37')), _step(_say("Done."))]),
    Task("append_line", "Add a line 'buy milk' at the end of todo.txt, keeping what is there.",
         _file("todo.txt", "call mum\nwater plants\nbuy milk"), files={"todo.txt": "call mum\nwater plants\n"},
         solution=[_step(_call("read_file", path="todo.txt")),
                   _step(_call("write_file", path="todo.txt", content="call mum\nwater plants\nbuy milk\n")), _step(_say("Added."))]),
    Task("find_in_files", "One of a.txt, b.txt and c.txt mentions a volcano. Which file? Answer with its name.", _says("b.txt"),
         files={"a.txt": "A walk by the sea.\n", "b.txt": "We climbed the volcano at dawn.\n", "c.txt": "A quiet library.\n"},
         solution=[_step(_call("read_file", path="a.txt")), _step(_call("read_file", path="b.txt")), _step(_say("b.txt"))]),
    Task("csv_average", "scores.csv has a 'score' column. Write the average score, as a whole number, to average.txt.",
         _file("average.txt", "8"), files={"scores.csv": "name,score\nada,7\nben,9\ncleo,8\n"},
         solution=[_step(_call("read_file", path="scores.csv")), _step(_call("write_file", path="average.txt", content="8")),
                   _step(_say("The average is 8."))]),
    Task("rename_variable", "In code.py, rename the variable x to count everywhere.",
         _file("code.py", "count = 1\ncount = count + 1\nprint(count)"), files={"code.py": "x = 1\nx = x + 1\nprint(x)\n"},
         solution=[_step(_call("read_file", path="code.py")),
                   _step(_call("write_file", path="code.py", content="count = 1\ncount = count + 1\nprint(count)\n")),
                   _step(_say("Renamed."))]),
    Task("two_files", "Create notes/a.md containing 'A' and notes/b.md containing 'B'.",
         lambda folder, answer, calls: _file("notes/a.md", "A")(folder, answer, calls) and _file("notes/b.md", "B")(folder, answer, calls),
         solution=[_step(_call("write_file", path="notes/a.md", content="A")), _step(_call("write_file", path="notes/b.md", content="B")),
                   _step(_say("Created both."))]),
]


@dataclass
class Result:
    task: str
    passed: bool
    steps: int
    tokens: int
    seconds: float
    error: str = ""


def run(model, tasks=None, step_limit=12):
    """Run the tasks on `model` (a Model or "provider:model"). Returns a list of Result."""
    from .model import resolve

    model = resolve(model)
    results = []
    for task in tasks or TASKS:
        tools = task.fresh() if task.fresh else task.tools
        steps, tokens, calls = [0], [0], []

        def hook(event, data):
            if event == "reply":
                steps[0] += 1
                tokens[0] = max(tokens[0], data.get("context") or 0)
                calls.extend(data.get("calls") or [])

        here = os.getcwd()
        with tempfile.TemporaryDirectory(prefix=f"nanoagent-bench-{task.name}-") as folder:
            folder = Path(folder)
            for name, content in task.files.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(content)
            started, answer, error = time.monotonic(), "", ""
            os.chdir(folder)
            try:
                agent = Agent("You are a careful assistant. Use the tools when they help; answer briefly.",
                              tools=tools, model=model, max_steps=step_limit)
                agent.hooks.append(hook)
                answer = agent.run(task.prompt)
                if agent.stop_reason == "steps":
                    error = f"stopped after {step_limit} steps"
            except Exception as failure:  # a task that crashes counts as failed
                error = f"{type(failure).__name__}: {failure}"
            finally:
                os.chdir(here)
            passed = not error and bool(task.check(folder, answer, calls))
            results.append(Result(task.name, passed, steps[0], tokens[0], round(time.monotonic() - started, 1), error))
    return results


def table(results, title=None):
    """The results as a Markdown table."""
    passed = sum(r.passed for r in results)
    lines = [f"### {title}", ""] if title else []
    lines += ["| task | passed | steps | tokens | seconds |", "|---|---|---|---|---|"]
    for r in results:
        mark = "yes" if r.passed else f"no ({r.error})" if r.error else "no"
        lines.append(f"| {r.task} | {mark} | {r.steps} | {r.tokens} | {r.seconds} |")
    lines += ["", f"Passed {passed} of {len(results)} ({round(100 * passed / max(1, len(results)))}%)."]
    return "\n".join(lines)


def solutions(tasks=None):
    """Every task's scripted solution, in order: a script for FakeModel."""
    return [step for task in tasks or TASKS for step in task.solution]
