"""Record and replay: see exactly what the agent saw, and turn a session into a test.

    agent.use(log("runs/"))                     # every request, reply, call and result
    nanoagent show runs/default-2026-11-15.jsonl

    model = FakeModel.from_log("runs/default-2026-11-15.jsonl")
    agent = Agent(tools=[...], model=model)     # replays the recorded replies; if the
    agent.run("the same request")               # agent's requests change, it says where

One JSON object per line, one file per conversation and day. Values of
environment variables whose names look like keys, tokens, secrets or
passwords are masked before anything is written.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

SECRET_NAMES = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)


def secrets():
    return sorted({value for name, value in os.environ.items() if SECRET_NAMES.search(name) and len(value) >= 6},
                  key=len, reverse=True)


class Log:
    def __init__(self, folder="runs"):
        self.folder = Path(folder).expanduser()
        self.agent = None

    def attach(self, agent):
        self.agent = agent
        self.folder.mkdir(parents=True, exist_ok=True)
        agent.hooks.append(self)

    def __call__(self, event, data):
        if event not in ("request", "reply", "tool"):
            return
        session = data.get("session", "default")
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event}
        if event == "request":  # the whole history is in earlier records: keep only what is new
            record.update(session=session, turns=len(data["messages"]), last=data["messages"][-1] if data["messages"] else None)
        else:
            record.update(data)
        line = json.dumps(record, ensure_ascii=False, default=str)
        for value in secrets():
            line = line.replace(value, "***")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session)[:60]
        with (self.folder / f"{safe}-{datetime.now():%Y-%m-%d}.jsonl").open("a") as file:
            file.write(line + "\n")


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def show(path):
    """A readable transcript of a log file."""
    lines = []
    for record in read(path):
        if record["event"] == "request":
            last = record.get("last") or {}
            if last.get("role") == "user":
                lines.append(f"you: {last['content']}")
        elif record["event"] == "reply":
            if record.get("text"):
                lines.append(f"agent: {record['text']}")
            for call in record.get("calls", []):
                lines.append(f"  -> {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})")
            if record.get("stop") in ("length", "refusal"):
                lines.append(f"  (stopped: {record['stop']})")
        elif record["event"] == "tool":
            result = record["result"]
            content = result["content"] if isinstance(result["content"], str) else "[text and images]"
            mark = "error" if result["error"] else "result"
            lines.append(f"  <- {mark}: {content[:200]}" + ("..." if len(content) > 200 else ""))
    return "\n".join(lines)


def log(folder="runs"):
    """Record every request, reply, tool call and result. See the module docstring."""
    return Log(folder)
