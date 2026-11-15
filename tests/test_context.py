"""Issue #5: agents that survive a 4k-8k context window."""

import json

import pytest

from fakes import FakeOpenAI
from nanoagent import Agent, tool
from nanoagent.model import estimate
from nanoagent.providers.openai import OpenAICompatible
from nanoagent.testing import FakeModel, call, reply, text


@tool
def dump(label: str) -> str:
    """Return a large log."""
    return f"{label}: " + "log line\n" * 6000  # about 54,000 characters


@tool
def small(label: str) -> str:
    """Return a little."""
    return f"{label} ok"


def size(request):
    tools = [[t["name"], t["description"], t["schema"]] for t in request["tools"]]
    return estimate(request["system"], tools, request["messages"])


def test_compaction_scales_with_window():
    model = FakeModel(context_window=8000)
    agent = Agent(model=model)
    assert agent.compact_at == 5600

    model.script = [reply(call("small", label="a"), tokens=5500), reply(text("done"), tokens=5500)]
    Agent(tools=[small], model=model).run("go")
    assert len(model.requests) == 2  # 5,500 is under the limit: no compaction

    model.script = [reply(call("small", label="a"), tokens=5700), reply(text("note")), reply(text("done"))]
    Agent(tools=[small], model=model).run("go")
    assert model.requests[3]["messages"][-1]["content"].startswith("Pause the task")


def test_large_windows_keep_the_old_defaults():
    agent = Agent(model=FakeModel(context_window=1_000_000))
    assert (agent.compact_at, agent.max_output) == (150_000, 20_000)


def test_clip_scales_with_window():
    model = FakeModel(reply(call("dump", label="x")), reply(text("ok")), context_window=8000)

    Agent(tools=[dump], model=model).run("go")

    content = model.tool_results()[0]["content"]
    assert len(content) < 8200  # about 2,000 tokens: a quarter of the window
    assert "characters omitted" in content


def test_estimate_when_no_usage():
    with FakeOpenAI() as server:
        server.script = [{**reply(call("dump", label=f"part {n}")), "usage": False} for n in range(3)]
        server.script += [{**reply(text("Handoff: the log was read.")), "usage": False}]
        server.script += [{**reply(text("done")), "usage": False}]
        model = OpenAICompatible(server.url, "m", context_window=1500)

        Agent(tools=[dump], model=model).run("read the log")

        assert [b["messages"][-1].get("content", "")[:14] for b in server.bodies][3] == "Pause the task"
        assert "Handoff: the log was read." in server.bodies[4]["messages"][1]["content"]


def test_ten_step_task_in_4k():
    script = [reply(call("dump", label=f"part {n}"), tokens=0) for n in range(10)]
    script += [reply(text("done"), tokens=0)]
    model = FakeModel(*script, context_window=4000)
    # Every request that is not a compaction gets the next scripted step;
    # compactions get a short note.
    original = model.reply

    def reply_or_note(system, tools, messages, on_text=None):
        if messages[-1].get("content", "").startswith("Pause the task"):
            model.script.insert(0, reply(text("Handoff: reading parts."), tokens=0))
        return original(system, tools, messages, on_text)

    model.reply = reply_or_note
    agent = Agent(tools=[dump], model=model)

    assert agent.run("read all ten parts") == "done"

    assert all(size(request) <= 4000 for request in model.requests), [size(r) for r in model.requests]
    steps = [r for r in model.requests if not r["messages"][-1].get("content", "").startswith("Pause the task")]
    assert len(steps) == 11


def test_instructions_survive_compaction():
    model = FakeModel(
        reply(call("small", label="a"), tokens=5000),
        reply(text("Handoff note.")),
        reply(text("done")),
        context_window=6000,
    )

    Agent("You are Jarvis.", tools=[small], model=model).run("go")

    after = model.requests[2]
    assert after["system"] == "You are Jarvis."
    assert [t["name"] for t in after["tools"]] == ["small"]
    assert len(after["messages"]) == 1 and "Handoff note." in after["messages"][0]["content"]


def test_summary_request_fits_even_when_history_does_not():
    history = []
    for n in range(8):
        history += [
            {"role": "user", "content": f"step {n}"},
            {"role": "assistant", "content": "", "calls": [{"id": str(n), "name": "small", "arguments": {}}], "raw": None, "by": "fake"},
            {"role": "tool", "results": [{"id": str(n), "name": "small", "content": "x" * 3000, "error": False}]},
        ]
    model = FakeModel(reply(text("note")), reply(text("done")), context_window=4000)
    agent = Agent(tools=[small], model=model)
    agent.messages = history
    old_turn = history[2]

    agent.run("continue")

    summary = model.requests[0]
    assert summary["messages"][-1]["content"].startswith("Pause the task")
    assert size(summary) <= 4000 * 0.8
    assert old_turn["results"][0]["content"] == "x" * 3000  # the real history was not edited
