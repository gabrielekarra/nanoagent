"""Issue #8 (async core) and issue #9 (parallel tool execution)."""

import asyncio
import threading
import time

import pytest

from nanoagent import Agent, Call, Model, Reply, tool
from nanoagent.testing import FakeModel, call, reply, text


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return f"sunny in {city}"


def script():
    return [reply(call("get_weather", "c1", city="Rome")), reply(text("Sunny."))]


# --- #8 async core -------------------------------------------------------------


def test_arun_matches_run():
    sync_model, async_model = FakeModel(*script()), FakeModel(*script())
    sync_agent = Agent(tools=[get_weather], model=sync_model)
    async_agent = Agent(tools=[get_weather], model=async_model)

    assert sync_agent.run("Rome?") == asyncio.run(async_agent.arun("Rome?")) == "Sunny."
    assert sync_agent.messages == async_agent.messages
    assert sync_model.requests == async_model.requests


def test_async_tool_is_awaited():
    @tool
    async def fetch(url: str) -> str:
        """Fetch a page."""
        await asyncio.sleep(0.01)
        return f"contents of {url}"

    model = FakeModel(reply(call("fetch", url="https://example.com")), reply(text("ok")))

    Agent(tools=[fetch], model=model).run("go")

    assert model.tool_results()[0]["content"] == "contents of https://example.com"


def test_sync_tool_does_not_block_the_loop():
    @tool
    def slow() -> str:
        """Take a while."""
        time.sleep(0.5)
        return "done"

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        background = asyncio.create_task(ticker())
        await Agent(tools=[slow], model=FakeModel(reply(call("slow")), reply(text("ok")))).arun("go")
        background.cancel()
        return ticks

    assert asyncio.run(main()) >= 5


def test_run_inside_a_running_loop_explains():
    async def main():
        Agent(model=FakeModel(reply(text("hi")))).run("hello")

    with pytest.raises(RuntimeError, match="await agent.arun"):
        asyncio.run(main())


def test_two_agents_concurrently():
    @tool
    async def pause(label: str) -> str:
        """Pause a moment."""
        await asyncio.sleep(0.05)
        return label

    async def main():
        first = Agent(tools=[pause], model=FakeModel(reply(call("pause", label="a")), reply(text("A"))))
        second = Agent(tools=[pause], model=FakeModel(reply(call("pause", label="b")), reply(text("B"))))
        answers = await asyncio.gather(first.arun("one"), second.arun("two"))
        return answers, first, second

    answers, first, second = asyncio.run(main())
    assert answers == ["A", "B"]
    assert first.messages[0]["content"] == "one" and second.messages[0]["content"] == "two"


def test_async_only_model():
    class AsyncModel(Model):
        async def areply(self, system, tools, messages, on_text=None):
            await asyncio.sleep(0)
            if messages[-1]["role"] == "user":
                return Reply(calls=[Call("1", "get_weather", {"city": "Oslo"})], stop="tool")
            return Reply(text="done")

    assert Agent(tools=[get_weather], model=AsyncModel()).run("Oslo?") == "done"
    assert AsyncModel().reply("", [], [{"role": "user", "content": "x"}]).calls[0].name == "get_weather"


def test_model_without_reply_explains():
    with pytest.raises(NotImplementedError, match="reply\\(\\) or areply\\(\\)"):
        Model().reply("", [], [])


def test_async_approval_callback():
    @tool(approve=True)
    def launch() -> str:
        """Launch."""
        return "launched"

    async def approve(name, arguments):
        await asyncio.sleep(0)
        return False

    model = FakeModel(reply(call("launch")), reply(text("ok")))

    Agent(tools=[launch], model=model, approve=approve).run("go")

    assert "declined" in model.tool_results()[0]["content"]


# --- #9 parallel tool execution ----------------------------------------------------


def sleeper(name, seconds, log=None, parallel=True):
    def body(label: str) -> str:
        if log is not None:
            log.append(("start", name, time.monotonic()))
        time.sleep(seconds)
        if log is not None:
            log.append(("end", name, time.monotonic()))
        return f"{name} {label}"

    body.__name__ = name
    body.__doc__ = f"Sleep {seconds}s."
    return tool(body, parallel=parallel)


def test_calls_run_concurrently():
    tools = [sleeper(f"t{n}", 0.3) for n in range(3)]
    model = FakeModel(reply(*(call(f"t{n}", label="x") for n in range(3))), reply(text("ok")))

    started = time.monotonic()
    Agent(tools=tools, model=model).run("go")

    assert time.monotonic() - started < 0.6


def test_results_keep_call_order():
    tools = [sleeper("slow", 0.3), sleeper("fast", 0.0)]
    model = FakeModel(reply(call("slow", label="1"), call("fast", label="2")), reply(text("ok")))

    Agent(tools=tools, model=model).run("go")

    assert [r["content"] for r in model.tool_results()] == ["slow 1", "fast 2"]


def test_parallel_limit():
    running, peak, lock = 0, 0, threading.Lock()

    @tool
    def work(label: str) -> str:
        """Work."""
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.1)
        with lock:
            running -= 1
        return label

    model = FakeModel(reply(*(call("work", label=str(n)) for n in range(6))), reply(text("ok")))

    Agent(tools=[work], model=model, parallel=2).run("go")

    assert peak == 2


def test_parallel_one_runs_in_series():
    log = []
    tools = [sleeper("a", 0.05, log), sleeper("b", 0.05, log)]
    model = FakeModel(reply(call("a", label="x"), call("b", label="y")), reply(text("ok")))

    Agent(tools=tools, model=model, parallel=1).run("go")

    assert [entry[:2] for entry in log] == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")]


def test_approvals_are_asked_before_anything_runs():
    events = []

    @tool(approve=True)
    def deploy(target: str) -> str:
        """Deploy."""
        events.append(("ran", target))
        return "deployed"

    @tool
    def status() -> str:
        """Status."""
        events.append(("ran", "status"))
        return "green"

    def approve(name, arguments):
        events.append(("asked", arguments["target"]))
        return arguments["target"] == "staging"

    model = FakeModel(
        reply(call("deploy", target="staging"), call("status"), call("deploy", target="production")),
        reply(text("ok")),
    )

    Agent(tools=[deploy, status], model=model, approve=approve).run("go")

    assert events[:2] == [("asked", "staging"), ("asked", "production")]
    assert sorted(events[2:]) == [("ran", "staging"), ("ran", "status")]
    results = model.tool_results()
    assert [r["error"] for r in results] == [False, False, True]


def test_serial_tool_runs_alone():
    log = []
    tools = [sleeper("p1", 0.1, log), sleeper("p2", 0.1, log), sleeper("solo", 0.05, log, parallel=False)]
    model = FakeModel(reply(call("p1", label="x"), call("solo", label="y"), call("p2", label="z")), reply(text("ok")))

    Agent(tools=tools, model=model).run("go")

    solo_start = next(t for kind, name, t in log if kind == "start" and name == "solo")
    solo_end = next(t for kind, name, t in log if kind == "end" and name == "solo")
    for kind, name, moment in log:
        if name != "solo":
            assert not (solo_start < moment < solo_end)
    assert [r["content"] for r in model.tool_results()] == ["p1 x", "solo y", "p2 z"]


def test_one_failure_does_not_cancel_the_others():
    @tool
    def broken() -> str:
        """Fail."""
        raise RuntimeError("boom")

    tools = [broken, sleeper("fine", 0.1)]
    model = FakeModel(reply(call("broken"), call("fine", label="x")), reply(text("ok")))

    Agent(tools=tools, model=model).run("go")

    assert [(r["error"], r["content"]) for r in model.tool_results()] == [(True, "RuntimeError: boom"), (False, "fine x")]
