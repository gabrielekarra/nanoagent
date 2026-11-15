"""Test an agent without calling a model.

`FakeModel` is a `Model` that gives the replies you scripted, in order, and
records what the agent sent it. It works the same whatever provider your
agent uses in production.

    from nanoagent import Agent
    from nanoagent.testing import FakeModel, call, reply, text

    def test_weather_agent():
        model = FakeModel(
            reply(call("get_weather", city="Rome")),    # the model asks for a tool
            reply(text("Sunny, no umbrella needed.")),  # then answers
        )
        agent = Agent(tools=[get_weather], model=model)
        assert agent.run("Umbrella in Rome?") == "Sunny, no umbrella needed."
        assert model.tool_results()[0]["content"] == "sunny"
"""

import asyncio
import copy
import itertools
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from .model import Call, Model, ModelError, Reply
from .runtime import Channel

_ids = itertools.count(1)


def text(value):
    """A text block in a scripted reply."""
    return {"type": "text", "text": value}


def call(name, _id=None, **arguments):
    """A tool call in a scripted reply."""
    return {"type": "tool_use", "id": _id or f"toolu_{next(_ids)}", "name": name, "input": arguments}


def reply(*blocks, stop=None, tokens=10):
    """One scripted model reply.

    stop     "end", "tool", "length" or "refusal"; by default "tool" if the
             reply holds a call, else "end"
    tokens   the context size the model reports after this reply
    """
    if stop is None:
        stop = "tool" if any(block["type"] == "tool_use" for block in blocks) else "end"
    return {"content": list(blocks), "stop": stop, "tokens": tokens}


class FakeModel(Model):
    """Serves scripted replies, in order, and records every request.

    script     the replies still to give; append to it at any time
    requests   what the agent sent each time: system, tools and messages
    """

    key = "fake"
    name = "fake"

    def __init__(self, *script, context_window=200_000):
        self.script = list(script)
        self.requests = []
        self.context_window = context_window

    def reply(self, system, tools, messages, on_text=None):
        self.requests.append({
            "system": system,
            "tools": [{"name": t.name, "description": t.description, "schema": t.schema} for t in tools],
            "messages": copy.deepcopy(messages),
        })
        if not self.script:
            raise ModelError("FakeModel: the script has no replies left")
        scripted = self.script.pop(0)
        texts = [block["text"] for block in scripted["content"] if block["type"] == "text"]
        for piece in texts:
            if on_text:
                on_text(piece)
        return Reply(
            text="".join(texts),
            calls=[Call(b["id"], b["name"], b["input"]) for b in scripted["content"] if b["type"] == "tool_use"],
            stop=scripted["stop"],
            context=scripted["tokens"],
        )

    def tool_results(self, request=-1):
        """The tool results the agent sent in a request (the last one by default)."""
        return self.requests[request]["messages"][-1]["results"]

    # Usable as a context manager, like the HTTP fakes in the test suite.
    def __enter__(self):
        return self

    def __exit__(self, *exception):
        pass


# --- the runtime ------------------------------------------------------------------


class FakeChannel(Channel):
    """A channel you drive from a test: `await channel.say(peer, text)`, then read `sent`."""

    def __init__(self, name="fake", default_peer=None):
        self.name = name
        self.default_peer = default_peer
        self.sent = []  # (peer, text)
        self.started = self.stopped = False

    async def start(self, deliver):
        self.started = True

    async def send(self, peer, text):
        self.sent.append((peer, text))

    async def stop(self):
        self.stopped = True

    async def say(self, peer, text, **extra):
        """A message arrives from `peer`."""
        await self._deliver(peer, text, **extra)

    async def wait_for(self, count, timeout=3):
        """Wait until at least `count` messages were sent; return them."""
        deadline = asyncio.get_running_loop().time() + timeout
        while len(self.sent) < count:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(f"{self.name}: expected {count} messages, got {self.sent}")
            await asyncio.sleep(0.005)
        return self.sent


class FakeClock:
    """A clock for `agent.clock` that only moves when told to."""

    def __init__(self, start=None):
        self.now = start or datetime(2026, 1, 5, 8, 0)  # a Monday

    def __call__(self):
        return self.now

    def advance(self, **delta):
        self.now += timedelta(**delta)
        return self.now


@asynccontextmanager
async def running(agent, **options):
    """Run `agent.alive()` for the duration of a block, then stop it.

        async with running(agent) as runtime:
            await channel.say("ada", "hello")
    """
    options.setdefault("tick", 3600)  # tests call runtime.tick_once() themselves
    task = asyncio.create_task(agent.alive(**options))
    while not (agent.runtime and agent.runtime.ready and agent.runtime.ready.is_set()):
        if task.done():
            task.result()
        await asyncio.sleep(0.005)
    runtime = agent.runtime
    try:
        yield runtime
    finally:
        runtime.stop()
        await asyncio.wait_for(task, 5)
