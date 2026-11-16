"""Issue #28: the tutorial chapters on any model, the runtime and a channel."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from fakes import FakeOpenAI, FakeTelegram, failure
from nanoagent.testing import call, reply, text

TUTORIAL = Path(__file__).parent.parent / "tutorial"


def load(filename, monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(filename[:-3], TUTORIAL / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- chapter 8: any model ------------------------------------------------------------


@pytest.fixture
def server():
    with FakeOpenAI() as fake:
        yield fake


def chapter8(server, monkeypatch):
    return load("08_any_model.py", monkeypatch, AGENT_MODEL_URL=server.url, AGENT_MODEL="tiny")


def test_chapter8_native_tools(server, monkeypatch, capsys):
    server.script = [reply(call("shell", "c1", command="echo hello")), reply(text("<think>done</think>It printed hello."))]
    chapter = chapter8(server, monkeypatch)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "say hello"}]

    assert chapter.run(messages) == "It printed hello."
    assert server.bodies[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "hello"}
    assert "think" not in capsys.readouterr().out


def test_chapter8_call_written_as_text(server, monkeypatch):
    server.script = [reply(text('{"name": "shell", "arguments": {"command": "echo hi"}}')), reply(text("ok"))]
    chapter = chapter8(server, monkeypatch)
    chapter.run([{"role": "system", "content": "s"}, {"role": "user", "content": "go"}])
    assert server.bodies[1]["messages"][-1]["content"] == "hi"


def test_chapter8_switches_to_guided(server, monkeypatch):
    server.script = [
        failure(400, "registry.ollama.ai/library/gemma2:2b does not support tools"),
        reply(text(json.dumps({"tool": "shell", "arguments": {"command": "echo guided"}}))),
        reply(text(json.dumps({"answer": "It printed guided."}))),
    ]
    chapter = chapter8(server, monkeypatch)

    assert chapter.run([{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]) == "It printed guided."
    guided = server.bodies[2]
    assert "tools" not in guided and guided["response_format"]["type"] == "json_schema"
    assert guided["messages"][-1] == {"role": "user", "content": "Result:\nguided"}


# --- chapter 9: always on -----------------------------------------------------------------


class Collect:
    name = "collect"

    def __init__(self, lines, event):
        self.lines, self.event, self.sent = lines, event, []

    async def start(self, runtime):
        for peer, line in self.lines:
            await runtime.queue.put(self.event(line, self, peer))

    async def send(self, peer, text):
        self.sent.append((peer, text))


async def until(condition, timeout=5):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def test_chapter9_runtime(monkeypatch):
    chapter = load("09_always_on.py", monkeypatch)

    async def main():
        channel = Collect([("ada", "hi"), ("bob", "hey"), ("ada", "again")], chapter.Event)
        runtime = chapter.Runtime([channel], [chapter.Every(0.05, "stretch")], notify=(channel, "ada"))
        task = asyncio.create_task(runtime.run())
        await until(lambda: len(channel.sent) >= 4)
        await runtime.queue.put(None)
        await task
        return channel.sent

    sent = asyncio.run(main())
    assert sent[:3] == [
        ("ada", "(message 1 of this conversation) You said: hi"),
        ("bob", "(message 1 of this conversation) You said: hey"),
        ("ada", "(message 3 of this conversation) You said: again"),
    ]
    assert sent[3] == ("ada", "(message 1 of this conversation) You said: stretch")


# --- chapter 10: a channel ------------------------------------------------------------------


def test_chapter10_telegram(monkeypatch):
    with FakeTelegram() as api:
        chapter = load("10_a_channel.py", monkeypatch, TELEGRAM_API=api.url)
        long_answer = "\n\n".join(["word " * 300] * 4)
        monkeypatch.setattr(chapter, "model", lambda messages: long_answer if messages[-1]["content"] == "long" else "ok")

        async def main():
            bot = chapter.Telegram(api.token, ["42"])
            runtime = chapter.Runtime([bot])
            task = asyncio.create_task(runtime.run())
            api.message(666, "stranger")
            api.message(42, "hi")
            api.message(42, "long")
            await until(lambda: len(api.sent()) >= 3)
            await asyncio.sleep(0.1)
            await runtime.queue.put(None)
            await task

        asyncio.run(main())
    sent = api.sent()
    assert {m["chat_id"] for m in sent} == {"42"}
    assert sent[0]["text"] == "ok"
    assert all(len(m["text"]) <= 4096 for m in sent) and len(sent) == 3
    assert " ".join(m["text"] for m in sent[1:]).split() == long_answer.split()
