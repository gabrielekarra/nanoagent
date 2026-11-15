"""Issue #4: Ollama's native API."""

import json

import pytest

from fakes import FakeOllama, failure
from nanoagent import Agent, ModelError, tool
from nanoagent.providers.ollama import Ollama
from nanoagent.testing import call, reply, text


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return f"sunny in {city}"


@pytest.fixture
def server():
    with FakeOllama() as fake:
        yield fake


def test_ollama_round_trip(server):
    server.script = [reply(call("get_weather", city="Rome")), reply(text("Sunny."))]

    assert Agent(tools=[get_weather], model=Ollama("qwen3:4b", host=server.url)).run("Rome?") == "Sunny."

    assert [r["path"] for r in server.requests] == ["/api/chat", "/api/chat"]
    first, second = server.bodies
    assert first["model"] == "qwen3:4b"
    assert first["tools"][0]["function"]["name"] == "get_weather"
    assert second["messages"][-2] == {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}
    ]}
    assert second["messages"][-1] == {"role": "tool", "tool_name": "get_weather", "content": "sunny in Rome"}


def test_ollama_streams_text(server):
    server.script = [reply(text("Hel"), text("lo!"))]
    seen = []

    Agent(model=Ollama("m", host=server.url), on_text=seen.append).run("hi")

    assert seen == ["Hel", "lo!"]


def test_ollama_sends_num_ctx(server):
    server.script = [reply(call("get_weather", city="Rome")), reply(text("ok"))]

    Agent(tools=[get_weather], model=Ollama("m", host=server.url, context=16_384, temperature=0.2)).run("go")

    assert all(body["options"] == {"num_ctx": 16_384, "temperature": 0.2} for body in server.bodies)


def test_ollama_guided_uses_format(server):
    server.script = [reply(text(json.dumps({"thought": "", "answer": "hi"})))]

    Agent(tools=[get_weather], model=Ollama("m", host=server.url, tools="guided")).run("hello")

    body = server.bodies[0]
    assert body["stream"] is False
    assert body["format"]["anyOf"][0]["properties"]["tool"]["enum"] == ["get_weather"]
    assert body["options"]["num_ctx"] == 8192


def test_ollama_switches_to_guided(server):
    server.script = [
        failure(400, "registry.ollama.ai/library/gemma2:2b does not support tools", shape="ollama"),
        reply(text(json.dumps({"thought": "", "tool": "get_weather", "arguments": {"city": "Rome"}}))),
        reply(text(json.dumps({"thought": "", "answer": "Sunny."}))),
    ]

    assert Agent(tools=[get_weather], model=Ollama("gemma2:2b", host=server.url)).run("Rome?") == "Sunny."
    assert "format" in server.bodies[1] and "format" in server.bodies[2]


def test_ollama_not_running_error():
    with pytest.raises(ModelError, match="ollama serve"):
        Agent(model=Ollama("m", host="http://127.0.0.1:9")).run("hi")


def test_ollama_missing_model_error(server):
    server.script = [failure(404, 'model "qwen3:4b" not found, try pulling it first', shape="ollama")]

    with pytest.raises(ModelError, match="ollama pull qwen3:4b"):
        Agent(model=Ollama("qwen3:4b", host=server.url)).run("hi")
    assert server.pulled == []


def test_ollama_pulls_when_asked(server, capsys):
    server.script = [failure(404, 'model "qwen3:4b" not found, try pulling it first', shape="ollama"), reply(text("hi"))]

    assert Agent(model=Ollama("qwen3:4b", host=server.url, pull=True)).run("hi") == "hi"
    assert server.pulled == ["qwen3:4b"]


def test_ollama_reports_tokens(server):
    server.script = [reply(text("hi"), tokens=1234)]
    model = Ollama("m", host=server.url)

    found = model.reply("", [], [{"role": "user", "content": "hello"}])

    assert found.context == 1234
    assert model.context_window == 8192


def test_ollama_length_stop(server):
    server.script = [reply(call("get_weather", city="Rome"), stop="length"), reply(text("ok"))]

    Agent(tools=[get_weather], model=Ollama("m", host=server.url)).run("go")

    assert "output limit" in server.bodies[1]["messages"][-1]["content"]
