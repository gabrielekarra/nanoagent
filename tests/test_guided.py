"""Issue #3: guided tool calling for models with no tool support."""

import json
import os
from pathlib import Path

import pytest

from fakes import FakeOpenAI, failure
from nanoagent import Agent, read_file, resolve, tool, write_file
from nanoagent.providers import guided
from nanoagent.providers.openai import OpenAICompatible
from nanoagent.testing import reply, text


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return f"sunny in {city}"


@tool
def get_time(city: str, hour24: bool = True) -> str:
    """Return the local time in a city."""
    return f"noon in {city}"


def step(**fields):
    return reply(text(json.dumps(fields)))


@pytest.fixture
def server():
    with FakeOpenAI() as fake:
        yield fake


def guided_agent(server, **options):
    model = OpenAICompatible(server.url, "smollm2:1.7b", tools="guided")
    return Agent("You are a travel assistant.", tools=[get_weather, get_time], model=model, **options)


def test_guided_round_trip(server):
    server.script = [
        step(thought="I need the weather", tool="get_weather", arguments={"city": "Rome"}),
        step(thought="And the time", tool="get_time", arguments={"city": "Rome"}),
        step(thought="I have both", answer="Sunny, and it is noon."),
    ]
    shown = []

    assert guided_agent(server, on_text=shown.append).run("Rome?") == "Sunny, and it is noon."

    assert shown == ["Sunny, and it is noon."]
    third = server.bodies[2]["messages"]
    assert [m["role"] for m in third] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert json.loads(third[2]["content"])["tool"] == "get_weather"
    assert third[3]["content"] == "Result of get_weather:\nsunny in Rome"
    assert third[5]["content"] == "Result of get_time:\nnoon in Rome"


def test_guided_request_carries_schema(server):
    server.script = [step(thought="", answer="hi")]

    guided_agent(server).run("hello")

    body = server.bodies[0]
    assert "tools" not in body
    assert body["stream"] is False
    shape = body["response_format"]
    assert shape["type"] == "json_schema"
    branches = shape["json_schema"]["schema"]["anyOf"]
    assert [b["properties"].get("tool", {}).get("enum") for b in branches] == [["get_weather"], ["get_time"], None]
    assert branches[0]["properties"]["arguments"] == get_weather.schema
    assert branches[2]["required"] == ["thought", "answer"]
    system = body["messages"][0]["content"]
    assert system.startswith("You are a travel assistant.")
    assert "- get_weather(city: string) - Return today's weather in a city." in system
    assert "- get_time(city: string, hour24?: boolean)" in system


def test_guided_falls_back_when_schema_rejected(server):
    server.script = [
        failure(400, "response_format json_schema is not supported"),
        failure(400, "response_format is not supported"),
        step(thought="", tool="get_weather", arguments={"city": "Rome"}),
        step(thought="", answer="Sunny."),
    ]

    assert guided_agent(server).run("Rome?") == "Sunny."

    formats = [body.get("response_format", {}).get("type") for body in server.bodies]
    assert formats == ["json_schema", "json_object", None, None]


@pytest.mark.parametrize(
    "output",
    [
        '{"thought": "x", "tool": "get_weather", "arguments": {"city": "Rome"}}',
        '```json\n{"thought": "x", "tool": "get_weather", "arguments": {"city": "Rome"}}\n```',
        'Sure! Here is my step:\n{"tool": "get_weather", "arguments": {"city": "Rome"}}\nHope that helps.',
        '{"thought": "x", "tool": "get_weather", "arguments": {"city": "Rome",},}',
        "{'thought': 'x', 'tool': 'get_weather', 'arguments': {'city': 'Rome'}}",
        '<think>The user wants weather.</think>{"tool": "get_weather", "arguments": {"city": "Rome"}}',
        'Reasoning first.</think>\n{"tool": "get_weather", "arguments": {"city": "Rome"}}',
        '{"name": "get_weather", "parameters": {"city": "Rome"}}',
        '{"action": "get_weather", "action_input": "{\\"city\\": \\"Rome\\"}"}',
        '{"tool_call": {"name": "get_weather", "arguments": {"city": "Rome"}}}',
    ],
)
def test_guided_parses_messy_output(output):
    found = guided.parse(output)
    assert found["tool"] == "get_weather"
    assert found["arguments"] == {"city": "Rome"}


@pytest.mark.parametrize(
    "output, answer",
    [
        ('{"thought": "done", "answer": "Sunny."}', "Sunny."),
        ('{"action": "Final Answer", "action_input": "Sunny."}', "Sunny."),
        ("```\n{'answer': 'Sunny.'}\n```", "Sunny."),
    ],
)
def test_guided_parses_answers(output, answer):
    assert guided.parse(output)["answer"] == answer


def test_guided_rejects_what_is_not_a_step():
    assert guided.parse("It is sunny in Rome.") is None
    assert guided.parse('{"city": "Rome"}') is None
    assert guided.parse('{"name": "Ada"}', tools={"get_weather"}) is None


def test_guided_retries_once_then_answers(server):
    server.script = [reply(text("It is probably sunny.")), reply(text("I really think it is sunny."))]

    assert guided_agent(server).run("Rome?") == "I really think it is sunny."

    first, second = server.bodies
    assert second["messages"][-2] == {"role": "assistant", "content": "It is probably sunny."}
    assert second["messages"][-1]["content"] == guided.CORRECTION


def test_guided_correction_can_succeed(server):
    server.script = [reply(text("Let me think...")), step(thought="", answer="Sunny.")]

    assert guided_agent(server).run("Rome?") == "Sunny."
    assert len(server.bodies) == 2


def test_guided_unknown_tool_is_reported(server):
    server.script = [step(thought="", tool="book_flight", arguments={}), step(thought="", answer="I cannot book.")]

    assert guided_agent(server).run("Book me a flight") == "I cannot book."

    result = server.bodies[1]["messages"][-1]["content"]
    assert result.startswith("Error from book_flight:\nThere is no tool named `book_flight`")


def test_guided_bad_arguments_are_reported(server):
    server.script = [step(thought="", tool="get_weather", arguments={"town": "Rome"}), step(thought="", answer="ok")]

    guided_agent(server).run("Rome?")

    assert "missing `city`" in server.bodies[1]["messages"][-1]["content"]


def test_auto_switches_to_guided(server):
    server.script = [
        failure(400, "registry.ollama.ai/library/gemma2:2b does not support tools"),
        step(thought="", tool="get_weather", arguments={"city": "Rome"}),
        step(thought="", answer="Sunny."),
    ]
    model = OpenAICompatible(server.url, "gemma2:2b")

    assert Agent(tools=[get_weather], model=model).run("Rome?") == "Sunny."

    assert "tools" in server.bodies[0]
    assert all("response_format" in body and "tools" not in body for body in server.bodies[1:])
    assert model.tools == "guided"


def test_auto_without_tools_stays_native(server):
    server.script = [reply(text("hi"))]

    Agent(model=OpenAICompatible(server.url, "m")).run("hello")

    assert server.bodies[0]["stream"] is True and "response_format" not in server.bodies[0]


@pytest.mark.local
@pytest.mark.skipif(not os.environ.get("NANOAGENT_LOCAL_MODEL"), reason="set NANOAGENT_LOCAL_MODEL, e.g. ollama:qwen2.5:1.5b")
def test_real_small_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("prices.txt").write_text("apples 3\npears 4\nfigs 5\n")
    agent = Agent(
        "You are a careful assistant. Use the tools to work with files.",
        tools=[read_file, write_file],
        model=resolve(os.environ["NANOAGENT_LOCAL_MODEL"]),
    )

    agent.run("Read prices.txt, add up the three numbers in it, and write only the total to total.txt.")

    assert Path("total.txt").read_text().strip() == "12"
