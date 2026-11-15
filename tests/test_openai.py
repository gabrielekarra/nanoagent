"""Issue #2: any OpenAI-compatible server."""

import json

import pytest

from fakes import FakeOpenAI, failure
from nanoagent import Agent, ModelError, tool
from nanoagent.providers.openai import OpenAICompatible, ThinkFilter
from nanoagent.testing import call, reply, text


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return f"sunny in {city}"


@tool
def get_time(city: str) -> str:
    """Return the local time in a city."""
    return f"noon in {city}"


@pytest.fixture
def server():
    with FakeOpenAI() as fake:
        yield fake


def agent_on(server, tools=(get_weather, get_time), **options):
    return Agent("Be brief.", tools=list(tools), model=OpenAICompatible(server.url, "local-model", **options))


def test_openai_tool_call_round_trip(server):
    server.script = [reply(call("get_weather", "call_7", city="Rome")), reply(text("Sunny in Rome."))]

    assert agent_on(server).run("Rome?") == "Sunny in Rome."

    first, second = server.bodies
    assert first["model"] == "local-model"
    assert first["messages"][0] == {"role": "system", "content": "Be brief."}
    assert first["tools"][0] == {"type": "function", "function": {
        "name": "get_weather", "description": "Return today's weather in a city.", "parameters": get_weather.schema,
    }}
    assert second["messages"][2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_7", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}
    ]}
    assert second["messages"][3] == {"role": "tool", "tool_call_id": "call_7", "content": "sunny in Rome"}


def test_openai_streams_text(server):
    server.script = [{**reply(text("The weather is lovely today.")), "chunks": 5}]
    seen = []

    answer = Agent(model=OpenAICompatible(server.url, "m"), on_text=seen.append).run("hi")

    assert len(seen) > 3
    assert "".join(seen) == answer == "The weather is lovely today."
    assert server.bodies[0]["stream"] is True
    assert server.bodies[0]["stream_options"] == {"include_usage": True}


def test_openai_several_calls_in_one_reply(server):
    server.script = [
        reply(call("get_weather", "a", city="Rome"), call("get_time", "b", city="Oslo")),
        reply(text("done")),
    ]

    agent_on(server).run("both")

    tool_messages = server.bodies[1]["messages"][3:]
    assert [(m["tool_call_id"], m["content"]) for m in tool_messages] == [("a", "sunny in Rome"), ("b", "noon in Oslo")]


def test_openai_arguments_split_across_chunks(server):
    server.script = [{**reply(call("get_weather", city="Reykjavik")), "split": 3}, reply(text("ok"))]

    agent_on(server).run("go")

    assert server.bodies[1]["messages"][-1]["content"] == "sunny in Reykjavik"


def test_openai_arguments_as_object(server):
    server.script = [{**reply(call("get_weather", city="Lima")), "object_args": True}, reply(text("ok"))]

    agent_on(server).run("go")

    assert server.bodies[1]["messages"][-1]["content"] == "sunny in Lima"


def test_openai_stop_with_tool_calls_still_runs_them(server):
    # Several servers report "stop" instead of "tool_calls".
    server.script = [{**reply(call("get_weather", city="Rome")), "finish": "stop"}, reply(text("ok"))]

    agent_on(server).run("go")

    assert server.bodies[1]["messages"][-1]["content"] == "sunny in Rome"


def test_openai_length_stop_does_not_run_calls(server):
    ran = []

    @tool
    def record(note: str) -> str:
        """Record a note."""
        ran.append(note)
        return "ok"

    server.script = [reply(call("record", note="half"), stop="length"), reply(text("ok"))]

    agent_on(server, tools=[record]).run("go")

    assert ran == []
    assert "output limit" in server.bodies[1]["messages"][-1]["content"]


def test_openai_http_error_is_readable(server):
    server.script = [failure(404, "model 'nope' not found")]

    with pytest.raises(ModelError, match="HTTP 404: model 'nope' not found") as caught:
        agent_on(server).run("go")
    assert caught.value.status == 404


def test_openai_unreachable_server_is_explained():
    with pytest.raises(ModelError, match="Is the server running"):
        Agent(model=OpenAICompatible("http://127.0.0.1:9", "m")).run("go")


def test_openai_sends_key_only_when_set(server):
    server.script = [reply(text("a")), reply(text("b"))]

    Agent(model=OpenAICompatible(server.url, "m", api_key="sk-123")).run("go")
    Agent(model=OpenAICompatible(server.url, "m")).run("go")

    assert server.requests[0]["headers"]["Authorization"] == "Bearer sk-123"
    assert "Authorization" not in server.requests[1]["headers"]


def test_reasoning_text_is_not_shown(server):
    server.script = [{**reply(text("<think>The user wants a greeting.</think>\n\nHello!")), "chunks": 4}]
    seen = []

    answer = Agent(model=OpenAICompatible(server.url, "m"), on_text=seen.append).run("hi")

    assert answer == "Hello!"
    assert "think" not in "".join(seen) and "greeting" not in "".join(seen)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 100])
def test_think_filter_handles_any_split(size):
    stream = "Hi <think>secret plan</think>there<think>more</think>!"
    pieces = [stream[i : i + size] for i in range(0, len(stream), size)]
    filter = ThinkFilter()
    assert "".join(filter.feed(p) for p in pieces) + filter.flush() == "Hi there!"


def test_openai_server_without_stream_options(server):
    server.script = [failure(400, "Unrecognized field: stream_options"), reply(text("hi")), reply(text("again"))]
    model = OpenAICompatible(server.url, "m")

    assert Agent(model=model).run("go") == "hi"
    Agent(model=model).run("go")

    assert "stream_options" in server.bodies[0]
    assert "stream_options" not in server.bodies[1] and "stream_options" not in server.bodies[2]


def test_openai_context_estimate_without_usage(server):
    server.script = [{**reply(text("x" * 400)), "usage": False}]
    model = OpenAICompatible(server.url, "m")

    found = model.reply("", [], [{"role": "user", "content": "y" * 4000}])

    assert 1000 < found.context < 1300


def test_openai_extra_options_are_sent(server):
    server.script = [reply(text("hi"))]

    Agent(model=OpenAICompatible(server.url, "m", temperature=0.1, max_tokens=500)).run("go")

    assert server.bodies[0]["temperature"] == 0.1
    assert server.bodies[0]["max_tokens"] == 500


def test_openai_consecutive_user_turns_are_merged(server):
    server.script = [reply(stop="refusal"), reply(text("ok"))]
    agent = Agent(model=OpenAICompatible(server.url, "m"))

    agent.run("one")
    agent.run("two")

    assert server.bodies[1]["messages"][1:] == [{"role": "user", "content": "one\n\ntwo"}]


def test_json_written_as_text_becomes_a_call(server):
    # A common small-model failure: the call is written as text instead of
    # using the tool-calling format.
    server.script = [
        reply(text('```json\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n```')),
        reply(text("Sunny.")),
    ]

    assert agent_on(server).run("Rome?") == "Sunny."

    assert server.bodies[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": server.bodies[1]["messages"][-2]["tool_calls"][0]["id"], "content": "sunny in Rome",
    }
    assert json.loads(server.bodies[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"]) == {"city": "Rome"}


def test_json_text_naming_no_tool_stays_an_answer(server):
    server.script = [reply(text('{"name": "Ada", "age": 36}'))]

    assert agent_on(server).run("Give me a JSON person") == '{"name": "Ada", "age": 36}'
