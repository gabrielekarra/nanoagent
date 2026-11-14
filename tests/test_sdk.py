from pathlib import Path
from typing import Annotated

import anthropic
import pytest

from nanoagent import CODING_TOOLS, Agent, tool
from nanoagent.testing import FakeModel, call, reply, text


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return f"sunny in {city}"


@tool(approve=True)
def launch(code: Annotated[int, "The launch code"], force: bool = False) -> str:
    """Launch the rocket."""
    return f"launched with {code}"


@pytest.fixture
def model():
    with FakeModel() as fake:
        yield fake


@pytest.fixture(autouse=True)
def empty_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# --- @tool ---------------------------------------------------------------------


def test_tool_schema_comes_from_the_function():
    assert get_weather.spec["name"] == "get_weather"
    assert get_weather.spec["description"] == "Return today's weather in a city."
    assert launch.schema == {
        "type": "object",
        "properties": {
            "code": {"type": "integer", "description": "The launch code"},
            "force": {"type": "boolean"},
        },
        "required": ["code"],
    }
    assert launch.approve and not get_weather.approve


def test_tool_is_still_a_function():
    assert get_weather("Rome") == "sunny in Rome"


def test_tool_rejects_unsupported_types():
    with pytest.raises(TypeError, match="only str, int, float and bool"):

        @tool
        def bad(items: list) -> str:
            return ""


@pytest.mark.parametrize(
    "arguments, problem",
    [
        ({"code": 7}, None),
        ({"code": 7, "force": True}, None),
        ({}, "missing `code`"),
        ({"code": "7"}, "`code` must be a integer"),
        ({"code": True}, "`code` must be a integer"),
        ({"code": 7, "extra": 1}, "unknown `extra`"),
        ("not an object", "the input must be an object"),
    ],
)
def test_tool_checks_model_written_input(arguments, problem):
    assert launch.check(arguments) == problem


# --- Agent ---------------------------------------------------------------------


def test_agent_uses_a_tool_and_answers(model):
    model.script = [reply(call("get_weather", city="Rome")), reply(text("No umbrella needed."))]
    agent = Agent("You are a travel assistant.", [get_weather], client=model.client)

    answer = agent.run("Umbrella in Rome?")

    assert answer == "No umbrella needed."
    assert agent.stop_reason == "end_turn"
    assert model.tool_results()[0]["content"] == "sunny in Rome"
    first = model.requests[0]
    assert first["system"] == "You are a travel assistant."
    assert first["model"] == "claude-opus-5-5"
    assert first["stream"] is True
    assert first["cache_control"] == {"type": "ephemeral"}
    assert [each["name"] for each in first["tools"]] == ["get_weather"]


def test_agent_keeps_the_conversation_between_runs(model):
    model.script = [reply(text("Hi Ada.")), reply(text("Ada."))]
    agent = Agent(client=model.client)

    agent.run("I am Ada.")
    agent.run("Who am I?")

    assert [m["role"] for m in model.requests[1]["messages"]] == ["user", "assistant", "user"]


def test_callbacks_see_text_and_tool_calls(model):
    model.script = [reply(text("Checking. "), call("get_weather", city="Oslo")), reply(text("Cold."))]
    written, used = [], []
    agent = Agent(tools=[get_weather], client=model.client, on_text=written.append, on_tool=lambda *a: used.append(a))

    agent.run("Oslo?")

    assert "".join(written) == "Checking. Cold."
    assert used == [("get_weather", {"city": "Oslo"})]


@pytest.mark.parametrize(
    "bad_call, message",
    [
        (call("nope"), "no tool named `nope`"),
        (call("get_weather"), "missing `city`"),
        (call("get_weather", city=3), "`city` must be a string"),
    ],
)
def test_bad_calls_are_reported_not_run(model, bad_call, message):
    model.script = [reply(bad_call), reply(text("sorry"))]

    Agent(tools=[get_weather], client=model.client).run("go")

    (result,) = model.tool_results()
    assert result["is_error"] is True and message in result["content"]


def test_tool_exception_goes_back_to_the_model(model):
    @tool
    def explode() -> str:
        """Fail."""
        raise RuntimeError("boom")

    model.script = [reply(call("explode")), reply(text("it failed"))]

    Agent(tools=[explode], client=model.client).run("go")

    assert model.tool_results()[0] == {
        "type": "tool_result",
        "tool_use_id": model.tool_results()[0]["tool_use_id"],
        "content": "RuntimeError: boom",
        "is_error": True,
    }


def test_approval_callback_can_decline(model):
    model.script = [reply(call("launch", code=1)), reply(call("get_weather", city="Rome")), reply(text("ok"))]
    asked = []

    def approve(name, arguments):
        asked.append((name, arguments))
        return False

    Agent(tools=[launch, get_weather], client=model.client, approve=approve).run("go")

    assert asked == [("launch", {"code": 1})]  # get_weather is not marked, so nobody is asked
    assert "declined" in model.tool_results(1)[0]["content"]
    assert model.tool_results(2)[0]["content"] == "sunny in Rome"


def test_without_an_approval_callback_marked_tools_run(model):
    model.script = [reply(call("launch", code=1)), reply(text("ok"))]

    Agent(tools=[launch], client=model.client).run("go")

    assert model.tool_results()[0]["content"] == "launched with 1"


def test_cut_off_reply_does_not_run_its_calls(model):
    model.script = [reply(call("launch", code=1), stop_reason="max_tokens"), reply(text("ok"))]

    Agent(tools=[launch], client=model.client).run("go")

    assert "output limit" in model.tool_results()[0]["content"]


def test_refusal_is_reported_through_stop_reason(model):
    model.script = [reply(stop_reason="refusal"), reply(text("sure"))]
    agent = Agent(client=model.client)

    assert agent.run("something") == ""
    assert agent.stop_reason == "refusal"
    assert agent.run("something else") == "sure"


def test_long_tool_output_is_clipped(model):
    @tool
    def dump() -> str:
        """Return a lot."""
        return "A" * 500 + "Z" * 500

    model.script = [reply(call("dump")), reply(text("ok"))]

    Agent(tools=[dump], client=model.client, max_output=100).run("go")

    content = model.tool_results()[0]["content"]
    assert content == "A" * 50 + "\n[... 900 characters omitted ...]\n" + "Z" * 50


def test_large_context_is_compacted(model):
    model.script = [
        reply(call("get_weather", city="Rome"), input_tokens=5000),
        reply(text("Goal: weather. Done: Rome is sunny.")),  # the handoff note
        reply(text("Sunny.")),
    ]
    agent = Agent(tools=[get_weather], client=model.client, compact_at=1000)

    assert agent.run("Weather in Rome?") == "Sunny."

    assert len(model.requests[2]["messages"]) == 1
    assert "Goal: weather. Done: Rome is sunny." in model.requests[2]["messages"][0]["content"]
    assert len(agent.messages) == 2


def test_interrupted_run_leaves_a_valid_conversation(model):
    @tool
    def wait() -> str:
        """Wait."""
        raise KeyboardInterrupt

    model.script = [reply(call("wait")), reply(text("ok"))]
    agent = Agent(tools=[wait], client=model.client)

    with pytest.raises(KeyboardInterrupt):
        agent.run("go")
    assert agent.run("never mind") == "ok"

    sent = model.requests[-1]["messages"]
    assert sent[2]["content"][0]["content"] == "Interrupted by the user."


def test_memory_is_loaded_and_extended(model):
    Path("NOTES.md").write_text("- Tests run with `make test`.\n")
    model.script = [reply(call("remember", note="The user prefers tabs.")), reply(text("noted"))]
    agent = Agent("Be useful.", client=model.client, memory="NOTES.md")

    agent.run("I prefer tabs")

    assert "make test" in model.requests[0]["system"]
    assert Path("NOTES.md").read_text() == "- Tests run with `make test`.\n- The user prefers tabs.\n"
    assert model.requests[1]["system"] == model.requests[0]["system"]  # fixed for the session


def test_subagent_gets_a_fresh_context_and_returns_only_its_report(model):
    model.script = [
        reply(call("task", prompt="Check Rome.")),  # main agent delegates
        reply(call("get_weather", city="Rome")),  # subagent works
        reply(text("Rome is sunny.")),  # subagent reports
        reply(text("You are fine.")),  # main agent answers
    ]
    written = []
    agent = Agent(tools=[get_weather], client=model.client, subagents=True, on_text=written.append)

    assert agent.run("Umbrella?") == "You are fine."

    main_first, sub_first, _, main_second = model.requests
    assert sub_first["messages"] == [{"role": "user", "content": "Check Rome."}]
    assert [each["name"] for each in main_first["tools"]] == ["get_weather", "task"]
    assert [each["name"] for each in sub_first["tools"]] == ["get_weather"]
    assert main_second["messages"][-1]["content"][0]["content"] == "Rome is sunny."
    assert "".join(written) == "You are fine."


def test_coding_tools_work_end_to_end(model):
    model.script = [
        reply(call("write_file", path="src/app.py", content="x = 1\n")),
        reply(call("edit_file", path="src/app.py", old="x = 1", new="x = 2")),
        reply(call("shell", command="cat src/app.py")),
        reply(text("done")),
    ]

    Agent(tools=CODING_TOOLS, client=model.client).run("make a file")

    assert Path("src/app.py").read_text() == "x = 2\n"
    assert model.tool_results()[0]["content"] == "x = 2"


# --- FakeModel -----------------------------------------------------------------


def test_fake_model_fails_loudly_when_the_script_runs_out(model):
    with pytest.raises(anthropic.BadRequestError, match="no replies left"):
        Agent(client=model.client).run("hello")


# --- the command line ----------------------------------------------------------


def test_command_line_agent(model, monkeypatch, capsys):
    from nanoagent import cli

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", model.url)
    monkeypatch.setattr(cli, "always_allowed", set())
    answers = iter(["create hello.txt", "y"])

    def typed(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", typed)
    model.script = [reply(call("write_file", path="hello.txt", content="hi")), reply(text("Created hello.txt."))]

    cli.main()

    assert Path("hello.txt").read_text() == "hi"
    out = capsys.readouterr().out
    assert "[write_file]" in out and "Created hello.txt." in out
