from pathlib import Path

import pytest

from conftest import CHAPTERS
from nanoagent.testing import call, reply, text


def user(request):
    return [{"role": "user", "content": request}]


def chapters_from(number):
    return [name for name in CHAPTERS if int(name[:2]) >= number]


# --- every chapter -----------------------------------------------------------


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_loop_runs_a_tool_and_feeds_the_result_back(chapter, api, load, capsys):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [reply(call("shell", "toolu_1", command="echo hello")), reply(text("It printed hello."))]

    messages = user("say hello")
    agent.run(messages)

    assert len(api.bodies) == 2
    (result,) = api.tool_results()
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_1"
    assert result["content"] == "hello"
    assert "It printed hello." in capsys.readouterr().out
    # user, assistant (tool call), user (tool result), assistant (final answer)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_several_calls_are_answered_in_one_message(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [
        reply(call("shell", "toolu_a", command="echo a"), call("shell", "toolu_b", command="echo b")),
        reply(text("done")),
    ]

    agent.run(user("two things"))

    assert [(r["tool_use_id"], r["content"]) for r in api.tool_results()] == [("toolu_a", "a"), ("toolu_b", "b")]


# --- chapter 2: tools --------------------------------------------------------


@pytest.mark.parametrize("chapter", chapters_from(2))
def test_file_tools(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [
        reply(call("write_file", path="src/app.py", content="x = 1\n")),
        reply(call("edit_file", path="src/app.py", old="x = 1", new="x = 2")),
        reply(call("read_file", path="src/app.py")),
        reply(text("done")),
    ]

    agent.run(user("make a file"))

    assert Path("src/app.py").read_text() == "x = 2\n"
    assert api.tool_results()[0]["content"] == "x = 2\n"


@pytest.mark.parametrize("chapter", chapters_from(2))
def test_a_failing_tool_is_reported_to_the_model(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [reply(call("read_file", path="missing.txt")), reply(text("no such file"))]

    agent.run(user("read it"))

    (result,) = api.tool_results()
    assert result["is_error"] is True
    assert "FileNotFoundError" in result["content"]


@pytest.mark.parametrize("chapter", chapters_from(2))
def test_edit_refuses_an_ambiguous_match(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    Path("f.txt").write_text("a a")
    api.script = [reply(call("edit_file", path="f.txt", old="a", new="b")), reply(text("ok"))]

    agent.run(user("edit"))

    assert api.tool_results()[0]["is_error"] is True
    assert Path("f.txt").read_text() == "a a"


# --- chapter 3: permissions --------------------------------------------------


@pytest.mark.parametrize("chapter", chapters_from(3))
def test_declined_action_does_not_run(chapter, api, load, monkeypatch):
    agent = load(chapter)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    api.script = [reply(call("write_file", path="f.txt", content="x")), reply(text("ok"))]

    agent.run(user("write"))

    assert not Path("f.txt").exists()
    assert api.tool_results()[0]["is_error"] is True
    assert "declined" in api.tool_results()[0]["content"]


@pytest.mark.parametrize("chapter", chapters_from(3))
def test_always_is_remembered_and_reads_never_ask(chapter, api, load, monkeypatch):
    agent = load(chapter)
    answers = ["a"]  # a second question would raise IndexError
    monkeypatch.setattr("builtins.input", lambda prompt="": answers.pop(0))
    api.script = [
        reply(call("write_file", path="f.txt", content="1")),
        reply(call("write_file", path="f.txt", content="2")),
        reply(call("read_file", path="f.txt")),
        reply(text("ok")),
    ]

    agent.run(user("write twice"))

    assert Path("f.txt").read_text() == "2"
    assert answers == []


@pytest.mark.parametrize("chapter", chapters_from(3))
def test_interrupted_turn_is_closed(chapter, api, load, monkeypatch):
    agent = load(chapter)

    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    api.script = [reply(call("shell", command="echo hi")), reply(text("ok"))]
    messages = user("go")

    with pytest.raises(KeyboardInterrupt):
        agent.run(messages)
    agent.close_open_calls(messages)
    agent.ALWAYS_ALLOWED.add("shell")
    messages.append({"role": "user", "content": "never mind"})
    agent.run(messages)

    sent = api.bodies[-1]["messages"]
    assert sent[2]["content"][0]["content"] == "Interrupted by the user."
    assert sent[3]["content"] == "never mind"


# --- chapter 4: streaming ----------------------------------------------------


@pytest.mark.parametrize("chapter", chapters_from(4))
def test_requests_are_streamed(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [reply(text("hi"))]

    agent.run(user("hello"))

    assert api.bodies[0]["stream"] is True
    assert all(tool["eager_input_streaming"] for tool in api.bodies[0]["tools"])


@pytest.mark.parametrize("chapter", chapters_from(4))
def test_invalid_input_is_rejected_before_running(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [reply(call("write_file", path="f.txt")), reply(text("ok"))]  # no content

    agent.run(user("write"))

    assert not Path("f.txt").exists()
    assert api.tool_results()[0]["is_error"] is True


@pytest.mark.parametrize("chapter", chapters_from(4))
def test_cut_off_call_does_not_run(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [
        reply(call("write_file", path="f.txt", content="half of the fi"), stop="length"),
        reply(text("ok")),
    ]

    agent.run(user("write"))

    assert not Path("f.txt").exists()
    assert "output limit" in api.tool_results()[0]["content"]


@pytest.mark.parametrize("chapter", chapters_from(4))
def test_refusal_ends_the_turn_cleanly(chapter, api, load, capsys):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [reply(stop="refusal"), reply(text("sure"))]
    messages = user("something")

    agent.run(messages)
    messages.append({"role": "user", "content": "something else"})
    agent.run(messages)

    assert "declined" in capsys.readouterr().out
    assert [m["role"] for m in api.bodies[-1]["messages"]] == ["user", "user"]


# --- chapter 5: context ------------------------------------------------------


@pytest.mark.parametrize("chapter", chapters_from(5))
def test_long_output_is_clipped(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    Path("big.txt").write_text("A" * 30_000 + "Z" * 30_000)
    api.script = [reply(call("read_file", path="big.txt")), reply(text("ok"))]

    agent.run(user("read"))

    content = api.tool_results()[0]["content"]
    assert len(content) < 21_000
    assert content.startswith("A") and content.endswith("Z")
    assert "40000 characters omitted" in content


@pytest.mark.parametrize("chapter", chapters_from(5))
def test_requests_ask_for_caching(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [reply(text("hi"))]

    agent.run(user("hello"))

    assert api.bodies[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("chapter", chapters_from(5))
def test_large_context_is_compacted(chapter, api, load):
    agent = load(chapter, AGENT_YOLO="1", AGENT_COMPACT_AT="1000")
    api.script = [
        reply(call("shell", command="echo hi"), tokens=5000),  # over the limit
        reply(text("Goal: say hi. Done: ran echo.")),  # the handoff note
        reply(text("all done")),
    ]
    messages = user("say hi")

    agent.run(messages)

    summary_request, final_request = api.bodies[1], api.bodies[2]
    assert "handoff note" in summary_request["messages"][-1]["content"]
    assert len(final_request["messages"]) == 1
    assert "Goal: say hi. Done: ran echo." in final_request["messages"][0]["content"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


# --- chapter 6: memory -------------------------------------------------------


@pytest.mark.parametrize("chapter", chapters_from(6))
def test_memory_file_is_loaded_and_extended(chapter, api, load):
    Path("AGENT.md").write_text("- Tests run with `make test`.\n")
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [reply(call("remember", note="The user prefers tabs.")), reply(text("noted"))]

    agent.run(user("I prefer tabs"))

    assert "make test" in api.bodies[0]["system"]
    assert Path("AGENT.md").read_text() == "- Tests run with `make test`.\n- The user prefers tabs.\n"
    # the system prompt does not change inside a session
    assert api.bodies[1]["system"] == api.bodies[0]["system"]


# --- chapter 7: subagents ----------------------------------------------------


@pytest.mark.parametrize("chapter", chapters_from(7))
def test_subagent_runs_in_a_fresh_context(chapter, api, load, capsys):
    agent = load(chapter, AGENT_YOLO="1")
    api.script = [
        reply(call("task", "toolu_main", prompt="Count the files here.")),  # main agent
        reply(call("shell", "toolu_sub", command="echo 3")),  # subagent
        reply(text("There are 3 files.")),  # subagent's report
        reply(text("The folder has 3 files.")),  # main agent
    ]

    answer = agent.run(user("how many files?"))

    main_first, sub_first, sub_second, main_second = api.bodies
    assert sub_first["messages"] == [{"role": "user", "content": "Count the files here."}]
    assert "task" in [tool["name"] for tool in main_first["tools"]]
    assert "task" not in [tool["name"] for tool in sub_first["tools"]]
    # only the report reaches the main conversation
    assert main_second["messages"][-1]["content"][0]["content"] == "There are 3 files."
    assert len(main_second["messages"]) == 3
    assert answer == "The folder has 3 files."
    assert "There are 3 files." not in capsys.readouterr().out
