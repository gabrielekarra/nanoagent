"""Issue #6: send a small model only the tools it needs."""

from nanoagent import Agent, Tool
from nanoagent.testing import FakeModel, call, reply, text

TOPICS = ["invoice", "calendar", "playlist", "garden", "parcel", "recipe", "taxes", "flight", "printer", "aquarium"]


def make_tool(name, description):
    return Tool(name, description, {"type": "object", "properties": {}, "required": []}, lambda: f"{name} done")


def many_tools():
    tools = [make_tool("get_weather", "Return today's weather forecast for a city.")]
    for topic in TOPICS:
        for verb in ("create", "delete", "list", "update"):
            if len(tools) < 40:
                tools.append(make_tool(f"{verb}_{topic}", f"{verb.title()} a {topic} entry."))
    return tools


def offered(model, request=0):
    return [each["name"] for each in model.requests[request]["tools"]]


def test_shortlist_keeps_request_small():
    model = FakeModel(reply(text("ok")), context_window=8000)
    Agent(tools=many_tools(), model=model).run("hello")
    assert len(offered(model)) == 9 and offered(model)[-1] == "find_tools"


def test_relevant_tool_is_shortlisted():
    model = FakeModel(reply(text("ok")), context_window=8000)
    Agent(tools=many_tools(), model=model).run("What's the weather in Rome?")
    assert "get_weather" in offered(model)


def test_used_tools_stay_available():
    script = [reply(call("list_aquarium"))] + [reply(call("get_weather")) for _ in range(3)] + [reply(text("ok"))]
    model = FakeModel(*script, context_window=8000)
    Agent(tools=many_tools(), model=model).run("Look at the aquarium, then the weather forecast a few times")
    assert all("list_aquarium" in offered(model, step) for step in range(1, 5))


def test_find_tools_expands_the_list():
    model = FakeModel(reply(call("find_tools", query="calendar")), reply(text("ok")), context_window=8000)
    Agent(tools=many_tools(), model=model).run("hello there")
    assert "calendar" in model.tool_results(1)[0]["content"]
    assert {"create_calendar", "delete_calendar", "list_calendar", "update_calendar"} <= set(offered(model, 1))


def test_call_outside_shortlist_still_runs():
    model = FakeModel(reply(call("update_printer")), reply(text("ok")), context_window=8000)
    Agent(tools=many_tools(), model=model).run("hello")
    assert "update_printer" not in offered(model, 0)
    assert model.tool_results(1)[0]["content"] == "update_printer done"


def test_off_for_large_windows_and_when_asked():
    big = FakeModel(reply(text("ok")), context_window=200_000)
    Agent(tools=many_tools(), model=big).run("hello")
    assert len(offered(big)) == 40
    small = FakeModel(reply(text("ok")), context_window=8000)
    Agent(tools=many_tools(), model=small, shortlist=0).run("hello")
    assert len(offered(small)) == 40
