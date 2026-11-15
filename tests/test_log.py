"""Issue #26: record and replay."""

import json

import pytest

from nanoagent import Agent, cli, log, tool
from nanoagent.testing import FakeModel, ReplayDivergence, call, reply, text


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return f"sunny in {city}"


def recorded(tmp_path):
    model = FakeModel(reply(text("Checking."), call("get_weather", "c1", city="Rome")),
                      reply(call("get_weather", "c2", city="Oslo")), reply(text("Rome sunny, Oslo sunny.")))
    agent = Agent(tools=[get_weather], model=model)
    agent.use(log(tmp_path / "runs"))
    agent.run("Weather in Rome and Oslo?")
    (path,) = (tmp_path / "runs").glob("*.jsonl")
    return path


def test_log_contains_every_step(tmp_path):
    path = recorded(tmp_path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in records] == ["request", "reply", "tool", "request", "reply", "tool", "request", "reply"]
    assert records[0]["last"] == {"role": "user", "content": "Weather in Rome and Oslo?"}
    assert records[1]["calls"] == [{"id": "c1", "name": "get_weather", "arguments": {"city": "Rome"}}]
    assert records[2]["result"]["content"] == "sunny in Rome"
    assert records[-1]["text"] == "Rome sunny, Oslo sunny."


def test_replay_reproduces_the_run(tmp_path):
    path = recorded(tmp_path)
    ran = []

    @tool
    def get_weather(city: str) -> str:  # noqa: F811
        """Return today's weather in a city."""
        ran.append(city)
        return f"sunny in {city}"

    answer = Agent(tools=[get_weather], model=FakeModel.from_log(path)).run("Weather in Rome and Oslo?")
    assert answer == "Rome sunny, Oslo sunny." and ran == ["Rome", "Oslo"]


def test_replay_detects_divergence(tmp_path):
    path = recorded(tmp_path)

    @tool
    def get_weather(city: str) -> str:  # noqa: F811
        """Return today's weather in a city."""
        return "rain"  # the tool changed since the recording

    with pytest.raises(ReplayDivergence) as caught:
        Agent(tools=[get_weather], model=FakeModel.from_log(path)).run("Weather in Rome and Oslo?")
    message = str(caught.value)
    assert "request 2" in message and "sunny in Rome" in message and "rain" in message


def test_secrets_are_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "sk-very-secret-123")

    @tool
    def leak() -> str:
        """Leak."""
        return "the key is sk-very-secret-123"

    agent = Agent(tools=[leak], model=FakeModel(reply(call("leak")), reply(text("ok"))))
    agent.use(log(tmp_path))
    agent.run("go")
    (path,) = tmp_path.glob("*.jsonl")
    assert "sk-very-secret-123" not in path.read_text() and "the key is ***" in path.read_text()


def test_show_prints_transcript(tmp_path, capsys):
    path = recorded(tmp_path)
    cli.main(argv=["show", str(path)])
    out = capsys.readouterr().out
    assert out.splitlines()[:4] == [
        "you: Weather in Rome and Oslo?",
        "agent: Checking.",
        '  -> get_weather({"city": "Rome"})',
        "  <- result: sunny in Rome",
    ]
    assert out.rstrip().endswith("agent: Rome sunny, Oslo sunny.")
