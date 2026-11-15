"""Issue #1: a provider-neutral core."""

import subprocess
import sys
import textwrap

import anthropic as anthropic_sdk
import pytest

from fakes import FakeAnthropic
from nanoagent import Agent, Call, Model, ModelError, Reply, resolve, tool
from nanoagent.providers.anthropic import FALLBACK_BETA, Anthropic
from nanoagent.providers.ollama import Ollama
from nanoagent.providers.openai import OpenAICompatible
from nanoagent.testing import FakeModel, call, reply, text


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return f"sunny in {city}"


def test_agent_runs_on_any_model_object():
    class Scripted(Model):
        def __init__(self):
            self.seen = []

        def reply(self, system, tools, messages, on_text=None):
            self.seen.append(messages[-1])
            if messages[-1]["role"] == "user":
                return Reply(calls=[Call("1", "get_weather", {"city": "Oslo"})], stop="tool")
            return Reply(text=f"It is {messages[-1]['results'][0]['content']}.")

    model = Scripted()
    assert Agent(tools=[get_weather], model=model).run("Weather in Oslo?") == "It is sunny in Oslo."
    assert model.seen[1] == {"role": "tool", "results": [
        {"id": "1", "name": "get_weather", "content": "sunny in Oslo", "error": False}
    ]}


@pytest.mark.parametrize(
    "spec, kind, url, name",
    [
        ("anthropic:claude-opus-5-5", Anthropic, None, "claude-opus-5-5"),
        ("claude-sonnet-5-5", Anthropic, None, "claude-sonnet-5-5"),
        ("ollama:qwen3:4b", Ollama, "http://localhost:11434", "qwen3:4b"),
        ("openai:gpt-5", OpenAICompatible, "https://api.openai.com/v1", "gpt-5"),
        ("openrouter:meta-llama/llama-3.3-70b-instruct", OpenAICompatible, "https://openrouter.ai/api/v1",
         "meta-llama/llama-3.3-70b-instruct"),
        ("lmstudio:qwen3-4b", OpenAICompatible, "http://localhost:1234/v1", "qwen3-4b"),
        ("llamacpp:any", OpenAICompatible, "http://localhost:8080/v1", "any"),
        ("vllm:Qwen/Qwen3-8B", OpenAICompatible, "http://localhost:8000/v1", "Qwen/Qwen3-8B"),
    ],
)
def test_model_string_is_resolved(spec, kind, url, name, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    model = resolve(spec)
    assert type(model) is kind
    assert model.model == name
    if url:
        assert model.base_url == url


def test_api_keys_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve("openrouter:x").api_key == "sk-or-1"
    assert resolve("openai:x").api_key is None
    assert resolve("lmstudio:x").api_key is None


def test_ollama_host_from_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    assert resolve("ollama:qwen3:4b").base_url == "http://gpu-box:11434"


def test_unknown_prefix_names_the_known_ones():
    with pytest.raises(ValueError, match="Known: anthropic, ollama, openai"):
        resolve("nope:model")


def test_no_model_explains_how_to_choose_one(monkeypatch):
    monkeypatch.delenv("NANOAGENT_MODEL", raising=False)
    with pytest.raises(ValueError, match="NANOAGENT_MODEL"):
        Agent()


def test_model_from_the_environment(monkeypatch):
    monkeypatch.setenv("NANOAGENT_MODEL", "lmstudio:qwen3-4b")
    assert Agent().model.model == "qwen3-4b"


def test_core_imports_without_anthropic():
    script = textwrap.dedent("""
        import sys
        sys.modules["anthropic"] = None   # any import of it fails
        import nanoagent
        from nanoagent.testing import FakeModel, reply, text
        assert nanoagent.Agent(model=FakeModel(reply(text("hi")))).run("hello") == "hi"
        try:
            nanoagent.Agent(model="claude-opus-5-5")
        except nanoagent.ModelError as error:
            assert "nanoagent[anthropic]" in str(error), error
        else:
            raise AssertionError("expected a ModelError")
        print("ok")
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.stdout.strip() == "ok", done.stderr


# --- the Anthropic adapter ----------------------------------------------------------


@pytest.fixture
def claude():
    with FakeAnthropic() as server:
        client = anthropic_sdk.Anthropic(api_key="fake", base_url=server.url, max_retries=0)
        yield server, client


def test_anthropic_adapter_replays_raw_reply(claude):
    server, client = claude
    server.script = [reply(text("Let me check."), call("get_weather", "toolu_9", city="Rome")), reply(text("Sunny."))]

    agent = Agent(tools=[get_weather], model=Anthropic("claude-opus-5-5", client=client))
    assert agent.run("Rome?") == "Sunny."

    first_reply = [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {"city": "Rome"}},
    ]
    second = server.bodies[1]
    assert second["messages"][1] == {"role": "assistant", "content": first_reply}
    assert second["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_9", "content": "sunny in Rome", "is_error": False}
    ]


def test_anthropic_request_shape(claude):
    server, client = claude
    server.script = [reply(text("hi"))]

    Agent("Be brief.", tools=[get_weather], model=Anthropic("claude-opus-5-5", client=client)).run("hello")

    (request,) = server.requests
    body = request["body"]
    assert body["stream"] is True
    assert body["system"] == "Be brief."
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0]["input_schema"] == get_weather.schema
    assert body["tools"][0]["eager_input_streaming"] is True
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in request["headers"]["anthropic-beta"]


def test_anthropic_fallbacks_only_where_supported(claude):
    server, client = claude
    server.script = [reply(text("hi"))]

    Agent(model=Anthropic("claude-haiku-5-5", client=client)).run("hello")

    assert "fallbacks" not in server.bodies[0]


def test_anthropic_reads_history_written_by_another_model(claude):
    server, client = claude
    server.script = [reply(text("Still sunny."))]
    agent = Agent(tools=[get_weather], model=FakeModel(reply(call("get_weather", "call_1", city="Rome")), reply(text("Sunny."))))
    agent.run("Rome?")

    agent.model = Anthropic("claude-opus-5-5", client=client)
    agent.run("And now?")

    messages = server.bodies[0]["messages"]
    assert messages[1] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Rome"}}
    ]}
    assert messages[3] == {"role": "assistant", "content": [{"type": "text", "text": "Sunny."}]}


def test_anthropic_stop_reasons(claude):
    server, client = claude
    server.script = [reply(stop="refusal"), reply(call("get_weather", city="x"), stop="length"), reply(text("ok"))]
    agent = Agent(tools=[get_weather], model=Anthropic("claude-opus-5-5", client=client))

    assert agent.run("one") == ""
    assert agent.stop_reason == "refusal"
    agent.run("two")
    assert "output limit" in server.bodies[-1]["messages"][-1]["content"][0]["content"]


def test_model_error_carries_status():
    error = ModelError("boom", 404, "not found")
    assert (error.status, error.detail, str(error)) == (404, "not found", "boom")
