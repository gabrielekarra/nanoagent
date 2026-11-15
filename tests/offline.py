"""Run code written for real models and services with nothing real behind it.

Used to check that the README and the examples run as written: every model
answers "ok", every channel is a FakeChannel that sends one message, MCP
servers, apps, the browser and Docker are stand-ins, and live() returns
after one exchange.
"""

import asyncio

import nanoagent
from nanoagent import Model, Reply
from nanoagent.agent import Agent
from nanoagent.providers.anthropic import Anthropic
from nanoagent.providers.openai import OpenAICompatible
from nanoagent.testing import FakeChannel, running


REAL = (Anthropic, OpenAICompatible)  # Ollama is an OpenAICompatible


class Okay(Model):
    key = "okay"

    def __init__(self, window=200_000, images=True):
        self.context_window, self.images = window, images
        self.requests = 0

    def reply(self, system, tools, messages, on_text=None):
        self.requests += 1
        return Reply(text="ok")


class StandIn:
    """A plugin that attaches and does nothing: an MCP server, an app, the browser, Docker."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def attach(self, agent, **options):
        agent.stand_ins = getattr(agent, "stand_ins", []) + [self]


def install(monkeypatch, delivered):
    """Patch nanoagent for an offline run. `delivered` collects (channel, peer, text) answers."""
    original = Agent.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if isinstance(self.model, REAL):  # models that would call a server; FakeModel and custom ones stay
            self.model = Okay(self.model.context_window, getattr(self.model, "images", True))

    def live(self, **options):
        async def once():
            async with running(self, **options):
                channels = [c for c in self.channels if isinstance(c, FakeChannel)]
                if channels:
                    await channels[0].say("me", "hello")
                    await channels[0].wait_for(1)
                    delivered.extend((c.name, peer, text) for c in channels for peer, text in c.sent)

        asyncio.run(once())

    monkeypatch.setattr(Agent, "__init__", init)
    monkeypatch.setattr(Agent, "live", live)
    for name in ("telegram", "whatsapp", "voice", "http", "discord", "slack", "email"):
        def channel(*args, _name=name, **kwargs):
            return FakeChannel(_name, default_peer="me")

        monkeypatch.setattr(nanoagent, name, channel)
    for name in ("mcp", "app", "browser", "computer"):
        monkeypatch.setattr(nanoagent, name, StandIn)
    real_sandbox = nanoagent.sandbox
    monkeypatch.setattr(nanoagent, "sandbox", lambda kind=None, **kwargs: StandIn() if kind else real_sandbox(**kwargs))
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.delenv("NANOAGENT_MODEL", raising=False)
