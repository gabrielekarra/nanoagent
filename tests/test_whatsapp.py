"""Issue #17: WhatsApp."""

import asyncio
import hashlib
import hmac
import json
import urllib.error
import urllib.request

import pytest

from fakes import FakeGraph
from nanoagent import Agent, Model, Reply
from nanoagent.channels import ServiceError
from nanoagent.channels.whatsapp import whatsapp
from nanoagent.testing import running

SECRET = "app-secret"


class Echo(Model):
    def __init__(self):
        self.requests = 0

    def reply(self, system, tools, messages, on_text=None):
        self.requests += 1
        return Reply(text=f"echo: {messages[-1]['content']}")


@pytest.fixture
def graph():
    with FakeGraph() as server:
        yield server


def channel(graph, allow=("393330000001",)):
    return whatsapp(allow=list(allow), token="wa-token", phone_id="555", app_secret=SECRET,
                    verify_token="my-verify", port=0, host="127.0.0.1", graph=graph.url)


def payload(peer, text, message_id="wamid.in.1"):
    return {"object": "whatsapp_business_account", "entry": [{"changes": [{"value": {"messages": [
        {"from": peer, "id": message_id, "type": "text", "text": {"body": text}}
    ]}}]}]}


def deliver(port, body, secret=SECRET):
    data = json.dumps(body).encode()
    signature = "sha256=" + hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/webhook", data,
                                     {"X-Hub-Signature-256": signature, "Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(request, timeout=5).status
    except urllib.error.HTTPError as error:
        return error.code


async def until(condition, timeout=3):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def run_with(graph, steps, model=None):
    async def main():
        wa = channel(graph)
        agent = Agent(model=model or Echo())
        agent.use(wa)
        async with running(agent):
            await steps(wa)
        return agent

    return asyncio.run(main())


def test_webhook_verification_handshake(graph):
    async def steps(wa):
        base = f"http://127.0.0.1:{wa.port}/webhook?hub.mode=subscribe&hub.challenge=4242"
        ok = await asyncio.to_thread(urllib.request.urlopen, base + "&hub.verify_token=my-verify")
        assert ok.read() == b"4242"
        with pytest.raises(urllib.error.HTTPError) as wrong:
            await asyncio.to_thread(urllib.request.urlopen, base + "&hub.verify_token=nope")
        assert wrong.value.code == 403

    run_with(graph, steps)


def test_message_in_answer_out(graph):
    async def steps(wa):
        assert await asyncio.to_thread(deliver, wa.port, payload("393330000001", "ciao")) == 200
        await until(lambda: graph.sent)

    run_with(graph, steps)
    assert graph.sent == [("393330000001", "echo: ciao")]


def test_bad_signature_is_rejected(graph):
    model = Echo()

    async def steps(wa):
        assert await asyncio.to_thread(deliver, wa.port, payload("393330000001", "hi"), "wrong") == 403
        await asyncio.sleep(0.1)

    run_with(graph, steps, model)
    assert model.requests == 0 and graph.sent == []


def test_strangers_are_ignored(graph):
    model = Echo()

    async def steps(wa):
        await asyncio.to_thread(deliver, wa.port, payload("15550000000", "hi"))
        await asyncio.sleep(0.1)

    run_with(graph, steps, model)
    assert model.requests == 0


def test_duplicate_delivery_is_answered_once(graph):
    async def steps(wa):
        for _ in range(2):
            await asyncio.to_thread(deliver, wa.port, payload("393330000001", "once", "wamid.same"))
        await until(lambda: graph.sent)
        await asyncio.sleep(0.1)

    run_with(graph, steps)
    assert len(graph.sent) == 1


def test_outside_reply_window_error_is_explained(graph):
    graph.outside_window.add("393330000001")
    wa = channel(graph)
    with pytest.raises(ServiceError, match="24 hours"):
        wa._send("393330000001", "Good morning")


def test_missing_settings_are_named(monkeypatch):
    for name in ("WHATSAPP_TOKEN", "WHATSAPP_PHONE_ID", "WHATSAPP_APP_SECRET", "WHATSAPP_VERIFY_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="token, phone_id, app_secret, verify_token"):
        whatsapp()
