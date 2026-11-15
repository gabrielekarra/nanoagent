"""Issue #20: every channel passes the same contract, and the guide's example is real code."""

import asyncio
import hashlib
import hmac
import json
import re
import time
import urllib.request
from pathlib import Path

import pytest

from fakes import FakeDiscord, FakeGraph, FakeMailbox, FakeSlack, FakeTelegram
from nanoagent.channels.discord import discord
from nanoagent.channels.email import email
from nanoagent.channels.http import http
from nanoagent.channels.slack import slack
from nanoagent.channels.telegram import telegram
from nanoagent.channels.whatsapp import whatsapp
from nanoagent.runtime import Channel
from nanoagent.testing import ChannelContract

GUIDE = Path(__file__).parent.parent / "docs" / "channels.md"


def signed_post(url, body, headers):
    request = urllib.request.Request(url, body, {"Content-Type": "application/json", **headers})
    return urllib.request.urlopen(request, timeout=5).status


class TestTelegramContract(ChannelContract):
    limit = 4096

    def setup_method(self):
        self.api = FakeTelegram()

    def teardown_method(self):
        self.api.close()

    def make(self, allow):
        return telegram(token=self.api.token, api=self.api.url, allow=allow, poll_timeout=1)

    def incoming(self, peer, text):
        self.api.message(int(peer), text)

    def outgoing(self):
        return [(m["chat_id"], m["text"]) for m in self.api.sent()]


class TestWhatsAppContract(ChannelContract):
    peer, stranger, limit = "393330000001", "15550000000", 4096

    def setup_method(self):
        self.graph = FakeGraph()
        self.count = 0

    def teardown_method(self):
        self.graph.close()

    def make(self, allow):
        self.channel = whatsapp(allow=allow, token="wa-token", phone_id="555", app_secret="s", verify_token="v",
                                port=0, host="127.0.0.1", graph=self.graph.url)
        return self.channel

    def incoming(self, peer, text):
        self.count += 1
        body = json.dumps({"entry": [{"changes": [{"value": {"messages": [
            {"from": peer, "id": f"m{self.count}", "type": "text", "text": {"body": text}}]}}]}]}).encode()
        signature = "sha256=" + hmac.new(b"s", body, hashlib.sha256).hexdigest()
        while not getattr(self.channel, "server", None):
            time.sleep(0.01)
        signed_post(f"http://127.0.0.1:{self.channel.port}/", body, {"X-Hub-Signature-256": signature})

    def outgoing(self):
        return list(self.graph.sent)


class TestSlackContract(ChannelContract):
    peer, stranger, limit = "D0001", "C6666", 3900

    def setup_method(self):
        self.api = FakeSlack()
        self.count = 0

    def teardown_method(self):
        self.api.close()

    def make(self, allow):
        self.channel = slack(allow=allow, token="xoxb-test", signing_secret="sign", port=0, host="127.0.0.1", api=self.api.url)
        return self.channel

    def incoming(self, peer, text):
        self.count += 1
        body = json.dumps({"type": "event_callback", "event_id": f"E{self.count}",
                           "event": {"type": "message", "channel": peer, "user": "U1", "text": text}}).encode()
        stamp = str(int(time.time()))
        signature = "v0=" + hmac.new(b"sign", f"v0:{stamp}:".encode() + body, hashlib.sha256).hexdigest()
        while not getattr(self.channel, "server", None):
            time.sleep(0.01)
        signed_post(f"http://127.0.0.1:{self.channel.port}/slack/events", body,
                    {"X-Slack-Request-Timestamp": stamp, "X-Slack-Signature": signature})

    def outgoing(self):
        return list(self.api.sent)


class TestDiscordContract(ChannelContract):
    limit = 2000

    def setup_method(self):
        self.fake = FakeDiscord()

    def teardown_method(self):
        self.fake.close()

    def make(self, allow):
        return discord(token="discord-token", allow=allow, api=self.fake.url, gateway=self.fake.gateway)

    def incoming(self, peer, text):
        self.fake.message(peer, text)

    def outgoing(self):
        return list(self.fake.sent)


class TestEmailContract(ChannelContract):
    peer, stranger = "ada@example.com", "spam@example.net"

    def setup_method(self):
        self.box = FakeMailbox()

    def make(self, allow):
        return email("agent@example.com", allow=allow, imap=self.box.imap, smtp=self.box.smtp, every=0.02)

    def incoming(self, peer, text):
        self.box.receive(peer, text)

    def outgoing(self):
        return [(m["To"], m.get_content().strip()) for m in self.box.outbox]


class TestHTTPContract(ChannelContract):
    stranger = None  # an HTTP endpoint has no allow list: it has a token

    def make(self, allow):
        self.channel = http(port=0)
        self.answers = []
        return self.channel

    def incoming(self, peer, text):
        while not getattr(self.channel, "server", None):
            time.sleep(0.01)
        request = urllib.request.Request(f"http://127.0.0.1:{self.channel.port}/message",
                                         json.dumps({"text": text, "session": peer}).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.answers.append((peer, json.loads(response.read())["answer"]))

    def outgoing(self):
        return list(self.answers)


# --- the contract catches mistakes ---------------------------------------------------------


class WrongPeer(Channel):
    name = "wrong"

    def __init__(self, sent):
        self.sent = sent

    async def send(self, peer, text):
        self.sent.append(("someone-else", text))


class BrokenContract(ChannelContract):
    stranger = None

    def make(self, allow):
        self.sent = []
        self.channel = WrongPeer(self.sent)
        return self.channel

    def incoming(self, peer, text):
        while not getattr(self.channel, "_loop", None):
            time.sleep(0.01)
        self.channel.post(peer, text)

    def outgoing(self):
        return self.sent


def test_contract_catches_a_broken_channel():
    with pytest.raises(AssertionError):
        BrokenContract().test_contract_delivers_to_the_sender()


# --- the guide ---------------------------------------------------------------------------


def test_guide_example_passes_the_contract():
    code = "\n".join(re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL))
    namespace = {"__name__": "guide"}
    exec(compile(code, str(GUIDE), "exec"), namespace)
    contract = namespace["TestFoldersContract"]
    for name in ("test_contract_delivers_to_the_sender", "test_contract_ignores_strangers", "test_contract_stops_quickly"):
        case = contract()
        case.setup_method()
        getattr(case, name)()
