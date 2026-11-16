"""Issue #20: details of the Slack, Discord and email channels beyond the contract."""

import asyncio
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request

import pytest

from fakes import FakeDiscord, FakeMailbox, FakeSlack
from nanoagent import Agent, Model, Reply
from nanoagent.channels.discord import discord
from nanoagent.channels.email import _text, email
from nanoagent.channels.slack import slack
from nanoagent.testing import running


class Echo(Model):
    def __init__(self):
        self.seen = []

    def reply(self, system, tools, messages, on_text=None):
        self.seen.append(messages[-1]["content"])
        return Reply(text="ok")


def test_slack_url_verification_and_signatures():
    with FakeSlack() as api:
        async def main():
            channel = slack(allow=["D1"], token="xoxb-test", signing_secret="sign", port=0, host="127.0.0.1", api=api.url)
            agent = Agent(model=Echo())
            agent.use(channel)
            async with running(agent):
                body = json.dumps({"type": "url_verification", "challenge": "abc"}).encode()

                def post(stamp, secret="sign"):
                    signature = "v0=" + hmac.new(secret.encode(), f"v0:{stamp}:".encode() + body, hashlib.sha256).hexdigest()
                    request = urllib.request.Request(f"http://127.0.0.1:{channel.port}/slack/events", body,
                                                     {"X-Slack-Request-Timestamp": stamp, "X-Slack-Signature": signature})
                    try:
                        with urllib.request.urlopen(request, timeout=5) as response:
                            return response.status, response.read()
                    except urllib.error.HTTPError as error:
                        return error.code, b""

                now = str(int(time.time()))
                return (await asyncio.to_thread(post, now),
                        await asyncio.to_thread(post, now, "wrong"),
                        await asyncio.to_thread(post, str(int(time.time()) - 3600)))

        good, forged, replayed = asyncio.run(main())
    assert good == (200, b'{"challenge": "abc"}')
    assert forged[0] == 403 and replayed[0] == 403


def test_discord_identifies_and_ignores_bots():
    with FakeDiscord() as fake:
        model = Echo()

        async def main():
            agent = Agent(model=model)
            agent.use(discord(token="discord-token", allow=["c1"], api=fake.url, gateway=fake.gateway))
            async with running(agent):
                await asyncio.to_thread(fake.message, "c1", "from a bot", "other-bot", True)
                await asyncio.to_thread(fake.message, "c1", "from me", "bot-1")
                await asyncio.to_thread(fake.message, "c1", "from a person")
                while not fake.sent:
                    await asyncio.sleep(0.01)

        asyncio.run(main())
    assert model.seen == ["from a person"]
    assert fake.identified[0]["token"] == "discord-token"
    assert fake.identified[0]["intents"] & (1 << 15)  # message content


def test_email_replies_in_thread():
    box = FakeMailbox()

    async def main():
        agent = Agent(model=Echo())
        agent.use(email("agent@example.com", allow=["ada@example.com"], imap=box.imap, smtp=box.smtp, every=0.02))
        async with running(agent):
            box.receive("Ada <ada@example.com>", "What's on today?\n\nOn Mon, Bob wrote:\n> old stuff", subject="Plans")
            while not box.outbox:
                await asyncio.sleep(0.01)

    asyncio.run(main())
    (sent,) = box.outbox
    assert sent["Subject"] == "Re: Plans"
    assert sent["In-Reply-To"] == "<m0@example.com>"
    assert sent["To"] == "ada@example.com"


def test_email_body_drops_the_quoted_part():
    import email as mail

    message = mail.message_from_string("From: a@b.c\nSubject: x\n\nHello\n\nOn Tue, X wrote:\n> quoted\n")
    assert _text(message) == "Hello"


def test_email_needs_settings(monkeypatch):
    for name in ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="IMAP and SMTP"):
        email()
