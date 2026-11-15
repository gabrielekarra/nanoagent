"""Issue #16: Telegram."""

import asyncio

import pytest

from fakes import FakeTelegram
from nanoagent import Agent, Model, Reply, tool
from nanoagent.channels import ServiceError, split_message
from nanoagent.channels.telegram import telegram
from nanoagent.testing import FakeModel, call, reply, running, text


class Echo(Model):
    def __init__(self):
        self.requests = 0

    def reply(self, system, tools, messages, on_text=None):
        self.requests += 1
        return Reply(text=f"echo: {messages[-1]['content']}")


@pytest.fixture
def api():
    with FakeTelegram() as server:
        yield server


def bot(api, **options):
    return telegram(token=api.token, api=api.url, poll_timeout=1, **options)


async def until(condition, timeout=3):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def test_message_in_answer_out(api):
    async def main():
        agent = Agent(model=Echo())
        agent.use(bot(api, allow=[42]))
        async with running(agent):
            api.message(42, "hello")
            await until(lambda: api.sent())

    asyncio.run(main())
    assert api.sent()[0]["chat_id"] == "42"
    assert api.sent()[0]["text"] == "echo: hello"
    assert api.sent("sendChatAction")[0] == {"chat_id": "42", "action": "typing"}


def test_strangers_are_ignored(api):
    async def main():
        model = Echo()
        agent = Agent(model=model)
        agent.use(bot(api, allow=[42]))
        async with running(agent):
            api.message(666, "let me in")
            api.message(42, "hi")
            await until(lambda: api.sent())
        return model

    model = asyncio.run(main())
    assert model.requests == 1
    assert [m["chat_id"] for m in api.sent()] == ["42"]


def test_pairing_code(api, tmp_path, capsys):
    async def main():
        agent = Agent(model=Echo(), state=tmp_path)
        channel = bot(api)
        agent.use(channel)
        async with running(agent):
            code = channel.gate.code
            api.message(7, "000000" if code != "000000" else "111111")
            api.message(8, code)
            await until(lambda: api.sent())
            api.message(8, "hi")
            await until(lambda: len(api.sent()) == 2)
        return channel, code

    channel, code = asyncio.run(main())
    assert f"send {code}" in capsys.readouterr().out
    assert channel.gate.owner == "8" and channel.default_peer == "8"
    assert [m["chat_id"] for m in api.sent()] == ["8", "8"]
    assert (tmp_path / "telegram.json").read_text() == '{"owner": "8"}'


def test_long_answer_is_split(api):
    paragraph = ("word " * 199 + "end.").strip()
    long_answer = "\n\n".join([paragraph] * 9)  # about 9,000 characters

    async def main():
        agent = Agent(model=FakeModel(reply(text(long_answer))))
        agent.use(bot(api, allow=[42]))
        async with running(agent):
            api.message(42, "tell me a lot")
            await until(lambda: len(api.sent()) >= 3)

    asyncio.run(main())
    parts = [m["text"] for m in api.sent()]
    assert all(len(part) <= 4096 for part in parts)
    assert all(part.endswith("end.") for part in parts)
    assert " ".join(parts).split() == long_answer.split()


def test_split_message_never_cuts_words():
    parts = split_message("alpha beta gamma delta", 11)
    assert parts == ["alpha beta", "gamma delta"]
    assert split_message("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]


def test_bad_markdown_falls_back_to_plain(api):
    async def main():
        agent = Agent(model=FakeModel(reply(text("Use snake_case names"))))
        agent.use(bot(api, allow=[42]))
        async with running(agent):
            api.message(42, "naming?")
            await until(lambda: api.sent())

    asyncio.run(main())
    (sent,) = api.sent()
    assert sent["text"] == "Use snake_case names" and "parse_mode" not in sent


def test_polling_survives_network_errors(api):
    api.fail_polls = 2

    async def main():
        agent = Agent(model=Echo())
        agent.use(bot(api, allow=[42]))
        async with running(agent):
            api.message(42, "one")
            api.message(42, "two")
            await until(lambda: len(api.sent()) == 2, timeout=5)
            await asyncio.sleep(0.1)

    asyncio.run(main())
    assert [m["text"] for m in api.sent()] == ["echo: one", "echo: two"]


def test_approval_buttons(api):
    deployed = []

    @tool(approve=True)
    def deploy() -> str:
        """Deploy."""
        deployed.append(True)
        return "ok"

    async def main():
        agent = Agent(tools=[deploy], model=FakeModel(reply(call("deploy")), reply(text("Deployed."))))
        agent.use(bot(api, allow=[42]))
        async with running(agent):
            api.message(42, "ship it")
            await until(lambda: api.sent())
            api.button(42, "approve:yes")
            await until(lambda: len(api.sent()) == 2)

    asyncio.run(main())
    question, done = api.sent()
    assert question["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "approve:yes"
    assert done["text"] == "Deployed." and deployed == [True]
    assert api.sent("answerCallbackQuery")


def test_token_never_appears_in_logs_or_errors(api, capsys):
    channel = bot(api)
    channel.api = api.url + "/nowhere"
    with pytest.raises(ServiceError) as caught:
        channel.call("getMe", {})
    assert api.token not in str(caught.value)
    channel.api = "http://127.0.0.1:9"
    with pytest.raises(ServiceError) as caught:
        channel.call("getMe", {})
    assert api.token not in str(caught.value)
    assert api.token not in capsys.readouterr().out


def test_missing_token_is_explained(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        telegram()
