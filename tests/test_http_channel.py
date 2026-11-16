"""Issue #18: the HTTP channel and the OpenAI-compatible endpoint."""

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from nanoagent import Agent, Model, Reply
from nanoagent.channels.http import http
from nanoagent.providers.openai import OpenAICompatible
from nanoagent.testing import running


class Writer(Model):
    """Streams its answer in three pieces and remembers the conversation length."""

    async def areply(self, system, tools, messages, on_text=None):
        turns = len([m for m in messages if m["role"] == "user"])
        pieces = ["You said ", repr(messages[-1]["content"]), f" (turn {turns})."]
        for piece in pieces:
            if on_text:
                on_text(piece)
            await asyncio.sleep(0.02)
        return Reply(text="".join(pieces))


def call(port, path, body, headers=None, token=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.headers.get("Content-Type"), response.read().decode()


def serve(steps, **options):
    async def main():
        channel = http(port=0, **options)
        agent = Agent(model=Writer())
        agent.use(channel)
        async with running(agent):
            return await asyncio.to_thread(steps, channel.port)

    return asyncio.run(main())


def test_post_message():
    kind, body = serve(lambda port: call(port, "/message", {"text": "hello"}))
    assert kind == "application/json"
    assert json.loads(body) == {"answer": "You said 'hello' (turn 1).", "session": "default"}


def test_stream_message():
    kind, body = serve(lambda port: call(port, "/message", {"text": "hi"}, {"Accept": "text/event-stream"}))
    assert kind == "text/event-stream"
    events = [block for block in body.split("\n\n") if block]
    assert [e.split("\n")[0] for e in events] == ["event: text"] * 3 + ["event: done"]
    assert json.loads(events[-1].split("data: ")[1]) == {"answer": "You said 'hi' (turn 1)."}


def test_chat_completions_shape():
    def steps(port):
        _, plain = call(port, "/v1/chat/completions", {"model": "jarvis", "messages": [{"role": "user", "content": "a"}]})
        _, streamed = call(port, "/v1/chat/completions", {"stream": True, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "b"}]}]})
        return json.loads(plain), streamed

    plain, streamed = serve(steps)
    assert plain["object"] == "chat.completion" and plain["model"] == "jarvis"
    assert plain["choices"][0]["message"] == {"role": "assistant", "content": "You said 'a' (turn 1)."}
    assert plain["choices"][0]["finish_reason"] == "stop"
    lines = [line for line in streamed.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "You said 'b' (turn 2)."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_openai_client_can_call_the_agent():
    def steps(port):
        model = OpenAICompatible(f"http://127.0.0.1:{port}/v1", "nanoagent")
        seen = []
        found = model.reply("", [], [{"role": "user", "content": "are you there?"}], seen.append)
        return found.text, seen

    answer, seen = serve(steps)
    assert answer == "You said 'are you there?' (turn 1)."
    assert len(seen) == 3


def test_token_required_off_localhost():
    async def main():
        agent = Agent(model=Writer())
        agent.use(http(port=0, host="0.0.0.0"))
        async with running(agent):
            pass

    with pytest.raises(ValueError, match="pass token"):
        asyncio.run(main())


def test_wrong_token_gets_401():
    def steps(port):
        with pytest.raises(urllib.error.HTTPError) as caught:
            call(port, "/message", {"text": "hi"}, token="wrong")
        _, ok = call(port, "/message", {"text": "hi"}, token="s3cret")
        return caught.value.code, json.loads(ok)

    status, ok = serve(steps, token="s3cret")
    assert status == 401 and ok["answer"].startswith("You said")


def test_sessions_by_header():
    def steps(port):
        call(port, "/message", {"text": "1"}, {"X-Session-Id": "ada"})
        call(port, "/message", {"text": "2"}, {"X-Session-Id": "ada"})
        _, bob = call(port, "/message", {"text": "3"}, {"X-Session-Id": "bob"})
        _, ada = call(port, "/message", {"text": "4"}, {"X-Session-Id": "ada"})
        return json.loads(ada)["answer"], json.loads(bob)["answer"]

    ada, bob = serve(steps)
    assert ada.endswith("(turn 3).") and bob.endswith("(turn 1).")
