"""Issues #10 (runtime), #11 (sessions), #12 (triggers), #13 (notify), #14 (approvals over channels)."""

import asyncio
import json
from datetime import datetime, timedelta

import pytest

from nanoagent import Agent, Channel, Model, Reply, at, every, tool
from nanoagent.testing import FakeChannel, FakeClock, FakeModel, call, reply, running, text


def answer(value):
    return reply(text(value))


class Echo(Model):
    """Answers with the last user message, upper-cased."""

    def reply(self, system, tools, messages, on_text=None):
        return Reply(text=messages[-1]["content"].upper())


# --- #10 runtime --------------------------------------------------------------------


def test_reply_goes_back_to_its_channel():
    async def main():
        first, second = FakeChannel("one"), FakeChannel("two")
        agent = Agent(model=Echo())
        agent.use(first)
        agent.use(second)
        async with running(agent):
            await first.say("ada", "hello")
            await second.say("bob", "ciao")
            await first.wait_for(1)
            await second.wait_for(1)
        return first, second

    first, second = asyncio.run(main())
    assert first.sent == [("ada", "HELLO")]
    assert second.sent == [("bob", "CIAO")]
    assert first.started and first.stopped and second.stopped


def test_events_run_one_at_a_time():
    class Slow(Model):
        def __init__(self):
            self.running = self.peak = 0

        async def areply(self, system, tools, messages, on_text=None):
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.05)
            self.running -= 1
            return Reply(text="ok")

    async def main():
        channel, model = FakeChannel(), Slow()
        agent = Agent(model=model)
        agent.use(channel)
        async with running(agent):
            await asyncio.gather(channel.say("a", "1"), channel.say("b", "2"), channel.say("c", "3"))
            await channel.wait_for(3)
        return model

    assert asyncio.run(main()).peak == 1


def test_failing_event_does_not_stop_the_loop():
    class Flaky(Model):
        def __init__(self):
            self.calls = 0

        def reply(self, system, tools, messages, on_text=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model crashed")
            return Reply(text="fine")

    async def main():
        channel = FakeChannel()
        agent = Agent(model=Flaky())
        agent.use(channel)
        async with running(agent):
            await channel.say("ada", "one")
            await channel.say("ada", "two")
            return await channel.wait_for(2)

    sent = asyncio.run(main())
    assert sent[0] == ("ada", "Sorry, something went wrong: RuntimeError: model crashed")
    assert sent[1] == ("ada", "fine")


def test_stop_shuts_down_channels():
    async def main():
        channel = FakeChannel()
        agent = Agent(model=Echo())
        agent.use(channel)
        task = asyncio.create_task(agent.alive())
        while not (agent.runtime and agent.runtime.ready and agent.runtime.ready.is_set()):
            await asyncio.sleep(0.005)
        started = asyncio.get_running_loop().time()
        agent.stop()
        await asyncio.wait_for(task, 1)
        return channel, asyncio.get_running_loop().time() - started

    channel, took = asyncio.run(main())
    assert channel.stopped and took < 1


def test_use_accepts_tools_channels_and_triggers():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(model=Echo())
    agent.use(ping)
    agent.use(FakeChannel())
    agent.use(every("1h", "check"))
    assert "ping" in agent.tools and len(agent.channels) == 1 and len(agent.triggers) == 1

    with pytest.raises(TypeError, match="a tool made with @tool, a list of tools, a channel, a trigger"):
        agent.use(42)


def test_tools_cannot_be_added_after_the_first_run():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(model=Echo())
    agent.run("hello")
    with pytest.raises(RuntimeError, match="before the agent's first run"):
        agent.use(ping)


class Terminal(Channel):
    """The example channel from the docs: about 15 lines."""

    name = "terminal"

    def __init__(self, lines):
        self.lines, self.printed = lines, []

    async def start(self, deliver):
        asyncio.get_running_loop().run_in_executor(None, self.read)

    def read(self):
        for line in self.lines:
            self.post("me", line)

    async def send(self, peer, text):
        self.printed.append(text)


def test_custom_channel_in_a_few_lines():
    async def main():
        terminal = Terminal(["hello", "world"])
        agent = Agent(model=Echo())
        agent.use(terminal)
        async with running(agent):
            while len(terminal.printed) < 2:
                await asyncio.sleep(0.005)
        return terminal

    assert asyncio.run(main()).printed == ["HELLO", "WORLD"]


def test_streamed_text_reaches_channels_that_want_it():
    class Streaming(FakeChannel):
        def __init__(self):
            super().__init__()
            self.pieces = []

        def on_text(self, peer, piece):
            self.pieces.append((peer, piece))

    async def main():
        channel = Streaming()
        agent = Agent(model=FakeModel(reply(text("Hel"), text("lo"))))
        agent.use(channel)
        async with running(agent):
            await channel.say("ada", "hi")
            await channel.wait_for(1)
        return channel

    channel = asyncio.run(main())
    assert channel.pieces == [("ada", "Hel"), ("ada", "lo")]


# --- #11 sessions on disk -------------------------------------------------------------


def test_conversation_survives_restart(tmp_path):
    first = Agent(model=FakeModel(answer("Noted.")), state=tmp_path)
    first.run("My name is Ada.")

    model = FakeModel(answer("Your name is Ada."))
    second = Agent(model=model, state=tmp_path)
    second.run("What is my name?")

    sent = model.requests[0]["messages"]
    assert [m["content"] for m in sent] == ["My name is Ada.", "Noted.", "What is my name?"]


def test_peers_have_separate_histories(tmp_path):
    async def main():
        channel = FakeChannel()
        model = FakeModel(answer("hi ada"), answer("hi bob"))
        agent = Agent(model=model, state=tmp_path)
        agent.use(channel)
        async with running(agent):
            await channel.say("ada", "I am Ada")
            await channel.say("bob", "I am Bob")
            await channel.wait_for(2)
        return model

    model = asyncio.run(main())
    assert [m["content"] for m in model.requests[1]["messages"]] == ["I am Bob"]
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 2


def test_interrupted_write_is_not_fatal(tmp_path):
    agent = Agent(model=FakeModel(answer("Noted.")), state=tmp_path)
    agent.run("hello")
    (path,) = (tmp_path / "sessions").glob("*.json")
    path.write_text('{"session": "default", "messages": [{"role": "us')

    with pytest.warns(UserWarning, match="unreadable"):
        fresh = Agent(model=FakeModel(answer("hi")), state=tmp_path)
        fresh.run("again")

    assert [m["content"] for m in fresh.messages] == ["again", "hi"]
    assert list((tmp_path / "sessions").glob("*.corrupt-*"))


def test_state_is_readable_json(tmp_path):
    Agent(model=FakeModel(answer("Noted.")), state=tmp_path).run("Remember the milk")

    (path,) = (tmp_path / "sessions").glob("*.json")
    saved = json.loads(path.read_text())
    assert saved["session"] == "default"
    assert saved["messages"][0]["content"] == "Remember the milk"


# --- #12 triggers ----------------------------------------------------------------------


def checked(trigger, clock, minutes, step=1):
    fired = []
    for _ in range(0, minutes, step):
        clock.advance(minutes=step)
        fired += trigger.check(clock())
    return fired


def test_every_fires_on_schedule():
    clock, trigger = FakeClock(), every("1h", "check the build")
    trigger.attach(Agent(model=Echo()))
    trigger.check(clock())

    assert checked(trigger, clock, 180) == ["check the build"] * 3


def test_at_fires_once_a_day():
    clock = FakeClock(datetime(2026, 1, 5, 8, 0))
    trigger = at("09:00", "plan the day")
    trigger.attach(Agent(model=Echo()))
    trigger.check(clock())

    assert checked(trigger, clock, 120) == ["plan the day"]  # 08:00 -> 10:00
    assert checked(trigger, clock, 22 * 60, step=5) == []  # until 08:00 the next day
    assert checked(trigger, clock, 120) == ["plan the day"]


def test_weekday_schedule():
    clock = FakeClock(datetime(2026, 1, 6, 8, 0))  # a Tuesday
    trigger = at("mon 09:00", "plan the week")
    trigger.attach(Agent(model=Echo()))
    trigger.check(clock())

    assert checked(trigger, clock, 6 * 24 * 60 + 120, step=10) == ["plan the week"]  # only on Monday 12th
    assert trigger.last == datetime(2026, 1, 12, 9, 0)


def test_no_double_fire_after_restart(tmp_path):
    clock = FakeClock()
    first = every("1h", "check", name="build")
    first.attach(Agent(model=Echo(), state=tmp_path))
    first.check(clock())
    assert checked(first, clock, 61) == ["check"]

    again = every("1h", "check", name="build")
    again.attach(Agent(model=Echo(), state=tmp_path))
    assert again.check(clock.advance(minutes=10)) == []


@pytest.mark.parametrize("catch_up, expected", [(True, ["check"]), (False, [])])
def test_missed_run_happens_once(tmp_path, catch_up, expected):
    clock = FakeClock()
    first = every("1h", "check", name="build")
    first.attach(Agent(model=Echo(), state=tmp_path))
    first.check(clock())

    clock.advance(hours=3, minutes=30)  # the agent was down for three periods
    again = every("1h", "check", name="build", catch_up=catch_up)
    again.attach(Agent(model=Echo(), state=tmp_path))
    assert again.check(clock()) == expected
    assert again.check(clock.advance(minutes=5)) == []


@pytest.mark.parametrize("spec", ["soon", "", "5 parsecs", "1h fast"])
def test_bad_period_is_rejected_early(spec):
    with pytest.raises(ValueError, match="Cannot read the period"):
        every(spec, "x")


@pytest.mark.parametrize("spec", ["9am", "mon tue 09:00", "funday 09:00", "25:00"])
def test_bad_time_is_rejected_early(spec):
    with pytest.raises(ValueError):
        at(spec, "x")


def test_trigger_answer_reaches_default_channel():
    async def main():
        channel, clock = FakeChannel(default_peer="ada"), FakeClock()
        agent = Agent(model=FakeModel(answer("The build is green.")))
        agent.clock = clock
        agent.use(channel, notify=True)
        agent.use(every("1h", "Check the build"))
        async with running(agent) as runtime:
            clock.advance(hours=1)
            await runtime.tick_once()
            return await channel.wait_for(1)

    assert asyncio.run(main()) == [("ada", "The build is green.")]


# --- #13 notify -------------------------------------------------------------------------


def test_notify_tool_sends_mid_task():
    seen_during_run = []

    @tool
    def check() -> str:
        """Look at the channel."""
        seen_during_run.append(list(channel.sent))
        return "checked"

    channel = FakeChannel(default_peer="ada")

    async def main():
        model = FakeModel(reply(call("notify", message="Starting the long job.")), reply(call("check")), answer("Done."))
        agent = Agent(tools=[check], model=model)
        agent.use(channel, notify=True)
        await agent.arun("do the long job")

    asyncio.run(main())
    assert seen_during_run == [[("ada", "Starting the long job.")]]


def test_notify_without_target_is_an_error_result():
    model = FakeModel(reply(call("notify", message="hello")), answer("ok"))
    agent = Agent(model=model)
    agent.use(FakeChannel())  # no notify=True

    agent.run("tell me")

    result = model.tool_results()[0]
    assert result["error"] and "nowhere to send" in result["content"]


def test_rate_limit():
    channel = FakeChannel(default_peer="ada")
    model = FakeModel(reply(*(call("notify", message=f"m{n}") for n in range(11))), answer("ok"))
    agent = Agent(model=model, parallel=1)
    agent.clock = FakeClock()
    agent.use(channel, notify=True)

    agent.run("spam")

    results = model.tool_results()
    assert [r["error"] for r in results] == [False] * 10 + [True]
    assert "Rate limit" in results[10]["content"]
    assert len(channel.sent) == 10


def test_quiet_hours_hold_messages():
    async def main():
        channel, clock = FakeChannel(default_peer="ada"), FakeClock(datetime(2026, 1, 5, 3, 0))
        model = FakeModel(reply(call("notify", message="Backup finished.")), answer("ok"))
        agent = Agent(model=model)
        agent.clock = clock
        agent.use(channel, notify=True, quiet=("23:00", "07:00"))
        async with running(agent) as runtime:
            await agent.arun("run the backup", session="cron")
            await runtime.tick_once()
            held = list(channel.sent)
            clock.advance(hours=4, minutes=1)
            await runtime.tick_once()
            return held, channel.sent, model.tool_results()[0]

    held, sent, result = asyncio.run(main())
    assert held == []
    assert sent == [("ada", "Backup finished.")]
    assert "Held" in result["content"]


# --- #14 approvals over any channel -----------------------------------------------------


@tool(approve=True)
def deploy(target: str) -> str:
    """Deploy the site."""
    deployed.append(target)
    return f"deployed to {target}"


deployed = []


@pytest.fixture(autouse=True)
def reset_deployed():
    deployed.clear()


def approval_agent(channel, **options):
    model = FakeModel(reply(call("deploy", target="production")), answer("Done."))
    agent = Agent(tools=[deploy], model=model)
    agent.use(channel)
    return agent, model


async def wait_until(condition, timeout=2):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.005)


def test_approval_is_asked_on_the_origin_channel():
    async def main():
        channel = FakeChannel()
        agent, model = approval_agent(channel)
        async with running(agent):
            await channel.say("ada", "deploy it")
            await channel.wait_for(1)
            await channel.say("ada", "yes")
            return await channel.wait_for(2), model

    sent, model = asyncio.run(main())
    assert sent[0] == ("ada", 'Allow deploy({"target": "production"})? Reply yes or no.')
    assert sent[1] == ("ada", "Done.")
    assert deployed == ["production"]


def test_no_declines():
    async def main():
        channel = FakeChannel()
        agent, model = approval_agent(channel)
        async with running(agent):
            await channel.say("ada", "deploy it")
            await channel.wait_for(1)
            await channel.say("ada", "no")
            await channel.wait_for(2)
        return model

    model = asyncio.run(main())
    assert deployed == []
    assert "declined" in model.tool_results()[0]["content"]


def test_timeout_declines():
    async def main():
        channel = FakeChannel()
        agent, model = approval_agent(channel)
        async with running(agent, timeout=0.05):
            await channel.say("ada", "deploy it")
            return await channel.wait_for(3)

    sent = asyncio.run(main())
    assert sent[1] == ("ada", "No answer, so I did not run deploy.")
    assert deployed == []


def test_only_the_asker_can_approve():
    async def main():
        channel = FakeChannel()
        agent, model = approval_agent(channel)
        model.script.append(answer("Who are you?"))
        async with running(agent, timeout=0.3):
            await channel.say("ada", "deploy it")
            await channel.wait_for(1)
            await channel.say("mallory", "yes")
            return await channel.wait_for(4)

    sent = asyncio.run(main())
    assert deployed == []
    assert ("mallory", "Who are you?") in sent


def test_buttons_answer_with_an_approval_value():
    async def main():
        channel = FakeChannel()
        agent, model = approval_agent(channel)
        async with running(agent):
            await channel.say("ada", "deploy it")
            await channel.wait_for(1)
            await channel.say("ada", "", approval=True)
            return await channel.wait_for(2)

    asyncio.run(main())
    assert deployed == ["production"]


@pytest.mark.parametrize("policy, expected", [("decline", []), ("ask_default", ["production"])])
def test_trigger_policy(policy, expected):
    async def main():
        channel, clock = FakeChannel(default_peer="ada"), FakeClock()
        model = FakeModel(reply(call("deploy", target="production")), answer("Deployed."))
        agent = Agent(tools=[deploy], model=model)
        agent.clock = clock
        agent.use(channel, notify=True)
        agent.use(every("1h", "Deploy if the build is green"))
        async with running(agent, approvals=policy) as runtime:
            clock.advance(hours=1)
            await runtime.tick_once()
            if policy == "ask_default":
                await channel.wait_for(1)
                await channel.say("ada", "yes")
            await channel.wait_for(2 if policy == "ask_default" else 1)

    asyncio.run(main())
    assert deployed == expected


def test_messages_during_approval_are_queued():
    async def main():
        channel = FakeChannel()
        agent, model = approval_agent(channel)
        model.script.append(answer("It is sunny."))
        async with running(agent):
            await channel.say("ada", "deploy it")
            await channel.wait_for(1)
            await channel.say("ada", "what's the weather?")
            await channel.say("ada", "yes")
            return await channel.wait_for(3)

    sent = asyncio.run(main())
    assert [text for _, text in sent[1:]] == ["Done.", "It is sunny."]
    assert deployed == ["production"]
