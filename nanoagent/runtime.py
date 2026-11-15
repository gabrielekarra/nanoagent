"""The runtime: what keeps an agent alive between requests.

`agent.live()` starts every channel (a way to talk to the agent) and every
trigger (a reason for it to act on its own), then answers events one at a
time: an event arrives, the agent works on it, the answer goes back where the
event came from.

A channel is a small class. This one answers on standard input and output:

    class Terminal(Channel):
        name = "terminal"

        async def start(self, deliver):
            asyncio.get_running_loop().run_in_executor(None, self.read)

        def read(self):
            for line in sys.stdin:
                self.post("me", line.strip())     # thread-safe delivery

        async def send(self, peer, text):
            print(text)

    agent.use(Terminal())
    agent.live()
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from functools import partial

YES = {"yes", "y", "ok", "okay", "sure", "allow", "approve", "go", "go ahead", "yep", "si", "sì", "do it"}
NO = {"no", "n", "nope", "deny", "stop", "cancel", "don't", "dont"}
STOP = object()


def verdict(text):
    """True for yes, False for no, None for anything else."""
    word = (text or "").strip().lower().rstrip(".!")
    return True if word in YES else False if word in NO else None


@dataclass
class Event:
    text: str
    channel: object = None
    peer: object = None
    session: str = None
    kind: str = "message"  # or "trigger"

    @property
    def key(self):
        """The conversation this event belongs to: one per channel and peer."""
        if self.session:
            return self.session
        return f"{self.channel.name}:{self.peer}" if self.channel else "triggers"


class Channel:
    """A way to talk to the agent. Implement `start`, `send` and, if needed, `stop`.

    start(deliver)   begin listening; for each incoming message, `await deliver(peer, text)`
                     on the event loop, or call `self.post(peer, text)` from any thread
    send(peer, text) deliver the agent's answer
    stop()           stop listening

    Optional: `typing(peer)` (show that the agent is working), `ask(peer, question)`
    (ask for an approval, e.g. with buttons), `on_text(peer, piece)` (stream the
    answer as it is written; may be called from any thread), and `default_peer`
    (who to contact first when the agent has something to say).
    """

    name = "channel"
    default_peer = None

    def attach(self, agent, notify=False, peer=None, quiet=None, per_minute=None):
        from .notify import enable

        self.agent = agent
        agent.channels.append(self)
        notifier = enable(agent)
        if notify:
            notifier.target = (self, peer)
        if quiet:
            notifier.quiet = quiet
        if per_minute:
            notifier.per_minute = per_minute

    async def start(self, deliver):
        pass

    async def send(self, peer, text):
        raise NotImplementedError

    async def stop(self):
        pass

    def post(self, peer, text, **extra):
        """Deliver an incoming message from any thread."""
        try:
            future = asyncio.run_coroutine_threadsafe(self._deliver(peer, text, **extra), self._loop)
            return future.result(timeout=30)
        except RuntimeError:
            return None  # the runtime has stopped


class Runtime:
    """Runs an agent's channels and triggers.

    approvals   what to do when a trigger's work needs an approval: "decline",
                or "ask_default" to ask whoever receives the agent's messages
    timeout     seconds to wait for an approval before treating it as a no
    tick        seconds between checks of the triggers
    """

    def __init__(self, agent, approvals="decline", timeout=300, tick=1.0):
        if approvals not in ("decline", "ask_default"):
            raise ValueError("approvals must be 'decline' or 'ask_default'")
        self.agent = agent
        self.approvals = approvals
        self.timeout = timeout
        self.tick = tick
        self.pending = {}  # (channel name, peer) -> future waiting for a yes or a no
        self.loop = None
        self.queue = None
        self.ready = None  # set once every channel has started

    async def run(self):
        agent = self.agent
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        self.ready = asyncio.Event()
        agent.runtime = self
        started, ticker = [], None
        try:
            for channel in agent.channels:
                deliver = partial(self.deliver, channel)
                channel._deliver, channel._loop = deliver, self.loop
                await channel.start(deliver)
                started.append(channel)
            ticker = asyncio.create_task(self._tick_forever())
            self.ready.set()
            while True:
                event = await self.queue.get()
                if event is STOP:
                    break
                await self.handle(event)
        finally:
            if ticker:
                ticker.cancel()
            for channel in reversed(started):
                try:
                    await asyncio.wait_for(channel.stop(), 5)
                except Exception as error:  # shutting down: report and carry on
                    print(f"nanoagent: {channel.name} did not stop cleanly: {error}", file=sys.stderr)
            agent.runtime = None
            agent.close()

    def stop(self):
        """Stop the runtime. Safe to call from any thread."""
        if self.loop and self.queue is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, STOP)

    async def submit(self, event):
        await self.queue.put(event)

    async def deliver(self, channel, peer, text, session=None, approval=None):
        """A message arrived on `channel`. It answers a pending approval, or it is queued."""
        waiting = self.pending.get((channel.name, peer))
        if waiting and not waiting.done():
            answer = approval if approval is not None else verdict(text)
            if answer is not None:
                waiting.set_result(answer)
                return
        if approval is not None:
            return  # a button pressed for a question nobody is waiting on any more
        await self.queue.put(Event(text, channel, peer, session))

    async def handle(self, event):
        agent, channel, peer = self.agent, event.channel, event.peer
        on_text = partial(channel.on_text, peer) if channel and hasattr(channel, "on_text") else None
        if channel and hasattr(channel, "typing"):
            try:
                await channel.typing(peer)
            except Exception:
                pass  # an indicator is not worth failing an event for
        try:
            answer = await agent.arun(event.text, session=event.key, on_text=on_text, approve=self._approver(event))
            if not answer and agent.stop_reason == "refusal":
                answer = "I can't help with that."
        except Exception as error:  # one failing event does not stop the agent
            answer = f"Sorry, something went wrong: {type(error).__name__}: {error}"
        try:
            if channel:
                await channel.send(peer, answer or "(no answer)")
            elif answer and agent.notifier and agent.notifier.target:
                await agent.notifier.send(answer)
            elif answer:
                print(answer)
        except Exception as error:
            print(f"nanoagent: could not deliver an answer: {type(error).__name__}: {error}", file=sys.stderr)

    # --- approvals --------------------------------------------------------------

    def _approver(self, event):
        if event.channel is not None:
            return partial(self._ask, event.channel, event.peer)
        notifier = self.agent.notifier
        if self.approvals == "ask_default" and notifier and notifier.target:
            channel, peer = notifier.where()
            return partial(self._ask, channel, peer)
        return lambda name, arguments: False

    async def _ask(self, channel, peer, name, arguments):
        if not getattr(channel, "can_approve", True):  # e.g. one HTTP request, one answer
            return False
        key = (channel.name, peer)
        waiting = self.loop.create_future()
        self.pending[key] = waiting
        question = f"Allow {name}({json.dumps(arguments, ensure_ascii=False)[:300]})? Reply yes or no."
        try:
            if hasattr(channel, "ask"):
                await channel.ask(peer, question)
            else:
                await channel.send(peer, question)
            return await asyncio.wait_for(waiting, self.timeout)
        except asyncio.TimeoutError:
            await channel.send(peer, f"No answer, so I did not run {name}.")
            return False
        finally:
            self.pending.pop(key, None)

    # --- triggers and held messages -----------------------------------------------

    async def tick_once(self):
        now = self.agent.clock()
        for trigger in self.agent.triggers:
            for prompt in trigger.check(now):
                await self.queue.put(Event(prompt, session=f"trigger:{trigger.name}", kind="trigger"))
        if self.agent.notifier:
            await self.agent.notifier.flush()

    async def _tick_forever(self):
        while True:
            try:
                await self.tick_once()
            except Exception as error:
                print(f"nanoagent: trigger check failed: {type(error).__name__}: {error}", file=sys.stderr)
            await asyncio.sleep(self.tick)
