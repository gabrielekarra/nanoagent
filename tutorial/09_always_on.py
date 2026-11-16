"""Chapter 9 - Always on.

So far the agent only ran while someone typed. To be an assistant it has to
wait for things to happen, and two kinds of things happen:

    - a message arrives on a channel: a terminal, a chat app, a microphone
    - a trigger fires: every hour, every morning at nine

Both become events in one queue. The runtime takes them one at a time, runs
the agent, and sends the answer back where the event came from. Each peer
(each person on each channel) gets a conversation of its own.

The model call is a stand-in here (see chapter 8 for a real one), so the
shape of the runtime is the only new thing on the page.

    python tutorial/09_always_on.py
"""

import asyncio
import sys
import time
from dataclasses import dataclass


@dataclass
class Event:
    text: str
    channel: object = None  # None for a trigger
    peer: str = None


def model(messages):
    """A stand-in for a real model: it says how long the conversation is."""
    return f"(message {len(messages)} of this conversation) You said: {messages[-1]['content']}"


class Terminal:
    """A channel: lines typed in the terminal are messages from "you"."""

    name = "terminal"

    async def start(self, runtime):
        loop = asyncio.get_running_loop()

        def read():  # input() blocks, so it runs on a thread of its own
            for line in sys.stdin:
                loop.call_soon_threadsafe(runtime.queue.put_nowait, Event(line.strip(), self, "you"))

        loop.run_in_executor(None, read)

    async def send(self, peer, text):
        print(f"[to {peer}] {text}")


class Every:
    """A trigger: a prompt to run every `seconds`."""

    def __init__(self, seconds, prompt):
        self.seconds, self.prompt, self.last = seconds, prompt, time.monotonic()

    def due(self, now):
        if now - self.last >= self.seconds:
            self.last = now
            return True
        return False


class Runtime:
    def __init__(self, channels, triggers, notify=None):
        self.channels, self.triggers, self.notify = channels, triggers, notify
        self.queue = asyncio.Queue()
        self.conversations = {}  # (channel name, peer) -> messages

    async def run(self):
        for channel in self.channels:
            await channel.start(self)
        ticker = asyncio.create_task(self.tick())
        try:
            while True:
                event = await self.queue.get()
                if event is None:
                    return
                await self.handle(event)
        finally:
            ticker.cancel()

    async def tick(self):
        while True:
            now = time.monotonic()
            for trigger in self.triggers:
                if trigger.due(now):
                    await self.queue.put(Event(trigger.prompt))
            await asyncio.sleep(0.05)

    async def handle(self, event):
        key = (event.channel.name, event.peer) if event.channel else ("trigger", event.text)
        messages = self.conversations.setdefault(key, [])
        messages.append({"role": "user", "content": event.text})
        try:
            answer = model(messages)
        except Exception as error:  # one failing event never stops the agent
            answer = f"Sorry, something went wrong: {error}"
        messages.append({"role": "assistant", "content": answer})
        if event.channel:  # back where it came from
            await event.channel.send(event.peer, answer)
        elif self.notify:  # a trigger: to whoever wants unprompted messages
            channel, peer = self.notify
            await channel.send(peer, answer)


if __name__ == "__main__":
    terminal = Terminal()
    runtime = Runtime([terminal], [Every(30, "Remind me to stretch.")], notify=(terminal, "you"))
    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        pass
