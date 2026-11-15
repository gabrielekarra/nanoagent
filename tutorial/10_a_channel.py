"""Chapter 10 - A channel: Telegram.

Chapter 9 left a slot for channels. This one fills it with a real messenger,
using only the standard library. A channel does three things:

    - it listens: here, a thread asks Telegram for new messages ("long polling")
    - it delivers each message to the runtime as an event
    - it sends answers back

and two that every messenger channel needs:

    - privacy: only allowed chats reach the model; anyone can message a bot
    - limits: Telegram refuses messages over 4096 characters, so long answers
      are split at paragraph boundaries

Create a bot with @BotFather, then:

    TELEGRAM_BOT_TOKEN=... TELEGRAM_ALLOW=<your chat id> python tutorial/10_a_channel.py
"""

import asyncio
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass

API = os.environ.get("TELEGRAM_API", "https://api.telegram.org")
LIMIT = 4096


@dataclass
class Event:
    text: str
    channel: object = None
    peer: str = None


def model(messages):
    """A stand-in for a real model (see chapter 8)."""
    return f"You said: {messages[-1]['content']}"


def split(text, limit=LIMIT):
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        cut = cut if cut > 0 else text.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return parts + [text]


class Telegram:
    name = "telegram"

    def __init__(self, token, allow):
        self.token, self.allow, self.running = token, {str(chat) for chat in allow}, True

    def call(self, method, payload):
        request = urllib.request.Request(f"{API}/bot{self.token}/{method}", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=40) as response:
            return json.load(response)["result"]

    async def start(self, runtime):
        loop = asyncio.get_running_loop()

        def poll():
            offset = None
            while self.running:
                try:
                    updates = self.call("getUpdates", {"offset": offset, "timeout": 30})
                except OSError:
                    time.sleep(2)  # the network failed: try again
                    continue
                for update in updates:
                    offset = update["update_id"] + 1
                    message = update.get("message") or {}
                    chat, text = str(message.get("chat", {}).get("id")), message.get("text")
                    if text and chat in self.allow:  # strangers never reach the model
                        loop.call_soon_threadsafe(runtime.queue.put_nowait, Event(text, self, chat))

        threading.Thread(target=poll, daemon=True).start()

    async def send(self, peer, text):
        for part in split(text):
            await asyncio.to_thread(self.call, "sendMessage", {"chat_id": peer, "text": part})

    async def stop(self):
        self.running = False


class Runtime:
    """Chapter 9's runtime, without triggers."""

    def __init__(self, channels):
        self.channels, self.queue, self.conversations = channels, asyncio.Queue(), {}

    async def run(self):
        for channel in self.channels:
            await channel.start(self)
        try:
            while (event := await self.queue.get()) is not None:
                messages = self.conversations.setdefault((event.channel.name, event.peer), [])
                messages.append({"role": "user", "content": event.text})
                answer = model(messages)
                messages.append({"role": "assistant", "content": answer})
                await event.channel.send(event.peer, answer)
        finally:
            for channel in self.channels:
                await channel.stop()


if __name__ == "__main__":
    bot = Telegram(os.environ["TELEGRAM_BOT_TOKEN"], os.environ.get("TELEGRAM_ALLOW", "").split(","))
    try:
        asyncio.run(Runtime([bot]).run())
    except KeyboardInterrupt:
        pass
