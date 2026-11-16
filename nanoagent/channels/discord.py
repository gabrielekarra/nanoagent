"""Discord: the agent as a bot in your server or your DMs.

    agent.use(discord(allow=["123456789012345678"]))   # DISCORD_BOT_TOKEN from the environment

Create an application at discord.com/developers, add a bot, turn on the
Message Content intent, and invite it. Peers are channel ids (a DM with the
bot is a channel too). With no allow list, a pairing code is printed at
startup.
"""

import asyncio
import json
import os
import random
import threading
import time

from ..runtime import Channel
from . import Gate, ServiceError, post_json, split_message
from .websocket import WebSocket

LIMIT = 2000
INTENTS = (1 << 0) | (1 << 9) | (1 << 12) | (1 << 15)  # guilds, guild and direct messages, message content


class Discord(Channel):
    name = "discord"

    def __init__(self, token=None, allow=None, api="https://discord.com/api/v10",
                 gateway="wss://gateway.discord.gg/?v=10&encoding=json"):
        self.token = token or os.environ.get("DISCORD_BOT_TOKEN")
        if not self.token:
            raise ValueError("Discord needs a bot token: discord(token=...) or set DISCORD_BOT_TOKEN.")
        self.api, self.gateway = api.rstrip("/"), gateway
        self.gate = Gate(self, allow)
        self.running = False
        self.socket = None
        self.me = None  # the bot's own user id, to ignore its own messages
        self.sequence = None

    def attach(self, agent, **options):
        super().attach(agent, **options)
        self.gate.attach(agent)

    @property
    def default_peer(self):
        return self.gate.default_peer

    async def start(self, deliver):
        self.gate.open()
        self.running = True
        threading.Thread(target=self._connect_forever, name="nanoagent-discord", daemon=True).start()

    async def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()

    async def send(self, peer, text):
        for part in split_message(text, LIMIT):
            await asyncio.to_thread(self._post, peer, part)

    async def typing(self, peer):
        await asyncio.to_thread(self._call, f"/channels/{peer}/typing", {})

    def _post(self, peer, text):
        self._call(f"/channels/{peer}/messages", {"content": text})

    def _call(self, path, payload):
        try:
            return post_json(self.api + path, payload, {"Authorization": f"Bot {self.token}"}, secret=self.token)
        except ServiceError as error:
            raise ServiceError(f"Discord {path}: {error}", error.status) from None

    # --- the gateway ---------------------------------------------------------------

    def _connect_forever(self):
        delay = 1
        while self.running:
            try:
                self._session()
                delay = 1
            except (OSError, ConnectionError, ValueError):
                if not self.running:
                    return
                time.sleep(delay)
                delay = min(60, delay * 2)

    def _session(self):
        self.socket = socket = WebSocket(self.gateway)
        hello = json.loads(socket.recv())
        interval = hello["d"]["heartbeat_interval"] / 1000
        socket.send(json.dumps({"op": 2, "d": {
            "token": self.token, "intents": INTENTS,
            "properties": {"os": "linux", "browser": "nanoagent", "device": "nanoagent"},
        }}))
        beating = threading.Event()

        def heartbeat():
            time.sleep(interval * random.random())  # jitter, as Discord asks
            while not beating.wait(interval):
                socket.send(json.dumps({"op": 1, "d": self.sequence}))

        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            while self.running:
                raw = socket.recv()
                if raw is None:
                    raise ConnectionError("the gateway closed the connection")
                self._event(json.loads(raw))
        finally:
            beating.set()

    def _event(self, event):
        self.sequence = event.get("s") or self.sequence
        if event.get("op") == 1:  # the gateway asks for a heartbeat now
            self.socket.send(json.dumps({"op": 1, "d": self.sequence}))
        if event.get("op") != 0:
            return
        data = event.get("d") or {}
        if event.get("t") == "READY":
            self.me = data.get("user", {}).get("id")
        elif event.get("t") == "MESSAGE_CREATE":
            author = data.get("author") or {}
            if author.get("bot") or author.get("id") == self.me:
                return
            peer, text = str(data.get("channel_id")), data.get("content") or ""
            if not text:
                return
            if not self.gate.admits(peer):
                if self.gate.pair(peer, text):
                    self._post(peer, "Paired: you are now the owner of this agent.")
                return
            self.post(peer, text)


def discord(token=None, allow=None, **options):
    """A Discord bot channel. See the module docstring."""
    return Discord(token, allow, **options)
