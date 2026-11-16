"""Telegram: the agent in your pocket.

    agent.use(telegram())                      # TELEGRAM_BOT_TOKEN from the environment
    agent.use(telegram(allow=[123456789]))     # only these chat ids
    agent.use(telegram(), notify=True)         # and it messages you first

Create a bot with @BotFather to get a token. Long polling on the Bot API: no
public address, no dependency. With no allow list, a pairing code is printed
at startup; the first chat that sends it becomes the owner.
"""

import asyncio
import os
import threading
import time

from ..runtime import Channel
from . import Gate, ServiceError, post_json, split_message

LIMIT = 4096


class Telegram(Channel):
    name = "telegram"

    def __init__(self, token=None, allow=None, api="https://api.telegram.org", poll_timeout=30):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram needs a bot token: telegram(token=...) or set TELEGRAM_BOT_TOKEN (from @BotFather).")
        self.api = api.rstrip("/")
        self.poll_timeout = poll_timeout
        self.gate = Gate(self, allow)
        self.offset = None
        self.running = False
        self.thread = None

    def attach(self, agent, **options):
        super().attach(agent, **options)
        self.gate.attach(agent)

    @property
    def default_peer(self):
        return self.gate.default_peer

    # --- the Channel interface ---------------------------------------------------

    async def start(self, deliver):
        self.gate.open()
        self.running = True
        # A thread of its own: a long poll cannot be cancelled, and it must not
        # hold up the event loop's shutdown.
        self.thread = threading.Thread(target=self._poll, name="nanoagent-telegram", daemon=True)
        self.thread.start()

    async def stop(self):
        self.running = False

    async def send(self, peer, text):
        for part in split_message(text, LIMIT):
            await asyncio.to_thread(self._send_message, peer, part)

    async def typing(self, peer):
        await asyncio.to_thread(self.call, "sendChatAction", {"chat_id": peer, "action": "typing"})

    async def ask(self, peer, question):
        buttons = {"inline_keyboard": [[
            {"text": "Yes", "callback_data": "approve:yes"},
            {"text": "No", "callback_data": "approve:no"},
        ]]}
        await asyncio.to_thread(self.call, "sendMessage", {"chat_id": peer, "text": question, "reply_markup": buttons})

    # --- the Bot API ---------------------------------------------------------------

    def call(self, method, payload):
        """One Bot API call. Errors never contain the token."""
        try:
            answer = post_json(f"{self.api}/bot{self.token}/{method}", payload,
                               timeout=self.poll_timeout + 10, secret=self.token)
        except ServiceError as error:
            raise ServiceError(f"Telegram {method}: {error}", error.status) from None
        return answer.get("result")

    def _send_message(self, peer, text):
        try:
            self.call("sendMessage", {"chat_id": peer, "text": text, "parse_mode": "Markdown"})
        except ServiceError as error:
            if error.status != 400:
                raise
            self.call("sendMessage", {"chat_id": peer, "text": text})  # Markdown that does not parse

    def _poll(self):
        failures = 0
        while self.running:
            try:
                updates = self.call("getUpdates", {
                    "offset": self.offset, "timeout": self.poll_timeout,
                    "allowed_updates": ["message", "callback_query"],
                }) or []
                failures = 0
            except ServiceError:
                failures += 1
                time.sleep(min(30, 0.2 * 2**failures))  # back off, then try again
                continue
            for update in updates:
                self.offset = update["update_id"] + 1
                if self.running:
                    self._handle(update)

    def _handle(self, update):
        if "callback_query" in update:
            query = update["callback_query"]
            peer = str(query["message"]["chat"]["id"])
            self.call("answerCallbackQuery", {"callback_query_id": query["id"]})
            if self.gate.admits(peer) and query.get("data", "").startswith("approve:"):
                self.post(peer, "", approval=query["data"] == "approve:yes")
            return
        message = update.get("message") or {}
        peer, text = str(message.get("chat", {}).get("id")), message.get("text")
        if not text:
            return
        if not self.gate.admits(peer):
            if self.gate.pair(peer, text):
                self._send_message(peer, "Paired: you are now the owner of this agent.")
            return  # strangers never reach the model
        self.post(peer, text)


def telegram(token=None, allow=None, **options):
    """A Telegram bot channel. See the module docstring."""
    return Telegram(token, allow, **options)
