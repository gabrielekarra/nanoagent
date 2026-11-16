"""WhatsApp, through the official WhatsApp Cloud API.

    agent.use(whatsapp(allow=["393331234567"]))

Setup, once: create an app at developers.facebook.com, add the WhatsApp
product, and note the phone number id, an access token and the app secret.
Then point the app's webhook at this machine (port 8443 by default, behind
HTTPS: a tunnel such as cloudflared or ngrok works) with a verify token of
your choice. The values come from the arguments or from WHATSAPP_TOKEN,
WHATSAPP_PHONE_ID, WHATSAPP_APP_SECRET and WHATSAPP_VERIFY_TOKEN.

WhatsApp only lets a business message someone within 24 hours of their last
message. Outside that window an unprompted message needs a template approved
by Meta, and sending fails with an explanation.
"""

import asyncio
import hashlib
import hmac
import json
import os
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..runtime import Channel
from . import Gate, ServiceError, post_json, split_message

LIMIT = 4096
OUTSIDE_WINDOW = 131047  # "Re-engagement message": more than 24 hours since the user wrote


class WhatsApp(Channel):
    name = "whatsapp"

    def __init__(self, token=None, phone_id=None, app_secret=None, verify_token=None, allow=None,
                 port=8443, host="0.0.0.0", graph="https://graph.facebook.com/v21.0"):
        env = os.environ.get
        self.token = token or env("WHATSAPP_TOKEN")
        self.phone_id = phone_id or env("WHATSAPP_PHONE_ID")
        self.app_secret = app_secret or env("WHATSAPP_APP_SECRET")
        self.verify_token = verify_token or env("WHATSAPP_VERIFY_TOKEN")
        missing = [name for name, value in (("token", self.token), ("phone_id", self.phone_id),
                                            ("app_secret", self.app_secret), ("verify_token", self.verify_token)) if not value]
        if missing:
            raise ValueError(f"WhatsApp needs {', '.join(missing)}: pass them or set WHATSAPP_* (see the module docstring).")
        self.graph = graph.rstrip("/")
        self.port, self.host = port, host
        self.gate = Gate(self, allow)
        self.seen = OrderedDict()  # message ids already handled; WhatsApp retries deliveries
        self.server = None

    def attach(self, agent, **options):
        super().attach(agent, **options)
        self.gate.attach(agent)

    @property
    def default_peer(self):
        return self.gate.default_peer

    async def start(self, deliver):
        self.gate.open()
        self.server = ThreadingHTTPServer((self.host, self.port), _handler(self))
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, args=(0.1,), daemon=True).start()

    async def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    async def send(self, peer, text):
        for part in split_message(text, LIMIT):
            await asyncio.to_thread(self._send, peer, part)

    def _send(self, peer, text):
        payload = {"messaging_product": "whatsapp", "to": peer, "type": "text", "text": {"body": text}}
        try:
            post_json(f"{self.graph}/{self.phone_id}/messages", payload,
                      {"Authorization": f"Bearer {self.token}"}, secret=self.token)
        except ServiceError as error:
            if error.code == OUTSIDE_WINDOW:
                raise ServiceError(
                    f"WhatsApp: {peer} has not written in the last 24 hours, so WhatsApp only allows a "
                    "template message approved by Meta. Wait for them to write first.", error.status, error.code
                ) from None
            raise ServiceError(f"WhatsApp: {error}", error.status, error.code) from None

    # --- the webhook --------------------------------------------------------------

    def signed(self, body, signature):
        expected = "sha256=" + hmac.new(self.app_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature or "")

    def receive(self, payload):
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                for message in change.get("value", {}).get("messages", []):
                    self._message(message)

    def _message(self, message):
        if message.get("id") in self.seen:
            return
        self.seen[message.get("id")] = True
        while len(self.seen) > 1000:
            self.seen.popitem(last=False)
        peer = message.get("from")
        text = (message.get("text") or {}).get("body")
        if message.get("type") == "interactive":
            reply = message.get("interactive", {}).get("button_reply", {})
            text = reply.get("title")
        if not peer or not text:
            return
        if not self.gate.admits(peer):
            if self.gate.pair(peer, text):
                self._send(peer, "Paired: you are now the owner of this agent.")
            return
        self.post(peer, text)


def _handler(channel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # Meta's verification handshake
            query = {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()}
            if query.get("hub.mode") == "subscribe" and query.get("hub.verify_token") == channel.verify_token:
                self._answer(200, query.get("hub.challenge", "").encode())
            else:
                self._answer(403, b"forbidden")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not channel.signed(body, self.headers.get("X-Hub-Signature-256")):
                return self._answer(403, b"bad signature")
            self._answer(200, b"ok")  # answer fast: Meta retries slow webhooks
            try:
                channel.receive(json.loads(body))
            except ValueError:
                pass

        def _answer(self, status, data):
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def whatsapp(allow=None, **options):
    """A WhatsApp Cloud API channel. See the module docstring."""
    return WhatsApp(allow=allow, **options)
