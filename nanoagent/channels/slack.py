"""Slack: the agent in your workspace.

    agent.use(slack(allow=["D0123ABCD"]))     # SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET

Create a Slack app with the `chat:write` scope, subscribe it to the
`message.im` and `app_mention` events, and set the request URL to this
machine (`/slack/events` on port 3000 by default, behind HTTPS). Peers are
conversation ids: a DM with the bot, or a channel it was mentioned in. With
no allow list, a pairing code is printed at startup.
"""

import asyncio
import hashlib
import hmac
import json
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..runtime import Channel
from . import Gate, ServiceError, post_json, split_message

LIMIT = 3900  # Slack truncates longer messages


class Slack(Channel):
    name = "slack"

    def __init__(self, token=None, signing_secret=None, allow=None, port=3000, host="0.0.0.0",
                 api="https://slack.com/api"):
        self.token = token or os.environ.get("SLACK_BOT_TOKEN")
        self.signing_secret = signing_secret or os.environ.get("SLACK_SIGNING_SECRET")
        if not self.token or not self.signing_secret:
            raise ValueError("Slack needs a bot token and a signing secret: set SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET.")
        self.api = api.rstrip("/")
        self.port, self.host = port, host
        self.gate = Gate(self, allow)
        self.seen = OrderedDict()
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
            await asyncio.to_thread(self._post, peer, part)

    def _post(self, peer, text):
        answer = post_json(f"{self.api}/chat.postMessage", {"channel": peer, "text": text},
                           {"Authorization": f"Bearer {self.token}"}, secret=self.token)
        if not answer.get("ok"):  # Slack reports errors with status 200
            raise ServiceError(f"Slack chat.postMessage: {answer.get('error')}")

    def signed(self, body, timestamp, signature):
        if not timestamp or abs(time.time() - int(timestamp)) > 300:
            return False  # an old request replayed
        base = f"v0:{timestamp}:".encode() + body
        expected = "v0=" + hmac.new(self.signing_secret.encode(), base, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature or "")

    def receive(self, payload):
        event = payload.get("event") or {}
        if payload.get("event_id") in self.seen:
            return
        self.seen[payload.get("event_id")] = True
        while len(self.seen) > 1000:
            self.seen.popitem(last=False)
        if event.get("type") not in ("message", "app_mention") or event.get("bot_id") or event.get("subtype"):
            return
        peer, text = event.get("channel"), (event.get("text") or "").strip()
        if not peer or not text:
            return
        if not self.gate.admits(peer):
            if self.gate.pair(peer, text):
                self._post(peer, "Paired: you are now the owner of this agent.")
            return
        self.post(peer, text)


def _handler(channel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not channel.signed(body, self.headers.get("X-Slack-Request-Timestamp"),
                                  self.headers.get("X-Slack-Signature")):
                return self._answer(403, b"bad signature")
            try:
                payload = json.loads(body)
            except ValueError:
                return self._answer(400, b"bad json")
            if payload.get("type") == "url_verification":
                return self._answer(200, json.dumps({"challenge": payload.get("challenge")}).encode())
            self._answer(200, b"ok")  # Slack retries requests that take over 3 seconds
            channel.receive(payload)

        def _answer(self, status, data):
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def slack(allow=None, **options):
    """A Slack app channel. See the module docstring."""
    return Slack(allow=allow, **options)
