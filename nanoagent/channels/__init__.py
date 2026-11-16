"""Built-in channels, and the pieces they share.

Every messaging channel is private by default: it answers only the people
in its allow list. With an empty list, it prints a pairing code at startup
and the first person to send that code becomes its owner.
"""

import json
import re
import secrets
import urllib.error
import urllib.request


def split_message(text, limit):
    """Split text into parts of at most `limit` characters, at paragraph, line or word boundaries."""
    text = text.strip() or "(empty)"
    parts = []
    while len(text) > limit:
        cut = -1
        for separator in ("\n\n", "\n", " "):
            cut = text.rfind(separator, 0, limit)
            if cut > 0:
                break
        if cut <= 0:  # one unbroken word longer than the limit
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    parts.append(text)
    return parts


class Gate:
    """Who may talk to a channel: an allow list, or one owner paired with a code."""

    def __init__(self, channel, allow=None):
        self.channel = channel
        self.allowed = {str(peer) for peer in allow or []}
        self.owner = None
        self.code = None

    def attach(self, agent):
        saved = self._read(agent)
        self.owner = saved.get("owner")

    def open(self):
        """At startup: print a pairing code if nobody is allowed yet."""
        if not self.allowed and not self.owner:
            self.code = f"{secrets.randbelow(10**6):06d}"
            print(f"nanoagent: send {self.code} to your {self.channel.name} bot to pair it with you.", flush=True)

    @property
    def default_peer(self):
        return self.owner or (sorted(self.allowed)[0] if self.allowed else None)

    def admits(self, peer):
        return str(peer) in self.allowed or (self.owner is not None and str(peer) == self.owner)

    def pair(self, peer, text):
        """True if `text` is the pairing code: `peer` becomes the owner."""
        if self.code and text.strip() == self.code:
            self.owner, self.code = str(peer), None
            self._write(getattr(self.channel, "agent", None))
            return True
        return False

    def _file(self, agent):
        state = getattr(agent, "state", None)
        return state / f"{self.channel.name}.json" if state else None

    def _read(self, agent):
        path = self._file(agent)
        try:
            return json.loads(path.read_text()) if path and path.exists() else {}
        except ValueError:
            return {}

    def _write(self, agent):
        path = self._file(agent)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"owner": self.owner}))


class ServiceError(Exception):
    """A messaging service answered with an error."""

    def __init__(self, message, status=None, code=None):
        super().__init__(message)
        self.status = status
        self.code = code


def post_json(url, payload, headers=None, timeout=60, secret=None):
    """POST JSON and return the decoded answer. Errors never mention `secret`."""
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json", **(headers or {})}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise ServiceError(_redact(body[:500], secret), error.code, _error_code(body)) from None
    except (urllib.error.URLError, OSError) as error:
        raise ServiceError(_redact(f"network error: {getattr(error, 'reason', error)}", secret)) from None


def _error_code(body):
    try:
        data = json.loads(body)
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    return error.get("code") if isinstance(error, dict) else data.get("error_code") if isinstance(data, dict) else None


def _redact(text, secret):
    return text.replace(secret, "***") if secret else text


def clean_markdown(text):
    """Plain text for channels that cannot render Markdown."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"`([^`]+)`", r"\1", text)
