"""Email: write to the agent, it writes back.

    agent.use(email(address="agent@example.com", allow=["me@example.com"]))

It checks the inbox over IMAP for unread mail from allowed senders and
answers over SMTP, in the same thread. Settings come from the arguments or
from EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST and EMAIL_SMTP_HOST (an
app password is usually needed). Peers are sender addresses. With no allow
list, a pairing code is printed at startup: send it as the subject or body.
"""

import asyncio
import email as mail
import imaplib
import os
import smtplib
import threading
import time
from email.message import EmailMessage
from email.utils import make_msgid, parseaddr

from ..runtime import Channel
from . import Gate


class Email(Channel):
    name = "email"

    def __init__(self, address=None, password=None, imap_host=None, smtp_host=None, allow=None,
                 every=30, imap=None, smtp=None):
        env = os.environ.get
        self.address = address or env("EMAIL_ADDRESS")
        self.password = password or env("EMAIL_PASSWORD")
        self.imap_host = imap_host or env("EMAIL_IMAP_HOST")
        self.smtp_host = smtp_host or env("EMAIL_SMTP_HOST")
        if not (imap and smtp) and not all((self.address, self.password, self.imap_host, self.smtp_host)):
            raise ValueError("Email needs an address, a password and the IMAP and SMTP hosts (or EMAIL_* variables).")
        self.imap = imap or (lambda: imaplib.IMAP4_SSL(self.imap_host))
        self.smtp = smtp or (lambda: smtplib.SMTP_SSL(self.smtp_host))
        self.every = every
        self.gate = Gate(self, [address.lower() for address in allow or []])
        self.threads = {}  # sender -> (subject, message id) of the last mail, to reply in thread
        self.running = False

    def attach(self, agent, **options):
        super().attach(agent, **options)
        self.gate.attach(agent)

    @property
    def default_peer(self):
        return self.gate.default_peer

    async def start(self, deliver):
        self.gate.open()
        self.running = True
        threading.Thread(target=self._poll, name="nanoagent-email", daemon=True).start()

    async def stop(self):
        self.running = False

    async def send(self, peer, text):
        await asyncio.to_thread(self._send, peer, text)

    def _send(self, peer, text):
        subject, reference = self.threads.get(peer, ("A message from your agent", None))
        message = EmailMessage()
        message["From"], message["To"] = self.address, peer
        message["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        # With no domain, make_msgid looks up this machine's name, which can hang for seconds.
        domain = self.address.split("@")[-1] if self.address and "@" in self.address else "nanoagent.local"
        message["Message-ID"] = make_msgid(domain=domain)
        if reference:
            message["In-Reply-To"] = message["References"] = reference
        message.set_content(text)
        server = self.smtp()
        try:
            if self.password:
                server.login(self.address, self.password)
            server.send_message(message)
        finally:
            server.quit()

    def check(self):
        """Read unread mail once. Returns how many messages were delivered."""
        delivered = 0
        box = self.imap()
        try:
            if self.password:
                box.login(self.address, self.password)
            box.select("INBOX")
            _, found = box.search(None, "UNSEEN")
            for number in (found[0] or b"").split():
                _, parts = box.fetch(number, "(RFC822)")
                box.store(number, "+FLAGS", "\\Seen")
                delivered += self._message(mail.message_from_bytes(parts[0][1]))
        finally:
            box.logout()
        return delivered

    def _message(self, message):
        sender = parseaddr(message.get("From", ""))[1].lower()
        subject = message.get("Subject", "") or "(no subject)"
        body = _text(message)
        if not sender:
            return 0
        self.threads[sender] = (subject, message.get("Message-ID"))
        if not self.gate.admits(sender):
            if self.gate.pair(sender, subject) or self.gate.pair(sender, body.strip().split("\n")[0]):
                self._send(sender, "Paired: you are now the owner of this agent.")
            return 0
        self.post(sender, body or subject)
        return 1

    def _poll(self):
        while self.running:
            try:
                self.check()
            except (OSError, imaplib.IMAP4.error):
                pass  # try again at the next check
            for _ in range(int(self.every * 10)):
                if not self.running:
                    return
                time.sleep(0.1)


def _text(message):
    """The plain-text body, without the quoted earlier mail."""
    part = message
    if message.is_multipart():
        part = next((p for p in message.walk() if p.get_content_type() == "text/plain"), None)
        if part is None:
            return ""
    text = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
    lines = []
    for line in text.splitlines():
        if line.startswith(">") or (line.startswith("On ") and line.rstrip().endswith("wrote:")):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def email(address=None, allow=None, **options):
    """An email channel. See the module docstring."""
    return Email(address, allow=allow, **options)
