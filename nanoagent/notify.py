"""The agent contacts you first.

    agent.use(telegram(), notify=True)            # unprompted messages go here
    agent.use(telegram(), notify=True, quiet=("23:00", "07:00"))

Answers to triggers are sent to that channel, and the model gets a `notify`
tool to send a message in the middle of a task. A rate limit keeps a confused
model from sending a hundred messages, and quiet hours hold messages until
the morning.
"""

from collections import deque
from datetime import time, timedelta
from typing import Annotated

from .tools import tool


class Notifier:
    def __init__(self, agent, per_minute=10, quiet=None):
        self.agent = agent
        self.target = None  # (channel, peer); peer None means the channel's default_peer
        self.per_minute = per_minute
        self.quiet = quiet  # ("23:00", "07:00") or None
        self.sent = deque()
        self.held = []

    def where(self):
        if not self.target:
            raise LookupError(
                "There is nowhere to send messages: add a channel with notify=True, "
                "e.g. agent.use(telegram(), notify=True)."
            )
        channel, peer = self.target
        peer = peer if peer is not None else channel.default_peer
        if peer is None:
            raise LookupError(f"{channel.name} does not know who to message yet.")
        return channel, peer

    def is_quiet(self, now):
        if not self.quiet:
            return False
        start, end = (time.fromisoformat(value) for value in self.quiet)
        moment = now.time()
        return start <= moment < end if start < end else moment >= start or moment < end

    async def send(self, text):
        """Send now, or hold during quiet hours. Returns "sent" or "held"."""
        channel, peer = self.where()
        now = self.agent.clock()
        if self.is_quiet(now):
            self.held.append(text)
            return "held"
        while self.sent and now - self.sent[0] > timedelta(minutes=1):
            self.sent.popleft()
        if len(self.sent) >= self.per_minute:
            raise RuntimeError(f"Rate limit: at most {self.per_minute} messages a minute. Wait before sending more.")
        self.sent.append(now)
        await channel.send(peer, text)
        return "sent"

    async def flush(self):
        """Deliver held messages once quiet hours are over."""
        if self.held and not self.is_quiet(self.agent.clock()):
            held, self.held = self.held, []
            channel, peer = self.where()
            await channel.send(peer, "\n\n".join(held))


def enable(agent):
    """Give the agent a notifier and the `notify` tool, once."""
    if agent.notifier is None:
        agent.notifier = Notifier(agent)
        agent._add_tools([_notify_tool(agent)])
    return agent.notifier


def _notify_tool(agent):
    @tool
    async def notify(message: Annotated[str, "What to tell the user"]) -> str:
        """Send the user a message right away, without ending your turn: progress on a
        long task, or something they asked to be told about."""
        status = await agent.notifier.send(message)
        return "Sent." if status == "sent" else "Held: it will be delivered when quiet hours end."

    return notify

