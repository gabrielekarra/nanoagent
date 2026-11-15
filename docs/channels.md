# Writing a channel

A channel is a way to talk to an agent: a messenger, a socket, a microphone, a folder. Adding one is a class with three methods, and `agent.use(YourChannel())` plugs it in.

## The interface

```text
start(deliver)     begin listening
send(peer, text)   deliver the agent's answer to `peer`
stop()             stop listening (optional)
```

All three are `async`. When a message arrives, hand it to the agent with `await deliver(peer, text)` if you are on the event loop, or `self.post(peer, text)` from any other thread. A peer is whoever the message came from: a chat id, an address, a name. The runtime keeps one conversation per channel and peer, and sends each answer back to the peer that asked.

Optional extras the runtime uses when they exist:

| Member | What it does |
|---|---|
| `default_peer` | who receives unprompted messages when the channel is added with `notify=True` |
| `typing(peer)` | show that the agent is working |
| `ask(peer, question)` | ask for an approval your own way, e.g. with buttons; answer with `deliver(peer, "", approval=True)` |
| `on_text(peer, piece)` | receive the answer while it is being written; may be called from any thread |
| `can_approve = False` | for channels that cannot ask a question and wait (tools that need approval are declined) |

Use `nanoagent.channels.split_message` if the service limits message length, and `nanoagent.channels.Gate` for an allow list with pairing.

## An example: a channel made of folders

Messages are text files dropped into `inbox/<peer>/`; answers are appended to `outbox/<peer>.txt`.

```python
import asyncio
import threading
from pathlib import Path

from nanoagent import Channel


class Folders(Channel):
    name = "folders"

    def __init__(self, root, allow=()):
        self.root, self.allow, self.running = Path(root), set(allow), False

    async def start(self, deliver):
        self.running = True
        threading.Thread(target=self.watch, daemon=True).start()

    async def stop(self):
        self.running = False

    def watch(self):
        while self.running:
            for path in sorted(self.root.glob("inbox/*/*.txt")):
                peer, text = path.parent.name, path.read_text()
                path.unlink()
                if peer in self.allow:
                    self.post(peer, text)
            threading.Event().wait(0.02)

    async def send(self, peer, text):
        outbox = self.root / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        with (outbox / f"{peer}.txt").open("a") as file:
            file.write(text + "\n")
```

## Testing it against the contract

`nanoagent.testing.ChannelContract` holds the behaviour every channel must have: answers go to the sender, strangers never reach the model, long answers are split within the service's limit, and the channel stops quickly. Inherit it and say how to drive your channel; pytest runs the rest.

```python
import itertools
import tempfile

from nanoagent.testing import ChannelContract


class TestFoldersContract(ChannelContract):
    def setup_method(self):
        self.root = Path(tempfile.mkdtemp())
        self.counter = itertools.count()

    def make(self, allow):
        return Folders(self.root, allow)

    def incoming(self, peer, text):
        inbox = self.root / "inbox" / peer
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / f"{next(self.counter):04}.txt").write_text(text)

    def outgoing(self):
        return [(path.stem, line) for path in sorted(self.root.glob("outbox/*.txt"))
                for line in path.read_text().splitlines()]
```

The folder channel has no length limit, so the splitting test is skipped (`limit = None`). A channel without an allow list sets `stranger = None`.

The built-in Telegram, WhatsApp, HTTP, Slack, Discord and email channels are all tested this way in `tests/test_contract.py`, against fake versions of their services in `tests/fakes.py`. A new channel should come with its fake too, so its tests run without a network.
