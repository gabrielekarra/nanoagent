"""nanoagent: any model becomes an agent, and every ability is one line."""

from .agent import Agent
from .model import Call, Model, ModelError, Reply, resolve
from .runtime import Channel, Event
from .tools import CODING_TOOLS, Tool, edit_file, read_file, shell, tool, write_file
from .triggers import at, every

__all__ = [
    "Agent", "Model", "Reply", "Call", "ModelError", "resolve",
    "Tool", "tool", "CODING_TOOLS", "shell", "read_file", "write_file", "edit_file",
    "Channel", "Event", "every", "at",
]
__version__ = "0.2.0.dev0"
