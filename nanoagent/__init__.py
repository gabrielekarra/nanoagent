"""nanoagent: any model becomes an agent, and every ability is one line."""

from .agent import Agent
from .apps import app
from .browser import browser
from .channels.discord import discord
from .channels.email import email
from .channels.http import http
from .channels.slack import slack
from .channels.telegram import telegram
from .channels.voice import voice
from .channels.whatsapp import whatsapp
from .computer import computer
from .mcp import mcp
from .model import Call, Model, ModelError, Reply, resolve
from .runtime import Channel, Event
from .safety import rules, sandbox
from .tools import CODING_TOOLS, Image, Tool, edit_file, read_file, shell, tool, write_file
from .triggers import at, every

__all__ = [
    "Agent", "Model", "Reply", "Call", "ModelError", "resolve",
    "Tool", "tool", "Image", "CODING_TOOLS", "shell", "read_file", "write_file", "edit_file",
    "Channel", "Event", "every", "at",
    "telegram", "whatsapp", "http", "voice", "discord", "slack", "email",
    "mcp", "app", "browser", "computer", "rules", "sandbox",
]
__version__ = "0.2.0.dev0"
