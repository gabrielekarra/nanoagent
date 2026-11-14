"""Chapter 2 - Tools.

A shell is enough to do anything, but purpose-built tools are easier for the
model to use correctly and easier for you to control. This chapter adds:

    - a `@tool` decorator, so a tool is just a Python function
    - file tools: read, write, and edit by exact replacement
    - error handling: a failed tool is reported to the model, not raised

The last point matters most. When a tool fails, the model reads the error
and tries something else. Errors are information, not crashes.
"""

import json
import os
import subprocess
from pathlib import Path

import anthropic

client = anthropic.Anthropic()
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5-5")
SYSTEM = (
    f"You are a coding agent working in {os.getcwd()}. "
    "Use your tools to do what the user asks, then say briefly what you did."
)

TOOLS = []  # the schemas sent to the model
HANDLERS = {}  # name -> the function that does the work


def tool(description, **params):
    """Register a function as a tool. Every parameter is a required string."""

    def register(function):
        TOOLS.append(
            {
                "name": function.__name__,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        name: {"type": "string", "description": about}
                        for name, about in params.items()
                    },
                    "required": list(params),
                },
            }
        )
        HANDLERS[function.__name__] = function
        return function

    return register


@tool("Run a shell command and return its output.", command="The command to run")
def shell(command):
    done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
    return (done.stdout + done.stderr).strip()


@tool("Read a text file.", path="Path of the file")
def read_file(path):
    return Path(path).read_text()


@tool(
    "Create a file, or overwrite it if it exists.",
    path="Path of the file",
    content="The complete new content",
)
def write_file(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
    return f"Wrote {path}"


@tool(
    "Replace one snippet of a file. `old` must match exactly and appear once.",
    path="Path of the file",
    old="The exact text to replace",
    new="The text to put in its place",
)
def edit_file(path, old, new):
    text = Path(path).read_text()
    if text.count(old) != 1:
        raise ValueError(f"`old` appears {text.count(old)} times in {path}, expected 1")
    Path(path).write_text(text.replace(old, new))
    return f"Edited {path}"


def execute(call):
    """Run one tool call and package the outcome as a `tool_result`."""
    print(f"  [{call.name}] {json.dumps(call.input)[:100]}")
    try:
        output, failed = str(HANDLERS[call.name](**call.input)), False
    except Exception as error:  # unknown tool, bad arguments, tool failure
        output, failed = f"{type(error).__name__}: {error}", True
    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": output or "(no output)",
        "is_error": failed,
    }


def run(messages):
    """Call the model until it stops asking for tools."""
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(block.text)

        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            return
        messages.append({"role": "user", "content": [execute(call) for call in calls]})


if __name__ == "__main__":
    messages = []
    while True:
        try:
            request = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        messages.append({"role": "user", "content": request})
        run(messages)
