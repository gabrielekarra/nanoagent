"""Chapter 3 - Permissions.

The agent from chapter 2 runs any command the model asks for. This chapter
puts a human between the request and the action:

    - tools that change things ask for approval first
    - a refusal is sent back to the model as a result, so it can adapt
    - Ctrl-C stops the agent without corrupting the conversation

The model proposes, the harness decides. That boundary is the entire
safety model of every coding agent you have used.
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

TOOLS = []
HANDLERS = {}

# Tools that change the machine. Reading is free; these need a yes.
NEEDS_APPROVAL = {"shell", "write_file", "edit_file"}
# Tools the user approved for the rest of the session. AGENT_YOLO=1 skips asking.
ALWAYS_ALLOWED = set(NEEDS_APPROVAL) if os.environ.get("AGENT_YOLO") else set()


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


def approved(call):
    """Ask the user before a tool that changes things."""
    if call.name not in NEEDS_APPROVAL or call.name in ALWAYS_ALLOWED:
        return True
    answer = input("  allow? [y]es / [n]o / [a]lways for this tool: ").strip().lower()
    if answer.startswith("a"):
        ALWAYS_ALLOWED.add(call.name)
    return answer.startswith(("y", "a"))


def result(call, output, failed=False):
    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": output or "(no output)",
        "is_error": failed,
    }


def execute(call):
    """Run one tool call and package the outcome as a `tool_result`."""
    print(f"  [{call.name}] {json.dumps(call.input)[:100]}")
    if not approved(call):
        return result(call, "The user declined this action. Ask them what to do instead.", True)
    try:
        return result(call, str(HANDLERS[call.name](**call.input)))
    except Exception as error:
        return result(call, f"{type(error).__name__}: {error}", True)


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


def close_open_calls(messages):
    """After an interruption, answer any tool call that never got a result.

    The API rejects a conversation where a `tool_use` has no `tool_result`,
    so an interrupted turn has to be closed before the next one starts.
    """
    last = messages[-1]
    if last["role"] == "assistant":
        calls = [block for block in last["content"] if block.type == "tool_use"]
        if calls:
            interrupted = [result(call, "Interrupted by the user.", True) for call in calls]
            messages.append({"role": "user", "content": interrupted})


if __name__ == "__main__":
    messages = []
    while True:
        try:
            request = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        messages.append({"role": "user", "content": request})
        try:
            run(messages)
        except KeyboardInterrupt:  # Ctrl-C stops the agent, not the program
            close_open_calls(messages)
            print("\n(interrupted)")
