"""Chapter 4 - Streaming.

Until now the agent was silent while the model wrote. This chapter streams
the reply so text appears as it is generated, and deals with what streaming
exposes:

    - tool inputs arrive as raw fragments that the API no longer validates,
      so the harness checks them against the schema before running anything
    - a reply can be cut off mid tool call when it hits the output limit;
      a cut-off call must never run
    - the model can decline a request, which is a stop reason, not an error
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
                # Stream this tool's input as it is written instead of
                # buffering it server-side. Faster, but unvalidated.
                "eager_input_streaming": True,
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


def valid(call):
    """Check a tool input against its schema: the right keys, all strings."""
    expected = next((t["input_schema"]["required"] for t in TOOLS if t["name"] == call.name), None)
    return (
        expected is not None
        and isinstance(call.input, dict)
        and set(call.input) == set(expected)
        and all(isinstance(value, str) for value in call.input.values())
    )


def execute(call, cut_off=False):
    """Run one tool call and package the outcome as a `tool_result`."""
    print(f"  [{call.name}] {json.dumps(call.input)[:100]}")
    if cut_off:
        return result(call, "Your reply hit the output limit before this call was complete, "
                            "so nothing ran. Try again with a smaller input.", True)
    if not valid(call):
        return result(call, f"Unknown tool or invalid input: {json.dumps(call.input)}", True)
    if not approved(call):
        return result(call, "The user declined this action. Ask them what to do instead.", True)
    try:
        return result(call, str(HANDLERS[call.name](**call.input)))
    except Exception as error:
        return result(call, f"{type(error).__name__}: {error}", True)


def ask(messages):
    """One model call. Text is printed while it is being generated."""
    for attempt in (1, 2):
        try:
            with client.messages.stream(
                model=MODEL, max_tokens=64000, system=SYSTEM, tools=TOOLS, messages=messages
            ) as stream:
                wrote = False
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    wrote = True
                if wrote:
                    print()
                return stream.get_final_message()
        except ValueError:  # a streamed tool input was not parseable JSON
            if attempt == 2:
                raise


def run(messages):
    """Call the model until it stops asking for tools."""
    while True:
        response = ask(messages)
        if response.content:  # a declined request can come back empty
            messages.append({"role": "assistant", "content": response.content})

        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            if response.stop_reason == "refusal":
                print("(the model declined this request)")
            return
        # Any stop reason other than "tool_use" means the reply ended early
        # and the calls in it may be incomplete.
        cut_off = response.stop_reason != "tool_use"
        results = [execute(call, cut_off) for call in calls]
        messages.append({"role": "user", "content": results})


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
