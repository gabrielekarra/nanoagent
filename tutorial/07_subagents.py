"""Chapter 7 - Subagents.

A long task fills the context with details that stop mattering: the forty
files read to find one function, the logs scanned for one error. This chapter
lets the agent delegate:

    - a `task` tool starts a second agent with an empty conversation
    - that agent runs the same loop with the same tools, minus `task`
    - only its final report comes back, as an ordinary tool result

A subagent is not a new mechanism. It is the loop from chapter 1, called
from inside a tool.
"""

import json
import os
import subprocess
from pathlib import Path

import anthropic

client = anthropic.Anthropic()
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5-5")
MEMORY = Path("AGENT.md")
SYSTEM = (
    f"You are a coding agent working in {os.getcwd()}. "
    "Use your tools to do what the user asks, then say briefly what you did. "
    "When you learn something that will matter in later sessions (a build "
    "command, a convention, a preference of the user), save it with `remember`."
)
if MEMORY.exists():
    SYSTEM += f"\n\nNotes from earlier sessions in this project:\n{MEMORY.read_text()}"

MAX_OUTPUT = 20_000  # characters kept from one tool result
COMPACT_AT = int(os.environ.get("AGENT_COMPACT_AT", 150_000))  # tokens of context
COMPACT_PROMPT = (
    "Pause the task. Without calling any tool, write a handoff note for an agent "
    "that will continue this work with no access to the conversation above: the "
    "user's goal, what is done, what is left, and every file path, command and "
    "decision it needs."
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


@tool(
    "Save a lasting fact about this project or the user for future sessions.",
    note="One self-contained sentence",
)
def remember(note):
    with MEMORY.open("a") as file:
        file.write(f"- {note}\n")
    return f"Saved to {MEMORY}"


@tool(
    "Hand a self-contained piece of work to a fresh agent and get back its final "
    "report. Good for research and side tasks whose details you do not need. The "
    "agent sees nothing of this conversation: put everything it needs in the prompt.",
    prompt="The complete task description",
)
def task(prompt):
    workers = [spec for spec in TOOLS if spec["name"] != "task"]  # no recursion
    return run([{"role": "user", "content": prompt}], tools=workers, quiet=True)


def approved(call):
    """Ask the user before a tool that changes things."""
    if call.name not in NEEDS_APPROVAL or call.name in ALWAYS_ALLOWED:
        return True
    answer = input("  allow? [y]es / [n]o / [a]lways for this tool: ").strip().lower()
    if answer.startswith("a"):
        ALWAYS_ALLOWED.add(call.name)
    return answer.startswith(("y", "a"))


def clip(text):
    """Keep the head and tail of a long output; the middle is rarely needed."""
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return f"{text[:half]}\n[... {len(text) - MAX_OUTPUT} characters omitted ...]\n{text[-half:]}"


def result(call, output, failed=False):
    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": clip(output) or "(no output)",
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


def ask(messages, tools, quiet):
    """One model call. Text is printed while it is generated, unless quiet."""
    for attempt in (1, 2):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=64000,
                system=SYSTEM,
                tools=tools,
                messages=messages,
                # Cache everything up to the last message. The next step
                # reads it back at a fraction of the normal input price.
                cache_control={"type": "ephemeral"},
            ) as stream:
                wrote = False
                for text in stream.text_stream:
                    if not quiet:
                        print(text, end="", flush=True)
                        wrote = True
                if wrote:
                    print()
                return stream.get_final_message()
        except ValueError:  # a streamed tool input was not parseable JSON
            if attempt == 2:
                raise


def context_size(usage):
    """Tokens the conversation occupies after this reply."""
    return (
        usage.input_tokens
        + (usage.cache_read_input_tokens or 0)
        + (usage.cache_creation_input_tokens or 0)
        + usage.output_tokens
    )


def compact(messages, tools):
    """Replace the whole history with a handoff note written by the model."""
    reply = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        tools=tools,  # same system and tools as always, so the cache still hits
        messages=messages + [{"role": "user", "content": COMPACT_PROMPT}],
    )
    note = "".join(block.text for block in reply.content if block.type == "text")
    if note:
        print("  (context compacted)")
        messages[:] = [
            {
                "role": "user",
                "content": "This session continues earlier work. Handoff note from "
                f"that work:\n\n{note}\n\nCarry on from where it left off.",
            }
        ]


def run(messages, tools=TOOLS, quiet=False):
    """Call the model until it stops asking for tools. Returns its last words."""
    while True:
        response = ask(messages, tools, quiet)
        if response.content:  # a declined request can come back empty
            messages.append({"role": "assistant", "content": response.content})

        calls = [block for block in response.content if block.type == "tool_use"]
        if calls:
            # Any stop reason other than "tool_use" means the reply ended
            # early and the calls in it may be incomplete.
            cut_off = response.stop_reason != "tool_use"
            results = [execute(call, cut_off) for call in calls]
            messages.append({"role": "user", "content": results})
        elif response.stop_reason == "refusal":
            print("(the model declined this request)")

        # Compact only here, when every tool call has its result.
        if context_size(response.usage) > COMPACT_AT:
            compact(messages, tools)
        if not calls:
            return "".join(block.text for block in response.content if block.type == "text")


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
