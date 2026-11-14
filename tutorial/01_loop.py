"""Chapter 1 - The loop.

An agent is a model, a tool, and a while loop:

    1. send the conversation to the model
    2. if it asked to use a tool, run the tool and append the result
    3. repeat until it stops asking

That is the whole idea. Everything in the later chapters makes this loop
safer, cheaper or more capable, but it never changes its shape.
"""

import os
import subprocess

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5-5")
SYSTEM = (
    f"You are a coding agent working in {os.getcwd()}. "
    "Use your tools to do what the user asks, then say briefly what you did."
)

# A tool is a name, a description and a JSON schema. The model never runs
# anything: it only asks, by returning a `tool_use` block that fits the schema.
TOOLS = [
    {
        "name": "shell",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


def shell(command):
    done = subprocess.run(command, shell=True, capture_output=True, text=True)
    return (done.stdout + done.stderr).strip() or "(no output)"


def run(messages):
    """Call the model until it stops asking for tools."""
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages
        )
        # Append the reply exactly as it came back. It holds more than text
        # (tool calls, the model's reasoning) and the API expects it unchanged.
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(block.text)

        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            return  # no tool requested: the model is done

        # Every tool call gets one result, all in a single user message.
        results = []
        for call in calls:
            print(f"  $ {call.input['command']}")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": shell(call.input["command"]),
                }
            )
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    messages = []
    while True:
        try:
            request = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        messages.append({"role": "user", "content": request})
        run(messages)
