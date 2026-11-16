"""Chapter 8 - Any model.

Chapters 1-7 talked to one provider. Most other servers, including every
local runtime (Ollama, llama.cpp, LM Studio, vLLM), speak one format: Chat
Completions. This chapter talks to it with nothing but the standard library,
and handles what small models get wrong:

    - many have no tool support at all: the server says so, and the agent
      switches to "guided" mode, where the model answers each step with one
      JSON object, {"tool": ..., "arguments": ...} or {"answer": ...},
      enforced by a JSON schema where the server supports it
    - some write the call as text instead of using the tool format
    - many wrap their reasoning in <think> tags, which the user should not see

    AGENT_MODEL_URL=http://localhost:11434/v1 AGENT_MODEL=qwen3:4b python tutorial/08_any_model.py
"""

import json
import os
import re
import subprocess
import urllib.error
import urllib.request

URL = os.environ.get("AGENT_MODEL_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("AGENT_MODEL", "qwen3:4b")
SYSTEM = f"You are a coding agent working in {os.getcwd()}. Use your tools, then say briefly what you did."

TOOLS = {
    "shell": {
        "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }
}


def shell(command):
    done = subprocess.run(command, shell=True, capture_output=True, text=True)
    return (done.stdout + done.stderr).strip() or "(no output)"


def post(body):
    request = urllib.request.Request(f"{URL}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)["choices"][0]["message"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from None


def visible(text):
    """The text without the model's reasoning."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


# --- native tool calling ---------------------------------------------------------


def native_step(messages):
    tools = [{"type": "function", "function": {"name": name, **spec}} for name, spec in TOOLS.items()]
    message = post({"model": MODEL, "messages": messages, "tools": tools})
    calls = [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
             for c in message.get("tool_calls") or []]
    text = visible(message.get("content"))
    if not calls and text.startswith("{"):  # the call written as text
        try:
            found = json.loads(text)
            if found.get("name") in TOOLS:
                calls, text = [("call_text", found["name"], found.get("arguments", {}))], ""
        except ValueError:
            pass
    return text, calls


# --- guided mode -----------------------------------------------------------------------

STEP_RULES = (
    "Reply to every message with exactly one JSON object. To use a tool: "
    '{"tool": "<name>", "arguments": {...}}. To reply to the user: {"answer": "..."}. Tools:\n'
    + "\n".join(f"- {name}: {spec['description']} Arguments: {json.dumps(spec['parameters']['properties'])}"
                for name, spec in TOOLS.items())
)
STEP_SCHEMA = {"anyOf": [
    *({"type": "object", "properties": {"tool": {"enum": [name]}, "arguments": spec["parameters"]},
       "required": ["tool", "arguments"]} for name, spec in TOOLS.items()),
    {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
]}


def guided_step(messages):
    # The model sees no tool roles: earlier steps are JSON text, results are user messages.
    chat = [{"role": "system", "content": messages[0]["content"] + "\n\n" + STEP_RULES}]
    for message in messages[1:]:
        if message["role"] == "tool":
            chat.append({"role": "user", "content": f"Result:\n{message['content']}"})
        elif message.get("tool_calls"):
            call = message["tool_calls"][0]["function"]
            chat.append({"role": "assistant", "content": json.dumps({"tool": call["name"], "arguments": json.loads(call["arguments"])})})
        else:
            chat.append({"role": message["role"], "content": message["content"]})
    shape = {"type": "json_schema", "json_schema": {"name": "step", "schema": STEP_SCHEMA}}
    text = visible(post({"model": MODEL, "messages": chat, "response_format": shape}).get("content"))
    try:
        step = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        return text, []  # not JSON at all: treat it as the answer
    if "answer" in step:
        return step["answer"], []
    return "", [("call_guided", step.get("tool"), step.get("arguments") or {})]


# --- the loop, unchanged in spirit since chapter 1 ----------------------------------------

mode = {"guided": False}


def run(messages):
    while True:
        try:
            text, calls = (guided_step if mode["guided"] else native_step)(messages)
        except RuntimeError as error:
            if "support tools" in str(error) or "--jinja" in str(error):
                mode["guided"] = True  # this model has no tool support: guide it
                continue
            raise
        if text:
            print(text)
        if not calls:
            messages.append({"role": "assistant", "content": text})
            return text
        messages.append({"role": "assistant", "content": text or None, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, n, a in calls]})
        for call_id, name, arguments in calls:
            print(f"  [{name}] {arguments}")
            output = shell(**arguments) if name in TOOLS else f"There is no tool named {name}."
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})


if __name__ == "__main__":
    messages = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            request = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        messages.append({"role": "user", "content": request})
        run(messages)
