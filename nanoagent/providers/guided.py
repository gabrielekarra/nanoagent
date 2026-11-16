"""Guided tool calling: make a model use tools without native tool support.

Small models often have no tool-calling template, or use it unreliably. In
guided mode nanoagent stops relying on it. The model answers every step with
one JSON object:

    {"thought": "I need the weather", "tool": "get_weather", "arguments": {"city": "Rome"}}
    {"thought": "I have it", "answer": "It is sunny in Rome."}

Where the server supports constrained decoding, the object is enforced by a
JSON schema built from the tools, so the model cannot produce anything else.
Whatever comes back is still parsed defensively, because some servers ignore
the schema and some models wrap the JSON in prose.

This module holds the parts every transport shares. A provider supplies
`complete(chat, schema) -> (text, context)`.
"""

import ast
import json
import re

RULES = """You work in steps. Reply to every message with exactly one JSON object and nothing else.

To use a tool:
{{"thought": "<what you need and why>", "tool": "<tool name>", "arguments": {{<the tool's arguments>}}}}

To reply to the user:
{{"thought": "<why you are done>", "answer": "<your reply>"}}

Use one tool per step. After each step you will see the tool's result. Answer only when you have what you need; if a tool fails, read the error and try something else.

Tools:
{tools}"""

CORRECTION = (
    "Your last reply was not one valid JSON object in the required format. "
    "Reply again with only the JSON object."
)


def instructions(system, tools):
    """The system prompt with the step rules and a compact list of the tools."""
    rules = RULES.format(tools="\n".join(describe(each) for each in tools))
    return f"{system}\n\n{rules}" if system else rules


def describe(tool):
    """`get_weather(city: string, days?: integer) - Return the weather.`"""
    properties, required = tool.schema.get("properties", {}), tool.schema.get("required", [])
    parameters = ", ".join(
        f"{name}{'' if name in required else '?'}: {spec.get('type', 'any')}" for name, spec in properties.items()
    )
    summary = (tool.description or "").strip().split("\n")[0]
    return f"- {tool.name}({parameters}) - {summary}"


def schema(tools):
    """The JSON schema of one step: one branch per tool, plus the answer."""
    branches = [
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "tool": {"type": "string", "enum": [each.name]},
                "arguments": each.schema,
            },
            "required": ["thought", "tool", "arguments"],
            "additionalProperties": False,
        }
        for each in tools
    ]
    branches.append({
        "type": "object",
        "properties": {"thought": {"type": "string"}, "answer": {"type": "string"}},
        "required": ["thought", "answer"],
        "additionalProperties": False,
    })
    return {"anyOf": branches}


def render(messages):
    """The conversation as plain chat turns, for a model that knows no tool roles."""
    chat = []
    for turn in messages:
        if turn["role"] == "user":
            chat.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "assistant":
            raw = turn.get("raw")
            steps = [raw["step"]] if isinstance(raw, dict) and "step" in raw else [
                {"thought": "", "tool": c["name"], "arguments": c["arguments"]} for c in turn.get("calls", [])
            ]
            if not steps and turn["content"]:
                steps = [{"thought": "", "answer": turn["content"]}]
            for step in steps:
                chat.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
        elif turn["role"] == "tool":
            chat.append({
                "role": "user",
                "content": "\n\n".join(
                    f"{'Error from' if r['error'] else 'Result of'} {r['name']}:\n{r['content']}"
                    for r in turn["results"]
                ),
            })
    return merge(chat)


def merge(chat):
    """Join consecutive turns of the same role. Many chat templates require alternation."""
    merged = []
    for turn in chat:
        if merged and merged[-1]["role"] == turn["role"] and turn["role"] != "tool" and not (
            merged[-1].get("tool_calls") or turn.get("tool_calls")
        ):
            merged[-1] = {**merged[-1], "content": f"{merged[-1]['content']}\n\n{turn['content']}"}
        else:
            merged.append(turn)
    return merged


def step(complete, system, tools, messages, on_text, ids):
    """Ask for one step, correct the model once if needed, and return a `Reply`."""
    from ..model import Call, Reply

    chat = [{"role": "system", "content": instructions(system, tools)}, *render(messages)]
    shape = schema(tools)
    text, context = complete(chat, shape)
    parsed = parse(text)
    if parsed is None:
        chat += [{"role": "assistant", "content": text}, {"role": "user", "content": CORRECTION}]
        text, context = complete(chat, shape)
        parsed = parse(text)
    if parsed is None:  # still no step: treat what it wrote as its answer
        answer = strip_thinking(text).strip()
        parsed = {"thought": "", "answer": answer}
    if "answer" in parsed:
        if on_text and parsed["answer"]:
            on_text(parsed["answer"])
        return Reply(text=parsed["answer"], stop="end", context=context, raw={"step": parsed})
    call = Call(f"call_{next(ids)}", parsed["tool"], parsed["arguments"])
    return Reply(calls=[call], stop="tool", context=context, raw={"step": parsed})


# --- reading whatever the model wrote ---------------------------------------

THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_thinking(text):
    text = THINKING.sub("", text)
    if "</think>" in text:  # the opening tag was part of the prompt
        text = text.split("</think>", 1)[1]
    return text


def parse(text, tools=None):
    """Find one step in a model's output, or None.

    Accepts the object alone, inside a code fence, surrounded by prose, with
    trailing commas or single quotes, after a <think> block, and under a few
    common alternative key names. With `tools`, only steps naming one of
    those tools (or answers) are accepted.
    """
    text = strip_thinking(text or "").strip()
    for candidate in _candidates(text):
        found = _normalize(_load(candidate))
        if found and (tools is None or "answer" in found or found["tool"] in tools):
            return found
    return None


def _candidates(text):
    yield text
    yield from FENCE.findall(text)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        yield text[start : end + 1]
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        yield json.dumps(obj)


def _load(candidate):
    candidate = candidate.strip()
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            return json.loads(attempt)
        except ValueError:
            pass
    try:  # single quotes and Python literals
        python = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", candidate)))
        return ast.literal_eval(python)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


TOOL_KEYS = ("tool", "name", "tool_name", "function", "action")
ARGUMENT_KEYS = ("arguments", "args", "parameters", "input", "action_input")


def _normalize(obj):
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("tool_call"), dict):
        obj = obj["tool_call"]
    thought = obj.get("thought") if isinstance(obj.get("thought"), str) else ""
    name = next((obj[key] for key in TOOL_KEYS if isinstance(obj.get(key), str)), None)
    if isinstance(obj.get("answer"), str) and not name:
        return {"thought": thought, "answer": obj["answer"]}
    if not name:
        return None
    arguments = next((obj[key] for key in ARGUMENT_KEYS if key in obj), {})
    if name.lower().replace("_", " ") == "final answer":
        return {"thought": thought, "answer": arguments if isinstance(arguments, str) else json.dumps(arguments)}
    if isinstance(arguments, str):
        loaded = _load(arguments) if arguments.strip() else {}
        arguments = loaded if isinstance(loaded, dict) else arguments
    if arguments is None:
        arguments = {}
    return {"thought": thought, "tool": name, "arguments": arguments}
