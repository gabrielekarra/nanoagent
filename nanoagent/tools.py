"""Tools: plain Python functions the model is allowed to call."""

import inspect
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
PYTHON_TYPES = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "object": dict, "array": list, "null": type(None),
}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    function: Callable  # a plain or an `async def` function
    approve: bool = False  # ask before running (the agent's `approve` callback, or the channel)
    parallel: bool = True  # may run at the same time as other calls of the same reply

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def check(self, arguments):
        """Return what is wrong with a model-written input, or None."""
        if not isinstance(arguments, dict):
            return "the input must be an object"
        return ", ".join(validate(arguments, self.schema)) or None


def validate(value, schema, path=None):
    """The problems of `value` against the parts of JSON Schema tools use."""
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds] if kinds else []
    if kinds and not any(_is(value, kind) for kind in kinds):
        return [f"`{path}` must be a {' or '.join(kinds)}"] if path else ["the input must be an object"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"`{path}` must be one of {', '.join(map(str, schema['enum']))}"]
    problems = []
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        problems += [f"missing `{_join(path, name)}`" for name in schema.get("required", []) if name not in value]
        for name, item in value.items():
            if name in properties:
                problems += validate(item, properties[name], _join(path, name))
            elif properties and schema.get("additionalProperties") is not True and not isinstance(
                schema.get("additionalProperties"), dict
            ):
                problems.append(f"unknown `{_join(path, name)}`")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            problems += validate(item, schema["items"], f"{path or 'input'}[{index}]")
    return problems


def _is(value, kind):
    if kind not in PYTHON_TYPES:
        return True
    if isinstance(value, bool) and kind in ("integer", "number"):
        return False  # bool is an int in Python, but not in JSON
    return isinstance(value, PYTHON_TYPES[kind])


def _join(path, name):
    return f"{path}.{name}" if path else name


def tool(function=None, *, approve=False, parallel=True):
    """Turn a function into a tool.

    The name, the description and the schema come from the function itself:
    its name, its docstring and its type hints. Parameters can be `str`,
    `int`, `float` or `bool`; one without a default is required. Describe a
    parameter with `Annotated[str, "what it is"]`.

        @tool
        def get_weather(city: str) -> str:
            "Return today's weather in a city."

        @tool(approve=True)   # the agent asks before running this one
        def delete_file(path: str) -> str: ...

        @tool
        async def fetch(url: str) -> str: ...   # async functions work too

    `parallel=False` keeps a tool from running at the same time as others.
    """

    def build(function):
        hints = typing.get_type_hints(function, include_extras=True)
        properties, required = {}, []
        for name, parameter in inspect.signature(function).parameters.items():
            hint, about = hints.get(name, str), None
            if typing.get_origin(hint) is Annotated:
                hint, about = typing.get_args(hint)[:2]
            if hint not in JSON_TYPES:
                raise TypeError(f"{function.__name__}({name}): only str, int, float and bool are supported")
            properties[name] = {"type": JSON_TYPES[hint]}
            if about:
                properties[name]["description"] = about
            if parameter.default is parameter.empty:
                required.append(name)
        schema = {"type": "object", "properties": properties, "required": required}
        return Tool(function.__name__, inspect.getdoc(function) or "", schema, function, approve, parallel)

    return build(function) if function else build


# --- a ready-made set for coding agents --------------------------------------


@tool(approve=True)
def shell(command: Annotated[str, "The command to run"]) -> str:
    """Run a shell command and return its output."""
    done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
    return (done.stdout + done.stderr).strip()


@tool
def read_file(path: Annotated[str, "Path of the file"]) -> str:
    """Read a text file."""
    return Path(path).read_text()


@tool(approve=True)
def write_file(
    path: Annotated[str, "Path of the file"],
    content: Annotated[str, "The complete new content"],
) -> str:
    """Create a file, or overwrite it if it exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
    return f"Wrote {path}"


@tool(approve=True)
def edit_file(
    path: Annotated[str, "Path of the file"],
    old: Annotated[str, "The exact text to replace"],
    new: Annotated[str, "The text to put in its place"],
) -> str:
    """Replace one snippet of a file. `old` must match exactly and appear once."""
    text = Path(path).read_text()
    if text.count(old) != 1:
        raise ValueError(f"`old` appears {text.count(old)} times in {path}, expected 1")
    Path(path).write_text(text.replace(old, new))
    return f"Edited {path}"


CODING_TOOLS = [shell, read_file, write_file, edit_file]
