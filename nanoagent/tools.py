"""Tools: plain Python functions the model is allowed to call."""

import inspect
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
PYTHON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    function: Callable
    approve: bool = False  # ask the agent's `approve` callback before running

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @property
    def spec(self):
        """The tool as the API wants it."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
            # Inputs stream as they are written instead of being buffered and
            # validated server-side. `check` below does the validation.
            "eager_input_streaming": True,
        }

    def check(self, arguments):
        """Return what is wrong with a model-written input, or None."""
        if not isinstance(arguments, dict):
            return "the input must be an object"
        known = self.schema["properties"]
        problems = [f"missing `{name}`" for name in self.schema["required"] if name not in arguments]
        for name, value in arguments.items():
            if name not in known:
                problems.append(f"unknown `{name}`")
                continue
            kind = known[name]["type"]
            is_bool = isinstance(value, bool)  # bool is an int in Python
            if not isinstance(value, PYTHON_TYPES[kind]) or (is_bool and kind != "boolean"):
                problems.append(f"`{name}` must be a {kind}")
        return ", ".join(problems) or None


def tool(function=None, *, approve=False):
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
        return Tool(function.__name__, inspect.getdoc(function) or "", schema, function, approve)

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
