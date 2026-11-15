"""Safety: rules instead of a question for every command, and a sandbox.

    agent.use(rules(allow=["git status", "git diff *", "ls *"], deny=["rm *", "curl *"]))
    agent.use(sandbox(root="."))          # file tools cannot leave this folder
    agent.use(sandbox("docker"))          # shell commands run in a container, no network

Rules apply before approval: an allowed command runs without asking, a
denied one never runs (even if approved), anything else is asked as usual.
A chained command is allowed only if every part is, and denied if any part
is: `git status && rm -rf /` is not allowed by a rule for `git status`.
Commands with substitutions ($(...) or backticks) are always asked.
Patterns use shell-style wildcards. A pattern that names a tool
(`allow=["write_file"]`) applies to that whole tool.
"""

import dataclasses
import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

SEPARATORS = re.compile(r"\|\||&&|[;|&\n]")


def parts(command):
    """The simple commands in a command line, or None if it hides commands inside others."""
    if "$(" in command or "`" in command or "<(" in command:
        return None
    return [" ".join(part.split()) for part in SEPARATORS.split(command) if part.strip()]


class Rules:
    def __init__(self, allow=(), deny=()):
        self.allow, self.deny = list(allow), list(deny)
        self.agent = None

    def attach(self, agent):
        self.agent = agent
        agent.guards.append(self)

    def __call__(self, name, arguments):
        tools = set(self.agent.tools) if self.agent else set()
        if name in self.deny:
            return "deny"
        if name in self.allow and name in tools and name != "shell":
            return "allow"
        if name != "shell" or "command" not in arguments:
            return None
        commands = parts(arguments["command"])
        if commands is None:
            return "deny" if self._matches(arguments["command"], self.deny) else None
        if any(self._matches(command, self.deny) for command in commands):
            return "deny"
        if commands and all(self._matches(command, self.allow) for command in commands):
            return "allow"
        return None

    @staticmethod
    def _matches(command, patterns):
        command = " ".join(command.split())
        return any(fnmatch.fnmatchcase(command, " ".join(pattern.split())) for pattern in patterns)


class Sandbox:
    def __init__(self, kind=None, root=".", network=False, image="python:3.12-slim"):
        if kind not in (None, "docker"):
            raise ValueError("sandbox() supports kind=None (files only) or 'docker'")
        self.kind, self.root, self.network, self.image = kind, Path(root).expanduser().resolve(), network, image

    def attach(self, agent):
        if self.kind == "docker" and not shutil.which("docker"):
            raise RuntimeError("sandbox('docker') needs Docker installed and running.")
        for name, wrapper in (("read_file", self._confined), ("write_file", self._confined),
                              ("edit_file", self._confined), ("shell", self._contained)):
            if name in agent.tools and (name != "shell" or self.kind == "docker"):
                # A copy: the built-in tools are shared by every agent.
                agent.tools[name] = dataclasses.replace(agent.tools[name], function=wrapper(agent.tools[name].function))

    def inside(self, path):
        """The resolved path, if it is inside the root; symlinks are followed first."""
        target = (self.root / Path(path).expanduser()).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"{path} is outside the sandbox ({self.root}).")
        return target

    def _confined(self, function):
        def confined(path, **rest):
            return function(str(self.inside(path)), **rest)

        confined.__name__ = function.__name__
        return confined

    def _contained(self, function):
        def contained(command):
            network = [] if self.network else ["--network", "none"]
            docker = ["docker", "run", "--rm", "-i", "-v", f"{self.root}:/work", "-w", "/work", *network,
                      self.image, "sh", "-c", command]
            done = subprocess.run(docker, capture_output=True, text=True, timeout=300)
            return (done.stdout + done.stderr).strip()

        return contained


def rules(allow=(), deny=()):
    """Allow and deny rules for commands and tools. See the module docstring."""
    return Rules(allow, deny)


def sandbox(kind=None, *, root=".", network=False, image="python:3.12-slim"):
    """Confine file tools to `root`; with "docker", run shell commands in a container. See the module docstring."""
    return Sandbox(kind, root, network, image)

