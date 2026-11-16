"""Issue #25: approval rules and a sandbox."""

import os
import shutil
from pathlib import Path

import pytest

from nanoagent import CODING_TOOLS, Agent, read_file, rules, sandbox
from nanoagent.safety import parts
from nanoagent.testing import FakeModel, call, reply, text


def run(commands, guard, approve=None, tools=CODING_TOOLS, extra=()):
    asked = []

    def asking(name, arguments):
        asked.append(arguments.get("command", name))
        return approve(name, arguments) if approve else False

    script = [reply(call("shell", command=command)) for command in commands] + [reply(text("ok"))]
    model = FakeModel(*script)
    agent = Agent(tools=tools, model=model, approve=asking, parallel=1)
    agent.use(guard)
    for each in extra:
        agent.use(each)
    agent.run("go")
    results = [model.tool_results(n + 1)[0] for n in range(len(commands))]
    return asked, results


def test_allow_rule_skips_the_question():
    asked, results = run(["echo hi", "echo  hi"], rules(allow=["echo *"]))
    assert asked == [] and [r["content"] for r in results] == ["hi", "hi"]


def test_other_commands_are_still_asked():
    asked, results = run(["echo hi", "date"], rules(allow=["echo *"]))
    assert asked == ["date"] and "declined" in results[1]["content"]


def test_deny_rule_wins_over_allow_and_approval():
    asked, results = run(["rm -rf build"], rules(allow=["rm *"], deny=["rm -rf *"]), approve=lambda *a: True)
    assert asked == [] and "rule set by the user blocks" in results[0]["content"]


@pytest.mark.parametrize("command", [
    "git status && rm -rf /", "git status; rm -rf /", "git status | rm -rf /", "git status || rm -rf /",
    "git status & rm -rf /", "git status\nrm -rf /",
])
def test_rules_see_through_chaining(command):
    guard = rules(allow=["git status"], deny=["rm *"])
    guard.agent = None
    assert guard("shell", {"command": command}) == "deny"
    loose = rules(allow=["git status"])
    assert loose("shell", {"command": command}) is None  # asked, not allowed


def test_substitutions_are_always_asked():
    guard = rules(allow=["echo *"])
    assert guard("shell", {"command": "echo $(cat ~/.ssh/id_rsa)"}) is None
    assert guard("shell", {"command": "echo `whoami`"}) is None
    assert parts("a && b; c") == ["a", "b", "c"]


def test_tool_rules():
    model = FakeModel(reply(call("write_file", path="a.txt", content="x")), reply(text("ok")))
    agent = Agent(tools=CODING_TOOLS, model=model, approve=lambda *a: False)
    agent.use(rules(allow=["write_file"]))
    agent.run("write")
    assert model.tool_results()[0]["content"].startswith("Wrote")


def test_file_tools_cannot_leave_the_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "inside.txt").write_text("fine")
    (tmp_path / "secret.txt").write_text("nope")
    os.symlink(tmp_path / "secret.txt", root / "link.txt")
    model = FakeModel(
        reply(call("read_file", path="inside.txt")),
        reply(call("read_file", path="../secret.txt")),
        reply(call("read_file", path="/etc/passwd")),
        reply(call("read_file", path="link.txt")),
        reply(call("write_file", path="../escape.txt", content="x")),
        reply(text("ok")),
    )
    agent = Agent(tools=CODING_TOOLS, model=model, parallel=1)
    agent.use(sandbox(root=root))
    agent.run("read")
    results = [model.tool_results(n)[0] for n in range(1, 6)]
    assert results[0]["content"] == "fine"
    assert all(r["error"] and "outside the sandbox" in r["content"] for r in results[1:])
    assert not (tmp_path / "escape.txt").exists()


def test_sandbox_does_not_change_other_agents(tmp_path):
    agent = Agent(tools=[read_file], model=FakeModel())
    agent.use(sandbox(root=tmp_path))
    assert agent.tools["read_file"].function is not read_file.function


def docker_running():
    import subprocess

    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


docker = pytest.mark.skipif(not docker_running(), reason="needs Docker running")


@pytest.mark.docker
@docker
def test_sandboxed_shell_cannot_see_home(tmp_path):
    model = FakeModel(reply(call("shell", command=f"ls {Path.home()} ; pwd")), reply(text("ok")))
    agent = Agent(tools=CODING_TOOLS, model=model)
    agent.use(sandbox("docker", root=tmp_path))
    agent.run("look")
    assert "/work" in model.tool_results()[0]["content"]
    assert "No such file" in model.tool_results()[0]["content"]


@pytest.mark.docker
@docker
def test_sandboxed_shell_has_no_network_by_default(tmp_path):
    model = FakeModel(reply(call("shell", command="python -c \"import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)\"")),
                      reply(text("ok")))
    agent = Agent(tools=CODING_TOOLS, model=model)
    agent.use(sandbox("docker", root=tmp_path))
    agent.run("fetch")
    assert "Error" in model.tool_results()[0]["content"]


def test_docker_sandbox_needs_docker(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="needs Docker"):
        Agent(tools=CODING_TOOLS, model=FakeModel()).use(sandbox("docker"))
