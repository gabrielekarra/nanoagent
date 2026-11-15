"""Issue #28: the README runs as written, and its numbers are current."""

import re
from pathlib import Path

import offline

ROOT = Path(__file__).parent.parent
README = (ROOT / "README.md").read_text()


def lines(path):
    return len(path.read_text().splitlines())


def test_readme_examples_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    delivered = []
    offline.install(monkeypatch, delivered)
    namespace = {"__name__": "readme"}
    for number, block in enumerate(re.findall(r"```python\n(.*?)```", README, re.DOTALL), 1):
        try:
            exec(compile(block, f"README block {number}", "exec"), namespace)
        except Exception as error:
            raise AssertionError(f"README python block {number} failed: {type(error).__name__}: {error}\n{block}") from error
    assert delivered, "agent.live() in the README answered nothing"


def test_readme_line_counts_are_current():
    for name, count in re.findall(r"\| \[`(\d\d_\w+\.py)`\]\(tutorial/\1\) \| (\d+) \|", README):
        assert lines(ROOT / "tutorial" / name) == int(count), name
    chapters = sorted(path.name for path in (ROOT / "tutorial").glob("*.py"))
    assert len(re.findall(r"\| \[`\d\d_", README)) == len(chapters)

    core = sum(lines(ROOT / "nanoagent" / name) for name in ("agent.py", "model.py", "tools.py"))
    package = sum(lines(path) for path in (ROOT / "nanoagent").rglob("*.py"))
    assert f"is {core} lines." in README, f"the core is now {core} lines"
    assert f"is {package} lines of Python" in README, f"the package is now {package} lines"


def test_every_example_is_listed():
    for path in (ROOT / "examples").glob("*.py"):
        if path.stem != "jarvis":
            assert f"examples/{path.name}" in README, path.name
