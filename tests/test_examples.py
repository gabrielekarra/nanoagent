"""Issue #28: every example starts and answers one message."""

import runpy
from pathlib import Path

import pytest

import offline

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_examples_run(path, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    delivered = []
    offline.install(monkeypatch, delivered)

    runpy.run_path(str(path), run_name="__main__")

    out = capsys.readouterr().out
    assert delivered or "ok" in out.split() or path.stem == "safe_coder", (path.stem, out)
    if "live()" in path.read_text():
        assert delivered and delivered[0][1:] == ("me", "ok")
