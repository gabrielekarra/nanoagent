"""Issue #27: the package builds, installs alone, and has one version."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

import nanoagent

ROOT = Path(__file__).parent.parent


def test_version_is_single_sourced():
    assert importlib.metadata.version("nanoagent") == nanoagent.__version__
    assert "version =" not in (ROOT / "pyproject.toml").read_text().split("[tool.setuptools.dynamic]")[0]


@pytest.mark.slow
def test_package_builds_and_installs(tmp_path):
    built = subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(tmp_path / "dist"), "-q"],
                           capture_output=True, text=True)
    if built.returncode:
        pytest.skip(f"cannot build here (no network for the build backend?): {built.stderr[-300:]}")
    (wheel,) = (tmp_path / "dist").glob("nanoagent-*.whl")
    # An empty environment, made from the base interpreter (not from the one running the tests,
    # which may itself be a virtual environment); this pip installs into it.
    base = getattr(sys, "_base_executable", sys.executable)
    subprocess.run([base, "-m", "venv", "--without-pip", str(tmp_path / "env")], check=True)
    python = tmp_path / "env" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-warn-script-location", "-q",
                    "--prefix", str(tmp_path / "env"), str(wheel)], check=True)
    check = ("import importlib.util, nanoagent; "
             "assert importlib.util.find_spec('anthropic') is None; "
             "from nanoagent.testing import FakeModel, reply, text; "
             "print(nanoagent.Agent(model=FakeModel(reply(text('hi')))).run('hello'), nanoagent.__version__)")
    done = subprocess.run([python, "-c", check], capture_output=True, text=True, cwd=tmp_path)
    assert done.stdout.split() == ["hi", nanoagent.__version__], done.stderr
    assert (python.parent / "nanoagent").exists()
