import importlib.util
from pathlib import Path

import pytest

from nanoagent.testing import FakeModel

TUTORIAL = Path(__file__).parent.parent / "tutorial"
CHAPTERS = sorted(path.name for path in TUTORIAL.glob("0*.py"))


@pytest.fixture
def api():
    with FakeModel() as model:
        yield model


@pytest.fixture
def load(api, tmp_path, monkeypatch):
    """Import a tutorial chapter as a module, pointed at the fake model, in an empty directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
    monkeypatch.delenv("AGENT_YOLO", raising=False)

    def _load(filename, **env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        spec = importlib.util.spec_from_file_location(filename[:-3], TUTORIAL / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load
