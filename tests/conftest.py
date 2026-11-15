import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeAnthropic  # noqa: E402

TUTORIAL = Path(__file__).parent.parent / "tutorial"
CHAPTERS = sorted(path.name for path in TUTORIAL.glob("0*.py") if int(path.name[:2]) <= 7)  # the Claude chapters


def pytest_configure(config):
    config.addinivalue_line("markers", "local: needs a real local model (set NANOAGENT_LOCAL_MODEL)")
    config.addinivalue_line("markers", "voice: needs the voice extra")
    config.addinivalue_line("markers", "docker: needs Docker")
    config.addinivalue_line("markers", "browser: needs Playwright and Chromium")
    config.addinivalue_line("markers", "slow: builds the package")


@pytest.fixture
def api():
    with FakeAnthropic() as server:
        yield server


@pytest.fixture
def load(api, tmp_path, monkeypatch):
    """Import a tutorial chapter as a module, pointed at the fake Anthropic API, in an empty directory."""
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
