"""Issue #22: control an app by name."""

import pytest

from nanoagent import cli
from nanoagent.apps import APPS, app


@pytest.mark.parametrize("name", sorted(APPS))
def test_app_resolves_to_its_command(name, monkeypatch):
    for variable in APPS[name].env:
        monkeypatch.setenv(variable, "value")
    server = app(name)
    assert server.words == APPS[name].command
    assert server.env == {variable: "value" for variable in APPS[name].env}


def test_arguments_are_passed_to_the_server(tmp_path):
    server = app("filesystem", str(tmp_path))
    assert server.words[-1] == str(tmp_path)


def test_missing_variable_is_explained(monkeypatch):
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError) as caught:
        app("github")
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in str(caught.value)
    assert "https://github.com/settings/personal-access-tokens" in str(caught.value)


def test_variable_from_env_argument(monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    assert app("notion", env={"NOTION_TOKEN": "secret"}).env == {"NOTION_TOKEN": "secret"}


def test_nothing_starts_until_used():
    server = app("time")
    assert server.process is None


def test_unknown_app_suggests_close_names():
    with pytest.raises(ValueError, match="Did you mean github"):
        app("gitub")


def test_registry_entries_are_complete():
    for entry in APPS.values():
        assert entry.command and entry.description and entry.maintainer
        assert entry.source.startswith("https://github.com/")
        if entry.env:
            assert entry.setup.startswith("https://")


def test_apps_command_lists_registry(capsys):
    cli.main(argv=["apps"])
    out = capsys.readouterr().out
    for name in APPS:
        assert name in out
    assert "needs GITHUB_PERSONAL_ACCESS_TOKEN" in out
