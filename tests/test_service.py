"""Issue #15: start at login, restart on crash."""

import os
import stat
from pathlib import Path

import pytest

from nanoagent import cli
from nanoagent.service import Service, read_env, run_file

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("system, expected", [("linux", "jarvis.service"), ("macos", "jarvis.plist")])
def test_service_file_matches_example(system, expected):
    service = Service("/home/ada/agents/jarvis.py", system=system, home="/home/ada", python="/usr/bin/python3")
    assert service.definition() == (DATA / expected).read_text()


def test_dry_run_prints_without_writing(tmp_path, capsys):
    script = tmp_path / "jarvis.py"
    script.write_text("print('hi')\n")

    cli.main(argv=["service", "install", str(script), "--dry-run", "--system", "linux"])

    out = capsys.readouterr().out
    assert out.startswith("# Managed by nanoagent")
    assert f"{script}" in out and "-m nanoagent run --env-file" in out


def test_install_writes_files_and_private_env(tmp_path):
    script = tmp_path / "jarvis.py"
    script.write_text("")
    commands = []
    service = Service(script, system="linux", home=tmp_path / "home", python="/usr/bin/python3")

    service.install({"TELEGRAM_BOT_TOKEN": "123:abc"}, run=lambda command, check: commands.append(command))

    assert service.file.read_text() == service.definition()
    assert stat.S_IMODE(os.stat(service.env_file).st_mode) == 0o600
    assert read_env(service.env_file) == {"TELEGRAM_BOT_TOKEN": "123:abc"}
    assert "123:abc" not in service.file.read_text()
    assert ["systemctl", "--user", "enable", "--now", "nanoagent-jarvis.service"] in commands


def test_install_keeps_earlier_variables(tmp_path):
    script = tmp_path / "jarvis.py"
    script.write_text("")
    service = Service(script, system="macos", home=tmp_path, python="/usr/bin/python3")
    quiet = lambda command, check: None  # noqa: E731

    service.install({"A": "1"}, run=quiet)
    service.install({"B": "2"}, run=quiet)

    assert read_env(service.env_file) == {"A": "1", "B": "2"}


def test_install_refuses_missing_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such script"):
        Service(tmp_path / "missing.py", system="linux", home=tmp_path).install(run=lambda *a, **k: None)


def test_uninstall_removes_only_its_own_files(tmp_path):
    script = tmp_path / "jarvis.py"
    script.write_text("")
    service = Service(script, system="linux", home=tmp_path, python="/usr/bin/python3")
    quiet = lambda command, check: None  # noqa: E731
    service.install(run=quiet)
    service.env_file.write_text("MINE=1\n")  # the user replaced it with their own file

    removed = service.uninstall(run=quiet)

    assert removed == [service.file]
    assert service.env_file.exists() and not service.file.exists()


def test_run_file_loads_env(tmp_path, monkeypatch):
    env = tmp_path / "agent.env"
    env.write_text("# comment\nGREETING=hello there\n")
    script = tmp_path / "agent.py"
    script.write_text("import os\nopen('out.txt', 'w').write(os.environ['GREETING'] + ' ' + __name__)\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GREETING", raising=False)

    run_file(script, env)

    assert Path("out.txt").read_text() == "hello there __main__"


def test_cli_run_command(tmp_path, monkeypatch):
    script = tmp_path / "agent.py"
    script.write_text("open('ran.txt', 'w').write('yes')\n")
    monkeypatch.chdir(tmp_path)

    cli.main(argv=["run", str(script)])

    assert Path("ran.txt").read_text() == "yes"
