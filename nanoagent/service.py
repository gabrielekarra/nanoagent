"""Keep an agent running: start it at login, restart it if it crashes.

    nanoagent run jarvis.py                        # run a file that calls agent.live()
    nanoagent service install jarvis.py            # systemd (Linux) or launchd (macOS)
    nanoagent service install jarvis.py --env TELEGRAM_BOT_TOKEN=...   # secrets, kept private
    nanoagent service status|logs|uninstall jarvis.py
    nanoagent service install jarvis.py --dry-run  # print the files instead of writing them

Secrets go to an environment file readable only by you, loaded by
`nanoagent run --env-file`, so they never appear in the service definition.
"""

import os
import platform as host_platform
import re
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

MARKER = "Managed by nanoagent"


def run_file(script, env_file=None):
    """Run a Python file as `__main__`, after loading an environment file if given."""
    if env_file:
        os.environ.update(read_env(Path(env_file)))
    script = Path(script).resolve()
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


def read_env(path):
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


class Service:
    """Where the files for one agent script go, and what they contain."""

    def __init__(self, script, *, system=None, home=None, python=None):
        self.script = Path(script).expanduser().resolve() if home is None else Path(script)
        self.name = re.sub(r"[^A-Za-z0-9_-]", "-", self.script.stem)
        self.system = system or ("macos" if host_platform.system() == "Darwin" else "linux")
        if self.system not in ("linux", "macos"):
            raise ValueError("system must be 'linux' or 'macos'")
        self.home = Path(home) if home else Path.home()
        self.python = python or sys.executable
        self.env_file = self.home / ".config" / "nanoagent" / f"{self.name}.env"
        if self.system == "linux":
            self.file = self.home / ".config" / "systemd" / "user" / f"nanoagent-{self.name}.service"
        else:
            self.file = self.home / "Library" / "LaunchAgents" / f"dev.nanoagent.{self.name}.plist"
        self.log = self.home / "Library" / "Logs" / f"nanoagent-{self.name}.log"

    def command(self):
        return [self.python, "-m", "nanoagent", "run", "--env-file", str(self.env_file), str(self.script)]

    def definition(self):
        if self.system == "linux":
            return f"""# {MARKER}. Remove with: nanoagent service uninstall {self.script}
[Unit]
Description=nanoagent: {self.name}
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={shlex.join(self.command())}
WorkingDirectory={self.script.parent}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
        arguments = "\n".join(f"        <string>{part}</string>" for part in self.command())
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- {MARKER}. Remove with: nanoagent service uninstall {self.script} -->
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>dev.nanoagent.{self.name}</string>
    <key>ProgramArguments</key>
    <array>
{arguments}
    </array>
    <key>WorkingDirectory</key>
    <string>{self.script.parent}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{self.log}</string>
    <key>StandardErrorPath</key>
    <string>{self.log}</string>
</dict>
</plist>
"""

    # --- actions ------------------------------------------------------------------

    def install(self, env=None, dry_run=False, run=subprocess.run):
        if not self.script.exists():
            raise FileNotFoundError(f"No such script: {self.script}")
        if dry_run:
            return self.definition()
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(self.definition())
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        existing = read_env(self.env_file) if self.env_file.exists() else {}
        existing.update(env or {})
        self.env_file.touch(mode=0o600)
        os.chmod(self.env_file, 0o600)  # secrets: readable only by you
        self.env_file.write_text(f"# {MARKER}\n" + "".join(f"{k}={v}\n" for k, v in existing.items()))
        if self.system == "linux":
            run(["systemctl", "--user", "daemon-reload"], check=False)
            run(["systemctl", "--user", "enable", "--now", self.file.name], check=False)
            note = "To keep it running when you are logged out: loginctl enable-linger"
        else:
            run(["launchctl", "load", "-w", str(self.file)], check=False)
            note = f"Logs: {self.log}"
        return f"Installed {self.file}\n{note}"

    def uninstall(self, run=subprocess.run):
        removed = []
        if self.system == "linux":
            run(["systemctl", "--user", "disable", "--now", self.file.name], check=False)
        elif self.file.exists():
            run(["launchctl", "unload", "-w", str(self.file)], check=False)
        for path in (self.file, self.env_file):
            if path.exists() and MARKER in path.read_text():  # never remove a file we did not write
                path.unlink()
                removed.append(path)
        return removed

    def status_command(self):
        if self.system == "linux":
            return ["systemctl", "--user", "status", self.file.name]
        return ["launchctl", "list", f"dev.nanoagent.{self.name}"]

    def logs_command(self):
        if self.system == "linux":
            return ["journalctl", "--user", "-u", self.file.name, "-n", "200", "--no-pager"]
        return ["tail", "-n", "200", str(self.log)]
