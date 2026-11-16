"""An agent that acts on its own and tells you on Telegram."""

import os
import shutil

from nanoagent import Agent, at, every, telegram, tool


@tool
def disk_usage() -> str:
    """How full the disk is."""
    total, used, free = shutil.disk_usage("/")
    return f"{used / total:.0%} used, {free // 2**30} GB free"


agent = Agent("You watch this computer for its owner.", tools=[disk_usage],
              model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"), state="~/.nanoagent/watcher")
agent.use(telegram(), notify=True, quiet=("23:00", "07:00"))
agent.use(every("1h", "Check the disk. Only if it is over 90% full, say so; otherwise answer 'fine'."))
agent.use(at("mon 09:00", "Report how full the disk is."))
agent.live()
