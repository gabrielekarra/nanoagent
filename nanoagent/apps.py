"""Control an app by name: `agent.use(app("github"))`.

A small, reviewed list of MCP servers. Each entry names the command that
starts it, the variables it needs, where its source lives and who maintains
it. `nanoagent apps` prints the list. Entries are added by pull request; an
entry must point to the server's official source.

    agent.use(app("github"))                       # needs GITHUB_PERSONAL_ACCESS_TOKEN
    agent.use(app("filesystem", "~/notes"))        # extra arguments go to the server
    agent.use(app("fetch", approve=False))         # options go to mcp()
"""

import difflib
import os
from dataclasses import dataclass, field

from .mcp import mcp


@dataclass
class App:
    name: str
    description: str
    command: list
    source: str
    maintainer: str
    env: list = field(default_factory=list)  # variables the server needs
    setup: str = ""  # where to get them
    arguments: str = ""  # what extra arguments mean, if any


APPS = {
    each.name: each
    for each in [
        App("filesystem", "Read, write and search files in the folders you allow.",
            ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem", "Model Context Protocol",
            arguments="one or more folders the server may use"),
        App("fetch", "Fetch web pages and turn them into Markdown.", ["uvx", "mcp-server-fetch"],
            "https://github.com/modelcontextprotocol/servers/tree/main/src/fetch", "Model Context Protocol"),
        App("git", "Read and change a git repository: status, diff, log, commit.", ["uvx", "mcp-server-git"],
            "https://github.com/modelcontextprotocol/servers/tree/main/src/git", "Model Context Protocol",
            arguments="--repository <path>"),
        App("time", "Current time and time-zone conversions.", ["uvx", "mcp-server-time"],
            "https://github.com/modelcontextprotocol/servers/tree/main/src/time", "Model Context Protocol"),
        App("memory", "A knowledge graph the agent keeps between sessions.",
            ["npx", "-y", "@modelcontextprotocol/server-memory"],
            "https://github.com/modelcontextprotocol/servers/tree/main/src/memory", "Model Context Protocol"),
        App("github", "Issues, pull requests, code and repositories on GitHub.",
            ["docker", "run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
            "https://github.com/github/github-mcp-server", "GitHub",
            env=["GITHUB_PERSONAL_ACCESS_TOKEN"], setup="https://github.com/settings/personal-access-tokens"),
        App("playwright", "Drive a real browser: open pages, click, type, read.", ["npx", "@playwright/mcp@latest"],
            "https://github.com/microsoft/playwright-mcp", "Microsoft"),
        App("notion", "Search and edit pages and databases in Notion.", ["npx", "-y", "@notionhq/notion-mcp-server"],
            "https://github.com/makenotion/notion-mcp-server", "Notion",
            env=["NOTION_TOKEN"], setup="https://www.notion.so/profile/integrations"),
        App("brave-search", "Web search through the Brave Search API.", ["npx", "-y", "@brave/brave-search-mcp-server"],
            "https://github.com/brave/brave-search-mcp-server", "Brave",
            env=["BRAVE_API_KEY"], setup="https://brave.com/search/api/"),
    ]
}


def app(name, *arguments, env=None, **options):
    """The MCP server registered as `name`, ready for `agent.use()`. Nothing starts until then."""
    entry = APPS.get(name)
    if entry is None:
        close = difflib.get_close_matches(name, APPS, n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"No app named {name!r}.{hint} Known: {', '.join(sorted(APPS))}. "
                         "For any other server use mcp('<command>').")
    values = dict(env or {})
    missing = [variable for variable in entry.env if not (values.get(variable) or os.environ.get(variable))]
    if missing:
        where = f" Get it at {entry.setup}." if entry.setup else ""
        raise ValueError(f"app({name!r}) needs {', '.join(missing)} in the environment or in env={{...}}.{where}")
    for variable in entry.env:
        values[variable] = values.get(variable) or os.environ[variable]
    return mcp([*entry.command, *map(os.path.expanduser, arguments)], env=values, name=name.replace("-", "_"), **options)


def listing():
    """The registry as text, for `nanoagent apps`."""
    width = max(map(len, APPS))
    lines = []
    for entry in sorted(APPS.values(), key=lambda each: each.name):
        needs = f"  needs {', '.join(entry.env)}" if entry.env else ""
        lines.append(f"{entry.name:<{width}}  {entry.description}{needs}\n{'':<{width}}  {entry.source} ({entry.maintainer})")
    return "\n".join(lines)
