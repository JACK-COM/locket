"""Claude Desktop: no hooks, so not a hook host; an install target that registers the
MCP server (`locket_mcp.py`) in Desktop's own config."""
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import install

# Where Claude Desktop keeps its config: macOS and Windows only. Elsewhere
# it is None, so `--desktop` refuses rather than write a file Desktop never reads.
CONFIG = (Path.home() / "Library/Application Support/Claude/claude_desktop_config.json" if sys.platform == "darwin"
          else Path(os.environ["APPDATA"]) / "Claude/claude_desktop_config.json" if os.environ.get("APPDATA")
          else None)


# Claude Code runs a hook through the user's shell, so `python3` survives an interpreter
# upgrade; Claude Desktop spawns a server with a minimal PATH, where a bare `python3` can
# be the macOS installer stub and hang, so the server takes the absolute path.
def merge_server(data):
    servers = data.setdefault("mcpServers", {})
    if "locket" in servers:
        return []
    cmd = install.command()
    servers["locket"] = ({"command": cmd, "args": ["mcp"]} if cmd else
                         {"command": sys.executable, "args": [str(install.HERE / "locket_mcp.py")]})
    return ["mcpServers.locket"]


def ours(entry):
    """Whether a `mcpServers.locket` entry runs Locket's server, as install writes it or
    `locket help install` gives it: an entry a person pointed elsewhere is theirs to keep."""
    if not isinstance(entry, dict):
        return False
    args = [str(a) for a in entry.get("args") or []]
    return ((Path(str(entry.get("command", ""))).name == "locket" and args == ["mcp"])
            or any(Path(a).name == "locket_mcp.py" for a in args))


DISPLAY = "Claude Desktop"
FLAG = "--desktop"
INSTALL = "registers the MCP server with Claude Desktop"
INSTALL_EXAMPLE = "plus the Claude Desktop server; restart the app after"


def register():
    if CONFIG is None:
        print("--desktop: no known Claude Desktop config path on this platform; add the "
              "mcpServers entry by hand (`locket help install`, Step 4)", file=sys.stderr)
        return 1
    rc = install.merge_json(CONFIG, merge_server)
    print("restart Claude Desktop once for the server to load")
    return rc


def removal():
    entry = None
    if CONFIG and CONFIG.is_file():
        try:
            entry = (json.loads(CONFIG.read_text()).get("mcpServers") or {}).get("locket")
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            entry = None
    if entry is None:
        return install.Removal([], None, [])
    if not ours(entry):
        return install.Removal([], None, [f"{CONFIG}: mcpServers.locket runs something else; left as it is"])

    def apply():
        assert CONFIG is not None           # `entry` is None without one, so apply is never built
        data = json.loads(CONFIG.read_text())
        data["mcpServers"].pop("locket", None)
        install.rewrite_json(CONFIG, data, "mcpServers.locket removed")
    return install.Removal([("mcp", f"{CONFIG}: mcpServers.locket")], apply, [])


def checks(m, row):
    """Doctor's row for the Desktop server entry."""
    if CONFIG is not None and CONFIG.is_file():
        try:
            srv = (json.loads(CONFIG.read_text()).get("mcpServers") or {}).get("locket")
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            srv = None
        if srv:
            parts = [Path(srv.get("command", ""))] + [Path(a) for a in srv.get("args", []) if str(a).endswith(".py")]
            gone = [str(p) for p in parts if not p.is_file()]
            row("fail", "desktop server", f"names a missing file: {', '.join(gone)}", "locket install --desktop") \
                if gone else row("ok", "desktop server", "registered; restart Claude Desktop after a change")


if TYPE_CHECKING:                           # the faces this module has, checked by mypy
    from adapters import Target, desktop as _self
    _target: Target = _self
