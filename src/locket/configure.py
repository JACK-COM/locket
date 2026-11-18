#!/usr/bin/env python3
"""configure: Locket's embedder settings, held in the Panoply's one settings file.

The file is ~/.panoply/config.json (PANOPLY_CONFIG names another), shared by every
Panoply piece. Its top-level `embed` section is the embedder every piece uses, and a
section named for a piece holds that piece's own overrides:

    {"embed":  {"model": "embeddinggemma-2:270m", "venv": "~/.panoply/venv"},
     "locket": {"embed": {"model": "nomic-embed-text"}}}

`locket configure embedder` writes Locket's own section; `--global` writes the top level,
which reaches every piece that does not override that key. Every way Locket runs, the
command line, the Claude Code, Codex and Hermes hooks, the MCP server, reads the file, so
one change reaches all of them.

Each setting resolves in this order, first found wins:
  1. an environment variable (MEMFIND_MODEL, OLLAMA_HOST, MEMFIND_NO_AUTOSTART=1,
     PANOPLY_VENV or LOCKET_VENV), for one shell or one test;
  2. Locket's own section;
  3. the top level;
  4. the shipped default.
A variable set in one place and not another is how two processes come to disagree, so
`locket configure` and `locket doctor` name every variable that is overriding the file.

An index is built in one embedder's space and tagged with it. After a model change every
store's index belongs to the old model: `find` and `index` rebuild a store when run, and
the hooks stay silent for that store until it is rebuilt, because a hook never rebuilds.
`--index` rebuilds every store as part of the change; without it, run `locket index all`
when convenient. `locket configure` lists which stores are behind.

Locket 0.10 kept these settings in ~/.locket/config.json. While that file remains, Locket
reads it below the top level; `locket migrate`, or the next `locket configure`, moves its
settings into the top level and renames it config.json.migrated.

Write the file with `configure`, never by hand while another configure runs: it holds a
lock, writes atomically and checks the result against the schema (`locket configure
schema` prints it).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _embed  # noqa: E402
import _settings  # noqa: E402

PIECE = "locket"
LEGACY = None           # Locket 0.10's file; resolved on use, so an unset HOME cannot break an import


def legacy_path():
    return LEGACY or Path.home() / ".locket" / "config.json"


def legacy_settings():
    """The usable embed settings of Locket 0.10's file while it remains, else None."""
    try:
        data = json.loads(legacy_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError):
        return None
    e = data.get("embed") if isinstance(data, dict) else None
    return {k: v for k, v in e.items() if k in _embed.CONFIG_KEYS and _embed._usable(k, e)} \
        if isinstance(e, dict) else None


def layers(data):
    """Locket's view of the embed settings in `data`, highest first: its own section, the
    top level, then an unmigrated ~/.locket/config.json."""
    out = _settings.layers(PIECE, "embed", data)
    legacy = legacy_settings()
    return out + [("~/.locket/config.json", legacy)] if legacy else out


def apply():
    """Put Locket's embed settings into the embedder, under the environment. Every module
    that ranks calls this once at import; it is cheap and changes nothing without a file."""
    try:
        return _embed.apply_config(layers(_settings.load(check=False)[0]))
    except Exception:                       # nothing in a settings file may stop a hook
        return _embed.apply_config(None)


def migrate():
    """Move ~/.locket/config.json's settings into the shared file's top level, where a key
    already set keeps its value, and rename the old file, both under the shared file's lock.
    Leaves everything in place when either file cannot be read. Returns a line, or None."""
    old = legacy_path()
    if not old.exists():
        return None
    legacy = legacy_settings()
    if legacy is None:
        return f"note: {old} has no readable embed section; left in place"
    with _settings.lock():
        data, findings = _settings.load()
        if _settings.unreadable(findings) or not isinstance(data.get("embed", {}), dict):
            return f"note: {_settings.path()} cannot be read, so {old} was left in place; fix it and rerun"
        top = data.get("embed", {})
        add = {k: v for k, v in legacy.items() if k not in top}
        if add:
            _settings.write_json(_settings.path(), _settings.edit(data, "embed", add))
        done = old.with_name("config.json.migrated")
        if done.exists():               # an earlier migration's copy is kept, never overwritten
            done = old.with_name(f"config.json.migrated-{int(time.time())}")
        try:
            old.rename(done)
        except FileNotFoundError:       # another process moved it first
            return None
    return f"moved {old} into {_settings.path()}; the old file is now {done.name}"


if __name__ == "__main__":
    from commands import settings
    sys.exit(settings.main(sys.argv[1:]))
