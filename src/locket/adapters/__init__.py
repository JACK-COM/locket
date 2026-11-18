"""The hook hosts: which one sent a payload, the payload read into Claude Code's shape,
and a decision delivered the way that host can hear it.

One module per host, each declaring NAME, `detects(payload)`, `native(payload, tool,
tool_input)` (its payload into Claude Code's tool names and fields), `deliver(payload,
msg, blocking, source, db)` (returns the exit code) and `status()` (one doctor line, or
None where the host is not installed). A new host is one module added to HOSTS.
"""
import json
import sys

import gate
from adapters import claude_code, codex, hermes

# Walked in order and the first host that detects a payload owns it; Claude Code,
# whose shape the others are read into, detects everything and so comes last.
HOSTS = (hermes, codex, claude_code)
BY_NAME = {h.NAME: h for h in HOSTS}


def host_of(payload):
    """Which host sent this hook payload, by NAME."""
    return next(h for h in HOSTS if h.detects(payload)).NAME


def read_payload():
    """The hook payload from stdin, normalised to Claude Code's shape, or None."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return None
    return normalise(payload)


def normalise(payload):
    """`read_payload` on an already-parsed dict, so a test can drive it. Sets `_host`
    to the sending host's NAME."""
    if not isinstance(payload, dict):
        return None
    ti = payload.get("tool_input")
    if not isinstance(ti, dict):
        ti = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    host = BY_NAME[host_of(payload)]
    tool, ti = host.native(payload, payload.get("tool_name"), dict(ti))
    # A relative path resolves against the SESSION's cwd, which the payload
    # carries, never against the hook subprocess's own. Left relative it misses
    # every corpus and the write passes with no gate, silently.
    fp = ti.get("file_path")
    if isinstance(fp, str) and fp:
        ti["file_path"] = gate.absolute_path(fp, payload.get("cwd"))
    return dict(payload, tool_input=ti, tool_name=tool, _host=host.NAME)


def emit(payload, msg, blocking, source="memscan", db=None):
    """Deliver a hook decision the way THIS host can hear it; returns the exit code.
    An empty message is no decision. `source` names the mode, so a host that records
    refusals keeps one record per mode; `db` is the host's state to read, for a test."""
    if not msg:
        return 0
    return BY_NAME.get(payload.get("_host"), claude_code).deliver(payload, msg, blocking, source, db)


def gate_status():
    """One line per installed host on whether the 2e question gate can see a session
    there, and whether a non-blocking nudge has a way to reach the model."""
    lines = [s for s in (h.status() for h in reversed(HOSTS)) if s]
    return "\n".join(lines) if lines else "no known host home found; CLI modes only"
