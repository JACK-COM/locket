"""The hosts Locket serves, each one module with two faces.

The hook face, for every hook host in HOSTS: NAME, `detects(payload)`, `native(payload,
tool, tool_input)` (its payload into Claude Code's tool names and fields), `deliver(payload,
msg, blocking, source, db)` (the exit code) and `status()` (one line, or None where the
host is not installed); `home()` (its own store), `asked(payload)` (the 2e answer when the
payload carries no transcript), and for the trigger hook `trigger_event(payload)` (its
payload in the shape `trigger.fire` reads), `seen(payload, home, pattern)` (whether the
session carries a pattern past its transcript) and `trigger_reply(payload, message, block,
home)` (the exit code).

The install face, for every target in TARGETS (the hook hosts and Desktop): FLAG (the
`locket install` flag, or None where the host's config is edited by hand), `register()`,
`removal()` (an `install.Removal`) and `checks(m, row)` (its doctor rows). The install
face imports probes and trigger inside its functions, so a hook never loads them.

A new host is one module added to HOSTS and TARGETS.
"""
import json
import sys

import gate
from adapters import claude_code, codex, desktop, hermes

# Walked in order and the first host that detects a payload owns it; Claude Code,
# whose shape the others are read into, detects everything and so comes last.
HOSTS = (hermes, codex, claude_code)
BY_NAME = {h.NAME: h for h in HOSTS}
# What `locket install`, `uninstall` and `doctor` walk: every hook host, and Desktop, which
# runs no hooks and takes the MCP server. Each declares FLAG (its install flag, or None
# where its config is edited by hand), `register()`, `removal()` and `checks(m, row)`.
TARGETS = (claude_code, codex, hermes, desktop)


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
        args = payload.get("args")
        ti = args if isinstance(args, dict) else {}
    host = BY_NAME[host_of(payload)]
    tool, ti = host.native(payload, payload.get("tool_name"), dict(ti))
    # A relative path resolves against the SESSION's cwd, which the payload
    # carries, never against the hook subprocess's own. Left relative it misses
    # every corpus and the write passes with no gate, silently.
    fp = ti.get("file_path")
    if isinstance(fp, str) and fp:
        ti["file_path"] = gate.absolute_path(fp, payload.get("cwd"))
    return dict(payload, tool_input=ti, tool_name=tool, _host=host.NAME)


def sender(payload):
    """The host module a normalised payload came from; Claude Code's when unmarked."""
    return BY_NAME.get(payload.get("_host"), claude_code)


def emit(payload, msg, blocking, source="memscan", db=None):
    """Deliver a hook decision the way THIS host can hear it; returns the exit code.
    An empty message is no decision. `source` names the mode, so a host that records
    refusals keeps one record per mode; `db` is the host's state to read, for a test."""
    if not msg:
        return 0
    return sender(payload).deliver(payload, msg, blocking, source, db)


def gate_status():
    """One line per installed host on whether the 2e question gate can see a session
    there, and whether a non-blocking nudge has a way to reach the model."""
    lines = [s for s in (h.status() for h in reversed(HOSTS)) if s]
    return "\n".join(lines) if lines else "no known host home found; CLI modes only"
