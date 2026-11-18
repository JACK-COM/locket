"""The clients Locket serves, each one module here with two faces: `HookHost`, for a host
whose hooks Locket runs in, and `Target`, for a client `locket install`, `uninstall` and
`doctor` reach. A new host is its module here and its row in `hosts.TABLE`, which names
the module; TARGETS and HOSTS are read off the table. Each module states which faces it
has under TYPE_CHECKING at its foot, so `make typecheck` fails while it lacks any part.
"""
from __future__ import annotations
import importlib

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, cast

import gate
import hosts

if TYPE_CHECKING:
    import doctor
    import install


class HookHost(Protocol):
    """The hook face. `native` reads the host's payload into Claude Code's tool names and
    fields; `deliver` hands a decision back and returns the exit code; `status` is one
    line, or None where the host is not installed; `home` is its own store; `asked` is the
    2e answer when the payload carries no transcript. For the trigger hook, `trigger_event`
    is the payload in the shape `trigger.fire` reads, `seen` whether the session carries a
    pattern past its transcript, and `trigger_reply` the exit code."""
    @property
    def NAME(self) -> str: ...
    @property
    def BASE(self) -> bool: ...
    @property
    def TRIGGER_EXIT(self) -> int: ...
    def detects(self, payload: dict) -> bool: ...
    def native(self, payload: dict, tool: Optional[str], ti: dict) -> tuple[Optional[str], dict]: ...
    def deliver(self, payload: dict, msg: str, blocking: bool, source: str, db: Optional[Path]) -> int: ...
    def status(self) -> Optional[str]: ...
    def home(self) -> Path: ...
    def asked(self, payload: dict) -> bool: ...
    def session_cwds(self) -> set[str]: ...
    def probe_write(self, store: Path, name: str, text: str) -> dict: ...
    def probe_call(self, cwd: str, command: str) -> dict: ...
    def trigger_event(self, payload: dict) -> dict: ...
    def seen(self, payload: dict, home: Optional[Path], pattern: str) -> bool: ...
    def trigger_reply(self, payload: dict, message: str, block: bool, home: Optional[Path]) -> int: ...


class Target(Protocol):
    """The install face. FLAG is the `locket install` flag, or None where the host's config
    is edited by hand (`register` then does nothing); `removal` is what uninstall does to
    its config; `checks` adds its doctor rows. It imports probes and trigger inside its
    functions, so a hook never loads them."""
    @property
    def DISPLAY(self) -> str: ...
    @property
    def FLAG(self) -> Optional[str]: ...
    @property
    def INSTALL(self) -> str: ...
    @property
    def INSTALL_EXAMPLE(self) -> str: ...
    def register(self) -> int: ...
    def removal(self) -> install.Removal: ...
    def checks(self, m: doctor.Machine, row: doctor.Row) -> None: ...


# Every client, in hosts.TABLE order: install, uninstall and doctor walk these. The
# cast stands where the static check cannot see; each module checks itself (above).
TARGETS: tuple[Target, ...] = tuple(cast(Target, importlib.import_module(f"adapters.{h.adapter}"))
                                    for h in hosts.TABLE)
# The ones with hooks, walked in order, the first that detects a payload owning it. The
# BASE host's shape is what the others are read into: it detects everything, so it comes last.
HOSTS: tuple[HookHost, ...] = tuple(sorted((cast(HookHost, m) for m in TARGETS if hasattr(m, "detects")),
                                           key=lambda m: m.BASE))
BY_NAME = {h.NAME: h for h in HOSTS}
BASE_HOST = HOSTS[-1]                       # for a payload no host marked


def session_cwds():
    """Every working directory any hook host has opened a session in."""
    return set().union(*(h.session_cwds() for h in HOSTS))


def named(mods):
    """`A, B or C` from the modules' DISPLAY names."""
    names = [m.DISPLAY for m in mods]
    if not names:
        return "no host"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


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
    return BY_NAME.get(payload.get("_host"), BASE_HOST)


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
    lines = [s for s in (cast(HookHost, h).status() for h in TARGETS if h in HOSTS) if s]
    return "\n".join(lines) if lines else "no known host home found; CLI modes only"
