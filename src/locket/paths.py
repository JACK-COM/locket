"""Where Locket keeps its own state, and where the hosts it serves live.

Every file Locket owns outside a store, and the host stores it knows on sight, are
fields of one `Paths`. `PATHS` is this process's, read from the environment at import,
so a hook or probe spawned with a different environment builds its own. Code reads
`paths.PATHS.<field>` at call time, never a copy taken at import; a test builds a
`Paths` of its own and installs it with `use(...)` for the length of a `with`.
"""
import contextlib
import dataclasses
import os
import tempfile
from pathlib import Path

HOME = Path.home()


@dataclasses.dataclass(frozen=True)
class Paths:
    # The registry: `{"corpora": [...], "pins": {path: digest}}`. Its directory is
    # Locket's state directory, outside every corpus; `LOCKET_REGISTRY` moves both.
    registry: Path
    # Hook health: live-run markers and per-verb counters.
    health_dir: Path
    # The grep-assist diagnostic log, newest half kept past its cap.
    grepassist_log: Path
    # Refusal records for ask-once gates. Keyed by uid and created owner-only: a
    # shared temp dir lets another user make the path first, as a readable
    # directory or a symlink elsewhere.
    pending_dir: Path
    # Each host's core directory, a store found without a manifest.
    host_stores: dict = dataclasses.field(default_factory=dict)

    @property
    def state(self):
        return self.registry.parent

    @property
    def cache(self):
        return self.state / "cache"

    @property
    def usage_ledger(self):
        return self.state / "usage.csv"

    @property
    def schema_file(self):
        return self.state / "locket.schema.json"

    @classmethod
    def under(cls, state, host_stores=None, pending_dir=None):
        """Every Locket file under `state`, for a test or a probe; host stores only as given."""
        state = Path(state)
        return cls(registry=state / "corpora.json", health_dir=state / "hook-health",
                   grepassist_log=state / "grepassist.csv",
                   pending_dir=Path(pending_dir) if pending_dir else state / "pending",
                   host_stores=dict(host_stores or {}))

    @classmethod
    def from_env(cls, env=os.environ):
        registry = Path(env.get("LOCKET_REGISTRY") or HOME / ".locket" / "corpora.json")
        state = registry.parent
        uid = os.getuid() if hasattr(os, "getuid") else "u"
        return cls(
            registry=registry,
            health_dir=Path(env.get("LOCKET_HEALTH_DIR") or state / "hook-health").expanduser(),
            grepassist_log=Path(env.get("LOCKET_GREPASSIST_LOG") or state / "grepassist.csv").expanduser(),
            pending_dir=Path(tempfile.gettempdir()) / f"memscan-pending-{uid}",
            # Each host's home: Claude Code's, Hermes Agent's, and Codex's, which CODEX_HOME moves.
            host_stores={"council": HOME / ".claude", "hermes": HOME / ".hermes",
                         "codex": Path(env.get("CODEX_HOME") or HOME / ".codex")},
        )

    def replace(self, **changes):
        """A copy with `changes`; the host-store table is copied too, so editing the
        copy's never reaches this one."""
        changes.setdefault("host_stores", dict(self.host_stores))
        return dataclasses.replace(self, **changes)


PATHS = Paths.from_env()


@contextlib.contextmanager
def use(p):
    """Install `p` as this process's paths until the block exits."""
    global PATHS
    saved, PATHS = PATHS, p
    try:
        yield p
    finally:
        PATHS = saved
