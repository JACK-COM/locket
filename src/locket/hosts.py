"""The clients Locket serves, one `Host` row each in `TABLE`: where each keeps its home,
which of its directories are corpus, and which module under `adapters/` speaks for it.

A new host is one row here and one module under `adapters/`; the hook, install, uninstall
and doctor walk the rows. A home is read at call time through `homes()`, so a test that
installs its own `Paths` with `host_stores` moves every host with it.
"""
import os
from pathlib import Path
from typing import NamedTuple, Optional

import fsio
import paths


class Host(NamedTuple):
    """One client. `name` keys its store; `adapter` is its module under `adapters/`; `home`
    is its directory under the user's home, None for a client with no store (Claude
    Desktop takes the MCP server only), and `env` the variable that moves it. `members`
    are the directories inside the home that are corpus, `excluded` host state inside a
    member, `foreign` a memory not laid out as this apparatus lays it out (so it inherits
    no index, ledger or holding-space defaults), and `code` a home holding scripts a
    memory may cite by `path:line`."""
    name: str
    adapter: str
    home: Optional[str] = None
    env: Optional[str] = None
    members: frozenset = frozenset()
    excluded: frozenset = frozenset()
    foreign: bool = False
    code: bool = False


# In the order install, uninstall and doctor walk them. A member list is an allowlist: the
# home is mostly host state (sessions, caches, plugins, transcripts, project stores) and a
# host adds directories between releases, so a directory not named is out until a
# manifest's `members` lets it in, where a denylist would leak each new one silently.
# Root-level markdown (CLAUDE.md, SOUL.md, AGENTS.md) is always in.
TABLE = (
    # Claude Code mirrors the account's synced skills into `skills/synced/<bucket>/`, a
    # verbatim copy of the live skill bodies that paired every skill against itself at 1.00.
    Host("council", "claude_code", ".claude",
         members=frozenset({"memory", "rules", "skills", "agents", "agent-memory"}),
         excluded=frozenset({"synced"}), code=True),
    # Codex writes its memories/ itself; its skills/ are bundled, not authored.
    Host("codex", "codex", ".codex", env="CODEX_HOME", members=frozenset({"memories"}), foreign=True),
    # Hermes's MEMORY.md is the whole memory, not an index over topic files.
    Host("hermes", "hermes", ".hermes", members=frozenset({"memories"}), foreign=True, code=True),
    Host("desktop", "desktop"),
)


def row(name: str) -> Host:
    """The row whose store is `name` or whose adapter module is `name`
    (`hermes`, `adapters.hermes`)."""
    key = name.rsplit(".", 1)[-1]
    return next(h for h in TABLE if key in (h.name, h.adapter))


def homes() -> dict:
    """{store name: home} for every host with a store: `Paths.host_stores` where a test or
    probe gave one, else each row's directory, moved by its `env`."""
    given = paths.PATHS.host_stores
    if given is not None:
        return given
    return {h.name: Path(os.environ.get(h.env or "") or paths.HOME / h.home) for h in TABLE if h.home}


def home(name: str) -> Path:
    """A host's home, or a path under Locket's state that never exists where a `Paths`
    was built without that host (`Paths.under`), so a lookup matches nothing and a
    glob finds nothing; a KeyError inside a hook would let the call through."""
    return homes().get(name) or paths.PATHS.state / "no-host" / name


def claude() -> Path:
    """Claude Code's home, the council store, whose `projects/` holds each project's memory."""
    return home("council")


def name_of(root: Optional[fsio.StrPath]) -> Optional[str]:
    """The host-store name whose home is `root`, or None: by spelling first, then by
    inode, so a home reached through a symlink or `/private` keeps its members and
    exclusions, as trust already matches it."""
    if root is None:
        return None
    root, known = Path(root), homes().items()
    return next((k for k, v in known if Path(v) == root), None) or next(
        (k for k, v in known if Path(v).is_dir() and fsio.same(v, root)), None)


def is_host(root: Optional[fsio.StrPath]) -> bool:
    """Whether `root` is a host's home."""
    return name_of(root) is not None


def foreign(root: Optional[fsio.StrPath]) -> bool:
    """Whether `root` is the home of a host whose files are not this apparatus's."""
    return name_of(root) in FOREIGN


# Derived from TABLE; tests patch these dicts for one case.
HOST_MEMBERS = {h.name: set(h.members) for h in TABLE if h.home}
FOREIGN = {h.name for h in TABLE if h.foreign}
HOST_EXCLUDED = {h.name: set(h.excluded) for h in TABLE if h.excluded}


def members_of(root: Optional[fsio.StrPath]) -> Optional[set[str]]:
    """A host store's member directories, or None where `root` is no host's home."""
    name = name_of(root)
    return set(HOST_MEMBERS[name]) if name in HOST_MEMBERS else None


def excluded_of(root: Optional[fsio.StrPath]) -> set[str]:
    """Directories a host excludes inside its members; empty where `root` is no host's home."""
    name = name_of(root)
    return set(HOST_EXCLUDED.get(name, ())) if name else set()


# Trees a package manager or a build writes: never prose, and their READMEs and
# changelogs would score as claims against the store they sit in. Unioned into every
# store's exclusions rather than defaulted, because a manifest's own `excluded_dirs`
# replaces the defaults and must not re-admit them. A dotted directory (.venv, .build,
# .git) is already skipped by name.
DEPENDENCY_DIRS = {"node_modules", "bower_components", "jspm_packages", "vendor",
                   "Pods", "Carthage", "DerivedData", "venv", "site-packages",
                   "__pycache__", "build", "dist"}
# Directories inside a corpus deliberately NOT deduplicated. An archive is a verbatim
# copy of superseded text, so it duplicates by design and would trip on every write; an
# in-flight thread appends until it closes; `discovery-research` holds angle files a
# cycle appends to with no session in front of it to read a nudge that BLOCKS the write.
# The cache is never prose, and a backup is a verbatim copy, so it forks by design.
EXCLUDED_DIRS = {"archive", "in-flight", "harness-logs", "discovery-research", ".memfind",
                 "backups", "backup"}
