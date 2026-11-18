"""The hosts Locket serves: where each keeps its home, and which of its directories are corpus.

A host's home is `paths.PATHS.host_stores[<name>]`, read at call time, so a test that
installs its own `Paths` moves the host with it. Nothing here copies a location at import.
"""
from pathlib import Path
from typing import Optional

import fsio
import paths


def _home(name: str) -> Path:
    """A host's home, or a path under Locket's state that never exists where a `Paths`
    was built without that host (`Paths.under`), so a lookup matches nothing and a
    glob finds nothing; a KeyError inside a hook would let the call through."""
    return paths.PATHS.host_stores.get(name) or paths.PATHS.state / "no-host" / name


def claude() -> Path:
    """Claude Code's home, the council store."""
    return _home("council")


def hermes() -> Path:
    """Hermes Agent's home."""
    return _home("hermes")


def codex() -> Path:
    """Codex's home, which CODEX_HOME moves."""
    return _home("codex")


def name_of(root: Optional[fsio.StrPath]) -> Optional[str]:
    """The host-store name whose home is `root`, or None: by spelling first, then by
    inode, so a home reached through a symlink or `/private` keeps its members and
    exclusions, as trust already matches it."""
    if root is None:
        return None
    root, homes = Path(root), paths.PATHS.host_stores.items()
    return next((k for k, v in homes if Path(v) == root), None) or next(
        (k for k, v in homes if Path(v).is_dir() and fsio.same(v, root)), None)


def is_host(root: Optional[fsio.StrPath]) -> bool:
    """Whether `root` is a host's home."""
    return name_of(root) is not None


def foreign(root: Optional[fsio.StrPath]) -> bool:
    """Whether `root` is the home of a host whose files are not this apparatus's."""
    return name_of(root) in FOREIGN


# What counts as corpus INSIDE a host's core directory, keyed by host-store name. The
# home is mostly host state (sessions, caches, plugins, transcripts, project stores),
# and the host adds directories between releases, so the default is an allowlist: a
# directory not named here is out until a manifest's `members` lets it in, which fails
# by omission where a denylist leaks every new directory silently. Root-level markdown
# (CLAUDE.md, SOUL.md, AGENTS.md) is always in. Hermes's and Codex's `skills/` are
# bundled with the agent rather than authored, so they are out.
HOST_MEMBERS = {
    "council": {"memory", "rules", "skills", "agents", "agent-memory"},
    "hermes": {"memories"},
    "codex": {"memories"},
}
# Hosts whose memory is not laid out as this apparatus lays it out: Hermes's MEMORY.md is
# the whole memory, not an index over topic files, and Codex writes its memories/ itself,
# so neither inherits the index, ledger and holding-space defaults.
FOREIGN = {"hermes", "codex"}
# Host state that lives INSIDE a member and is excluded whatever a manifest says:
# Claude Code mirrors the account's synced skills into `skills/synced/<bucket>/`, a
# verbatim copy of the live skill bodies, so every skill paired against itself at 1.00
# on the first audit of the whole home.
HOST_EXCLUDED = {
    "council": {"synced"},
}


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
