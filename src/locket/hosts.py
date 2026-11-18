"""The hosts Locket serves: where each keeps its home, and which of its directories are corpus.

A host's home is `paths.PATHS.host_stores[<name>]`, read at call time, so a test that
installs its own `Paths` moves the host with it. Nothing here copies a location at import.
"""
from pathlib import Path

import paths


def claude():
    """Claude Code's home, the council store."""
    return paths.PATHS.host_stores["council"]


def hermes():
    """Hermes Agent's home."""
    return paths.PATHS.host_stores["hermes"]


def codex():
    """Codex's home, which CODEX_HOME moves."""
    return paths.PATHS.host_stores["codex"]


def name_of(root):
    """The host-store name whose home is `root`, or None."""
    if root is None:
        return None
    root = Path(root)
    return next((k for k, v in paths.PATHS.host_stores.items() if Path(v) == root), None)


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
# Host state that lives INSIDE a member and is excluded whatever a manifest says:
# Claude Code mirrors the account's synced skills into `skills/synced/<bucket>/`, a
# verbatim copy of the live skill bodies, so every skill paired against itself at 1.00
# on the first audit of the whole home.
HOST_EXCLUDED = {
    "council": {"synced"},
}


def members_of(root):
    """A host store's member directories, or None where `root` is no host's home."""
    name = name_of(root)
    return set(HOST_MEMBERS[name]) if name in HOST_MEMBERS else None


def excluded_of(root):
    """Directories a host excludes inside its members; empty where `root` is no host's home."""
    return set(HOST_EXCLUDED.get(name_of(root), ()))


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
