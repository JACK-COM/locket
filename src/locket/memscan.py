#!/usr/bin/env python3
"""memscan: duplication, drift and link checks over a flat memory corpus.

The unit is the CLAIM, not the file. Document-level similarity cannot see one
clause restated inside four files that are otherwise about different subjects,
and it scores sibling records sharing a whole vocabulary as duplicates.

Modes:

  audit    cross-file verbatim claim pairs. These are copies that currently
           AGREE, so they are fork RISK rather than proof of a fork.
  dupes    repeated claims INSIDE one file, which `audit` skips by construction.
           The uniqueness invariant on a holding space (prime-memory-discipline §2).
  budget   §5 as counting: bytes, dated history clauses (each listed by line,
           the reconsolidation worklist), longest prose paragraph. `audit` is
           blind to a file that only disagrees with the world, so nothing else
           measures this.
  pointers every index line paired to the file it points at, for adjudication
           (`hooks` is its older name). Needs a reader: a hook asserting what its topic file now refutes
           scores LOW on similarity and is invisible to `audit`.
  links    wikilink targets that resolve to no file.
  cites    every `path:line` citation resolved against the code it points at.
           The one claim class nothing else here can see, because the claim is
           ABOUT the code rather than IN it and no other mode opens the file.
  corpora  every corpus on the machine, with its file count.
  init     make a directory a corpus every mode can find: manifest, registry
           row, index. `forget` drops the row again.
  against  candidate owners for an incoming statement. Used by the write-time
           hook.

The corpus argument takes a registered name (`notes`), any substring of a project
slug (`myapp`), or a literal path. With no argument it takes THIS project's corpus.

Stdlib only, by design: this runs inside a PreToolUse hook on every memory
write, so it pays interpreter startup and nothing else.
"""
import ast
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import shlex
import stat
import sys
import tempfile
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath

import paths
from paths import CLAUDE, CODEX, HERMES, HOME
# A per-corpus manifest, optional. Its presence makes ANY directory a corpus for
# the write-time gate and for a literal path, so a memory store on a host this
# file has never heard of is covered by writing one file into it; its absence
# means the host defaults below. Every key it may carry is listed once, under
# MANIFEST in USAGE; `settings`, `csv_sources` and `all_corpora` read them, and
# only from a trusted store (`trusted`). It is the only thing this tool owns
# inside a corpus; the index lives in ~/.locket/cache.
MANIFEST = "locket.json"
OLD_MANIFEST = "memfind.json"      # the name before 2026-09-21; read, migrated by `migrate`, never written
LEGACY_MANIFEST = "reference-sources.json"


def manifest_file(root):
    """The manifest path for `root`: `locket.json`, else the older `memfind.json`
    where only that exists, else the new name, for writing. Every read goes
    through here so a store not yet migrated keeps working."""
    root = Path(root)
    new = root / MANIFEST
    if new.is_file():
        return new
    old = root / OLD_MANIFEST
    return old if old.is_file() else new


def has_manifest(root):
    return manifest_file(root).is_file()
# The registry: manifest directories that `corpora`, `all` and a substring
# argument can find. The gate derives a corpus from the written path and needs
# no registry, but enumeration has nothing to derive from, and a whole-disk
# search for manifests is not an option. `init <dir>` writes the manifest if
# absent and registers the path; the manifest stays the authority that a
# directory IS a corpus, so a row whose manifest is gone is reported, never
# trusted. Lives beside memfind's venv, outside every corpus; `LOCKET_REGISTRY`
# points a test at a scratch one.
# Each host's core directory is its council store, found without a manifest:
# the whole of `~/.claude` (memory, rules, skills, agents and their memories,
# the constitution at its root) is one corpus, so a rule restating a memory
# fact draws the write-time nudge instead of surfacing only in `across`.
# Claude Code's per-project auto-memory is derived from the path instead
# (`corpus_for`) and is a store of its own, never a member of council.
# What counts as corpus INSIDE a host's core directory. The home is mostly host
# state (sessions, caches, plugins, transcripts, the project stores above), and
# the host adds directories between releases, so the default is an allowlist:
# a directory not named here is out until a manifest's `members` lets it in,
# which fails by omission where a denylist leaks every new directory silently.
# Root-level markdown (CLAUDE.md, SOUL.md, AGENTS.md) is always in. Hermes's and
# Codex's `skills/` are bundled with the agent rather than authored, so they are out.
HOST_MEMBERS = {
    CLAUDE: {"memory", "rules", "skills", "agents", "agent-memory"},
    HERMES: {"memories"},
    CODEX: {"memories"},
}
# Host state that lives INSIDE a member and is excluded whatever a manifest
# says, like `.memfind`: Claude Code mirrors the account's synced skills into
# `skills/synced/<bucket>/`, a verbatim copy of the live skill bodies, so every
# skill paired against itself at 1.00 on the first audit of the whole home.
HOST_EXCLUDED = {
    CLAUDE: {"synced"},
}
# Shorthands for the CLI only. Any other argument is taken as a literal path, so
# `memscan audit ~/.claude/projects/<slug>/memory` works without a name here.
# Directories inside a corpus that are deliberately NOT deduplicated. An archive
# is a verbatim copy of superseded text, so it duplicates by design and would trip
# on every write; an in-flight investigation thread appends until it closes
# (prime-memory-discipline §6); `discovery-research` holds Mode-2 angle files the
# surveillance cycle appends to on a schedule, with no session in front of it to
# read a nudge that BLOCKS the write. The cache directory is never prose.
# Trees a package manager or a build writes: never prose, and their READMEs and
# changelogs would score as claims against the store they sit in. Unioned into
# every store's exclusions rather than defaulted, because a manifest's own
# `excluded_dirs` replaces the defaults and must not re-admit them. A dotted
# directory (.venv, .build, .git) is already skipped by name.
DEPENDENCY_DIRS = {"node_modules", "bower_components", "jspm_packages", "vendor",
                   "Pods", "Carthage", "DerivedData", "venv", "site-packages",
                   "__pycache__", "build", "dist"}
EXCLUDED_DIRS = {"archive", "in-flight", "harness-logs", "discovery-research", ".memfind",
                 "backups", "backup"}   # a backup is a verbatim copy, so it forks by design


def _registry_data():
    try:
        data = json.loads(paths.PATHS.registry.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def registry():
    """Registered corpus paths, as `init` wrote them. [] where the file is absent
    or unreadable, so enumeration degrades to the host stores rather than dies."""
    rows = _registry_data().get("corpora")
    return [Path(x) for x in rows if isinstance(x, str)] if isinstance(rows, list) else []


def registry_pins():
    """{registered path: digest of its configuration when a person trusted it}. A row
    with no pin predates pinning and is trusted as it stands until `migrate` pins it."""
    pins = _registry_data().get("pins")
    return {str(k): v for k, v in pins.items() if isinstance(v, str)} if isinstance(pins, dict) else {}


_DIGESTS = {}


def config_digest(d):
    """sha256 over every file that makes a store act: its manifest, its legacy sources
    file and its rows file, each named, absent ones included. Trust is pinned to it,
    so a `git pull` that adds a block-all row or a source from elsewhere leaves the
    store inert until a person trusts what it now says."""
    d = Path(d)
    m = _manifest_data(d)
    rows = m.get("triggers") if isinstance(m.get("triggers"), str) and m.get("triggers") else "triggers.json"
    rows = Path(rows).expanduser()
    files = [manifest_file(d), d / LEGACY_MANIFEST, rows if rows.is_absolute() else d / rows]
    sig = []
    for f in files:
        try:
            st = os.lstat(f)                # ctime: a restored mtime cannot reset it
            sig.append((str(f), st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size))
        except OSError:
            sig.append((str(f), None))
    sig = tuple(sig)
    hit = _DIGESTS.get(str(d))
    if hit and hit[0] == sig:
        return hit[1]
    h = hashlib.sha256()
    for f in files:
        try:
            with open_safe(f, binary=True) as fh:
                body = fh.read()
        except OSError:
            body = None
        tag = b"absent" if body is None else str(len(body)).encode()
        h.update(f.name.encode() + b"\0" + tag + b"\0" + (body or b""))
    _DIGESTS[str(d)] = (sig, h.hexdigest())
    return h.hexdigest()


def _inode(p):
    """(st_dev, st_ino) of a path, or None. Identity for trust: `Path` equality
    calls `/users` and `/Users` two directories on a case-insensitive disk."""
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        return None
    return st.st_dev, st.st_ino


_TRUST_CACHE = {}


def _trusted_ids():
    """Inodes of the host stores and every registered path, recomputed only when
    the registry file changes, because the walk asks once per manifest it meets."""
    try:
        st = paths.PATHS.registry.stat()
        key = (str(paths.PATHS.registry), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(paths.PATHS.registry), None, None)
    key += tuple(str(p) for p in paths.PATHS.host_stores.values())
    if key not in _TRUST_CACHE:
        hosts = {_inode(p) for p in paths.PATHS.host_stores.values()}
        hosts.discard(None)
        pins = registry_pins()
        rows = {}
        for p in registry():
            ident = _inode(p)
            if ident is not None:
                rows[ident] = pins.get(str(p))
        _TRUST_CACHE.clear()
        _TRUST_CACHE[key] = (hosts, rows)
    return _TRUST_CACHE[key]


def trusted(d):
    """Whether directory `d` may act as a store: a host store, Claude Code's
    `projects/<slug>/memory`, or a path a human registered (`locket trust`,
    `locket init` at a terminal). A manifest anywhere else is inert, because a
    cloned repository carries whatever manifest its author wrote: trigger rows
    that block every tool call, ledger schemas, `sources` outside the directory."""
    return _granted(Path(d))


def _granted(d):
    """The real trust check, which the tests' fixture trust never replaces:
    `grant_trust` asks it, so a fixture is still registered for real."""
    ident = _inode(d)
    if ident is None:
        return False
    hosts, rows = _trusted_ids()
    if ident in hosts:
        return True
    if ident in rows:
        pin = rows[ident]
        return pin is not None and pin == config_digest(d)
    return d.name == "memory" and _inode(d.parent.parent) == _inode(CLAUDE / "projects")


def changed_since_trust(d):
    """Whether `d` is registered but its configuration no longer matches its pin."""
    ident = _inode(d)
    rows = _trusted_ids()[1]
    return ident in rows and rows[ident] is not None and rows[ident] != config_digest(d)


def registry_stale():
    """Registered paths whose manifest is gone. Listed, never enumerated."""
    return [p for p in registry() if not has_manifest(p)]


_INVISIBLE = frozenset(("Cc", "Cf", "Co", "Cs", "Cn", "Zl", "Zp"))


def relay(text, limit=200, lines=False):
    """Text from a store, made safe to put in front of an agent or on a terminal:
    every control, format, private-use, unassigned and line-separator character
    becomes a space (a newline is how a planted label poses as a fresh instruction;
    an escape sequence rewrites a terminal; bidi overrides, zero-width marks and tag
    characters hide text from the person reading it), runs collapse, and the result
    is cut to `limit` characters."""
    import unicodedata
    if lines:                               # a row the user wrote across lines keeps them
        return "\n".join(relay(ln, limit) for ln in str(text)[:limit * 4].split("\n"))[:limit]
    flat = "".join(" " if unicodedata.category(c) in _INVISIBLE else c for c in str(text)[:limit * 4])
    flat = " ".join(flat.split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "\u2026"


class SafeOut:
    """A text stream that passes every line through `relay(lines=True)`'s character
    rule: whatever a mode prints (a title, a file name, a history line from a store)
    reaches a terminal or an agent with no control, bidi or invisible character."""

    def __init__(self, stream):
        self._s = stream

    def write(self, text):
        import unicodedata
        keep = "\n\t"
        return self._s.write("".join(c if c in keep or unicodedata.category(c) not in _INVISIBLE else " "
                                     for c in str(text)))

    def __getattr__(self, name):
        return getattr(self._s, name)


def _same(a, b):
    """Whether two paths name one directory: by inode where both exist, else by
    resolved path. Spelling is not identity on a case-insensitive disk."""
    ia, ib = _inode(a), _inode(b)
    if ia is not None and ib is not None:
        return ia == ib
    pa, pb = Path(a).expanduser().resolve(), Path(b).expanduser().resolve()
    if pa == pb:
        return True
    # not yet on disk: the same name under one directory, case aside, is taken as one
    # file, since on a case-insensitive disk it is, and a gate errs toward refusing
    return (pa.name.casefold() == pb.name.casefold() and pa.parent != pa
            and _same(pa.parent, pb.parent))


def write_store_file(path, text):
    """Write a file Locket owns inside a store (a manifest, a scaffolded ledger) by
    atomic rename, refusing a symlink at the path: a cloned repo's `locket.json`
    linked to `~/.claude/settings.json` must not turn `init` into an overwrite."""
    p = Path(path)
    if p.is_symlink():
        raise OSError(f"{p} is a symlink; Locket writes no file through one")
    # Locket's own edit to a manifest a person trusted keeps that trust: the pin moves
    # to what Locket wrote. Any other edit leaves the store inert until re-trusted.
    repin = (p.name in (MANIFEST, OLD_MANIFEST, LEGACY_MANIFEST) and _granted(p.parent)
             and any(_same(p.parent, r) for r in registry()))
    try:                                    # an edited file keeps its mode; a new one takes the umask's
        mode = os.stat(p).st_mode & 0o777
    except FileNotFoundError:
        mask = os.umask(0)
        os.umask(mask)
        mode = 0o666 & ~mask
    _replace_atomic(p, text, f".{p.name}.", mode)
    if repin:
        register(p.parent)


def pin_unpinned():
    """Pin every registered store that has no pin yet (rows from before 0.12) to its
    configuration as it stands, once a person agrees: an unpinned row is untrusted,
    since a registry written by hand carries no pins. Returns the stores pinned, or
    None when a person was needed and none answered."""
    pins = registry_pins()
    todo = [r for r in registry() if str(r) not in pins and has_manifest(r)]
    if todo and not person_confirms(
            f"Pin {len(todo)} stores registered before 0.12 to their configuration as it stands?",
            [str(r) for r in todo[:8]], "pin them"):
        return None
    for r in todo:
        register(r)
    return todo


def _write_registry(rows, pins=None):
    paths.PATHS.state.mkdir(parents=True, exist_ok=True)
    keep = sorted({str(p) for p in rows})
    pins = {k: v for k, v in (registry_pins() if pins is None else pins).items() if k in keep}
    _replace_atomic(paths.PATHS.registry, json.dumps({"corpora": keep, "pins": pins}, indent=2) + "\n", ".corpora.")


def register(root):
    """Add a corpus path to the registry, pinned to its configuration as it stands,
    or re-pin a row already there. True when the row was new."""
    root = Path(root).resolve()
    rows = registry()
    old = next((r for r in rows if _same(root, r)), None)
    pins = registry_pins()
    pins.pop(str(old), None)
    pins[str(old or root)] = config_digest(root)
    _write_registry(rows if old else rows + [root], pins)
    return old is None


def _first_cwd(path, limit=60):
    """The first `"cwd"` value in a session's JSON lines, or None."""
    try:
        with open(path, errors="ignore") as f:
            for _, line in zip(range(limit), f):
                if '"cwd"' not in line:
                    continue
                m = re.search(r'"cwd"\s*:\s*"((?:[^"\\]|\\.)*)"', line)
                if m:
                    return json.loads(f'"{m.group(1)}"')
    except (OSError, ValueError):
        pass
    return None


def _unslug(slug, budget=2000):
    """The existing directory Claude Code slugified into `slug`, or None. The slug
    replaces every non-alphanumeric character with `-`, so `a-b` may be `a/b`,
    `a-b`, `a.b` or `a_b`; matched against the disk one component at a time."""
    def slugify(name):
        return re.sub(r"[^A-Za-z0-9]", "-", name)

    steps = [0]

    def walk(d, rest):
        if not rest:
            return d
        steps[0] += 1
        if steps[0] > budget:
            return None
        try:
            names = os.listdir(d)
        except OSError:
            return None
        for n in sorted(names, key=len, reverse=True):
            sn = slugify(n)
            if rest == sn or rest.startswith(sn + "-"):
                p = os.path.join(d, n)
                if os.path.isdir(p):
                    hit = walk(p, rest[len(sn) + 1:])
                    if hit:
                        return hit
        return None

    return walk("/", slug[1:]) if slug.startswith("-") else None


def session_dirs():
    """Every working directory a host's session record names: Claude Code's
    project transcripts, Codex rollouts, Hermes `state.db`. The detection surface
    for an untrusted store; a directory no host has opened is not on it."""
    out = set()
    def mtime(f):
        try:
            return f.stat().st_mtime
        except OSError:                     # a transcript rotated away mid-scan
            return 0.0

    for proj in (CLAUDE / "projects").glob("*"):
        hit = _unslug(proj.name)
        if hit:
            out.add(hit)
        logs = sorted(proj.glob("*.jsonl"), key=mtime, reverse=True)
        for f in logs[:3]:
            cwd = _first_cwd(f)
            if cwd:
                out.add(cwd)
                break
    rollouts = sorted((CODEX / "sessions").glob("*/*/*/*.jsonl"), reverse=True)
    for f in rollouts[:500]:
        cwd = _first_cwd(f, limit=1)
        if cwd:
            out.add(cwd)
    db = HERMES / "state.db"
    try:
        import sqlite3
    except ImportError:                     # a Python built without it reads no Hermes record
        return out
    if db.is_file():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
            try:
                out.update(r[0] for r in con.execute(
                    "SELECT DISTINCT cwd FROM sessions WHERE cwd IS NOT NULL AND cwd != ''"))
            finally:
                con.close()
        except sqlite3.Error:
            pass
    return out


def untrusted_stores():
    """Directories holding a manifest that is not trusted, found on the session
    record's working directories and their parents up to the home directory."""
    found, seen = set(), set()
    for cwd in session_dirs():
        d = Path(cwd)
        for p in (d, *d.parents):
            if p in seen:
                break
            seen.add(p)
            if p == HOME or p == p.parent:
                break
            if has_manifest(p) and not _granted(p):
                found.add(p)
    found.update(r for r in registry() if has_manifest(r) and changed_since_trust(r))
    return sorted(found)


def human_confirms(d):
    """True only when a person at a terminal types `yes`. Trust turns a passive
    manifest into one that can block tool calls and read files outside its
    directory, so it is never granted on an agent's word: a hook-run agent has
    no terminal, and a harmless-looking `locket init .` planted in a README must
    not be enough. A determined agent can fake a terminal (`script`, a pty); that
    is a deliberate circumvention, the same class as editing the registry by hand,
    which the write gate refuses separately."""
    return person_confirms(
        f"Trust {d} as a Locket store? Its manifest can then add trigger rows that refuse "
        f"tool calls, ledger schemas and source files. Only trust a directory you made or have read.",
        trust_summary(d), "trust it")


def person_confirms(question, details, act):
    """True only when a person at a terminal types `yes` to `question`. The one
    gate for every setting an agent must not be able to change on its own word."""
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return False
    say = lambda line: print(line, file=sys.stderr)   # one stream for the whole exchange
    say(question)
    for line in details:
        say(f"  {relay(line)}")
    sys.stderr.write(f"Type yes to {act}: "); sys.stderr.flush()
    try:
        return sys.stdin.readline().strip() == "yes"
    except (EOFError, OSError):
        return False


def trust_summary(d):
    """What trusting `d` would switch on, one line each, read from the files
    themselves, so the person deciding sees the rows and paths, not a promise."""
    m = _manifest_data(Path(d))
    out = [f"{manifest_file(Path(d)).name}: keys {', '.join(sorted(m)) or 'none'}"]
    for e in m.get("sources") or []:
        if isinstance(e, dict):
            out.append(f"source file: {e.get('path')}")
    for e in m.get("ledgers") or []:
        if isinstance(e, dict):
            out.append(f"ledger schema for: {e.get('path')}")
    rows_name = m.get("triggers") if isinstance(m.get("triggers"), str) else "triggers.json"
    rows = Path(d) / Path(rows_name).expanduser()
    try:
        data = json.loads(read_safe(rows))
        rws = [r for r in (data.get("rows") or []) if isinstance(r, dict)] if isinstance(data, dict) else []
        out.append(f"trigger rows: {len(rws)} in {rows}, "
                   f"{sum(1 for r in rws if r.get('block'))} of them blocking")
        for r in rws[:12]:
            out.append(f"{'BLOCKS' if r.get('block') else 'asks'} [{r.get('id', '?')}] when "
                       f"{r.get('tools')!s} matches /{r.get('content')!s}/: {r.get('question')!s}")
    except (OSError, ValueError):
        pass
    return out


def grant_trust(d, confirm=None):
    """Register `d` once `confirm(d)` says a person agreed, printing the outcome.
    True when `d` is trusted afterwards."""
    d = Path(d).expanduser().resolve()
    if _granted(d):
        print(f"registry: {paths.PATHS.registry} (already listed)" if any(_same(d, r) for r in registry())
              else f"trust: {d} is a store by convention on this host")
        return True
    changed = changed_since_trust(d)
    if not (confirm or human_confirms)(d):
        why = " Its configuration changed since it was trusted." if changed else ""
        print(f"trust: not granted.{why} Trust is given by a person at a terminal, not by an "
              f"agent; ask the user to run:\n  locket trust {shlex.quote(str(d))}")
        return False
    added = register(d)
    print(f"registry: {paths.PATHS.registry} (" + ("added" if added else "trust renewed for its current configuration"
                                        if changed else "already listed") + ")")
    return True


def migrate(root):
    """Rename a store's `memfind.json` to `locket.json` in place. Returns
    "migrated", "already" or "none". Content untouched, so a store under git
    sees one rename."""
    root = Path(root).expanduser().resolve()
    if (root / MANIFEST).is_file():
        return "already"
    old = root / OLD_MANIFEST
    if not old.is_file():
        return "none"
    repin = _granted(root) and any(_same(root, r) for r in registry())
    old.rename(root / MANIFEST)
    if repin:
        register(root)
    settings.cache_clear()
    return "migrated"


def forget(root):
    """Drop a path from the registry. True when a row was removed. The manifest
    and the cache inside the directory are the user's to delete; this only
    stops the listing naming it."""
    root = Path(root).expanduser().resolve()
    rows = registry()
    keep = [r for r in rows if not _same(root, r)]
    if len(keep) == len(rows):
        return False
    _write_registry(keep)
    return True


def _raw_corpora():
    """Every physical store on the machine, joined or not: each host store that
    exists, one per Claude Code project, and every registered directory whose
    manifest is still there. A registered store is keyed by the `name` in its
    manifest, else its basename, else its full path where either collides. A
    project store is keyed by its manifest's `name` too, else its slug, which
    `resolve` still accepts; `init` writes the name and never registers one."""
    out = {k: v for k, v in paths.PATHS.host_stores.items() if v.is_dir()}
    reg = [p for p in registry() if has_manifest(p)]
    # a project's name yields to a host key and to a registered store's name, which
    # a `belongs_to` elsewhere may already point at
    taken = set(paths.PATHS.host_stores) | {_read_manifest(p).get("name") for p in reg}
    for d in sorted((CLAUDE / "projects").glob("*/memory")):
        name = _read_manifest(d).get("name")
        ok = isinstance(name, str) and name and name not in taken and name not in out
        out[name if ok else d.parent.name] = d
    for p in reg:
        if p in out.values():
            continue
        name = _read_manifest(p).get("name")
        key = name if isinstance(name, str) and name else p.name
        out[key if key not in out else str(p)] = p
    return out


def all_corpora():
    """Every corpus a name can address: the physical stores, less each one that
    `belongs_to` another. A joined store is reached through its parent's name
    and counted in the parent's files, so it is listed once, as the parent."""
    raw = _raw_corpora()
    return {k: v for k, v in raw.items() if parent_of(v, raw) is None}


def _by_name(raw, name):
    """The store `raw` keys as `name`, or the project store whose slug it is: a
    `belongs_to` written by slug keeps resolving once that store takes a name."""
    hit = raw.get(name)
    if hit is None and name.startswith("-"):
        hit = next((d for d in raw.values() if d.name == "memory" and d.parent.name == name), None)
    return hit


def parent_of(root, raw=None):
    """The store `root` declares it belongs to, or None. `belongs_to` in the
    manifest names a corpus as `corpora` lists it; a name that resolves to no
    store, to the store itself, to a store it physically contains, to a store
    containing it inside that store's walk, or into a cycle, is ignored and the
    store stands alone, because this runs inside the write-time hook and a bad
    manifest must degrade. A store under its parent but outside the parent's
    members (a host home's own auto-memory under `projects/`) joins."""
    root = Path(root)
    name = _read_manifest(root).get("belongs_to")
    if not isinstance(name, str) or not name:
        return None
    raw = raw if raw is not None else _raw_corpora()
    target = _by_name(raw, name)
    if target is None or target == root or root in target.parents or _walked_by(target, root):
        return None
    seen, cur = {root}, target                  # a cycle means nobody is the top
    while cur is not None:
        if cur in seen:
            return None
        seen.add(cur)
        nxt = _read_manifest(cur).get("belongs_to")
        cur = _by_name(raw, nxt) if isinstance(nxt, str) else None
    return target


def _walked_by(outer, inner):
    """Whether store `outer`'s own walk reaches directory `inner`: it sits under
    `outer` on disk, inside a member directory and no excluded or dot-directory.
    Only such a nesting would score a file against two stores; one outside the
    walk is a separate store that may join."""
    outer, inner = Path(outer), Path(inner)
    if outer not in inner.parents:
        return False
    rel = inner.relative_to(outer).parts
    members, excluded = _own_walk(outer)
    if members is not None and rel[0] not in members:
        return False
    return not any(q in excluded or q.startswith(".") for q in rel)


def top_of(root):
    """The corpus a physical store is part of: itself, or the store at the end of
    its `belongs_to` chain. Every resolver returns this, so a write into a joined
    store, the cwd default and a name all land on the one corpus."""
    root = Path(root)
    raw = _raw_corpora()
    while True:
        up = parent_of(root, raw)
        if up is None:
            return root
        root = up


def children_of(root, raw=None):
    """Physical stores whose `belongs_to` resolves to `root`, sorted."""
    raw = raw if raw is not None else _raw_corpora()
    root = Path(root)
    return sorted(r for r in set(raw.values()) if r != root and parent_of(r, raw) == root)


def roots_of(root):
    """Every physical directory composing the corpus `root`: itself, then each
    store that belongs to it, recursively. A store is one of these to itself."""
    raw = _raw_corpora()
    out, todo = [], [Path(root)]
    while todo:
        r = todo.pop(0)
        if r in out:
            continue
        out.append(r)
        todo.extend(children_of(r, raw))
    return out


def _label(root):
    """What a joined store's files are prefixed with in every report and index
    key: the manifest's `name`, else the directory's basename."""
    name = _read_manifest(root).get("name")
    return name if isinstance(name, str) and name else Path(root).name


def corpus_key(root, path, roots=None):
    """The corpus-relative key of a file: its path under `root`, or under a joined
    store as `<label>/<path>`. One key per file across every mode and the index,
    because a joined store's files are not under the root on disk. A loop over a
    store's files passes `roots=roots_of(root)` once: each call otherwise re-reads
    the registry and every manifest, which was 1.9 of a 2.6 s hook on a store
    whose files sit mostly in a joined auto-memory."""
    root, path = Path(root), Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        pass
    for r in (roots_of(root) if roots is None else roots)[1:]:
        try:
            return f"{_label(r)}/{path.relative_to(r)}"
        except ValueError:
            continue
    return str(path)


def resolve(name):
    """A corpus path from a shorthand, a project-slug substring, or a literal path.

    Returns None and prints the candidates where the name matches no corpus or
    more than one. The substring matches against the slugified project directory
    (`-Users-me-Software-myapp-ios`), so `myapp`
    resolves it. A literal path comes back ABSOLUTE: `corpus_for` compares
    resolved parents, so a relative path handed on unresolved reported the
    write-time hook as NOT PROTECTED over a directory it guards.
    """
    if name in paths.PATHS.host_stores:
        return top_of(paths.PATHS.host_stores[name])
    p = Path(name).expanduser()
    if p.is_dir():
        return top_of(p.resolve())
    listed, low = all_corpora(), name.lower()
    # A Claude Code project store also answers to its slug, though a named one is
    # listed only by name. The passes run exact key, exact slug, key substring,
    # slug substring, first non-empty wins: an exact slug then beats a longer
    # sibling slug containing it (`-…-DevDocs` opening `-…-DevDocs-Persona`), and
    # `devdocs` beside `devdocs-persona` is not ambiguous. Slugs come only from
    # `projects/`, or a registered `~/vault/memory` would answer to `vault`.
    slugs = {d.parent.name: d for d in listed.values() if d.parent.parent == CLAUDE / "projects"}
    exact, part = (lambda k: k.lower() == low), (lambda k: low in k.lower())
    hits = {}
    for keyed, test in ((listed, exact), (slugs, exact), (listed, part), (slugs, part)):
        hits = {k: v for k, v in keyed.items() if test(k)}
        if hits:
            break
    if len(hits) == 1:
        return next(iter(hits.values()))
    if not hits:
        print(f"no corpus matches {name!r}. Known:", file=sys.stderr)
    else:
        print(f"{name!r} is ambiguous:", file=sys.stderr)
    for k, v in sorted((hits or all_corpora()).items()):
        print(f"  {k:70s} {len(md_files(v)):4d} files", file=sys.stderr)
    return None


def corpus_for_cwd(cwd):
    """The auto-memory corpus belonging to a working directory, or None.

    Claude Code slugifies the absolute path into `~/.claude/projects/<slug>/memory`,
    replacing every non-alphanumeric character with a dash.
    """
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(Path(cwd).resolve()))
    d = CLAUDE / "projects" / slug / "memory"
    return top_of(d) if d.is_dir() else None


def corpus_for(path):
    """The corpus a written path belongs to, or None.

    Derived from the path rather than matched against a named list, so EVERY
    project's auto-memory is covered, including one that does not exist yet.
    Four shapes qualify: a directory holding a `locket.json` manifest (any host,
    any location), the host stores in `Paths.host_stores`, and Claude Code's
    `~/.claude/projects/<slug>/memory`. A non-markdown path and anything under
    the corpus's excluded directories return None.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if p.suffix.lower() != ".md":           # MEMORY.MD is MEMORY.md on a case-insensitive disk
        return None
    root = _phys_root_of(p.parent)          # membership and exclusion are the physical store's
    if root is None:
        return None
    if not in_members(root, p):
        return None
    between = {q.name for q in p.parents if root in q.parents}
    if settings(root)["excluded_dirs"] & between:
        return None
    if excluded_file(root, p):
        return None
    return top_of(root)                     # the corpus it is scored against


def in_members(root, path):
    """Whether `path` sits at the root or under one of the store's member
    directories. Always true for a store with no `members` (a manifest store,
    a project memory); for a host store, false under `sessions`, `plugins`,
    `projects` and every other directory HOST_MEMBERS does not name."""
    mem = settings(root)["members"]
    if mem is None:
        return True
    rel = Path(path).relative_to(root).parts
    return len(rel) == 1 or rel[0] in mem


def _phys_root_of(d):
    """The physical store that directory `d` is or sits inside, or None: a host
    store, a directory holding a manifest, or Claude Code's
    `~/.claude/projects/<slug>/memory`. Membership, exclusion and the walk are
    the physical store's; the corpus it belongs to is `top_of` it."""
    projects = CLAUDE / "projects"
    stores = set(paths.PATHS.host_stores.values())
    if len(str(d)) > PATH_MAX:              # no file lives there; walking it cost a stat per part
        return None
    for parent in (d, *d.parents):
        if parent in stores:
            return parent
        if has_manifest(parent):
            # An untrusted manifest makes its subtree no store at all, never a
            # member of a trusted store above it: the parent's walk stops here too.
            return parent if trusted(parent) else None
        if parent.name == "memory" and parent.parent.parent == projects:
            return parent
    return None


def _root_of(d):
    """The corpus root that directory `d` is or sits inside, or None. One walker
    for the gate, the CLI default and the MCP, so they cannot disagree about
    what a corpus is; a store that `belongs_to` another resolves to that one."""
    phys = _phys_root_of(d)
    return None if phys is None else top_of(phys)


def corpus_here(cwd):
    """The corpus a working directory belongs to: the one it sits inside, else
    Claude Code's auto-memory for it, else None. The default used to be the
    auto-memory alone, so `--index` run inside a manifest directory looked for
    `~/.claude/projects/<slug>/memory` instead: it failed before that folder
    existed and indexed the wrong, empty one once a session had created it."""
    cwd = Path(cwd)
    return _root_of(cwd) or corpus_for_cwd(cwd)


def search_root(words, cwd=None):
    """The corpus the first of a search's path arguments belongs to, or None.

    Each word is read as the shell would expand it (`~`, `$HOME`, a path relative
    to the session's cwd) before it is placed, so no path shape is privileged; a
    regex for paths opening on `/` read `~/.claude/memory` as `/.claude/memory`.
    Only a word naming something that exists counts, cut at its first glob
    character, or its directory where that cut leaves a partial name, so a search
    pattern resolved against the cwd never shadows the path argument after it."""
    for w in words:
        w, cut = re.split(r"[*?\[]", str(w), maxsplit=1)[0], bool(re.search(r"[*?\[]", str(w)))
        if not w or w.startswith("-"):
            continue
        p = Path(os.path.expanduser(os.path.expandvars(w)))
        if not p.is_absolute():
            p = Path(cwd or os.getcwd()) / p
        p = os.path.normpath(p)             # `..` collapsed, or an excluded dir it passes through still counts
        if not os.path.exists(p):           # os.path swallows every OSError, a name too long included
            # only a glob cut may fall back to its directory (`~/brain/fact*.md` cut to `~/brain/fact`);
            # any other missing word, the verb `grep` first, would fall back to the cwd's own store
            p = os.path.dirname(p) if cut else ""
            if not p or not os.path.exists(p):
                continue
        p = Path(p)
        root = corpus_for((p if p.is_dir() else p.parent) / "_probe.md")
        if root is not None:
            return root
    return None


def _read_manifest(root):
    """The manifest as a dict, or {} where it is absent, unreadable or untrusted.
    A broken manifest must degrade to the defaults rather than take a hook down
    with it, and an untrusted one is read as no manifest at all."""
    if root is None or not trusted(root):
        return {}
    return _manifest_data(root)


def _manifest_data(root):
    """The manifest file's own contents, trusted or not: for `init` editing a
    file it is about to rewrite, never for acting on one."""
    m = manifest_file(root)
    if not m.is_file():
        return {}
    try:
        with open_safe(m, binary=True) as f:
            data = json.loads(f.read().decode("utf-8"))     # strict: a garbled name is no name
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):           # JSONDecodeError, and a non-UTF-8 file
        return {}


# The manifest's shape, as JSON Schema, in the one place the reader lives. `schema`
# writes it to ~/.locket/locket.schema.json for an editor to read by filename and
# `validate_manifest` checks a manifest against it at `init`, so an agent authoring
# one by hand gets the same hint an editor would. Every key optional; nothing here
# is required for a directory to be a store.
_STR_LIST = {"type": "array", "items": {"type": "string"}}
MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "locket.schema.json",
    "title": "Locket manifest (locket.json)",
    "description": "Per-corpus configuration for Locket (memscan.py and memfind.py). "
                   "Every key is optional and a missing key takes the tool's default; "
                   "`locket --help` MANIFEST is the authority.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "$schema": {"type": "string", "description": "Editor hint only; ignored by the tool."},
        "_comment": {"type": "string", "description": "Free text for the human reader; ignored by the tool."},
        "name": {"type": "string",
                 "description": "What do `corpora` and a store argument call this store? Default: the "
                                "directory's basename. Host stores and project memories ignore it."},
        "members": {**_STR_LIST,
                    "description": "Which top-level directories are corpus? The rest of the root is "
                                   "host state. Root-level markdown is always in. Replaces the default."},
        "excluded_dirs": {**_STR_LIST,
                          "description": "Which directories, at any depth, are not corpus? Replaces the "
                                         "default (archive, in-flight, harness-logs, discovery-research)."},
        "excluded_files": {**_STR_LIST,
                           "description": "Which single files are not corpus? Corpus-relative paths or "
                                          "globs, invisible to every mode, the index and the write-time "
                                          "gate. For a file that holds no claims, such as a draft or a template."},
        "index_files": {**_STR_LIST,
                        "description": "Which files POINT at other files? A line restating its target "
                                       "is not a fork. Surface role, by basename; replaces the default."},
        "ledger_surfaces": {**_STR_LIST,
                            "description": "Which markdown files grow by appending entries? Never scored "
                                           "for forks. Not `ledgers`, which checks CSV row shape. Surface "
                                           "role, by basename; replaces the default."},
        "holding_spaces": {**_STR_LIST,
                           "description": "Which files stage entries on their way to a durable home? "
                                          "Scored only against themselves, yet fully corpus, unlike an "
                                          "excluded file. Surface role, by basename; replaces the default."},
        "whole_read_exempt": {**_STR_LIST,
                              "description": "Which files may outgrow a whole read? The size sweep skips "
                                             "them and nothing else does: a mixed surface never read whole "
                                             "by its own contract, or a file a project instrument already "
                                             "caps. Default: none."},
        "code_roots": {**_STR_LIST,
                       "description": "Where do `cites` path:line citations resolve? Directories, "
                                      "absolute or relative to the store."},
        "sources": {
            "type": "array",
            "description": "Which CSVs hold claims? Their rows are scored for forks like "
                           "any sentence in the corpus.",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["path", "text"],
                      "properties": {
                          "path": {"type": "string", "description": "The CSV, absolute or store-relative."},
                          "text": {"type": "string", "description": "Column holding the claim sentence."},
                          "label": {"type": "string", "description": "Column that names a row in output (an Area)."}}}},
        "ledgers": {
            "type": "array",
            "description": "What shape must a CSV ledger's rows take? A row outside its "
                           "schema is refused at write time; forks are `sources`' business.",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["path"],
                      "properties": {
                          "path": {"type": "string", "description": "Basename glob or store-relative path of the ledger."},
                          "extends": {"type": "string",
                                      "description": "A built-in's glob this entry merges onto though its path falls outside it."},
                          "columns": {**_STR_LIST, "description": "The exact header, in order."},
                          "required": {**_STR_LIST, "description": "Columns that may not be empty."},
                          "caps": {"type": "object", "additionalProperties": {"type": "integer"},
                                   "description": "{column: maximum characters}."},
                          "enum": {"type": "object", "additionalProperties": _STR_LIST,
                                   "description": "{column: allowed values}; include \"\" where empty is allowed."},
                          "pattern": {"type": "object", "additionalProperties": {"type": "string"},
                                      "description": "{column: regex the whole value must match}."}}}},
        "triggers": {"type": "string",
                     "description": "Which rows file does `locket trigger` read for this store? Store-relative "
                                    "or absolute. Default: triggers.json at the store root."},
        "belongs_to": {"type": "string",
                       "description": "Which corpus does this directory JOIN? Its name as `corpora` lists it; "
                                      "written by `init <dir> --parent <name>`."},
    },
}


def cmd_schema(quiet=False):
    """Write the manifest schema beside the registry, so an editor can be pointed
    at one predictable path per machine and the tool can check a manifest against
    the same file. Rerun after an update; `init` runs it."""
    schema = paths.PATHS.schema_file
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(json.dumps(MANIFEST_SCHEMA, indent=2) + "\n")
    if not quiet:
        print(f"schema: {schema} (written)")
        print(f"editor: VS Code reads it by filename through one user-settings entry; "
              f"`locket help install` (Later) names it")
    return 0


def _check(value, schema, path, out):
    """Structural check of `value` against the manifest schema's subset of JSON Schema,
    through panoply-lib's checker, a generated copy beside this file."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import _manifest_schema
    _manifest_schema.check_value(value, schema, path, out)


def validate_manifest(root):
    """Findings for the manifest at `root` against MANIFEST_SCHEMA; [] when clean
    or absent. Structure only: a key the reader would silently ignore, a wrong
    type, a malformed source or ledger entry. Whether the values are right is
    the modes' business."""
    m = manifest_file(root)
    if not m.is_file():
        return []
    try:
        data = json.loads(read_safe(m))
    except (json.JSONDecodeError, OSError) as e:
        return [f"manifest does not parse: {e}"]
    out = []
    _check(data, MANIFEST_SCHEMA, "", out)
    for i, e in enumerate(data.get("ledgers") or [] if isinstance(data, dict) else []):
        x = e.get("extends") if isinstance(e, dict) else None
        if x is not None and x not in LEDGER_SCHEMAS:
            out.append(f"ledgers[{i}].extends: {x!r} is no built-in; one of {', '.join(LEDGER_SCHEMAS)}")
    return out


def _inside(rel):
    """Whether a manifest's relative path stays inside its store: not absolute, no
    `..`. A member `/` walked the whole disk inside the write hook."""
    p = PurePosixPath(str(rel).replace("\\", "/"))
    return bool(str(rel).strip()) and not p.is_absolute() and ".." not in p.parts and not str(rel).startswith("~")


def _own_walk(root, m=None):
    """-> (members, excluded_dirs): what one store's own walk covers, from its own
    manifest alone. Never unioned across a join, so `parent_of` can ask it of a
    parent without recursing through `settings`, which unions name-sets upward."""
    m = _read_manifest(root) if m is None else m
    mem = m.get("members")
    members = ({str(x) for x in mem if isinstance(x, str) and _inside(x)} if isinstance(mem, list)
               else (set(HOST_MEMBERS[Path(root)]) if root is not None
                     and Path(root) in HOST_MEMBERS else None))
    val = m.get("excluded_dirs")
    excluded = {str(x) for x in val} if isinstance(val, list) else set(EXCLUDED_DIRS)
    excluded.add(".memfind")                  # the cache is never prose, whatever the manifest says
    if root is not None and Path(root) in HOST_EXCLUDED:
        excluded |= HOST_EXCLUDED[Path(root)]
    excluded.update({"backups", "backup"})    # nor is a backup, which forks by design
    excluded |= DEPENDENCY_DIRS               # nor a dependency or build tree, whatever the manifest says
    return members, excluded


def settings(root):
    """`_settings_for`, cached per store and per state of the trust registry, so a store
    trusted or untrusted in another process is seen without a restart."""
    _trusted_ids()
    pinned = root is not None and _inode(root) in _TRUST_CACHE[next(iter(_TRUST_CACHE))][1]
    return _settings_for(root, (next(iter(_TRUST_CACHE)), config_digest(root) if pinned else None))


def _clear_settings():
    _settings_for.cache_clear()


settings.cache_clear = _clear_settings      # every caller's reset still works


@lru_cache(maxsize=None)
def _settings_for(root, _trust_key):
    """Per-corpus name-sets, the module defaults overridden by the manifest.

    Every place that once read a module-level set reads this instead, keyed on
    the corpus, so two corpora on one machine can name their ledgers differently.
    The defaults are this apparatus's file names and are what a corpus without a
    manifest gets, which keeps the Claude Code install zero-config.
    """
    m = _read_manifest(root)
    members, excluded = _own_walk(root, m)
    # A host store whose files are not this apparatus's: Hermes's MEMORY.md is
    # the whole memory, not an index over topic files, and Codex writes its
    # memories/ itself, so neither inherits the index exemption. A manifest
    # there still wins.
    if root is not None and Path(root) in (paths.PATHS.host_stores.get("hermes"), paths.PATHS.host_stores.get("codex")):
        m = {"index_files": [], "ledger_surfaces": [], "holding_spaces": [], **m}
    out = {}
    # `members`: the top-level directories that ARE corpus, everything else under
    # the root being host state. None means every directory, which is what a
    # manifest store or a project memory gets; a host store takes HOST_MEMBERS.
    out["members"] = members
    for key, default in (("index_files", INDEX_FILES),
                         ("ledger_surfaces", LEDGER_SURFACES),
                         ("holding_spaces", HOLDING_SPACES),
                         ("whole_read_exempt", ())):
        val = m.get(key)
        out[key] = ({str(x) for x in val} if isinstance(val, list) else set(default))
    out["excluded_dirs"] = excluded
    ef = m.get("excluded_files")
    out["excluded_files"] = ([str(x) for x in ef if isinstance(x, str)]
                             if isinstance(ef, list) else [])
    out["append_surfaces"] = out["ledger_surfaces"] | out["holding_spaces"]
    cr = m.get("code_roots")
    out["code_roots"] = ([Path(x).expanduser() for x in cr if isinstance(x, str)]
                         if isinstance(cr, list) else None)
    # A joined store keeps its own members, exclusions and code roots for its own
    # walk; the NAME-sets union upward, so the gate asked about a file by name
    # (an index line, a holding-space entry) answers the same at the top as it
    # would have in the store the file physically sits in.
    for child in children_of(root) if root is not None else []:
        cs = settings(child)
        for key in ("index_files", "ledger_surfaces", "holding_spaces", "append_surfaces",
                    "whole_read_exempt"):
            out[key] = out[key] | cs[key]
        out["excluded_files"] = out["excluded_files"] + [x for x in cs["excluded_files"]
                                                         if x not in out["excluded_files"]]
    return out
# Files whose job is to POINT at other files. A hook line restating its target is
# doing its job; only a FALSE, decaying or uninformative restatement is a defect.
# SELF.md and SHORT_TERM.md are deliberately absent: they carry no index lines, and
# treating them as indexes mislabels the promotion-flow duplication between them.
INDEX_FILES = {"MEMORY.md", "ARCHIVE.md", "CURATION-INDEX.md", "CURATION-SOURCES-INDEX.md",
               "FRONTEND-INDEX.md", "PLATFORM-INDEX.md"}
# Files that accumulate ENTRIES rather than stating facts. Appending to one is the
# discipline rather than a breach of it, so both sets are exempt from the ordinary
# corpus-wide nudge. The stake is higher than noise: the hook exits 2, and the
# cycles write these on a schedule with no session in front of them, so a wrong
# nudge here is a blocked cycle that cannot report its own failure.
#
# LEDGER: chronology is the whole value, entries repeat by design, nothing is
# checked. HOLDING: a staging area whose entries are promoted or discarded after
# observation, so a DUPLICATE ENTRY is a defect (Peer ruling 2026-08-15) while a
# new one is the file doing its job. A holding space is therefore scored against
# ITSELF and nothing else: restating an item it already holds fires, and turning
# over an observation about a fact that lives in MEMORY.md does not.
LEDGER_SURFACES = {"HARNESS.md", "CHECKPOINTS.md", "EXPLORATIONS.md"}
HOLDING_SPACES = {"SHORT_TERM.md", "OPEN_QUESTIONS.md", "MEMORY-proposed.md",
                  "PROMOTIONS.md"}

STOP = set("""a an the and or but if then than that this these those is are was were be been being
of in on at to for with from by as it its not no we i you they our your their my what which who
when where how why can could should would will may might must do does did done so such very more
most other some any each all both few own same too don now here there into over under again
further once about against between through during before after above below up down out off only
just also because while has have had having me him her them us""".split())
WORD = re.compile(r"[a-z0-9_./-]{3,}")
SENT = re.compile(r"(?<=[.!?;:])\s+|\n")
# A hard-wrapped sentence is one sentence. `SENT` splits on every newline, so
# before it runs a wrapped line is joined to the one before it wherever the
# earlier line does not end a sentence and the later one is not line-meaningful
# markdown (a bullet, a heading, a quote, a table row, a numbered item, a fence,
# a blank). Measured before the join on a hard-wrapped store: 30% of its claims
# stopped at the wrap, and a tail under MIN_TOKENS was in no claim at all, so a
# promotion paraphrased across a wrap was invisible to `graduated` and `across`.
# The version rides in memfind's fingerprint, so a change here re-embeds every
# store once instead of leaving stale claims under an unchanged file.
CLAIMS_VERSION = 2
_ENDS = re.compile(r"""[.!?;:]["')\]*_`]*\s*$""")
_LINE_MEANINGFUL = re.compile(r"""\s*(?:[-*+]\s|#|>|\||\d+[.)]\s|```|$)""")
_NO_JOIN_FROM = re.compile(r"""\s*(?:#|\||```|$)""")   # a bullet's continuation is still its sentence


def unwrap(txt):
    """`txt` with hard wraps inside a sentence joined by a space. Linear: a joined
    line is kept as its segments and joined once, and only its tail is tested for an
    ending, because rebuilding and re-searching the whole line on every wrap made one
    unpunctuated 200 KB file cost seconds inside the write hook."""
    out, joinable = [], False               # each line as its segments; may the last take a wrap
    for ln in txt.split("\n"):
        if (joinable and not _ENDS.search(_tail(out[-1]))
                and ln.strip() and not _LINE_MEANINGFUL.match(ln)):
            out[-1][-1] = out[-1][-1].rstrip()
            out[-1].append(" " + ln.strip())
        else:
            out.append([ln])
            # decided once per line: its start never changes as wraps join onto it
            joinable = bool(ln.strip()) and not _NO_JOIN_FROM.match(ln)
    return "\n".join("".join(c) for c in out)


def _tail(segments, width=256):
    """The last `width` or more characters of a line held as segments: all `_ENDS`
    needs, since an ending is a stop followed only by quotes and space."""
    parts, n = [], 0
    for seg in reversed(segments):
        parts.append(seg)
        n += len(seg)
        if n >= width:
            break
    return "".join(reversed(parts))
LINK = re.compile(r"\[\[([A-Za-z0-9_\-]+)\]\]")
# an index line: "- [Title](target.md) — hook text"
IDX_LINE = re.compile(r"^\s*[-*]\s*\[(?P<title>[^\]]+)\]\((?P<target>[a-z0-9_\-]+\.md)\)(?P<hook>.*)$")

MIN_TOKENS = 5          # content tokens a sentence needs to be a claim. Not 7: that
                        # drops short pithy rules like 'exactly two paragraphs'
PAIR_FLOOR = 0.55       # jaccard over claim tokens
BUCKET_CAP = 400        # a token in more claims than this is not a signal
MIN_SHARED = 5          # content tokens two claims must actually share
ENTRY_TOKENS = 8        # content tokens a line needs to BE an entry, not a stem


def similar(ta, tb):
    """Jaccard over two token sets, or 0.0 where they share fewer than MIN_SHARED.

    A ratio alone cannot tell a restated fact from two short lines ending in the
    same attribution: `claims` splits on the colon, so a heading arrives as a
    six-token claim whose overlap is mostly a date and the word `ruling`, and
    scores 0.50 against an unrelated heading. A verbatim five-token claim still
    fires, because it shares all five.
    """
    shared = len(ta & tb)
    return shared / len(ta | tb) if shared >= MIN_SHARED else 0.0


def strip(txt):
    """Body text with frontmatter, code fences and markdown furniture removed."""
    if txt.startswith("---"):
        end = txt.find("\n---", 3)
        if end != -1:
            txt = txt[end + 4:]
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = unwrap(txt)                  # before the furniture goes: a heading is still a heading here
    # wikilinks go entirely, not just their brackets: [[a_b_c]] survives
    # bracket-stripping as the word sequence "a b c" and pairs at 1.00 against
    # every other mention of the same slug. A pointer is not a claim.
    txt = re.sub(r"\[\[[^\]]+\]\]", " ", txt)
    return re.sub(r"[*_`#>|\[\]]", " ", txt)


def tokens(s):
    return {w for w in WORD.findall(s.lower()) if w not in STOP}


def claims(txt):
    """(sentence, token-set) for every sentence carrying enough to be a claim."""
    for s in SENT.split(txt):
        s = " ".join(s.split())
        ts = tokens(s)
        if len(ts) >= MIN_TOKENS:
            yield s, ts


def unclaimed(txt):
    """The sentences `claims` drops as under the floor, with at least one
    content token. What the instrument cannot see is reported beside what it
    saw, because a check cannot name the claims it never formed and a clean
    result over a corpus it half-indexed reads exactly like a clean corpus."""
    for s in SENT.split(txt):
        s = " ".join(s.split())
        ts = tokens(s)
        if 0 < len(ts) < MIN_TOKENS:
            yield s


def md_files(root):
    """Every markdown file the corpus contains: the root's own, subdirectories
    included, then each joined store's own. Sorted by path, so a joined store's
    files sit together.

    ⚠ This must agree with `corpus_for`, which walks PARENTS and therefore covers
    subdirectories. A flat `glob("*.md")` here scores a subdirectory at write time
    against a corpus it is not a member of, so forks inside it are invisible while
    the nudge still fires on it. One walker, both ends.
    """
    return sorted(p for r in roots_of(root) for p in _own_md(r))


def _own_md(root, pattern="*.md", honor_excluded_files=True):
    """The markdown physically under one store, by that store's own settings.
    `pattern` walks another file type the same way (the ledger sweep takes
    `*.csv`), and a sweep that must see a file the manifest hides from claims
    passes `honor_excluded_files=False`."""
    root = Path(root)
    return sorted(p for p in _walk_files(root, pattern)
                  if not (honor_excluded_files and excluded_file(root, p)))


def _corpus_dirs(root):
    """Every directory one store's own walk enters: the root, then each member
    directory and everything below it, never descending into a dot-directory,
    an excluded directory or a nested store. Yields (dir, filenames).

    ⚠ Pruned on the way DOWN, never filtered after: the hook pays this walk on
    every memory write under a 10-second budget, and a host home holds tens of
    thousands of directories of session and job state beside a corpus of a few
    hundred files. A walk that entered them and filtered afterwards took 25 s on
    one council store, past the timeout, so the gate failed open. Anything that
    asks "what is under this store" walks through here, so what is skipped stays
    one decision.

    A subdirectory carrying its own manifest is a corpus of its own: `_root_of`
    resolves a write inside it to the inner store, so its files are not this
    store's. `init` refuses to nest, but a manifest written by hand can. A
    listed store below this one (a project memory under the council home carries
    no manifest) belongs to itself too, whatever the member list says.

    ⚠ An unlistable directory yields nothing and its siblings still walk: a host
    root that is searchable but not listable must not take its members with it,
    or every write into the store passes unscored."""
    root = Path(root)
    s = settings(root)
    skip, mem = s["excluded_dirs"], s["members"]
    listed = {r for r in _raw_corpora().values() if r != root and root in r.parents}

    def entered(name):
        return name not in skip and not name.startswith(".")

    if mem is None:
        tops = [root]
    else:
        # The root's own files count; its other directories are host state.
        own = []
        try:
            with os.scandir(root) as it:
                for e in it:
                    try:
                        if not e.is_dir():
                            own.append(e.name)
                    except OSError:         # one entry that cannot be stat'd drops only itself
                        continue
        except OSError:
            pass
        own.sort()
        yield root, own
        # a member that is a symlink would walk wherever it points, outside the store
        real_root = os.path.realpath(root)
        tops = [root / n for n in sorted(mem)
                if all(entered(part) for part in Path(n).parts)
                and (root / n).is_dir() and not (root / n).is_symlink()
                # no component may be a link out: `link/sub` walked wherever `link` pointed
                and os.path.realpath(root / n).startswith(real_root + os.sep)]
    for top in tops:
        for dirpath, dirnames, filenames in os.walk(top):
            here = Path(dirpath)
            if here != root and (here in listed or MANIFEST in filenames
                                 or OLD_MANIFEST in filenames):
                dirnames[:] = []
                continue
            dirnames[:] = sorted(d for d in dirnames if entered(d))
            yield here, sorted(filenames)


def has_content(root, csv=True):
    """Whether a store, its joined stores included, holds anything to score: a non-empty
    markdown file its manifest does not exclude, or with `csv`, a non-empty registered CSV
    source. Stops at the first one, so a store with memory costs almost nothing to ask.
    `csv=False` asks what the meaning index can build from, which is markdown only."""
    def nonempty(p):
        try:
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False
    return any(any(nonempty(p) and not excluded_file(r, p) for p in _walk_files(r))
               or (csv and any(nonempty(Path(pth)) for pth, _t, _l in csv_sources(r)))
               for r in roots_of(root))


MAX_FILE_BYTES = 8 * 1024 * 1024      # past the 5 MB a declared CSV is reported at
MAX_SCAN = 64 * 1024                  # model-written text a pattern is run over, at most
PATH_MAX = 4096                       # Linux's; longer names no file on any host we serve
MAX_STORE_FILES = 4000
MAX_STORE_BYTES = 64 * 1024 * 1024


def _regular(p):
    """A plain file, by `lstat`: no symlink, no FIFO, no device, one hard link. A
    store is read only through files it holds itself, so a link planted in a cloned
    repository cannot make the walk read a key or a config from elsewhere on disk."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_nlink == 1


def open_safe(path, binary=False, newline=None, max_bytes=MAX_FILE_BYTES):
    """`open` for a file a store controls, checked on the descriptor that is read:
    refused unless it is a regular file of one link under `max_bytes` (None for no
    cap), so a swap between the walk's check and this read gains nothing, and a FIFO
    never blocks it. `newline=""` for a CSV; the default translates CRLF as
    `read_text` did. Raises OSError on refusal."""
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1
                or (max_bytes is not None and st.st_size > max_bytes)):
            raise OSError(f"{path}: not a plain file" + (f" under {max_bytes} bytes" if max_bytes else ""))
        return (os.fdopen(fd, "rb") if binary
                else os.fdopen(fd, "r", newline=newline, errors="ignore", encoding="utf-8"))
    except BaseException:
        os.close(fd)
        raise


def read_safe(path):
    """A corpus file's text by `open_safe`; raises OSError on refusal."""
    with open_safe(path) as f:
        return f.read()


def read_or_empty(path):
    """`read_safe`, or "" for a file that is gone, unreadable or refused."""
    try:
        return read_safe(path)
    except OSError:
        return ""


def skipped_files(root):
    """Files Locket would read in one store but passes over, as (path, why): markdown
    the walk meets, and the CSVs the manifest declares, each a symlink, a hard link,
    not a file, or (markdown only) past MAX_FILE_BYTES. For `doctor`, so a skip is
    reported once rather than reading as a store that holds nothing there."""
    out = []
    declared = [(e[0], True) for e in _own_csv_sources_unchecked(root)]
    walked = [(d / n, False) for d, names in _corpus_dirs(root) for n in names if fnmatch.fnmatch(n, "*.md")]
    for p, is_csv in walked + declared:
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if stat.S_ISLNK(st.st_mode):
            out.append((p, "a symlink"))
        elif not stat.S_ISREG(st.st_mode):
            out.append((p, "not a plain file"))
        elif st.st_nlink != 1:
            out.append((p, "a hard link"))
        elif st.st_size > MAX_FILE_BYTES and not is_csv:
            out.append((p, f"over {MAX_FILE_BYTES // (1024 * 1024)} MB"))
    return out


def _own_csv_sources_unchecked(root):
    """The declared CSV paths, links included, so `skipped_files` can name the ones
    `_own_csv_sources` drops."""
    entries = _read_manifest(root).get("sources")
    out = []
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and isinstance(e.get("path"), str):
            pth = Path(e["path"]).expanduser()
            out.append((pth if pth.is_absolute() else Path(root) / pth,))
    return out


def _walk_files(root, pattern="*.md"):
    """Every readable file under one store matching `pattern`, by the walk above.
    The match follows the platform's case rule, as `Path.glob` does. Only plain
    files (`_regular`): a symlink, dangling or not, is dropped, and so is a hard
    link. The walk stops past MAX_STORE_FILES or MAX_STORE_BYTES and records it."""
    count = size = 0
    for d, names in _corpus_dirs(root):
        for n in names:
            p = d / n
            if not (fnmatch.fnmatch(n, pattern) and _regular(p)):
                continue
            count += 1
            try:
                size += os.lstat(p).st_size
            except OSError:
                continue
            if count > MAX_STORE_FILES or size > MAX_STORE_BYTES:
                health_error("walk", "bound", OSError(
                    f"{root}: walk stopped past {MAX_STORE_FILES} files or {MAX_STORE_BYTES} bytes"))
                return
            yield p


def excluded_file(root, path):
    """Whether the manifest's `excluded_files` names this file. Each pattern is
    tried against the corpus-relative posix path and against the basename, so
    `notes/draft.md`, `*.csv` and `scratch-*` each match what they read as. A
    file outside the corpus (a registered CSV elsewhere on disk) is matched on
    its absolute path and its basename. Applied by `md_files`, `csv_sources`,
    `index_hooks` and `corpus_for`, so an excluded file is invisible to every
    mode and to the write-time gate alike."""
    pats = settings(root)["excluded_files"]
    if not pats:
        return False
    p = Path(path)
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        rel = p.as_posix()
    return any(fnmatch.fnmatchcase(rel, pat) or fnmatch.fnmatchcase(p.name, pat)
               for pat in pats)


def csv_sources(root):
    """Reference CSVs registered to this corpus: [(path, text column, label column)].

    From `sources` in `memfind.json`, or the older standalone `reference-sources.json`,
    still read. Each entry is {path, text, label}: `text` names the column holding the
    claim, `label` the column that identifies a row in output (an Area). A missing
    file is skipped and a malformed manifest yields [], because this runs inside the
    write-time hook and a guardrail that dies on a bad file blocks the write it
    exists to advise on.
    """
    out, seen = [], set()
    for r in roots_of(root):
        for pth, text, label in _own_csv_sources(r):
            key = pth.resolve()               # two stores registering one CSV is one source
            if key not in seen:
                seen.add(key)
                out.append((pth, text, label))
    return out


def _own_csv_sources(root):
    """The CSVs one physical store's manifest registers. A relative `path`
    resolves against that store's root."""
    entries = _read_manifest(root).get("sources")
    if not isinstance(entries, list):
        m = Path(root) / LEGACY_MANIFEST
        if not m.is_file() or not trusted(root):
            return []
        try:
            entries = json.loads(read_safe(m))
        except (json.JSONDecodeError, OSError):
            return []
    out = []
    try:
        for e in entries:
            pth = Path(e["path"]).expanduser()
            if not pth.is_absolute():
                pth = Path(root) / pth
            if pth.exists() and not excluded_file(root, pth):
                out.append((pth, e["text"], e.get("label", "")))
    except (KeyError, TypeError):
        return []
    return out


def csv_rows(root):
    """[(label, text)] for every registered CSV row, in file order, labelled
    `<file>.csv#<label value>`. Errors in one file skip that file, never raise."""
    import csv as _csv
    out = []
    for pth, textcol, labelcol in csv_sources(root):
        try:
            with open_safe(pth, newline="", max_bytes=None) as fh:   # declared CSVs are never capped
                for r in _csv.DictReader(fh):
                    t = (r.get(textcol) or "").strip()
                    if not t:
                        continue
                    tag = (r.get(labelcol) or "").strip() if labelcol else ""
                    out.append((f"{pth.name}#{tag}", t))
        except (OSError, _csv.Error, UnicodeDecodeError):
            continue
    return out


def corpus(root, csv=True):
    """{name: stripped body} for every claim source in the corpus.

    The registered CSVs are IN the corpus, grouped one pseudo-document per label
    (`CLAIMS.csv#dev-loop`), so a row copied into prose is a fork `audit` and the
    write-time gate can see. Measured the other way first: a store whose checks
    read markdown alone reported clean while thirty-two of thirty-three CSV rows
    restated a markdown sentence. `memfind` passes `csv=False` and indexes the
    rows as its own per-row units instead, or they would be embedded twice.
    """
    rs = roots_of(root)
    docs = {}
    for p in md_files(root):
        # One unreadable file (mode 000, or deleted since the walk) is skipped: raised,
        # it emptied the whole read inside the hook, whose catch-all let the write through.
        try:
            docs[corpus_key(root, p, rs)] = strip(read_safe(p))
        except OSError:
            continue
    if csv:
        grouped = defaultdict(list)
        for label, text in csv_rows(root):
            grouped[label].append(strip(text))
        for label, texts in grouped.items():
            docs[label] = "\n".join(texts)
    return docs


@lru_cache(maxsize=None)
def archived_claims(root):
    """Token-sets of every claim already filed under `<corpus>/archive/`.

    Read as a PROVENANCE record, never as corpus: `archive` stays in
    EXCLUDED_DIRS, so nothing here is ever scored against or reported as a fork.
    prime-memory-discipline.md §4 requires superseded text be archived BEFORE it is
    rewritten over, so a claim found here was moved or rewritten rather than
    forked, and that is the whole signal.

    ⚠ This makes the archive load-bearing for the write gate. Pruning it, or
    filing a paraphrase where §4 asks for verbatim, silently narrows what the
    gate will forgive.
    """
    out = []
    for p in sorted(_archive_md(root)):
        try:
            body = read_safe(p)
        except OSError:
            continue
        out.extend(ts for _, ts in claims(strip(body)))
    return out


def _archive_md(root):
    """Markdown under every directory named `archive` inside the store's members
    (`memory/archive/` under the council home, `archive/` in a flat store) and
    inside each joined store's, so a move out of any member surface keeps its
    provenance. Skips a nested store, a dot-directory and any other excluded
    directory on the way down."""
    for r in roots_of(root):
        yield from _own_archive_md(r)


def _own_archive_md(root):
    # Each `archive/` is found as a child of a directory the walk enters, because
    # the walk prunes `archive` itself while it stays in `excluded_dirs`. An
    # archive carrying its own manifest is a nested store, not provenance. A host
    # root's own `archive/` is host state like its other directories.
    root = Path(root)
    hosted = settings(root)["members"] is not None
    for d, _ in _corpus_dirs(root):
        if hosted and d == root:
            continue
        a = d / "archive"
        if a.is_dir() and not a.is_symlink() and not has_manifest(a):
            found = []
            for dirpath, dirnames, filenames in os.walk(a):     # never through a linked directory
                found += [Path(dirpath) / n for n in filenames
                          if n.endswith(".md") and _regular(Path(dirpath) / n)]
                if len(found) > MAX_STORE_FILES:
                    health_error("walk", "bound", OSError(f"{a}: archive past {MAX_STORE_FILES} files"))
                    break
            yield from sorted(found[:MAX_STORE_FILES])


def subtract_archived(added, arch):
    """`added` with every claim that is already archived removed.

    The gate asks "is this text novel?" as a proxy for "is this fact new?", and
    the proxy breaks whenever text MOVES. Provenance separates the two, and it is
    a subtraction rather than a new threshold: a pure relocation reduces to
    nothing and exits on the length check without the live corpus ever being
    read, while a move carrying one new sentence is still scored on that
    sentence.

    ⚠ Returns STRIPPED text and strips internally, because `corpus()` stores
    stripped bodies. Comparing raw text against them matches nothing and no-ops
    silently while looking exactly like a working patch, so owning the strip here
    is what makes that mismatch impossible rather than merely tested for.

    Granularity is the CLAIM, which is the unit `score_against` consumes.
    Subtracting whole LINES cannot work at either setting, and both were measured
    against a real relocation: dropping a line on any archived claim carries an
    unrelated new claim out of reach of the scorer, and dropping it only when
    every claim is archived misses a compressed rewrite entirely, where each line
    mixes reused and new material. ⚠ This is why the holding-space path must NOT
    use this function: `_entry_lines` groups by line to keep a dated entry
    attached to its date, and re-emitting claims severs the two.

    Cost is added claims by archived claims, 0.27s against a 6,539-claim archive
    for a 25KB whole-file write and near-nil for an ordinary edit. It grows with
    the archive, which only ever grows.
    """
    if not arch:
        return strip(added)
    return "\n".join(s for s, ts in claims(strip(added))
                     if not any(similar(ts, a) >= FIRE_FLOOR for a in arch))


def _inverted(items, df):
    """Bucket by the 3 rarest tokens, so candidate generation stays near-linear.

    Ties in rarity break on the token itself: broken by set order they followed
    the per-process string hash, so each run bucketed differently and a pair
    found by one `audit` was missing from the next on an unchanged store."""
    inv = defaultdict(list)
    for i, (_, _, ts) in enumerate(items):
        for w in sorted(ts, key=lambda w: (df[w], w))[:3]:
            inv[w].append(i)
    return inv


def cross_file_pairs(docs, floor=PAIR_FLOOR, same_file=False):
    items, df = [], Counter()
    for name, txt in docs.items():
        for s, ts in claims(txt):
            items.append((name, s, ts))
            df.update(ts)
    seen, out = set(), []
    for _, idxs in _inverted(items, df).items():
        if len(idxs) > BUCKET_CAP:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = sorted((idxs[a], idxs[b]))
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                fa, sa, ta = items[i]
                fb, sb, tb = items[j]
                if (fa == fb) != same_file:     # cross-file forks, or within-file dupes
                    continue
                jac = similar(ta, tb)
                if jac >= floor:
                    out.append((jac, fa, fb, sa, sb))
    out.sort(reverse=True)
    return out


# A reference target: a URL, a file with an extension (path included), a path
# anchored at ~, . or / or ending in /, or a ledger query (`Area=substrate`).
# A bare slash is not a path: "AD/mandate" and "CE-525 / CE-525S" are prose, so an
# unanchored extensionless path ("memory/projects") is not a target either. The
# extension list is the memory formats; a generic `.ext` would take "Node.js" and
# "e.g.". Any `k=v` counts, harmless since a claim keeps its tokens past the floor.
# Matched on the stripped sentence, so backticks are gone.
_TARGET = re.compile(r"""https?://\S+|[\w~./-]*\.(?:md|csv|jsonl?|py|sh|txt|toml|ya?ml|pdf)\b|(?<![\w/])[~.]?/[\w./-]+|\b[\w.-]+/(?=\s|$)|\b\w+=[\w.-]+""", re.I)
_REDIRECT = re.compile(r"(?:see|cf\.?|composes)\s")   # a verb whose object is the reference


def pointer_only(sentence):
    """True where a sentence's payload is a cross-reference, not a claim.

    The criterion: an opening redirect ("See X", "Composes Y"), or a sentence naming a
    reference target that is no longer a claim once the target is removed, by
    the same MIN_TOKENS floor `claims` uses. These repeat across files by design
    and are the noise floor of `audit`. It replaced a closed list of opening
    phrases, under which a pointer worded any other way paired as a fork.
    """
    s = sentence.lower().lstrip("-* ")
    if _REDIRECT.match(s):
        return True
    if not _TARGET.search(s):
        return False
    return len(tokens(_TARGET.sub(" ", s))) < MIN_TOKENS


def score_against(text, docs):
    """Rank files by how much of `text` they already assert.

    Returns (combined, claim_score, filename, (incoming, existing)) best first,
    empty where `text` carries no claim.
    """
    incoming = list(claims(strip(text)))
    if not incoming:
        return []
    inc_all = set().union(*(ts for _, ts in incoming))
    ranked = []
    for name, body in docs.items():
        best, best_pair = 0.0, None
        overlap = 0
        for s, ts in claims(body):
            overlap += len(ts & inc_all)
            for isent, its in incoming:
                jac = similar(ts, its)
                if jac > best:
                    best, best_pair = jac, (isent, s)
        # claim-level match dominates; bulk token overlap breaks ties so a
        # paraphrase still surfaces the right topical neighbour.
        ranked.append((best + min(overlap / 400.0, 0.35), best, name, best_pair))
    ranked.sort(reverse=True)
    return ranked


def index_hooks(root):
    """(index key, target key, title, hook text, target path) for every index line
    with a target, in every member directory and joined store. A target resolves
    beside its index file, never at the root: a composite store's `memory/MEMORY.md`
    points into `memory/`."""
    out = []
    index_files = settings(root)["index_files"]
    rs = roots_of(root)
    for p in md_files(root):
        if p.name not in index_files:
            continue
        for line in read_or_empty(p).splitlines():
            m = IDX_LINE.match(line)
            if m:
                hook = m.group("hook").lstrip(" —-").strip()
                if hook:
                    t = p.parent / m.group("target")
                    out.append((corpus_key(root, p, rs), corpus_key(root, t, rs), m.group("title"), hook, t))
    return out


def front_name(txt):
    """The top-level `name:` in a file's frontmatter, or None. Read only inside the
    opening block, so a `name:` line in the body never counts."""
    if not txt.startswith("---\n"):
        return None
    end = txt.find("\n---", 3)
    m = re.search(r"^name:[ \t]*['\"]?([^'\"\n]+?)['\"]?[ \t]*$", txt[4:end] if end != -1 else "", re.M)
    return m.group(1) if m else None


def dangling(root):
    """{target: [files]} for every wikilink resolving to no file. A target resolves
    by file stem or by the frontmatter `name:`, which is Claude Code's auto-memory
    convention (`[[verify-end-to-end]]` -> `feedback_verify_end_to_end.md`)."""
    files = set()
    texts = {p: read_or_empty(p) for p in md_files(root)}
    for p, txt in texts.items():
        files.add(p.stem.replace("-", "_"))
        name = front_name(txt)
        if name:
            files.add(name.replace("-", "_"))
    out = defaultdict(list)
    for p, txt in texts.items():
        for t in LINK.findall(txt):
            if t.replace("-", "_") not in files:
                out[t].append(p.name)
    return out


# A pointer at CSV ROWS: the file in backticks, the word `query` within the same
# clause, and a `Column=value|value` filter in backticks. A row has no stable ID,
# so this is the `[[wikilink]]` of a ruling, and it is checkable: the CSV must be
# registered, the column must exist, and every value must match at least one row,
# so that retiring a ruling exposes every passage that depended on it.
QUERY_PTR = re.compile(r"`([^`\s]+\.csv)`[^`\n]{0,40}`([A-Za-z_][\w -]*?)=([^`]+)`")


def _column_values(path, column):
    """Set of a column's values, or None where the column is absent."""
    import csv as _csv
    try:
        with open_safe(path, newline="") as fh:
            rd = _csv.DictReader(fh)
            if column not in (rd.fieldnames or []):
                return None
            return {(r.get(column) or "").strip() for r in rd}
    except (OSError, _csv.Error):
        return None


def csv_pointers(root):
    """[(file, csv, column, values, status)] for every query pointer in the corpus.
    Status is `ok`, `UNREGISTERED` (no such CSV in the manifest), `NO COLUMN`, or
    `DANGLING <values>` naming the filter values no row carries."""
    sources = {p.name: p for p, _, _ in csv_sources(root)}
    cache, out = {}, []
    for p in md_files(root):
        txt = read_or_empty(p)
        for m in QUERY_PTR.finditer(txt):
            if not re.search(r"\bquery\b", txt[max(0, m.start() - 60):m.end()], re.I):
                continue
            name, col, vals = m.group(1), m.group(2).strip(), m.group(3)
            if name not in sources:
                out.append((p.name, name, col, vals, "UNREGISTERED"))
                continue
            if (name, col) not in cache:
                cache[(name, col)] = _column_values(sources[name], col)
            have = cache[(name, col)]
            if have is None:
                out.append((p.name, name, col, vals, "NO COLUMN"))
                continue
            missing = [v.strip() for v in vals.split("|") if v.strip() and v.strip() not in have]
            out.append((p.name, name, col, vals, f"DANGLING {'|'.join(missing)}" if missing else "ok"))
    return out


# ------------------------------------------------------------- code citations
#
# A `path:line` written into a memory file is the one claim class every other
# mode is blind to: `audit` compares claims to claims, the write-time gate
# scores novelty, `memfind` ranks semantics, and not one of them opens the cited
# file. So a citation is correct for exactly one commit and then decays in
# silence. Measured on a production corpus 2026-08-17: of 54 citations,
# roughly one in eight had drifted, one pointed past EOF and one named a script
# deleted six weeks earlier.
CITE = re.compile(
    r"(?<![\w./-])"
    r"([A-Za-z0-9_][A-Za-z0-9_./-]*"
    r"\.(?:py|js|jsx|ts|tsx|html|css|json|toml|ya?ml|sh|sql|kt|swift|rb|go))"
    r":(\d+)"
)
# Vendored and generated trees. A citation resolving into `node_modules` is
# noise, and walking them is most of the runtime.
SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", "dist", "build",
             "site-packages", ".mypy_cache", ".pytest_cache", ".next", ".git"}


def project_dir_for(corpus_root):
    """The working directory a project corpus was created from, or None.

    `corpus_for_cwd` slugifies a path by replacing every non-alphanumeric
    character with a dash, so the slug cannot simply be split: directory names
    contain dashes, dots and underscores themselves, and all three arrive as the
    same character. Walk the filesystem instead and re-slugify each real entry,
    taking the longest match at every level. That is exact where string
    splitting only guesses.
    """
    try:
        rel = Path(corpus_root).resolve().relative_to((CLAUDE / "projects").resolve())
    except (ValueError, OSError):
        return None
    rest, cur = rel.parts[0].lstrip("-"), Path("/")
    while rest:
        best = None
        try:
            entries = os.listdir(cur)
        except OSError:
            return None
        for e in entries:
            s = re.sub(r"[^A-Za-z0-9]", "-", e)
            if rest != s and not rest.startswith(s + "-"):
                continue
            if (cur / e).is_dir() and (best is None or len(s) > len(best[1])):
                best = (e, s)
        if best is None:
            return None
        cur, rest = cur / best[0], rest[len(best[1]):].lstrip("-")
    return cur


# Excluded when walking `~/.claude` itself. Scoped to that root rather than
# added to SKIP_DIRS, because `cache` and `backups` are ordinary directory names
# a real repository is allowed to have.
# ⚠ `file-history` is the load-bearing one: it stores historical COPIES of edited
# files, so leaving it in resolves a citation against a stale snapshot of the
# file it names and reports a scope that has not been true for weeks.
CLAUDE_SKIP = frozenset({
    "file-history", "projects", "plugins", "backups", "cache", "paste-cache",
    "shell-snapshots", "session-env", "debug", "history", "statsig", "todos",
    "ide", "daemon", "tasks", "chrome", "jobs", "sessions", "state",
})
# A host home directory past this many subdirectories is reported by `code_index`,
# because the skip list above is a denylist and a new host-state tree otherwise shows
# up only as a slow `cites` run: `jobs/` once added 6 s to every run before it was listed.
LOUD_SUBTREE = 5000


def code_roots(corpus_root):
    """(tree, extra skips) for every tree a citation in this corpus could reach.

    The project's own working directory, plus `~/.claude` as a whole. Council
    and rules memory cite hooks, scripts and the harness; a project corpus cites
    its repos and sometimes a hook as well. Naming `~/.claude` whole rather than
    listing its code directories is deliberate: an explicit list silently misses
    whatever gets added next, and it already missed `harness/` once, which turned
    a live citation into a reported FILE NOT FOUND.
    """
    roots, seen = [], set()
    for r in roots_of(corpus_root):
        for tree, skips in _own_code_roots(r):
            if tree not in seen:
                seen.add(tree)
                roots.append((tree, skips))
    return roots


def _own_code_roots(corpus_root):
    if not trusted(corpus_root):
        return []                            # an untrusted folder's citations reach no host file
    declared = settings(corpus_root)["code_roots"]
    if declared is not None:                 # the manifest names them outright
        return [((Path(corpus_root) / d) if not d.is_absolute() else d, frozenset())
                for d in declared if ((Path(corpus_root) / d) if not d.is_absolute() else d).is_dir()]
    roots = []
    proj = project_dir_for(corpus_root)
    if proj:
        roots.append((proj, frozenset()))
    for home in (CLAUDE, HERMES):
        if home.is_dir():
            roots.append((home, CLAUDE_SKIP))
    return roots


def code_index(roots):
    """basename -> [absolute paths]. One walk for the whole run.

    Built up front rather than searched per citation: resolving fifty citations
    by fifty walks re-reads the tree fifty times.

    A host home is walked whole minus CLAUDE_SKIP, so an authored directory added
    later (`harness/`) is still searched. The cost of that denylist is that a new
    host-state tree is walked too, so any top-level directory of a host home past
    LOUD_SUBTREE subdirectories is named on stderr instead of costing time silently.
    """
    idx = defaultdict(list)
    for base, extra in roots:
        sizes = Counter()
        base_s = str(base)
        for dirpath, dirnames, filenames in os.walk(base):
            # sorted, so the first hit `resolve_citation` reads is the same every run
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in SKIP_DIRS and d not in extra
                                 and not d.startswith("."))
            if extra and dirpath != base_s:
                sizes[dirpath[len(base_s):].lstrip(os.sep).split(os.sep, 1)[0]] += 1
            for fn in filenames:
                idx[fn].append(os.path.join(dirpath, fn))
        for top, n in sorted(sizes.items()):
            if n > LOUD_SUBTREE:
                print(f"note: walked {n} directories under {Path(base) / top}; if it is host "
                      "state rather than code, it belongs in memscan.CLAUDE_SKIP", file=sys.stderr)
    return idx


def enclosing_scope(lines, n):
    """The `def`/`class` a 1-indexed line sits inside, or None at module level.

    The whole point of the mode: a matched string tells you a line, and only the
    enclosing scope tells you a location.
    """
    if not 0 < n <= len(lines):
        return None
    line = lines[n - 1]
    indent = len(line) - len(line.lstrip())
    name = lambda s: s.split("(")[0].split(":")[0].strip()
    # ⚠ A line at column 0 is at MODULE level and has no enclosing scope. Walking
    # back from it finds the last def in the file and reports a constant as
    # living inside an unrelated function, which is the exact wrong-citation
    # failure this mode exists to catch. Caught 2026-08-17 attributing
    # `CURRENT_CURATION_STANDARD` (globals.py:125) to `_read_version` (:41), and
    # it nearly got two CORRECT memory citations "fixed".
    if indent == 0:
        stripped = line.lstrip()
        return name(stripped) if stripped.startswith(
            ("def ", "async def ", "class ")) else None
    for i in range(n - 2, -1, -1):
        stripped = lines[i].lstrip()
        if not stripped.startswith(("def ", "async def ", "class ")):
            continue
        if len(lines[i]) - len(stripped) < indent:
            return name(stripped)
    return None


def citations(root):
    """{(path, line): {citing file, ...}} across the corpus."""
    out = defaultdict(set)
    for p in md_files(root):
        for m in CITE.finditer(read_or_empty(p)):
            out[(m.group(1), int(m.group(2)))].add(p.name)
    return out


def resolve_citation(path, line, idx):
    """(status, detail) for one citation. Status is ok / eof / missing; a missing citation's
    detail is how many indexed files share its name somewhere else."""
    path = posixpath.normpath(path.replace("\\", "/"))
    hits = idx.get(posixpath.basename(path), [])
    if "/" in path:
        # A citation that names a directory means that directory: a file of the same name
        # elsewhere is a different file, and resolving to it reported a stranger as live.
        # Its first component may abbreviate a repository (`ui/app/x.py` for
        # `choose-my-plane-ui/app/x.py`) once two or more components follow it.
        parts = path.split("/")
        norm = {h: h.replace(os.sep, "/") for h in hits}
        at = lambda t: [h for h in hits if norm[h] == t or norm[h].endswith("/" + t)]
        exact = at(path) or (at("/".join(parts[1:])) if len(parts) >= 3 else [])
        if not exact:
            return "missing", len(hits)
        hits = exact
    if not hits:
        return "missing", 0
    # splitlines, not split("\n"): the latter counts a trailing newline as a
    # 89th line in an 88-line file, so a citation one past the end reads as ok.
    # An ambiguous name resolves to the first candidate long enough to hold the line:
    # PAST EOF means no candidate reaches it, never that the first one sorted was short.
    first = None
    for h in hits:
        try:
            lines = read_safe(h).splitlines()     # never through a link, a FIFO or past the cap
        except OSError:
            continue
        if line <= len(lines):
            return "ok", (h, lines, len(hits))
        first = first or (h, len(lines))
    if first is None:
        return "missing", 0
    return "eof", (first[0], first[1], len(hits))


# ---------------------------------------------------------------- entry points

def _filter_pairs(pairs, store=lambda f: ""):
    """(kept, dropped): the pairs with boilerplate (a sentence in three or more
    files of ONE store) and pointer-only sentences removed. One filter shared by
    `audit`, `graduated` and `across`, so no two modes disagree about what a
    pair is. `store` names the store a file belongs to; `across` passes the
    union prefix so that a sentence propagated to three stores, one file each,
    is the finding and not furniture. Counted over `store::file` keys it was
    three files and vanished, while the mode's own closing line promised it
    would fire forever."""
    txt_files = defaultdict(set)
    for _, fa, fb, sa, sb in pairs:
        txt_files[(store(fa), " ".join(sa.lower().split()))].add(fa)
        txt_files[(store(fb), " ".join(sb.lower().split()))].add(fb)
    boiler = {t for (_, t), f in txt_files.items() if len(f) >= 3}

    kept, dropped = [], 0
    for jac, fa, fb, sa, sb in pairs:
        # Test BOTH sides: a near-duplicate pair carrying the boilerplate on the
        # B side only survives a one-sided test.
        if ({" ".join(sa.lower().split()), " ".join(sb.lower().split())} & boiler
                or pointer_only(sa) or pointer_only(sb)):
            dropped += 1
            continue
        kept.append((jac, fa, fb, sa, sb))
    return kept, dropped


def _audit_pairs(root, floor=PAIR_FLOOR):
    """(docs, raw pairs, kept pairs, dropped count) for one corpus."""
    docs = corpus(root)
    pairs = cross_file_pairs(docs, floor)
    kept, dropped = _filter_pairs(pairs)
    return docs, pairs, kept, dropped


def across_pairs(stores, floor=PAIR_FLOOR):
    """Claim pairs that NO single store holds both sides of.

    `stores` is {name: root}. Every corpus is loaded into one union and paired
    once, so the cost is one `audit` over the sum of the stores (0.5s over 490
    files here), and a pair some store's own `audit` already owns is dropped.
    Returns (union, raw pairs, dropped, cross). The gap this closes: every other
    check scores a corpus against itself, so a fact written into two stores is
    invisible to both stores' gates and rots in whichever one stops being
    maintained.

    A file is one identity however many manifests register it: a markdown file
    by its resolved path, a CSV pseudo-document by the CSV's resolved path plus
    its label. Keyed by `<store>::<name>` instead, a ledger registered under
    `sources` in two stores entered the union twice and paired every row against
    itself, thirty phantom pairs above the one real one on the first store that
    shared a ledger; and a rule file restating a row of the ledger it points into
    printed here too, though the rule store's own `audit` already reports it. A
    shared file's union key names every holder (`alpha+beta::RULES.csv#x`), so
    the pair lines say it is shared without a second mechanism. Edge worth
    knowing: `_filter_pairs` counts furniture per prefix, so a shared file sits
    in its own bucket rather than in each holder's.
    """
    holders, bodies, names = {}, {}, {}
    for k, root in sorted(stores.items()):
        rs = roots_of(root)
        md = {corpus_key(root, p, rs): str(p.resolve()) for p in md_files(root)}
        csvs = {p.name: str(p.resolve()) for p, _, _ in csv_sources(root)}
        for name, body in corpus(root).items():
            if name in md:
                ident = md[name]
            elif "#" in name and name.split("#", 1)[0] in csvs:
                fname, tag = name.split("#", 1)
                ident = f"{csvs[fname]}#{tag}"
            else:
                ident = f"{k}::{name}"
            holders.setdefault(ident, []).append(k)
            bodies.setdefault(ident, body)
            names.setdefault(ident, name)
    union, owners = {}, {}
    for ident, ks in holders.items():
        key = f"{'+'.join(ks)}::{names[ident]}"
        union[key] = bodies[ident]
        owners[key] = set(ks)
    pairs = cross_file_pairs(union, floor)
    kept, dropped = _filter_pairs(pairs, store=lambda f: f.split("::", 1)[0])
    cross = [p for p in kept if owners[p[1]].isdisjoint(owners[p[2]])]
    return union, pairs, dropped, cross


def _side_kind(key, stores):
    """`holding`, `ledger` or `durable` for one side of an `across` pair, read
    from the manifest of every store holding it; `mixed` where they disagree."""
    prefix, name = key.split("::", 1)
    base = Path(name).name
    kinds = set()
    for k in prefix.split("+"):
        s = settings(stores[k])
        kinds.add("holding" if base in s["holding_spaces"]
                  else "ledger" if base in s["ledger_surfaces"] else "durable")
    return kinds.pop() if len(kinds) == 1 else "mixed"


def graduated_across(cross, stores):
    """The `across` pairs with a holding-space side and a durable side, holding
    side first: an entry promoted into ANOTHER store and never retired.
    `graduated` scores a holding space against the durable files of its own
    store, so a promotion that crossed a store boundary is invisible to it, and
    this is the one place both sides are in hand. Found on the first store that
    kept its rules in a second corpus: five holding entries whose lessons had
    moved there weeks before, `graduated` reporting zero and right to."""
    out = []
    for jac, fa, fb, sa, sb in cross:
        ka, kb = _side_kind(fa, stores), _side_kind(fb, stores)
        if {ka, kb} == {"holding", "durable"}:
            out.append((jac, fa, fb, sa, sb) if ka == "holding" else (jac, fb, fa, sb, sa))
    return out


def cmd_across(floor=PAIR_FLOOR):
    stores = {k: v for k, v in all_corpora().items() if md_files(v) or csv_rows(v)}
    union, pairs, dropped, cross = across_pairs(stores, floor)
    grad = graduated_across(cross, stores)
    print(f"{len(stores)} corpora, {len(union)} files; {len(pairs)} raw claim pairs at "
          f"jaccard {floor}; {dropped} filtered as pointer/boilerplate; "
          f"{len(cross)} across corpora to read, {len(grad)} with a holding-space side "
          f"(promoted into another store, never retired; `graduated` scores one store "
          f"and cannot see these)\n")
    by = Counter((fa.split("::", 1)[0], fb.split("::", 1)[0]) for _, fa, fb, _, _ in cross)
    for (a, b), n in by.most_common():
        print(f"  {n:4d}  {a}  x  {b}")
    if by:
        print()
    tagged = {(fa, fb, sa) for _, fa, fb, sa, _ in grad}
    for jac, fa, fb, sa, sb in cross:
        tag = " [holding side: promoted, never retired]" if (fa, fb, sa) in tagged or (fb, fa, sb) in tagged else ""
        print(f"[{jac:.2f}]{tag} {fa}\n       {fb}\n   A: {relay(sa)}\n   B: {relay(sb)}\n")
    print("Pairs that currently AGREE, so fork risk rather than divergence: once two copies "
          "drift apart the jaccard falls and nothing here sees them. A store derived from "
          "another (a vault distilled from a host's council store) fires here forever once "
          "registered; read it as expected, never exempt it, and register it, because a "
          "directory no manifest names is watched by nothing and a lesson promoted into it "
          "reads exactly like one never promoted. Inside one host the skills, rules and "
          "agents sit in the council store with the memory, so their restatements are "
          "`audit`'s to report, not this mode's.")
    return len(cross)


def cmd_audit(root, floor=PAIR_FLOOR):
    docs, pairs, kept, dropped = _audit_pairs(root, floor)
    index_files = settings(root)["index_files"]
    idx_pairs = [p for p in kept if Path(p[1]).name in index_files or Path(p[2]).name in index_files]
    under = sum(1 for body in docs.values() for _ in unclaimed(body))
    print(f"{len(docs)} files; {len(pairs)} raw cross-file claim pairs at jaccard {floor}; "
          f"{dropped} filtered as pointer/boilerplate; {len(kept)} to read "
          f"({len(idx_pairs)} index-restates-its-target); {under} sentences under the "
          f"{MIN_TOKENS}-token floor formed no claim and are in nothing above\n")
    for jac, fa, fb, sa, sb in kept:
        print(f"[{jac:.2f}] {fa}\n       {fb}\n   A: {relay(sa)}\n   B: {relay(sb)}\n")
    return len(kept)


def cmd_graduated(root, floor=PAIR_FLOOR):
    """Holding-space claims that a durable file now holds: promoted, never retired.

    A holding space stages an entry until it is promoted or discarded, so an entry
    whose claim now sits in a durable file or a registered CSV row has graduated
    and was never retired: retire it, or rewrite it to what is still open. `audit`
    prints these same pairs among every other kind; this mode keeps only the pairs
    with exactly one holding side and no ledger side, because a holding-to-holding
    pair is a ledger restating a gate by design, a ledger entry quoting a gate is
    chronology doing its job (on the council store those were 20 of 20 hits), and
    a durable-to-durable pair is `audit`'s. Scores ONE store: an entry promoted
    into a second store is invisible here and is tagged by `across`, which is
    the only mode holding both sides.

    Never silent. The scope line prints on a clean run too: a zero from a store
    with no holding file and a zero from one holding file scored against eleven
    durable sources are different answers, and a mode that prints nothing when
    clean cannot tell its reader which one it gave. Informs, never blocks.
    """
    sets = settings(root)
    holding, ledgers = sets["holding_spaces"], sets["ledger_surfaces"]
    docs, _, kept, _ = _audit_pairs(root, floor)
    present = sorted(n for n in docs if Path(n).name in holding)
    durable = [n for n in docs if Path(n).name not in holding and Path(n).name not in ledgers]
    hits = []
    for jac, fa, fb, sa, sb in kept:
        if Path(fa).name in ledgers or Path(fb).name in ledgers:
            continue
        ha, hb = Path(fa).name in holding, Path(fb).name in holding
        if ha != hb:                       # holding side first, so a reader works one file at a time
            hits.append((jac, fa, fb, sa, sb) if ha else (jac, fb, fa, sb, sa))
    hits.sort(key=lambda p: (p[1], -p[0]))
    if not holding:
        scope = "no holding spaces are configured for this store"
    elif not present:
        scope = f"none of the configured holding spaces ({', '.join(sorted(holding))}) is present"
    else:
        scope = (f"{len(present)} holding file{'s' if len(present) > 1 else ''} "
                 f"({', '.join(present)}) against {len(durable)} durable sources, ledgers excluded")
    print(f"{len(hits)} holding-space claims also held by a durable file; {scope}\n")
    for jac, fh, fd, sh, sd in hits:
        print(f"[{jac:.2f}] {fh}\n       {fd}\n   holding: {relay(sh)}\n   durable: {relay(sd)}\n")
    return len(hits)


def corpora_lines(width=70):
    """One line per store: markdown files plus registered CSV rows, the same
    sources `audit` counts. The CLI and the MCP server both print this, because
    a listing that counted markdown alone read as a half-indexed store beside an
    audit that counted six."""
    out = []
    for k, v in sorted(all_corpora().items()):
        rows = len(csv_rows(v))
        joined = len(roots_of(v)) - 1
        out.append(f"  {k:{width}s} {len(md_files(v)):4d} files"
                   + (f" + {rows} csv rows" if rows else "")
                   + (f"  [{joined} joined store{'s' if joined > 1 else ''}]" if joined else ""))
    for p in registry_stale():
        out.append(f"  {str(p):{width}s} registered, but its {MANIFEST} is gone; "
                   f"`init` it again to keep it, `forget` it to drop the row")
    return out


def _stores_under(d):
    """Corpora strictly inside directory `d`: every listed store (host, auto-memory
    and registry) whose parents include `d`, plus the first hand-written manifest
    found below it. Sorted, host stores first, so the message names a real one."""
    d = Path(d)
    found = sorted({r for r in _raw_corpora().values() if r != d and d in r.parents})
    if found:
        return found
    hit = next((m.parent for name in (MANIFEST, OLD_MANIFEST)
                for m in d.rglob(name) if m.parent != d), None)
    return [hit] if hit else []


def cmd_init(target, index=True, parent=None, name=None, ledgers=None, confirm=None):
    """Make a directory a corpus every mode can find: write an empty manifest
    where none exists, register the path, build the index where an embedder
    answers, print the corpora line. Running it twice changes nothing. A
    directory inside an existing corpus is refused, and so is one containing a
    corpus, since nesting either way would score every file against two stores;
    a host store or a project memory is already a corpus by convention and
    takes no manifest.

    With `parent`, the directory JOINS that corpus instead of standing alone:
    `belongs_to: <name>` goes into its manifest (written even for a store by
    convention, where the manifest is otherwise optional), every mode and the
    gate then resolve it to the parent, and the index is built at the parent.
    Two folders in two places, one corpus. `name` sets the joining store's label
    in the same write, for the case the label would collide: a host's
    auto-memory is always a directory called `memory`, the name the host
    store's own member directory takes.

    With `ledgers` (a name, or "" for the store's own), `scaffold_ledgers` adds
    the three built-in ledgers, writing a manifest even on a store by convention,
    because asking for ledgers is asking for configuration."""
    d = Path(target).expanduser()
    if d.is_dir() and d.resolve() in (Path("/"), HOME.resolve()):
        print(f"{d} is too broad to be a store; init the folder that holds the memory", file=sys.stderr)
        return 1
    if not d.is_dir():
        print(f"{d} is not a directory", file=sys.stderr)
        return 1
    d = d.resolve()
    if ledgers:
        if ledger_name(ledgers) is None:
            print(f"{ledgers!r} cannot name a ledger: it goes into file names", file=sys.stderr)
            return 1
    elif ledgers is not None:           # the store's own name, made safe for a file name
        ledgers = re.sub(r"[^\w.-]+", "-", name or _label(d)).strip("-.") or "notes"
    inside = _phys_root_of(d)
    if inside is not None and inside != d:
        print(f"{d} is inside the corpus {inside} already; nothing to do", file=sys.stderr)
        return 1
    # The guard runs both ways. A folder containing a store is the likelier
    # accident, because the parent is what a person names when they mean "all
    # of my configuration": on the first store to try it, init on the parent of
    # two registered stores was accepted, took a registry row and began indexing
    # caches and session state before it was killed. A host home IS that parent
    # by design, with the project stores under it as non-members, so a store
    # by convention skips the check rather than refusing itself.
    within = [] if inside == d else _stores_under(d)
    if within:
        more = f" and {len(within) - 1} more" if len(within) > 1 else ""
        print(f"{d} contains the corpus {within[0]}{more}; a store cannot contain another. "
              f"`init` the directory you mean, not its parent", file=sys.stderr)
        return 1
    m = manifest_file(d)
    try:
        cmd_schema(quiet=True)              # the editor's copy tracks the installed tool
    except OSError:
        pass
    for f in validate_manifest(d):
        print(f"manifest: {f}", file=sys.stderr)
    if parent is not None:
        if d in set(paths.PATHS.host_stores.values()):
            print(f"{d} is a host store and cannot belong to another corpus", file=sys.stderr)
            return 1
        raw = _raw_corpora()
        target = raw.get(parent)
        if target is None:
            print(f"no corpus is named {parent!r}; `corpora` lists the names", file=sys.stderr)
            return 1
        if target == d:
            print(f"{d} cannot belong to itself", file=sys.stderr)
            return 1
        if d in target.parents or _walked_by(target, d):
            print(f"{d} and {target} nest on disk inside {target.name}'s walk; a nested "
                  f"directory there is a `members` entry, not a joined store", file=sys.stderr)
            return 1
        label = name or _label(d)
        if "/" in label or label in (".", ".."):
            print(f"{label!r} cannot be a label: it is prefixed to paths", file=sys.stderr)
            return 1
        if label == _label(target) or (target / label).is_dir():
            print(f"{d}'s label {label!r} collides with a directory or the name of {target}; "
                  f"label the joining store: locket init {d} --parent {parent} --name <label>",
                  file=sys.stderr)
            return 1
        man = dict(_manifest_data(d))
        if name:
            man["name"] = name
        man["belongs_to"] = parent
        write_store_file(m, json.dumps(man, indent=2) + "\n")
        settings.cache_clear()
        if not grant_trust(d, confirm):
            return 1
        settings.cache_clear()
        if parent_of(d, _raw_corpora()) != target:
            print(f"{d} did not resolve to {target} after the write; check for a cycle "
                  f"through {parent!r}", file=sys.stderr)
            return 1
        print(f"manifest: {m} (belongs_to {parent!r})")
        if ledgers is not None and scaffold_ledgers(d, ledgers):
            return 1
        top = top_of(d)
        stale = d / ".memfind"
        if stale.is_dir():
            print(f"index: {stale} is no longer read; delete it")
        n, rows = len(md_files(top)), len(csv_rows(top))
        print(f"corpus: {top}  {n} files" + (f" + {rows} csv rows" if rows else "")
              + f"  [{len(roots_of(top)) - 1} joined]")
        if not index:
            return 0
        try:
            import memfind
            memfind.build_index(top, quiet=True)
            print(f"index: built for {top}")
        except ImportError:
            print(f"index: memfind.py not found beside memscan.py; run  locket index {top}")
        except RuntimeError as e:
            print(f"index: not built ({e}); run  locket index {top}  once an embedder answers")
        except OSError as e:
            print(f"index: not built, the cache could not be written ({e})")
        return 0
    # A host store or a project memory is listed on sight and takes no registry
    # row; a manifest there is optional config (sources, exclusions), kept if
    # present and never written for it.
    # A standalone store's manifest carries its `name`, the folder's unless
    # `--name` says otherwise, so the label is visible and editable in the one
    # file that owns it. Only the name: a default written into a manifest
    # replaces the built-in and freezes it there, so a store would stop
    # receiving a changed default without anyone having chosen that.
    if name is None and not (m.is_file() or inside == d):
        other = _raw_corpora().get(d.name)
        if other is not None and other != d:
            print(f"the name {d.name!r} is taken by {other}; name this store:  "
                  f"locket init {d} --name <label>", file=sys.stderr)
            return 1
    if name is not None:
        if "/" in name or name in (".", "..") or not name.strip():
            print(f"{name!r} cannot be a name: it is prefixed to paths", file=sys.stderr)
            return 1
        other = _raw_corpora().get(name)
        if other is not None and other != d:
            print(f"the name {name!r} is taken by {other}; `corpora` lists the names", file=sys.stderr)
            return 1
    by_convention = inside == d and not m.is_file()
    if m.is_file():
        by_convention = d in set(paths.PATHS.host_stores.values()) or (
            d.name == "memory" and d.parent.parent == CLAUDE / "projects")
        man = _manifest_data(d)
        if name is not None and man.get("name") != name:
            write_store_file(m, json.dumps({**man, "name": name}, indent=2) + "\n")
            print(f"manifest: {m} (kept, name {name!r})")
        else:
            print(f"manifest: {m} (kept)")
    elif by_convention and name is None:
        print(f"manifest: none needed; {d} is a corpus by convention on this host")
    else:
        write_store_file(m, json.dumps({"name": name or d.name}, indent=2) + "\n")
        print(f"manifest: {m} (written, name {name or d.name!r})")
    settings.cache_clear()
    if by_convention:
        print("registry: not needed; listed by convention")
    else:
        if not grant_trust(d, confirm):
            return 1
        settings.cache_clear()
    if ledgers is not None and scaffold_ledgers(d, ledgers):
        return 1
    n, rows = len(md_files(d)), len(csv_rows(d))
    print(f"corpus: {d}  {n} files" + (f" + {rows} csv rows" if rows else ""))
    if not index:
        return 0
    if n == 0 and rows == 0:
        print("index: nothing to index yet; the store is registered and is read once it has markdown")
        return 0
    try:                                    # memfind.py sits beside this file
        import memfind
        memfind.build_index(d, quiet=True)
        print(f"index: built for {d}")
    except ImportError:
        print(f"index: memfind.py not found beside memscan.py; run  locket index {d}")
    except RuntimeError as e:
        print(f"index: not built ({e}); run  locket index {d}  once an embedder answers")
    except OSError as e:
        print(f"index: not built, the cache could not be written ({e})")
    return 0


def session_ledger(root):
    """The store's session ledger (`HISTORY-*-Sessions.csv`) at its root or the root of a
    store joined to it, or None. The top store's own ledger wins."""
    for r in roots_of(top_of(Path(root))):
        try:
            hit = next(iter(sorted(Path(r).glob("HISTORY-*-Sessions.csv"))), None)
        except OSError:
            continue
        if hit is not None:
            return hit
    return None


LEDGER_NAME = re.compile(r"\w[\w.-]*")


def ledger_name(name):
    """The name a ledger file carries, or None when it cannot go into a file name."""
    return name if isinstance(name, str) and LEDGER_NAME.fullmatch(name) else None


LEDGER_FILES = (("HISTORY-{}-Sessions.csv", "HISTORY-*-Sessions.csv", None),
                ("RULINGS-{}.csv", "RULINGS*.csv", "Ruling"), ("CLAIMS-{}.csv", "CLAIMS*.csv", "Claim"))


def scaffold_ledgers(d, name):
    """Create the three built-in ledgers at a store's root, header row only, and
    register RULINGS and CLAIMS as sources so their rows rank beside the markdown.
    The session ledger stays unregistered: its rows are chronology, never claims.
    An existing file is never touched and a source is never listed twice, so a
    second run changes nothing. Headers come from LEDGER_SCHEMAS, the one home of
    the columns the write-time check enforces. Returns 0, or 1 on a bad name."""
    if ledger_name(name) is None:
        print(f"{name!r} cannot name a ledger: it goes into file names", file=sys.stderr)
        return 1
    m = manifest_file(d)
    man = dict(_read_manifest(d))
    # ⚠ `_read_manifest` reads an unparseable file as {}, and writing that back
    # would erase every key the user had; refuse instead
    if not man and m.is_file() and read_or_empty(m).strip():
        print(f"{m} does not parse; fix it (`locket schema {d}`) before adding ledgers", file=sys.stderr)
        return 1
    sources = man.get("sources") or []
    if not isinstance(sources, list):
        print(f"{m}: `sources` must be a list; fix it before adding ledgers", file=sys.stderr)
        return 1
    sources = list(sources)

    def where(e):                   # as `_own_csv_sources` reads an entry
        p = Path(e["path"]).expanduser()
        return (p if p.is_absolute() else d / p).resolve()
    listed = {where(e) for e in sources if isinstance(e, dict) and isinstance(e.get("path"), str)}
    added = []
    for pattern, builtin, text in LEDGER_FILES:
        # a ledger already here under any name is the store's; a second would split its rows
        have = sorted(d.glob(builtin))
        f = have[0] if have else d / pattern.format(name)
        schema = ledger_schema_for(f)[0] or LEDGER_SCHEMAS[builtin]   # a store's `extends` wins
        if f.exists():
            print(f"ledger: {f.name} (kept)")
        else:
            write_store_file(f, ",".join(map(str, schema["columns"])) + "\n")
            print(f"ledger: {f.name} (written, header only)")
        if text and f.resolve() not in listed:
            sources.append({"path": f.name, "text": text, "label": "Area"})
            added.append(f.name)
    if added:
        write_store_file(m, json.dumps({**man, "sources": sources}, indent=2) + "\n")
        settings.cache_clear()
        print(f"manifest: {', '.join(added)} registered as sources")
    return 0


def cmd_pointers(root):
    hooks = index_hooks(root)
    missing = [(i, t, h) for i, t, _, h, path in hooks if not path.exists()]
    print(f"{len(hooks)} index lines with a target; {len(missing)} point at a missing file\n")
    for i, t, h in missing:
        print(f"  MISSING TARGET  {i} -> {t}  :: {h[:90]}")
    if missing:
        print("  (an index line naming a file that was never written, or was removed: write the "
              "file or cut the line. A [[wikilink]] with no file is `links`' to report.)")
    for i, t, title, h, _ in hooks:
        print(f"\n### {i} -> {t}\nTITLE: {title}\nHOOK: {h}")
    return len(hooks)


def cmd_dupes(root, floor=PAIR_FLOOR, files=None):
    """Repeated claims INSIDE one file. The uniqueness invariant, measured.

    Defaults to the holding spaces, where uniqueness is the rule (Peer ruling
    2026-08-15); pass `files` to check any other. A ledger is exempt on purpose,
    since repetition is its shape.
    """
    docs = corpus(root)
    want = files or sorted(settings(root)["holding_spaces"] & set(docs))
    scope = {k: v for k, v in docs.items() if k in want}
    if not scope:
        print(f"no holding space in {root}; name a file to check one directly")
        return 0
    # Same ENTRY_TOKENS floor the write-time check uses. A report that flags what
    # the hook lets through is the drift this instrument exists to prevent.
    pairs = [p for p in cross_file_pairs(scope, floor, same_file=True)
             if not pointer_only(p[3]) and len(tokens(p[3])) >= ENTRY_TOKENS]
    per = Counter(p[1] for p in pairs)
    print(f"{len(scope)} file(s) checked for repeated entries: " +
          ", ".join(f"{k} ({per.get(k, 0)})" for k in sorted(scope)) + "\n")
    for jac, fa, _fb, sa, sb in pairs:
        print(f"[{jac:.2f}] {fa}\n   A: {relay(sa)}\n   B: {relay(sb)}\n")
    return len(pairs)


_DATE = r"20\d\d[-/]\d\d(?:[-/]\d\d)?"
# A dated HISTORY clause, in the shapes loaded-whole files actually carry: `Update 2026-08-08`,
# `⚠ UPDATE (2026-08-08)`, `2026-08-08: **Resolved:**`, and the italic sub-header that opens a
# stacked block, `*Refinement, 2026-05-19:*`. The `update <date>` form alone matched none of
# SELF.md's ten stacks. A date that scopes a live behaviour (`since 2026-09-16`, `(Peer ruling
# 2026-08-15)`) is not history and must not match; the negatives in tests/test_memscan.py hold that line.
HISTORY_CLAUSE = re.compile(
    r"(?:^\s*[*_]{1,2}(?:refinement|sharpening|second sharpening|third sharpening|surface note|"
    r"instances since)\b[^\n]{0,60}?" + _DATE +            # the italic stack header, line-start only
    r"|\b(?:update|correction)\s*[\(:]?\s*" + _DATE +       # `Update 2026-08-08`, `⚠ UPDATE (2026-08-08)`
    r"|" + _DATE + r"\s*[:)]?\s*\**\s*(?:resolved|update|correction|superseded)\b)",  # date first
    re.I | re.M)


def bare_date(added):
    """The first non-heading line of `added` carrying a date, or None.

    Any date written onto a loaded-whole surface is ASKED about and never blocks: an
    attribution (`Peer ruling 2026-09-11`), a `since <date>` or an `added on <date>` is
    bookkeeping §1 cuts, while a version or a cutover is scoped and stays, and a regex
    cannot tell the two apart, so under §2d this is a proxy and warns only. A heading
    carries its date by convention (a session log, a dated section) and is skipped.
    """
    for line in added.split("\n"):
        if line.lstrip().startswith("#"):
            continue
        if re.search(_DATE, line):
            return line.strip()
    return None


ENTRY_BYTES = 10_240       # §5 flag. 3 KB is the one-fact-one-file design target, not this
                           # number: it sat BELOW the corpus median (211 of 317 files, median
                           # 4,004 bytes) and so ranked nothing. Re-derive against the median
                           # before moving it; 10 KB was ~p90.
UPDATE_CAP = 3             # §5: three or more dated history clauses
PARA_WORDS = 200           # a paragraph past this is a session narrative, not a fact


def _prose_paras(txt):
    """Word counts of the PROSE blocks only.

    A bullet list, a table or a heading run carries no blank lines, so measuring
    every block reads a whole index file as one 6,000-word paragraph. The defect
    this counts is a paragraph nobody can read.

    A NUMBERED list counts as list-ish too. Without that, a 7-item checklist read
    as one 525-word paragraph and put its file at the top of the [P] table, which
    is a miscount rather than a finding.

    A line that follows a bullet or numbered item with no blank line between is
    that item's continuation and is credited to it, indented or not, which is how
    markdown renders it. Counting marker lines alone read a hard-wrapped 62-line
    index carrying 16 markers as a 784-word paragraph in a file whose longest
    real paragraph was 298: the same miscount with item length as the variable.
    Only a list marker opens a run, so a heading, a table row or a quote line
    followed by unbroken prose leaves that prose counted.
    """
    numbered = re.compile(r"^\d+[.)]\s")

    def opens(l):
        s = l.lstrip()
        return s[:2].rstrip() in ("-", "*") or bool(numbered.match(s))

    for block in re.split(r"\n\s*\n", txt):
        lines = [l for l in block.splitlines() if l.strip()]
        listish, in_item = 0, False
        for l in lines:
            if opens(l):
                listish, in_item = listish + 1, True
            elif l.lstrip()[:2].rstrip() in ("|", "#", ">"):
                listish, in_item = listish + 1, False
            elif in_item:
                listish += 1
        if lines and listish <= len(lines) / 2:
            yield len(block.split())


def cmd_budget(root):
    """Files breaching an §5 budget: bytes, dated history clauses, longest prose paragraph.

    Every history clause is listed by line, whatever the file's count, because the list
    is the reconsolidation worklist: a rewrite happens per passage, not per file.
    """
    rows, ledgers, history = [], [], {}
    ledger_names = settings(root)["ledger_surfaces"]
    rs = roots_of(root)
    for p in md_files(root):
        if p.name in ledger_names:           # chronology: length is the design, not a breach
            ledgers.append(f"{p.stat().st_size // 1024}KB {corpus_key(root, p, rs)}")
            continue
        txt = read_or_empty(p)
        name = corpus_key(root, p, rs)
        hits = [(i + 1, l.strip()[:110]) for i, l in enumerate(txt.splitlines())
                if HISTORY_CLAUSE.search(l)]
        if hits:
            history[name] = hits
        rows.append((p.stat().st_size, len(hits), max(list(_prose_paras(txt)) or [0]), name))
    if ledgers:
        print("ledgers, exempt (append-only chronology, folded in place, never reconsolidated): "
              + ", ".join(ledgers) + "\n")
    flagged = [r for r in sorted(rows, reverse=True)
               if r[0] > ENTRY_BYTES or r[1] >= UPDATE_CAP or r[2] > PARA_WORDS]
    b = sum(1 for r in flagged if r[0] > ENTRY_BYTES)
    u = sum(1 for r in flagged if r[1] >= UPDATE_CAP)
    w = sum(1 for r in flagged if r[2] > PARA_WORDS)
    print(f"{len(rows)} files; {len(flagged)} over an §5 budget — "
          f"{b} past {ENTRY_BYTES // 1024}KB [B], {u} with {UPDATE_CAP}+ update clauses [U], "
          f"{w} with a {PARA_WORDS}+ word paragraph [P]."
          + (f" Largest {min(25, len(flagged))}:\n" if flagged else ""))
    def mark_of(size, upd, para):
        return "".join(c for c, hit in
                       (("B", size > ENTRY_BYTES), ("U", upd >= UPDATE_CAP), ("P", para > PARA_WORDS))
                       if hit)

    if flagged:
        print(f"  {'bytes':>7} {'upd':>4} {'para':>5}  file")
    for size, upd, para, name in flagged[:25]:
        print(f"  {size:7d} {upd:4d} {para:5d}  {name}  [{mark_of(size, upd, para)}]")

    # The table sorts by bytes, so a small file holding one enormous paragraph
    # never reaches it: a 677-word paragraph in a 10KB file stayed invisible while
    # a 566-word one in a 28KB file showed. Sort [P] separately as well.
    shown = {r[3] for r in flagged[:25]}
    by_para = [r for r in sorted(flagged, key=lambda r: -r[2])
               if r[2] > PARA_WORDS and r[3] not in shown][:10]
    if by_para:
        print("\n  Longest paragraphs not in the table above:")
        for size, upd, para, name in by_para:
            print(f"  {size:7d} {upd:4d} {para:5d}  {name}  [{mark_of(size, upd, para)}]")
    if history:
        print("\n  Dated history clauses, each a passage to rewrite to what is now true "
              "(prime-memory-discipline §2), never to append to:")
        for name, hits in sorted(history.items(), key=lambda kv: -len(kv[1])):
            print(f"  {name} ({len(hits)})")
            for ln, text in hits[:8]:
                print(f"    {ln:5d}  {text}")
            if len(hits) > 8:
                print(f"          … {len(hits) - 8} more")
    return len(flagged)


def cmd_links(root):
    """Dangling links are three different things and only one is a defect.

    RESOLVABLE: written in an older naming convention, pointing at a file that
    does exist under a type prefix. Real rot; repair these.
    unresolved: a forward reference to a memory not written yet, which
    prime-memory-discipline allows, or a pointer to a skill or rule file that was never
    a memory. Neither is a defect, so neither is reported as one.
    """
    d = dangling(root)
    stems = [p.stem for p in md_files(root)]      # same walker as `dangling`, or
                                                  # the classifier and the thing it
                                                  # classifies disagree
    resolvable, unresolved = {}, {}
    for t, files in d.items():
        key = t.replace("-", "_")
        hits = [f for f in stems if f.endswith("_" + key)]
        (resolvable if len(hits) == 1 else unresolved)[t] = (hits, files)
    total = sum(len(v) for v in d.values())
    print(f"{len(d)} dangling wikilink targets, {total} occurrences: "
          f"{len(resolvable)} RESOLVABLE, {len(unresolved)} unresolved\n")
    for t, (hits, files) in sorted(resolvable.items()):
        print(f"  RESOLVABLE  [[{t}]] -> [[{hits[0]}]]   ({len(set(files))} files)")
    for t, (hits, files) in sorted(unresolved.items()):
        why = f"ambiguous {hits}" if hits else "forward reference or non-memory target"
        print(f"  unresolved  [[{t}]]  {why}   <- {', '.join(sorted(set(files)))}")
    ptrs = csv_pointers(root)
    if ptrs:
        bad = [x for x in ptrs if x[4] != "ok"]
        print(f"\n{len(ptrs)} CSV query pointers (`file.csv` ... query `Column=value`), "
              f"{len(ptrs) - len(bad)} resolve, {len(bad)} do not")
        for f, name, col, vals, status in sorted(bad, key=lambda x: (x[4], x[0])):
            print(f"  {status:14s} `{name}` query `{col}={vals}`   <- {f}")
    return len(resolvable) + sum(1 for x in ptrs if x[4] != "ok")


def cmd_cites(root, filters=None):
    """Resolve every `path:line` citation against the code it points at.

    Two findings are decidable here and are printed as defects: a cited file
    that no longer exists, and a line past the end of one that does. Everything
    else NEEDS A READER, the same way `pointers` does. This mode can say which
    function a line sits in; only a reader knows whether that is the function
    the sentence claimed. So the resolved list prints the enclosing scope beside
    each citation and stops there.

    The count itself is the headline. Every citation here is a pointer that will
    break without telling anyone, so the number is debt to convert to symbols,
    not a score to drive to zero by editing code.

    Two things read as defects and are not. A citation quoted as EVIDENCE that
    citations rot is a deliberately dead pointer, and this mode cannot tell it
    from a live one. A bare basename (`aircraft.py:80`, four files deep in three
    repos) is reported with the match count beside it: the scope shown belongs
    to whichever file sorted first, which is a guess, and the citation was
    already ambiguous before this mode read it.
    """
    cites = citations(root)
    if not cites:
        print("no path:line citations. The corpus cites by symbol.")
        return 0
    roots = code_roots(root)
    if not roots:
        print("no code tree resolved for this corpus; nothing to check against.")
        return 0
    print("searching: " + ", ".join(str(r) for r, _ in roots))
    idx = code_index(roots)

    defects, resolved = [], []
    for (path, line), sources in sorted(cites.items()):
        if filters and not any(f in path for f in filters):
            continue
        status, detail = resolve_citation(path, line, idx)
        who = ", ".join(sorted(sources))
        if status == "missing":
            elsewhere = (f" ({detail} file{'s share' if detail != 1 else ' shares'} the name, none at this path)"
                         if detail else "")
            defects.append(f"!! {path}:{line}  FILE NOT FOUND{elsewhere}   <- {who}")
        elif status == "eof":
            _, length, _ = detail
            defects.append(f"!! {path}:{line}  PAST EOF (file is {length} lines)   <- {who}")
        else:
            _, lines, n_hits = detail
            scope = enclosing_scope(lines, line) if path.endswith(".py") else None
            where = scope or lines[line - 1].strip()[:58] or "(blank line)"
            amb = f"  [{n_hits} files match this name]" if n_hits > 1 else ""
            resolved.append(f"   {path}:{line}{amb}\n       {where}\n       <- {who}")

    citing = {f for s in cites.values() for f in s}
    total = len(defects) + len(resolved)
    s = lambda n, w: f"{n} {w}" + ("" if n == 1 else "s")
    print(f"\n{s(total, 'distinct path:line citation')} across "
          f"{s(len(citing), 'memory file')}. {len(defects)} resolve to nothing.")
    print("Cite the SYMBOL instead: no other mode opens the cited file, so these "
          "decay in silence.\n")
    for d in defects:
        print(d)
    if defects and resolved:
        print()
    for r in resolved:
        print(r)
    return len(defects)


def cmd_against(root, text, top=3):
    docs = corpus(root)
    ranked = score_against(text, docs)[:top]
    for combined, best, name, pair in ranked:
        print(f"[{best:.2f}] {name}")
        if pair and best >= 0.4:
            print(f"      existing: {pair[1][:180]}")
    return ranked


# Fire only on a real claim-level restatement, which is the CLAIM score and never
# the combined one. The bulk-overlap term alone reaches 0.35, so gating on the
# combined score fires on novel text at claim scores of 0.12 to 0.25 wherever it
# shares a vocabulary with the corpus.
FIRE_FLOOR = 0.45


def added_text(tool_input, text, prior):
    """The NEW material in a write, or None where the write reconsolidates.

    Three shapes add material: a new or empty file, a whole-file Write that leaves
    the old body intact somewhere inside it, and an Edit that inserts more than
    three times what it replaces and over 400 characters more. The second is how
    an append happens at session close, so a whole-file Write is not exempt on its
    own. Rewriting in place returns None, because nudging on it would fire hardest
    exactly when the discipline is being followed.

    ⚠ Only the ADDED material is returned, never the whole text. The prior body is
    already in the corpus, so scoring it self-matches at 1.00 and names the file
    being written on every append.
    """
    old = str(tool_input.get("old_string") or "")
    if old:                                   # Edit: an inserted block, not a swap
        if not (len(text) > 3 * len(old) and len(text) - len(old) > 400):
            return None
        # Subtract the anchor LINE BY LINE, not as one block. An edit that inserts
        # material BETWEEN two anchor lines leaves them separated in `new_string`,
        # so a contiguous replace is a no-op and the anchor reaches the scorer as
        # if it were new, matching at 1.00 the very line being edited around.
        anchor = {l.strip() for l in old.splitlines() if l.strip()}
        return "\n".join(l for l in text.splitlines() if l.strip() not in anchor)
    body = prior.strip()
    if not body:                              # new file, or an empty one filled
        return text
    if body not in text:                      # passages rewritten: reconsolidation
        return None
    return text.replace(body, " ", 1)


# A holding space has its own furniture: "Promotion target if this holds across
# instances:" is a stem several entries share, not an entry any of them duplicates.
# The discriminator is SUBSTANCE, not repetition: stems carry 5 to 6 content
# tokens and genuine duplicate entries carry 10 to 14, while both match the same
# number of existing claims at or above FIRE_FLOOR once MIN_SHARED is applied.
# ENTRY_TOKENS sits with the other floors at the top, because `dupes` applies it too.


ISO_DATE = re.compile(r"20\d\d-\d\d-\d\d")

# One nudge, two forces, so the opening clause has to state which happened. Kept
# as constants because `main` rewrites a warning into the blocking form when the
# 2e gate refuses the same write, and two string literals drifting apart would
# leave the message asserting a write landed when it did not.
LEAD_BLOCKING = "A memory is about to be written."
LEAD_WARNING = "This memory write landed."


def dated_series_pair(a, b):
    """True where two lines are successive readings rather than one entry twice.

    A measurement series inside a holding space is a QUERIED surface living in a
    loaded-whole file: chronology is its whole value, so §2's uniqueness invariant
    reads its repeated scaffolding as a duplicate and blocks the next reading. The
    discriminator is the date the entry carries about ITSELF, so two readings of
    one sentence carrying different dates and different figures both land.

    ⚠ This carve-out is narrower than it looks and is NOT the general answer to a
    repeated null. Where the content is genuinely identical, collapse the entries
    to one line carrying a count and a last-seen date and update it in place; an
    in-place Edit never reaches the scorer at all. Run-length encoding is the fix
    for a repeated NOTHING, and this is the fix for repeated SOMETHINGS (Peer
    ruling 2026-08-16).
    """
    da, db = set(ISO_DATE.findall(a)), set(ISO_DATE.findall(b))
    return bool(da) and bool(db) and not (da & db)


def _entry_lines(txt, floor):
    """(line, [(claim, tokens)]) for every line carrying claims at `floor`.

    ⚠ An entry is a LINE; a claim is a fragment of one. `claims` splits on the
    colon, which is exactly where a dated entry keeps its date, so the substance
    half of `*Measurement update, 2026-08-16*: the ledger reached 341539 bytes`
    reaches the comparison carrying no date at all. Grouping by line is what puts
    the date back in reach of the sentence it qualifies.
    """
    for line in strip(txt).splitlines():
        cs = [(s, ts) for s, ts in claims(line) if len(ts) >= floor]
        if cs:
            yield line, cs


def holding_space_hit(added, prior):
    """The specific entry a holding space already holds, or None.

    Judged per incoming line, and only for a line carrying enough material to BE
    an entry. A short shared stem says nothing about whether the entry is new.
    """
    body = list(_entry_lines(prior, MIN_TOKENS))
    best = None
    for aline, acs in _entry_lines(added, ENTRY_TOKENS):
        for bline, bcs in body:
            if dated_series_pair(aline, bline):
                continue
            for isent, its in acs:
                for s, ts in bcs:
                    jac = similar(ts, its)
                    if jac >= FIRE_FLOOR and (best is None or jac > best[0]):
                        best = (jac, isent, s)
    return best


def hook_decision(name, tool_input, text, prior, docs, archived=None,
                  autonomous=False, sets=None, root=None):
    """The write-time nudge as one function: `(message, blocking)`.

    `docs` and `archived` are CALLABLES, invoked only after the cheap gates,
    because reading the corpus and the archive are the only expensive things the
    hook does and it runs on every memory write. Composed as one function
    returning a decision rather than an exit code so a test can drive the
    whole branch, including the append case where `added_text` and
    `score_against` are each correct and the pair still fires.

    Blocking is reserved for an autonomous run, and this is the reverse of the
    obvious reading. A supervised session has an adjudicator in the room who can
    read a nudge and act on it, which is the entire premise of the
    cycle/conversation split; §2d already calls this a nudge that "names
    candidate owners" and "does not adjudicate", and an exit-2 nudge is neither.
    The measured cost of blocking it was four legitimate operations refused in
    one session, each of them the discipline being followed, with the only way
    through being the Bash path §2d forbids. Cycle behaviour is unchanged: a
    cycle cannot act on a warning either, so the guard it gets stays hard.
    """
    sets = sets or settings(None)
    if name in sets["index_files"]:           # an index line is meant to point
        return None, False
    if name in sets["ledger_surfaces"]:       # chronology is the whole value
        return None, False
    added = added_text(tool_input, text, prior)
    if added is None or len(added) < 200:
        return None, False
    lead = LEAD_BLOCKING if autonomous else LEAD_WARNING
    if name in sets["holding_spaces"]:
        # Judged on the UNSUBTRACTED text: a holding space is scored against its
        # own prior body, so archived provenance says nothing about whether the
        # entry is a repeat, and `holding_space_hit` needs whole lines anyway.
        hit = holding_space_hit(added, prior)
        if hit is None:
            return None, False
        return (f"{lead} {name} is a holding space and its entries are unique by rule: "
                f"it already holds this one (closest line {hit[0]:.2f}).\n"
                f"      existing: {relay(hit[2], 160)}\n"
                f"Promote it, discard it, or fold it into the entry already there "
                f"(prime-memory-discipline.md §2)."), autonomous
    # A dated history clause stacked on a durable surface is §2's forbidden shape,
    # decidable from the added text alone (a regex, with §1's scoped dates as its
    # negatives in tests/test_memscan.py), so it fires here before any corpus is read and
    # blocks a cycle like the novelty guard does. Holding spaces and ledgers never
    # reach this line: a gate transition there appends one dated clause by rule.
    stacked = HISTORY_CLAUSE.search(added)
    dated = None if stacked else bare_date(added)
    if stacked:
        history = (f"{lead} It stacks a dated history clause on a surface that loads whole "
                   f"({stacked.group(0).strip()[:60]!r}). Rewrite the passage to what is now true "
                   f"and cut the journey; a date stays only where something is scoped by it "
                   f"(prime-memory-discipline.md §2).")
    elif dated:
        # Warn-only in both modes, so the lead never claims a block it does not make.
        history = (f"{LEAD_WARNING} It writes a date onto a surface that loads whole "
                   f"({dated[:90]!r}). A date stays only where it scopes something, a version "
                   f"or a cutover; an attribution, a 'since' or an 'added on' is bookkeeping and "
                   f"is cut (prime-memory-discipline.md §1).")
    else:
        history = None
    ledger = session_ledger(root) if history and root is not None else None
    if ledger:
        history += f" If the date records what a session did, it is a row in {ledger.name}."
    blocks = bool(stacked and autonomous)
    # Provenance before novelty: archived text is a move, not a fork. Subtraction
    # only removes, so a pure relocation exits here without the corpus being read.
    # ⚠ The floor now measures stripped text, which is shorter than raw, so this
    # gate is marginally tighter than the one above. Fewer nudges, never more.
    added = subtract_archived(added, archived() if archived else [])
    if len(added) < 200:
        return history, blocks
    corpus = docs()
    # The replaced span stops existing the moment the edit lands, so it must leave
    # the corpus before scoring or an in-place rewrite matches the sentences it
    # retires. `added_text` subtracts the anchor line by line, and a markdown
    # paragraph is one physical line, so an extended paragraph is never
    # byte-identical and survives that subtraction.
    # ⚠ `corpus()` stores STRIPPED bodies. Comparing a raw `old_string` against
    # them matches nothing and no-ops silently while looking like a working patch.
    old = strip(str(tool_input.get("old_string") or "")).strip()
    if old and name in corpus:
        corpus = dict(corpus)
        corpus[name] = corpus[name].replace(old, " ", 1)
    ranked = [r for r in score_against(added, corpus)[:3] if r[1] >= FIRE_FLOOR]
    if not ranked:
        return history, blocks
    lines = ([history, ""] if history else []) + [f"{lead} These files may already own the fact it carries:"]
    for _combined, best, fname, pair in ranked:
        lines.append(f"  - {fname}   (closest claim {best:.2f})")
        if pair:
            lines.append(f"      existing: {relay(pair[1], 160)}")
    lines.append("Rewrite the passage that already owns it rather than leaving a sibling "
                 "(prime-memory-discipline.md §2c). If this genuinely is a new fact, nothing "
                 "to do. Provenance is already subtracted: archived text does not reach "
                 "this list, so a match here is a fork rather than a move.")
    return "\n".join(lines), autonomous


ENV_ASSIGN = re.compile(r"[A-Za-z_]\w*=\S*")
# Shell separators that start a fresh command. A segment boundary is the only
# place a command name can sit, which is the whole discriminator below.
SHELL_SPLIT = re.compile(r"\|\||&&|[;|&\n()`]")


HEREDOC = re.compile(r"""<<-?\s*['"]?(\w+)['"]?""")


_INTERPRETER = re.compile(r"\b(?:python[0-9.]*|node|deno|bun|bash|sh|zsh|dash|fish|perl|ruby|php|osascript|lua)\b")


def command_text(command):
    """`command` as the guards that refuse by name read it: a heredoc body fed to a
    program that only stores or prints it (`git commit -F -`, `cat > notes`, `gh ...
    --notes-file -`) is data and dropped, and one fed to an interpreter is code and
    kept. Release notes that name the trust verb are not a trust command; a Python
    heredoc that opens a rows file is a write."""
    out, lines, i = [], command.split("\n"), 0
    while i < len(lines):
        out.append(lines[i])
        m = HEREDOC.search(lines[i])
        if m:
            keep = bool(_INTERPRETER.search(lines[i][:m.start()]))
            delim = m.group(1)
            i += 1
            while i < len(lines) and lines[i].strip() != delim:
                if keep:
                    out.append(lines[i])
                i += 1
        i += 1
    return "\n".join(out)


def _strip_heredocs(command):
    """`command` with heredoc BODIES removed, openers kept.

    A heredoc body is data the shell feeds to a program, so an invocation quoted
    inside one is a fixture or a doc example rather than a command. ⚠ Measured on
    a real transcript: the two false positives surviving the position check were
    both test fixture strings inside `python3 - <<'PYEOF'` bodies, and one
    read as command-position because the opening quote attached to the path's
    first component and left the basename bare.
    """
    out, lines, i = [], command.split("\n"), 0
    while i < len(lines):
        out.append(lines[i])
        m = HEREDOC.search(lines[i])
        if m:
            delim = m.group(1)
            i += 1
            while i < len(lines) and lines[i].strip() != delim:
                i += 1
        i += 1
    return "\n".join(out)


def _runs_memfind(command):
    """True where `command` EXECUTES memfind.py rather than merely naming it.

    Position is the discriminator, not presence. `grep memfind.py`, `cat
    memfind.py` and a heredoc quoting an invocation all NAME the script; only a
    segment whose command word is the script, optionally behind an interpreter
    and its flags, runs it. ⚠ The looseness this replaces was not theoretical: a
    `printf` writing a test fixture that happened to contain the string
    discharged the gate for a whole supervised session, silently.
    """
    for seg in SHELL_SPLIT.split(_strip_heredocs(command)):
        toks = seg.split()
        i = 0
        while i < len(toks) and ENV_ASSIGN.fullmatch(toks[i]):
            i += 1                                  # FOO=bar python3 ...
        if i >= len(toks) or toks[i][0] in "'\"":
            continue          # a quoted literal in command position is a string
        if PurePosixPath(toks[i]).name.startswith("python"):
            # Scan the rest of the segment rather than pinning the script to a
            # fixed offset: interpreter flags vary and some take a separate value
            # (`-X dev`, `-W once`), so counting them is a false negative waiting
            # to happen, and a false negative here has no escape hatch. A quoted
            # occurrence keeps its quotes through `split`, so `-c "print(...)"`
            # does not match on its own payload.
            rest = [PurePosixPath(t).name for t in toks[i + 1:]]
            if "memfind.py" in rest:
                return True
            if "locket.py" in rest and _locket_verb(rest[rest.index("locket.py") + 1:]) == "find":
                return True
        else:
            name = PurePosixPath(toks[i]).name
            if name == "memfind.py":
                return True
            if name in ("locket", "locket.py") and _locket_verb(toks[i + 1:]) == "find":
                return True
    return False


def _locket_verb(toks):
    """The verb `locket` was given: the first token after the command word, with
    a leading `--` stripped (`--index` and `index` are one verb). `locket find`
    is the question the 2e gate is about; `locket index` builds a cache and is
    not an asking."""
    return toks[0].lstrip("-") if toks else ""


def memfind_asked(transcript):
    """True where this session has actually run memfind, as opposed to naming it.

    A session reasoning about the tool mentions its filename constantly, and
    treating a mention as an invocation silences the gate on exactly the sessions
    doing the most memory work. So a candidate line is parsed and its Bash
    command inspected by `_runs_memfind`; the cheap byte test only decides which
    lines are worth parsing.

    ⚠ A tool RESULT is excluded, and this file is why: the test fixtures in tests/test_memscan.py
    carry the markers, so a session that merely Reads memscan.py would otherwise
    satisfy the gate from the Read's own result line. Reading a file that quotes an
    invocation is not invoking it.

    Fails open three ways, because a false negative here has no escape hatch: the
    write gate runs on Write and Edit, which carry no `MEMWRITE-OK` equivalent, so
    a session wrongly judged not to have asked cannot clear the gate by asking. An
    absent or unreadable transcript reads as asked. So does one whose lines carry
    the marker but expose no message content this parser recognises, which is what
    a changed transcript schema looks like from in here. ⚠ Recognising the SCHEMA
    is the fail-open test, not finding a Bash block: a line naming the script in
    prose is a shape this function understands perfectly and must answer no to.
    """
    if not transcript:
        return True
    seen_marker = parsed = False
    try:
        with open(transcript, "rb") as f:
            for line in f:
                if (b"memfind.py" not in line and b"locket" not in line) or b"toolUseResult" in line:
                    continue
                seen_marker = True
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if rec.get("type") == "response_item" and isinstance(rec.get("payload"), dict):
                    parsed = True               # a Codex rollout line
                    if _rollout_runs_memfind(rec["payload"]):
                        return True
                    continue
                content = ((rec.get("message") or {}).get("content")) or []
                if not isinstance(content, list) or not content:
                    continue
                parsed = True
                for block in content:
                    if not isinstance(block, dict) or block.get("name") != "Bash":
                        continue
                    cmd = (block.get("input") or {}).get("command")
                    if isinstance(cmd, str) and _runs_memfind(cmd):
                        return True
    except OSError:
        return True
    return seen_marker and not parsed


def _rollout_runs_memfind(item):
    """Whether one Codex rollout `response_item` is a shell call that runs memfind.
    A call is `{"type": "function_call", "name": ..., "arguments": "<JSON>"}`; the
    unified-exec tool puts the command in `cmd`, the older shell tool in `command`,
    as a string or an argv list. A `function_call_output` is a result, never a run."""
    if item.get("type") != "function_call":
        return False
    try:
        args = json.loads(item.get("arguments") or "{}")
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    cmd = args.get("cmd") or args.get("command") if isinstance(args, dict) else None
    if isinstance(cmd, list):               # ["bash", "-lc", "<script>"] runs the script
        cmd = (str(cmd[-1]) if len(cmd) >= 3 and cmd[-2] in ("-c", "-lc")
               else shlex.join(str(c) for c in cmd))
    return isinstance(cmd, str) and _runs_memfind(cmd)


def semantic_neighbours(root, text, top=3, timeout=4):
    """Closest existing claims by MEANING as (rows, stale), or None on any failure.

    Never raises and never builds an index; a missing or unreachable embedder
    returns None. This is the half lexical scoring is blind to: a paraphrase shares
    fewer than MIN_SHARED tokens and scores 0.0, indistinguishable from a novel
    fact. ⚠ Report the rows and never threshold them. The paraphrase and
    novel-fact score distributions overlap by 0.027, so a verdict drawn from one is
    wrong often enough to be worse than silence.
    """
    try:
        import memfind
        # no server spawn inside a hook's budget; the block restores the settings, so a
        # long-lived caller (the MCP server) keeps its own
        with memfind.settings(TIMEOUT=timeout, AUTOSTART=False):
            idx = memfind.load_cached(root)
            if idx is None:
                return None
            items, flat, dim, stale = idx
            rows = memfind.rank(text[:2000], root, top=top, idx=(items, flat, dim), lexical=False)
            return rows, stale
    except Exception:                       # ollama down, cache corrupt, anything
        return None


def memfind_gate(payload, name, tool_input, text, prior, root):
    """Fire the §2e question once per session carrying its own answer, else None.

    A process gate, not a verdict: it asks whether the question was posed, which is
    decidable, rather than whether the fact is new, which is not. The neighbour
    list rides along as information and decides nothing, and running memfind
    satisfies the gate even where ollama is down and it returns nothing.

    Autonomous runs are exempt. A cycle writes on a schedule with no session in
    front of it, cannot act on a nudge, and cannot report being blocked.
    """
    if os.environ.get("CLAUDE_AUTONOMOUS") == "1":
        return None
    # INDEX_FILES is deliberately NOT exempt here. Its lexical exemption exists
    # because a pointer line is meant to restate its target; that says nothing
    # about whether the corpus already holds an incoming fact, and MEMORY.md is
    # where a durable fact most often lands.
    sets = settings(root)
    if name in sets["append_surfaces"]:
        return None                         # chronology, and staging
    added = added_text(tool_input, text, prior)
    if added is None or len(added) < 200:
        return None
    # The same provenance subtraction the novelty nudge applies. Without it a
    # relocation clears that nudge and trips this one instead, which resurrects
    # the exact fault provenance-awareness exists to retire.
    added = subtract_archived(added, archived_claims(root))
    if len(added) < 200:
        return None
    if session_asked(payload):
        return None

    out = [f"Net-new material is about to land in {name}, and this session has not asked "
           f"the memory-corpus question yet: does this corpus already hold this idea? "
           f"Lexical scanning cannot answer it, because a paraphrase shares no tokens "
           f"with the claim it restates (prime-memory-discipline.md §2e)."]
    found = semantic_neighbours(root, added)
    if found is None:
        out.append("memfind could not run here, which does not lift the question. Ask it "
                   "directly, then repeat this write:")
    else:
        rows, stale = found
        out.append("Closest existing claims by meaning"
                   + (" (from a stale index):" if stale else ":"))
        for c, f, s in rows:
            out.append(f"  - {relay(f, 120)}   ({c:.2f})")
            out.append(f"      {relay(s, 150)}")
        out.append("Ranked neighbours, not a verdict: read the top file, then repeat this "
                   "write. To ask about any other statement:")
    out.append(f'  python3 {shlex.quote(str(Path(__file__).resolve().parent / "locket.py"))} find '
               f'"<statement>" {shlex.quote(str(root))}')
    out.append("Fires once per session.")
    return "\n".join(out)


def memfind_asked_db(db_path, session_id):
    """`memfind_asked` for a host that keeps sessions in SQLite rather than a
    transcript file: Hermes's `~/.hermes/state.db`, whose `messages` table holds
    one row per assistant turn with `tool_calls` as JSON. Fails open on a missing
    or locked database and on any shape it does not recognise, for the same
    reason the transcript reader does: a wrong no here cannot be cleared by asking.
    """
    if not session_id or not Path(db_path).is_file():
        return True
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = con.execute(
                "select tool_calls from messages where session_id=? "
                "and (tool_calls like ? or tool_calls like ?)",
                (session_id, "%memfind%", "%locket%")).fetchall()
        finally:
            con.close()
    except Exception:                        # locked, missing table, schema drift
        return True
    for (raw,) in rows:
        try:
            calls = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, ValueError):
            continue
        for call in calls if isinstance(calls, list) else []:
            fn = (call or {}).get("function") or {}
            if fn.get("name") not in ("terminal", "Bash"):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    continue
            cmd = (args or {}).get("command")
            if isinstance(cmd, str) and _runs_memfind(cmd):
                return True
    return False


def session_asked(payload):
    """Has THIS session run memfind, whichever host produced the payload.

    Claude Code hands the hook a `transcript_path`; Hermes hands a `session_id`
    and keeps the transcript in `state.db`. With neither source the answer is
    yes, which is fail-open by design and the reason `locket status` reports
    whether the gate is armed on this host: a gate that cannot see the session
    must not block it, and must not pretend it looked.
    """
    if payload.get("transcript_path"):
        return memfind_asked(payload.get("transcript_path"))
    if payload.get("session_id") and payload.get("_host") == "hermes":
        return memfind_asked_db(HERMES / "state.db", payload.get("session_id"))
    if payload.get("_host") == "mcp":       # the server lives the session and keeps count
        return bool(payload.get("_asked"))
    return True


USAGE = """memscan: duplication, drift and link checks over a memory corpus.

USAGE
  locket <mode> [corpus]          (memscan.py <mode> [corpus] by path)

  The corpus argument is optional and takes a shorthand (council, hermes)
  or registered name, any substring of a Claude Code project slug (myapp), a
  literal path, or `all` for every corpus on the machine in turn. Omit it to
  use the corpus the current directory sits inside, else the one belonging to
  it.

  A corpus is a directory of markdown files. A host's core directory is its
  COUNCIL store, recognised on sight: `~/.claude` on Claude Code, where the
  members are memory, rules, skills, agents and agent-memory plus the markdown
  at the root, and `~/.hermes` on Hermes, where the member is memories plus
  SOUL.md. Everything else under the home is host state and never corpus, and
  each `~/.claude/projects/<slug>/memory` is a store of its own. Any other
  directory becomes one with `init <dir>`, which writes a `locket.json` manifest into it
  and registers the path in `~/.locket/corpora.json` so that `corpora`, `all`
  and a name can find it. The manifest alone is honoured by the write-time
  gate and by a literal path; only the registry makes a store LISTABLE, since
  nothing searches the disk for manifests. The manifest may also name the
  store, rename the ledger, holding-space and index files this tool treats
  specially, exclude individual files (`excluded_files`: relative paths or
  globs, skipped by every mode and by the gate), declare the code roots
  `cites` resolves against, and register reference CSVs. A manifest acts only
  once a person has trusted its directory (`locket trust`); until then it is
  read as no manifest. The manifest, the registry row and the index under
  ~/.locket/cache are all this tool owns for a corpus.

MANIFEST
  `locket.json` at the corpus root, a JSON object (the older name `memfind.json`
  is still read; `migrate` renames it); `schema` writes its JSON
  Schema for an editor and checks a manifest against it. Every key is optional and
  a missing key takes the default shown; a broken file degrades to all the
  defaults rather than taking a hook down. `$schema` and `_comment` are ignored.
  This is the whole list, grouped by the question each key answers.

  IS THIS FILE CORPUS AT ALL?
  An excluded file does not exist for the tool: nothing indexes, pairs or
  guards it. To keep a file in the corpus but change how it is scored, give it
  a surface role in the next group instead.

  members          Which top-level directories are corpus? A list of names;
                   the rest of the root is host state, and markdown at the
                   root is always in. Default: every directory, except on a
                   host's core directory, where it is the host's member list
                   above. REPLACES that list, so a Hermes that authors its own
                   skills writes ["memories", "skills"].
  excluded_dirs    Which directories, at any depth, are not corpus? A list of
                   names. REPLACES the default: archive, in-flight,
                   harness-logs, discovery-research. Whatever the list says,
                   .memfind, backups, backup and every dot-directory stay
                   excluded.
  excluded_files   Which single files are not corpus? A list of
                   corpus-relative paths or glob patterns, each tried against
                   the relative path and the basename: "notes/draft.md",
                   "*.csv", "scratch-*". Skipped by every mode, the index and
                   the write-time gate. Also removes a registered CSV it
                   matches. Use it for a file that holds no claims, such as a
                   draft or a template whose instances copy its shape.
                   Default: none.

  HOW IS THIS FILE SCORED FOR FORKS?
  Each list names files by basename and gives them a surface role. A corpus
  file in none of them is DURABLE: every claim in it is scored against every
  other file in the store. Each list REPLACES its default.

  index_files      Which files POINT at other files? A line restating its
                   target is the index doing its job, not a fork. Default:
                   MEMORY.md, ARCHIVE.md, CURATION-INDEX.md,
                   CURATION-SOURCES-INDEX.md, FRONTEND-INDEX.md,
                   PLATFORM-INDEX.md.
  ledger_surfaces  Which markdown files grow by appending entries? They are
                   never scored, because chronology is their value and an
                   append is the discipline, not a breach. Not the same as
                   `ledgers` below, which checks the row shape of a CSV.
                   Default: HARNESS.md, CHECKPOINTS.md, EXPLORATIONS.md.
  holding_spaces   Which files stage entries on their way to a durable home?
                   Scored only against THEMSELVES: a restated entry fires, a
                   fact already held elsewhere does not, since an entry is
                   meant to resemble what it may become. Unlike an excluded
                   file it is fully corpus: indexed, searchable, and read by
                   `graduated` for entries whose lesson has landed elsewhere.
                   Default: SHORT_TERM.md, OPEN_QUESTIONS.md,
                   MEMORY-proposed.md, PROMOTIONS.md.

  MAY THIS FILE OUTGROW A WHOLE READ?
  whole_read_exempt
                   Which files does the SessionStart size sweep skip? A list
                   of names: a mixed surface its own load contract says is
                   never read whole, or a file a project instrument already
                   caps. It exempts nothing else; every other mode still reads
                   the file, and a file may carry this and a surface role at
                   once. Default: none.

  WHICH CSV ROWS COUNT, AND IN WHAT SHAPE?
  A CSV is scored for forks only when `sources` lists it, and its rows are
  checked for shape only when a `ledgers` schema governs it; a rulings file
  usually takes both.

  sources          Which CSVs hold claims? A list of {"path", "text",
                   "label"}: `path` is the file, absolute or relative to the
                   store, `text` the column holding the sentence, `label` the
                   column that names a row in output (an Area). Default: the
                   older standalone reference-sources.json if present, else
                   none.
  ledgers          What shape must a CSV ledger's rows take? A list of
                   {"path", "extends", "columns", "required", "caps", "enum",
                   "pattern"}; a row outside its schema is refused at write
                   time. It says nothing about forks. `path` is a basename
                   glob or a store-relative path; `columns` the exact header;
                   `required` the columns that may not be empty; `caps`
                   {column: max characters}; `enum` {column: [values]}, ""
                   among them where empty is allowed; `pattern` {column: regex
                   the whole value must match}. Every key but `path` is
                   optional. A built-in schema governs its glob wherever the
                   file sits, corpus or not, and is the authority on it. An
                   entry naming that glob, or a path under it, merges onto it
                   for this store: `caps`, `enum` and `pattern` change only the
                   columns they name, `columns` and `required` replace.
                   `extends` names a built-in's glob to merge onto for a file
                   whose name falls outside it.
                   Default: the built-ins alone, which are:
@@BUILTIN_LEDGERS@@

  HOW DOES THIS STORE CONNECT TO THE REST?
  name             What do `corpora` and a store argument call this store? A
                   string. Default: the directory's basename. Host stores and
                   project memories are named by the host and ignore it.
  belongs_to       Which corpus does this directory JOIN? The name of another
                   corpus as `corpora` lists it. Every mode, the index and the
                   write-time gate then resolve a file here to the parent, the
                   listing shows the parent alone, and its files appear as
                   `<name>/<path>` (this store's `name`, else its basename).
                   The store keeps its own `members`, `excluded_*` and
                   `code_roots` for its own walk; the file-name lists union
                   upward. Written by `init <dir> --parent <corpus>`. A name
                   that resolves to nothing, to itself, to a store under it on
                   disk or whose own walk reaches it, or into a cycle is
                   ignored and the store stands alone. Default: none.
  code_roots       Where do `cites` path:line citations resolve? A list of
                   directories, taken outright. Default: the project's own
                   working directory (for a project memory) plus ~/.claude and
                   ~/.hermes where they exist.
  triggers         Which rows file does `locket trigger` read for this store?
                   A path, store-relative or absolute. Default: triggers.json
                   at the store root.

  On Hermes's ~/.hermes the three surface-role lists default to empty
  instead, because its MEMORY.md is the whole memory rather than an index; a
  manifest there still wins. Store the file under version control: it is
  authored configuration, where the index is derived and rebuilt.

MODES
  audit     cross-file claim pairs that currently AGREE, so fork RISK rather
            than proof of a fork. An index line matching its own target is the
            index doing its job, not a finding.
  across    claim pairs that NO single store holds both sides of, over every
            store on the machine at once. Every other mode and the write-time
            gate score a corpus against itself, so a fact written into two
            stores is invisible to both until this runs. A file registered by
            two manifests is one file here, named with every holder. Tags a
            pair with a holding-space side: promoted into another store and
            never retired, which `graduated` cannot see. Takes no corpus.
  dupes     repeated claims INSIDE one file, which audit skips by construction.
            Use on the holding spaces (SHORT_TERM, OPEN_QUESTIONS, proposed).
  graduated holding-space claims a durable file or CSV row now holds: an entry
            promoted and never retired. Prints its scope even when clean, so a
            zero says what it was scored against. Scores ONE store: a promotion
            into a second store is `across`'s to tag. Run with `links` and
            `across` at session close.
  ledgers   every schema-governed CSV in the corpus, row by row: a header off
            the schema, a required field empty, a field over its cap, outside
            its enum or off its pattern. The same check the write-time gate
            runs on a row as it is written, over the rows already there.
            Then every declared CSV (a ledger or a registered source) past
            5 MB, each a finding: the file stays whole, and `doctor` times
            the write hook on the largest.
  budget    bytes, dated history clauses listed by line (the reconsolidation
            worklist: `*Refinement, <date>:*`, `<date>: Resolved:`, `Update
            <date>`), longest prose paragraph. Nothing else measures a file
            that only disagrees with the world.
  pointers  every index line paired to the file it points at, a missing
            target first. NEEDS A READER: a hook asserting what its target now
            refutes scores LOW and is invisible to audit. `hooks` is its older
            name; whether the write-time gate is armed is `locket status`.
  links     wikilink targets resolving to no file.
  cites     `path:line` citations resolved against the code, with the enclosing
            def/class for each. Prints a missing file or a past-EOF line as a
            defect; everything else NEEDS A READER, since only a reader knows
            whether the scope is the one the sentence claimed. Takes optional
            path substrings to narrow the list.
  corpora   every corpus on this machine, with file and CSV-row counts, and
            any registered path whose manifest has since gone.
  init DIR  make DIR a corpus every mode can find: manifest, registry row,
            index. Idempotent. The one command for a folder that already
            holds markdown. With `--parent NAME`, DIR joins the corpus NAME
            instead (`belongs_to` in its manifest, index rebuilt at the
            parent): two folders in two places, one corpus. Works on a
            project memory too, which otherwise takes no manifest. The
            joined files are keyed `<label>/<path>`, the label being the
            manifest's `name` or else the folder's; `--name LABEL` sets it,
            and a label clashing with the parent's name or one of its
            directories is refused. A host auto-memory folder is always
            `memory`, so it takes `--name`. A standalone store's manifest is
            written with `name` (the folder's, or `--name LABEL`) and nothing
            else, so every other key keeps tracking the built-in default.
            With `--ledgers[=NAME]`, also writes HISTORY-NAME-Sessions.csv,
            RULINGS-NAME.csv and CLAIMS-NAME.csv at DIR (header row only,
            NAME defaulting to the store's) and registers the last two as
            sources: rows the write-time check refuses off-schema, and
            rulings and claims ranked beside the markdown. Existing files
            are kept.
  schema [DIR]  write the manifest's JSON Schema to ~/.locket/locket.schema.json
            (`init` does this too), so an editor mapped to that path hints every
            `locket.json` by filename; with DIR, check that directory's manifest
            against it instead: an unknown key with a did-you-mean, a wrong type,
            a malformed `sources` or `ledgers` entry. `locket help install`
            names the editor side.
  migrate [DIR|all]  rename a store's older `memfind.json` manifest to
            `locket.json` in place; every store on the machine when DIR is
            omitted. Until a store is migrated the old name is still read.
  trust DIR  let DIR's manifest act: its trigger rows, ledger schemas and
            sources. Asks a person at a terminal to type yes; an agent cannot
            grant it. Until then a manifest outside the host stores is inert.
            Trust is pinned to DIR's locket.json, reference-sources.json and
            rows file as they stand: after editing any of them, or a pull that
            changes them, the store is inert until you trust it again.
  forget DIR  drop DIR's registry row, which also withdraws its trust (`untrust`
            is the same verb), once a person confirms at a terminal. Its
            manifest stays, inert.

HOOK MODES (PreToolUse entry points, JSON on stdin; not for manual use)
  Claude Code (matcher: tool names below), Codex (the same, in hooks.json) and
  Hermes (`pre_tool_call`, matcher write_file|patch, terminal, search_files)
  hand the same payload; the tool names are mapped on the way in, and a Codex
  patch, native or through the shell, is read as the Write or Edit it amounts
  to. Hermes can hear a BLOCK and nothing else on a
  tool event, so a nudge there refuses the call once, leading with how to
  proceed: the same call repeated passes once state.db shows the nudge was
  delivered, and the third attempt passes regardless.
  deliver     retired; does nothing, so an older Hermes config that still
              registers it on `pre_llm_call` keeps running. Remove the entry.
  hook        Write|Edit. On a `.csv` path a schema governs, the ledger check:
              a row over a cap, outside its enum or off its pattern BLOCKS,
              the one refusal here on a question the tool can decide (§2d).
              On markdown, the 2c novelty nudge and the 2e memfind gate. Both
              subtract text already filed under archive/ first, since 4 archives
              superseded text BEFORE it is rewritten over, so a match there is a
              move rather than a fork. The novelty nudge WARNS in a supervised
              session and blocks only under CLAUDE_AUTONOMOUS; the 2e gate
              blocks, being dischargeable by running one command.
  bashguard   Bash. A heredoc or echo into a schema-governed ledger is read as
              rows and checked as `hook` checks them; a redirect whose rows it
              cannot read warns and names the sweep. Then the same corpora, written through a shell redirect, a
              heredoc or an in-place edit (sed, perl or ruby -i, gawk -i
              inplace), which `hook` never sees because it matches TOOLS
              rather than paths. Blocks an unambiguous write, warns on a
              scripted one and on a memory file handed to any command not
              known to be read-only. A Python heredoc or `-c` program is
              parsed and judged by where its writing calls write, so a
              memory file named only as data passes. Silent for ledgers, holding spaces,
              archive/, the scratchpad and autonomous runs. `MEMWRITE-OK:
              <reason>` in the command declares the exception and passes.
  grepassist  Grep|Bash. Runs memfind on a prose-shaped search pattern and
              injects the ranked neighbours, because 2e says ask the question
              BEFORE the grep and prose alone did not achieve it. Fires only on
              a pattern carrying no regex metacharacter and two words of four
              characters or more, so an identifier or an alternation never
              reaches it. Resolves the corpus from the search path, else from
              the cwd. Silent with the embedder down. Every grep it reads is
              logged, returned or declined and why, with the pattern's first
              80 characters, to ~/.locket/grepassist.csv (mode 0600;
              LOCKET_GREPASSIST_LOG), so a hook gone quiet shows as a rate.

EXAMPLES
  locket corpora                 # what corpora exist, and their sizes
  locket audit all               # forks in every corpus, one after another
  locket audit myapp             # forks across one project's corpus
  locket audit                   # ... same, if cwd is inside that project
  locket budget council          # oversized files across the council store
  locket dupes ~/.claude/memory  # a literal path scopes a run to one member
  locket links council           # dangling [[wikilinks]] and CSV pointers
  locket cites myapp             # do the code citations still point anywhere
  locket cites myapp models.py   # ... only the ones naming models.py

Every mode prints the directory it resolved and whether the write-time hook
reaches it, because a literal path can name a directory the hook never guards.
The module docstring carries the design rationale."""


# ---------------------------------------------------------------------------
# Bash-mediated writes, and searches, are the two blind spots of a Write|Edit gate
# ---------------------------------------------------------------------------
# The write-time gate matches the Write and Edit TOOLS. A memory file rewritten by
# a shell redirect or a python heredoc never reaches it, so the 2c/2e checks do not
# fire and nothing says they did not. A full memory-maintenance session wrote
# every one of its corrections through heredocs and tripped the gate zero times,
# which is the measurement behind this file.
# This is a deliberate carve-out from any session directive preferring Bash for
# file edits. It applies to memory corpora only; everywhere else Bash is fine.

# --- Ledger CSVs: schema-checked append surfaces ------------------------------
# A ledger is queried, never loaded whole, so it may grow freely; what it may not
# do is grow a row past the shape a query can scan. The schema is the one thing
# about a ledger that is decidable at write time (§2d: an instrument blocks only
# on a question it can decide), so a row over a cap, outside an enum or off its
# pattern is REFUSED, where every prose check only warns. The built-in schemas are
# the session-history ledger every project keeps and the rulings and claims
# ledgers a store keeps once it has earned them (`RULINGS.csv` in a single store,
# `RULINGS-<Project>.csv` in a project, so one glob covers both). Each is keyed by
# a basename glob and applies wherever the file is, corpus or not, because the
# project root a ledger sits in is usually no store at all. Rulings and claims
# carry no text cap: their rows range past 2,000 characters in live stores, and a
# store that wants one sets it in its manifest. `Category` and `Area` take one
# lowercase key or several joined by `|`, because `Area=X` is how every pointer
# queries them; only a claim carries a `Source`. A manifest's `ledgers` list
# adds a schema for any other CSV in that store, or adjusts a built-in for its
# own file by naming the same glob or a path under it, merging onto it
# (`_merge_schema`). This constant is each schema's one home: `usage()` renders
# it into the help, and the project-dev skill points here rather than restating it. The date takes an optional UTC time, because
# a local-evening session lands on the next UTC day and a bare date cannot say
# which side of midnight it came from. `repo` names the repository the hash
# resolves in (`group/sub/repo` allowed), `issue` is one tracker-agnostic ticket
# key (`#12`, `ENG-42`) and `pr` a GitHub PR or GitLab MR (`#155`, `!155`).
_LEDGER_KEYS = r"[a-z][a-z0-9-]*(?:\|[a-z][a-z0-9-]*)*"
LEDGER_SCHEMAS = {
    "HISTORY-*-Sessions.csv": {
        "columns": ["date", "title", "summary", "area", "repo", "commit_hash", "issue", "pr"],
        "required": ["date", "title", "summary"],
        "caps": {"title": 90, "summary": 240},
        "pattern": {"date": r"20\d\d-\d\d-\d\d(?:T\d\d:\d\d(?::\d\d)?Z)?",
                    "repo": r"(?:[\w.-]+(?:/[\w.-]+)*)?",
                    "commit_hash": r"(?:[0-9a-f]{7,40})?",
                    "issue": r"(?:\S{1,40})?",
                    "pr": r"(?:[#!]\d+)?"},
    },
    "RULINGS*.csv": {
        "columns": ["Ruling", "Category", "Area", "Date"],
        "required": ["Ruling", "Category", "Area", "Date"],
        "pattern": {"Category": _LEDGER_KEYS, "Area": _LEDGER_KEYS, "Date": r"20\d\d-\d\d-\d\d"},
    },
    "CLAIMS*.csv": {
        "columns": ["Claim", "Category", "Area", "Source", "Date"],
        "required": ["Claim", "Category", "Area", "Source", "Date"],
        "pattern": {"Category": _LEDGER_KEYS, "Area": _LEDGER_KEYS, "Date": r"20\d\d-\d\d-\d\d"},
    },
}
_SCHEMA_KEYS = ("columns", "required", "caps", "enum", "pattern")
# The size past which `ledgers` and `doctor` report a declared CSV. A report, never
# a cap: the file keeps every row and stays whole (RULINGS.csv on Area=substrate).
CSV_REPORT_BYTES = 5 * 1024 * 1024


def manifest_help():
    """The MANIFEST section of `usage()` alone, for `locket help manifest`: a
    slice of the one text, never a second copy of it."""
    text = usage()
    start = text.index("\nMANIFEST\n") + 1
    return text[start:text.index("\nMODES\n", start)].rstrip() + "\n"


def usage():
    """USAGE with the built-in ledger schemas rendered from LEDGER_SCHEMAS, so the
    help is a view of the constant and never a copy of it."""
    pad = " " * 19
    lines = []
    for glob, sch in LEDGER_SCHEMAS.items():
        lines.append(f"{pad}  {glob}: {','.join(sch.get('columns', []))}")
        if sch.get("required"):
            lines.append(f"{pad}    required {', '.join(sch['required'])}")
        if sch.get("caps"):
            lines.append(f"{pad}    caps " + ", ".join(f"{c} ≤{n}" for c, n in sch["caps"].items()))
        for c, vals in sch.get("enum", {}).items():
            lines.append(f"{pad}    {c} in " + "|".join(v or "<empty>" for v in vals))
        for c, rx in sch.get("pattern", {}).items():
            lines.append(f"{pad}    {c} matches {rx}")
    return USAGE.replace("@@BUILTIN_LEDGERS@@", "\n".join(lines))


def ledger_schemas(root):
    """{glob: schema} in force for a corpus: the built-ins, then every `ledgers`
    entry in the manifest of each physical store composing it, later entries
    overriding earlier ones by glob. Malformed entries are skipped."""
    out = {k: dict(v, _from="built in") for k, v in LEDGER_SCHEMAS.items()}
    for r in (roots_of(root) if root is not None else []):
        entries = _read_manifest(r).get("ledgers")
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("path"), str):
                continue
            own = {k: e[k] for k in _SCHEMA_KEYS if k in e}
            base = LEDGER_SCHEMAS.get(e.get("extends")) or _builtin_for(e["path"])
            out[e["path"]] = _merge_schema(base, own)
            out[e["path"]]["_from"] = f"{manifest_file(r).name} in {r}"
    return out


def _builtin_for(path):
    """The built-in schema a manifest entry's path falls under, by its own text
    or its basename, or {}."""
    name = PurePosixPath(path).name
    for glob, schema in LEDGER_SCHEMAS.items():
        if path == glob or fnmatch.fnmatchcase(path, glob) or fnmatch.fnmatchcase(name, glob):
            return schema
    return {}


def _merge_schema(base, own):
    """`own` over `base`: a dict-valued key (caps, enum, pattern) merges column by
    column, a list-valued key (columns, required) replaces. A store overriding one
    column's enum keeps the built-in's date pattern and caps, rather than having to
    restate them in a copy that goes stale when the built-in changes."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in own.items():
        out[k] = ({**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v)
    return out


def ledger_schema_for(path):
    """(schema, glob) governing a CSV path, or (None, None). A manifest entry is
    matched against the path relative to its store and against the basename;
    a built-in against the basename anywhere inside a trusted store. Outside
    every store no schema applies: a cloned repo's `RULINGS.csv` must not be
    able to block a write or put its header text in front of the agent."""
    p = Path(path).expanduser()
    if p.suffix.lower() != ".csv":
        return None, None
    phys = _phys_root_of(p.parent)
    if phys is None:
        return None, None
    root = top_of(phys)
    try:
        rel_path = p.relative_to(root).as_posix() if root is not None else p.name
    except ValueError:
        rel_path = p.name
    for glob, schema in reversed(list(ledger_schemas(root).items())):
        if fnmatch.fnmatchcase(p.name, glob) or fnmatch.fnmatchcase(rel_path, glob):
            return schema, glob
    return None, None


def check_ledger_rows(schema, header, rows, first=2):
    """Findings for data rows against a schema, each `row N: <what>`; N counts
    from `first`, the file line the first data row sits on. A header that does
    not carry the schema's columns is one finding and ends the check, because
    every column test below would then be naming the wrong field."""
    cols = schema.get("columns")
    if isinstance(cols, list) and [c.strip() for c in header] != list(cols):
        return [f"header is {','.join(header)}; the schema wants {','.join(cols)}"]
    idx = {c.strip(): i for i, c in enumerate(header)}
    out = []
    for n, row in enumerate(rows, first):
        if not any(x.strip() for x in row):
            continue                                   # a blank line is not a row
        if len(row) != len(header):
            out.append(f"row {n}: {len(row)} fields, header has {len(header)}")
            continue
        val = lambda c: row[idx[c]].strip() if c in idx else None
        for c in schema.get("required", []) or []:
            if val(c) == "":
                out.append(f"row {n}: {c} is required and is empty")
        for c, cap in (schema.get("caps") or {}).items():
            v = val(c)
            if v is not None and isinstance(cap, int) and len(v) > cap:
                out.append(f"row {n}: {c} is {len(v)} characters, cap {cap}")
        for c, allowed in (schema.get("enum") or {}).items():
            v = val(c)
            if v is not None and isinstance(allowed, list) and v not in allowed:
                shown = "|".join(a if a else "<empty>" for a in allowed)
                out.append(f"row {n}: {c} is {v!r}, allowed {shown}")
        for c, pat in (schema.get("pattern") or {}).items():
            v = val(c)
            try:                            # a value past MAX_SCAN is never handed to a pattern
                ok = v is None or (len(v) <= MAX_SCAN and re.fullmatch(pat, v) is not None)
            except re.error:
                ok = True
            if not ok:
                out.append(f"row {n}: {c} is {v!r}, wanted /{pat}/")
    return out


def _csv_rows_of(text):
    import csv as _csv, io as _io
    return list(_csv.reader(_io.StringIO(text)))


def check_ledger_text(schema, text, header=None):
    """Findings for CSV `text`. With `header`, every parsed line is a data row
    (an Edit fragment, a heredoc); a line whose field count differs from the
    header's is not a row and is skipped, since a fragment can cut a row. Without
    it, the first line is the header (a whole-file Write, the sweep)."""
    rows = _csv_rows_of(text)
    if header is None:
        if not rows:
            return []
        return check_ledger_rows(schema, rows[0], rows[1:], first=2)
    rows = [r for r in rows if len(r) == len(header)
            and [c.strip() for c in r] != [c.strip() for c in header]]
    return check_ledger_rows(schema, header, rows, first=1)


def _ledger_header(path):
    try:
        with open_safe(path, newline="") as fh:
            rows = _csv_rows_of(fh.read(4096))
    except OSError:
        return None
    return rows[0] if rows else None


def ledger_message(path, glob, schema, findings, via):
    return (f"A ledger row breaks its schema and the write is refused ({Path(path).name}, "
            f"schema {glob!r}, {schema.get('_from', 'built in')}):\n  - " + "\n  - ".join(relay(f, 240) for f in findings[:12])
            + (f"\n  - ... {len(findings) - 12} more" if len(findings) > 12 else "")
            + f"\nA ledger is queried, never read whole, so a row is scanned rather than read: "
              f"shorten the field to the cap, or file the detail where the row points. "
              f"The cap is the ledger's design and {via} enforces it; `locket ledgers` "
              f"sweeps the rows already there.")


def ledger_decision(path, tool_input):
    """(message, blocking) for a Write or Edit into a schema-governed CSV."""
    schema, glob = ledger_schema_for(path)
    if schema is None:
        return None, False
    if "content" in tool_input and tool_input.get("content") is not None:
        findings = check_ledger_text(schema, str(tool_input["content"]))
    else:
        header = _ledger_header(path)
        if header is None:
            return None, False
        findings = check_ledger_text(schema, str(tool_input.get("new_string") or ""), header)
    if not findings:
        return None, False
    return ledger_message(path, glob, schema, findings, "the write-time hook"), True


def _source_texts(body, textcol):
    import csv as _csv, io
    try:
        return [(r.get(textcol) or "").strip() for r in _csv.DictReader(io.StringIO(body))]
    except _csv.Error:
        return []


def source_row_decision(path, tool_input, prior, append=False):
    """(message, False) for a write into a registered source CSV: each row it adds,
    scored against the rest of the store, so a ruling that restates prose is named
    before `audit` would find it. Warn-only in every mode, a cycle included:
    whether a fact is new is not decidable, and moving a passage into a row
    rightly draws this until the prose is cut."""
    p = Path(path).expanduser()
    phys = _phys_root_of(p.parent)
    if phys is None:
        return None, False
    root = top_of(phys)
    src = [(s, t, lab) for s, t, lab in csv_sources(root) if s.resolve() == p.resolve()]
    if not src:
        return None, False
    textcol = src[0][1]
    if tool_input.get("content") is not None:
        post = str(tool_input["content"])
    elif tool_input.get("old_string"):
        post = prior.replace(str(tool_input["old_string"]), str(tool_input.get("new_string") or ""), 1)
    elif append:
        post = prior.rstrip("\n") + "\n" + str(tool_input.get("new_string") or "")
    else:
        return None, False
    before = Counter(_source_texts(prior, textcol))
    after = _source_texts(post, textcol)
    added, seen = [], Counter()
    for t in after:
        seen[t] += 1
        if t and seen[t] > before[t]:
            added.append(t)
    if not added:
        return None, False
    # The file's own rows are scored as they will stand after the write, minus the
    # new ones, so an edited row never matches the version it replaces.
    docs = {k: v for k, v in corpus(root).items() if not k.startswith(f"{p.name}#")}
    kept = Counter(after) - Counter(added)
    docs[p.name] = "\n".join(strip(t) for t in kept.elements() if t)
    lines = []
    for row in added:
        ranked = [r for r in score_against(row, docs)[:3] if r[1] >= FIRE_FLOOR]
        if ranked:
            lines.append(f"  row: {relay(row, 120)}")
            for _c, best, fname, pair in ranked:
                lines.append(f"    - {fname}   (closest claim {best:.2f})"
                             + (f"\n        existing: {relay(pair[1], 160)}" if pair else ""))
    if not lines:
        return None, False
    return (f"{LEAD_WARNING} {p.name} is a registered source, and a row here may restate a claim "
            f"the store already carries:\n" + "\n".join(lines) +
            "\nOne fact, one row: cut the prose copy to a pointer, or fold the row into the one "
            "that owns it (prime-memory-discipline.md §2c). A new fact needs nothing."), False


_QUOTED = re.compile(r"""(?:echo|printf)\s+(?:-[a-zA-Z]+\s+)*(?:"((?:[^"\\]|\\.)*)"|'([^']*)')""")
_CSV_REDIRECT = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*(?P<p>[^\s;&|<>]+\.csv)")
# a pipeline boundary: a pipe stays inside, because `printf ... | tee -a x.csv` is one write,
# and so does a backslash-continued newline
_PIPELINE_END = re.compile(r"\|\||&&|;|(?<!\\)\n")


def _pipeline_around(command, start, end):
    """The text of the pipeline holding command[start:end]."""
    before = [m.end() for m in _PIPELINE_END.finditer(command, 0, start)]
    after = _PIPELINE_END.search(command, end)
    return command[before[-1] if before else 0:after.start() if after else len(command)]


def ledger_bash_decision(command, cwd):
    """(message, blocking) for a shell command appending to a schema-governed CSV.

    A heredoc body or an echo/printf string is read as rows and checked, and a
    violation blocks. Each body counts only against the ledger its own pipeline
    redirects into, so a second redirect elsewhere in a compound command is never
    read as the ledger's rows. A redirect into the ledger whose rows cannot be read
    off the command (a script, a variable) warns instead, because the check cannot
    decide it, and names the sweep that can."""
    if ".csv" not in command:
        return None, False
    targets, owner = {}, {}
    for m in _CSV_REDIRECT.finditer(command):
        tok = m.group("p").strip("'\"")
        cands = [tok] if tok.startswith("/") else [str(Path(r) / tok) for r in _cd_targets(command, cwd, m.start("p"))]
        for c in cands:
            schema, glob = ledger_schema_for(c)
            if schema is not None:
                owner[tok] = c
                targets[c] = (schema, glob)
                break
    if not targets:
        return None, False
    # keyed by resolved path, so two spellings of one ledger share one verdict
    bodies = {c: [] for c in targets}

    def attribute(pipeline, body):
        for c in {owner.get(m.group("p").strip("'\"")) for m in _CSV_REDIRECT.finditer(pipeline)}:
            if c in bodies:
                bodies[c].append(body)

    lines, i, starts = command.split("\n"), 0, []
    pos = 0
    for line in lines:
        starts.append(pos); pos += len(line) + 1
    while i < len(lines):
        m = HEREDOC.search(lines[i])
        if m:
            at, delim, body = starts[i], m.group(1), []
            i += 1
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i]); i += 1
            # the opener's pipeline in the full command, so a redirect on a continued line counts
            attribute(_pipeline_around(command, at + m.start(), at + m.end()), "\n".join(body))
        i += 1
    for m in _QUOTED.finditer(command):
        attribute(_pipeline_around(command, m.start(), m.end()),
                  (m.group(1) if m.group(1) is not None else m.group(2))
                  .replace("\\n", "\n").replace('\\"', '"'))
    # every target is checked before any warns: an opaque write earlier in the command
    # must never stand in for a readable, violating one later
    for path, (schema, glob) in targets.items():
        header = _ledger_header(path)
        findings = []
        for body in bodies[path]:
            if header is None:
                findings += check_ledger_text(schema, body)
            else:
                findings += check_ledger_text(schema, body, header)
        if findings:
            return ledger_message(path, glob, schema, findings, "bashguard"), True
    for path in targets:
        if not bodies[path]:
            return (f"This appends to a schema-governed ledger ({Path(path).name}) with rows the "
                    f"guard cannot read off the command, so the caps go unchecked here. Run "
                    f"`locket ledgers` after it, or write the row through Write/Edit, "
                    f"where the check is mechanical."), False
    return None, False


def _schema_csvs(root):
    """[(path, schema, glob)] for every schema-governed CSV the store's walk finds."""
    seen, files = set(), []
    # The store's own walker, members and excluded directories honoured, so a
    # ledger under a member directory (a store's `memory/RULINGS.csv`) is swept;
    # a root-only glob reported such a store clean while checking nothing.
    # `excluded_files` hides a file from claims, not from its schema.
    for r in roots_of(root):
        for p in _own_md(r, "*.csv", honor_excluded_files=False):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            # the gate's own resolver, so a file an entry names inside a
            # built-in's glob is checked once, under the entry, as it is written
            schema, g = ledger_schema_for(p)
            if schema is not None:
                files.append((p, schema, g))
    return files


def declared_csvs(root):
    """[(path, text column or None)] for every CSV the store declares: each
    schema-governed ledger and each registered source, once. These are the
    files the write hook reads on every row written into them."""
    out, seen = [], set()
    for p, text, _label in csv_sources(root):
        seen.add(p.resolve())
        out.append((p, text))
    for p, _schema, _glob in _schema_csvs(root):
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append((p, None))
    return out


def large_csvs(root, limit=None):
    """[(path, bytes)] for each declared CSV past CSV_REPORT_BYTES, largest first.
    A declared CSV keeps every row in one file and is never split, because a split
    hides rows from every reader that names the table (RULINGS.csv on
    Area=substrate); this is its breach reading, a monitor rather than a cap."""
    limit = CSV_REPORT_BYTES if limit is None else limit
    out = []
    for p, _t in declared_csvs(root):
        try:
            n = p.stat().st_size
        except OSError:
            continue
        if n > limit:
            out.append((p, n))
    return sorted(out, key=lambda x: -x[1])


def cmd_ledgers(root):
    """Every schema-governed CSV in the corpus, each row checked, then every
    declared CSV past the size reading. A clean run prints its scope, so a zero
    says what it was scored against."""
    schemas = ledger_schemas(root)
    files = _schema_csvs(root)
    big = large_csvs(root)
    if not files and not big:
        print(f"no schema-governed ledger in {root} (schemas: {', '.join(schemas)})")
        return 0
    bad = 0
    for p, schema, glob in files:
        try:
            body = read_safe(p)
        except OSError as e:
            print(f"{corpus_key(root, p)}  schema {glob!r}  NOT READ: {e}")
            bad += 1
            continue
        findings = check_ledger_text(schema, body)
        rows = max(len(_csv_rows_of(body)) - 1, 0)
        print(f"{corpus_key(root, p)}  schema {glob!r}  {rows} rows, {len(findings)} findings")
        for f in findings:
            print(f"  {f}")
        bad += len(findings)
    for p, n in big:
        print(f"{corpus_key(root, p)}  size {n / 1048576:.1f} MB, past the "
              f"{CSV_REPORT_BYTES / 1048576:g} MB reading: run `locket doctor` for the write "
              f"hook's latency on it, and choose a remedy against what it shows; never split it")
    if big:
        print(f"{len(big)} declared CSV(s) past the {CSV_REPORT_BYTES / 1048576:g} MB reading")
    bad += len(big)
    print(f"{len(files)} ledger(s) checked, {bad} finding(s)")
    return bad


SCRATCH_MARKERS = ("/scratchpad/", "/private/tmp/", "/tmp/")
MEMWRITE_ESCAPE = "MEMWRITE-OK:"
# Destination-position writes. Direction is the whole point: a corpus path as the
# SOURCE of a cp is a read, and blocking that would break every
# backup-before-rewrite command this discipline actually requires.
BASH_WRITE = (
    re.compile(r">>?\s*(?P<p>[^\s;&|<>]+\.md)"),
    # the path starts a word (after a space, a quote or `=`): starting it mid-word made one
    # 32 KB argument cost seconds of backtracking inside a single match attempt
    re.compile(r"\btee\b[^;&|]*?(?<![^\s'\"=])(?P<p>[^\s;&|<>]+\.md)"),
    # `-i` in a flag cluster of its own (`-i`, `-Ei`, `-i.bak`, `-ibak`, `--in-place`), never
    # the `-i` inside `make-it-yours.md`, which read a `sed -n` as a write to `t-yours.md`
    re.compile(r"\bg?sed\b[^;&|]*?(?<!\S)-(?:[nEsuzr]*i|-in-place)[^;&|]*?(?<![^\s'\"])(?P<p>[^\s;&|<>]+\.md)"),
    # perl and ruby share sed's `-i` (`-pi`, `-i.bak`, `-0pi`); gawk spells it `-i inplace`
    # only argument-free flags before the `i`, as in _INPLACE: `-MList::Util` and `-CSDio` are not it
    re.compile(r"\b(?:perl|ruby)\b[^;&|]*?(?<!\S)-[0-9lnpaWwsSTtUuXc]*i[^;&|]*?(?<![^\s'\"])(?P<p>[^\s;&|<>]+\.md)"),
    re.compile(r"\bg?awk\b[^;&|]*?(?<!\S)-i\s*inplace\b[^;&|]*?(?<![^\s'\"])(?P<p>[^\s;&|<>]+\.md)"),
    re.compile(r"\b(?:cp|mv|rsync)\b[^;&|]+?\s(?P<p>[^\s;&|<>]+\.md)\s*(?:$|[;&|])"),
)
# Where each pattern above may start: the command word, or None for a redirect. A
# pattern is tried from its first start in a clause and nowhere after a miss there,
# because a lazy span retried from every `sed -i` in one clause cost 9.5 s on 4 KB:
# a match from a later start is also a match from the first, so one miss settles it.
BASH_WRITE_STARTS = (None, re.compile(r"\btee\b"), re.compile(r"\bg?sed\b"),
                     re.compile(r"\b(?:perl|ruby)\b"), re.compile(r"\bg?awk\b"),
                     re.compile(r"\b(?:cp|mv|rsync)\b"))
_CLAUSE = re.compile(r"[^;&|]+")


def write_matches(pattern, start, command):
    """`pattern.finditer(command)` in linear time: per clause, matched from the first
    `start` word, then from the next one after each hit, never after a miss."""
    if start is None:
        yield from pattern.finditer(command)
        return
    for clause in _CLAUSE.finditer(command):
        cur, end = clause.start(), clause.end()
        while True:
            s0 = start.search(command, cur, end)
            if not s0:
                break
            m = pattern.match(command, s0.start(), end)
            if not m:
                break
            yield m
            cur = m.end()


# Destination computed at run time, so it cannot be read off the command text.
# Warn rather than block: a false block here costs more than a missed one.
BASH_WRITE_OPAQUE = re.compile(
    r"write_text\(|open\([^)]*['\"][wa]['\"]|\.writelines\(|shutil\.(?:copy|move)")
# Commands known to leave a file argument unwritten, unless an in-place flag
# (_INPLACE) or a writing git subcommand (_GIT_WRITES) says otherwise. A corpus path handed to any
# other command warns. ⚠ BASH_WRITE alone is a closed list of writers, and
# `perl -pi` passed it silently for a week of real edits; listing the readers
# instead makes the next unlisted writer announce itself, at the cost of a
# warning on an unlisted reader.
BASH_READERS = frozenset((
    "cat head tail less more bat glow nl grep egrep fgrep rg ag ack wc diff cmp comm "
    "ls stat file find du tree test [ realpath readlink basename dirname md5 md5sum shasum "
    "sha256sum sort uniq cut column fold jq yq strings xxd od hexdump echo printf open "
    "git gh locket memscan memfind sed gsed awk gawk rm trash touch mkdir chmod "
    # directional: BASH_WRITE reads which argument is the destination
    "cp mv rsync tee").split())
# Flags whose value a command only reads, keyed by the command they belong to: the file
# after one of these is an input whatever else the command writes. Scoped per command,
# never a blanket reader entry, because the same CLI has writing flags (`claude
# --debug-file`). `ollama launch <cmd>` is judged as `<cmd>`.
_READ_FLAGS = {"claude": frozenset(("--append-system-prompt-file", "--system-prompt-file"))}
# Words that run the command after them; the word after is the one judged.
_SHELL_PREFIXES = frozenset("sudo env command nohup time exec builtin nice xargs caffeinate "
                            "timeout stdbuf ionice setsid "
                            "do then else if while until ! {".split())


def _expand_home(tok):
    """A shell path with its home spelling expanded: `~`, `$HOME` and `${HOME}`,
    the forms a command writes a memory path in most often, which a literal
    `startswith("/")` test read as relative and resolved to nothing."""
    tok = tok.replace('"', "").replace("'", "")     # `"$HOME/dir"/f.md` quotes mid-token
    for pre in ("${HOME}", "$HOME"):
        if tok == pre or tok.startswith(pre + "/"):
            tok = str(HOME) + tok[len(pre):]
    # os.path, not Path: an unknown `~user` stays literal, as the shell leaves it,
    # where Path.expanduser raises and silently disables the guard for the command
    return os.path.expanduser(tok) if tok.startswith("~") else tok


def _shell_code(command):
    """`command` with quoted text, heredoc bodies and comments blanked to spaces, every
    offset kept, so a scan for shell syntax sees only what the shell runs. Quote
    characters themselves stay, so a quoted argument still reads as one."""
    out, i, n = list(command), 0, len(command)
    pending = []                            # heredoc delimiters opened on the current line
    while i < n:
        c = command[i]
        if c in "'\"":
            j = i + 1
            while j < n and command[j] != c:
                j += 2 if c == '"' and command[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                if command[k] != "\n":
                    out[k] = " "
            i = j + 1
            continue
        if c == "#" and (i == 0 or command[i - 1] in " \t\n;&|("):
            while i < n and command[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if c == "<" and command.startswith("<<", i) and not command.startswith("<<<", i):
            m = HEREDOC.match(command, i)
            if m:
                pending.append((m.group(1), command.startswith("<<-", i)))
                i = m.end()
                continue
        if c == "\n" and pending:
            i += 1
            for word, dash in pending:      # blank each body up to its delimiter line
                while i < n:
                    end = command.find("\n", i)
                    end = n if end < 0 else end
                    line = command[i:end]
                    if (line.lstrip("\t") if dash else line) == word:
                        i = end
                        break
                    for k in range(i, end):
                        out[k] = " "
                    i = end + 1
            pending = []
            continue
        i += 1
    return "".join(out)


# `cd` or `pushd` in command position only; the argument runs to the next separator,
# so `"$(mktemp -d)"` is one argument
_CD = re.compile(r"(?:^|(?<=[;&|({]))[ \t]*(cd|pushd|popd)(?=[ \t;&|)\n]|$)(?:[ \t]+([^;&|\n)]*?))?[ \t]*(?=$|[;&|\n)])",
                 re.M)


def _cd_targets(command, cwd, pos=None):
    """The directory a bare filename at `pos` resolves against, as a one-item list, or []
    when it cannot be known.

    The shell resolves against the last `cd` before the name, so that is the one
    taken: resolving against the session cwd first read `cd /elsewhere && sed -i … x.md`
    as a write to the session's own `x.md`. A `cd` whose target is computed at run time
    (`cd "$(mktemp -d)"`) makes every later bare name unknowable, and an unknowable
    target is not claimed as a corpus write. The scan runs over `_shell_code`: a
    `cd /elsewhere` inside a string, a comment or a heredoc body moved nothing, and
    reading it as a move let a real write into the cwd's corpus pass silently."""
    times, dirs = _cd_timeline(command, cwd)
    if pos is None:
        here = dirs[-1]
    else:
        import bisect
        here = dirs[bisect.bisect_right(times, pos - 1)]   # the last cd starting before pos
    return [here] if here else []


@lru_cache(maxsize=8)
def _cd_timeline(command, cwd):
    """(starts, dirs): `dirs[0]` is the cwd and `dirs[k]` the directory after the k-th
    `cd`, which starts at `starts[k - 1]`. Built once per command, because resolving
    each path by rescanning the whole command made a long one quadratic."""
    here, times, dirs = cwd, [], [cwd]
    for m in _CD.finditer(_shell_code(command)):
        if m.group(1) == "popd":
            here = None                          # the stack is not tracked
        else:
            d = command[m.start(2):m.end(2)].strip() if m.group(2) is not None else ""
            d = d or None
            if d is None or d == "~":
                here = str(HOME)
            elif any(c in d for c in "$`(") and not d.strip("'\"").startswith(("$HOME", "${HOME}")):
                here = None
            elif d == "-":
                here = None                      # the previous directory; not tracked
            else:
                d = _expand_home(d)
                here = d if d.startswith("/") else (str(Path(here) / d) if here else None)
            if here and len(here) > PATH_MAX:
                here = None                      # no such directory: `cd x; ` repeated, not a path
        times.append(m.start())
        dirs.append(here)
    return times, dirs


# An in-place flag read off the tokens, so a `;` inside the program text (`sed -i
# 's/a/b/; s/c/d/'`, which stops BASH_WRITE's clause scan) cannot hide it.
# Only flags that take no argument may precede the `i`: `-fimport.sed` and `-ne'print'`
# glue an argument on, and the `i` inside it is not the flag.
_INPLACE = {"sed": r"-[nEsuzr]*i.*|--in-place.*", "gsed": r"-[nEsuzr]*i.*|--in-place.*",
            "perl": r"-[0-9lnpaWwsSTtUuXc]*i.*", "ruby": r"-[0-9lnpaWwsSTtUuXc]*i.*"}
# git subcommands that write the paths they are handed; the rest of git reads
_GIT_WRITES = frozenset("checkout restore mv rm reset stash apply am merge rebase cherry-pick "
                        "revert pull switch clean checkout-index read-tree sparse-checkout "
                        "submodule worktree bisect".split())
_GIT_VALUED = frozenset(("-C", "-c", "--git-dir", "--work-tree", "--namespace"))   # take the next word


def _unlisted_commands(command):
    """[(command word, `.md` argument, offset, in_place)] for every simple command whose
    word is not in BASH_READERS, or that runs with an in-place flag, or is a git
    subcommand that writes. `in_place` marks a write decidable from the tokens. Heredoc bodies are data and are skipped, and so are a
    quoted string holding whitespace (prose, a program), a glob, a redirect target
    (BASH_WRITE reads those), a comment, and a `for`/`case` header. A command the
    tokenizer cannot split (an unbalanced quote) yields nothing: that line of shell
    would not run either."""
    text = _strip_heredocs(command).replace("\\\n", " ").replace("\n", " ; ")
    text = text.replace("\\(", " ").replace("\\)", " ")    # `find … \( -o \)` groups, not subshells
    text = text.replace("`", " ; ")                       # a backtick substitution is its own command
    lex = shlex.shlex(text, posix=True, punctuation_chars=True)
    lex.whitespace_split, lex.commenters = True, ""      # `a#b` is a word, `#` alone a comment
    try:
        toks = list(lex)
    except ValueError:
        return []
    out, word, skip, cursor, redirect, inplace, prev = [], None, False, 0, False, False, ""
    for t in toks:
        # `<(` and `$(` arrive fused: a process or command substitution opens a new command
        # only a token of shell punctuation: a quoted pattern like "(a|b)" is an argument
        # a bare `<`/`>` is a redirect, below; `<(` and `>(` arrive fused and open a command
        if t and set(t) <= set("();|&<>$") and ("<" not in t and ">" not in t or "(" in t or ")" in t):
            word, skip, redirect, inplace = None, False, False, False
            continue
        if t in ("-exec", "-execdir", "-ok", "-okdir"):    # find runs the command after it
            word, inplace = None, False
            continue
        if skip:
            continue
        if t.startswith("#"):
            skip = True                                  # to the end of the line
            continue
        if set(t) <= set("<>&0123456789") and ("<" in t or ">" in t):
            redirect = True
            continue
        if redirect:
            redirect = False
            continue
        if word is None:
            if (t in _SHELL_PREFIXES or "=" in t.split("/")[0] or t.startswith("-")
                    or re.fullmatch(r"\d+[smhd]?", t)):         # `timeout 5`, `nice -n 5`
                continue
            if t in ("for", "case", "select", "in", "function"):
                skip = True
                continue
            word, prev = os.path.basename(t), ""
            continue
        if word in _INPLACE and re.fullmatch(_INPLACE[word], t):
            inplace = True
        elif word in ("awk", "gawk") and (t in ("-iinplace", "--include=inplace")
                                          or (prev in ("-i", "--include") and t == "inplace")):
            inplace = True
        elif word == "git" and not t.startswith("-") and prev not in _GIT_VALUED:
            word = f"git {t}"                            # judged by its subcommand
        elif word == "ollama" and t == "launch":
            word = None                                  # the launched command is the one judged
            continue
        flag, prev = prev, t
        if not t.endswith(".md") or any(c in t for c in "*?[ \t") or "://" in t:
            continue
        reads = _READ_FLAGS.get(word, ())
        if flag in reads or t.partition("=")[0] in reads:
            continue                                     # the value of a read-only flag
        reader = word in BASH_READERS or (word.startswith("git ") and word[4:] not in _GIT_WRITES)
        if reader and not inplace:
            continue
        at = command.find(os.path.basename(t), cursor)
        cursor = at if at >= 0 else cursor
        out.append((word, t, cursor, inplace))
    return out


# A Python program the command runs: `python3 - <<'EOF'` bodies and `python3 -c '…'`.
# Its write targets are read off the program rather than off every `.md` token in it:
# a name inside `.replace("a.md", "b.md")` is data, and `FD / 'X.md'` puts the
# directory in another token, so a token scan both over-fired (two CMP sessions'
# routine edits) and could not see the directory that decides membership.
_PY_WORD = re.compile(r"(?<![\w.-])python[\d.]*(?![\w.-])")
_PY_HEREDOC = re.compile(r"""<<(-?)\s*(['"]?)(\w+)\2""")
_PY_DASH_C = re.compile(r"""(?<![\w.-])python[\d.]*(?:\s+-[A-Za-z]+)*?\s+-c\s+(?:'([^']*)'|"((?:[^"\\]|\\.)*)")""")
_HOLE = "\0"                                     # an f-string value that does not fold
_FOLD_STEPS = 10_000                              # fold calls per program, the whole run
_FOLD_CAP = 64                                   # candidate strings per expression

# The program counts as read only when EVERY call in it is one of these or a known
# writer below; any other call (subprocess, an alias, getattr, an unknown module)
# leaves it unread and the token scan of old answers. ⚠ The first cut listed writers
# only, and an audit found six silent passes through calls it had not listed; this
# is BASH_READERS' inversion, so the next unlisted writer announces itself.
_PY_PURE_BUILTINS = frozenset(
    "len print sorted reversed list dict set frozenset tuple str int float bool bytes "
    "enumerate zip range any all min max sum abs round isinstance issubclass repr format "
    "map filter iter next hasattr type id hash chr ord divmod super".split())
_PY_PURE_MODULES = frozenset(
    "re json textwrap collections itertools functools math string difflib hashlib "
    "fnmatch shlex unicodedata datetime time statistics operator copy pprint typing "
    "dataclasses enum csv glob tomllib yaml os.path".split())
_PY_PURE_FUNCS = frozenset(
    "os.getcwd os.listdir os.walk os.getenv os.environ.get os.scandir os.stat os.remove "
    "os.unlink os.makedirs os.mkdir sys.exit pathlib.Path pathlib.PurePath "
    "pathlib.PosixPath pathlib.Path.home pathlib.Path.cwd Path PurePath".split())
# methods on a value (a string, a list, a path, a match, a handle) that write nothing
# outside a handle; a handle's own destination is read where `open` made it. Deleting
# (`unlink`, `os.remove`) is not writing here, as `rm` is a reader to BASH_READERS.
_PY_PURE_METHODS = frozenset(
    {m for m in dir(str) if not m.startswith("_")} | set(
        "append extend insert pop remove clear sort reverse index count get items keys "
        "values update setdefault add discard union intersection difference copy "
        "read_text read_bytes read readline readlines exists is_file is_dir glob rglob "
        "iterdir stat resolve expanduser absolute with_suffix with_name with_stem "
        "joinpath relative_to as_posix samefile group groups groupdict start end span "
        "sub subn search match fullmatch findall finditer write writelines close flush seek "
        "tell home cwd decode mkdir unlink is_symlink".split()))
# writers: canonical name -> index of the destination argument
_PY_WRITE_FUNCS = {"open": 0, "io.open": 0, "codecs.open": 0, "os.open": 0,
                   "shutil.copy": 1, "shutil.copy2": 1, "shutil.copyfile": 1,
                   "shutil.move": 1, "shutil.copytree": 1, "os.rename": 1,
                   "os.replace": 1, "os.link": 1, "os.symlink": 1}
_PY_ESCAPES = frozenset(("getattr", "eval", "exec", "compile", "__import__", "os.system",
                         "os.popen", "pty.spawn"))
_PY_INTO_DIR = frozenset(("shutil.copy", "shutil.copy2", "shutil.move", "shutil.copytree"))
_PY_WRITE_METHODS = frozenset(("write_text", "write_bytes", "touch", "symlink_to", "hardlink_to"))
_PY_PATH_FUNCS = frozenset(("pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath",
                            "os.path.join", "Path"))
_PY_SAME_FUNCS = frozenset(("str", "os.fspath", "os.path.expanduser", "os.path.abspath",
                            "os.path.realpath", "os.path.normpath"))


def _python_programs(command):
    """[(source, start, end, unreadable)] for each Python program in `command`;
    offsets span the program text. `unreadable` marks a body whose literals are not
    the program's: one the shell rewrites first (an unquoted heredoc or a
    double-quoted `-c` holding `$` or a backtick), or a `-c` body the shell splices
    on with `'\\''`, which the match sees only the first piece of."""
    out, pos, lines = [], 0, command.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        pos += len(line) + 1
        i += 1
        m = _PY_HEREDOC.search(line)
        if not (m and _PY_WORD.search(line[:m.start()])):
            continue
        body_start, body = pos, []
        while i < len(lines):
            ln = lines[i]
            i += 1
            pos += len(ln) + 1
            if (ln.lstrip("\t") if m.group(1) else ln) == m.group(3):
                break
            body.append(ln)
        src = "\n".join(body)
        out.append((src, body_start, body_start + len(src),
                    not m.group(2) and any(c in src for c in "$`")))
    for m in _PY_DASH_C.finditer(command):
        g = 1 if m.group(1) is not None else 2
        src = m.group(g)
        spliced = command[m.end():m.end() + 1] not in ("", " ", "\t", "\n", ";", "&", "|", ")")
        out.append((src, m.start(g), m.end(g), spliced or (g == 2 and any(c in src for c in "$`"))))
    return out


class _PyWrites:
    """What a parsed Python program writes: `targets`, the folded destination of every
    writing call (`_HOLE` where an f-string value does not fold); `writes`, whether it
    makes any such call; `decided`, whether every destination folded and every call
    is known. Relative targets are joined to each folded `os.chdir` base as well as
    returned bare, since the walk is flow-insensitive.

    A name's values are the union of every binding it gets anywhere in the program;
    any binding that does not fold (a parameter, a call result, an `except … as`, a
    tuple target) leaves the name unknown. Membership turns on the directory and the
    file pattern, so a target whose directory folds is decided even with a hole in
    its file name, and one with a hole in its directory is not."""

    def __init__(self, tree):
        self.steps = 0                      # fold calls this run, against _FOLD_STEPS
        self.binds, self.unknown, self.aliases, self.defs = {}, set(), {}, set()
        self.targets, self.writes, self.decided, self.bases = [], False, True, []
        accepted = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    self._bind(t, n.value, accepted)
            elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and n.value is not None:
                self._bind(n.target, n.value, accepted)
            elif isinstance(n, (ast.For, ast.comprehension)) and isinstance(
                    n.iter, (ast.List, ast.Tuple, ast.Set)) and isinstance(n.target, ast.Name):
                for e in n.iter.elts:
                    self._bind(n.target, e, accepted)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs.add(n.name)
            elif isinstance(n, ast.Import):
                for al in n.names:                        # `import a.b` binds `a`; `as x` binds all of it
                    if al.asname:
                        self.aliases[al.asname] = al.name
                    else:
                        self.aliases[al.name.split(".")[0]] = al.name.split(".")[0]
            elif isinstance(n, ast.ImportFrom) and n.module:
                for al in n.names:
                    self.aliases[al.asname or al.name] = f"{n.module}.{al.name}"
        # every other binding site leaves its name unknown, whatever construct made it
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load) and id(n) not in accepted:
                self.unknown.add(n.id)
            elif isinstance(n, ast.arg):
                self.unknown.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                self.unknown.add(n.name)
            for attr in ("name", "rest"):                 # match captures (3.10+)
                if type(n).__name__ in ("MatchAs", "MatchStar", "MatchMapping") and getattr(n, attr, None):
                    self.unknown.add(getattr(n, attr))
        self._calls(tree)

    def _bind(self, target, value, accepted):
        if isinstance(target, ast.Name):
            self.binds.setdefault(target.id, []).append(value)
            accepted.add(id(target))

    def canon(self, f):
        """The dotted name a callee resolves to through imports (`sh.copy` ->
        `shutil.copy`), `.method` for a method on a value, or None when the callee
        is itself computed."""
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if not isinstance(f, ast.Name):
            return "." + parts[0] if parts else None
        root = f.id
        if root in self.aliases:
            root = self.aliases[root]
        elif parts:
            return "." + parts[0]                         # a method on a variable
        return ".".join([root] + list(reversed(parts)))

    def fold(self, e, seen=()):
        """Set of strings `e` can be, or None when it cannot be known. A name bound
        to itself twice (`x1 = x0 + x0`) doubles the walk per line, so the run stops
        at _FOLD_STEPS and the rest is unknown. No memo: a fold's answer depends on
        `seen`, so one cached per node would answer for the wrong path."""
        self.steps += 1
        if self.steps > _FOLD_STEPS:
            return None
        if isinstance(e, ast.Constant):
            return {e.value} if isinstance(e.value, str) else None
        if isinstance(e, ast.JoinedStr):
            parts = [{str(v.value)} if isinstance(v, ast.Constant) else (self.fold(v.value, seen) or {_HOLE})
                     for v in e.values]
            return _cross(parts, lambda a, b: a + b) if parts else {""}
        if isinstance(e, ast.Name):
            if e.id in self.unknown or e.id not in self.binds or e.id in seen:
                return None
            out = set()
            for v in self.binds[e.id]:
                f = self.fold(v, seen + (e.id,))
                if f is None:
                    return None
                out |= f
            return out if len(out) <= _FOLD_CAP else None
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Div, ast.Add)):
            return _cross([self.fold(e.left, seen), self.fold(e.right, seen)],
                          _join if isinstance(e.op, ast.Div) else (lambda a, b: a + b))
        if isinstance(e, ast.Attribute) and e.attr == "parent":
            v = self.fold(e.value, seen)
            return None if v is None else {os.path.dirname(x.rstrip("/")) for x in v}
        if isinstance(e, ast.Call) and not e.keywords:
            name = self.canon(e.func) or ""
            if name in _PY_PATH_FUNCS:
                return _cross([self.fold(a, seen) for a in e.args], _join) if e.args else {"."}
            if name == "pathlib.Path.home" or (name == ".home" and not e.args):
                return {"~"}
            if name in _PY_SAME_FUNCS and len(e.args) == 1:
                return self.fold(e.args[0], seen)
            if name in (".expanduser", ".resolve", ".absolute") and not e.args:
                return self.fold(e.func.value, seen)
        return None

    def _calls(self, tree):
        callees = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
        for n in ast.walk(tree):
            # a writer handed on as a value (`o = open`, `map(open, …)`) is called unseen
            if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Load) \
                    and id(n) not in callees:
                name = self.canon(n) if isinstance(n, ast.Attribute) else self.aliases.get(n.id, n.id)
                if name in _PY_WRITE_FUNCS or (name or "").lstrip(".") in _PY_WRITE_METHODS:
                    self.decided, self.writes = False, True
            if not isinstance(n, ast.Call):
                continue
            name = self.canon(n.func)
            dest = None
            if name is None:
                self._unknown(n, n.func)                  # `x[0]()`, `f()()`
                continue
            if isinstance(n.func, ast.Name) and n.func.id in self.defs:
                continue                                  # its body is walked on its own
            if name in _PY_ESCAPES or name.startswith(("os.exec", "os.spawn", "subprocess.")):
                self._unknown(n)                          # a shell or dynamic call: write if it names one
                continue
            if isinstance(n.func, ast.Name) and (n.func.id in self.binds or n.func.id in self.unknown) \
                    and n.func.id not in self.aliases:
                self._unknown(n)                          # a variable called: unknown callee
                continue
            if name in ("os.chdir",):
                v = self.fold(n.args[0]) if n.args else None
                if v is None or any(_HOLE in x for x in v):
                    self.decided = False
                else:
                    self.bases.extend(v)
                continue
            method = name[1:] if name.startswith(".") else None
            if method in _PY_WRITE_METHODS:
                dest = n.func.value
            elif method in ("rename", "replace") and len(n.args) + len(n.keywords) == 1 \
                    and (n.args or n.keywords[0].arg == "target"):
                dest = self._arg(n, 0, "target")          # Path.rename(dst); str.replace takes two
            elif method == "open" or name in ("open", "io.open", "codecs.open"):
                builtin = method is None
                path = (self._arg(n, 0, "file") or self._arg(n, 0, "filename")) if builtin else n.func.value
                mode = self._arg(n, 1 if builtin else 0, "mode")
                m = {"r"} if mode is None else self.fold(mode)
                if m is None or any(set(x) & set("wax+") for x in m):
                    dest = path
            elif name in _PY_WRITE_FUNCS:
                dest = self._arg(n, _PY_WRITE_FUNCS[name], "dst" if _PY_WRITE_FUNCS[name] else "path")
                if name in _PY_INTO_DIR:                  # `copy(src, dir)` lands at dir/basename(src)
                    self._into_dir(dest, self._arg(n, 0, "src"), tree=name.endswith("copytree"))
            elif (method == "copy" and len(n.args) >= 2) or not (
                    method in _PY_PURE_METHODS or name in _PY_PURE_BUILTINS
                    or name in _PY_PURE_FUNCS or ("." in name.lstrip(".") and name.rsplit(".", 1)[0] in _PY_PURE_MODULES)):
                self._unknown(n)                          # `x.copy(a, b)` is not dict.copy()
                continue
            if dest is None:
                continue
            self.writes = True
            v = self.fold(dest)
            if v is None:
                self.decided = False
                continue
            for x in v:
                head = x.rsplit("/", 1)[0] if "/" in x else ""
                if x.startswith(_HOLE) or _HOLE in head:
                    self.decided = False                  # the directory itself is unknown
                else:
                    self.targets.append(x)

    def _mdish(self, e, seen=()):
        """Whether expression `e` can evaluate to a `.md` path: a literal or f-string
        ending in `.md`, a join or concatenation ending in one, a path constructor over
        one, a collection or subscript holding one, or a name any binding of which can.
        A call's RESULT is not its argument: `Path('MEMORY.md').read_text()` is the
        file's text, and reading it as a path re-fired the routine transform."""
        if isinstance(e, ast.Constant):
            return isinstance(e.value, str) and e.value.rstrip().endswith(".md")
        if isinstance(e, ast.JoinedStr):
            return bool(e.values) and self._mdish(e.values[-1], seen)
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Div)):
            return self._mdish(e.right, seen)
        if isinstance(e, (ast.List, ast.Tuple, ast.Set)):
            return any(self._mdish(x, seen) for x in e.elts)
        if isinstance(e, ast.Dict):
            return any(self._mdish(x, seen) for x in e.values if x is not None)
        if isinstance(e, ast.Subscript):
            return self._mdish(e.value, seen)
        if isinstance(e, ast.Starred):
            return self._mdish(e.value, seen)
        if isinstance(e, ast.Name):
            return e.id in self.binds and e.id not in seen and any(
                self._mdish(v, seen + (e.id,)) for v in self.binds[e.id])
        if isinstance(e, ast.Call):
            name = self.canon(e.func) or ""
            if name in _PY_PATH_FUNCS or name in _PY_SAME_FUNCS:
                return bool(e.args) and self._mdish(e.args[-1], seen)
            if name in (".expanduser", ".resolve", ".absolute", ".with_name", ".with_suffix"):
                return True if name == ".with_suffix" and e.args and self._mdish(e.args[0], seen) \
                    else self._mdish(e.func.value, seen)
        return False

    def _unknown(self, call, *extra):
        """An unknown call: a write only where its arguments (or receiver) reach a `.md`
        string. One that never does is neutral whatever its name, so `datetime.now()`,
        `sys.stdout.write` and `argparse` stay quiet while `mylib.save('MEMORY.md')`,
        `os.truncate('MEMORY.md', 0)` and `fileinput.input(..., inplace=True)` warn.
        A write computed with no `.md` text in sight is unseen here and at the token
        scan alike."""
        parts = [*call.args, *(k.value for k in call.keywords), *extra]
        if isinstance(call.func, ast.Attribute):
            parts.append(call.func.value)
        if any(self._mdish(p) for p in parts):
            self.decided, self.writes = False, True

    def _into_dir(self, dst, src, tree=False):
        """Targets of a copy or move whose destination may be a directory."""
        d, s = (self.fold(dst) if dst is not None else None), (self.fold(src) if src is not None else None)
        if d is None:
            return
        for x in d:
            if x.endswith(".md") or _HOLE in x:
                continue                                  # a file destination, or a directory unknown
            if tree:
                self.targets.append(_join(x, _HOLE + ".md"))
            elif s is None:
                self.targets.append(_join(x, _HOLE + ".md"))
            else:
                self.targets.extend(_join(x, os.path.basename(y)) for y in s)

    @staticmethod
    def _arg(call, i, kw):
        for k in call.keywords:
            if k.arg == kw:
                return k.value
        return call.args[i] if len(call.args) > i else None


def _cross(sets, op):
    """Every `op`-combination of one string from each set, or None past the cap."""
    if any(s is None for s in sets):
        return None
    out = set(sets[0])
    for s in sets[1:]:
        out = {op(x, y) for x in out for y in s}
        if len(out) > _FOLD_CAP:
            return None
    return out


def _join(a, b):
    return b if b.startswith(("/", "~")) else (a.rstrip("/") + "/" + b if a else b)


def _python_writes(source):
    """`_PyWrites` for a program, or None when it does not parse or is too deep to
    walk (a RecursionError from a long `+` chain is undecided, never a crash)."""
    try:
        return _PyWrites(ast.parse(source))
    except (SyntaxError, ValueError, RecursionError):
        return None


def bash_write_targets(command, cwd):
    """(blocking, opaque, unlisted) corpus files a Bash command appears to write;
    `unlisted` pairs each with the command word outside BASH_READERS that takes it.

    Resolves a bare `MEMORY.md` against the payload cwd, or the last `cd` before it
    in the command, because that is the shape a heredoc rewrite actually takes.
    Resolving rather than name-matching is why the same command is a corpus write
    from inside a corpus and nothing at all from outside one.

    A path the shell expands at run time (any variable but HOME, a substitution)
    cannot be decided from the command, so it never blocks: it is expanded against
    this process's environment and, if that lands in a corpus, warns. Read
    literally, `"$TMPDIR/x.md"` is relative and joins the cwd, which would block
    every temp-file heredoc written from inside a corpus.
    """
    def resolve(tok, pos):
        """(path, decidable) for the corpus file `tok` names, or None."""
        tok = _expand_home(tok)
        decidable = not any(c in tok for c in "$`")
        if not decidable:
            tok = os.path.expandvars(tok)
            if any(c in tok for c in "$`"):          # unset, or a substitution
                return None
        cands = ([tok] if tok.startswith("/")
                 else [str(Path(r) / tok) for r in _cd_targets(command, cwd, pos)])
        for c in cands:
            if corpus_for(c) is not None:
                return Path(c), decidable
        return None

    blocking, opaque = [], []
    for pat, start in zip(BASH_WRITE, BASH_WRITE_STARTS):
        for m in write_matches(pat, start, command):
            hit = resolve(m.group("p"), m.start("p"))
            if hit:
                (blocking if hit[1] else opaque).append(hit[0])

    def flag(hit):
        if hit and hit[0] not in blocking and hit[0] not in opaque:
            opaque.append(hit[0])
    # A Python program that parses is read on its own, gate or no gate, so a writer the
    # gate's regex does not spell (`write_bytes`, mode `'wb'`) is still seen. Where every
    # destination folds and every call is known, its folded targets are the answer;
    # otherwise it gets the token scan below, as does all text outside a read program.
    rest = list(command)
    for src, start, end, unreadable in _python_programs(command):
        w = None if unreadable else _python_writes(src)
        if w is None:
            continue                            # unread: its tokens stay in `rest`
        rest[start:end] = " " * (end - start)
        for t in w.targets:
            # a hole at the end may still be `.md`; inside the name it stands for any text
            probe = (t + ".md" if t.endswith(_HOLE) else t).replace(_HOLE, "*")
            if not probe.endswith(".md"):
                continue
            flag(resolve(probe, start))
            if not probe.startswith(("/", "~")):
                for base in w.bases:            # after an `os.chdir` the program's cwd is that
                    flag(resolve(_join(base, probe), start))
        if not w.decided and (w.writes or BASH_WRITE_OPAQUE.search(src)):
            for m in re.finditer(r"[\w./~-]+\.md", src):
                # `f'{d}/MEMORY.md'` scans as `/MEMORY.md`: an unknown directory may be the cwd's
                for tok in {m.group(), os.path.basename(m.group())}:
                    flag(resolve(tok, start + m.start()))
                    for base in w.bases:
                        flag(resolve(_join(base, tok), start + m.start()))
    rest = "".join(rest)
    if BASH_WRITE_OPAQUE.search(rest):
        for m in re.finditer(r"[\w./~-]+\.md", rest):
            flag(resolve(m.group(), m.start()))
    unlisted = []
    for word, tok, pos, inplace in _unlisted_commands(command):
        hit = resolve(tok, pos)
        if not hit or hit[0] in blocking:
            continue
        if inplace and hit[1]:                  # a literal path, edited in place: decidable
            blocking.append(hit[0])
        elif hit[0] not in opaque:
            unlisted.append((word, hit[0]))
    return blocking, opaque, unlisted


def bash_write_decision(command, cwd, autonomous=False):
    """(message, blocking) for a Bash command touching a corpus, or (None, False).

    Exempts what 2d exempts, plus the scratchpad and autonomous runs, and honours a
    declared MEMWRITE-OK escape the way the dossier hook honours NOT-CURATION: a
    reword records nothing, a declaration records the decision.
    """
    if ".md" not in command or autonomous or MEMWRITE_ESCAPE in command:
        return None, False
    if any(mark in command for mark in SCRATCH_MARKERS):
        return None, False
    blocking, opaque, unlisted = bash_write_targets(command, cwd)

    def live(q):                             # exempt by ITS corpus's names, not ours
        sets = settings(corpus_for(q))
        return q.name not in sets["append_surfaces"] and "archive" not in q.parts
    blocking = [q for q in blocking if live(q)]
    opaque = [q for q in opaque if live(q)]
    unlisted = [(w, q) for w, q in unlisted if live(q)]
    if blocking:
        names = ", ".join(sorted({q.name for q in blocking}))
        return (f"This writes to a memory corpus through Bash ({names}), where the write-time "
                f"gate cannot see it. That gate matches the Write and Edit tools, so a shell "
                f"redirect skips the 2c one-authority check and the 2e already-has-a-home "
                f"question, silently.\n"
                f"  - Use Write or Edit for memory files, even where a session directive "
                f"prefers Bash. Memory corpora are the carve-out.\n"
                f"  - A ledger, a holding space, anything under archive/ and the scratchpad "
                f"are exempt and never reach this message.\n"
                f"  - If Bash is genuinely right here (an atomic multi-file patch, a scripted "
                f"migration), put `{MEMWRITE_ESCAPE} <reason>` in the command and it passes."), True
    if opaque:
        names = ", ".join(sorted({q.name for q in opaque}))
        return (f"This looks like a scripted write into a memory corpus ({names}). The "
                f"write-time gate matches Write and Edit, so if it does write there the 2c and "
                f"2e checks will not fire and nothing will report that they did not. Route "
                f"the edit through Write/Edit, or put `{MEMWRITE_ESCAPE} <reason>` in the "
                f"command if Bash is genuinely right. Running `memfind` first does not "
                f"substitute: it answers 2e and leaves the 2c check unrun."), False
    if unlisted:
        what = ", ".join(sorted({f"`{w}` on {q.name}" for w, q in unlisted}))
        return (f"This hands a memory file to a command Locket does not know to be read-only "
                f"({what}). If it writes the file, the write-time gate never sees it, because "
                f"that gate matches Write and Edit. Route the edit through Write/Edit, or put "
                f"`{MEMWRITE_ESCAPE} <reason>` in the command if Bash is genuinely right; a "
                f"read-only command can be ignored."), False
    return None, False


# A regex handed to an embedding ranker is noise, so the injector fires only on a
# pattern that reads as a question: two content words, no metacharacters. Words are
# separated by whitespace only: a token joined by a hyphen, dot, underscore or slash
# (`angle-scout`, `settings.json`, `spec_provenance`) is one identifier, never a phrase.
GREP_META = set("|[]()\\^$*+?{}")


def prose_like(pattern):
    if not pattern or GREP_META & set(pattern):
        return False
    words = (w.strip(",;:.!'\"") for w in pattern.split())
    return len([w for w in words if re.fullmatch(r"[A-Za-z0-9]{4,}", w)]) >= 2


GREP_VERBS = {"grep", "egrep", "rg", "ugrep"}


def grep_words(cmd, cwd=None, _nested=False):
    """(pattern, path words) for the grep in a shell command, or None when it runs none.

    Only a grep's own words count: the command is split at `;`, `&&`, `|` and redirects
    first, and every word naming a grep verb is tried until one yields a pattern, so `which
    grep; grep ...` still finds the second. A command with no verb is searched once inside
    the command string a shell takes after `-c` (`bash -c '...'`), never in any other
    quoted word, where a commit message or a heredoc that mentions grep would fire it.
    The pattern is the `-e`/`--regexp` value where one is given, otherwise the first word
    after the verb that is not a path, a bare count or a non-question value of a bare long
    option, and it counts only when shaped like a question (`prose_like`); no per-tool list
    of value-taking flags is needed. The path words are every
    other word after an `-e` pattern, else the words after the pattern, last first, so a
    value-taking flag between pattern and path cannot shadow the path."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        toks = list(lex)
    except ValueError:                      # unbalanced quotes: words are still words
        toks = cmd.split()
    verbs = [k for k, t in enumerate(toks) if os.path.basename(t) in GREP_VERBS]
    if not verbs:
        if not _nested:
            for flag, t in zip(toks, toks[1:]):
                if re.fullmatch(r"-[A-Za-z]*c", flag):
                    got = grep_words(t, cwd, True)
                    if got:
                        return got
        return None

    def pathlike(w):
        if "/" in w or w.startswith(("~", "$", ".")):
            return True
        return os.path.exists(os.path.join(cwd or os.getcwd(), w))

    for i in verbs:
        seg = []
        for t in toks[i + 1:]:
            if t and not t.strip("();<>|&"):   # an operator ends the grep's words
                if t[0] in "<>" and seg and seg[-1].isdigit():
                    seg.pop()               # the `2` of `2>/dev/null` is a descriptor, not a word
                break
            seg.append(t)
        flagged, positional, opts, it, after_opt = None, [], True, iter(seg), False
        for w in it:
            # whether the word before this one was an option that could take it as a value
            follows_opt = after_opt
            after_opt = opts and w.startswith("-") and w not in ("-", "--") and "=" not in w
            attached = re.fullmatch(r"-[A-Za-z]*e(.+)", w, re.S) if opts else None
            if opts and w == "--":
                opts = False
            elif opts and w.startswith("--regexp="):
                flagged = flagged or w.partition("=")[2]
            elif opts and (w == "--regexp" or re.fullmatch(r"-[A-Za-z]*e", w)):
                value = next(it, None)      # consumed even when an earlier -e won
                flagged = flagged or value
            elif attached and not w.startswith("--"):
                flagged = flagged or attached.group(1)
            elif not (opts and w.startswith("-") and w != "-"):
                positional.append((w, follows_opt))
        if flagged:
            return flagged, [w for w, _ in positional][::-1]
        # The pattern is the FIRST word that can be one: a path, a bare count (`-A 12`) and a
        # non-question word right after an option (`-g '*.md'`, `-t py`, `--exclude-dir
        # node_modules`) are flag values or targets. When that first word is not a question
        # the grep has none to rank, so the scan stops there; searching further took a
        # filename (`settings.json`) as the query. ⚠ A regex right after a boolean option
        # (`grep -r 'a|b' 'two words here'`) is skipped as a value, so a later prose-shaped
        # path with spaces would be taken; per-tool flag tables were rejected as the fix.
        at = next((j for j, (w, fl) in enumerate(positional)
                   if not pathlike(w) and not w.isdigit() and not (fl and not prose_like(w))), None)
        if at is not None and prose_like(positional[at][0]):
            return positional[at][0], [w for w, _ in positional[at + 1:]][::-1]
    return "", []


GREPASSIST_LOG_CAP = 1 << 20                # past this the file keeps its newest half


def _trim_log(path, cap):
    """Keep the newest half of `path` by bytes once it passes `cap`. The kept half is
    written beside the log and renamed over it, so a parallel hook never sees a truncated
    file; an append racing the rename lands on the old inode and is lost, which a
    diagnostic log tolerates."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size <= cap:
        return
    with path.open("rb") as fh:
        fh.seek(size // 2)
        fh.readline()                       # drop the partial row the seek landed in
        kept = fh.read()
    _replace_atomic(path, kept, ".grepassist-")


# Hook health: whether a hook run was abandoned or failed open, read from live runs.
# A hook the host kills at its timeout, or one whose catch-all swallows an error, lets
# the call through and says nothing; a doctor probe sees only the runs it makes itself.
# Both surfaces are bounded by construction (RULINGS.csv on Area=substrate): a run's
# marker lives only while the run does, so the directory is empty at rest and capped,
# and the counters are one record per known hook verb, rewritten in place.
HEALTH_VERBS = ("hook", "bashguard", "grepassist")
HEALTH_MARKER_CAP = 256          # past this the oldest count as abandoned: that many live hooks is a fault
HEALTH_STALE_POSIX = 3600        # a hook alive this long is past every timeout (doctor caps them at 600 s)
HEALTH_STALE_OTHER = 600         # no liveness probe off POSIX, so age alone: past any hook timeout
HEALTH_LOCK_WAIT = 0.2           # seconds; a stuck lock holder costs a retry, never a hang
HEALTH_TEMP_AGE = 60             # seconds; a temp or claim left this long belongs to a dead writer


def health_dir():
    """Where hook health lives: `Paths.health_dir`, read at call time."""
    return paths.PATHS.health_dir


def _pid_alive(pid):
    """Whether a process exists. Off POSIX there is no harmless probe (os.kill there
    terminates), so the answer is always yes and a marker folds on age alone."""
    if os.name != "posix":
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError, ValueError):
        return False
    except OSError:                         # exists, owned by someone else
        return True
    return True


def _utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def _replace_atomic(path, data, prefix, mode=None):
    """Write `data` (str or bytes) beside `path` and rename it over, so a reader never sees
    a partial file. The temp is removed when the write fails; a writer killed mid-way can
    still leave one, which each surface's own sweep removes."""
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), prefix=prefix)
    try:
        with os.fdopen(fd, "wb" if isinstance(data, bytes) else "w") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _HealthLock:
    """A non-blocking exclusive lock on `<dir>/.lock`, retried for HEALTH_LOCK_WAIT. `held`
    is False when it could not be taken, and the caller then drops or defers its update:
    a hook waiting here would be killed by its host and feed the record it guards."""

    def __init__(self, d):
        self.d, self.fh, self.held = d, None, False

    def __enter__(self):
        try:
            import fcntl
            take = lambda fh: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            try:
                import msvcrt
            except ImportError:
                return self                 # no lock primitive: the update is dropped, never run unlocked
            def take(fh):
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            self.fh = open(self.d / ".lock", "a+")
        except OSError:
            return self
        deadline = time.monotonic() + HEALTH_LOCK_WAIT
        while True:
            try:
                take(self.fh)
                self.held = True
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    return self
                time.sleep(0.01)

    def __exit__(self, *exc):
        if self.fh is not None:
            self.fh.close()                 # closing releases either lock
        return False


def _valid_record(rec):
    return (isinstance(rec, dict) and isinstance(rec.get("verbs"), dict)
            and all(isinstance(v, dict) for v in rec["verbs"].values()))


def _health_update(change, d=None):
    """Apply `change(record)` to the counters under the lock and replace the file whole.
    True when it landed. Never raises: telemetry must not break the hook it watches. An
    unreadable record is set aside as `health.json.corrupt` (one file, overwritten) and the
    fresh record says so, rather than zeroing the history in silence."""
    d = health_dir() if d is None else Path(d)
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _HealthLock(d) as lock:
            if not lock.held:
                return False
            f = d / "health.json"
            try:
                rec = json.loads(f.read_text())
                if not _valid_record(rec):
                    raise ValueError("not a health record")
            except FileNotFoundError:
                rec = {"since": _utc(), "verbs": {}}
            except Exception:
                try:
                    os.replace(f, d / "health.json.corrupt")
                    note = f"an unreadable record was set aside as health.json.corrupt at {_utc()}"
                except OSError:
                    note = f"an unreadable record was overwritten at {_utc()}"
                rec = {"since": _utc(), "verbs": {}, "reset": note}
            change(rec)
            _replace_atomic(f, json.dumps(rec, indent=1, sort_keys=True), ".health-")
        return True
    except Exception:
        return False


def _count(rec, verb, kind, detail):
    verb = verb if verb in HEALTH_VERBS else "?"     # one record per known verb: bounded
    v = rec["verbs"].setdefault(verb, {"killed": 0, "errors": 0})
    v[kind] = int(v.get(kind, 0)) + 1
    v["last_killed" if kind == "killed" else "last_error"] = dict(detail, at=_utc())


def _marker_name(verb, pid, start_ns):
    """Start time first, so a name sort is a time sort and the cap folds the oldest."""
    return f"{start_ns:020d}-{pid}-{verb}"


def health_error(verb, where, exc):
    """Record an error a hook swallowed to let the call through. When the counters cannot
    be written now, the error is left as a marker of a finished process, which the next
    fold counts, so contention delays the count and never drops it."""
    what = type(exc).__name__
    if _health_update(lambda rec: _count(rec, verb, "errors", {"where": where, "what": what})):
        return
    try:
        marks = health_dir() / "inflight"
        marks.mkdir(mode=0o700, parents=True, exist_ok=True)
        (marks / _marker_name(f"{verb}!error", 0, time.time_ns())).write_text(
            json.dumps({"where": str(where)[:40], "what": what}))
    except Exception:
        pass


def _fold_dead_markers(d=None, now=None):
    """Count every run that ended without removing its marker, the trace of a run the host
    abandoned, and every deferred error, then remove those markers. Counted before removed:
    when the count cannot land, each marker is put back for a later fold, so evidence is
    never destroyed uncounted. Returns how many dead markers remain uncounted.

    A marker is claimed by an atomic rename first, so two folds count it once; a claim whose
    folder died is reclaimed once it has been a claim for HEALTH_TEMP_AGE. Temps a dead
    writer left are removed here too. Past HEALTH_MARKER_CAP markers the oldest fold
    whatever their state, so the directory stays bounded."""
    d = health_dir() if d is None else Path(d)
    marks = d / "inflight"
    now = time.time() if now is None else now
    stale = HEALTH_STALE_POSIX if os.name == "posix" else HEALTH_STALE_OTHER
    try:
        names = sorted(os.listdir(marks))
    except OSError:
        return 0
    plain = [n for n in names if not n.startswith(".")]
    excess = len(plain) - HEALTH_MARKER_CAP
    take = []                               # (original name, verb, age)
    for i, n in enumerate(plain):
        try:
            start_ns, pid, verb = n.split("-", 2)
            age = max(0.0, now - int(start_ns) / 1e9)   # a clock stepped back reads as young
            pid = int(pid)
        except ValueError:
            take.append((n, "?", 0.0)); continue
        if i < excess or verb.endswith("!error") or age > stale or not _pid_alive(pid):
            take.append((n, verb, age))
    for n in names:                         # claims and finished markers a dead process left
        if not n.startswith("."):
            continue
        try:
            old = now - (marks / n).stat().st_mtime > HEALTH_TEMP_AGE
        except OSError:
            continue
        if n.startswith(".claim-") and old and n.count("-") >= 2:
            orig = n.split("-", 2)[2]
            try:
                os.replace(marks / n, marks / orig)     # reclaimable like any marker
                start_ns, _pid, verb = orig.split("-", 2)
                take.append((orig, verb, max(0.0, now - int(start_ns) / 1e9)))
            except (OSError, ValueError):
                pass
        elif n.startswith(".done-") and old:
            try:
                (marks / n).unlink()
            except OSError:
                pass
    claimed = []
    for n, verb, age in take:
        c = marks / f".claim-{os.getpid()}-{n}"
        try:
            os.rename(marks / n, c)
            os.utime(c)                     # a claim's age is time since claimed, not since the run began
        except OSError:
            continue                        # another fold claimed it
        try:
            detail = json.loads(c.read_text())
            detail = detail if isinstance(detail, dict) else {}
        except Exception:
            detail = {}
        claimed.append((c, n, verb, age, detail))
    if claimed:
        def change(rec):
            for _c, _n, verb, age, detail in claimed:
                if verb.endswith("!error"):
                    _count(rec, verb[:-len("!error")], "errors",
                           {"where": str(detail.get("where", "?")), "what": str(detail.get("what", "?"))})
                else:
                    _count(rec, verb, "killed", {"cwd": str(detail.get("cwd", ""))[:200],
                                                 "seconds": round(age, 1)})
        landed = _health_update(change, d)
        for c, n, *_ in claimed:
            try:
                if landed:
                    c.unlink()
                else:
                    os.replace(c, marks / n)
            except OSError:
                pass
        if not landed:
            return len(claimed)
    for t in d.glob(".health-*"):           # temps a writer killed mid-replace left behind
        try:
            if now - t.stat().st_mtime > HEALTH_TEMP_AGE:
                t.unlink()
        except OSError:
            pass
    return 0


class HookRun:
    """`with HookRun(verb):` around one hook invocation: leaves a marker for this run,
    folds the traces of earlier abandoned runs, and removes the marker on the way out. A
    run the host kills never reaches the exit, so its marker is what a later fold counts.
    The marker is written before the fold, so a run killed while folding is still seen."""

    def __init__(self, verb):
        self.verb, self.marker = verb, None

    def __enter__(self):
        try:
            d = health_dir()
            marks = d / "inflight"
            marks.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.listdir(marks)               # unlistable: no fold could ever bound it, so write nothing
            try:
                cwd = os.getcwd()
            except OSError:
                cwd = ""
            m = marks / _marker_name(self.verb, os.getpid(), time.time_ns())
            m.write_text(json.dumps({"cwd": cwd}))
            self.marker = m
            _fold_dead_markers(d)
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        if self.marker is not None:
            try:
                self.marker.unlink()
            except OSError:                 # held open elsewhere: mark it finished, never abandoned
                try:
                    os.replace(self.marker, self.marker.with_name(".done-" + self.marker.name))
                except OSError:
                    pass
        return False


def health_report(d=None, fold=True, kinds=("killed", "errors")):
    """(record, problems): the counters, and one line per verb and kind in `kinds` with a
    count since they were last acknowledged. Folds first, so an abandoned run shows even
    when no hook has run since; a dead marker the fold could not count yet, and a record
    that cannot be read, are problems too, never a clean reading."""
    d = health_dir() if d is None else Path(d)
    pending = _fold_dead_markers(d) if fold else 0
    out = [f"{pending} abandoned run(s) not yet counted (the record was busy)"] if pending else []
    f = d / "health.json"
    try:
        rec = json.loads(f.read_text())
        if not _valid_record(rec):
            raise ValueError("not a health record")
    except FileNotFoundError:
        return None, out
    except Exception:
        return None, out + [f"the hook-health record {f} cannot be read; the next write sets it aside"]
    for verb, v in sorted(rec["verbs"].items()):
        if "killed" in kinds and v.get("killed"):
            lk = v.get("last_killed") or {}
            out.append(f"{verb} was abandoned {v['killed']}x (last {lk.get('at', '?')}, "
                       f"{lk.get('seconds', '?')}s, in {lk.get('cwd') or '?'}); those calls ran unchecked")
        if "errors" in kinds and v.get("errors"):
            le = v.get("last_error") or {}
            out.append(f"{verb} swallowed {v['errors']} error(s) (last {le.get('at', '?')}: "
                       f"{le.get('what', '?')} in {le.get('where', '?')}); those calls ran unchecked")
    if rec.get("reset"):
        out.append(str(rec["reset"]))
    return rec, out


def health_ack(d=None):
    """Zero the counters and restart `since`: the reading is seen. True when it landed."""
    def change(rec):
        rec.clear(); rec.update({"since": _utc(), "verbs": {}})
    return _health_update(change, d)


def _grepassist_log(tool, decision, reason, pattern):
    """Record one grepassist decision and return 0, the hook's exit code.

    The hook runs on every grep and returns neighbours only for a question-shaped one, so
    a silent call is either a correct decline or a broken hook; this row is what tells
    the two apart. Columns: UTC time, tool, decision (`returned`/`declined`), reason,
    pattern's first 80 characters. A grep that never parsed as one is not logged; a grep
    outside every corpus is (`no-corpus`), because a broken corpus lookup would show as
    exactly that. The file is 0600, since a pattern can carry a secret. Writing must never
    break the hook, so every failure here is swallowed."""
    try:
        _trim_log(paths.PATHS.grepassist_log, GREPASSIST_LOG_CAP)
        clean = re.sub(r"[\r\n\t,\"]+", " ", str(pattern))[:80]
        if clean[:1] in ("=", "+", "-", "@"):
            clean = "'" + clean             # a spreadsheet reads these as a formula
        row = f"{_utc()},{tool},{decision},{reason},{clean}\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        try:
            fd = os.open(str(paths.PATHS.grepassist_log), flags, 0o600)
        except FileNotFoundError:
            paths.PATHS.grepassist_log.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(paths.PATHS.grepassist_log), flags, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(row)                   # one short write: atomic under O_APPEND
    except Exception:
        pass
    return 0


def grep_assist(pattern, root, top=4):
    """Ranked semantic neighbours for a grep about to run over a memory corpus.

    Reference-CSV rows rank here beside the markdown wherever the corpus registers one
    in `reference-sources.json`, because this routes through `memfind.rank` and memfind
    indexes both, rows a markdown-only index cannot see.

    The gap: a grep answers which file contains these WORDS, and a paraphrase shares
    fewer than MIN_SHARED tokens, so it scores 0.0 and reads as absent. 2e says run
    memfind BEFORE the grep, never as a second opinion after one, and that
    instruction was live and ignored twice by supervised sessions, which is why it
    is mechanical here rather than prose.
    A literal grep pattern ranks worse than a phrased question, so this is a floor
    rather than a replacement for asking properly.
    """
    got = semantic_neighbours(root, pattern, top=top, timeout=3)
    if not got or not got[0]:
        return None
    rows, _stale = got
    lines = [f"  [{sc:.3f}] {relay(name, 120)}  ::  {relay(claim, 110)}" for sc, name, claim in rows]
    return ("Before the grep: these rank closest to that text by MEANING, which is what a "
            "grep cannot see (a paraphrase shares too few tokens to match at all).\n"
            + "\n".join(lines)
            + "\nRead them as candidates, never as a verdict; the scores adjudicate nothing "
              "(prime-memory-discipline 2e). A `<file>.csv#<area>` result is one ROW of a reference "
              "CSV rather than a file, so query that CSV on `Area=` for the rest.")


# ---------------------------------------------------------------------------
# Host adapters: the hook modes above speak one payload shape, and three hosts
# produce it. Claude Code's PreToolUse, Codex's PreToolUse and Hermes's
# pre_tool_call carry the same stdin JSON by design (`hook_event_name`,
# `tool_name`, `tool_input`, `cwd`), and differ in tool vocabulary, in how a
# file edit arrives, and in what a reply may say.
# ---------------------------------------------------------------------------
# Per host: how a decision reaches the model (`host_of` tells the payloads apart). `advice` is "context" where a non-blocking message rides the tool event,
# "refuse-once" where the only same-turn channel is a block; `block` names the
# reply shape a refusal takes.
HOSTS = {
    "claude-code": {"advice": "context", "block": "exit2"},
    "codex": {"advice": "context", "block": "deny-json"},
    "hermes": {"advice": "refuse-once", "block": "exit2-json"},
}
# Hermes tool name -> the Claude Code tool the hook modes are written against.
HERMES_TOOLS = {"write_file": "Write", "patch": "Edit", "terminal": "Bash",
                "search_files": "Grep", "read_file": "Read"}
# A Codex patch, as the native `apply_patch` tool's `command` or as the body of an
# `apply_patch` heredoc when the model has no native tool (every model Codex has no
# metadata for, glm through Ollama among them). Same grammar either way.
PATCH_BODY = re.compile(r"\*\*\* Begin Patch\n(.*?)\n?\*\*\* End Patch", re.S)
PATCH_FILE = re.compile(r"\*\*\* (Add|Update|Delete) File: (.+)")
APPLY_PATCH_CMD = re.compile(r"\s*apply_patch\b")


def _absolute(path, cwd):
    """`path` resolved against the SESSION's cwd, never the hook subprocess's own."""
    return path if Path(path).expanduser().is_absolute() else str(Path(cwd or os.getcwd()) / path)


def host_of(payload):
    """Which host sent this hook payload. Hermes names its events differently;
    Codex shares Claude Code's event names and is told apart by the fields only it
    sends (`turn_id`) and by its rollout transcript's file name."""
    event = str(payload.get("hook_event_name") or "")
    if event in ("pre_tool_call", "post_tool_call"):
        return "hermes"
    if payload.get("turn_id") or Path(str(payload.get("transcript_path") or "")).name.startswith("rollout-"):
        return "codex"
    return "claude-code"


def parse_patch(text):
    """[(op, path, added, removed)] for each file a Codex patch touches, op one of
    add | update | delete; `added` and `removed` are the `+` and `-` lines joined.
    A `*** Move to:` retargets the update to its destination. [] where the text
    holds no patch."""
    # CRLF first: the body match needs a bare newline, and a stray \r would ride into
    # every added line; a patch that fails to parse passes the gate silently
    m = PATCH_BODY.search((text or "").replace("\r\n", "\n"))
    if not m:
        return []
    out, cur = [], None
    for line in m.group(1).split("\n"):
        f = PATCH_FILE.fullmatch(line.rstrip())
        if f:
            cur = [f.group(1).lower(), f.group(2).strip(), [], []]
            out.append(cur)
        elif cur is None:
            continue
        elif line.startswith("*** Move to: "):
            cur[1] = line[len("*** Move to: "):].strip()
        elif line.startswith("+"):
            cur[2].append(line[1:])
        elif line.startswith("-"):
            cur[3].append(line[1:])
    return [(op, path, "\n".join(a), "\n".join(r)) for op, path, a, r in out]


def _patch_input(patch, cwd):
    """The Write or Edit a Codex patch amounts to, as (tool, tool_input), or None
    when it adds nothing. A patch can touch several files and a hook judges one, so
    the most decidable is judged first: the trust registry or a store's configuration
    (refused outright, so a benign first file cannot carry a manifest past the gate),
    then a CSV (whose ledger gate may block), then the first file inside a store."""
    files = [f for f in parse_patch(patch) if f[0] != "delete"]
    refused = lambda f: is_registry(_absolute(f[1], cwd)) or store_config(_absolute(f[1], cwd)) is not None
    pick = (next((f for f in files if refused(f)), None)
            or next((f for f in files if f[1].lower().endswith(".csv")), None)
            or next((f for f in files if corpus_for(_absolute(f[1], cwd)) is not None), None)
            or (files[0] if files else None))
    if pick is None:
        return None
    op, path, added, removed = pick
    if op == "add":
        return "Write", {"file_path": path, "content": added}
    return "Edit", {"file_path": path, "old_string": removed, "new_string": added}


def read_payload():
    """The hook payload from stdin, normalised to Claude Code's shape, or None.

    Sets `_host` ("claude-code" | "codex" | "hermes"), maps Hermes tool names, and
    copies Hermes's `path` into `file_path`. A Codex patch, native or through the
    shell, becomes the Write or Edit it amounts to. A `patch` call in Hermes's
    unadvertised V4A mode carries a whole patch string and no `new_string`; it
    stays opaque and the write modes see an empty text, which is silence rather
    than a wrong nudge.
    """
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return None
    return normalise(payload)


def normalise(payload):
    """`read_payload` on an already-parsed dict, so a test can drive it."""
    if not isinstance(payload, dict):
        return None
    ti = payload.get("tool_input")
    if not isinstance(ti, dict):
        ti = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    ti = dict(ti)
    host = host_of(payload)
    tool = payload.get("tool_name")
    if host == "codex" and (tool == "apply_patch" or (
            tool == "Bash" and APPLY_PATCH_CMD.match(str(ti.get("command", ""))))):
        edit = _patch_input(str(ti.get("command", "")), payload.get("cwd"))
        if edit:
            tool, ti = edit
    if host == "hermes" and tool == "memory":
        # Hermes's own memory writes go through this tool, never write_file, so
        # without this mapping the store it exists for is the one it never sees.
        # add = append (an Edit with no anchor), replace = Edit with old_text,
        # remove = nothing added; `operations` batches several of those.
        ops = ti.get("operations") if isinstance(ti.get("operations"), list) else [ti]
        added, old = [], []
        for op in ops:
            if not isinstance(op, dict):
                continue
            if isinstance(op.get("old_text"), str):
                old.append(op["old_text"])   # a removed span leaves the corpus too
            if op.get("action") == "remove":
                continue
            txt = op.get("content") or op.get("new_text")
            if isinstance(txt, str):
                added.append(txt)
        target = ti.get("target") or "memory"   # top-level in both shapes
        ti = {"file_path": str(HERMES / "memories" / ("USER.md" if target == "user" else "MEMORY.md"))}
        if old:                              # replace: an Edit with its anchor
            ti.update(old_string="\n".join(old), new_string="\n".join(added))
        else:                                # add: an append, so the hook sees
            ti.update(content="\n".join(added), _append=True)   # prior + entry
        tool = "Edit"
    elif host == "hermes":
        tool = HERMES_TOOLS.get(tool, tool)
        if "file_path" not in ti and "path" in ti:
            ti["file_path"] = ti["path"]
    # A relative path resolves against the SESSION's cwd, which the payload
    # carries, never against the hook subprocess's own. Left relative it misses
    # every corpus and the write passes with no gate, silently.
    fp = ti.get("file_path")
    if isinstance(fp, str) and fp:
        ti["file_path"] = _absolute(fp, payload.get("cwd"))
    payload = dict(payload, tool_input=ti, tool_name=tool, _host=host)
    return payload


def emit(payload, msg, blocking, source="memscan", db=None):
    """Deliver a hook decision the way THIS host can hear it; returns the exit code.

    Claude Code: exit 2 with the message on stderr blocks; stdout JSON carrying
    `additionalContext` reaches the model without blocking. Codex: the same
    `additionalContext` advises, and a block is a `permissionDecision: deny` on
    stdout, which reaches the model as `Command blocked by PreToolUse hook:
    <reason>` and the write does not land (measured live on codex-cli 0.157.0;
    openai/codex#27833 reports older builds ignoring it); exit 0, since exit 2 on
    a tool event was not measured. Hermes: exit 2 blocks,
    and the message travels as stdout JSON so it is not cut at the 400 characters
    Hermes keeps of stderr. ⚠ Hermes
    has NO same-turn channel for a non-blocking message on a tool event:
    `pre_tool_call` honours only block, modify and approve, a `post_tool_call`
    context is discarded, and `pre_llm_call` runs once per user turn, so advice
    parked for it arrives after the agent has finished. Advice is therefore
    refused once (`refuse_once`), and `source` names the mode so two modes
    refusing one call keep separate records; `db` is the state.db to read.
    """
    if not msg:
        return 0
    host = payload.get("_host", "claude-code")
    if HOSTS.get(host, {}).get("block") == "deny-json" and blocking:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": msg}}))
        return 0
    if HOSTS.get(host, {}).get("advice") == "refuse-once" and not blocking:
        msg = refuse_once(payload, msg.replace(LEAD_WARNING, LEAD_BLOCKING, 1), source, db)
        if msg is None:
            return 0
        blocking = True
    if blocking:
        if host == "hermes":
            print(json.dumps({"action": "block", "message": msg}))
        print(msg, file=sys.stderr)
        return 2
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "additionalContext": msg}}))
    return 0


PENDING_TTL = 3600      # seconds; a refusal record older than this is stale
ONCE_NOTE = "Locket asks this once; to proceed, repeat the same call unchanged."
ONCE_MAX = 2            # refusals of one call before it passes whatever state.db says


def _pending_ours():
    """The record directory exists, is no symlink, and is this user's. A directory
    someone else made could forge or erase the records that let a call through."""
    try:
        return (paths.PATHS.pending_dir.is_dir() and not paths.PATHS.pending_dir.is_symlink()
                and (not hasattr(os, "getuid") or paths.PATHS.pending_dir.stat().st_uid == os.getuid()))
    except OSError:
        return False


def _strings(value):
    """Every string inside a parsed JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def hermes_shown(session_id, message, since, db=None):
    """Whether a tool result Hermes stored for this session at or after `since`
    carries the start of `message`: True, False, or None when state.db cannot say.
    Hermes stores a blocked call's message as that call's result, JSON-encoded,
    before the next tool round runs."""
    db = HERMES / "state.db" if db is None else Path(db)
    if not session_id or not db.is_file():
        return None
    head = message[:300]            # stored JSON-encoded, so it is read back through json
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = con.execute("select content from messages where session_id=? and role='tool' "
                               "and timestamp>=?", (session_id, since - 1)).fetchall()
        finally:
            con.close()
    except Exception:                        # locked, missing table, schema drift
        return None
    for (content,) in rows:
        if not isinstance(content, str):
            continue
        try:
            if any(head in s for s in _strings(json.loads(content))):
                return True
        except (json.JSONDecodeError, ValueError):
            continue
    return False


def refuse_once(payload, msg, source, db=None):
    """The message to refuse a Hermes tool call with, or None to let it pass.

    The same call from the same mode is refused until state.db shows the message
    reached the model, and at most ONCE_MAX times. Hermes runs every pre_tool_call
    hook but shows the model only the first block, so a record alone would let a
    retry pass a second mode's message nobody saw; the cap keeps a result Hermes
    moved to a file, or a state.db that cannot be read, from refusing forever. The
    note leads so the model reads how to proceed before the advice.
    """
    shown = f"{ONCE_NOTE}\n\n{msg}"
    try:
        paths.PATHS.pending_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not _pending_ours():
            return None                      # no record means no retry could pass
        now = time.time()
        for p in paths.PATHS.pending_dir.iterdir():
            try:
                if now - p.stat().st_mtime > PENDING_TTL:
                    p.unlink()
            except OSError:
                continue
        call = json.dumps([source, payload.get("tool_name"), payload.get("tool_input")],
                          sort_keys=True, default=str)
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(payload.get("session_id") or "any"))
        record = paths.PATHS.pending_dir / f"once-{sid}-{hashlib.sha256(call.encode()).hexdigest()[:32]}"
        count = int(record.read_text() or 0) if record.is_file() else 0
        if count and (count >= ONCE_MAX or hermes_shown(payload.get("session_id"), shown,
                                                        record.stat().st_mtime, db) is True):
            record.unlink()
            return None
        record.write_text(str(count + 1))
        return shown
    except Exception:                        # a guardrail never breaks the call it watched
        return None


def gate_status():
    """One line per host on whether the 2e question gate can see a session here,
    and whether a non-blocking nudge has a way to reach the model."""
    lines = []
    if CLAUDE.is_dir():
        lines.append("2e gate on Claude Code: armed (reads the session transcript "
                     "Claude Code hands the hook); nudges reach the model on the write")
    if CODEX.is_dir():
        lines.append("2e gate on Codex: armed (reads the rollout transcript Codex hands the "
                     "hook); nudges reach the model on the write, once /hooks has trusted them")
    if HERMES.is_dir():
        db = HERMES / "state.db"
        lines.append("2e gate on Hermes: " + ("armed (reads state.db by session_id)"
                                              if db.is_file() else
                                              "UNARMED, state.db not found, so every "
                                              "write passes the question silently")
                     + "; a nudge refuses the call once and the same call repeated passes")
    return "\n".join(lines) if lines else "no known host home found; CLI modes only"


CONFIG_REFUSAL = ("This write edits {name}, configuration that decides what Locket does in {store}: "
                  "trigger rows that refuse tool calls, ledger schemas, files read from elsewhere. "
                  "A person edits it by hand; an agent does not, so a README cannot reconfigure "
                  "a store through you. Tell the user what you would change and why.")


def store_config(path):
    """(store, name) when `path` is a file that configures a store Locket acts on: a
    manifest, a legacy sources file, or a rows file, in a trusted store or a host
    store; else None. Decidable from the path, so the gate on it may block."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if len(str(p)) > PATH_MAX:
        return None
    names = {n.lower() for n in (MANIFEST, OLD_MANIFEST, LEGACY_MANIFEST, "triggers.json")}
    for q in (p, Path(os.path.realpath(p))):  # by name on a case-insensitive disk, and through a link
        d = q.parent
        if q.name.lower() in names and (trusted(d) or d in set(paths.PATHS.host_stores.values())):
            return d, q.name
    phys = _phys_root_of(d)
    if phys is not None:
        rows = _read_manifest(phys).get("triggers")
        if isinstance(rows, str) and rows:
            target = Path(rows).expanduser()
            target = target if target.is_absolute() else phys / target
            if _same(target, p):
                return phys, p.name
    return None


_CONFIG_WORD = re.compile(r"[^\s;&|<>()]*(?:locket\.json|memfind\.json|reference-sources\.json|triggers\.json)\b")


_TRUST_VERB = re.compile(r"\b(?:locket|memscan(?:\.py)?|locket\.py)\s+(?:trust|untrust|forget)\b")
TRUST_REFUSAL = ("Granting or withdrawing a store's trust is a person's act at their own terminal, "
                 "never through an agent's shell. Tell the user the command to run.")


def shell_store_config(command, cwd):
    """(store, name) for the first store configuration file a shell command names,
    read as the shell would resolve it, else None. Naming one is enough: no agent task
    needs to touch a store's configuration from a shell, and whether a command writes a
    file cannot be read off its text, while whether it names one can."""
    for i, m in enumerate(_CONFIG_WORD.finditer(command[:MAX_SCAN])):
        if i >= 100:
            break
        tok = m.group(0)
        bases = [""] if tok.startswith(("/", "~", "$HOME", "${HOME}")) else _cd_targets(command, cwd, m.start())
        for b in bases:
            path = _expand_home(tok) if not b else str(Path(b) / tok)
            hit = store_config(path)
            if hit is not None:
                return hit
    return None


REGISTRY_REFUSAL = ("This touches Locket's trust registry or embedder settings ({reg}). Both are a "
                    "person's to change at a terminal: ask the user to run  locket trust <dir>, "
                    "locket untrust <dir>  or  locket configure embedder ...  themselves.")


_PATHLIKE = re.compile(r"""[^\s'"<>;&|()]+\.(?:md|csv)\b""")


def oversized_decision(command, cwd):
    """(message, blocking) for a shell command too long to scan. Its parsers are
    linear, but not run past MAX_SCAN, so padding cannot buy a slow hook; and since
    a skipped scan must not read as a clean one, a command naming a store file
    anywhere is treated as a write to it, never as undecided."""
    if MEMWRITE_ESCAPE in command:
        return None, False
    autonomous = os.environ.get("CLAUDE_AUTONOMOUS") == "1"
    for i, m in enumerate(_PATHLIKE.finditer(command)):
        if i >= 500:
            break
        tok = m.group(0)
        cands = [tok] if tok.startswith(("/", "~")) else [
            str(Path(r) / tok) for r in _cd_targets(command, cwd, m.start())]
        for c in cands:
            c = str(Path(c).expanduser())
            if corpus_for(c) is not None or ledger_schema_for(c)[0] is not None:
                return (f"This command is {len(command)} characters, past the {MAX_SCAN} the guard "
                        f"scans, and names {relay(c, 160)} in a memory store, so it is read as a write "
                        f"to it. Split it, or add {MEMWRITE_ESCAPE} <reason> if the write is intended."), autonomous
    return None, False


def is_registry(path):
    """Whether `path` names a file that decides what Locket trusts or where it sends
    text: the trust registry, or the Panoply settings that name the embedder host.
    By identity, so a case variant or a link is the same file; decidable, so it may block."""
    guarded = [paths.PATHS.registry, Path(os.environ.get("PANOPLY_CONFIG", str(HOME / ".panoply" / "config.json"))),
               paths.PATHS.state / "config.json"]
    try:
        return any(_same(path, g) for g in guarded)
    except (OSError, RuntimeError, ValueError):
        return False


def hook_mode(payload):
    """The write-time gate on one normalised Write or Edit: the trust registry, the
    ledger schema for a CSV, the lexical checks, then the 2e question. Reached from
    `hook`, and from `bashguard` when a shell call turned out to be a Codex patch."""
    ti = payload["tool_input"]
    path = str(ti.get("file_path", ""))
    real = os.path.realpath(_absolute(path, payload.get("cwd"))) if path else ""
    if real and real != path and (is_registry(real) or store_config(real) is not None
                                  or corpus_for(real) is not None):
        path = real                         # a link or alias is judged as the file it reaches
        ti = {**ti, "file_path": real}
        payload = {**payload, "tool_input": ti}
    if path and is_registry(path):
        return emit(payload, REGISTRY_REFUSAL.format(reg=paths.PATHS.registry), True, "hook")
    cfg = store_config(path) if path else None
    if cfg is not None:
        return emit(payload, CONFIG_REFUSAL.format(name=cfg[1], store=cfg[0]), True, "hook")
    if path.lower().endswith(".csv"):  # a ledger row: schema, decidable, may block
        try:
            msg, blocking = ledger_decision(path, ti)
            if not blocking:
                prior = read_or_empty(path)
                msg = source_row_decision(path, ti, prior)[0]
        except Exception as e:          # a guardrail must never break a write
            health_error("hook", "ledger", e)
            return 0
        return emit(payload, msg, blocking, "hook")
    root = corpus_for(path)
    if root is None:
        return 0
    text = str(ti.get("content") or ti.get("new_string") or "")
    prior = read_or_empty(path)         # "" for a new file, an unreadable one or a refused one
    if ti.get("_append") and prior:     # an append is a Write of prior + entry
        text = prior.rstrip("\n") + "\n" + text
    name = Path(path).name
    try:
        msg, blocking = hook_decision(
            name, ti, text, prior, lambda: corpus(root),
            lambda: archived_claims(root),
            autonomous=os.environ.get("CLAUDE_AUTONOMOUS") == "1",
            sets=settings(root), root=root)
    except Exception as e:              # a guardrail must never break a write: a
        health_error("hook", "lexical", e)  # dangling symlink in the corpus raised here
        return 0
    if msg and blocking:
        return emit(payload, msg, True)
    # The silent path is the paraphrase blind spot: lexical scoring found
    # nothing, which is also what it returns for a fact restated in new words.
    try:
        gate = memfind_gate(payload, name, ti, text, prior, root)
    except Exception as e:              # a guardrail must never break a write
        health_error("hook", "2e", e)
        gate = None
    # A non-blocking nudge must still reach the model when the 2e gate blocks,
    # or the warning is discarded by the very write that gets refused. ⚠ It is
    # re-led on the way: exit 2 means the write did NOT land, and a warning
    # worded for the landed case would assert the opposite of what happened.
    if gate:
        warn = msg.replace(LEAD_WARNING, LEAD_BLOCKING, 1) if msg else None
        return emit(payload, "\n\n".join(x for x in (warn, gate) if x), True)
    return emit(payload, msg, False, "hook")


def main(argv):
    if len(argv) < 2 or argv[1] in ("--help", "-h", "help"):
        print(usage())
        return 0 if len(argv) > 1 else 1
    mode = "pointers" if argv[1] == "hooks" else argv[1]   # the older name
    if mode in ("hook", "bashguard", "grepassist"):
        with HookRun(mode):
            return _hook_entry(mode)

    if mode == "deliver":                   # retired: an older Hermes config still calls it
        sys.stdin.read()
        return 0

    return _main_rest(mode, argv)


def _hook_entry(mode):
    """The three PreToolUse entry points, each reading its JSON payload from stdin."""
    if mode == "hook":
        payload = read_payload()
        if payload is None:
            return 0
        return hook_mode(payload)

    if mode == "bashguard":                 # PreToolUse on Bash: the writes it misses
        payload = read_payload()
        if payload is not None and payload["tool_name"] in ("Write", "Edit"):
            return hook_mode(payload)       # a Codex patch sent through the shell
        if payload is None or payload["tool_name"] != "Bash":
            return 0
        command = str(payload["tool_input"].get("command", ""))
        if len(command) > MAX_SCAN:
            try:
                msg, blocking = oversized_decision(command, payload.get("cwd"))
            except Exception as e:
                health_error("bashguard", "decision", e)
                return 0
            return emit(payload, msg, blocking, mode)
        bare = re.sub(r"""['"\\]""", "", command_text(command))
        if _TRUST_VERB.search(bare):
            return emit(payload, TRUST_REFUSAL, True, mode)
        cfg = shell_store_config(bare, payload.get("cwd"))
        if cfg is not None:
            return emit(payload, CONFIG_REFUSAL.format(name=cfg[1], store=cfg[0]), True, mode)
        if ((".locket" in bare and ("corpora.json" in bare or "config.json" in bare))
                or (".panoply" in bare and "config.json" in bare) or str(paths.PATHS.registry) in bare):
            # a shell command naming the registry is refused whether or not it writes: no
            # agent task needs the file, and a write to it would grant trust
            return emit(payload, REGISTRY_REFUSAL.format(reg=paths.PATHS.registry), True, mode)
        try:
            msg, blocking = ledger_bash_decision(command, payload.get("cwd"))
            if msg is None:
                msg, blocking = bash_write_decision(
                    command, payload.get("cwd"),
                    autonomous=os.environ.get("CLAUDE_AUTONOMOUS") == "1")
        except Exception as e:              # a guardrail must never break the shell
            health_error("bashguard", "decision", e)
            return 0
        return emit(payload, msg, blocking, mode)

    if mode == "grepassist":                # PreToolUse on Grep|Bash: 2e, automatic
        tool, pattern = "?", ""
        try:
            payload = read_payload()
            if payload is None:
                return 0
            ti = payload["tool_input"]
            tool = payload["tool_name"]
            cwd = payload.get("cwd") or os.getcwd()
            if tool == "Grep":
                pattern, words = str(ti.get("pattern", "")), [ti["path"]] if ti.get("path") else []
            elif tool == "Bash":
                got = grep_words(str(ti.get("command", "")), cwd)
                if got is None:
                    return 0
                pattern, words = got
            else:
                return 0
            root = search_root(words, cwd) or corpus_here(cwd)
            if root is None:
                return _grepassist_log(tool, "declined", "no-corpus", pattern)
            if not prose_like(pattern):
                return _grepassist_log(tool, "declined", "not-a-question", pattern)
            out = grep_assist(pattern, root)
        except Exception as e:              # a bad payload, anything; an absent embedder returns None
            health_error("grepassist", "rank", e)
            return _grepassist_log(tool, "declined", f"error:{type(e).__name__}", pattern)
        _grepassist_log(tool, "returned" if out else "declined", "ranked" if out else "no-neighbours", pattern)
        return emit(payload, out, False, mode)
    return 0


def _main_rest(mode, argv):
    """Every non-hook mode of `main`."""
    if mode == "init":                      # needs no corpus of its own: it makes one
        rest, flags = list(argv[2:]), {}
        led = [a for a in rest if a == "--ledgers" or a.startswith("--ledgers=")]
        for a in led:
            rest.remove(a)
        ledgers = led[-1].partition("=")[2] if led else None
        for f in ("--parent", "--name"):
            if f in rest:
                i = rest.index(f)
                if i + 1 >= len(rest):
                    print(f"{f} needs a value", file=sys.stderr)
                    return 1
                flags[f] = rest[i + 1]; del rest[i:i + 2]
        if len(rest) != 1:
            print("init needs a directory:  locket init <dir> [--name <label>] [--parent <corpus>] "
                  "[--ledgers[=<Name>]]  (a ledger name takes `=`, never a space)", file=sys.stderr)
            return 1
        try:
            return cmd_init(rest[0], parent=flags.get("--parent"), name=flags.get("--name"), ledgers=ledgers)
        except OSError as e:                # a refused write (a symlinked manifest or ledger) is named
            print(f"init: {e}", file=sys.stderr)
            return 1
    if mode == "migrate":                   # rename memfind.json -> locket.json, one store or all
        one = len(argv) > 2 and argv[2] != "all"
        targets = [resolve(argv[2])] if one else list(_raw_corpora().values())   # joined stores too
        seen, n = set(), 0
        for r in targets:
            if r is None or r in seen:
                continue
            seen.add(r)
            what = migrate(r)
            if what not in ("none", "already") or one:  # done or never needed is not news on a sweep
                print(f"{r}: {what}"); n += 1
        if not n:
            print("migrate: no store manifest needs renaming")
        return 0
    if mode in ("forget", "untrust"):
        if len(argv) < 3:
            print(f"{mode} needs a directory:  locket {mode} <dir>", file=sys.stderr)
            return 1
        d = Path(argv[2]).expanduser()
        if any(_same(d, r) for r in registry()) and not person_confirms(
                f"Withdraw trust from {d}? Its trigger rows, ledger schemas and sources stop acting.",
                [], "withdraw it"):
            print(f"registry: unchanged. Withdrawing a store's guardrails is a person's act at a "
                  f"terminal; ask the user to run:  locket {mode} {shlex.quote(str(d))}")
            return 1
        print(f"registry: {paths.PATHS.registry} ({'row removed' if forget(argv[2]) else 'not listed'})")
        return 0
    if mode == "trust":
        if len(argv) < 3 or not Path(argv[2]).expanduser().is_dir():
            print("trust needs a directory:  locket trust <dir>", file=sys.stderr)
            return 1
        return 0 if grant_trust(argv[2]) else 1
    if mode == "schema":                    # needs no corpus: writes the manifest schema
        if len(argv) > 2:
            findings = validate_manifest(Path(argv[2]).expanduser())
            print("\n".join(findings) if findings else f"{manifest_file(Path(argv[2]).expanduser())}: valid")
            return 1 if findings else 0
        return cmd_schema()
    if mode == "corpora":                   # needs no corpus of its own
        print("\n".join(corpora_lines()))
        return 0
    if mode == "across":                    # every corpus at once, by construction
        # `across all` is what every other Step 5 line teaches; it is not a floor.
        floor = next((float(a) for a in argv[2:3] if re.fullmatch(r"[0-9.]+", a)), PAIR_FLOOR)
        cmd_across(floor)
        return 0
    if len(argv) > 2 and argv[2] == "all":
        # Every corpus on the machine, one after another. The shape a user with
        # one global memory wants, and the only one that needs no vocabulary.
        rc = 0
        for k, v in sorted(all_corpora().items()):
            if not md_files(v):              # an empty store has nothing to report
                continue
            print(f"== {k}")
            rc |= main([argv[0], mode, str(v)] + argv[3:])
            print()
        return rc
    if len(argv) > 2:
        root = resolve(argv[2])
    else:
        # No argument means THIS directory's corpus, never a hardcoded one: a
        # fixed default makes `memscan audit` in another repo sweep that corpus
        # and report it as yours.
        root = corpus_here(os.getcwd())
        if root is None:
            print("no corpus for this directory; name one, or `init` it:", file=sys.stderr)
            for k, v in sorted(all_corpora().items()):
                print(f"  {k:70s} {len(md_files(v)):4d} files", file=sys.stderr)
    if root is None:
        return 1
    if mode != "against":
        # Which directory, and whether the write-time hook reaches it. `resolve`
        # accepts any literal path and `corpus_for` is strict, so the CLI can audit
        # a directory the hook will never protect.
        guarded = corpus_for(root / "_probe.md") is not None
        print(f"corpus: {root}"
              f"  [write-time hook: {'ACTIVE' if guarded else 'NOT PROTECTED — audit only'}]")
    if mode == "audit":
        cmd_audit(root, float(argv[3]) if len(argv) > 3 else PAIR_FLOOR)
    elif mode == "pointers":
        cmd_pointers(root)
    elif mode == "links":
        cmd_links(root)
    elif mode == "cites":
        cmd_cites(root, filters=argv[3:] or None)
    elif mode == "budget":
        cmd_budget(root)
    elif mode == "ledgers":
        cmd_ledgers(root)
    elif mode == "dupes":
        cmd_dupes(root, files=argv[3:] or None)
    elif mode == "graduated":
        cmd_graduated(root, float(argv[3]) if len(argv) > 3 else PAIR_FLOOR)
    elif mode == "against":
        cmd_against(root, sys.stdin.read())
    else:
        # An unknown mode is usually the corpus argument with the mode omitted.
        print(f"Unknown mode {mode!r}. Did you forget a mode "
              f"(audit, across, dupes, budget, ledgers, pointers, links, cites, corpora, init, trust, untrust, schema, migrate, forget, against)?")
        print(usage())
        return 1
    return 0


if __name__ == "__main__":
    # memfind imports memscan; run as a script, this module is __main__, and without the
    # alias that import would load and run a second copy with caches of its own
    sys.modules.setdefault("memscan", sys.modules[__name__])
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:      # piping any mode into `head` is normal use
        sys.stderr.close()
        sys.exit(0)
