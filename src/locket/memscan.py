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
import fnmatch
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath

HOME = Path.home()
CLAUDE = HOME / ".claude"                # Claude Code's home
HERMES = HOME / ".hermes"                # Hermes Agent's home
# A per-corpus manifest, optional. Its presence makes ANY directory a corpus for
# the write-time gate and for a literal path, so a memory store on a host this
# file has never heard of is covered by writing one file into it; its absence
# means the host defaults below. Every key it may carry is listed once, under
# MANIFEST in USAGE; `settings`, `csv_sources` and `all_corpora` read them. It
# and the `.memfind/` cache are the only two things this tool ever owns inside
# a corpus.
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
REGISTRY = Path(os.environ.get("LOCKET_REGISTRY", str(HOME / ".locket" / "corpora.json")))
# Each host's core directory is its council store, found without a manifest:
# the whole of `~/.claude` (memory, rules, skills, agents and their memories,
# the constitution at its root) is one corpus, so a rule restating a memory
# fact draws the write-time nudge instead of surfacing only in `across`.
# Claude Code's per-project auto-memory is derived from the path instead
# (`corpus_for`) and is a store of its own, never a member of council.
HOST_STORES = {
    "council": CLAUDE,
    "hermes": HERMES,
}
# What counts as corpus INSIDE a host's core directory. The home is mostly host
# state (sessions, caches, plugins, transcripts, the project stores above), and
# the host adds directories between releases, so the default is an allowlist:
# a directory not named here is out until a manifest's `members` lets it in,
# which fails by omission where a denylist leaks every new directory silently.
# Root-level markdown (CLAUDE.md, SOUL.md, REBUILD.md) is always in. Hermes's
# `skills/` is bundled with the agent rather than authored, so it is out.
HOST_MEMBERS = {
    CLAUDE: {"memory", "rules", "skills", "agents", "agent-memory"},
    HERMES: {"memories"},
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
CORPORA = dict(HOST_STORES)
# Directories inside a corpus that are deliberately NOT deduplicated. An archive
# is a verbatim copy of superseded text, so it duplicates by design and would trip
# on every write; an in-flight investigation thread appends until it closes
# (prime-memory-discipline §6); `discovery-research` holds Mode-2 angle files the
# surveillance cycle appends to on a schedule, with no session in front of it to
# read a nudge that BLOCKS the write. The cache directory is never prose.
EXCLUDED_DIRS = {"archive", "in-flight", "harness-logs", "discovery-research", ".memfind",
                 "backups", "backup"}   # a backup is a verbatim copy, so it forks by design


def registry():
    """Registered corpus paths, as `init` wrote them. [] where the file is absent
    or unreadable, so enumeration degrades to the host stores rather than dies."""
    try:
        data = json.loads(REGISTRY.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    rows = data.get("corpora") if isinstance(data, dict) else None
    return [Path(x) for x in rows if isinstance(x, str)] if isinstance(rows, list) else []


def registry_stale():
    """Registered paths whose manifest is gone. Listed, never enumerated."""
    return [p for p in registry() if not has_manifest(p)]


def register(root):
    """Add a corpus path to the registry, once. True when it was new."""
    root = Path(root).resolve()
    rows = registry()
    if root in rows:
        return False
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(
        {"corpora": sorted(str(p) for p in rows + [root])}, indent=2) + "\n")
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
    old.rename(root / MANIFEST)
    settings.cache_clear()
    return "migrated"


def forget(root):
    """Drop a path from the registry. True when a row was removed. The manifest
    and the cache inside the directory are the user's to delete; this only
    stops the listing naming it."""
    root = Path(root).expanduser().resolve()
    rows = registry()
    if root not in rows:
        return False
    REGISTRY.write_text(json.dumps(
        {"corpora": sorted(str(p) for p in rows if p != root)}, indent=2) + "\n")
    return True


def _raw_corpora():
    """Every physical store on the machine, joined or not: each host store that
    exists, one per Claude Code project, and every registered directory whose
    manifest is still there. A registered store is keyed by the `name` in its
    manifest, else its basename, else its full path where either collides."""
    out = {k: v for k, v in HOST_STORES.items() if v.is_dir()}
    for d in sorted((CLAUDE / "projects").glob("*/memory")):
        out[d.parent.name] = d
    for p in registry():
        if not has_manifest(p) or p in out.values():
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
    target = raw.get(name)
    if target is None or target == root or root in target.parents or _walked_by(target, root):
        return None
    seen, cur = {root}, target                  # a cycle means nobody is the top
    while cur is not None:
        if cur in seen:
            return None
        seen.add(cur)
        nxt = _read_manifest(cur).get("belongs_to")
        cur = raw.get(nxt) if isinstance(nxt, str) else None
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


def corpus_key(root, path):
    """The corpus-relative key of a file: its path under `root`, or under a joined
    store as `<label>/<path>`. One key per file across every mode and the index,
    because a joined store's files are not under the root on disk."""
    root, path = Path(root), Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        pass
    for r in roots_of(root)[1:]:
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
    if name in CORPORA:
        return top_of(CORPORA[name])
    p = Path(name).expanduser()
    if p.is_dir():
        return top_of(p.resolve())
    hits = {k: v for k, v in all_corpora().items() if name.lower() in k.lower()}
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
    any location), the host stores in HOST_STORES, and Claude Code's
    `~/.claude/projects/<slug>/memory`. A non-markdown path and anything under
    the corpus's excluded directories return None.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if p.suffix != ".md":
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
    stores = set(HOST_STORES.values())
    for parent in (d, *d.parents):
        if parent in stores or has_manifest(parent):
            return parent
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


def _read_manifest(root):
    """The manifest as a dict, or {} where it is absent or unreadable. A broken
    manifest must degrade to the defaults rather than take a hook down with it."""
    if root is None:
        return {}
    m = manifest_file(root)
    if not m.is_file():
        return {}
    try:
        data = json.loads(m.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
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
                 "description": "What `corpora` and a store argument call this store. Default: the "
                                "directory's basename. Host stores and project memories ignore it."},
        "members": {**_STR_LIST,
                    "description": "Top-level directory names that ARE corpus; the rest of the root is "
                                   "host state. Root-level markdown is always in. Replaces the default."},
        "excluded_dirs": {**_STR_LIST,
                          "description": "Directory names, at any depth, that are not corpus. Replaces the "
                                         "default (archive, in-flight, harness-logs, discovery-research)."},
        "excluded_files": {**_STR_LIST,
                           "description": "Corpus-relative paths or globs of files that are not corpus, "
                                          "skipped by every mode, the index and the write-time gate."},
        "index_files": {**_STR_LIST,
                        "description": "File names whose lines POINT at other files, so a line restating "
                                       "its target is not a fork. Replaces the default."},
        "ledger_surfaces": {**_STR_LIST,
                            "description": "File names that accumulate entries by design and are never "
                                           "checked. Replaces the default."},
        "holding_spaces": {**_STR_LIST,
                           "description": "File names scored only against themselves. Replaces the default."},
        "whole_read_exempt": {**_STR_LIST,
                              "description": "File names the size sweep skips: a mixed surface never read "
                                             "whole by its own contract, or a file a project instrument "
                                             "already caps. Default: none."},
        "code_roots": {**_STR_LIST,
                       "description": "Directories `cites` resolves path:line citations against, "
                                      "absolute or relative to the store."},
        "sources": {
            "type": "array",
            "description": "Reference CSVs whose rows are claims in this corpus.",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["path", "text"],
                      "properties": {
                          "path": {"type": "string", "description": "The CSV, absolute or store-relative."},
                          "text": {"type": "string", "description": "Column holding the claim sentence."},
                          "label": {"type": "string", "description": "Column that names a row in output (an Area)."}}}},
        "ledgers": {
            "type": "array",
            "description": "Schemas for CSV ledgers in this store; a row outside one is refused at write time.",
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
        "belongs_to": {"type": "string",
                       "description": "Name of another corpus this directory JOINS, as `corpora` lists it; "
                                      "written by `init <dir> --parent <name>`."},
    },
}
SCHEMA_FILE = REGISTRY.parent / "locket.schema.json"


def cmd_schema(quiet=False):
    """Write the manifest schema beside the registry, so an editor can be pointed
    at one predictable path per machine and the tool can check a manifest against
    the same file. Rerun after an update; `init` runs it."""
    SCHEMA_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_FILE.write_text(json.dumps(MANIFEST_SCHEMA, indent=2) + "\n")
    if not quiet:
        print(f"schema: {SCHEMA_FILE} (written)")
        print(f"editor: VS Code reads it by filename through one user-settings entry; "
              f"`locket help install` (Later) names it")
    return 0


def _check(value, schema, path, out):
    """Structural check of `value` against the subset of JSON Schema the manifest
    schema uses: type, properties, additionalProperties, required, items."""
    import difflib
    t = schema.get("type")
    kinds = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool}
    if t in kinds and (not isinstance(value, kinds[t]) or (t == "integer" and isinstance(value, bool))):
        out.append(f"{path or 'manifest'}: expected {t}, got {type(value).__name__}")
        return
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                out.append(f"{path or 'manifest'}: missing required key {k!r}")
        for k, v in value.items():
            if k in props:
                _check(v, props[k], f"{path}.{k}" if path else k, out)
            elif schema.get("additionalProperties") is False:
                near = difflib.get_close_matches(k, props, n=1, cutoff=0.6)
                out.append(f"{path or 'manifest'}: unknown key {k!r}"
                           + (f", did you mean {near[0]!r}?" if near else ""))
            elif isinstance(schema.get("additionalProperties"), dict):
                _check(v, schema["additionalProperties"], f"{path}.{k}" if path else k, out)
    elif t == "array" and "items" in schema:
        for i, v in enumerate(value):
            _check(v, schema["items"], f"{path}[{i}]", out)


def validate_manifest(root):
    """Findings for the manifest at `root` against MANIFEST_SCHEMA; [] when clean
    or absent. Structure only: a key the reader would silently ignore, a wrong
    type, a malformed source or ledger entry. Whether the values are right is
    the modes' business."""
    m = manifest_file(root)
    if not m.is_file():
        return []
    try:
        data = json.loads(m.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"manifest does not parse: {e}"]
    out = []
    _check(data, MANIFEST_SCHEMA, "", out)
    for i, e in enumerate(data.get("ledgers") or [] if isinstance(data, dict) else []):
        x = e.get("extends") if isinstance(e, dict) else None
        if x is not None and x not in LEDGER_SCHEMAS:
            out.append(f"ledgers[{i}].extends: {x!r} is no built-in; one of {', '.join(LEDGER_SCHEMAS)}")
    return out


def _own_walk(root, m=None):
    """-> (members, excluded_dirs): what one store's own walk covers, from its own
    manifest alone. Never unioned across a join, so `parent_of` can ask it of a
    parent without recursing through `settings`, which unions name-sets upward."""
    m = _read_manifest(root) if m is None else m
    mem = m.get("members")
    members = ({str(x) for x in mem if isinstance(x, str)} if isinstance(mem, list)
               else (set(HOST_MEMBERS[Path(root)]) if root is not None
                     and Path(root) in HOST_MEMBERS else None))
    val = m.get("excluded_dirs")
    excluded = {str(x) for x in val} if isinstance(val, list) else set(EXCLUDED_DIRS)
    excluded.add(".memfind")                  # the cache is never prose, whatever the manifest says
    if root is not None and Path(root) in HOST_EXCLUDED:
        excluded |= HOST_EXCLUDED[Path(root)]
    excluded.update({"backups", "backup"})    # nor is a backup, which forks by design
    return members, excluded


@lru_cache(maxsize=None)
def settings(root):
    """Per-corpus name-sets, the module defaults overridden by the manifest.

    Every place that once read a module-level set reads this instead, keyed on
    the corpus, so two corpora on one machine can name their ledgers differently.
    The defaults are this apparatus's file names and are what a corpus without a
    manifest gets, which keeps the Claude Code install zero-config.
    """
    m = _read_manifest(root)
    members, excluded = _own_walk(root, m)
    # A host store whose files are not this apparatus's: Hermes's MEMORY.md is
    # the whole memory, not an index over topic files, so it must not inherit
    # the index exemption. A manifest there still wins.
    if root is not None and Path(root) == HOST_STORES["hermes"]:
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
APPEND_SURFACES = LEDGER_SURFACES | HOLDING_SPACES

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
    """`txt` with hard wraps inside a sentence joined by a space."""
    lines = txt.split("\n")
    out = []
    for ln in lines:
        prev = out[-1] if out else None
        if (prev is not None and prev.strip() and not _ENDS.search(prev)
                and not _NO_JOIN_FROM.match(prev)
                and ln.strip() and not _LINE_MEANINGFUL.match(ln)):
            out[-1] = prev.rstrip() + " " + ln.strip()
        else:
            out.append(ln)
    return "\n".join(out)
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


def _own_md(root):
    """The markdown physically under one store, by that store's own settings."""
    root = Path(root)
    skip = settings(root)["excluded_dirs"]
    # A subdirectory carrying its own manifest is a corpus of its own: `_root_of`
    # resolves a write inside it to the inner store, so counting its files here
    # too would score them against two stores and report a file against itself
    # in `across`. `init` refuses to nest, but a manifest written by hand can.
    nested = {m.parent.relative_to(root) for name in (MANIFEST, OLD_MANIFEST)
              for m in root.rglob(name) if m.parent != root}
    # A listed store below this one (a project memory under the council home
    # carries no manifest) belongs to itself, whatever the member list says.
    nested |= {r.relative_to(root) for r in _raw_corpora().values()
               if r != root and root in r.parents}
    return sorted(p for p in _member_md(root)
                  if not any(q.name in skip or q.name.startswith(".") or q in nested
                             for q in p.relative_to(root).parents)
                  and not excluded_file(root, p))


def _member_md(root):
    """Every markdown file under the root's member directories, plus the root's
    own. Walking only the members is what keeps a host store cheap: the council
    home holds session transcripts and plugin trees that a bare rglob would
    walk on every write-time hook."""
    root = Path(root)
    mem = settings(root)["members"]
    if mem is None:
        yield from root.rglob("*.md")
        return
    yield from root.glob("*.md")
    for name in sorted(mem):
        d = root / name
        if d.is_dir():
            yield from d.rglob("*.md")


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
        if not m.is_file():
            return []
        try:
            entries = json.loads(m.read_text())
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
            with open(pth, newline="", errors="ignore") as fh:
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
    docs = {corpus_key(root, p): strip(p.read_text(errors="ignore"))
            for p in md_files(root)}
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
            body = p.read_text(errors="ignore")
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
    root = Path(root)
    s = settings(root)
    others = s["excluded_dirs"] - {"archive"}
    tops = [root] if s["members"] is None else [root / m for m in sorted(s["members"])]
    for top in tops:
        if not top.is_dir():
            continue
        for a in top.rglob("archive"):
            if not a.is_dir():
                continue
            rel = a.relative_to(root).parts[:-1]
            if any(q.startswith(".") or q in others for q in rel):
                continue
            if any(has_manifest(root / Path(*rel[:i + 1])) for i in range(len(rel))):
                continue
            yield from a.rglob("*.md")


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
    """Bucket by the 3 rarest tokens, so candidate generation stays near-linear."""
    inv = defaultdict(list)
    for i, (_, _, ts) in enumerate(items):
        for w in sorted(ts, key=lambda w: df[w])[:3]:
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


def pointer_only(sentence):
    """True where a sentence's payload is a cross-reference, not a claim.

    These repeat across files by design (see X, composes Y) and are the noise
    floor of `audit`.
    """
    s = sentence.lower().lstrip("- ")
    return s.startswith(("see ", "composes ", "full per-record ledger", "re-verify", "ship-state"))


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
    for p in md_files(root):
        if p.name not in index_files:
            continue
        for line in p.read_text(errors="ignore").splitlines():
            m = IDX_LINE.match(line)
            if m:
                hook = m.group("hook").lstrip(" —-").strip()
                if hook:
                    t = p.parent / m.group("target")
                    out.append((corpus_key(root, p), corpus_key(root, t), m.group("title"), hook, t))
    return out


def dangling(root):
    files = {p.stem.replace("-", "_") for p in md_files(root)}
    out = defaultdict(list)
    for p in md_files(root):
        for t in LINK.findall(p.read_text(errors="ignore")):
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
        with open(path, newline="", errors="ignore") as fh:
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
        txt = p.read_text(errors="ignore")
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
    "ide", "daemon", "tasks", "chrome",
})


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
    """
    idx = defaultdict(list)
    for base, extra in roots:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and d not in extra
                           and not d.startswith(".")]
            for fn in filenames:
                idx[fn].append(os.path.join(dirpath, fn))
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
        for m in CITE.finditer(p.read_text(errors="ignore")):
            out[(m.group(1), int(m.group(2)))].add(p.name)
    return out


def resolve_citation(path, line, idx):
    """(status, detail) for one citation. Status is ok / eof / missing."""
    hits = idx.get(os.path.basename(path), [])
    exact = [h for h in hits if h.endswith("/" + path) or h.endswith(path)]
    hits = exact or hits
    if not hits:
        return "missing", None
    # splitlines, not split("\n"): the latter counts a trailing newline as a
    # 89th line in an 88-line file, so a citation one past the end reads as ok.
    lines = Path(hits[0]).read_text(errors="replace").splitlines()
    if line > len(lines):
        return "eof", (hits[0], len(lines), len(hits))
    return "ok", (hits[0], lines, len(hits))


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
        md = {corpus_key(root, p): str(p.resolve()) for p in md_files(root)}
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
        print(f"[{jac:.2f}]{tag} {fa}\n       {fb}\n   A: {sa[:200]}\n   B: {sb[:200]}\n")
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
        print(f"[{jac:.2f}] {fa}\n       {fb}\n   A: {sa[:200]}\n   B: {sb[:200]}\n")
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
        print(f"[{jac:.2f}] {fh}\n       {fd}\n   holding: {sh[:200]}\n   durable: {sd[:200]}\n")
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


def cmd_init(target, index=True, parent=None, name=None):
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
    store's own member directory takes."""
    d = Path(target).expanduser()
    if not d.is_dir():
        print(f"{d} is not a directory", file=sys.stderr)
        return 1
    d = d.resolve()
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
        if d in set(HOST_STORES.values()):
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
        man = dict(_read_manifest(d))
        if name:
            man["name"] = name
        man["belongs_to"] = parent
        m.write_text(json.dumps(man, indent=2) + "\n")
        settings.cache_clear()
        if parent_of(d, _raw_corpora()) != target:
            print(f"{d} did not resolve to {target} after the write; check for a cycle "
                  f"through {parent!r}", file=sys.stderr)
            return 1
        print(f"manifest: {m} (belongs_to {parent!r})")
        top = top_of(d)
        stale = d / ".memfind"
        if stale.is_dir():
            print(f"index: {stale} is no longer read; the corpus indexes at {top}")
        n, rows = len(md_files(top)), len(csv_rows(top))
        print(f"corpus: {top}  {n} files" + (f" + {rows} csv rows" if rows else "")
              + f"  [{len(roots_of(top)) - 1} joined]")
        if not index:
            return 0
        try:
            import memfind
            memfind.build_index(top, quiet=True)
            print(f"index: {top / '.memfind'} built")
        except ImportError:
            print(f"index: memfind.py not found beside memscan.py; run  locket index {top}")
        except RuntimeError as e:
            print(f"index: not built ({e}); run  locket index {top}  once an embedder answers")
        return 0
    # A host store or a project memory is listed on sight and takes no registry
    # row; a manifest there is optional config (sources, exclusions), kept if
    # present and never written for it.
    by_convention = inside == d and not m.is_file()
    if m.is_file():
        print(f"manifest: {m} (kept)")
        by_convention = d in set(HOST_STORES.values()) or (
            d.name == "memory" and d.parent.parent == CLAUDE / "projects")
    elif by_convention:
        print(f"manifest: none needed; {d} is a corpus by convention on this host")
    else:
        m.write_text("{}\n")
        print(f"manifest: {m} (written)")
    settings.cache_clear()
    if by_convention:
        print("registry: not needed; listed by convention")
    else:
        print(f"registry: {REGISTRY} ({'added' if register(d) else 'already listed'})")
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
        print(f"index: {d / '.memfind'} built")
    except ImportError:
        print(f"index: memfind.py not found beside memscan.py; run  locket index {d}")
    except RuntimeError as e:
        print(f"index: not built ({e}); run  locket index {d}  once an embedder answers")
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
        print(f"[{jac:.2f}] {fa}\n   A: {sa[:200]}\n   B: {sb[:200]}\n")
    return len(pairs)


_DATE = r"20\d\d[-/]\d\d(?:[-/]\d\d)?"
# A dated HISTORY clause, in the shapes loaded-whole files actually carry: `Update 2026-08-08`,
# `⚠ UPDATE (2026-08-08)`, `2026-08-08: **Resolved:**`, and the italic sub-header that opens a
# stacked block, `*Refinement, 2026-05-19:*`. The `update <date>` form alone matched none of
# SELF.md's ten stacks. A date that scopes a live behaviour (`since 2026-09-16`, `(Peer ruling
# 2026-08-15)`) is not history and must not match; the negatives in `selftest` hold that line.
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
    for p in md_files(root):
        if p.name in ledger_names:           # chronology: length is the design, not a breach
            ledgers.append(f"{p.stat().st_size // 1024}KB {corpus_key(root, p)}")
            continue
        txt = p.read_text(errors="ignore")
        name = corpus_key(root, p)
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
          f"{w} with a {PARA_WORDS}+ word paragraph [P]. Largest 25:\n")
    def mark_of(size, upd, para):
        return "".join(c for c, hit in
                       (("B", size > ENTRY_BYTES), ("U", upd >= UPDATE_CAP), ("P", para > PARA_WORDS))
                       if hit)

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
            defects.append(f"!! {path}:{line}  FILE NOT FOUND   <- {who}")
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


def selftest():
    # Every fixture is synthetic. The live memory store is never read, because a
    # test that reads it encodes the ORIGIN's manifest: on a store with different
    # ledger and holding-space names the assertions fail, and the first instance
    # to install this cold made them pass by rewriting its own manifest to match
    # ours. A council-shaped store is built in a temp dir and HOST_STORES is
    # pointed at it for the duration.
    import tempfile, shutil
    fixture = Path(tempfile.mkdtemp())
    COUNCIL = fixture / "memory"
    COUNCIL.mkdir()
    saved_council = HOST_STORES["council"]
    HOST_STORES["council"] = COUNCIL
    settings.cache_clear()
    docs = {
        "a.md": "The default execution path is the dossier pipeline run interactively.\n"
                "Unrelated sentence about hangar rates and insurance quotes entirely.",
        "b.md": "Something else here about propeller pitch enumerations and defaults.\n"
                "The default execution path is the dossier pipeline, run interactively.",
        "c.md": "See project doc basis per engine class for the whole seam and ladder.",
    }
    wrapped = ("- first item that wraps\n  onto a second line\n  and onto a third\n"
               "- second item that wraps\n  onto a second line\n  and a third line too\n")
    assert list(_prose_paras(wrapped)) == [], "a hard-wrapped list is a list, not a paragraph"
    lazy = "- one item\ncontinued without indentation\nand still the item\n"
    assert list(_prose_paras(lazy)) == [], "a lazy continuation belongs to its item"
    headed = "# Heading\nProse follows the heading directly.\nMore prose.\nAnd more.\n"
    assert list(_prose_paras(headed)) == [len(headed.split())], "a heading opens no run"
    indented = "    Indented prose under no marker\n    is still prose.\n"
    assert list(_prose_paras(indented)) == [len(indented.split())], "indentation alone is not a list"

    # §5 dated history clauses: the shapes loaded-whole files carry, not only `Update <date>`.
    # Positive control is SELF.md's stack form; the negatives are scoped dates §1 permits,
    # and a regex that fires on them turns a worklist into noise.
    for pos in ("*Refinement, 2026-05-19: session-end discipline is structurally load-bearing.*",
                "**Some stale assertion**. 2026-08-08: **Resolved:** the Peer had a sandwich.",
                "⚠ UPDATE (2026-08-08): the endpoint moved to the API repo.",
                "*Instances since graduation, folded from the Workspace 2026-09-11 (supervised).*",
                "Update 2026-08-08: the cap went to $5."):
        assert HISTORY_CLAUSE.search(pos), f"history clause not matched: {pos}"
    for neg in ("It moved into the API repo on 2026-08-02 and is no longer at the workspace root.",
                "### trading *(2026-07-27 -- resolved 2026-08-02, personal not work)*",
                "plus since 2026-09-18 one row in the registry at `~/.locket/corpora.json`",
                "a gate transition appends one dated clause (Peer ruling 2026-09-11).",
                "Rebrand SHIPPED (v1.1.3), superseding the prior contractor iteration.",
                "thread texture is in `in-flight/closed/architectural-stake-refinement-2026-05-12.md`.",
                "**Target:** SELF.md *Architectural stake* Refinement 2026-05-12, as the mechanism face."):
        assert not HISTORY_CLAUSE.search(neg), f"scoped date matched as history: {neg}"

    # ★ A sentence spanning a hard wrap is ONE claim; line-meaningful markdown is not joined.
    wrapped = ("On any file-op failure the diagnostic order is: wrong machine, unloaded tool, unanswered\n"
               "authorization prompt, then channel health. Channel health is last and has not once been the answer.\n")
    cs = [c for c, _ in claims(strip(wrapped))]
    assert any("unanswered authorization prompt, then channel health" in c for c in cs), cs
    assert not any(c.endswith("unanswered") for c in cs), "the wrap must not end a claim"
    bul = "- first bullet carries no terminal stop at all here\n- second bullet carries none either here\n"
    assert len([c for c, _ in claims(strip(bul))]) == 2, "two bullets stay two claims"
    cont = "- first item that wraps onto\n  a second indented line of the same item\n"
    assert len([c for c, _ in claims(strip(cont))]) == 1, "a wrapped bullet is one claim"
    lead = "The three steps that follow are these\n- measured step one is the first of them\n"
    assert not any("these measured" in c for c in [c for c, _ in claims(strip(lead))]), \
        "a line before a list is not joined to its first item"
    head = "# A heading that carries no stop\nProse that follows the heading directly and at length.\n"
    assert not any("stop Prose" in c for c in [c for c, _ in claims(strip(head))]), \
        "a heading is not joined to the prose under it"
    tbl = "| alpha column | beta column here |\n| gamma column | delta column here |\n"
    assert not any("here gamma" in c for c in [c for c, _ in claims(strip(tbl))]), "table rows stay rows"
    bold = "**Status:** N=1 and explicitly not met\n**Target:** a rule file somewhere\n"
    assert len([c for c, _ in claims(strip(bold))]) <= 2 and \
        not any("met Target" in c for c in [c for c, _ in claims(strip(bold))]), \
        "a line ending in a bolded colon-field value still ends at its stop"
    assert list(unclaimed("Short tail here.\nA long enough sentence to be a claim on its own terms.")) \
        == ["Short tail here."], "the floor's rejects are reportable"

    pairs = cross_file_pairs(docs)
    assert pairs, "identical claim across two files must pair"
    assert {pairs[0][1], pairs[0][2]} == {"a.md", "b.md"}, pairs[0]
    assert pairs[0][0] >= 0.8, pairs[0][0]

    # one shared clause inside otherwise-unrelated files still pairs: this is the
    # case document-level similarity cannot see.
    docs2 = {
        "x.md": "Turbine overhaul programs cover the engine reserve entirely here.\n"
                "The Overview section is exactly two paragraphs long and no more.",
        "y.md": "Silhouette glyphs scale by wingspan across the comparison cards.\n"
                "The Overview section is exactly two paragraphs long and no more.",
    }
    assert cross_file_pairs(docs2), "cross-subject single-clause restatement must pair"

    # across: one claim held by two stores pairs exactly once, and a pair inside
    # one store is audit's, not this mode's.
    two = fixture / "two"
    for store, files in (("alpha", {"m.md": docs["a.md"], "n.md": docs["b.md"]}),
                         ("beta", {"o.md": docs["a.md"] + "\nA sentence only beta holds here today."})):
        (two / store).mkdir(parents=True)
        (two / store / MANIFEST).write_text("{}\n")
        for f, body in files.items():
            (two / store / f).write_text(body)
    _, _, _, cross = across_pairs({"alpha": two / "alpha", "beta": two / "beta"})
    assert {(a.split("::")[0], b.split("::")[0]) for _, a, b, _, _ in cross} == {("alpha", "beta")}, cross
    # alpha/m holds both of beta/o's shared sentences, alpha/n holds one: three pairs
    assert len(cross) == 3, [(a, b) for _, a, b, _, _ in cross]
    # a sentence propagated to three stores, one file each, is the finding; the
    # same sentence in three files of ONE store is furniture and drops the pair.
    spread = "Carry cost structure in curated prose and never the cost level itself."
    (two / "gamma").mkdir(); (two / "gamma" / MANIFEST).write_text("{}\n")
    (two / "gamma" / "g.md").write_text(spread)
    (two / "alpha" / "m.md").write_text(docs["a.md"] + "\n" + spread)
    (two / "beta" / "o.md").write_text(docs["a.md"] + "\n" + spread)
    stores3 = {"alpha": two / "alpha", "beta": two / "beta", "gamma": two / "gamma"}
    _, _, _, cross3 = across_pairs(stores3)
    assert any("gamma::g.md" in (a, b) for _, a, b, _, _ in cross3), "propagated sentence must survive"
    for f in ("q.md", "r.md"):
        (two / "alpha" / f).write_text(spread)
    _, _, _, cross4 = across_pairs(stores3)
    assert not any("gamma::g.md" in (a, b) for _, a, b, _, _ in cross4), "furniture in one store is filtered"
    # a CSV registered by two manifests is ONE file: it never pairs against
    # itself, and a store's own restatement of a row is that store's audit, not
    # this mode's. A holding-space entry whose claim a second store's durable
    # file now holds is tagged, because `graduated` scores one store only.
    row = "A reference ledger registered by two manifests is scored as one file and never as two."
    shared = two / "shared.csv"
    shared.write_text(f"Ruling,Area\n\"{row}\",locket\n")
    reg = json.dumps({"sources": [{"path": str(shared), "text": "Ruling", "label": "Area"}]})
    (two / "alpha" / MANIFEST).write_text(reg)
    (two / "beta" / MANIFEST).write_text(reg)
    (two / "beta" / "o.md").write_text(docs["a.md"] + "\n" + row)
    (two / "alpha" / "SHORT_TERM.md").write_text(docs["a.md"].split("\n")[0])
    settings.cache_clear()
    union5, _, _, cross5 = across_pairs(stores3)
    assert "alpha+beta::shared.csv#locket" in union5, sorted(union5)
    assert not any("shared.csv" in a or "shared.csv" in b for _, a, b, _, _ in cross5), \
        "a shared CSV pairs against neither itself nor a holder's own restatement"
    assert any({a, b} == {"alpha::m.md", "beta::o.md"} for _, a, b, _, _ in cross5), "md pairs still fire"
    grad = graduated_across(cross5, stores3)
    assert [(a, b) for _, a, b, _, _ in grad] == [("alpha::SHORT_TERM.md", "beta::o.md")], grad
    (two / "alpha" / "SHORT_TERM.md").unlink()
    # a manifest directory inside a store is a store of its own: the outer walker
    # leaves its files to it, and a write there resolves to it.
    inner = two / "alpha" / "inner"
    inner.mkdir()
    (inner / MANIFEST).write_text("{}\n")
    (inner / "p.md").write_text(docs["a.md"])
    assert not [f for f in md_files(two / "alpha") if "inner" in str(f)], "nested store must not be walked"
    assert _root_of(inner / "p.md") == inner, "a write inside the nested store belongs to it"

    assert pointer_only("See project doc basis per engine class.")
    assert not pointer_only("The default execution path is the dossier pipeline run interactively.")

    ranked = score_against("The default execution path is the dossier pipeline run interactively.", docs)
    assert ranked[0][2] in ("a.md", "b.md"), ranked[0]
    assert ranked[0][1] >= 0.8, ranked[0][1]
    assert hook_payload_from(docs, "wholly unrelated text about runway lighting and taxiways") is None
    # the hard negative: heavy vocabulary overlap, no restated claim. Gating on the
    # combined score passes this and fires; gating on the claim score does not.
    assert hook_payload_from(docs, "Propeller pitch defaults and hangar insurance quotes differ by "
                                   "execution path, and the interactively run pipeline for hangar "
                                   "rates is unrelated to any default here.") is None

    m = IDX_LINE.match("- [Some Title](feedback_thing.md) — the hook text here")
    assert m and m.group("target") == "feedback_thing.md", m

    found = {(a, int(b)) for a, b in CITE.findall(
        "see src/models/aircraft.py:2493 and results.html:109 and app.js:7-9, "
        "shipped in 4.2.31 at 2026-08-17T03:23:17Z, cost 11.88:2 not a cite")}
    assert found == {("src/models/aircraft.py", 2493), ("results.html", 109),
                     ("app.js", 7)}, found

    src = ["class Thing:", "    def method(self):", "        return 1",
           "", "def top():", "    pass", "", "CONSTANT = 3"]
    assert enclosing_scope(src, 3) == "def method", enclosing_scope(src, 3)
    assert enclosing_scope(src, 6) == "def top", enclosing_scope(src, 6)
    assert enclosing_scope(src, 5) == "def top", enclosing_scope(src, 5)
    assert enclosing_scope(src, 1) == "class Thing", enclosing_scope(src, 1)
    assert enclosing_scope(src, 99) is None
    # the regression: a module-level constant BELOW a def is not inside it.
    assert enclosing_scope(src, 8) is None, enclosing_scope(src, 8)

    # The slug is lossy (dots, dashes and underscores all arrive as `-`), so the
    # decode is only trustworthy if it round-trips through the encoder.
    for name, c in sorted(all_corpora().items()):
        if name in ("council", "home"):
            continue
        d = project_dir_for(c)
        if d is not None:
            assert corpus_for_cwd(d) == c, (name, d)

    # corpus resolution reaches EVERY project, not a named list
    assert corpus_for(COUNCIL / "x.md") == COUNCIL
    # ★ A host's core directory is its council store, its members the corpus.
    # Built here rather than read off ~/.claude, whose layout is one machine's.
    home = (fixture / "home").resolve()     # init resolves; /var is /private/var here
    for rel, body in (("CLAUDE.md", "The constitution at the root is corpus.\n"),
                      ("memory/a.md", "A memory fact that lives in the memory member.\n"),
                      ("rules/r.md", "A rule restating nothing, in the rules member.\n"),
                      ("plugins/p.md", "A vendored plugin page that is host state.\n"),
                      ("sessions/s.md", "A transcript that is host state as well.\n"),
                      ("memory/archive/old.md",
                       "The superseded sentence filed verbatim before the rewrite happened.\n"),
                      ("memory/vault/memfind.json", "{}\n"),
                      ("memory/vault/v.md", "A nested manifest store inside a member.\n")):
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(body)
    import io as _io, contextlib as _ctx
    HOST_STORES["home"] = home
    HOST_MEMBERS[home] = {"memory", "rules"}
    settings.cache_clear()
    assert corpus_for(home / "rules/r.md") == home, "a member directory resolves to the home"
    assert corpus_for(home / "CLAUDE.md") == home, "root-level markdown is corpus"
    assert corpus_for(home / "plugins/p.md") is None, "a non-member directory is host state"
    assert corpus_for(home / "memory/archive/old.md") is None, "archive stays provenance"
    assert corpus_for(home / "memory/vault/v.md") == home / "memory/vault", "a nested store is its own"
    got = {str(q.relative_to(home)) for q in md_files(home)}
    assert got == {"CLAUDE.md", "memory/a.md", "rules/r.md"}, got
    assert any("superseded" in " ".join(ts) for ts in archived_claims(home)), \
        "provenance is read from the archive under a member, not root/archive alone"
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        rc = cmd_init(home, index=False)
    assert rc == 0 and "by convention" in buf.getvalue() and not (home / MANIFEST).exists(), \
        "init on a host home is a no-op that does not refuse itself for the stores under it"
    (home / MANIFEST).write_text(json.dumps({"members": ["memory"]}))
    settings.cache_clear()
    assert corpus_for(home / "rules/r.md") is None, "a manifest's members replaces the host default"
    assert corpus_for(home / "memory/a.md") == home
    (home / MANIFEST).unlink(); settings.cache_clear()
    del HOST_STORES["home"]; del HOST_MEMBERS[home]
    settings.cache_clear()
    for slug in ("-a-notes", "-b-myapp", "-c-brand-new-project"):
        p = CLAUDE / f"projects/{slug}/memory/x.md"
        assert corpus_for(p) == p.parent, slug
    assert corpus_for(COUNCIL / "archive/x.md") is None                       # verbatim by design
    assert corpus_for(COUNCIL / "in-flight/x.md") is None                     # appends until it closes
    assert corpus_for(CLAUDE / "projects/-a/memory/archive/x.md") is None
    assert corpus_for(COUNCIL / "x.txt") is None                              # not markdown
    assert corpus_for("/Users/x/Software/notes/x.md") is None                 # outside the corpus
    assert corpus_for(CLAUDE / "projects/-a/x.md") is None                    # project root, not memory

    big = "x" * 900
    assert added_text({}, big, "") == big                                     # new or empty file
    assert added_text({"old_string": "y" * 800}, big, "y" * 800) is None       # in-place rewrite
    assert added_text({"old_string": "y" * 20}, big, "z") == big               # inserted block
    # an Edit that inserts BETWEEN two anchor lines: both anchors must be subtracted
    anchored = added_text({"old_string": "TOP LINE\nBOTTOM LINE"},
                          "TOP LINE\n" + big + "\nBOTTOM LINE", "z")
    assert "TOP LINE" not in anchored and "BOTTOM LINE" not in anchored, anchored
    # a whole-file Write: an append LEAVES THE OLD BODY INTACT, a reconsolidation
    # does not. This is the dominant append shape at session close.
    prior = "The hangar rate structure is quote-gated on this airframe entirely.\n"
    assert added_text({}, prior + big, prior).strip() == big                   # appended
    assert added_text({}, "wholly rewritten body " + big, prior) is None       # reconsolidated

    # ★ THE HARD CASE, and the reason the containment test alone is not the fix:
    # the prior body is IN the text and is already in the corpus, so scoring the
    # whole text self-matches at 1.00 and names the file being written on every
    # append. Only the delta may reach the scorer.
    docs3 = dict(docs, t=prior)
    # `hook_decision` returns (message, blocking); these cases assert on the
    # message, and the blocking branch has its own case at the end.
    hd = lambda *a, **k: hook_decision(*a, **k)[0]
    novel = ("Turbine hot-section inspections fall due on calendar time as well as hours, "
             "and an owner flying below the annual utilisation assumption pays the "
             "reserve twice over the life of the engine program regardless. " * 2)
    assert hd("t", {}, prior + novel, prior, lambda: docs3) is None, \
        "an append whose DELTA is novel must not fire on its own prior body"
    restated = ("The default execution path is the dossier pipeline run interactively "
                "for every one of these. " * 4)
    assert hd("t", {}, prior + restated, prior, lambda: docs3), \
        "an append whose delta restates another file MUST fire"
    assert hd("MEMORY.md", {}, prior + restated, prior, lambda: docs3) is None
    assert hd("CHECKPOINTS.md", {}, prior + restated, prior, lambda: docs3) is None

    # ★ THE SECOND HARD CASE: an in-place rewrite that EXTENDS what it replaces.
    # A markdown paragraph is one physical line, so the extended paragraph is
    # never byte-identical and survives `added_text`'s anchor subtraction; the
    # replaced span must therefore leave the corpus before scoring.
    # `marked` lives ONLY in doc `t`, so any hit on it is the file matching
    # itself; a retiring sentence another doc also asserts would be a true
    # positive and prove nothing. ⚠ It carries MARKDOWN deliberately, because the
    # corpus holds stripped bodies and this case fails without the strip.
    marked = "The `hangar` rate structure is **quote-gated** on this airframe entirely.\n"
    docs_md = dict(docs, t=strip(marked))
    assert hd("t", {"old_string": marked}, marked + novel, marked,
                         lambda: docs_md) is None, \
        "a rewrite must not be scored against the passage it is retiring"
    # and the guard must still bite: the same rewrite restating ANOTHER file fires.
    # `restated * 2` because an Edit must also clear the 400-character delta gate
    # in `added_text`; a single `restated` sits just under it and would pass this
    # assertion for the wrong reason.
    assert hd("t", {"old_string": prior}, prior + restated * 2, prior,
                         lambda: docs3), \
        "a rewrite whose new material restates another file MUST still fire"

    # a holding space is scored against ITSELF: a repeated entry fires, and an entry
    # restating a fact that lives elsewhere in the corpus does not, because turning
    # over something already known is the surface's job.
    held = ("The incubator is holding the observation that autonomous cycles skip "
            "reconsolidation whenever the surface is already large. ")
    stem = "Promotion target if this holds across further instances: "
    body = held + stem + "Ramp availability shifts seasonally with transient traffic volumes. "
    assert hd("SHORT_TERM.md", {}, body + held * 2, body,
                         lambda: docs3), "a repeated entry in a holding space must fire"
    assert hd("SHORT_TERM.md", {}, body + restated, body,
                         lambda: docs3) is None, "a holding space must not score against the corpus"
    # the file's own furniture must not fire: a stem shared by three entries says
    # nothing about whether a fourth entry is new.
    assert hd("SHORT_TERM.md", {}, body + stem * 3 + stem + novel, body + stem * 3,
                         lambda: docs3) is None, "a shared stem is not a duplicate entry"

    # a dated measurement series is chronology, not repetition: the same sentence
    # carrying a later date and a different figure must be allowed to land, while
    # the undated twin of the same sentence must still fire.
    # ⚠ Each reading must clear the 200-character floor in `hook_decision` on its
    # own, or both assertions below pass for the wrong reason: a short `added`
    # never reaches the scorer, so the carve-out would look correct untested.
    tail = ("bytes on this reading of the ledger growth series, measured against the "
            "previous reading and recorded here so the trend stays visible.\n")
    r1 = ("Measurement update, 2026-08-13 (conscious cycle 312): the checkpoints ledger "
          "reached 300232 " + tail)
    r2 = ("Measurement update, 2026-08-16 (weekly-synthesis cycle 10): the checkpoints ledger "
          "reached 341539 " + tail)
    assert len(r1) > 200 and len(r2) > 200, "readings must clear the hook's length floor"
    assert dated_series_pair(r1, r2) and not dated_series_pair(r1, r1)
    assert hd("OPEN_QUESTIONS.md", {}, body + r1 + r2, body + r1,
                         lambda: docs3) is None, "a later reading in a dated series must not fire"
    undated = r2.replace("2026-08-16", "").replace("341539", "300232")
    assert hd("OPEN_QUESTIONS.md", {}, body + r1 + undated, body + r1,
                         lambda: docs3), "an undated restatement is still a duplicate entry"

    # ★ PROVENANCE: text already filed under archive/ is a MOVE, not a fork, and
    # must not be scored against the source it is moving away from. `moved` lives
    # in doc `t` and is being written into a new file while the source is still
    # live, which is exactly the ordering 4 prescribes and the shape that
    # blocked four legitimate operations before this subtraction existed.
    moved = ("The conscious cycle runs at 09:00 Pacific on the Opus model under a "
             "three dollar ceiling, and the four subconscious cycles follow it at "
             "three-hourly intervals on Sonnet. " * 3)
    docs_mv = dict(docs, t=moved)
    assert hd("new.md", {}, moved, "", lambda: docs_mv) is not None, \
        "control: without provenance the relocation fires against its own source"
    arch = [ts for _, ts in claims(strip(moved))]
    assert hd("new.md", {}, moved, "", lambda: docs_mv, lambda: arch) is None, \
        "a relocation of ARCHIVED text must not fire against its still-live source"
    # ⚠ the archive is read STRIPPED: a filed copy carrying markdown furniture
    # must still match, or the patch no-ops silently while looking like it works.
    arch_md = [ts for _, ts in claims(strip(
        "The **conscious cycle** runs at `09:00` Pacific on the [[opus]] model under a "
        "three dollar ceiling, and the four subconscious cycles follow it at "
        "three-hourly intervals on Sonnet.\n"))]
    assert hd("new.md", {}, moved, "", lambda: docs_mv, lambda: arch_md) is None, \
        "archived text carrying markdown must still subtract"
    # ★ and the guard must not over-subtract: a move carrying one NEW claim that
    # restates another file still fires on that claim alone.
    assert hd("new.md", {}, moved + restated * 2, "", lambda: docs3, lambda: arch), \
        "a mixed write must still fire on the material that is not archived"
    # an unarchived duplicate is untouched by provenance
    assert hd("t", {}, prior + restated, prior, lambda: docs3, lambda: arch), \
        "provenance must not excuse a duplicate that was never archived"

    # ★ a dated history clause on a durable surface fires on the shape alone, warns
    # supervised, blocks a cycle, and never fires on a holding space or a ledger.
    stack = ("Some durable entry about hangar rates and the quote-gated structure of them.\n"
             "⚠ UPDATE (2026-08-08): the endpoint moved to the API repo and the rates changed.\n") * 2
    _m, _b = hook_decision("entry.md", {}, stack, "", lambda: {})
    assert _m and "dated history clause" in _m and _b is False, _m
    _m, _b = hook_decision("entry.md", {}, stack, "", lambda: {}, autonomous=True)
    assert _m and _b is True, "a cycle stacking a dated clause is blocked"
    scoped = ("Rebrand SHIPPED (v1.1.3), superseding the prior contractor iteration entirely.\n"
              "It moved into the API repo on 2026-08-02 and is no longer at the workspace root.\n") * 2
    _m, _b = hook_decision("entry.md", {}, scoped, "", lambda: {})
    assert _m and "dated history clause" not in _m and "writes a date" in _m and _b is False, \
        f"a scoped date is not a history clause, and is still asked about: {_m}"
    _m, _b = hook_decision("PROMOTIONS.md", {}, stack, "", lambda: {})
    assert not (_m and "dated history clause" in _m), "a holding space takes its dated transition"
    _m, _b = hook_decision("HARNESS.md", {}, stack, "", lambda: {})
    assert _m is None, "a ledger is never checked"
    # ★ any other date on a durable surface is asked about: warns supervised, never blocks a
    # cycle, skips a dated heading, and is silent on a holding space or a ledger.
    attributed = ("The trim keeps only rules that generalize to any store (Peer ruling 2026-09-20).\n"
                  "Its apparatus clauses move to a Prime-only rule and the file ships to the Partial.\n") * 2
    _m, _b = hook_decision("entry.md", {}, attributed, "", lambda: {})
    assert _m and "writes a date" in _m and _b is False, _m
    _m, _b = hook_decision("entry.md", {}, attributed, "", lambda: {}, autonomous=True)
    assert _m and "writes a date" in _m and _b is False, "a bare date never blocks a cycle"
    headed = ("## Session 33, 2026-09-20\n"
              "The build passes and the widget target is unchanged since the convention audit.\n") * 2
    _m, _b = hook_decision("entry.md", {}, headed, "", lambda: {})
    assert _m is None, f"a dated heading is convention: {_m}"
    _m, _b = hook_decision("PROMOTIONS.md", {}, attributed, "", lambda: {})
    assert not (_m and "writes a date" in _m), "a holding space dates its transitions"
    _m, _b = hook_decision("HARNESS.md", {}, attributed, "", lambda: {})
    assert _m is None, "a ledger is never checked for dates"

    # ★ the supervised/autonomous branch: same finding, different force. A cycle
    # cannot read a warning, so its guard stays hard; a session can, so it warns.
    _m, blocking = hook_decision("t", {}, prior + restated, prior, lambda: docs3)
    assert _m and blocking is False, "a supervised nudge must warn, not block"
    _m, blocking = hook_decision("t", {}, prior + restated, prior, lambda: docs3,
                                 autonomous=True)
    assert _m and blocking is True, "an autonomous nudge must still block"
    assert "about to be written" in _m and "landed" not in _m, _m
    _m, _b = hook_decision("t", {}, prior + restated, prior, lambda: docs3)
    assert _m.startswith(LEAD_WARNING), "the warn channel must not claim the write was stopped"
    # ⚠ `main` re-leads a warning into the blocking form when the 2e gate refuses
    # the same write. That swap is a literal replace, so it silently no-ops if the
    # two leads ever drift; this is the assertion that keeps them in step.
    assert _m.replace(LEAD_WARNING, LEAD_BLOCKING, 1).startswith(LEAD_BLOCKING), \
        "the blocking re-lead must actually match what the warn channel emits"

    # within-file pairs are the uniqueness check and are invisible to `audit`
    dup = {"h.md": held * 2 + "\nUnrelated line about ramp availability and transient traffic.\n" + held}
    assert cross_file_pairs(dup, same_file=True), "repeated claim inside one file must pair"
    assert not cross_file_pairs(dup), "the same file must not pair against itself cross-file"

    # the walker and the write-time gate must cover the same files, or a
    # subdirectory is nudged against a corpus it is not a member of
    for d in EXCLUDED_DIRS:
        assert corpus_for(COUNCIL / f"{d}/x.md") is None, d

    # --- the §2e question gate -------------------------------------------------
    # Driveable without ollama: an unreachable embedder degrades the message and
    # never changes the decision, which is the property that lets it block at all.
    d = Path(tempfile.mkdtemp())
    (d / "asked.jsonl").write_bytes(
        b'{"message":{"content":[{"type":"tool_use","name":"Bash",'
        b'"input":{"command":"python3 ~/.claude/scripts/memfind.py \\"x\\" council"}}]}}\n')
    (d / "discussed.jsonl").write_bytes(
        b'{"message":{"content":[{"type":"text","text":"memfind.py ranks candidate owners"}]}}\n')
    assert memfind_asked(str(d / "asked.jsonl")), "a Bash invocation must count as asked"
    assert not memfind_asked(str(d / "discussed.jsonl")), \
        "prose naming the script must NOT count; a session doing memory work names it constantly"
    assert memfind_asked(str(d / "absent.jsonl")), "an unreadable transcript must not block"
    assert memfind_asked(None)

    # ★ POSITION, not presence. Every string below contains the script name; only
    # the first group runs it. The `printf` case is the one that was live: it wrote
    # a test fixture containing the name and discharged a whole session's gate.
    for cmd in ('python3 ~/.claude/scripts/memfind.py "q" council',
                '~/.claude/scripts/memfind.py "q"',
                'locket find "q" council',
                'python3 ~/.claude/scripts/locket.py find "q"',
                'cd /tmp && locket find "q" | head',
                'cd /tmp && python3 scripts/memfind.py "q" memory',
                'python3 -X faulthandler memfind.py "q"',
                'PYTHONPATH=. python3 memfind.py "q"',
                'memfind.py "q" | head -20'):
        assert _runs_memfind(cmd), cmd
    for cmd in ('grep -n memfind.py scripts/*.py',
                'locket index council',
                'locket audit',
                'echo "locket find q"',
                'python3 ~/.claude/scripts/locket.py --index notes',
                'cat ~/.claude/scripts/memfind.py',
                'ls -l memfind.py',
                'sed -n "1,40p" memfind.py',
                'wc -l memfind.py memscan.py',
                'printf \'{"command":"python3 memfind.py x"}\' > fixture.jsonl',
                'echo "run memfind.py next"',
                '\'~/.claude/scripts/memfind.py "q"\',',
                'python3 - <<\'PY\'\npython3 scripts/memfind.py "q" x\nPY',
                'python3 -c "import re; print(\'memfind.py\')"'):
        assert not _runs_memfind(cmd), cmd

    (d / "mentioned.jsonl").write_bytes(
        b'{"message":{"content":[{"type":"tool_use","name":"Bash","input":'
        b'{"command":"grep -n memfind.py ~/.claude/scripts/memscan.py"}}]}}\n')
    assert not memfind_asked(str(d / "mentioned.jsonl")), \
        "a Bash command that only NAMES the script must not discharge the gate"

    # ⚠ fail-open on an unrecognised schema. A false negative here cannot be
    # cleared by asking, because Write and Edit carry no MEMWRITE-OK equivalent,
    # so a transcript shape this function cannot read must never block.
    (d / "future-schema.jsonl").write_bytes(
        b'{"envelope":{"parts":[{"tool":"Bash","args":'
        b'{"cmd":"python3 memfind.py q"}}]}}\n')
    assert memfind_asked(str(d / "future-schema.jsonl")), \
        "a marker this parser cannot structure must fail open, not block"

    unasked = {"transcript_path": str(d / "discussed.jsonl")}
    root = COUNCIL
    assert memfind_gate(unasked, "t.md", {}, novel * 3, "", root), \
        "net-new material with the question unasked must fire"
    assert memfind_gate({"transcript_path": str(d / "asked.jsonl")}, "t.md", {}, novel * 3, "",
                        root) is None, "asking the question once must silence the gate"
    for exempt in ("CHECKPOINTS.md", "SHORT_TERM.md"):
        assert memfind_gate(unasked, exempt, {}, novel * 3, "", root) is None, exempt
    assert memfind_gate(unasked, "MEMORY.md", {}, novel * 3, "", root), \
        "an index file is exempt from the lexical nudge, never from the question"
    assert memfind_gate(unasked, "t.md", {}, novel, novel, root) is None, \
        "a reconsolidation adds nothing and must not fire"
    os.environ["CLAUDE_AUTONOMOUS"] = "1"
    assert memfind_gate(unasked, "t.md", {}, novel * 3, "", root) is None, \
        "a cycle cannot act on a nudge and must never be blocked by one"
    del os.environ["CLAUDE_AUTONOMOUS"]
    shutil.rmtree(d, ignore_errors=True)
    # --- bashguard: fires on the thing, silent on every near-miss -------------
    # a command naming a scratch path is exempt by design, so under a HOME in
    # /tmp (Homebrew's test sandbox) every absolute-path case would pass silently
    if not any(m in str(CLAUDE) + "/" for m in SCRATCH_MARKERS):
        cmem = str(CLAUDE / "projects" / "-Users-x-proj" / "memory")
        def bw(cmd, cwd=cmem, auto=False):
            return bash_write_decision(cmd, cwd, autonomous=auto)
        assert bw("cat > MEMORY.md <<EOF")[1], "a redirect into a corpus file must block"
        assert bw("sed -i '' 's/a/b/' MEMORY.md")[1], "sed -i must block"
        assert bw(f"echo x >> {cmem}/topic_note.md")[1], "an absolute corpus path must block"
        rel = Path(cmem).relative_to(HOME)
        for spelt in (f"~/{rel}", f"$HOME/{rel}", f"${{HOME}}/{rel}", f'"$HOME/{rel}"', f"'~/{rel}'"):
            assert bw(f"echo x >> {spelt}/topic_note.md")[1], f"a home-relative path must block: {spelt}"
        assert bw(f"cd ~/{rel} && echo x >> topic_note.md")[1], "a cd through ~ resolves a bare name"
        assert bw(f"cd ~nosuchuser9/x && echo x >> {cmem}/topic_note.md")[1], \
            "an unknown ~user elsewhere in the command must not disable the check"
        msg, blk = bw("python3 - <<'EOF'\npathlib.Path('MEMORY.md').write_text(s)\nEOF")
        assert msg and not blk, "a scripted write warns rather than blocks"
        for quiet in (
            "cat MEMORY.md",                               # a read
            "grep -n foo MEMORY.md",                       # a read
            "cp MEMORY.md /private/tmp/scratchpad/x.bak",  # backup OUT of the corpus
            "cat > CHECKPOINTS.md",                        # ledger, exempt by 2d
            "cat > SHORT_TERM.md",                         # holding space, exempt
            "cat > archive/ARCHIVE-old.md",                # archived, exempt
        ):
            assert bw(quiet) == (None, False), f"decoy fired: {quiet}"
        assert bw("cat > MEMORY.md", auto=True) == (None, False), \
            "an autonomous cycle must never be blocked (2d: it cannot report the failure)"
        assert bw("cat > MEMORY.md  # MEMWRITE-OK: scripted multi-file patch") == (None, False), \
            "a declared escape must pass"
        # Resolving rather than name-matching: the SAME command is a corpus write from
        # inside a corpus and nothing at all from outside one.
        assert bw("cat > README.md")[1], "a .md written into a corpus dir is a corpus write"
        assert bw("cat > README.md", cwd="/Users/x/some-repo") == (None, False), \
            "the same command outside a corpus must stay silent"

    # --- grepassist: only a question, never a regex ---------------------------
    assert prose_like("workspace root is a git repository")
    assert not prose_like("waive|prototype"), "an alternation is not a question"
    assert not prose_like("v_ne\\b"), "a metacharacter disqualifies"
    assert not prose_like("repo"), "one content word is not a question"
    assert not prose_like(""), "an empty pattern must not fire"

    # --- portability: a manifest makes any directory a corpus -----------------
    d = Path(tempfile.mkdtemp())
    store = d / "brain"
    (store / "old").mkdir(parents=True)
    (store / "LOG.md").write_text("entry one\n")
    (store / "facts.md").write_text("A fact.\n")
    assert corpus_for(store / "facts.md") is None, "no manifest, no host store: not a corpus"
    (store / MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "ledger_surfaces": ["LOG.md"],
        "holding_spaces": ["INBOX.md"], "index_files": [],
        "code_roots": [str(d)]}))
    settings.cache_clear()
    assert corpus_for(store / "facts.md") == store, "a manifest names the corpus"
    assert corpus_for(store / "old" / "x.md") is None, "the manifest's excluded dir holds"
    assert corpus_for(store / "archive" / "x.md") == store, \
        "a manifest REPLACES the excluded set; `archive` is ours, not theirs"
    s = settings(store)
    assert s["ledger_surfaces"] == {"LOG.md"} and "INBOX.md" in s["append_surfaces"], s
    assert ".memfind" in s["excluded_dirs"], "the cache dir is excluded whatever the manifest says"
    assert [r for r, _ in code_roots(store)] == [d], "declared code roots win outright"
    assert md_files(store) == [store / "LOG.md", store / "facts.md"], md_files(store)
    assert bash_write_decision("cat > LOG.md", str(store)) == (None, False), \
        "a ledger named by the manifest is exempt in bashguard too"
    assert bash_write_decision("cat > facts.md", str(store))[1], \
        "a plain file in a manifest corpus blocks"
    assert "old" in settings(store)["excluded_dirs"] and "backups" in settings(store)["excluded_dirs"], \
        "a backup directory is excluded whatever the manifest says"
    (store / ".hidden").mkdir(); (store / ".hidden" / "h.md").write_text("A hidden note.\n")
    assert store / ".hidden" / "h.md" not in md_files(store), "dot-directories are never corpus"
    # an index in a member subdirectory points beside itself, not at the root
    (store / "sub").mkdir(); (store / "sub" / "t.md").write_text("A target.\n")
    (store / "sub" / "MEMORY.md").write_text("- [T](t.md) the target\n- [G](gone.md) removed\n")
    (store / MANIFEST).write_text(json.dumps({"excluded_dirs": ["old"], "index_files": ["MEMORY.md"]}))
    settings.cache_clear()
    ih = {h[1]: h[4].exists() for h in index_hooks(store)}
    assert ih == {"sub/t.md": True, "sub/gone.md": False}, ih
    shutil.rmtree(store / "sub")

    # --- the CSV boundary: registered rows are IN the corpus --------------------
    (store / "RULES.csv").write_text(
        "Ruling,Area,Date\n"
        '"Serve the estimate bottom-up from priced components and never from an aggregate.",cost,2026-01-01\n'
        '"A rebase fights the watcher, so build the sheet and commit it before rebasing.",dev-loop,2026-01-02\n')
    (store / MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "index_files": [], "ledger_surfaces": ["LOG.md"],
        "holding_spaces": ["INBOX.md"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    settings.cache_clear()
    c = corpus(store)
    assert "RULES.csv#cost" in c and "RULES.csv#dev-loop" in c, sorted(c)
    assert "RULES.csv#cost" not in corpus(store, csv=False), "memfind takes rows as its own units"
    (store / "notes.md").write_text(
        "Serve the estimate bottom-up from priced components and never from an aggregate.\n")
    pairs = cross_file_pairs(corpus(store))
    assert any({fa, fb} == {"notes.md", "RULES.csv#cost"} for _, fa, fb, _, _ in pairs), \
        "a CSV row copied into prose is a fork the audit must see"
    # and the reverse at write time: a new row restating prose warns, never blocks
    rules_prior = (store / "RULES.csv").read_text()
    row = '"Serve the estimate bottom-up from priced components, never from an aggregate.",pricing,2026-01-03\n'
    msg, blocking = source_row_decision(str(store / "RULES.csv"), {"new_string": row}, rules_prior, append=True)
    assert msg and "notes.md" in msg and not blocking, msg
    assert source_row_decision(str(store / "RULES.csv"), {"content": rules_prior}, rules_prior) == (None, False), \
        "a rewrite adding no row draws nothing"
    edit = {"old_string": "never from an aggregate", "new_string": "not from an aggregate"}
    msg = source_row_decision(str(store / "RULES.csv"), edit, rules_prior)[0] or ""
    assert "- RULES.csv " not in msg, "an edited row never matches the version it replaces"
    msg, _ = hook_decision("new.md", {}, "A rebase fights the watcher, so build the sheet and commit it "
                           "before rebasing. " * 3, "", lambda: corpus(store), lambda: [], sets=settings(store))
    assert msg and "RULES.csv#dev-loop" in msg, msg
    # query pointers: the [[wikilink]] of a ruling, checked against the rows
    (store / "guide.md").write_text(
        "Query `RULES.csv` on `Area=cost` before pricing anything.\n"
        "The rebase ruling (`RULES.csv`, query `Area=dev-loop|frontend`) governs the sheet.\n"
        "See `OTHER.csv` (query `Area=cost`) for the rest.\n"
        "Never query `RULES.csv` on `Zone=cost`, which is the old column.\n"
        "The file `RULES.csv` holds `Area=cost` rows, but this sentence is not a pointer.\n")
    st = {(x[1], x[2], x[3]): x[4] for x in csv_pointers(store)}
    assert st[("RULES.csv", "Area", "cost")] == "ok", st
    assert st[("RULES.csv", "Area", "dev-loop|frontend")] == "DANGLING frontend", st
    assert st[("OTHER.csv", "Area", "cost")] == "UNREGISTERED", st
    assert st[("RULES.csv", "Zone", "cost")] == "NO COLUMN", st
    assert len(csv_pointers(store)) == 4, "a mention without the word query is not a pointer"

    # --- graduated: a holding-space claim that a durable source now holds -------
    import contextlib, io
    (store / MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "index_files": [], "ledger_surfaces": ["LOG.md"],
        "holding_spaces": ["INBOX.md", "QUEUE.md"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    settings.cache_clear()
    (store / "INBOX.md").write_text(
        "A rebase fights the watcher, so build the sheet and commit it before rebasing.\n"
        "Gate: a third instance where the watcher ordering bites on a freshly built sheet.\n")
    (store / "QUEUE.md").write_text(
        "Gate: a third instance where the watcher ordering bites on a freshly built sheet.\n")
    (store / "LOG.md").write_text(
        "entry one\nGate: a third instance where the watcher ordering bites on a freshly built sheet.\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = cmd_graduated(store)
    txt = buf.getvalue()
    assert n == 1 and "INBOX.md" in txt and "RULES.csv#dev-loop" in txt, txt
    body = txt.split("\n\n", 1)[1]
    assert "QUEUE.md" not in body, "a holding-to-holding pair is a ledger restating a gate, not a graduation"
    assert "LOG.md" not in body, "a ledger entry quoting a gate is chronology, not a graduation"
    assert "2 holding files (INBOX.md, QUEUE.md) against" in txt, txt
    (store / "INBOX.md").unlink(); (store / "QUEUE.md").unlink()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cmd_graduated(store) == 0
    assert "none of the configured holding spaces" in buf.getvalue(), \
        "a clean run still says what it was scored against"
    (store / MANIFEST).write_text("{not json")
    settings.cache_clear()
    assert settings(store)["ledger_surfaces"] == LEDGER_SURFACES, \
        "a broken manifest degrades to the defaults rather than taking the hook down"
    (store / MANIFEST).unlink()
    settings.cache_clear()
    assert resolve(str(store)).is_absolute(), "a literal path resolves absolute"

    # --- excluded_files: one file ignored at both ends --------------------------
    (store / "notes").mkdir()
    (store / "notes" / "draft.md").write_text("A draft fact nobody should score against.\n")
    (store / "export.md").write_text("An exported artifact that is not memory.\n")
    (store / MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "index_files": ["facts.md"], "ledger_surfaces": ["LOG.md"],
        "excluded_files": ["notes/draft.md", "export*"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    settings.cache_clear()
    got = md_files(store)
    assert store / "notes" / "draft.md" not in got, "a relative path excludes that file"
    assert store / "export.md" not in got, "a glob excludes by basename"
    assert store / "facts.md" in got and store / "guide.md" in got, got
    assert corpus_for(store / "notes" / "draft.md") is None, "the gate skips an excluded file too"
    assert corpus_for(store / "facts.md") == store
    assert csv_sources(store), "a registered CSV that no pattern names stays a source"
    (store / MANIFEST).write_text(json.dumps({
        "excluded_files": ["*.csv"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    settings.cache_clear()
    assert csv_sources(store) == [], "an excluded CSV is not a source even when registered"
    assert "RULES.csv#cost" not in corpus(store)
    (store / MANIFEST).write_text("{}\n"); settings.cache_clear()

    # --- init and the registry: a manifest directory the listing can find -----
    saved, globals()["REGISTRY"] = REGISTRY, d / "locket" / "corpora.json"
    try:
        vault = d / "vault"; (vault / "sub").mkdir(parents=True)
        vault = vault.resolve()             # the registry stores resolved paths; /var is /private/var here
        (vault / "one.md").write_text("A note in a folder nobody has registered.\n")
        assert registry() == [] and vault not in all_corpora().values()
        assert corpus_here(vault) is None, "no manifest, no auto-memory: the default has nothing"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_init(vault, index=False)
        assert rc == 0 and (vault / MANIFEST).read_text() == "{}\n", buf.getvalue()
        assert registry() == [vault] and vault in all_corpora().values(), \
            "init makes the listing see a manifest directory"
        assert corpus_here(vault) == vault and corpus_here(vault / "sub") == vault, \
            "the cwd default finds a manifest directory from anywhere inside it"
        assert corpus_for(vault / "sub" / "x.md") == vault
        assert not register(vault), "registering twice is a no-op"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cmd_init(vault, index=False) == 0
        assert "(kept)" in buf.getvalue() and "already listed" in buf.getvalue(), buf.getvalue()
        with contextlib.redirect_stderr(io.StringIO()):
            assert cmd_init(vault / "sub", index=False) == 1, "a directory inside a corpus is refused"
        # and so is a directory CONTAINING one, whether the inner store is listed
        # or only carries a manifest by hand; the parent is left untouched.
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert cmd_init(d, index=False) == 1, "a directory containing a corpus is refused"
        assert "contains the corpus" in err.getvalue() and not (d / MANIFEST).exists(), err.getvalue()
        hand = d / "hand"; hand.mkdir(); (hand / MANIFEST).write_text("{}\n")
        assert forget(vault)
        with contextlib.redirect_stderr(io.StringIO()):
            assert cmd_init(d, index=False) == 1, "an unlisted manifest below is found too"
        assert not (d / MANIFEST).exists()
        (hand / MANIFEST).unlink(); hand.rmdir()
        assert register(vault)
        (vault / MANIFEST).write_text(json.dumps({"name": "vault-of-one"})); settings.cache_clear()
        assert all_corpora().get("vault-of-one") == vault, "the manifest's name is the listing key"
        assert resolve("vault-of") == vault, "and a substring of it resolves"
        (vault / MANIFEST).unlink(); settings.cache_clear()
        assert vault not in all_corpora().values() and registry_stale() == [vault], \
            "a registered path whose manifest is gone is reported, never enumerated"
        assert any("is gone" in l for l in corpora_lines()), corpora_lines()
        assert forget(vault) and not forget(vault) and registry() == [], "forget drops the row once"
        assert not any("is gone" in l for l in corpora_lines())
        with contextlib.redirect_stderr(io.StringIO()):
            assert cmd_init(vault / "one.md", index=False) == 1, "a file is not a corpus"
        hostlike = (d / "hostlike").resolve(); hostlike.mkdir()
        (hostlike / "a.md").write_text("A host store's note.\n")
        HOST_STORES["_selftest"] = hostlike        # a store listed on sight, for this block only
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert cmd_init(hostlike, index=False) == 0
            assert "by convention" in buf.getvalue() and registry() == [] \
                and not (hostlike / MANIFEST).exists(), \
                "a host store is listed on sight, so init writes it no manifest and no row"
        finally:
            del HOST_STORES["_selftest"]
    finally:
        globals()["REGISTRY"] = saved

    # --- belongs_to: two folders in two places, one corpus ------------------
    saved, globals()["REGISTRY"] = REGISTRY, d / "locket2" / "corpora.json"
    try:
        par = (d / "par").resolve(); (par / "docs").mkdir(parents=True)
        (par / "IMPL.md").write_text("The impl points at [[topic_one]] and [[topic_two]].\n"
                                     "See `RULES.csv` query `Area=cost` for the ruling.\n")
        (par / "docs" / "d.md").write_text("A doc under the parent's member directory.\n")
        (par / "RULES.csv").write_text("Area,Ruling\ncost,The hangar rate is quote-gated on every airframe.\n")
        (par / MANIFEST).write_text(json.dumps({
            "name": "par", "members": ["docs"], "ledger_surfaces": ["LOG.md"],
            "whole_read_exempt": ["IMPL.md"],
            "sources": [{"path": "RULES.csv", "text": "Ruling", "label": "Area"}]}))
        settings.cache_clear(); register(par)
        kid = (d / "elsewhere" / "memory").resolve(); (kid / "archive").mkdir(parents=True)
        (kid / "topic_one.md").write_text("Topic one holds a durable fact about hangars.\n")
        (kid / "SHORT.md").write_text("- an entry\n")
        (kid / "LOG.md").write_text("2026-01-01 a ledger line\n")
        (kid / "archive" / "old.md").write_text("A superseded sentence kept as provenance here.\n")
        (kid / LEGACY_MANIFEST).write_text(json.dumps(
            [{"path": str(par / "RULES.csv"), "text": "Ruling", "label": "Area"}]))
        (kid / MANIFEST).write_text(json.dumps({"name": "kid", "holding_spaces": ["SHORT.md"]}))
        settings.cache_clear(); register(kid)
        assert {"par", "kid"} <= set(all_corpora()), "before joining, both are listed"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cmd_init(kid, index=False, parent="par") == 0, buf.getvalue()
        settings.cache_clear()
        assert _read_manifest(kid)["belongs_to"] == "par" and _read_manifest(kid)["name"] == "kid", \
            "init --parent adds belongs_to and keeps the rest of the manifest"
        assert "kid" not in all_corpora() and all_corpora()["par"] == par, \
            "a joined store is listed through its parent alone"
        assert parent_of(kid) == par and top_of(kid) == par and top_of(par) == par
        assert roots_of(par) == [par, kid]
        assert corpus_for(kid / "topic_one.md") == par, "a write into the joined store is scored at the parent"
        assert corpus_for(kid / "archive" / "old.md") is None, "the joined store's own exclusions hold"
        assert corpus_here(kid) == par and resolve(str(kid)) == par and resolve("par") == par
        got = {corpus_key(par, q) for q in md_files(par)}
        assert got == {"IMPL.md", "docs/d.md", "kid/LOG.md", "kid/SHORT.md", "kid/topic_one.md"}, got
        assert set(corpus(par)) >= {"kid/topic_one.md", "RULES.csv#cost"}, "keys are composite; one CSV, once"
        assert len(csv_sources(par)) == 1, "a CSV registered by both stores is one source"
        st = settings(par)
        assert "SHORT.md" in st["holding_spaces"] and "LOG.md" in st["ledger_surfaces"] \
            and "SHORT.md" in st["append_surfaces"], "name-sets union upward"
        assert "SHORT.md" not in settings(kid)["ledger_surfaces"], "and never downward"
        assert st["whole_read_exempt"] == {"IMPL.md"} and settings(kid)["whole_read_exempt"] == set(), \
            "the size-sweep exemption is a name-set like the others: read, unioned upward, never down"
        assert not dangling(par).get("topic_one") and dangling(par).get("topic_two"), \
            "a wikilink into the joined store resolves; a missing one still dangles"
        assert any("superseded" in " ".join(ts) for ts in archived_claims(par)), \
            "provenance is read from the joined store's archive too"
        assert any(s[4] == "ok" for s in csv_pointers(par)), csv_pointers(par)
        assert any("[1 joined store]" in l for l in corpora_lines()), corpora_lines()
        # refusals: unknown parent, self, a store nesting on disk, a cycle
        for bad in ("nobody", "kid"):
            with contextlib.redirect_stderr(io.StringIO()):
                assert cmd_init(kid, index=False, parent=bad) == 1, bad
        sub = par / "docs"; (sub / MANIFEST).write_text("{}\n"); settings.cache_clear()
        with contextlib.redirect_stderr(io.StringIO()):
            assert cmd_init(sub, index=False, parent="par") == 1, "a nested directory is a member, not a join"
        (sub / MANIFEST).unlink(); settings.cache_clear()
        (par / MANIFEST).write_text(json.dumps({"name": "par", "belongs_to": "kid"}))
        settings.cache_clear()
        assert parent_of(par) is None and parent_of(kid) is None, "a cycle leaves both standing alone"
        assert {"par", "kid"} <= set(all_corpora())
        (par / MANIFEST).write_text(json.dumps({"name": "par"})); settings.cache_clear()
        assert parent_of(kid) == par
        # nested on disk but outside the parent's members: a host home's own
        # auto-memory under projects/ is the case, and it joins
        (par / MANIFEST).write_text(json.dumps({"name": "par", "members": ["docs"]})); settings.cache_clear()
        out = par / "projects" / "m"; out.mkdir(parents=True)
        (out / "o.md").write_text("A nested store outside the parent's members joins it.\n")
        (out / MANIFEST).write_text(json.dumps({"name": "outm"})); settings.cache_clear()
        register(out)                          # a real auto-memory is listed on sight
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert cmd_init(out, index=False, parent="par") == 0, "a store outside the members joins"
        assert parent_of(out) == par and corpus_for(out / "o.md") == par
        assert [q.name for q in md_files(par)].count("o.md") == 1, "counted once, as a joined store"
        # a joining folder called like one of the parent's directories needs a label
        am = (d / "host" / "projects" / "slug" / "memory").resolve(); am.mkdir(parents=True)
        (am / "a.md").write_text("An auto-memory fact about the council store here.\n")
        (am / MANIFEST).write_text("{}\n"); settings.cache_clear(); register(am)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            assert cmd_init(am, index=False, parent="par", name="docs") == 1, "a label naming a parent dir"
            assert cmd_init(am, index=False, parent="par", name="a/b") == 1, "a label with a slash"
        assert "--name <label>" in err.getvalue(), err.getvalue()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert cmd_init(am, index=False, parent="par", name="auto-memory") == 0
        assert _read_manifest(am) == {"name": "auto-memory", "belongs_to": "par"}, _read_manifest(am)
        assert corpus_key(par, am / "a.md") == "auto-memory/a.md"
        assert forget(am)
        assert forget(out)
        assert forget(par) and forget(kid)
    finally:
        globals()["REGISTRY"] = saved
        settings.cache_clear()

    # --- the manifest schema: hints for the hand that writes one ---------------
    sd = (d / "schemadir").resolve(); sd.mkdir()
    assert validate_manifest(sd) == [], "no manifest, nothing to say"
    (sd / MANIFEST).write_text(json.dumps({
        "name": "s", "members": ["docs"], "belongs_to": "x",
        "sources": [{"path": "R.csv", "text": "Ruling", "label": "Area"}],
        "ledgers": [{"path": "L.csv", "caps": {"note": 10}, "enum": {"k": ["a", ""]}}]}))
    assert validate_manifest(sd) == [], validate_manifest(sd)
    (sd / MANIFEST).write_text(json.dumps({
        "belongs-to": "x", "members": "docs", "sources": [{"text": "Ruling"}],
        "ledgers": [{"path": "L.csv", "caps": {"note": "ten"}, "extra": 1}]}))
    f = validate_manifest(sd)
    assert any("unknown key 'belongs-to', did you mean 'belongs_to'?" in x for x in f), f
    assert any("members: expected array" in x for x in f), f
    assert any("sources[0]: missing required key 'path'" in x for x in f), f
    assert any("ledgers[0].caps.note: expected integer" in x for x in f), f
    assert any("ledgers[0]: unknown key 'extra'" in x for x in f), f
    (sd / MANIFEST).write_text("{not json")
    assert validate_manifest(sd) and "does not parse" in validate_manifest(sd)[0]
    (sd / MANIFEST).unlink()
    assert json.loads(json.dumps(MANIFEST_SCHEMA))["properties"].keys() >= {
        "belongs_to", "ledgers", "sources", "members"}, "the schema names the keys the reader takes"

    # --- ledger schemas: the one decidable refusal ---------------------------
    led = (d / "proj").resolve(); led.mkdir()
    hist = led / "HISTORY-Proj-Sessions.csv"
    hdr = "date,title,summary,area,repo,commit_hash,issue,pr\n"
    good = '2026-09-20,Composite corpus,"Two stores joined into one, links resolved.",api,grp/sub/proj,5ab427a,ENG-42,#155\n'
    hist.write_text(hdr + good)
    sch, glob = ledger_schema_for(hist)
    assert sch is not None and glob == "HISTORY-*-Sessions.csv", "the built-in governs the file outside any store"
    assert ledger_schema_for(led / "notes.csv") == (None, None)
    assert ledger_decision(str(hist), {"content": hdr + good}) == (None, False), "a clean whole-file write passes"
    assert ledger_decision(str(hist), {"content": hdr}) == (None, False), "a header alone passes"
    long_row = f'2026-09-20,{"t" * 91},{"s" * 241},ios,a b,zz,two words,155\n'
    msg, blocking = ledger_decision(str(hist), {"content": hdr + good + long_row})
    assert blocking and "row 3: title is 91 characters, cap 90" in msg \
        and "summary is 241 characters, cap 240" in msg and "commit_hash is 'zz'" in msg and "issue is 'two words'" in msg and "pr is '155'" in msg \
        and "repo is 'a b'" in msg, msg
    mr = good.replace("#155", "!155")
    assert ledger_decision(str(hist), {"content": hdr + mr}) == (None, False), "a GitLab MR is a pr"
    msg, blocking = ledger_decision(str(hist), {"content": "date,title\n2026-01-01,x\n"})
    assert blocking and "header is date,title" in msg, "a header off the schema is one finding"
    msg, blocking = ledger_decision(str(hist), {"old_string": "x", "new_string": long_row})
    assert blocking and "row 1: title is 91" in msg, "an Edit fragment is checked against the header on disk"
    assert ledger_decision(str(hist), {"old_string": "x", "new_string": "Composite"}) == (None, False), \
        "a fragment that is not a row is not judged"
    assert ledger_decision(str(hist), {"old_string": "x", "new_string": ",,,,,,,\n"}) == (None, False), \
        "an empty row is a blank line"
    msg, blocking = ledger_decision(str(hist), {"old_string": "x", "new_string": "2026-09-20,,ok,,,,,\n"})
    assert blocking and "title is required" in msg, msg
    for dt in ("2026-09-23T03:12Z", "2026-09-23T03:12:45Z"):
        assert ledger_decision(str(hist), {"old_string": "x", "new_string": f"{dt},t,s,,,,,\n"}) == (None, False), \
            f"a UTC instant is a date: {dt}"
    for dt in ("2026-09-23T03:12", "2026-09-23T03:12:45-07:00", "2026-09-23 03:12Z"):
        msg, blocking = ledger_decision(str(hist), {"old_string": "x", "new_string": f"{dt},t,s,,,,,\n"})
        assert blocking and "date" in msg, f"only a Z-suffixed instant passes: {dt}"
    # bashguard: heredoc rows and echo rows are read; an opaque append warns
    msg, blocking = ledger_bash_decision(f"cat >> {hist} <<'EOF'\n{long_row}EOF\n", None)
    assert blocking and "bashguard" in msg and "cap 90" in msg, msg
    msg, blocking = ledger_bash_decision(f'echo "{good.strip()}" >> {hist}', None)
    assert (msg, blocking) == (None, False), msg
    msg, blocking = ledger_bash_decision(f"python3 gen.py >> {hist}", None)
    assert msg and not blocking and "cannot read" in msg, msg
    assert ledger_bash_decision(f"cat {hist}", None) == (None, False), "a read is nothing"
    assert ledger_bash_decision("echo hi >> other.csv", str(led)) == (None, False)
    # a body counts only against the ledger its own pipeline writes
    msg, blocking = ledger_bash_decision(f"printf '{good.strip()}\\n' >> {hist} && echo '# t' > a.md", None)
    assert (msg, blocking) == (None, False), f"a second redirect is not the ledger's content: {msg}"
    msg, blocking = ledger_bash_decision(f"printf 'not-a-date,t,s,,,,,\\n' | tee -a {hist}; echo '# t' > a.md", None)
    assert blocking and "date" in msg and "header" not in msg, msg
    msg, blocking = ledger_bash_decision(f"echo '# t' > a.md; python3 gen.py >> {hist}", None)
    assert msg and not blocking and "cannot read" in msg, "an unrelated echo does not vouch for an opaque append"
    msg, blocking = ledger_bash_decision(f"cat <<'EOF' > {hist}\n{long_row}EOF\n", None)
    assert blocking and "cap 90" in msg, "a redirect after the heredoc opener is the same pipeline"
    hist2 = led / "HISTORY-Other-Sessions.csv"; hist2.write_text(hdr)
    msg, blocking = ledger_bash_decision(
        f"python3 gen.py >> {hist} && printf 'not-a-date,t,s,,,,,\\n' >> {hist2}", None)
    assert blocking and "date" in msg, "an opaque write first never excuses a readable bad row after it"
    msg, blocking = ledger_bash_decision(f"cat <<'EOF' \\\n  >> {hist}\n{long_row}EOF\n", None)
    assert blocking and "cap 90" in msg, "a redirect on a continued opener line is the same pipeline"
    hist2.unlink()
    # a manifest overrides the built-in for its store, and governs another CSV
    (led / MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": "HISTORY-*-Sessions.csv", "caps": {"summary": 20}},
        {"path": "LEDGER-Closures.csv", "required": ["slug"], "caps": {"note": 10}}]}))
    settings.cache_clear()
    msg, blocking = ledger_decision(str(hist), {"content": hdr + good})
    assert blocking and "characters, cap 20" in msg and MANIFEST in msg, msg
    clos = led / "LEDGER-Closures.csv"; clos.write_text("slug,note\n")
    msg, blocking = ledger_decision(str(clos), {"old_string": "x", "new_string": ",a very long note here\n"})
    assert blocking and "slug is required" in msg and "note is 21 characters" in msg, msg
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = cmd_ledgers(led)
    assert n == 1 and "2 ledger(s) checked, 1 finding(s)" in buf.getvalue(), buf.getvalue()
    (led / MANIFEST).unlink(); settings.cache_clear()
    assert "@@" not in usage() and "caps title ≤90" in usage(), "the help renders the built-ins from the constant"
    # an entry naming a built-in's glob MERGES onto it: only the keys it names change
    (led / MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": "HISTORY-*-Sessions.csv", "enum": {"area": ["docs", ""]}}]}))
    settings.cache_clear()
    ok = '2026-09-20,Docs pass,"A row under the store\'s own areas.",docs,,,,\n'
    assert ledger_decision(str(hist), {"content": hdr + ok}) == (None, False), "a store's own area passes"
    msg, blocking = ledger_decision(str(hist), {"content": hdr + ok.replace(",docs,", ",ui,")})
    assert blocking and "area is 'ui'" in msg, "the store's enum replaces the built-in's for that column"
    msg, blocking = ledger_decision(str(hist), {"content": hdr + ok.replace("2026-09-20", "20 Sep")})
    assert blocking and "date" in msg, "the built-in date pattern survives an enum override"
    msg, blocking = ledger_decision(str(hist), {"content": hdr + f'2026-09-20,{"t" * 91},s,docs,,,,\n'})
    assert blocking and "cap 90" in msg, "the built-in caps survive an enum override"
    (led / MANIFEST).unlink(); settings.cache_clear()
    # `extends`: a file named outside the glob inherits a built-in by declaration
    odd = led / "HISTORY-session-summary.csv"
    assert ledger_schema_for(odd) == (None, None), "outside the glob, nothing governs it"
    (led / MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": odd.name, "extends": "HISTORY-*-Sessions.csv", "enum": {"area": ["docs", ""]}}]}))
    settings.cache_clear()
    assert validate_manifest(led) == [], validate_manifest(led)
    assert ledger_decision(str(odd), {"content": hdr + ok}) == (None, False), "extends inherits the built-in"
    msg, blocking = ledger_decision(str(odd), {"content": hdr + ok.replace("2026-09-20", "20 Sep")})
    assert blocking and "date" in msg, "the inherited date pattern holds"
    (led / MANIFEST).write_text(json.dumps({"ledgers": [{"path": odd.name, "extends": "nope"}]}))
    settings.cache_clear()
    assert any("is no built-in" in f for f in validate_manifest(led)), validate_manifest(led)
    (led / MANIFEST).unlink(); settings.cache_clear()
    # an entry naming one file inside the built-in's glob governs the sweep alone
    (led / MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": hist.name, "enum": {"area": ["docs", ""]}}]}))
    settings.cache_clear(); hist.write_text(hdr + ok)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = cmd_ledgers(led)
    assert n == 0 and "1 ledger(s) checked, 0 finding(s)" in buf.getvalue(), buf.getvalue()
    (led / MANIFEST).unlink(); settings.cache_clear(); hist.write_text(hdr + good)

    # --- host adapters: Hermes speaks the same payload in its own vocabulary ---
    h = normalise({"hook_event_name": "pre_tool_call", "tool_name": "write_file",
                   "tool_input": {"path": "/x/MEMORY.md", "content": "c"},
                   "session_id": "s1"})
    assert h["_host"] == "hermes" and h["tool_name"] == "Write", h
    assert h["tool_input"]["file_path"] == "/x/MEMORY.md", h
    h = normalise({"hook_event_name": "pre_tool_call", "tool_name": "terminal",
                   "args": {"command": "ls"}})
    assert h["tool_name"] == "Bash" and h["tool_input"]["command"] == "ls", \
        "Hermes may carry the arguments under `args`"
    c = normalise({"hook_event_name": "PreToolUse", "tool_name": "Edit",
                   "tool_input": {"file_path": "/x/a.md", "new_string": "n"}})
    assert c["_host"] == "claude-code" and c["tool_name"] == "Edit", c
    m = normalise({"hook_event_name": "pre_tool_call", "tool_name": "memory",
                   "tool_input": {"action": "add", "target": "user", "content": "likes tea"}})
    assert m["tool_name"] == "Edit" and m["tool_input"]["file_path"].endswith("memories/USER.md"), m
    assert m["tool_input"]["content"] == "likes tea" and m["tool_input"]["_append"], m
    assert added_text({}, "old body\nlikes tea", "old body\n") == " \nlikes tea", \
        "an append reaches the scorer as the entry alone"
    m = normalise({"hook_event_name": "pre_tool_call", "tool_name": "memory",
                   "tool_input": {"operations": [
                       {"action": "replace", "target": "memory", "old_text": "a", "content": "b"},
                       {"action": "remove", "target": "memory", "old_text": "z"},
                       {"action": "add", "target": "memory", "new_text": "c"}]}})
    assert m["tool_input"]["file_path"].endswith("memories/MEMORY.md"), m
    assert m["tool_input"]["new_string"] == "b\nc" and m["tool_input"]["old_string"] == "a\nz", m
    assert "MEMORY.md" not in settings(HERMES)["index_files"], \
        "Hermes's MEMORY.md is its whole memory and must not inherit the index exemption"
    r = normalise({"hook_event_name": "pre_tool_call", "tool_name": "write_file",
                   "tool_input": {"path": "MEMORY.md", "content": "c"},
                   "cwd": str(CLAUDE / "memory")})
    assert r["tool_input"]["file_path"] == str(CLAUDE / "memory" / "MEMORY.md"), \
        "a relative path resolves against the payload cwd, or the write escapes the gate"
    assert session_asked({"_host": "hermes"}), "no session source at all fails open"

    # a Hermes nudge that cannot block is stashed and delivered on the next model call
    global PENDING_DIR
    saved, PENDING_DIR = PENDING_DIR, d / "pending"
    try:
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):
            assert emit({"_host": "hermes", "session_id": "s1"}, "nudge one", False) == 0
            assert emit({"_host": "hermes"}, "nudge for anyone", False) == 0
        assert deliver_pending("other") == "nudge for anyone", "another session gets only the unowned one"
        assert deliver_pending("s1") == "nudge one", "the owner gets its own"
        assert deliver_pending("s1") == "", "delivered once, then gone"
    finally:
        PENDING_DIR = saved

    # the 2e gate reads Hermes's state.db the way it reads Claude Code's transcript
    import sqlite3
    db = d / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (session_id text, role text, tool_calls text)")
    call = lambda cmd: json.dumps([{"function": {"name": "terminal",
                                                 "arguments": json.dumps({"command": cmd})}}])
    con.execute("insert into messages values ('asked','assistant',?)",
                (call("python3 ~/.claude/scripts/memfind.py 'q' council"),))
    con.execute("insert into messages values ('named','assistant',?)",
                (call("grep -n memfind.py memscan.py"),))
    con.commit(); con.close()
    assert memfind_asked_db(db, "asked"), "a terminal call that RUNS memfind discharges"
    assert not memfind_asked_db(db, "named"), "one that only names it does not"
    assert not memfind_asked_db(db, "silent"), "a session with no such call has not asked"
    assert memfind_asked_db(d / "missing.db", "asked"), "no database fails open"
    shutil.rmtree(d, ignore_errors=True)

    HOST_STORES["council"] = saved_council
    settings.cache_clear()
    shutil.rmtree(fixture, ignore_errors=True)
    print("selftest ok")


def hook_payload_from(docs, text, top=3):
    """selftest seam: the FIRE_FLOOR gate against an in-memory corpus."""
    ranked = [r for r in score_against(text, docs)[:top] if r[1] >= FIRE_FLOOR]
    return None if not ranked else "hit"


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
                  autonomous=False, sets=None):
    """The write-time nudge as one function: `(message, blocking)`.

    `docs` and `archived` are CALLABLES, invoked only after the cheap gates,
    because reading the corpus and the archive are the only expensive things the
    hook does and it runs on every memory write. Composed as one function
    returning a decision rather than an exit code so the selftest can drive the
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
                f"      existing: {hit[2][:160]}\n"
                f"Promote it, discard it, or fold it into the entry already there "
                f"(prime-memory-discipline.md §2)."), autonomous
    # A dated history clause stacked on a durable surface is §2's forbidden shape,
    # decidable from the added text alone (a regex, with §1's scoped dates as its
    # negatives in the selftest), so it fires here before any corpus is read and
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
            lines.append(f"      existing: {pair[1][:160]}")
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


def _strip_heredocs(command):
    """`command` with heredoc BODIES removed, openers kept.

    A heredoc body is data the shell feeds to a program, so an invocation quoted
    inside one is a fixture or a doc example rather than a command. ⚠ Measured on
    a real transcript: the two false positives surviving the position check were
    both selftest fixture strings inside `python3 - <<'PYEOF'` bodies, and one
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

    ⚠ A tool RESULT is excluded, and this file is why: the selftest fixtures below
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
        saved = (memfind.TIMEOUT, memfind.AUTOSTART)
        memfind.TIMEOUT, memfind.AUTOSTART = timeout, False   # no server spawn inside a hook's budget
        try:
            idx = memfind.load_cached(root)
            if idx is None:
                return None
            items, flat, dim, stale = idx
            rows = memfind.rank(text[:2000], root, top=top, idx=(items, flat, dim), lexical=False)
            return rows, stale
        finally:                             # a long-lived caller (the MCP server) keeps its own
            memfind.TIMEOUT, memfind.AUTOSTART = saved
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
            out.append(f"  - {f}   ({c:.2f})")
            out.append(f"      {s[:150]}")
        out.append("Ranked neighbours, not a verdict: read the top file, then repeat this "
                   "write. To ask about any other statement:")
    out.append(f'  python3 {Path(__file__).resolve().parent / "locket.py"} find "<statement>" {root}')
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
  `cites` resolves against, and register reference CSVs. The manifest, the
  `.memfind/` cache and the registry row are all this tool owns for a corpus.

MANIFEST
  `locket.json` at the corpus root, a JSON object (the older name `memfind.json`
  is still read; `migrate` renames it); `schema` writes its JSON
  Schema for an editor and checks a manifest against it. Every key is optional and
  a missing key takes the default shown; a broken file degrades to all the
  defaults rather than taking a hook down. This is the whole list.

  name             string. What `corpora` and a store argument call this
                   store. Default: the directory's basename. Host stores and
                   project memories are named by the host and ignore it.
  members          list of top-level directory names that ARE corpus, the
                   rest of the root being host state; markdown at the root is
                   always in. Default: every directory, except on a host's
                   core directory, where it is the host's member list above.
                   REPLACES that list, so a Hermes that authors its own skills
                   writes ["memories", "skills"].
  excluded_dirs    list of directory names, at any depth, that are not corpus.
                   REPLACES the default: archive, in-flight, harness-logs,
                   discovery-research. Whatever the list says, .memfind,
                   backups, backup and every dot-directory stay excluded.
  excluded_files   list of corpus-relative paths or glob patterns naming files
                   that are not corpus, each tried against the relative path
                   and the basename: "notes/draft.md", "*.csv", "scratch-*".
                   Skipped by every mode, the index and the write-time gate.
                   Also removes a registered CSV it matches. Default: none.
  index_files      list of file names whose lines POINT at other files, so a
                   line restating its target is not a fork. Default:
                   MEMORY.md, ARCHIVE.md, CURATION-INDEX.md,
                   CURATION-SOURCES-INDEX.md, FRONTEND-INDEX.md,
                   PLATFORM-INDEX.md. Replaces the default.
  ledger_surfaces  list of file names that accumulate entries by design and
                   are never checked (chronology is the value). Default:
                   HARNESS.md, CHECKPOINTS.md, EXPLORATIONS.md. Replaces.
  holding_spaces   list of file names scored only against THEMSELVES: a
                   restated entry fires, a fact held elsewhere does not.
                   Default: SHORT_TERM.md, OPEN_QUESTIONS.md,
                   MEMORY-proposed.md, PROMOTIONS.md. Replaces.
  whole_read_exempt
                   list of file names the SessionStart size sweep skips: a
                   mixed surface its own load contract says is never read
                   whole, or a file a project instrument already caps. It
                   exempts nothing else; every other mode still reads the
                   file. Default: none.
  code_roots       list of directories `cites` resolves path:line citations
                   against, outright. Default: the project's own working
                   directory (for a project memory) plus ~/.claude and
                   ~/.hermes where they exist.
  sources          list of {"path", "text", "label"}: a reference CSV whose
                   rows are claims in this corpus. `path` is the file, absolute
                   or relative to the store, `text` the column holding the
                   sentence, `label` the column that names a row in output (an
                   Area). Default: the older standalone reference-sources.json
                   if present, else none.
  ledgers          list of {"path", "extends", "columns", "required", "caps",
                   "enum", "pattern"}: a schema for a CSV ledger in this store. `path`
                   is a basename glob or a store-relative path; `columns` the
                   exact header; `required` the columns that may not be empty;
                   `caps` {column: max characters}; `enum` {column: [values]},
                   "" among them where empty is allowed; `pattern` {column:
                   regex the whole value must match}. Every key but `path` is
                   optional. A built-in schema governs its glob wherever the
                   file sits, corpus or not, and is the authority on it. An
                   entry naming that glob, or a path under it, merges onto it
                   for this store: `caps`, `enum` and `pattern` change only the
                   columns they name, `columns` and `required` replace.
                   `extends` names a built-in's glob to merge onto for a file
                   whose name falls outside it.
                   Default: the built-ins alone, which are:
@@BUILTIN_LEDGERS@@
  belongs_to       string, the name of another corpus as `corpora` lists it.
                   This directory then JOINS that corpus: every mode, the index
                   and the write-time gate resolve a file here to the parent,
                   the listing shows the parent alone, and its files appear as
                   `<name>/<path>` (this store's `name`, else its basename).
                   The store keeps its own `members`, `excluded_*` and
                   `code_roots` for its own walk; the file-name lists union
                   upward. Written by `init <dir> --parent <corpus>`. A name
                   that resolves to nothing, to itself, to a store under it
                   on disk or whose own walk reaches it, or into a cycle is ignored and the store
                   stands alone. Default: none.

  On Hermes's ~/.hermes the three file-name lists default to empty
  instead, because its MEMORY.md is the whole memory rather than an index; a
  manifest there still wins. Store the file under version control: it is
  authored configuration, where .memfind/ is derived and rebuilt.

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
            `memory`, so it takes `--name`.
  schema [DIR]  write the manifest's JSON Schema to ~/.locket/locket.schema.json
            (`init` does this too), so an editor mapped to that path hints every
            `locket.json` by filename; with DIR, check that directory's manifest
            against it instead: an unknown key with a did-you-mean, a wrong type,
            a malformed `sources` or `ledgers` entry. `locket help install`
            names the editor side.
  migrate [DIR|all]  rename a store's older `memfind.json` manifest to
            `locket.json` in place; every store on the machine when DIR is
            omitted. Until a store is migrated the old name is still read.
  forget DIR  drop DIR's registry row. Its manifest and `.memfind/` are yours
            to delete; without them the directory is no longer a corpus.
  selftest  run the internal checks.

HOOK MODES (PreToolUse entry points, JSON on stdin; not for manual use)
  Claude Code (matcher: tool names below) and Hermes (`pre_tool_call`, matcher
  write_file|patch, terminal, search_files) hand the same payload; the tool
  names are mapped on the way in. Hermes can hear a BLOCK and nothing else on a
  tool event, so a nudge there is stashed and handed over by `deliver`.
  deliver     Hermes `pre_llm_call` only. Prints every nudge stashed for this
              session as `context`, which Hermes prepends to the model call that
              follows the tool result. One hop later than Claude Code, never
              lost. Stashes older than an hour are purged unread.
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
              heredoc or sed -i, which `hook` never sees because it matches
              TOOLS rather than paths. Blocks an unambiguous write, warns on a
              scripted one, and is silent for ledgers, holding spaces,
              archive/, the scratchpad and autonomous runs. `MEMWRITE-OK:
              <reason>` in the command declares the exception and passes.
  grepassist  Grep|Bash. Runs memfind on a prose-shaped search pattern and
              injects the ranked neighbours, because 2e says ask the question
              BEFORE the grep and prose alone did not achieve it. Fires only on
              a pattern carrying no regex metacharacter and two words of four
              characters or more, so an identifier or an alternation never
              reaches it. Resolves the corpus from the search path, else from
              the cwd. Silent with the embedder down.

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
# pattern is REFUSED, where every prose check only warns. The built-in schema is
# the session-history ledger every project keeps; it is keyed by a basename
# glob and applies wherever the file is, corpus or not, because the project root
# a HISTORY file sits in is usually no store at all. A manifest's `ledgers` list
# adds a schema for any other CSV in that store, or adjusts a built-in for its
# own file by naming the same glob or a path under it, merging onto it
# (`_merge_schema`). This constant is each schema's one home: `usage()` renders
# it into the help, and the project-dev skill points here rather than restating it. The date takes an optional UTC time, because
# a local-evening session lands on the next UTC day and a bare date cannot say
# which side of midnight it came from. `repo` names the repository the hash
# resolves in (`group/sub/repo` allowed), `issue` is one tracker-agnostic ticket
# key (`#12`, `ENG-42`) and `pr` a GitHub PR or GitLab MR (`#155`, `!155`).
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
}
_SCHEMA_KEYS = ("columns", "required", "caps", "enum", "pattern")


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
    a built-in against the basename anywhere on disk."""
    p = Path(path).expanduser()
    if p.suffix.lower() != ".csv":
        return None, None
    phys = _phys_root_of(p.parent)
    root = top_of(phys) if phys is not None else None
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
            try:
                ok = v is None or re.fullmatch(pat, v) is not None
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
        with open(path, newline="", errors="ignore") as fh:
            rows = _csv_rows_of(fh.read(4096))
    except OSError:
        return None
    return rows[0] if rows else None


def ledger_message(path, glob, schema, findings, via):
    return (f"A ledger row breaks its schema and the write is refused ({Path(path).name}, "
            f"schema {glob!r}, {schema.get('_from', 'built in')}):\n  - " + "\n  - ".join(findings[:12])
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
            lines.append(f"  row: {row[:120]}")
            for _c, best, fname, pair in ranked:
                lines.append(f"    - {fname}   (closest claim {best:.2f})"
                             + (f"\n        existing: {pair[1][:160]}" if pair else ""))
    if not lines:
        return None, False
    return (f"{LEAD_WARNING} {p.name} is a registered source, and a row here may restate a claim "
            f"the store already carries:\n" + "\n".join(lines) +
            "\nOne fact, one row: cut the prose copy to a pointer, or fold the row into the one "
            "that owns it (prime-memory-discipline.md §2c). A new fact needs nothing."), False


_QUOTED = re.compile(r"""(?:echo|printf)\s+(?:-[a-zA-Z]+\s+)*(?:"((?:[^"\\]|\\.)*)"|'([^']*)')""")
_CSV_REDIRECT = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*(?P<p>[^\s;&|]+\.csv)")
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
        cands = [tok] if tok.startswith("/") else [str(Path(r) / tok) for r in _cd_targets(command, cwd)]
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


def cmd_ledgers(root):
    """Every schema-governed CSV in the corpus, each row checked. A clean run
    prints its scope, so a zero says what it was scored against."""
    schemas = ledger_schemas(root)
    seen, files = set(), []
    for r in roots_of(root):
        for glob in schemas:
            for p in sorted(Path(r).glob(glob) if "/" not in glob else [Path(r) / glob]):
                rp = p.resolve()
                if not p.is_file() or rp in seen:
                    continue
                seen.add(rp)
                # the gate's own resolver, so a file an entry names inside a
                # built-in's glob is checked once, under the entry, as it is written
                schema, g = ledger_schema_for(p)
                if schema is not None:
                    files.append((p, schema, g))
    if not files:
        print(f"no schema-governed ledger in {root} (schemas: {', '.join(schemas)})")
        return 0
    bad = 0
    for p, schema, glob in files:
        findings = check_ledger_text(schema, p.read_text(errors="ignore"))
        rows = max(len(_csv_rows_of(p.read_text(errors="ignore"))) - 1, 0)
        print(f"{corpus_key(root, p)}  schema {glob!r}  {rows} rows, {len(findings)} findings")
        for f in findings:
            print(f"  {f}")
        bad += len(findings)
    print(f"{len(files)} ledger(s) checked, {bad} finding(s)")
    return bad


SCRATCH_MARKERS = ("/scratchpad/", "/private/tmp/", "/tmp/")
MEMWRITE_ESCAPE = "MEMWRITE-OK:"
# Destination-position writes. Direction is the whole point: a corpus path as the
# SOURCE of a cp is a read, and blocking that would break every
# backup-before-rewrite command this discipline actually requires.
BASH_WRITE = (
    re.compile(r">>?\s*(?P<p>[^\s;&|]+\.md)"),
    re.compile(r"\btee\b[^;&|]*?(?P<p>[^\s;&|]+\.md)"),
    re.compile(r"\bsed\b[^;&|]*?-i[^;&|]*?(?P<p>[^\s;&|]+\.md)"),
    re.compile(r"\b(?:cp|mv|rsync)\b[^;&|]+?\s(?P<p>[^\s;&|]+\.md)\s*(?:$|[;&|])"),
)
# Destination computed at run time, so it cannot be read off the command text.
# Warn rather than block: a false block here costs more than a missed one.
BASH_WRITE_OPAQUE = re.compile(
    r"write_text\(|open\([^)]*['\"][wa]['\"]|\.writelines\(|shutil\.(?:copy|move)")


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


def _cd_targets(command, cwd):
    """Directories this command may resolve a bare filename against."""
    roots = [cwd] if cwd else []
    roots += [_expand_home(d) for d in re.findall(r"\bcd\s+([^\s;&|]+)", command)]
    return roots


def bash_write_targets(command, cwd):
    """(blocking, opaque) corpus files a Bash command appears to write.

    Resolves a bare `MEMORY.md` against the payload cwd and any `cd` in the
    command, because that is the shape a heredoc rewrite actually takes. Resolving
    rather than name-matching is why the same command is a corpus write from inside
    a corpus and nothing at all from outside one.
    """
    def resolve(tok):
        tok = _expand_home(tok)
        cands = ([tok] if tok.startswith("/")
                 else [str(Path(r) / tok) for r in _cd_targets(command, cwd)])
        for c in cands:
            if corpus_for(c) is not None:
                return Path(c)
        return None

    blocking = []
    for pat in BASH_WRITE:
        for m in pat.finditer(command):
            hit = resolve(m.group("p"))
            if hit:
                blocking.append(hit)
    opaque = []
    if BASH_WRITE_OPAQUE.search(command):
        for tok in re.findall(r"[\w./~-]+\.md", command):
            hit = resolve(tok)
            if hit and hit not in blocking:
                opaque.append(hit)
    return blocking, opaque


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
    blocking, opaque = bash_write_targets(command, cwd)

    def live(q):                             # exempt by ITS corpus's names, not ours
        sets = settings(corpus_for(q))
        return q.name not in sets["append_surfaces"] and "archive" not in q.parts
    blocking = [q for q in blocking if live(q)]
    opaque = [q for q in opaque if live(q)]
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
                f"2e checks will not fire and nothing will report that they did not. Run "
                f"`memfind` on what you are about to assert, or route the edit through "
                f"Write/Edit."), False
    return None, False


# A regex handed to an embedding ranker is noise, so the injector fires only on a
# pattern that reads as a question: two content words, no metacharacters.
GREP_META = set("|[]()\\^$*+?{}")


def prose_like(pattern):
    if not pattern or GREP_META & set(pattern):
        return False
    return len([w for w in re.split(r"[^A-Za-z0-9_]+", pattern) if len(w) > 3]) >= 2


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
    lines = [f"  [{sc:.3f}] {name}  ::  {claim[:110]}" for sc, name, claim in rows]
    return ("Before the grep: these rank closest to that text by MEANING, which is what a "
            "grep cannot see (a paraphrase shares too few tokens to match at all).\n"
            + "\n".join(lines)
            + "\nRead them as candidates, never as a verdict; the scores adjudicate nothing "
              "(prime-memory-discipline 2e). A `<file>.csv#<area>` result is one ROW of a reference "
              "CSV rather than a file, so query that CSV on `Area=` for the rest.")


# ---------------------------------------------------------------------------
# Host adapters: the hook modes above speak one payload shape, and two hosts
# produce it. Claude Code's PreToolUse and Hermes's pre_tool_call carry the same
# stdin JSON by design (`hook_event_name`, `tool_name`, `tool_input`, `cwd`), and
# differ in tool vocabulary and in what a reply may say.
# ---------------------------------------------------------------------------
# Hermes tool name -> the Claude Code tool the hook modes are written against.
HERMES_TOOLS = {"write_file": "Write", "patch": "Edit", "terminal": "Bash",
                "search_files": "Grep"}


def read_payload():
    """The hook payload from stdin, normalised to Claude Code's shape, or None.

    Sets `_host` ("claude-code" | "hermes"), maps Hermes tool names, and copies
    Hermes's `path` into `file_path`. A `patch` call in Hermes's unadvertised V4A
    mode carries a whole patch string and no `new_string`; it stays opaque and the
    write modes see an empty text, which is silence rather than a wrong nudge.
    """
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return None
    return normalise(payload)


def normalise(payload):
    """`read_payload` on an already-parsed dict, so the selftest can drive it."""
    if not isinstance(payload, dict):
        return None
    ti = payload.get("tool_input")
    if not isinstance(ti, dict):
        ti = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    ti = dict(ti)
    event = str(payload.get("hook_event_name") or "")
    host = "hermes" if event in ("pre_tool_call", "post_tool_call") else "claude-code"
    tool = payload.get("tool_name")
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
    if isinstance(fp, str) and fp and not Path(fp).expanduser().is_absolute():
        ti["file_path"] = str(Path(payload.get("cwd") or os.getcwd()) / fp)
    payload = dict(payload, tool_input=ti, tool_name=tool, _host=host)
    return payload


def emit(payload, msg, blocking):
    """Deliver a hook decision the way THIS host can hear it; returns the exit code.

    Claude Code: exit 2 with the message on stderr blocks; stdout JSON carrying
    `additionalContext` reaches the model without blocking. Hermes: exit 2 blocks,
    and the message travels as stdout JSON so it is not cut at the 400 characters
    Hermes keeps of stderr. ⚠ Hermes has NO channel for a non-blocking message on
    a tool event: `pre_tool_call` honours only block and modify, and a
    `post_tool_call` context is parsed and then discarded. What it does have is
    `pre_llm_call`, whose `context` is prepended to the next model call, so a
    warning is STASHED here and the `deliver` mode, registered on that event,
    hands it to the model one hop later, after the tool result and before the
    model reasons on it. Without `deliver` registered the stash is purged unread.
    """
    if not msg:
        return 0
    host = payload.get("_host", "claude-code")
    if blocking:
        if host == "hermes":
            print(json.dumps({"action": "block", "message": msg}))
        print(msg, file=sys.stderr)
        return 2
    if host == "hermes":
        stash(payload.get("session_id"), msg)
        print(msg, file=sys.stderr)
        return 0
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "additionalContext": msg}}))
    return 0


# Keyed by uid and created owner-only: a shared temp dir lets another user make the
# path first, as a readable directory or a symlink elsewhere.
PENDING_DIR = Path(tempfile.gettempdir()) / f"memscan-pending-{os.getuid() if hasattr(os, 'getuid') else 'u'}"
PENDING_TTL = 3600      # seconds; a stash nobody delivered is stale, not owed


def _pending_ours():
    """The stash directory exists, is no symlink, and is this user's. `deliver` hands its
    files to the model, so a directory someone else made would be an injection route."""
    try:
        return (PENDING_DIR.is_dir() and not PENDING_DIR.is_symlink()
                and (not hasattr(os, "getuid") or PENDING_DIR.stat().st_uid == os.getuid()))
    except OSError:
        return False


def stash(session_id, msg):
    """Park a non-blocking message for `deliver` to hand to the next model call."""
    try:
        PENDING_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not _pending_ours():
            return
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id or "any"))
        (PENDING_DIR / f"{sid}.{time.time_ns()}.txt").write_text(msg)
    except OSError:
        pass                                 # a guardrail never breaks the write it watched


def deliver_pending(session_id):
    """Every message stashed for this session (and for no session), joined; the
    files are removed on read and stale ones purged, so nothing is delivered twice."""
    if not _pending_ours():
        return ""
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id or "any"))
    now, out = time.time(), []
    for p in sorted(PENDING_DIR.iterdir()):
        try:
            owner = p.name.split(".", 1)[0]
            if now - p.stat().st_mtime > PENDING_TTL:
                p.unlink()
                continue
            if owner not in (sid, "any"):
                continue
            out.append(p.read_text())
            p.unlink()
        except OSError:
            continue
    return "\n\n".join(out)


def gate_status():
    """One line per host on whether the 2e question gate can see a session here,
    and whether a non-blocking nudge has a way to reach the model."""
    lines = []
    if CLAUDE.is_dir():
        lines.append("2e gate on Claude Code: armed (reads the session transcript "
                     "Claude Code hands the hook); nudges reach the model on the write")
    if HERMES.is_dir():
        db = HERMES / "state.db"
        lines.append("2e gate on Hermes: " + ("armed (reads state.db by session_id)"
                                              if db.is_file() else
                                              "UNARMED, state.db not found, so every "
                                              "write passes the question silently")
                     + "; nudges reach the model one call later, only if `deliver` "
                       "is registered on pre_llm_call")
    return "\n".join(lines) if lines else "no known host home found; CLI modes only"


def main(argv):
    if len(argv) < 2 or argv[1] in ("--help", "-h", "help"):
        print(usage())
        return 0 if len(argv) > 1 else 1
    mode = "pointers" if argv[1] == "hooks" else argv[1]   # the older name
    if mode == "selftest":
        selftest()
        return 0
    if mode == "hook":                      # PreToolUse entry, JSON on stdin
        payload = read_payload()
        if payload is None:
            return 0
        ti = payload["tool_input"]
        path = str(ti.get("file_path", ""))
        if path.lower().endswith(".csv"):  # a ledger row: schema, decidable, may block
            try:
                msg, blocking = ledger_decision(path, ti)
                if not blocking:
                    try:
                        prior = Path(path).read_text(errors="ignore")
                    except OSError:
                        prior = ""
                    msg = source_row_decision(path, ti, prior)[0]
            except Exception:               # a guardrail must never break a write
                return 0
            return emit(payload, msg, blocking)
        root = corpus_for(path)
        if root is None:
            return 0
        text = str(ti.get("content") or ti.get("new_string") or "")
        try:
            prior = Path(path).read_text(errors="ignore")
        except OSError:                     # new file, or unreadable
            prior = ""
        if ti.get("_append") and prior:     # an append is a Write of prior + entry
            text = prior.rstrip("\n") + "\n" + text
        name = Path(path).name
        try:
            msg, blocking = hook_decision(
                name, ti, text, prior, lambda: corpus(root),
                lambda: archived_claims(root),
                autonomous=os.environ.get("CLAUDE_AUTONOMOUS") == "1",
                sets=settings(root))
        except Exception:                   # a guardrail must never break a write: a
            return 0                        # dangling symlink in the corpus raised here
        if msg and blocking:
            return emit(payload, msg, True)
        # The silent path is the paraphrase blind spot: lexical scoring found
        # nothing, which is also what it returns for a fact restated in new words.
        try:
            gate = memfind_gate(payload, name, ti, text, prior, root)
        except Exception:                   # a guardrail must never break a write
            gate = None
        # A non-blocking nudge must still reach the model when the 2e gate blocks,
        # or the warning is discarded by the very write that gets refused. ⚠ It is
        # re-led on the way: exit 2 means the write did NOT land, and a warning
        # worded for the landed case would assert the opposite of what happened.
        if gate:
            warn = msg.replace(LEAD_WARNING, LEAD_BLOCKING, 1) if msg else None
            return emit(payload, "\n\n".join(x for x in (warn, gate) if x), True)
        return emit(payload, msg, False)

    if mode == "deliver":                   # Hermes pre_llm_call: the stashed nudges
        raw = sys.stdin.read()              # drain it; the history inside is not needed
        try:
            sid = (json.loads(raw) or {}).get("session_id")
        except (json.JSONDecodeError, ValueError, AttributeError):
            sid = None
        msg = deliver_pending(sid)
        if msg:
            # Hermes caps a hook's context at 10,000 characters and spills the
            # rest to a file the model is told about; keep the whole nudge inline.
            if len(msg) > 9000:
                msg = msg[:9000] + "\n[nudges truncated; run memscan/memfind directly for the rest]"
            print(json.dumps({"context": msg}))
        return 0

    if mode == "bashguard":                 # PreToolUse on Bash: the writes it misses
        payload = read_payload()
        if payload is None or payload["tool_name"] != "Bash":
            return 0
        command = str(payload["tool_input"].get("command", ""))
        try:
            msg, blocking = ledger_bash_decision(command, payload.get("cwd"))
            if msg is None:
                msg, blocking = bash_write_decision(
                    command, payload.get("cwd"),
                    autonomous=os.environ.get("CLAUDE_AUTONOMOUS") == "1")
        except Exception:                   # a guardrail must never break the shell
            return 0
        return emit(payload, msg, blocking)

    if mode == "grepassist":                # PreToolUse on Grep|Bash: 2e, automatic
        try:
            payload = read_payload()
            if payload is None:
                return 0
            ti = payload["tool_input"]
            tool = payload["tool_name"]
            where = None
            if tool == "Grep":
                pattern, where = str(ti.get("pattern", "")), ti.get("path")
            elif tool == "Bash":
                cmd = str(ti.get("command", ""))
                m = re.search(r"(?<![\w-])(?:grep|egrep|rg|ugrep)(?![\w-])"
                              r"[^;&|]*?['\"]([^'\"]+)['\"]", cmd)
                if not m:
                    return 0
                pattern = m.group(1)
                where = next((x for x in re.findall(r"(/[\w./~-]+)", cmd)
                              if corpus_for(x + "/_probe.md")), None)
            else:
                return 0
            root = corpus_for(str(Path(where) / "_probe.md")) if where else None
            if root is None:
                root = corpus_here(payload.get("cwd") or os.getcwd())
            if root is None or not prose_like(pattern):
                return 0
            out = grep_assist(pattern, root)
        except Exception:                   # ollama down, bad payload, anything
            return 0
        return emit(payload, out, False)

    if mode == "init":                      # needs no corpus of its own: it makes one
        rest, flags = list(argv[2:]), {}
        for f in ("--parent", "--name"):
            if f in rest:
                i = rest.index(f)
                if i + 1 >= len(rest):
                    print(f"{f} needs a value", file=sys.stderr)
                    return 1
                flags[f] = rest[i + 1]; del rest[i:i + 2]
        if len(rest) != 1:
            print("init needs a directory:  locket init <dir> [--parent <corpus> [--name <label>]]",
                  file=sys.stderr)
            return 1
        if "--name" in flags and "--parent" not in flags:
            print("--name labels a joining store and needs --parent", file=sys.stderr)
            return 1
        return cmd_init(rest[0], parent=flags.get("--parent"), name=flags.get("--name"))
    if mode == "migrate":                   # rename memfind.json -> locket.json, one store or all
        one = len(argv) > 2 and argv[2] != "all"
        targets = [resolve(argv[2])] if one else list(_raw_corpora().values())   # joined stores too
        seen, n = set(), 0
        for r in targets:
            if r is None or r in seen:
                continue
            seen.add(r)
            what = migrate(r)
            if what != "none" or one:       # a store with no manifest at all is not news
                print(f"{r}: {what}"); n += 1
        if not n:
            print("migrate: nothing to rename")
        return 0
    if mode == "forget":
        if len(argv) < 3:
            print("forget needs a directory:  locket forget <dir>", file=sys.stderr)
            return 1
        print(f"registry: {REGISTRY} ({'row removed' if forget(argv[2]) else 'not listed'})")
        return 0
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
              f"(audit, across, dupes, budget, ledgers, pointers, links, cites, corpora, init, schema, migrate, forget, against)?")
        print(usage())
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:      # piping any mode into `head` is normal use
        sys.stderr.close()
        sys.exit(0)
