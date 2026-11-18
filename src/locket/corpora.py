"""Which stores exist and how they join: the host stores, Claude Code's project stores and
registered stores, each store's parent and children, the store a path belongs to, and
each store's settings, its manifest over the host defaults with the name-sets of its
children unioned in.
"""
import fnmatch
import os
import re
from functools import lru_cache
from pathlib import Path

import fsio
import hosts
import manifest
import paths
import trust


def raw_corpora():
    """Every physical store on the machine, joined or not: each host store that
    exists, one per Claude Code project, and every registered directory whose
    manifest is still there. A registered store is keyed by the `name` in its
    manifest, else its basename, else its full path where either collides. A
    project store is keyed by its manifest's `name` too, else its slug, which
    `resolve` still accepts; `init` writes the name and never registers one."""
    out = {k: v for k, v in paths.PATHS.host_stores.items() if v.is_dir()}
    reg = [p for p in trust.registry() if manifest.has_manifest(p)]
    # a project's name yields to a host key and to a registered store's name, which
    # a `belongs_to` elsewhere may already point at
    taken = set(paths.PATHS.host_stores) | {trust.read_manifest(p).get("name") for p in reg}
    for d in sorted((hosts.claude() / "projects").glob("*/memory")):
        name = trust.read_manifest(d).get("name")
        ok = isinstance(name, str) and name and name not in taken and name not in out
        out[name if ok else d.parent.name] = d
    for p in reg:
        if p in out.values():
            continue
        name = trust.read_manifest(p).get("name")
        key = name if isinstance(name, str) and name else p.name
        out[key if key not in out else str(p)] = p
    return out


def all_corpora():
    """Every corpus a name can address: the physical stores, less each one that
    `belongs_to` another. A joined store is reached through its parent's name
    and counted in the parent's files, so it is listed once, as the parent."""
    raw = raw_corpora()
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
    name = trust.read_manifest(root).get("belongs_to")
    if not isinstance(name, str) or not name:
        return None
    raw = raw if raw is not None else raw_corpora()
    target = _by_name(raw, name)
    if target is None or target == root or root in target.parents or walked_by(target, root):
        return None
    seen, cur = {root}, target                  # a cycle means nobody is the top
    while cur is not None:
        if cur in seen:
            return None
        seen.add(cur)
        nxt = trust.read_manifest(cur).get("belongs_to")
        cur = _by_name(raw, nxt) if isinstance(nxt, str) else None
    return target


def walked_by(outer, inner):
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
    raw = raw_corpora()
    while True:
        up = parent_of(root, raw)
        if up is None:
            return root
        root = up


def children_of(root, raw=None):
    """Physical stores whose `belongs_to` resolves to `root`, sorted."""
    raw = raw if raw is not None else raw_corpora()
    root = Path(root)
    return sorted(r for r in set(raw.values()) if r != root and parent_of(r, raw) == root)


def roots_of(root):
    """Every physical directory composing the corpus `root`: itself, then each
    store that belongs to it, recursively. A store is one of these to itself."""
    raw = raw_corpora()
    out, todo = [], [Path(root)]
    while todo:
        r = todo.pop(0)
        if r in out:
            continue
        out.append(r)
        todo.extend(children_of(r, raw))
    return out


def store_label(root):
    """What a joined store's files are prefixed with in every report and index
    key: the manifest's `name`, else the directory's basename."""
    name = trust.read_manifest(root).get("name")
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
            return f"{store_label(r)}/{path.relative_to(r)}"
        except ValueError:
            continue
    return str(path)


def corpus_for_cwd(cwd):
    """The auto-memory corpus belonging to a working directory, or None.

    Claude Code slugifies the absolute path into `~/.claude/projects/<slug>/memory`,
    replacing every non-alphanumeric character with a dash.
    """
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(Path(cwd).resolve()))
    d = hosts.claude() / "projects" / slug / "memory"
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
    root = phys_root_of(p.parent)          # membership and exclusion are the physical store's
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


def phys_root_of(d):
    """The physical store that directory `d` is or sits inside, or None: a host
    store, a directory holding a manifest, or Claude Code's
    `~/.claude/projects/<slug>/memory`. Membership, exclusion and the walk are
    the physical store's; the corpus it belongs to is `top_of` it."""
    projects = hosts.claude() / "projects"
    stores = set(paths.PATHS.host_stores.values())
    if len(str(d)) > fsio.PATH_MAX:              # no file lives there; walking it cost a stat per part
        return None
    for parent in (d, *d.parents):
        if parent in stores:
            return parent
        if manifest.has_manifest(parent):
            # An untrusted manifest makes its subtree no store at all, never a
            # member of a trusted store above it: the parent's walk stops here too.
            return parent if trust.trusted(parent) else None
        if parent.name == "memory" and parent.parent.parent == projects:
            return parent
    return None


def _root_of(d):
    """The corpus root that directory `d` is or sits inside, or None. One walker
    for the gate, the CLI default and the MCP, so they cannot disagree about
    what a corpus is; a store that `belongs_to` another resolves to that one."""
    phys = phys_root_of(d)
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
        s = os.path.normpath(p)             # `..` collapsed, or an excluded dir it passes through still counts
        if not os.path.exists(s):           # os.path swallows every OSError, a name too long included
            # only a glob cut may fall back to its directory (`~/brain/fact*.md` cut to `~/brain/fact`);
            # any other missing word, the verb `grep` first, would fall back to the cwd's own store
            s = os.path.dirname(s) if cut else ""
            if not s or not os.path.exists(s):
                continue
        p = Path(s)
        root = corpus_for((p if p.is_dir() else p.parent) / "_probe.md")
        if root is not None:
            return root
    return None


def _own_walk(root, m=None):
    """-> (members, excluded_dirs): what one store's own walk covers, from its own
    manifest alone. Never unioned across a join, so `parent_of` can ask it of a
    parent without recursing through `settings`, which unions name-sets upward."""
    m = trust.read_manifest(root) if m is None else m
    mem = m.get("members")
    members = ({str(x) for x in mem if isinstance(x, str) and manifest.inside(x)} if isinstance(mem, list)
               else hosts.members_of(root))
    val = m.get("excluded_dirs")
    excluded = {str(x) for x in val} if isinstance(val, list) else set(hosts.EXCLUDED_DIRS)
    excluded.add(".memfind")                  # the cache is never prose, whatever the manifest says
    excluded |= hosts.excluded_of(root)
    excluded.update({"backups", "backup"})    # nor is a backup, which forks by design
    excluded |= hosts.DEPENDENCY_DIRS               # nor a dependency or build tree, whatever the manifest says
    return members, excluded


def settings(root):
    """`_settings_for`, cached per store and per state of the trust registry, so a store
    trusted or untrusted in another process is seen without a restart."""
    trust.trusted_ids()
    pinned = root is not None and fsio.inode(root) in trust.TRUST_CACHE[next(iter(trust.TRUST_CACHE))][1]
    return _settings_for(root, (next(iter(trust.TRUST_CACHE)), trust.config_digest(root) if pinned else None))


def clear_settings():
    """Forget every store's settings, after a manifest or the registry changed in this process."""
    _settings_for.cache_clear()


@lru_cache(maxsize=None)
def _settings_for(root, _trust_key):
    """Per-corpus name-sets, the module defaults overridden by the manifest.

    Every place that once read a module-level set reads this instead, keyed on
    the corpus, so two corpora on one machine can name their ledgers differently.
    The defaults are this apparatus's file names and are what a corpus without a
    manifest gets, which keeps the Claude Code install zero-config.
    """
    m = trust.read_manifest(root)
    members, excluded = _own_walk(root, m)
    # A host store whose files are not this apparatus's inherits none of the defaults
    # below (`hosts.FOREIGN`); a manifest there still wins.
    if hosts.foreign(root):
        m = {"index_files": [], "ledger_surfaces": [], "holding_spaces": [], **m}
    out = {}
    # `members`: the top-level directories that ARE corpus, everything else under
    # the root being host state. None means every directory, which is what a
    # manifest store or a project memory gets; a host store takes HOST_MEMBERS.
    out["members"] = members
    for key, default in (("index_files", manifest.INDEX_FILES),
                         ("ledger_surfaces", manifest.LEDGER_SURFACES),
                         ("holding_spaces", manifest.HOLDING_SPACES),
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
