"""Which stores may act. A manifest is inert until a person trusts its store at a terminal;
trust is pinned to a digest of the store's configuration files, and any change but
Locket's own makes the store inert again: write_store_file writes a store's
configuration and moves the pin with the edit.
"""
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

import fsio
import hosts
import manifest
import paths


# The registry: manifest directories that `corpora`, `all` and a substring
# argument can find. The gate derives a corpus from the written path and needs
# no registry, but enumeration has nothing to derive from, and a whole-disk
# search for manifests is not an option. `init <dir>` writes the manifest if
# absent and registers the path; the manifest stays the authority that a
# directory IS a corpus, so a row whose manifest is gone is reported, never
# trusted. Lives in Locket's state directory, outside every corpus;
# `LOCKET_REGISTRY` points a test at a scratch one.
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
    m = manifest.manifest_data(d)
    rows = m.get("triggers") if isinstance(m.get("triggers"), str) and m.get("triggers") else "triggers.json"
    rows = Path(rows).expanduser()
    files = [manifest.manifest_file(d), d / manifest.LEGACY_MANIFEST, rows if rows.is_absolute() else d / rows]
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
            with fsio.open_safe(f, binary=True) as fh:
                body = fh.read()
        except OSError:
            body = None
        tag = b"absent" if body is None else str(len(body)).encode()
        h.update(f.name.encode() + b"\0" + tag + b"\0" + (body or b""))
    _DIGESTS[str(d)] = (sig, h.hexdigest())
    return h.hexdigest()


TRUST_CACHE = {}


def trusted_ids():
    """Inodes of the host stores and every registered path, recomputed only when
    the registry file changes, because the walk asks once per manifest it meets."""
    try:
        st = paths.PATHS.registry.stat()
        key = (str(paths.PATHS.registry), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(paths.PATHS.registry), None, None)
    key += tuple(str(p) for p in paths.PATHS.host_stores.values())
    if key not in TRUST_CACHE:
        host_ids = {fsio.inode(p) for p in paths.PATHS.host_stores.values()}
        host_ids.discard(None)
        pins = registry_pins()
        rows = {}
        for p in registry():
            ident = fsio.inode(p)
            if ident is not None:
                rows[ident] = pins.get(str(p))
        TRUST_CACHE.clear()
        TRUST_CACHE[key] = (host_ids, rows)
    return TRUST_CACHE[key]


def trusted(d):
    """Whether directory `d` may act as a store: a host store, Claude Code's
    `projects/<slug>/memory`, or a path a human registered (`locket trust`,
    `locket init` at a terminal). A manifest anywhere else is inert, because a
    cloned repository carries whatever manifest its author wrote: trigger rows
    that block every tool call, ledger schemas, `sources` outside the directory."""
    return granted(Path(d))


def granted(d):
    """The real trust check, which the tests' fixture trust never replaces:
    `grant_trust` asks it, so a fixture is still registered for real."""
    ident = fsio.inode(d)
    if ident is None:
        return False
    host_ids, rows = trusted_ids()
    if ident in host_ids:
        return True
    if ident in rows:
        pin = rows[ident]
        return pin is not None and pin == config_digest(d)
    return d.name == "memory" and fsio.inode(d.parent.parent) == fsio.inode(hosts.claude() / "projects")


def changed_since_trust(d):
    """Whether `d` is registered but its configuration no longer matches its pin."""
    ident = fsio.inode(d)
    rows = trusted_ids()[1]
    return ident in rows and rows[ident] is not None and rows[ident] != config_digest(d)


def registry_stale():
    """Registered paths whose manifest is gone. Listed, never enumerated."""
    return [p for p in registry() if not manifest.has_manifest(p)]


def write_store_file(path, text):
    """Write a file Locket owns inside a store (a manifest, a scaffolded ledger) by
    atomic rename, refusing a symlink at the path: a cloned repo's `locket.json`
    linked to `~/.claude/settings.json` must not turn `init` into an overwrite."""
    p = Path(path)
    if p.is_symlink():
        raise OSError(f"{p} is a symlink; Locket writes no file through one")
    # Locket's own edit to a manifest a person trusted keeps that trust: the pin moves
    # to what Locket wrote. Any other edit leaves the store inert until re-trusted.
    repin = (p.name in (manifest.MANIFEST, manifest.OLD_MANIFEST, manifest.LEGACY_MANIFEST) and granted(p.parent)
             and any(fsio.same(p.parent, r) for r in registry()))
    try:                                    # an edited file keeps its mode; a new one takes the umask's
        mode = os.stat(p).st_mode & 0o777
    except FileNotFoundError:
        mask = os.umask(0)
        os.umask(mask)
        mode = 0o666 & ~mask
    fsio.replace_atomic(p, text, f".{p.name}.", mode)
    if repin:
        register(p.parent)


def pin_unpinned():
    """Pin every registered store that has no pin yet (rows from before 0.12) to its
    configuration as it stands, once a person agrees: an unpinned row is untrusted,
    since a registry written by hand carries no pins. Returns the stores pinned, or
    None when a person was needed and none answered."""
    pins = registry_pins()
    todo = [r for r in registry() if str(r) not in pins and manifest.has_manifest(r)]
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
    fsio.replace_atomic(paths.PATHS.registry, json.dumps({"corpora": keep, "pins": pins}, indent=2) + "\n", ".corpora.")


def register(root):
    """Add a corpus path to the registry, pinned to its configuration as it stands,
    or re-pin a row already there. True when the row was new."""
    root = Path(root).resolve()
    rows = registry()
    old = next((r for r in rows if fsio.same(root, r)), None)
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

    for proj in (hosts.claude() / "projects").glob("*"):
        hit = _unslug(proj.name)
        if hit:
            out.add(hit)
        logs = sorted(proj.glob("*.jsonl"), key=mtime, reverse=True)
        for f in logs[:3]:
            cwd = _first_cwd(f)
            if cwd:
                out.add(cwd)
                break
    rollouts = sorted((hosts.codex() / "sessions").glob("*/*/*/*.jsonl"), reverse=True)
    for f in rollouts[:500]:
        cwd = _first_cwd(f, limit=1)
        if cwd:
            out.add(cwd)
    db = hosts.hermes() / "state.db"
    if db.is_file():
        out.update(r[0] for r in fsio.sqlite_rows(
            db, "SELECT DISTINCT cwd FROM sessions WHERE cwd IS NOT NULL AND cwd != ''") or ())
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
            if p == paths.HOME or p == p.parent:
                break
            if manifest.has_manifest(p) and not granted(p):
                found.add(p)
    found.update(r for r in registry() if manifest.has_manifest(r) and changed_since_trust(r))
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
        say(f"  {fsio.relay(line)}")
    sys.stderr.write(f"Type yes to {act}: "); sys.stderr.flush()
    try:
        return sys.stdin.readline().strip() == "yes"
    except (EOFError, OSError):
        return False


def trust_summary(d):
    """What trusting `d` would switch on, one line each, read from the files
    themselves, so the person deciding sees the rows and paths, not a promise."""
    m = manifest.manifest_data(Path(d))
    out = [f"{manifest.manifest_file(Path(d)).name}: keys {', '.join(sorted(m)) or 'none'}"]
    for e in m.get("sources") or []:
        if isinstance(e, dict):
            out.append(f"source file: {e.get('path')}")
    for e in m.get("ledgers") or []:
        if isinstance(e, dict):
            out.append(f"ledger schema for: {e.get('path')}")
    rows_name = m.get("triggers") if isinstance(m.get("triggers"), str) else "triggers.json"
    rows = Path(d) / Path(rows_name).expanduser()
    try:
        data = json.loads(fsio.read_safe(rows))
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
    if granted(d):
        print(f"registry: {paths.PATHS.registry} (already listed)" if any(fsio.same(d, r) for r in registry())
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


def forget(root):
    """Drop a path from the registry. True when a row was removed. The manifest
    and the cache inside the directory are the user's to delete; this only
    stops the listing naming it."""
    root = Path(root).expanduser().resolve()
    rows = registry()
    keep = [r for r in rows if not fsio.same(root, r)]
    if len(keep) == len(rows):
        return False
    _write_registry(keep)
    return True


def read_manifest(root):
    """The manifest as a dict, or {} where it is absent, unreadable or untrusted.
    A broken manifest must degrade to the defaults rather than take a hook down
    with it, and an untrusted one is read as no manifest at all."""
    if root is None or not trusted(root):
        return {}
    return manifest.manifest_data(root)
