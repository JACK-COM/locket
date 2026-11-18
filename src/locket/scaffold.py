"""Making a store: init writes a manifest and its ledgers into a directory and registers
it, schema writes the manifest schema, and migrate renames a store's older manifest.
"""
import json
import re
import sys
from pathlib import Path

import corpora
import corpus
import fsio
import hosts
import ledger
import manifest
import paths
import trust


def migrate(root):
    """Rename a store's `memfind.json` to `locket.json` in place. Returns
    "migrated", "already" or "none". Content untouched, so a store under git
    sees one rename."""
    root = Path(root).expanduser().resolve()
    if (root / manifest.MANIFEST).is_file():
        return "already"
    old = root / manifest.OLD_MANIFEST
    if not old.is_file():
        return "none"
    repin = trust._granted(root) and any(fsio._same(root, r) for r in trust.registry())
    old.rename(root / manifest.MANIFEST)
    if repin:
        trust.register(root)
    corpora.settings.cache_clear()
    return "migrated"


def cmd_schema(quiet=False):
    """Write the manifest schema beside the registry, so an editor can be pointed
    at one predictable path per machine and the tool can check a manifest against
    the same file. Rerun after an update; `init` runs it."""
    schema = paths.PATHS.schema_file
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(json.dumps(manifest.MANIFEST_SCHEMA, indent=2) + "\n")
    if not quiet:
        print(f"schema: {schema} (written)")
        print(f"editor: VS Code reads it by filename through one user-settings entry; "
              f"`locket help install` (Later) names it")
    return 0


def _stores_under(d):
    """Corpora strictly inside directory `d`: every listed store (host, auto-memory
    and registry) whose parents include `d`, plus the first hand-written manifest
    found below it. Sorted, host stores first, so the message names a real one."""
    d = Path(d)
    found = sorted({r for r in corpora._raw_corpora().values() if r != d and d in r.parents})
    if found:
        return found
    hit = next((m.parent for name in (manifest.MANIFEST, manifest.OLD_MANIFEST)
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
    if d.is_dir() and d.resolve() in (Path("/"), paths.HOME.resolve()):
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
        ledgers = re.sub(r"[^\w.-]+", "-", name or corpora._label(d)).strip("-.") or "notes"
    inside = corpora._phys_root_of(d)
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
    m = manifest.manifest_file(d)
    try:
        cmd_schema(quiet=True)              # the editor's copy tracks the installed tool
    except OSError:
        pass
    for f in manifest.validate_manifest(d):
        print(f"manifest: {f}", file=sys.stderr)
    if parent is not None:
        if d in set(paths.PATHS.host_stores.values()):
            print(f"{d} is a host store and cannot belong to another corpus", file=sys.stderr)
            return 1
        raw = corpora._raw_corpora()
        target = raw.get(parent)
        if target is None:
            print(f"no corpus is named {parent!r}; `corpora` lists the names", file=sys.stderr)
            return 1
        if target == d:
            print(f"{d} cannot belong to itself", file=sys.stderr)
            return 1
        if d in target.parents or corpora._walked_by(target, d):
            print(f"{d} and {target} nest on disk inside {target.name}'s walk; a nested "
                  f"directory there is a `members` entry, not a joined store", file=sys.stderr)
            return 1
        label = name or corpora._label(d)
        if "/" in label or label in (".", ".."):
            print(f"{label!r} cannot be a label: it is prefixed to paths", file=sys.stderr)
            return 1
        if label == corpora._label(target) or (target / label).is_dir():
            print(f"{d}'s label {label!r} collides with a directory or the name of {target}; "
                  f"label the joining store: locket init {d} --parent {parent} --name <label>",
                  file=sys.stderr)
            return 1
        man = dict(manifest._manifest_data(d))
        if name:
            man["name"] = name
        man["belongs_to"] = parent
        trust.write_store_file(m, json.dumps(man, indent=2) + "\n")
        corpora.settings.cache_clear()
        if not trust.grant_trust(d, confirm):
            return 1
        corpora.settings.cache_clear()
        if corpora.parent_of(d, corpora._raw_corpora()) != target:
            print(f"{d} did not resolve to {target} after the write; check for a cycle "
                  f"through {parent!r}", file=sys.stderr)
            return 1
        print(f"manifest: {m} (belongs_to {parent!r})")
        if ledgers is not None and scaffold_ledgers(d, ledgers):
            return 1
        top = corpora.top_of(d)
        stale = d / ".memfind"
        if stale.is_dir():
            print(f"index: {stale} is no longer read; delete it")
        n, rows = len(corpus.md_files(top)), len(corpus.csv_rows(top))
        print(f"corpus: {top}  {n} files" + (f" + {rows} csv rows" if rows else "")
              + f"  [{len(corpora.roots_of(top)) - 1} joined]")
        if not index:
            return 0
        try:
            import memfind
            memfind.build_index(top, quiet=True)
            print(f"index: built for {top}")
        except ImportError as e:              # memfind.py absent, or numpy, which it needs
            print(f"index: not built ({e}); run  locket index {top}")
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
        other = corpora._raw_corpora().get(d.name)
        if other is not None and other != d:
            print(f"the name {d.name!r} is taken by {other}; name this store:  "
                  f"locket init {d} --name <label>", file=sys.stderr)
            return 1
    if name is not None:
        if "/" in name or name in (".", "..") or not name.strip():
            print(f"{name!r} cannot be a name: it is prefixed to paths", file=sys.stderr)
            return 1
        other = corpora._raw_corpora().get(name)
        if other is not None and other != d:
            print(f"the name {name!r} is taken by {other}; `corpora` lists the names", file=sys.stderr)
            return 1
    by_convention = inside == d and not m.is_file()
    if m.is_file():
        by_convention = d in set(paths.PATHS.host_stores.values()) or (
            d.name == "memory" and d.parent.parent == hosts.claude() / "projects")
        man = manifest._manifest_data(d)
        if name is not None and man.get("name") != name:
            trust.write_store_file(m, json.dumps({**man, "name": name}, indent=2) + "\n")
            print(f"manifest: {m} (kept, name {name!r})")
        else:
            print(f"manifest: {m} (kept)")
    elif by_convention and name is None:
        print(f"manifest: none needed; {d} is a corpus by convention on this host")
    else:
        trust.write_store_file(m, json.dumps({"name": name or d.name}, indent=2) + "\n")
        print(f"manifest: {m} (written, name {name or d.name!r})")
    corpora.settings.cache_clear()
    if by_convention:
        print("registry: not needed; listed by convention")
    else:
        if not trust.grant_trust(d, confirm):
            return 1
        corpora.settings.cache_clear()
    if ledgers is not None and scaffold_ledgers(d, ledgers):
        return 1
    n, rows = len(corpus.md_files(d)), len(corpus.csv_rows(d))
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
    except ImportError as e:              # memfind.py absent, or numpy, which it needs
        print(f"index: not built ({e}); run  locket index {d}")
    except RuntimeError as e:
        print(f"index: not built ({e}); run  locket index {d}  once an embedder answers")
    except OSError as e:
        print(f"index: not built, the cache could not be written ({e})")
    return 0


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
    m = manifest.manifest_file(d)
    man = dict(trust._read_manifest(d))
    # ⚠ `_read_manifest` reads an unparseable file as {}, and writing that back
    # would erase every key the user had; refuse instead
    if not man and m.is_file() and fsio.read_or_empty(m).strip():
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
        schema = ledger.ledger_schema_for(f)[0] or manifest.LEDGER_SCHEMAS[builtin]   # a store's `extends` wins
        if f.exists():
            print(f"ledger: {f.name} (kept)")
        else:
            trust.write_store_file(f, ",".join(map(str, schema["columns"])) + "\n")
            print(f"ledger: {f.name} (written, header only)")
        if text and f.resolve() not in listed:
            sources.append({"path": f.name, "text": text, "label": "Area"})
            added.append(f.name)
    if added:
        trust.write_store_file(m, json.dumps({**man, "sources": sources}, indent=2) + "\n")
        corpora.settings.cache_clear()
        print(f"manifest: {', '.join(added)} registered as sources")
    return 0
