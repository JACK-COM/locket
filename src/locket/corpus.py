"""A store's content: the markdown files its walk covers, pruned on the way down and
bounded in count and bytes, its registered CSV rows, its claims keyed by file, and its
archive, whose claims are subtracted as provenance.
"""
import fnmatch
import json
import os
import stat
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import claims
import corpora
import fsio
import health
import hosts
import manifest
import paths
import trust


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
        return corpora.top_of(paths.PATHS.host_stores[name])
    p = Path(name).expanduser()
    if p.is_dir():
        return corpora.top_of(p.resolve())
    listed, low = corpora.all_corpora(), name.lower()
    # A Claude Code project store also answers to its slug, though a named one is
    # listed only by name. The passes run exact key, exact slug, key substring,
    # slug substring, first non-empty wins: an exact slug then beats a longer
    # sibling slug containing it (`-…-DevDocs` opening `-…-DevDocs-Persona`), and
    # `devdocs` beside `devdocs-persona` is not ambiguous. Slugs come only from
    # `projects/`, or a registered `~/vault/memory` would answer to `vault`.
    slugs = {d.parent.name: d for d in listed.values() if d.parent.parent == hosts.claude() / "projects"}
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
    for k, v in sorted((hits or corpora.all_corpora()).items()):
        print(f"  {k:70s} {len(md_files(v)):4d} files", file=sys.stderr)
    return None


def md_files(root):
    """Every markdown file the corpus contains: the root's own, subdirectories
    included, then each joined store's own. Sorted by path, so a joined store's
    files sit together.

    ⚠ This must agree with `corpus_for`, which walks PARENTS and therefore covers
    subdirectories. A flat `glob("*.md")` here scores a subdirectory at write time
    against a corpus it is not a member of, so forks inside it are invisible while
    the nudge still fires on it. One walker, both ends.
    """
    return sorted(p for r in corpora.roots_of(root) for p in _own_md(r))


def _own_md(root, pattern="*.md", honor_excluded_files=True):
    """The markdown physically under one store, by that store's own settings.
    `pattern` walks another file type the same way (the ledger sweep takes
    `*.csv`), and a sweep that must see a file the manifest hides from claims
    passes `honor_excluded_files=False`."""
    root = Path(root)
    return sorted(p for p in _walk_files(root, pattern)
                  if not (honor_excluded_files and corpora.excluded_file(root, p)))


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
    s = corpora.settings(root)
    skip, mem = s["excluded_dirs"], s["members"]
    listed = {r for r in corpora._raw_corpora().values() if r != root and root in r.parents}

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
            if here != root and (here in listed or manifest.MANIFEST in filenames
                                 or manifest.OLD_MANIFEST in filenames):
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
    return any(any(nonempty(p) and not corpora.excluded_file(r, p) for p in _walk_files(r))
               or (csv and any(nonempty(Path(pth)) for pth, _t, _l in csv_sources(r)))
               for r in corpora.roots_of(root))


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
        elif st.st_size > fsio.MAX_FILE_BYTES and not is_csv:
            out.append((p, f"over {fsio.MAX_FILE_BYTES // (1024 * 1024)} MB"))
    return out


def _own_csv_sources_unchecked(root):
    """The declared CSV paths, links included, so `skipped_files` can name the ones
    `_own_csv_sources` drops."""
    entries = trust._read_manifest(root).get("sources")
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
            if not (fnmatch.fnmatch(n, pattern) and fsio._regular(p)):
                continue
            count += 1
            try:
                size += os.lstat(p).st_size
            except OSError:
                continue
            if count > fsio.MAX_STORE_FILES or size > fsio.MAX_STORE_BYTES:
                health.health_error("walk", "bound", OSError(
                    f"{root}: walk stopped past {fsio.MAX_STORE_FILES} files or {fsio.MAX_STORE_BYTES} bytes"))
                return
            yield p


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
    for r in corpora.roots_of(root):
        for pth, text, label in _own_csv_sources(r):
            key = pth.resolve()               # two stores registering one CSV is one source
            if key not in seen:
                seen.add(key)
                out.append((pth, text, label))
    return out


def _own_csv_sources(root):
    """The CSVs one physical store's manifest registers. A relative `path`
    resolves against that store's root."""
    entries = trust._read_manifest(root).get("sources")
    if not isinstance(entries, list):
        m = Path(root) / manifest.LEGACY_MANIFEST
        if not m.is_file() or not trust.trusted(root):
            return []
        try:
            entries = json.loads(fsio.read_safe(m))
        except (json.JSONDecodeError, OSError):
            return []
    out = []
    try:
        for e in entries:
            pth = Path(e["path"]).expanduser()
            if not pth.is_absolute():
                pth = Path(root) / pth
            if pth.exists() and not corpora.excluded_file(root, pth):
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
            with fsio.open_safe(pth, newline="", max_bytes=None) as fh:   # declared CSVs are never capped
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
    rs = corpora.roots_of(root)
    docs = {}
    for p in md_files(root):
        # One unreadable file (mode 000, or deleted since the walk) is skipped: raised,
        # it emptied the whole read inside the hook, whose catch-all let the write through.
        try:
            docs[corpora.corpus_key(root, p, rs)] = claims.strip(fsio.read_safe(p))
        except OSError:
            continue
    if csv:
        grouped = defaultdict(list)
        for label, text in csv_rows(root):
            grouped[label].append(claims.strip(text))
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
            body = fsio.read_safe(p)
        except OSError:
            continue
        out.extend(ts for _, ts in claims.claims(claims.strip(body)))
    return out


def _archive_md(root):
    """Markdown under every directory named `archive` inside the store's members
    (`memory/archive/` under the council home, `archive/` in a flat store) and
    inside each joined store's, so a move out of any member surface keeps its
    provenance. Skips a nested store, a dot-directory and any other excluded
    directory on the way down."""
    for r in corpora.roots_of(root):
        yield from _own_archive_md(r)


def _own_archive_md(root):
    # Each `archive/` is found as a child of a directory the walk enters, because
    # the walk prunes `archive` itself while it stays in `excluded_dirs`. An
    # archive carrying its own manifest is a nested store, not provenance. A host
    # root's own `archive/` is host state like its other directories.
    root = Path(root)
    hosted = corpora.settings(root)["members"] is not None
    for d, _ in _corpus_dirs(root):
        if hosted and d == root:
            continue
        a = d / "archive"
        if a.is_dir() and not a.is_symlink() and not manifest.has_manifest(a):
            found = []
            for dirpath, dirnames, filenames in os.walk(a):     # never through a linked directory
                found += [Path(dirpath) / n for n in filenames
                          if n.endswith(".md") and fsio._regular(Path(dirpath) / n)]
                if len(found) > fsio.MAX_STORE_FILES:
                    health.health_error("walk", "bound", OSError(f"{a}: archive past {fsio.MAX_STORE_FILES} files"))
                    break
            yield from sorted(found[:fsio.MAX_STORE_FILES])


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
        return claims.strip(added)
    return "\n".join(s for s, ts in claims.claims(claims.strip(added))
                     if not any(claims.similar(ts, a) >= claims.FIRE_FLOOR for a in arch))
