"""Code citations in a memory store: which source trees a store's citations point into, an
index of each tree's definitions, and whether a cited file:line still holds what the
citation names.
"""
import os
import posixpath
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import corpora
import corpus
import fsio
import hosts
import trust


# ------------------------------------------------------------- code citations
#
# A `path:line` written into a memory file is the one claim class every other
# mode is blind to: `audit` compares claims to claims, the write-time gate
# scores novelty, `memfind` ranks semantics, and not one of them opens the cited
# file. So a citation is correct for exactly one commit and then decays in
# silence. On one production corpus, of 54 citations,
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
        rel = Path(corpus_root).resolve().relative_to((hosts.claude() / "projects").resolve())
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
    for r in corpora.roots_of(corpus_root):
        for tree, skips in _own_code_roots(r):
            if tree not in seen:
                seen.add(tree)
                roots.append((tree, skips))
    return roots


def _own_code_roots(corpus_root):
    if not trust.trusted(corpus_root):
        return []                            # an untrusted folder's citations reach no host file
    declared = corpora.settings(corpus_root)["code_roots"]
    if declared is not None:                 # the manifest names them outright
        return [((Path(corpus_root) / d) if not d.is_absolute() else d, frozenset())
                for d in declared if ((Path(corpus_root) / d) if not d.is_absolute() else d).is_dir()]
    roots: list[tuple[Path, frozenset]] = []
    proj = project_dir_for(corpus_root)
    if proj:
        roots.append((proj, frozenset()))
    for home in (hosts.claude(), hosts.hermes()):
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
        sizes: Counter[str] = Counter()
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
                      "state rather than code, it belongs in cites.CLAUDE_SKIP", file=sys.stderr)
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
    # failure this mode exists to catch, and it gets correct citations "fixed".
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
    for p in corpus.md_files(root):
        for m in CITE.finditer(fsio.read_or_empty(p)):
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
            lines = fsio.read_safe(h).splitlines()     # never through a link, a FIFO or past the cap
        except OSError:
            continue
        if line <= len(lines):
            return "ok", (h, lines, len(hits))
        first = first or (h, len(lines))
    if first is None:
        return "missing", 0
    return "eof", (first[0], first[1], len(hits))
