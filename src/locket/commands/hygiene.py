"""Whether a corpus still reads true against itself and the world: index lines against
their targets (pointers), wikilinks and CSV query pointers (links), path:line citations
against the code (cites), files past a §5 budget (budget), and ledger rows against their
schema (ledgers).
"""
import re
from collections import defaultdict

import cites
import claims
import corpora
import corpus
import fsio
import gate
import ledger
from commands import Report


def index_hooks(root):
    """(index key, target key, title, hook text, target path) for every index line
    with a target, in every member directory and joined store. A target resolves
    beside its index file, never at the root: a composite store's `memory/MEMORY.md`
    points into `memory/`."""
    out = []
    index_files = corpora.settings(root)["index_files"]
    rs = corpora.roots_of(root)
    for p in corpus.md_files(root):
        if p.name not in index_files:
            continue
        for line in fsio.read_or_empty(p).splitlines():
            m = claims.IDX_LINE.match(line)
            if m:
                hook = m.group("hook").lstrip(" —-").strip()
                if hook:
                    t = p.parent / m.group("target")
                    out.append((corpora.corpus_key(root, p, rs), corpora.corpus_key(root, t, rs), m.group("title"), hook, t))
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
    texts = {p: fsio.read_or_empty(p) for p in corpus.md_files(root)}
    for p, txt in texts.items():
        files.add(p.stem.replace("-", "_"))
        name = front_name(txt)
        if name:
            files.add(name.replace("-", "_"))
    out = defaultdict(list)
    for p, txt in texts.items():
        for t in claims.LINK.findall(txt):
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
        with fsio.open_safe(path, newline="") as fh:
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
    sources = {p.name: p for p, _, _ in corpus.csv_sources(root)}
    cache, out = {}, []
    for p in corpus.md_files(root):
        txt = fsio.read_or_empty(p)
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


def cmd_pointers(root):
    out = []
    hooks = index_hooks(root)
    missing = [(i, t, h) for i, t, _, h, path in hooks if not path.exists()]
    out.append(f"{len(hooks)} index lines with a target; {len(missing)} point at a missing file\n")
    for i, t, h in missing:
        out.append(f"  MISSING TARGET  {i} -> {t}  :: {h[:90]}")
    if missing:
        out.append("  (an index line naming a file that was never written, or was removed: write the "
                   "file or cut the line. A [[wikilink]] with no file is `links`' to report.)")
    for i, t, title, h, _ in hooks:
        out.append(f"\n### {i} -> {t}\nTITLE: {title}\nHOOK: {h}")
    return Report(out, len(hooks))


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
    out = []
    rows, ledgers, history = [], [], {}
    ledger_names = corpora.settings(root)["ledger_surfaces"]
    rs = corpora.roots_of(root)
    for p in corpus.md_files(root):
        if p.name in ledger_names:           # chronology: length is the design, not a breach
            ledgers.append(f"{p.stat().st_size // 1024}KB {corpora.corpus_key(root, p, rs)}")
            continue
        txt = fsio.read_or_empty(p)
        name = corpora.corpus_key(root, p, rs)
        hits = [(i + 1, l.strip()[:110]) for i, l in enumerate(txt.splitlines())
                if gate.HISTORY_CLAUSE.search(l)]
        if hits:
            history[name] = hits
        rows.append((p.stat().st_size, len(hits), max(list(_prose_paras(txt)) or [0]), name))
    if ledgers:
        out.append("ledgers, exempt (append-only chronology, folded in place, never reconsolidated): "
                   + ", ".join(ledgers) + "\n")
    flagged = [r for r in sorted(rows, reverse=True)
               if r[0] > ENTRY_BYTES or r[1] >= UPDATE_CAP or r[2] > PARA_WORDS]
    b = sum(1 for r in flagged if r[0] > ENTRY_BYTES)
    u = sum(1 for r in flagged if r[1] >= UPDATE_CAP)
    w = sum(1 for r in flagged if r[2] > PARA_WORDS)
    out.append(f"{len(rows)} files; {len(flagged)} over an §5 budget — "
               f"{b} past {ENTRY_BYTES // 1024}KB [B], {u} with {UPDATE_CAP}+ update clauses [U], "
               f"{w} with a {PARA_WORDS}+ word paragraph [P]."
               + (f" Largest {min(25, len(flagged))}:\n" if flagged else ""))
    def mark_of(size, upd, para):
        return "".join(c for c, hit in
                       (("B", size > ENTRY_BYTES), ("U", upd >= UPDATE_CAP), ("P", para > PARA_WORDS))
                       if hit)

    if flagged:
        out.append(f"  {'bytes':>7} {'upd':>4} {'para':>5}  file")
    for size, upd, para, name in flagged[:25]:
        out.append(f"  {size:7d} {upd:4d} {para:5d}  {name}  [{mark_of(size, upd, para)}]")

    # The table sorts by bytes, so a small file holding one enormous paragraph
    # never reaches it: a 677-word paragraph in a 10KB file stayed invisible while
    # a 566-word one in a 28KB file showed. Sort [P] separately as well.
    shown = {r[3] for r in flagged[:25]}
    by_para = [r for r in sorted(flagged, key=lambda r: -r[2])
               if r[2] > PARA_WORDS and r[3] not in shown][:10]
    if by_para:
        out.append("\n  Longest paragraphs not in the table above:")
        for size, upd, para, name in by_para:
            out.append(f"  {size:7d} {upd:4d} {para:5d}  {name}  [{mark_of(size, upd, para)}]")
    if history:
        out.append("\n  Dated history clauses, each a passage to rewrite to what is now true "
                   "(prime-memory-discipline §2), never to append to:")
        for name, hits in sorted(history.items(), key=lambda kv: -len(kv[1])):
            out.append(f"  {name} ({len(hits)})")
            for ln, text in hits[:8]:
                out.append(f"    {ln:5d}  {text}")
            if len(hits) > 8:
                out.append(f"          … {len(hits) - 8} more")
    return Report(out, len(flagged))


def cmd_links(root):
    """Dangling links are three different things and only one is a defect.

    RESOLVABLE: written in an older naming convention, pointing at a file that
    does exist under a type prefix. Real rot; repair these.
    unresolved: a forward reference to a memory not written yet, which
    prime-memory-discipline allows, or a pointer to a skill or rule file that was never
    a memory. Neither is a defect, so neither is reported as one.
    """
    out = []
    d = dangling(root)
    stems = [p.stem for p in corpus.md_files(root)]      # same walker as `dangling`, or
                                                  # the classifier and the thing it
                                                  # classifies disagree
    resolvable, unresolved = {}, {}
    for t, files in d.items():
        key = t.replace("-", "_")
        hits = [f for f in stems if f.endswith("_" + key)]
        (resolvable if len(hits) == 1 else unresolved)[t] = (hits, files)
    total = sum(len(v) for v in d.values())
    out.append(f"{len(d)} dangling wikilink targets, {total} occurrences: "
               f"{len(resolvable)} RESOLVABLE, {len(unresolved)} unresolved\n")
    for t, (hits, files) in sorted(resolvable.items()):
        out.append(f"  RESOLVABLE  [[{t}]] -> [[{hits[0]}]]   ({len(set(files))} files)")
    for t, (hits, files) in sorted(unresolved.items()):
        why = f"ambiguous {hits}" if hits else "forward reference or non-memory target"
        out.append(f"  unresolved  [[{t}]]  {why}   <- {', '.join(sorted(set(files)))}")
    ptrs = csv_pointers(root)
    if ptrs:
        bad = [x for x in ptrs if x[4] != "ok"]
        out.append(f"\n{len(ptrs)} CSV query pointers (`file.csv` ... query `Column=value`), "
                   f"{len(ptrs) - len(bad)} resolve, {len(bad)} do not")
        for f, name, col, vals, status in sorted(bad, key=lambda x: (x[4], x[0])):
            out.append(f"  {status:14s} `{name}` query `{col}={vals}`   <- {f}")
    return Report(out, len(resolvable) + sum(1 for x in ptrs if x[4] != "ok"))


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
    out = []
    found = cites.citations(root)
    if not found:
        out.append("no path:line citations. The corpus cites by symbol.")
        return Report(out, 0)
    roots = cites.code_roots(root)
    if not roots:
        out.append("no code tree resolved for this corpus; nothing to check against.")
        return Report(out, 0)
    out.append("searching: " + ", ".join(str(r) for r, _ in roots))
    idx = cites.code_index(roots)

    defects, resolved = [], []
    for (path, line), sources in sorted(found.items()):
        if filters and not any(f in path for f in filters):
            continue
        status, detail = cites.resolve_citation(path, line, idx)
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
            scope = cites.enclosing_scope(lines, line) if path.endswith(".py") else None
            where = scope or lines[line - 1].strip()[:58] or "(blank line)"
            amb = f"  [{n_hits} files match this name]" if n_hits > 1 else ""
            resolved.append(f"   {path}:{line}{amb}\n       {where}\n       <- {who}")

    citing = {f for s in found.values() for f in s}
    total = len(defects) + len(resolved)
    s = lambda n, w: f"{n} {w}" + ("" if n == 1 else "s")
    out.append(f"\n{s(total, 'distinct path:line citation')} across "
               f"{s(len(citing), 'memory file')}. {len(defects)} resolve to nothing.")
    out.append("Cite the SYMBOL instead: no other mode opens the cited file, so these "
               "decay in silence.\n")
    for d in defects:
        out.append(d)
    if defects and resolved:
        out.append("")
    for r in resolved:
        out.append(r)
    return Report(out, len(defects))


def cmd_ledgers(root):
    """Every schema-governed CSV in the corpus, each row checked, then every
    declared CSV past the size reading. A clean run prints its scope, so a zero
    says what it was scored against."""
    out = []
    schemas = ledger.ledger_schemas(root)
    files = ledger.schema_csvs(root)
    big = ledger.large_csvs(root)
    if not files and not big:
        out.append(f"no schema-governed ledger in {root} (schemas: {', '.join(schemas)})")
        return Report(out, 0)
    bad = 0
    for p, schema, glob in files:
        try:
            body = fsio.read_safe(p)
        except OSError as e:
            out.append(f"{corpora.corpus_key(root, p)}  schema {glob!r}  NOT READ: {e}")
            bad += 1
            continue
        findings = ledger.check_ledger_text(schema, body)
        rows = max(len(ledger._csv_rows_of(body)) - 1, 0)
        out.append(f"{corpora.corpus_key(root, p)}  schema {glob!r}  {rows} rows, {len(findings)} findings")
        for f in findings:
            out.append(f"  {f}")
        bad += len(findings)
    for p, n in big:
        out.append(f"{corpora.corpus_key(root, p)}  size {n / 1048576:.1f} MB, past the "
                   f"{ledger.CSV_REPORT_BYTES / 1048576:g} MB reading: run `locket doctor` for the write "
                   f"hook's latency on it, and choose a remedy against what it shows; never split it")
    if big:
        out.append(f"{len(big)} declared CSV(s) past the {ledger.CSV_REPORT_BYTES / 1048576:g} MB reading")
    bad += len(big)
    out.append(f"{len(files)} ledger(s) checked, {bad} finding(s)")
    return Report(out, bad)
