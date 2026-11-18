"""The decidable checks on a CSV write: the ledger schema a store declares, checked row
by row, and a registered source's new rows scored against the prose of its store.

A schema violation blocks, because whether a row fits its ledger is decidable from the
row; a source row that restates a claim warns, because whether a fact is new is not.
"""
import fnmatch
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Optional

import claims
import corpora
import corpus
import fsio
import manifest
import trust


# One nudge, two forces, so the opening clause has to state which happened. Kept
# as constants because the gate rewrites a warning into the blocking form when the
# 2e gate refuses the same write, and two string literals drifting apart would
# leave the message asserting a write landed when it did not.
LEAD_BLOCKING = "A memory is about to be written."
LEAD_WARNING = "This memory write landed."


def ledger_schemas(root):
    """{glob: schema} in force for a corpus: the built-ins, then every `ledgers`
    entry in the manifest of each physical store composing it, later entries
    overriding earlier ones by glob. Malformed entries are skipped."""
    out = {k: dict(v, _from="built in") for k, v in manifest.LEDGER_SCHEMAS.items()}
    for r in (corpora.roots_of(root) if root is not None else []):
        entries = trust.read_manifest(r).get("ledgers")
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("path"), str):
                continue
            own = {k: e[k] for k in manifest.SCHEMA_KEYS if k in e}
            base = manifest.LEDGER_SCHEMAS.get(str(e.get("extends"))) or _builtin_for(e["path"])
            out[e["path"]] = _merge_schema(base, own)
            out[e["path"]]["_from"] = f"{manifest.manifest_file(r).name} in {r}"
    return out


def _builtin_for(path):
    """The built-in schema a manifest entry's path falls under, by its own text
    or its basename, or {}."""
    name = PurePosixPath(path).name
    for glob, schema in manifest.LEDGER_SCHEMAS.items():
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
    phys = corpora.phys_root_of(p.parent)
    if phys is None:
        return None, None
    root = corpora.top_of(phys)
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
                ok = v is None or (len(v) <= fsio.MAX_SCAN and re.fullmatch(pat, v) is not None)
            except re.error:
                ok = True
            if not ok:
                out.append(f"row {n}: {c} is {v!r}, wanted /{pat}/")
    return out


def csv_rows_of(text):
    import csv as _csv, io as _io
    return list(_csv.reader(_io.StringIO(text)))


def check_ledger_text(schema, text, header=None):
    """Findings for CSV `text`. With `header`, every parsed line is a data row
    (an Edit fragment, a heredoc); a line whose field count differs from the
    header's is not a row and is skipped, since a fragment can cut a row. Without
    it, the first line is the header (a whole-file Write, the sweep)."""
    rows = csv_rows_of(text)
    if header is None:
        if not rows:
            return []
        return check_ledger_rows(schema, rows[0], rows[1:], first=2)
    rows = [r for r in rows if len(r) == len(header)
            and [c.strip() for c in r] != [c.strip() for c in header]]
    return check_ledger_rows(schema, header, rows, first=1)


def ledger_header(path):
    try:
        with fsio.open_safe(path, newline="") as fh:
            rows = csv_rows_of(fh.read(4096))
    except OSError:
        return None
    return rows[0] if rows else None


def ledger_message(path, glob, schema, findings, via):
    return (f"A ledger row breaks its schema and the write is refused ({Path(path).name}, "
            f"schema {glob!r}, {schema.get('_from', 'built in')}):\n  - " + "\n  - ".join(fsio.relay(f, 240) for f in findings[:12])
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
        header = ledger_header(path)
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
    phys = corpora.phys_root_of(p.parent)
    if phys is None:
        return None, False
    root = corpora.top_of(phys)
    src = [(s, t, lab) for s, t, lab in corpus.csv_sources(root) if s.resolve() == p.resolve()]
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
    added: list[str] = []
    seen: Counter[str] = Counter()
    for t in after:
        seen[t] += 1
        if t and seen[t] > before[t]:
            added.append(t)
    if not added:
        return None, False
    # The file's own rows are scored as they will stand after the write, minus the
    # new ones, so an edited row never matches the version it replaces.
    index, own = corpus.claim_index(root), f"{p.name}#"
    kept = Counter(after) - Counter(added)
    stands = {p.name: "\n".join(claims.strip(t) for t in kept.elements() if t)}
    lines = []
    for row in added:
        ranked = [r for r in claims.score_against(row, index, hide=lambda k: k.startswith(own), extra=stands)[:3]
                  if r[1] >= claims.FIRE_FLOOR]
        if ranked:
            lines.append(f"  row: {fsio.relay(row, 120)}")
            for _c, best, fname, pair in ranked:
                lines.append(f"    - {fname}   (closest claim {best:.2f})"
                             + (f"\n        existing: {fsio.relay(pair[1], 160)}" if pair else ""))
    if not lines:
        return None, False
    return (f"{LEAD_WARNING} {p.name} is a registered source, and a row here may restate a claim "
            f"the store already carries:\n" + "\n".join(lines) +
            "\nOne fact, one row: cut the prose copy to a pointer, or fold the row into the one "
            "that owns it (prime-memory-discipline.md §2c). A new fact needs nothing."), False


# The size past which `ledgers` and `doctor` report a declared CSV. A report, never
# a cap: the file keeps every row and stays whole (RULINGS.csv on Area=substrate).
CSV_REPORT_BYTES = 5 * 1024 * 1024


def schema_csvs(root):
    """[(path, schema, glob)] for every schema-governed CSV the store's walk finds."""
    seen, files = set(), []
    # The store's own walker, members and excluded directories honoured, so a
    # ledger under a member directory (a store's `memory/RULINGS.csv`) is swept;
    # a root-only glob reported such a store clean while checking nothing.
    # `excluded_files` hides a file from claims, not from its schema.
    for r in corpora.roots_of(root):
        for p in corpus.own_md(r, "*.csv", honor_excluded_files=False):
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
    out: list[tuple[Path, Optional[str]]] = []
    seen = set()
    for p, text, _label in corpus.csv_sources(root):
        seen.add(p.resolve())
        out.append((p, text))
    for p, _schema, _glob in schema_csvs(root):
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
