"""Where a corpus holds one claim twice: claim pairs across files (audit), across stores
(across), inside one file (dupes), between a holding space and a durable file
(graduated), and the files an incoming text already restates (against). On a registered
CSV, `dupes` ranks rows by meaning through `rows`, which loads memfind only then.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import claims
import corpora
import corpus
import fsio
from commands import Report, Tool, Verb, as_text, corpus_verb, floor_arg, report_tool, take_n, take_opt


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
                or claims.pointer_only(sa) or claims.pointer_only(sb)):
            dropped += 1
            continue
        kept.append((jac, fa, fb, sa, sb))
    return kept, dropped


def _audit_pairs(root, floor=claims.PAIR_FLOOR):
    """(docs, raw pairs, kept pairs, dropped count) for one corpus."""
    docs = corpus.corpus(root)
    pairs = claims.cross_file_pairs(docs, floor)
    kept, dropped = _filter_pairs(pairs)
    return docs, pairs, kept, dropped


def across_pairs(stores, floor=claims.PAIR_FLOOR):
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
    holders: dict[str, list[str]] = {}
    bodies: dict[str, str] = {}
    names: dict[str, str] = {}
    for k, root in sorted(stores.items()):
        rs = corpora.roots_of(root)
        md = {corpora.corpus_key(root, p, rs): str(p.resolve()) for p in corpus.md_files(root)}
        # keyed as a row label keys it, so two CSVs sharing a name stay two files
        csvs = {k: str(p.resolve()) for p, k in corpus.csv_keys(root).items()}
        for name, body in corpus.corpus(root).items():
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
    pairs = claims.cross_file_pairs(union, floor)
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
        s = corpora.settings(stores[k])
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


def cmd_across(floor=claims.PAIR_FLOOR):
    out = []
    stores = {k: v for k, v in corpora.all_corpora().items() if corpus.md_files(v) or corpus.csv_rows(v)}
    union, pairs, dropped, cross = across_pairs(stores, floor)
    grad = graduated_across(cross, stores)
    out.append(f"{len(stores)} corpora, {len(union)} files; {len(pairs)} raw claim pairs at "
               f"jaccard {floor}; {dropped} filtered as pointer/boilerplate; "
               f"{len(cross)} across corpora to read, {len(grad)} with a holding-space side "
               f"(promoted into another store, never retired; `graduated` scores one store "
               f"and cannot see these)\n")
    by = Counter((fa.split("::", 1)[0], fb.split("::", 1)[0]) for _, fa, fb, _, _ in cross)
    for (a, b), n in by.most_common():
        out.append(f"  {n:4d}  {a}  x  {b}")
    if by:
        out.append("")
    tagged = {(fa, fb, sa) for _, fa, fb, sa, _ in grad}
    for jac, fa, fb, sa, sb in cross:
        tag = " [holding side: promoted, never retired]" if (fa, fb, sa) in tagged or (fb, fa, sb) in tagged else ""
        out.append(f"[{jac:.2f}]{tag} {fa}\n       {fb}\n   A: {fsio.relay(sa)}\n   B: {fsio.relay(sb)}\n")
    out.append("Pairs that currently AGREE, so fork risk rather than divergence: once two copies "
               "drift apart the jaccard falls and nothing here sees them. A store derived from "
               "another (a vault distilled from a host's council store) fires here forever once "
               "registered; read it as expected, never exempt it, and register it, because a "
               "directory no manifest names is watched by nothing and a lesson promoted into it "
               "reads exactly like one never promoted. Inside one host the skills, rules and "
               "agents sit in the council store with the memory, so their restatements are "
               "`audit`'s to report, not this mode's.")
    return Report(out, len(cross))


# A relative markdown link, `](sub/file.md)` or `](file.md#part)`: the target an index line names.
_MD_LINK = re.compile(r"\]\((?![a-z]+:)([^)\s#]+\.md)(?:#[^)]*)?\)")


def index_links(root):
    """{(index key, target key)} for every relative markdown link in the store's index
    files, the target resolved beside the index that names it."""
    index_files = corpora.settings(root)["index_files"]
    rs = corpora.roots_of(root)
    out = set()
    for p in corpus.md_files(root):
        if p.name in index_files:
            key = corpora.corpus_key(root, p, rs)
            for target in _MD_LINK.findall(fsio.read_or_empty(p)):
                out.add((key, corpora.corpus_key(root, Path(os.path.normpath(p.parent / target)), rs)))
    return out


def cmd_audit(root, floor=claims.PAIR_FLOOR):
    out = []
    docs, pairs, kept, dropped = _audit_pairs(root, floor)
    # An index restating a file it links to is the index doing its job; restating any
    # other file, it is a finding like any other pair.
    links = index_links(root)
    own = [p for p in kept if (p[1], p[2]) in links or (p[2], p[1]) in links]
    kept = [p for p in kept if not ((p[1], p[2]) in links or (p[2], p[1]) in links)]
    under = sum(1 for body in docs.values() for _ in claims.unclaimed(body))
    out.append(f"{len(docs)} files; {len(pairs)} raw cross-file claim pairs at jaccard {floor}; "
               f"{dropped} filtered as pointer/boilerplate, {len(own)} as an index line restating "
               f"the file it links to; {len(kept)} to read; {under} sentences under the "
               f"{claims.MIN_TOKENS}-token floor formed no claim and are in nothing above\n")
    for jac, fa, fb, sa, sb in kept:
        out.append(f"[{jac:.2f}] {fa}\n       {fb}\n   A: {fsio.relay(sa)}\n   B: {fsio.relay(sb)}\n")
    return Report(out, len(kept))


def cmd_graduated(root, floor=claims.PAIR_FLOOR):
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
    out = []
    sets = corpora.settings(root)
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
    out.append(f"{len(hits)} holding-space claims also held by a durable file; {scope}\n")
    for jac, fh, fd, sh, sd in hits:
        out.append(f"[{jac:.2f}] {fh}\n       {fd}\n   holding: {fsio.relay(sh)}\n   durable: {fsio.relay(sd)}\n")
    return Report(out, len(hits))


def _tables_named(root, name):
    """[(path, text column, key)] of the registered CSVs a user-supplied name means: its
    key in the store (`sub/RULINGS.csv`), its resolved path, or a bare file name
    (`./x.csv` included), which matches every registered CSV of that name. An exact
    key or path wins over the bare-name set, so a root `RULINGS.csv` stays nameable
    beside a `memory/RULINGS.csv`. More than one match is the caller's to refuse,
    never to pick from."""
    p = Path(name)
    sources = corpus.csv_sources(Path(root))
    keys = corpus.csv_keys(Path(root), sources)
    exact = [(pth, text, keys[pth]) for pth, text, _label in sources
             if keys[pth] == p.as_posix() or os.path.realpath(p) == os.path.realpath(pth)]
    if exact:
        return exact
    return [(pth, text, keys[pth]) for pth, text, _label in sources
            if len(p.parts) == 1 and p.name == pth.name]


def _lexical_table(pth, text, key, top, as_json, why):
    """Word-overlap row pairs inside one CSV, for a machine with no embedder: each row
    is its own document, so a pair is two rows sharing a claim. Said so in the report,
    and on stderr under `--json`, so the worklist still parses."""
    docs = {f"row {i}": row for i, (_label, row) in enumerate(corpus.csv_rows_of(pth, text))}
    best: dict = {}                    # one entry per row pair, at its best-matching claim
    for jac, fa, fb, _sa, _sb in claims.cross_file_pairs(docs, claims.PAIR_FLOOR):
        pair = tuple(sorted((fa, fb)))
        best[pair] = max(best.get(pair, 0.0), jac)
    pairs = sorted(((j, docs[a], docs[b]) for (a, b), j in best.items()), key=lambda t: -t[0])
    pairs = pairs[:top or 30]
    warn = (f"{key}: WORD OVERLAP ONLY. No embedder answered ({why}), so a paraphrased row "
            f"is invisible here; {len(pairs)} pair(s) at jaccard {claims.PAIR_FLOOR}")
    if as_json:
        print(fsio.relay(warn, 400), file=sys.stderr)
        return Report([json.dumps([{"score": round(j, 3), "a_label": key, "a": a,
                                    "b_label": key, "b": b, "verdict": None} for j, a, b in pairs],
                                  indent=1)], len(pairs))
    out = [warn + "\n"] + [f"[{j:.2f}]\n   A: {fsio.relay(a)}\n   B: {fsio.relay(b)}\n" for j, a, b in pairs]
    return Report(out, len(pairs))


def _table_report(root, pth, text, key, top, incoming, against, as_json):
    """One registered CSV by meaning, through `meaning`, or by word overlap when no
    embedder answers. Incoming and covered rows have no word-overlap form: they say so."""
    from commands import meaning              # memfind, and numpy with it, loads only here
    try:
        return meaning.cmd_table_dupes(root, pth, text, key, top=top, incoming=incoming,
                                       against=against, as_json=as_json)
    except meaning._rows().NoEmbedder as e:
        if incoming or against:
            return Report([f"{key}: this needs an embedder, and none answered ({fsio.relay(str(e))})"], 0)
        return _lexical_table(pth, text, key, top, as_json, fsio.relay(str(e)))


def cmd_dupes(root, floor=claims.PAIR_FLOOR, files=None, top=None, incoming=None, against=None,
              as_json=False):
    """Repeated claims INSIDE one file. The uniqueness invariant, measured.

    Defaults to the holding spaces, where uniqueness is the rule (Peer ruling
    2026-08-15); pass `files` to check any other. A ledger is exempt on purpose,
    since repetition is its shape. A reference CSV registered under `sources` holds
    one fact per row and breaks that by paraphrase, so one named here is ranked by
    meaning instead (`meaning.cmd_table_dupes`), and the options apply to it alone.
    A markdown file stays on word overlap: this report has to flag exactly what the
    write hook refuses, and the hook is lexical.
    """
    out: list[str] = []
    found = 0
    tables = [(f, _tables_named(root, f)) for f in files or ()]
    if any(t for _f, t in tables):
        for f, t in tables:
            if len(t) > 1:
                refusal = (f"{f} names {len(t)} registered CSVs ({', '.join(k for _p, _t, k in t)}); "
                           "name one by its path in the store")
                if as_json:                      # stdout stays parseable JSON
                    print(fsio.relay(refusal, 400), file=sys.stderr)
                else:
                    out.append(refusal)
            elif t:
                pth, text, key = t[0]
                rep = _table_report(root, pth, text, key, top, incoming, against, as_json)
                out += rep.lines
                found += rep.count
        files = [f for f, t in tables if not t]
        if not files:
            return Report(out, found)
    docs = corpus.corpus(root)
    missing = []
    if files:
        want = set()
        for f in files:
            hit = _doc_keys(root, f, docs)
            want |= hit
            if not hit:
                missing.append(f)
    else:
        # Holding spaces are named by file name and docs are keyed by path inside
        # the store, so match on the name, as `graduated` does.
        holding = corpora.settings(root)["holding_spaces"]
        want = {k for k in docs if Path(k).name in holding}
    scope = {k: v for k, v in docs.items() if k in want}
    if not scope:
        out.append(f"no file named {', '.join(missing)} in {root}" if missing
                   else f"no holding space in {root}; name a file to check one directly")
        return Report(out, found)
    if missing:
        out.append(f"not in {root}, skipped: {', '.join(missing)}")
    # Same ENTRY_TOKENS floor the write-time check uses. A report that flags what
    # the hook lets through is the drift this instrument exists to prevent.
    pairs = [p for p in claims.cross_file_pairs(scope, floor, same_file=True)
             if not claims.pointer_only(p[3]) and len(claims.tokens(p[3])) >= claims.ENTRY_TOKENS]
    per = Counter(p[1] for p in pairs)
    out.append(f"{len(scope)} file(s) checked for repeated entries: " +
               ", ".join(f"{k} ({per.get(k, 0)})" for k in sorted(scope)) + "\n")
    for jac, fa, _fb, sa, sb in pairs:
        out.append(f"[{jac:.2f}] {fa}\n   A: {fsio.relay(sa)}\n   B: {fsio.relay(sb)}\n")
    return Report(out, found + len(pairs))


def cmd_against(root, text, top=3):
    out = []
    docs = corpus.claim_index(root)
    ranked = claims.score_against(text, docs)[:top]
    for combined, best, name, pair in ranked:
        out.append(f"[{best:.2f}] {name}")
        if pair and best >= 0.4:
            out.append(f"      existing: {pair[1][:180]}")
    return Report(out, len(ranked))


def _across(rest):
    # `across all` is what every other Step 5 line teaches; it is not a floor.
    floor = next((float(a) for a in rest[:1] if re.fullmatch(r"[0-9.]+", a)), claims.PAIR_FLOOR)
    cmd_across(floor).show()
    return 0


def _doc_keys(root, name, docs):
    """The store's files a user-supplied name resolves to: its key in the store, an
    absolute path under the store or a joined store, or a bare file name or
    trailing path, which matches that file in every member directory holding one.
    An exact key wins over a trailing-path match."""
    if name in docs:
        return {name}
    p = Path(name)
    if p.is_absolute():
        # The file's own symlink is kept (a store file may link outside the store);
        # the root is tried as given and resolved, since either may be the
        # spelling the registry holds.
        roots = corpora.roots_of(Path(root))
        for r, f in ((Path(root), Path(os.path.abspath(p))),
                     (Path(root).resolve(), Path(os.path.realpath(p.parent)) / p.name)):
            key = corpora.corpus_key(r, f, roots if r == Path(root) else None)
            if key in docs:
                return {key}
        return set()
    tail = "/" + p.as_posix()
    return {k for k in docs if ("/" + Path(k).as_posix()).endswith(tail)}


def _dupes(root, rest):
    """`dupes`'s command line: file names, and the options a registered CSV takes. An
    option the verb does not know, one left without its value, or one that cannot
    apply is refused rather than read as a file name or dropped."""
    try:
        top, rest = take_n(rest, None)
        incoming, rest = take_opt(rest, "--incoming")
        against, rest = take_opt(rest, "--against")
    except ValueError as e:
        return Report([str(e)], 0)
    as_json = "--json" in rest
    files = [f for f in rest if f != "--json"]
    unknown = [f for f in files if f.startswith("-")]
    if unknown:
        return Report([f"dupes does not take {', '.join(unknown)}; see  locket dupes -h"], 0)
    if incoming and against:
        return Report(["--incoming and --against are two different questions; ask one at a time"], 0)
    if as_json and (incoming or against):
        return Report(["--json writes the row-pair worklist only; drop --incoming or --against"], 0)
    if (incoming or against or as_json or top) and not any(_tables_named(root, f) for f in files):
        return Report(["-n, --incoming, --against and --json apply to a CSV registered under "
                       "`sources`; name one"], 0)
    return cmd_dupes(root, files=files or None, top=top, incoming=incoming, against=against,
                     as_json=as_json)


def _files(args):
    files = args.get("files") or None
    return [files] if isinstance(files, str) else files     # a bare string would match as a substring


VERBS = (
    Verb("audit", "[corpus [floor]]", "claim pairs across files that agree: fork risk",
         "locket audit                          this directory's store\n"
         "locket audit all                      every store, one after another\n"
         "locket audit notes 0.7                only pairs at least 0.7 alike (default 0.55, 1.0 is identical)",
         run=corpus_verb(lambda r, rest: cmd_audit(r, floor_arg(rest, claims.PAIR_FLOOR))),
         mcp=report_tool("Claims that appear in two files: fork risk, not proof.",
                         lambda r, a: cmd_audit(r))),
    Verb("across", "[floor]", "claim pairs split across two stores; always reads every store",
         "locket across                         pairs at least 0.55 alike\n"
         "locket across 0.8                     only near-identical pairs",
         run=_across,
         mcp=Tool("Claims held by two different stores, over every store on "
                  "this machine at once. Each store's own checks score it "
                  "against itself, so a fact written into two stores is "
                  "invisible to both until this runs. Fork risk, not proof.",
                  lambda a, s: as_text(cmd_across()) or "no memory stores with markdown files found")),
    Verb("dupes", "[corpus] [file ...] [-n N] [--incoming rows.csv | --against file.md] [--json]",
         "claims repeated inside one file; use on scratch and inbox files, or a registered CSV",
         "locket dupes                          the scratch and inbox files the store names\n"
         "locket dupes council SHORT_TERM.md    one file, named as the store lists it\n"
         "locket dupes notes RULINGS.csv        a registered CSV's closest row pairs, by meaning\n"
         "locket dupes notes RULINGS.csv -n 50 --json   the same as a worklist with a verdict to fill\n"
         "locket dupes notes RULINGS.csv --incoming new.csv   each new row against its nearest row\n"
         "locket dupes notes RULINGS.csv --against rules/x.md rows a whole-loaded file already states",
         # no header under --json, so the worklist redirected to a file parses as JSON
         run=lambda rest: corpus_verb(_dupes, header="--json" not in rest)(rest),
         mcp=report_tool("Claims repeated inside one file (holding spaces by default). A CSV "
                         "registered under `sources` named here has its 30 closest row pairs "
                         "ranked by meaning instead: a worklist, never a verdict.",
                         lambda r, a: cmd_dupes(r, files=_files(a)),
                         files={"type": "array", "items": {"type": "string"},
                                "description": "File names to check instead of the defaults."})),
    Verb("graduated", "[corpus [floor]]", "scratch-file entries a permanent file already states: ready to clear",
         "locket graduated                      this directory's store\n"
         "locket graduated notes 0.7            only entries at least 0.7 alike (default 0.55)",
         run=corpus_verb(lambda r, rest: cmd_graduated(r, floor_arg(rest, claims.PAIR_FLOOR))),
         mcp=report_tool("Holding-space claims a durable file or CSV row now holds: "
                         "entries promoted and never retired. Prints its scope even "
                         "when clean. Run with locket_links at session close.",
                         lambda r, a: cmd_graduated(r))),
    Verb("against", "[corpus] < text", "rank files by how much of the text on stdin they already state, "
         "word by word; what the write hook runs",
         'echo "a sentence" | locket against council\n'
         "locket against notes < draft.md       a whole draft before it is written",
         run=corpus_verb(lambda r, rest: cmd_against(r, sys.stdin.read()), header=False)),
)

SEE = "scan"
