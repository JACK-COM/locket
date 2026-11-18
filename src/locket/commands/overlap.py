"""Where a corpus holds one claim twice: claim pairs across files (audit), across stores
(across), inside one file (dupes), between a holding space and a durable file
(graduated), and the files an incoming text already restates (against).
"""
from collections import Counter, defaultdict
from pathlib import Path

import claims
import corpora
import corpus
import fsio
from commands import Report


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
    holders, bodies, names = {}, {}, {}
    for k, root in sorted(stores.items()):
        rs = corpora.roots_of(root)
        md = {corpora.corpus_key(root, p, rs): str(p.resolve()) for p in corpus.md_files(root)}
        csvs = {p.name: str(p.resolve()) for p, _, _ in corpus.csv_sources(root)}
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


def cmd_audit(root, floor=claims.PAIR_FLOOR):
    out = []
    docs, pairs, kept, dropped = _audit_pairs(root, floor)
    index_files = corpora.settings(root)["index_files"]
    idx_pairs = [p for p in kept if Path(p[1]).name in index_files or Path(p[2]).name in index_files]
    under = sum(1 for body in docs.values() for _ in claims.unclaimed(body))
    out.append(f"{len(docs)} files; {len(pairs)} raw cross-file claim pairs at jaccard {floor}; "
               f"{dropped} filtered as pointer/boilerplate; {len(kept)} to read "
               f"({len(idx_pairs)} index-restates-its-target); {under} sentences under the "
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


def cmd_dupes(root, floor=claims.PAIR_FLOOR, files=None):
    """Repeated claims INSIDE one file. The uniqueness invariant, measured.

    Defaults to the holding spaces, where uniqueness is the rule (Peer ruling
    2026-08-15); pass `files` to check any other. A ledger is exempt on purpose,
    since repetition is its shape.
    """
    out = []
    docs = corpus.corpus(root)
    want = files or sorted(corpora.settings(root)["holding_spaces"] & set(docs))
    scope = {k: v for k, v in docs.items() if k in want}
    if not scope:
        out.append(f"no holding space in {root}; name a file to check one directly")
        return Report(out, 0)
    # Same ENTRY_TOKENS floor the write-time check uses. A report that flags what
    # the hook lets through is the drift this instrument exists to prevent.
    pairs = [p for p in claims.cross_file_pairs(scope, floor, same_file=True)
             if not claims.pointer_only(p[3]) and len(claims.tokens(p[3])) >= claims.ENTRY_TOKENS]
    per = Counter(p[1] for p in pairs)
    out.append(f"{len(scope)} file(s) checked for repeated entries: " +
               ", ".join(f"{k} ({per.get(k, 0)})" for k in sorted(scope)) + "\n")
    for jac, fa, _fb, sa, sb in pairs:
        out.append(f"[{jac:.2f}] {fa}\n   A: {fsio.relay(sa)}\n   B: {fsio.relay(sb)}\n")
    return Report(out, len(pairs))


def cmd_against(root, text, top=3):
    out = []
    docs = corpus.claim_index(root)
    ranked = claims.score_against(text, docs)[:top]
    for combined, best, name, pair in ranked:
        out.append(f"[{best:.2f}] {name}")
        if pair and best >= 0.4:
            out.append(f"      existing: {pair[1][:180]}")
    return Report(out, len(ranked))
