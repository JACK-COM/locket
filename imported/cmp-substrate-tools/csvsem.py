#!/usr/bin/env python3
"""Semantic near-duplicate detection over the CMP reference CSVs.

  csvsem.py dupes  [rulings|claims|both] [-n N]   rank the most-similar existing pairs
  csvsem.py check  <batch.json> [-n N]            score incoming rows against what exists

Why this exists: `csvtool.py merge` compares CHARACTERS (a content-token floor, then
SequenceMatcher >= 0.88). That catches a verbatim row and a Never->Do not edit, and it
misses a paraphrase completely — measured 2026-08-22, when two rewordings of the live
"never write a result cap at a layer that cannot see which filter resolves" row both
merged clean. `memscan audit` and the write-time hook do not cover these files at all,
because they resolve a corpus from a path under memory/ and the CSVs sit at the
workspace root.

★★ It RANKS and never decides. memfind.py's own docstring records a novel-versus-
paraphrase threshold as having FAILED: the two distributions overlap, so any cutoff is
wrong often enough to be worse than no cutoff (memory-discipline 2d — an instrument may
block only on a question it can DECIDE). Output is a reader's worklist, not a verdict,
and nothing here refuses a write.

Measured on the full claims corpus 2026-08-22, adjudicating all 226 flagged pairs: 50%
precision in the 0.85+ band and 21% overall, so claims is NOT blockable at any cutoff.
Rulings ran 46 to 58% overall over the same detector. The dominant claims false positive
is a per-variant certificated-limits row (PA-28 TCDS 2A13 items, Cessna 525 series) that
scores to 0.889 while naming a different model, so a high score is no evidence about
subject identity. Re-run `adjudicate.py score` rather than quoting these figures.

Mean-centring is required rather than cosmetic. nomic's space is anisotropic: raw cosine
puts almost every pair between 0.97 and 0.99. Subtracting the mean vector recovers a
usable spread. Needs ollama serving nomic-embed-text; degrades to a clear message.

⚠ Keep this vector cache even though `memfind` now indexes the same rows. The two stores
are not interchangeable: memfind stores each vector L2-NORMALISED and nothing else, while
every figure below was measured by centring on the corpus mean FIRST and normalising after.
Reusing memfind's vectors here would shift every score and silently orphan the calibration,
so the apparent one-authority-per-fact merge is the wrong move. memfind answers *which rows
resemble this statement*, one query at a time; this answers *which rows resemble each other*,
all pairs at once, and only the second needs the centring.
"""
import csv, hashlib, json, math, os, sys
from array import array

sys.path.insert(0, os.path.expanduser("~/.claude/scripts"))
import memfind

ROOT = "/Users/raphaelokure/Software/jackcom-projects/choose-my-plane"
FILES = {"rulings": (os.path.join(ROOT, "REFERENCE-CMP-Rulings.csv"), "Ruling"),
         "claims":  (os.path.join(ROOT, "REFERENCE-CMP-Claims.csv"), "Claim")}
CACHE = os.path.expanduser("~/.claude/memory/.memfind/csvsem.json")


def load_cache():
    try:
        with open(CACHE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_cache(c):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(c, fh)


def vectors(texts):
    """Embed with a content-hash cache, so a re-run after one new wave is near-free."""
    cache = load_cache()
    need = [t for t in texts if hashlib.sha1(t.encode()).hexdigest() not in cache]
    if need:
        uniq = list(dict.fromkeys(need))
        print(f"  embedding {len(uniq)} new rows", file=sys.stderr)
        for t, v in zip(uniq, memfind.embed(uniq)):
            cache[hashlib.sha1(t.encode()).hexdigest()] = v
        save_cache(cache)
    return [cache[hashlib.sha1(t.encode()).hexdigest()] for t in texts]


def centred_unit(vecs):
    """Mean-centre, then L2-normalise, so a dot product IS the cosine."""
    n, d = len(vecs), len(vecs[0])
    mean = [sum(v[i] for v in vecs) / n for i in range(d)]
    out = []
    for v in vecs:
        c = [v[i] - mean[i] for i in range(d)]
        norm = math.sqrt(sum(x * x for x in c)) or 1.0
        out.append(array("f", [x / norm for x in c]))
    return out


def rows_of(kind):
    path, col = FILES[kind]
    with open(path, newline="") as fh:
        return [(i, r) for i, r in enumerate(csv.DictReader(fh), start=2)], col


def cmd_dupes(kind, topn):
    for k in (["rulings", "claims"] if kind == "both" else [kind]):
        rows, col = rows_of(k)
        texts = [r[col] for _, r in rows]
        print(f"\n=== {k}: {len(texts)} rows ===", file=sys.stderr)
        U = centred_unit(vectors(texts))
        pairs = []
        for i in range(len(U)):
            a = U[i]
            for j in range(i + 1, len(U)):
                b = U[j]
                s = 0.0
                for x, y in zip(a, b):
                    s += x * y
                if s > 0.55:
                    pairs.append((s, i, j))
        pairs.sort(reverse=True)
        print(f"\n### {k}: {len(pairs)} pairs above 0.55, top {topn}\n")
        for s, i, j in pairs[:topn]:
            li, ri = rows[i]; lj, rj = rows[j]
            flag = "  <<< same Area" if ri.get("Area") == rj.get("Area") else ""
            print(f"[{s:.3f}] L{li} :: L{lj}   ({ri.get('Area')} / {rj.get('Area')}){flag}")
            print(f"    A: {ri[col][:165]}")
            print(f"    B: {rj[col][:165]}\n")


def cmd_check(batchfile, topn):
    batch = json.load(open(batchfile))
    for k, key in (("rulings", "rulings"), ("claims", "claims")):
        incoming = batch.get(key) or []
        if not incoming:
            continue
        rows, col = rows_of(k)
        existing = [r[col] for _, r in rows]
        new = [r[col] for r in incoming]
        U = centred_unit(vectors(existing + new))
        E, N = U[:len(existing)], U[len(existing):]
        print(f"\n### incoming {k}: {len(new)} rows scored against {len(existing)} existing\n")
        # NO THRESHOLD. Score every incoming row against its nearest existing neighbour, sort
        # by that score, and print the head. A cutoff was tried and abandoned twice: an
        # absolute one because paraphrase and novel overlap on score, and a rank-1 MARGIN
        # because it was calibrated at ~670 rows and does not survive the corpus growing.
        # Measured 2026-08-22 at 2,400 rows: a deliberate reword of a live cost-model ruling
        # put its true owner at rank 1 and scored 0.502, only 1.15x clear of rank 2, because
        # a denser corpus crowds every neighbourhood. Ranking held; the trigger did not.
        # So rank and let the reader read, which is what memfind concluded for the same reason.
        scored = []
        for vec, r in zip(N, incoming):
            best, bi = max(((sum(x * y for x, y in zip(vec, e)), i) for i, e in enumerate(E)))
            scored.append((best, r[col], rows[bi][0], existing[bi]))
        scored.sort(reverse=True)
        for best, inc, line, near in scored[:topn]:
            print(f"[{best:.3f}] INCOMING: {inc[:150]}")
            print(f"          NEAREST L{line}: {near[:150]}\n")
        if len(scored) > topn:
            print(f"({len(scored) - topn} further rows not shown, all at or below "
                  f"{scored[topn][0]:.3f}; raise -n to see them.)")
        print(f"Every incoming row is ranked against its nearest neighbour and NOTHING is "
              f"filtered out. Read the head of this list. No score here decides anything.")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "-n"]
    topn = 30
    if "-n" in sys.argv:
        topn = int(sys.argv[sys.argv.index("-n") + 1]); a = [x for x in a if x != str(topn)]
    try:
        if a[0] == "check":
            cmd_check(a[1], topn)
        else:
            cmd_dupes(a[1] if len(a) > 1 else "both", topn)
    except RuntimeError as e:
        print(f"csvsem: {e}", file=sys.stderr); sys.exit(1)
