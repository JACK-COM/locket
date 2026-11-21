#!/usr/bin/env python3
"""Adjudicate flagged CSV row pairs, and measure what a BLOCKING gate would have done.

  adjudicate.py pairs  [rulings|claims] [-t 0.55]   emit the worklist as JSON
  adjudicate.py score  <verdicts.json>              precision per band, from the verdicts

The cleanup is the smaller half. The real output is the number that decides whether
csvsem may ever refuse a write once it folds into memfind: `memory-discipline` 2d lets
an instrument block only on a question it can DECIDE, and "is this row a duplicate" is a
proxy read off text. So measure the proxy against adjudicated truth before granting it
authority over an unattended agent.

The cost is asymmetric and that sets the bar. A MISSED duplicate is recoverable by the
next sweep. A WRONG BLOCK lands on a mid-wave agent that cannot adjudicate it and cannot
report why it stopped — the unattended-cycle failure 2d exists to prevent. So the figure
that matters is precision at the proposed cutoff, not overall accuracy.

Verdict vocabulary, one per pair:
  dup-a     same rule; keep row A, delete B
  dup-b     same rule; keep row B, delete A
  merge     each carries something the other lacks; write one row holding both
  distinct  FALSE POSITIVE: different rules that merely share a topic
"""
import csv, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csvsem

BANDS = [(0.85, "0.85+"), (0.75, "0.75-0.85"), (0.65, "0.65-0.75"), (0.0, "below 0.65")]


def band_of(s):
    for lo, name in BANDS:
        if s >= lo:
            return name
    return "below 0.65"


def cmd_pairs(kind, thresh):
    rows, col = csvsem.rows_of(kind)
    texts = [r[col] for _, r in rows]
    U = csvsem.centred_unit(csvsem.vectors(texts))
    out = []
    for i in range(len(U)):
        a = U[i]
        best = []
        for j in range(i + 1, len(U)):
            s = 0.0
            for x, y in zip(a, U[j]):
                s += x * y
            if s >= thresh:
                best.append((s, j))
        for s, j in best:
            li, ri = rows[i]; lj, rj = rows[j]
            out.append({
                "score": round(s, 3), "band": band_of(s),
                "same_area": ri.get("Area") == rj.get("Area"),
                "a_line": li, "a_area": ri.get("Area"), "a": ri[col],
                "b_line": lj, "b_area": rj.get("Area"), "b": rj[col],
                "verdict": None,
            })
    out.sort(key=lambda d: -d["score"])
    print(json.dumps(out, indent=1))
    print(f"\n{len(out)} pairs at or above {thresh}", file=sys.stderr)


def cmd_score(fn):
    pairs = json.load(open(fn))
    done = [p for p in pairs if p.get("verdict")]
    if not done:
        print("no verdicts recorded yet"); return
    tally = {}
    for p in done:
        b = p["band"]
        t = tally.setdefault(b, {"dup": 0, "distinct": 0})
        t["dup" if p["verdict"] != "distinct" else "distinct"] += 1
    print(f"{len(done)} pairs adjudicated of {len(pairs)} flagged\n")
    print(f"{'band':<12}{'true dup':>10}{'false pos':>11}{'precision':>11}")
    cum_d = cum_f = 0
    for _, name in BANDS:
        t = tally.get(name)
        if not t:
            continue
        d, f = t["dup"], t["distinct"]
        cum_d += d; cum_f += f
        prec = d / (d + f) if (d + f) else 0
        print(f"{name:<12}{d:>10}{f:>11}{prec:>10.0%}")
    print(f"{'ALL':<12}{cum_d:>10}{cum_f:>11}{cum_d/(cum_d+cum_f):>10.0%}")
    print("\nA cutoff is safe to BLOCK on only where its band's false-positive count is 0:")
    running_f = 0
    for _, name in BANDS:
        t = tally.get(name)
        if not t:
            continue
        running_f += t["distinct"]
        verdict = "blockable" if running_f == 0 else f"NOT blockable ({running_f} would be wrongly refused)"
        print(f"  cut at {name:<12} {verdict}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "score":
        cmd_score(a[1])
    else:
        t = float(a[a.index("-t") + 1]) if "-t" in a else 0.55
        cmd_pairs(a[1] if len(a) > 1 and not a[1].startswith("-") else "rulings", t)
