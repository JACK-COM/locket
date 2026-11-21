#!/usr/bin/env python3
"""Precision by score band, from an adjudicated `locket dupes <store> <file.csv> --json` worklist.

    scripts/score-verdicts.py worklist.json [band width, default 0.05]

A reader fills each pair's `verdict`: `dup-a` (keep A), `dup-b` (keep B), `merge` (each
holds something the other lacks) or `distinct` (a false positive). This reports, per
band from the top score down, how many flagged pairs were real, and the highest cutoff
that would have refused no distinct pair.

It exists for one decision: whether a duplicate check may ever BLOCK a write. An
instrument may block only on a question it can decide, and "is this row a duplicate"
is read off text, so the proxy is measured against adjudicated truth first. A missed
duplicate is recovered by the next sweep; a wrong block lands on an unattended agent
that cannot adjudicate it. So the figure that matters is precision at the cutoff, and a
band is blockable only where nothing above it was distinct. Every measurement so far
said no: on a claims table half the pairs above 0.85 were different facts. Bands are a
width rather than fixed edges because a cutoff belongs to one embedder and one table.
"""
import json
import math
import sys


VERDICTS = {"dup-a", "dup-b", "merge", "distinct"}


def band(score, width):
    """The lower edge of the band holding `score`. The quotient is rounded before the
    floor, because 0.70 / 0.05 is 13.999999999999998 in floating point and an edge
    score, common at three decimals, would land one band low."""
    return math.floor(round(score / width, 9)) * width


def main(argv):
    if len(argv) < 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 1
    try:
        width = float(argv[2]) if len(argv) > 2 else 0.05
    except ValueError:
        width = 0.0
    if not 0 < width <= 1:
        print("the band width is a number above 0 and at most 1, as in 0.05", file=sys.stderr)
        return 1
    with open(argv[1]) as fh:
        pairs = json.load(fh)
    done = [p for p in pairs if p.get("verdict")]
    unknown = sorted({str(p["verdict"]) for p in done} - VERDICTS)
    if unknown:
        # A typo counted as a real duplicate inflates precision and can mark a band
        # blockable, which is the one decision this script serves.
        print(f"unknown verdict(s) {', '.join(unknown)}; use {', '.join(sorted(VERDICTS))}",
              file=sys.stderr)
        return 1
    if not done:
        print("no verdicts recorded yet")
        return 1
    bands = {}
    for p in done:
        t = bands.setdefault(round(band(p["score"], width), 6), [0, 0])
        t[p["verdict"] == "distinct"] += 1
    print(f"{len(done)} of {len(pairs)} pairs adjudicated\n")
    print(f"{'band':<14}{'real':>6}{'distinct':>10}{'precision':>11}   cut here")
    real = wrong = 0
    for lo in sorted(bands, reverse=True):
        d, f = bands[lo]
        real, wrong = real + d, wrong + f
        cut = "blockable" if wrong == 0 else f"{wrong} wrongly refused"
        label = f"{lo:.2f}-{lo + width:.2f}"
        print(f"{label:<14}{d:>6}{f:>10}{d / (d + f):>11.0%}   {cut}")
    print(f"{'all':<14}{real:>6}{wrong:>10}{real / (real + wrong):>11.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
