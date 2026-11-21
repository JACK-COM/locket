#!/usr/bin/env python3
"""Which CSV rows are now covered by a rule file, and may therefore be deleted?

A fact promoted into `.claude/rules/` needs no CSV row: a rule file loads whole and
unconditionally, so its reader already holds the fact and never queries for it, and a
second copy is a separately-editable twin. This RANKS coverage and decides nothing; the
cut is read and made by hand, because a row scoring low may still be covered in substance
and a row scoring high may carry detail the compacted rule deliberately dropped.
"""
import csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csvsem

ROOT = "/Users/raphaelokure/Software/jackcom-projects/choose-my-plane/"
RULES = ["workspace-mechanics.md", "api-code.md", "source-authority.md"]
PROMOTED = ["feedback_cmp_commit_signing", "feedback_bash_cwd_persists_across_repos",
            "reference_authed_dev_dashboard_localhost_9696", "reference_root_doc_archival_lifecycle",
            "feedback_db_enum_value_vs_name", "feedback_no_scraping_reproducibility"]

sents = []
for rf in RULES:
    txt = open(ROOT + ".claude/rules/" + rf).read()
    txt = re.sub(r"`{1,3}[^`]*`{1,3}", lambda m: m.group(0).strip("`"), txt)
    for s in re.split(r"(?<=[.!?])\s+|\n", txt):
        s = re.sub(r"^[\s\-*#>★⚠]+", "", s).strip()
        if len(s.split()) >= 6:
            sents.append(s)

seen = {}
for f in ("pilot.json", "wave1.json", "wave2.json", "wave3.json"):
    for r in json.load(open(f)):
        seen[r["file"][:-3]] = r
cand = []
for slug in PROMOTED:
    for key, col in (("rulings", "Ruling"), ("claims", "Claim")):
        for x in (seen[slug].get(key) or []):
            cand.append({"slug": slug, "kind": key, "text": x[col]})

live = {}
for kind, fn, col in (("rulings", "REFERENCE-CMP-Rulings.csv", "Ruling"),
                      ("claims", "REFERENCE-CMP-Claims.csv", "Claim")):
    live[kind] = {r[col] for r in csv.DictReader(open(ROOT + fn, newline=""))}

U = csvsem.centred_unit(csvsem.vectors(sents + [c["text"] for c in cand]))
R, C = U[:len(sents)], U[len(sents):]
for c, v in zip(cand, C):
    best, bi = max(((sum(x * y for x, y in zip(v, r)), i) for i, r in enumerate(R)))
    c["score"], c["nearest"] = round(best, 3), sents[bi]
    c["in_csv"] = c["text"] in live[c["kind"]]

cand.sort(key=lambda c: -c["score"])
json.dump(cand, open("rulecover.json", "w"), indent=1)
print(f"{len(cand)} promoted-file rows scored; {sum(1 for c in cand if c['in_csv'])} still in a CSV")
for band in ((0.70, 9.9), (0.55, 0.70), (0.40, 0.55), (0.0, 0.40)):
    n = [c for c in cand if band[0] <= c["score"] < band[1] and c["in_csv"]]
    print(f"  {band[0]:.2f}+ : {len(n)}")
