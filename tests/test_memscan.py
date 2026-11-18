"""memscan's behaviour suite: the corpus walk, claims, the write gate, bashguard,
grepassist, hook health, init and the registry, joins, and the Hermes ask-once gate.
Every store is synthetic and built under a temp dir; `fixture_env` points every
Locket path there."""
import json
import os
import shlex
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.harness import SRC, fixture_env

import memscan as ms  # noqa: E402
import cites  # noqa: E402
import grepassist  # noqa: E402
import help_text  # noqa: E402
from adapters import hermes  # noqa: E402
import hosts  # noqa: E402
import ledger  # noqa: E402
import paths  # noqa: E402
from commands import hygiene, overlap, stores  # noqa: E402


def _shown(report):
    """Print a report as the CLI does and return its finding count."""
    report.show()
    return report.count


def _hook_health(fixture):
    """Hook health against a throwaway directory, in this process and in every child it
    spawns; the installed paths and the environment are both restored on the way out."""
    live_dir = ms.health_dir()
    d = fixture / "hook-health"
    with paths.use(paths.PATHS.replace(health_dir=d)), mock.patch.dict(os.environ, LOCKET_HEALTH_DIR=str(d)):
        _hook_health_body(d)
    assert ms.health_dir() == live_dir, "the test left hook health pointed at its fixture"


def _hook_health_body(d):
    import signal, subprocess
    marks = d / "inflight"
    ls = lambda: sorted(os.listdir(marks))
    with ms.HookRun("hook"):
        assert len(ls()) == 1, "a running hook leaves its marker"
    assert not ls(), "a finished hook leaves a marker behind"
    assert ms.health_report() == (None, []), "a clean run wrote counters"
    # A run the host kills never reaches its exit: its marker is the only trace. Killed for
    # real, so the test is the mechanism and not a model of it. POSIX only: elsewhere a dead
    # run is known by age alone.
    if os.name == "posix":
        prog = (f"import sys, time; sys.path.insert(0, {str(SRC)!r}); import memscan\n"
                "with memscan.HookRun('grepassist'):\n    print('in', flush=True); time.sleep(60)\n")
        child = subprocess.Popen([sys.executable, "-c", prog], stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline().strip() == "in"
        child.send_signal(signal.SIGKILL)
        child.wait()
        child.stdout.close()
        assert len(ls()) == 1, "a killed run left no marker"
        rec, problems = ms.health_report()         # the report folds: no later hook run needed
        assert rec["verbs"]["grepassist"]["killed"] == 1 and len(problems) == 1, rec
        assert not ls(), "the killed run's marker was not folded"
        assert "?" not in problems[0].split("(last ")[1][:4], problems
    ms.health_ack()
    # A live run is never folded early, and a clock stepped back reads as young; past the
    # stale age it folds whoever holds it, which covers a reused PID.
    stale = ms.HEALTH_STALE_POSIX if os.name == "posix" else ms.HEALTH_STALE_OTHER
    live = marks / ms._marker_name("hook", os.getpid(), time.time_ns())
    live.write_text("{}")
    ms._fold_dead_markers(now=time.time() - 3600)
    ms._fold_dead_markers()
    assert live.exists(), "a live run's marker was folded"
    ms._fold_dead_markers(now=time.time() + stale + 1)
    assert not live.exists() and ms.health_report()[0]["verbs"]["hook"]["killed"] == 1
    # Counted before removed: with the lock held elsewhere the count cannot land, and the
    # marker must survive for a later fold rather than vanish uncounted.
    dead = marks / ms._marker_name("bashguard", 2 ** 22 + 7, time.time_ns() - 10 ** 12)
    dead.write_text("{}")
    with open(d / ".lock", "a+") as held:
        try:
            import fcntl
            fcntl.flock(held, fcntl.LOCK_EX)
            t0 = time.monotonic()
            ms._fold_dead_markers(now=time.time() + stale + 1)
            assert time.monotonic() - t0 < 2, "a held lock made the fold wait"
            assert dead.exists(), "a marker was removed although its count could not land"
        except ImportError:
            pass
    ms._fold_dead_markers(now=time.time() + stale + 1)
    assert not dead.exists() and ms.health_report()[0]["verbs"]["bashguard"]["killed"] == 1
    # A claim or a temp a dead writer left is reclaimed or removed, never kept forever.
    orphan = marks / f".claim-999999-{ms._marker_name('hook', 999999, time.time_ns())}"
    orphan.write_text("{}")
    temp = d / ".health-leftover"
    temp.write_text("{")
    old = time.time() - ms.HEALTH_TEMP_AGE - 5
    os.utime(orphan, (old, old)); os.utime(temp, (old, old))
    ms.health_ack()
    ms._fold_dead_markers(now=time.time() + stale + 1)
    assert not ls() and not temp.exists(), (ls(), temp.exists())
    assert ms.health_report()[0]["verbs"]["hook"]["killed"] == 1, "an orphaned claim was not counted"
    # The directory is bounded: past the cap the OLDEST fold, by start time across verbs.
    ms.health_ack()
    base = time.time_ns()
    for i in range(ms.HEALTH_MARKER_CAP + 10):
        verb = ("bashguard", "hook")[i % 2]     # mixed verbs: a name sort must still be a time sort
        (marks / ms._marker_name(verb, os.getpid(), base + i)).write_text("{}")
    ms._fold_dead_markers()
    kept = ls()
    assert len(kept) == ms.HEALTH_MARKER_CAP, len(kept)
    assert kept[0] == ms._marker_name("bashguard", os.getpid(), base + 10), "the cap did not fold the oldest"
    for n in kept:
        (marks / n).unlink()
    # A swallowed error is counted with its detail, the record stays one entry per verb,
    # an unreadable record is set aside and said so, and an ack zeroes the counters.
    ms.health_ack()
    for _ in range(3):
        ms.health_error("hook", "lexical", ValueError("x"))
    rec, problems = ms.health_report()
    assert rec["verbs"]["hook"]["errors"] == 3 and set(rec["verbs"]) == {"hook"}, rec
    assert any("swallowed 3 error" in p and "ValueError in lexical" in p for p in problems), problems
    (d / "health.json").write_text("{not json")
    ms.health_error("hook", "2e", OSError())
    rec, problems = ms.health_report()
    assert (d / "health.json.corrupt").exists() and any("set aside" in p for p in problems), problems
    assert ms.health_ack() and ms.health_report()[1] == []
    # A record that cannot be read is a problem, never a clean reading.
    (d / "health.json").write_text("[]")
    assert any("cannot be read" in p for p in ms.health_report()[1])
    ms.health_ack()
    # Under contention nothing is dropped: an error waits as a marker and is counted at the
    # next fold, and a dead marker the fold could not count is reported as pending.
    dead = marks / ms._marker_name("hook", 2 ** 22 + 9, time.time_ns() - 10 ** 12)
    dead.write_text("{}")
    with open(d / ".lock", "a+") as held:
        try:
            import fcntl
            fcntl.flock(held, fcntl.LOCK_EX)
            ms.health_error("grepassist", "rank", KeyError())
            assert any("not yet counted" in p for p in ms.health_report()[1]), "a pending abandoned run read clean"
        except ImportError:
            pass
    rec, problems = ms.health_report()
    assert rec["verbs"]["grepassist"]["errors"] == 1 and rec["verbs"]["hook"]["killed"] == 1, rec
    assert any("KeyError in rank" in p for p in problems), problems
    # Only known verbs get a record, and a PID too large for the OS reads as dead.
    ms.health_ack()
    (marks / ms._marker_name("anything-at-all", 2 ** 70, time.time_ns())).write_text("{}")
    (marks / ms._marker_name("other", 2 ** 70, time.time_ns() + 1)).write_text("{}")
    rec, _p = ms.health_report()
    assert set(rec["verbs"]) == {"?"} and rec["verbs"]["?"]["killed"] == 2, rec
    assert ms.health_ack() and not os.listdir(marks)
def _selftest_inner():
    # Every fixture is synthetic. The live memory store is never read, because a
    # test that reads it encodes the ORIGIN's manifest: on a store with different
    # ledger and holding-space names the assertions fail, and the first instance
    # to install this cold made them pass by rewriting its own manifest to match
    # ours. A council-shaped store is built in a temp dir and `Paths.host_stores` is
    # pointed at it for the duration.
    import tempfile, shutil
    fixture = Path(tempfile.mkdtemp())
    COUNCIL = fixture / "memory"
    COUNCIL.mkdir()
    saved_council = paths.PATHS.host_stores["council"]
    _hook_health(fixture)
    paths.PATHS.host_stores["council"] = COUNCIL
    ms.settings.cache_clear()
    docs = {
        "a.md": "The default execution path is the dossier pipeline run interactively.\n"
                "Unrelated sentence about hangar rates and insurance quotes entirely.",
        "b.md": "Something else here about propeller pitch enumerations and defaults.\n"
                "The default execution path is the dossier pipeline, run interactively.",
        "c.md": "See project doc basis per engine class for the whole seam and ladder.",
    }
    wrapped = ("- first item that wraps\n  onto a second line\n  and onto a third\n"
               "- second item that wraps\n  onto a second line\n  and a third line too\n")
    assert list(hygiene._prose_paras(wrapped)) == [], "a hard-wrapped list is a list, not a paragraph"
    lazy = "- one item\ncontinued without indentation\nand still the item\n"
    assert list(hygiene._prose_paras(lazy)) == [], "a lazy continuation belongs to its item"
    headed = "# Heading\nProse follows the heading directly.\nMore prose.\nAnd more.\n"
    assert list(hygiene._prose_paras(headed)) == [len(headed.split())], "a heading opens no run"
    indented = "    Indented prose under no marker\n    is still prose.\n"
    assert list(hygiene._prose_paras(indented)) == [len(indented.split())], "indentation alone is not a list"

    # §5 dated history clauses: the shapes loaded-whole files carry, not only `Update <date>`.
    # Positive control is SELF.md's stack form; the negatives are scoped dates §1 permits,
    # and a regex that fires on them turns a worklist into noise.
    for pos in ("*Refinement, 2026-05-19: session-end discipline is structurally load-bearing.*",
                "**Some stale assertion**. 2026-08-08: **Resolved:** the Peer had a sandwich.",
                "⚠ UPDATE (2026-08-08): the endpoint moved to the API repo.",
                "*Instances since graduation, folded from the Workspace 2026-09-11 (supervised).*",
                "Update 2026-08-08: the cap went to $5."):
        assert ms.HISTORY_CLAUSE.search(pos), f"history clause not matched: {pos}"
    for neg in ("It moved into the API repo on 2026-08-02 and is no longer at the workspace root.",
                "### trading *(2026-07-27 -- resolved 2026-08-02, personal not work)*",
                "plus since 2026-09-18 one row in the registry at `~/.locket/corpora.json`",
                "a gate transition appends one dated clause (Peer ruling 2026-09-11).",
                "Rebrand SHIPPED (v1.1.3), superseding the prior contractor iteration.",
                "thread texture is in `in-flight/closed/architectural-stake-refinement-2026-05-12.md`.",
                "**Target:** SELF.md *Architectural stake* Refinement 2026-05-12, as the mechanism face."):
        assert not ms.HISTORY_CLAUSE.search(neg), f"scoped date matched as history: {neg}"

    # ★ A sentence spanning a hard wrap is ONE claim; line-meaningful markdown is not joined.
    wrapped = ("On any file-op failure the diagnostic order is: wrong machine, unloaded tool, unanswered\n"
               "authorization prompt, then channel health. Channel health is last and has not once been the answer.\n")
    cs = [c for c, _ in ms.claims(ms.strip(wrapped))]
    assert any("unanswered authorization prompt, then channel health" in c for c in cs), cs
    assert not any(c.endswith("unanswered") for c in cs), "the wrap must not end a claim"
    bul = "- first bullet carries no terminal stop at all here\n- second bullet carries none either here\n"
    assert len([c for c, _ in ms.claims(ms.strip(bul))]) == 2, "two bullets stay two claims"
    cont = "- first item that wraps onto\n  a second indented line of the same item\n"
    assert len([c for c, _ in ms.claims(ms.strip(cont))]) == 1, "a wrapped bullet is one claim"
    lead = "The three steps that follow are these\n- measured step one is the first of them\n"
    assert not any("these measured" in c for c in [c for c, _ in ms.claims(ms.strip(lead))]), \
        "a line before a list is not joined to its first item"
    head = "# A heading that carries no stop\nProse that follows the heading directly and at length.\n"
    assert not any("stop Prose" in c for c in [c for c, _ in ms.claims(ms.strip(head))]), \
        "a heading is not joined to the prose under it"
    tbl = "| alpha column | beta column here |\n| gamma column | delta column here |\n"
    assert not any("here gamma" in c for c in [c for c, _ in ms.claims(ms.strip(tbl))]), "table rows stay rows"
    bold = "**Status:** N=1 and explicitly not met\n**Target:** a rule file somewhere\n"
    assert len([c for c, _ in ms.claims(ms.strip(bold))]) <= 2 and \
        not any("met Target" in c for c in [c for c, _ in ms.claims(ms.strip(bold))]), \
        "a line ending in a bolded colon-field value still ends at its stop"
    assert list(ms.unclaimed("Short tail here.\nA long enough sentence to be a claim on its own terms.")) \
        == ["Short tail here."], "the floor's rejects are reportable"

    pairs = ms.cross_file_pairs(docs)
    assert pairs, "identical claim across two files must pair"
    assert {pairs[0][1], pairs[0][2]} == {"a.md", "b.md"}, pairs[0]
    assert pairs[0][0] >= 0.8, pairs[0][0]

    # one shared clause inside otherwise-unrelated files still pairs: this is the
    # case document-level similarity cannot see.
    docs2 = {
        "x.md": "Turbine overhaul programs cover the engine reserve entirely here.\n"
                "The Overview section is exactly two paragraphs long and no more.",
        "y.md": "Silhouette glyphs scale by wingspan across the comparison cards.\n"
                "The Overview section is exactly two paragraphs long and no more.",
    }
    assert ms.cross_file_pairs(docs2), "cross-subject single-clause restatement must pair"

    # across: one claim held by two stores pairs exactly once, and a pair inside
    # one store is audit's, not this mode's.
    two = fixture / "two"
    for store, files in (("alpha", {"m.md": docs["a.md"], "n.md": docs["b.md"]}),
                         ("beta", {"o.md": docs["a.md"] + "\nA sentence only beta holds here today."})):
        (two / store).mkdir(parents=True)
        (two / store / ms.MANIFEST).write_text("{}\n")
        for f, body in files.items():
            (two / store / f).write_text(body)
    _, _, _, cross = overlap.across_pairs({"alpha": two / "alpha", "beta": two / "beta"})
    assert {(a.split("::")[0], b.split("::")[0]) for _, a, b, _, _ in cross} == {("alpha", "beta")}, cross
    # alpha/m holds both of beta/o's shared sentences, alpha/n holds one: three pairs
    assert len(cross) == 3, [(a, b) for _, a, b, _, _ in cross]
    # a sentence propagated to three stores, one file each, is the finding; the
    # same sentence in three files of ONE store is furniture and drops the pair.
    spread = "Carry cost structure in curated prose and never the cost level itself."
    (two / "gamma").mkdir(); (two / "gamma" / ms.MANIFEST).write_text("{}\n")
    (two / "gamma" / "g.md").write_text(spread)
    (two / "alpha" / "m.md").write_text(docs["a.md"] + "\n" + spread)
    (two / "beta" / "o.md").write_text(docs["a.md"] + "\n" + spread)
    stores3 = {"alpha": two / "alpha", "beta": two / "beta", "gamma": two / "gamma"}
    _, _, _, cross3 = overlap.across_pairs(stores3)
    assert any("gamma::g.md" in (a, b) for _, a, b, _, _ in cross3), "propagated sentence must survive"
    for f in ("q.md", "r.md"):
        (two / "alpha" / f).write_text(spread)
    _, _, _, cross4 = overlap.across_pairs(stores3)
    assert not any("gamma::g.md" in (a, b) for _, a, b, _, _ in cross4), "furniture in one store is filtered"
    # a CSV registered by two manifests is ONE file: it never pairs against
    # itself, and a store's own restatement of a row is that store's audit, not
    # this mode's. A holding-space entry whose claim a second store's durable
    # file now holds is tagged, because `graduated` scores one store only.
    row = "A reference ledger registered by two manifests is scored as one file and never as two."
    shared = two / "shared.csv"
    shared.write_text(f"Ruling,Area\n\"{row}\",locket\n")
    reg = json.dumps({"sources": [{"path": str(shared), "text": "Ruling", "label": "Area"}]})
    (two / "alpha" / ms.MANIFEST).write_text(reg)
    (two / "beta" / ms.MANIFEST).write_text(reg)
    (two / "beta" / "o.md").write_text(docs["a.md"] + "\n" + row)
    (two / "alpha" / "SHORT_TERM.md").write_text(docs["a.md"].split("\n")[0])
    ms.settings.cache_clear()
    union5, _, _, cross5 = overlap.across_pairs(stores3)
    assert "alpha+beta::shared.csv#locket" in union5, sorted(union5)
    assert not any("shared.csv" in a or "shared.csv" in b for _, a, b, _, _ in cross5), \
        "a shared CSV pairs against neither itself nor a holder's own restatement"
    assert any({a, b} == {"alpha::m.md", "beta::o.md"} for _, a, b, _, _ in cross5), "md pairs still fire"
    grad = overlap.graduated_across(cross5, stores3)
    assert [(a, b) for _, a, b, _, _ in grad] == [("alpha::SHORT_TERM.md", "beta::o.md")], grad
    (two / "alpha" / "SHORT_TERM.md").unlink()
    # a manifest directory inside a store is a store of its own: the outer walker
    # leaves its files to it, and a write there resolves to it.
    inner = two / "alpha" / "inner"
    inner.mkdir()
    (inner / ms.MANIFEST).write_text("{}\n")
    (inner / "p.md").write_text(docs["a.md"])
    assert not [f for f in ms.md_files(two / "alpha") if "inner" in str(f)], "nested store must not be walked"
    assert ms._root_of(inner / "p.md") == inner, "a write inside the nested store belongs to it"

    assert ms.pointer_only("See project doc basis per engine class.")
    assert not ms.pointer_only("The default execution path is the dossier pipeline run interactively.")
    # a pointer that does not open with "see" is still a pointer
    assert ms.pointer_only("Durable context lives in memory/projects/locket.md for this project.")
    assert ms.pointer_only("The ruling is a RULINGS.csv row on Area=substrate.")
    # a claim that happens to cite a file keeps its claim
    assert not ms.pointer_only("The write gate in memscan.py refuses a fork unless locket find ran earlier in the session.")

    ranked = ms.score_against("The default execution path is the dossier pipeline run interactively.", docs)
    assert ranked[0][2] in ("a.md", "b.md"), ranked[0]
    assert ranked[0][1] >= 0.8, ranked[0][1]
    assert hook_payload_from(docs, "wholly unrelated text about runway lighting and taxiways") is None
    # the hard negative: heavy vocabulary overlap, no restated claim. Gating on the
    # combined score passes this and fires; gating on the claim score does not.
    assert hook_payload_from(docs, "Propeller pitch defaults and hangar insurance quotes differ by "
                                   "execution path, and the interactively run pipeline for hangar "
                                   "rates is unrelated to any default here.") is None

    m = ms.IDX_LINE.match("- [Some Title](feedback_thing.md) — the hook text here")
    assert m and m.group("target") == "feedback_thing.md", m

    found = {(a, int(b)) for a, b in ms.CITE.findall(
        "see src/models/aircraft.py:2493 and results.html:109 and app.js:7-9, "
        "shipped in 4.2.31 at 2026-08-17T03:23:17Z, cost 11.88:2 not a cite")}
    assert found == {("src/models/aircraft.py", 2493), ("results.html", 109),
                     ("app.js", 7)}, found

    src = ["class Thing:", "    def method(self):", "        return 1",
           "", "def top():", "    pass", "", "CONSTANT = 3"]
    assert ms.enclosing_scope(src, 3) == "def method", ms.enclosing_scope(src, 3)
    assert ms.enclosing_scope(src, 6) == "def top", ms.enclosing_scope(src, 6)
    assert ms.enclosing_scope(src, 5) == "def top", ms.enclosing_scope(src, 5)
    assert ms.enclosing_scope(src, 1) == "class Thing", ms.enclosing_scope(src, 1)
    assert ms.enclosing_scope(src, 99) is None
    # the regression: a module-level constant BELOW a def is not inside it.
    assert ms.enclosing_scope(src, 8) is None, ms.enclosing_scope(src, 8)

    # The slug is lossy (dots, dashes and underscores all arrive as `-`), so the
    # decode is only trustworthy if it round-trips through the encoder.
    for name, c in sorted(ms.all_corpora().items()):
        if name in ("council", "home"):
            continue
        d = ms.project_dir_for(c)
        if d is not None:
            assert ms.corpus_for_cwd(d) == c, (name, d)

    # corpus resolution reaches EVERY project, not a named list
    assert ms.corpus_for(COUNCIL / "x.md") == COUNCIL
    # ★ A host's core directory is its council store, its members the corpus.
    # Built here rather than read off ~/.claude, whose layout is one machine's.
    home = (fixture / "home").resolve()     # init resolves; /var is /private/var here
    for rel, body in (("CLAUDE.md", "The constitution at the root is corpus.\n"),
                      ("memory/a.md", "A memory fact that lives in the memory member.\n"),
                      ("rules/r.md", "A rule restating nothing, in the rules member.\n"),
                      ("plugins/p.md", "A vendored plugin page that is host state.\n"),
                      ("sessions/s.md", "A transcript that is host state as well.\n"),
                      ("memory/archive/old.md",
                       "The superseded sentence filed verbatim before the rewrite happened.\n"),
                      ("memory/vault/memfind.json", "{}\n"),
                      ("memory/vault/v.md", "A nested manifest store inside a member.\n")):
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(body)
    import io as _io, contextlib as _ctx
    paths.PATHS.host_stores["home"] = home
    hosts.HOST_MEMBERS["home"] = {"memory", "rules"}
    ms.settings.cache_clear()
    assert ms.corpus_for(home / "rules/r.md") == home, "a member directory resolves to the home"
    assert ms.corpus_for(home / "CLAUDE.md") == home, "root-level markdown is corpus"
    assert ms.corpus_for(home / "plugins/p.md") is None, "a non-member directory is host state"
    assert ms.corpus_for(home / "memory/archive/old.md") is None, "archive stays provenance"
    assert ms.corpus_for(home / "memory/vault/v.md") == home / "memory/vault", "a nested store is its own"
    got = {str(q.relative_to(home)) for q in ms.md_files(home)}
    assert got == {"CLAUDE.md", "memory/a.md", "rules/r.md"}, got
    # Host state must not be ENTERED, not merely filtered out afterwards: a walk that
    # visited a home's job and session trees pushed the write hook past its timeout,
    # so the gate failed open. os.walk lists through os.scandir on every version;
    # Path.glob and rglob do only on some (3.9 binds scandir at class creation), so
    # an rglob is caught by name: one rooted at the home or in host state is the
    # regression, whatever it lists through.
    (home / "plugins/deep/er/locket.json").parent.mkdir(parents=True, exist_ok=True)
    (home / "plugins/deep/er/locket.json").write_text("{}\n")
    scanned, real_scandir, real_rglob, real_glob = [], os.scandir, Path.rglob, Path.glob
    def _spy(p="."):
        scanned.append(Path(p)); return real_scandir(p)
    def _rglob_spy(self, *a, **k):
        scanned.append(Path(self) / "**"); return real_rglob(self, *a, **k)
    def _glob_spy(self, pattern, *a, **k):
        if "**" in str(pattern):
            scanned.append(Path(self) / "**")
        return real_glob(self, pattern, *a, **k)
    os.scandir, Path.rglob, Path.glob = _spy, _rglob_spy, _glob_spy
    try:
        ms.md_files(home); list(ms._archive_md(home)); list(ms._walk_files(home, "*.csv"))
    finally:
        os.scandir, Path.rglob, Path.glob = real_scandir, real_rglob, real_glob
    strays = sorted({str(p.relative_to(home)) for p in scanned
                     if (p == home / "**") or (home in p.parents
                                               and p.relative_to(home).parts[0] not in ("memory", "rules"))})
    assert not strays, f"the corpus walk entered host state: {strays}"
    # A store holds content when any markdown or registered row exists; an empty one does not.
    assert ms.has_content(home)
    (fixture / "hollow").mkdir(exist_ok=True)
    assert not ms.has_content(fixture / "hollow"), "an empty folder read as a store with memory"
    # An unlistable home keeps its members: the root listing failing must not empty
    # the store, or every write into it passes unscored.
    if hasattr(os, "getuid") and os.getuid() != 0:
        home.chmod(0o311)
        try:
            got = {str(q.relative_to(home)) for q in ms.md_files(home)}
        finally:
            home.chmod(0o755)
        assert got == {"memory/a.md", "rules/r.md"}, f"an unlistable root dropped its members: {got}"
    # An ambiguous citation is PAST EOF only when no same-named file reaches the line.
    (home / "rules/a").mkdir(); (home / "rules/a/x.py").write_text("a\n")
    (home / "rules/b").mkdir(); (home / "rules/b/x.py").write_text("a\nb\nc\n")
    assert ms.code_index([(home / "rules", frozenset())])["x.py"][0].endswith("a/x.py"), \
        "the short candidate must sort first, or this test passes on first-match resolution"
    cidx = ms.code_index([(home / "rules", frozenset())])
    assert ms.resolve_citation("x.py", 3, cidx)[0] == "ok", "a short first candidate hid a long one"
    assert ms.resolve_citation("x.py", 9, cidx)[0] == "eof"
    # A citation naming a directory never resolves to a same-named file in another one.
    assert ms.resolve_citation("b/x.py", 3, cidx)[0] == "ok"
    assert ms.resolve_citation("c/x.py", 1, cidx) == ("missing", 2), "a path resolved to a stranger"
    assert ms.resolve_citation("./b//x.py", 3, cidx)[0] == "ok", "a normalisable path read as missing"
    (home / "rules/repo-ui/app").mkdir(parents=True); (home / "rules/repo-ui/app/y.py").write_text("y\n")
    cidx = ms.code_index([(home / "rules", frozenset())])
    assert ms.resolve_citation("ui/app/y.py", 1, cidx)[0] == "ok", "an abbreviated repository root read as missing"
    assert ms.resolve_citation("ui/y.py", 1, cidx)[0] == "missing", "a two-part path matched a stranger's folder"
    # The mode end to end: defects counted, every citation listed, the scope named.
    cited = {("x.py", 3): {"m.md"}, ("x.py", 9): {"m.md"}, ("gone.py", 1): {"n.md"}}
    with mock.patch.object(cites, "citations", return_value=cited), \
            mock.patch.object(cites, "code_roots", return_value=[(home / "rules", frozenset())]):
        r = hygiene.cmd_cites(home)
    assert r.count == 2 and "gone.py:1  FILE NOT FOUND" in r.text and "x.py:9  PAST EOF" in r.text \
        and "3 distinct path:line citations across 2 memory files" in r.text, r.text
    # A member path is entered only if every component would be: `memory/archive`
    # stays provenance even when a manifest names it as a member.
    hosts.HOST_MEMBERS["home"] = {"memory", "rules", "memory/archive"}; ms.settings.cache_clear()
    assert not any("archive" in str(q) for q in ms.md_files(home)), "a member path entered an excluded dir"
    hosts.HOST_MEMBERS["home"] = {"memory", "rules"}; ms.settings.cache_clear()
    # One unreadable file is skipped and the rest still score: raised, it emptied the
    # whole read inside the hook and the catch-all let the write through.
    if hasattr(os, "getuid") and os.getuid() != 0:
        (home / "memory/locked.md").write_text("A sentence nobody can read here.\n")
        (home / "memory/locked.md").chmod(0)
        try:
            got = ms.corpus(home)
        finally:
            (home / "memory/locked.md").chmod(0o644); (home / "memory/locked.md").unlink()
        assert "memory/a.md" in got and "memory/locked.md" not in got, sorted(got)
    # A dangling symlink is not a file: reading it raised inside the hook.
    (home / "memory/gone.md").symlink_to(home / "memory/never-there.md")
    assert "memory/gone.md" not in {str(q.relative_to(home)) for q in ms.md_files(home)}
    (home / "memory/gone.md").unlink()
    assert any("superseded" in " ".join(ts) for _, _, ts in ms.archived_claims(home).items), \
        "provenance is read from the archive under a member, not root/archive alone"
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        rc = ms.cmd_init(home, index=False)
    assert rc == 0 and "by convention" in buf.getvalue() and not (home / ms.MANIFEST).exists(), \
        "init on a host home is a no-op that does not refuse itself for the stores under it"
    (home / ms.MANIFEST).write_text(json.dumps({"members": ["memory"]}))
    ms.settings.cache_clear()
    assert ms.corpus_for(home / "rules/r.md") is None, "a manifest's members replaces the host default"
    assert ms.corpus_for(home / "memory/a.md") == home
    (home / ms.MANIFEST).unlink(); ms.settings.cache_clear()
    del paths.PATHS.host_stores["home"]; del hosts.HOST_MEMBERS["home"]
    ms.settings.cache_clear()
    for slug in ("-a-notes", "-b-myapp", "-c-brand-new-project"):
        p = hosts.claude() / f"projects/{slug}/memory/x.md"
        assert ms.corpus_for(p) == p.parent, slug
    assert ms.corpus_for(COUNCIL / "archive/x.md") is None                       # verbatim by design
    assert ms.corpus_for(COUNCIL / "in-flight/x.md") is None                     # appends until it closes
    assert ms.corpus_for(hosts.claude() / "projects/-a/memory/archive/x.md") is None
    assert ms.corpus_for(COUNCIL / "x.txt") is None                              # not markdown
    assert ms.corpus_for("/Users/x/Software/notes/x.md") is None                 # outside the corpus
    assert ms.corpus_for(hosts.claude() / "projects/-a/x.md") is None                    # project root, not memory

    big = "x" * 900
    assert ms.added_text({}, big, "") == big                                     # new or empty file
    assert ms.added_text({"old_string": "y" * 800}, big, "y" * 800) is None       # in-place rewrite
    assert ms.added_text({"old_string": "y" * 20}, big, "z") == big               # inserted block
    # an Edit that inserts BETWEEN two anchor lines: both anchors must be subtracted
    anchored = ms.added_text({"old_string": "TOP LINE\nBOTTOM LINE"},
                          "TOP LINE\n" + big + "\nBOTTOM LINE", "z")
    assert "TOP LINE" not in anchored and "BOTTOM LINE" not in anchored, anchored
    # a whole-file Write: an append LEAVES THE OLD BODY INTACT, a reconsolidation
    # does not. This is the dominant append shape at session close.
    prior = "The hangar rate structure is quote-gated on this airframe entirely.\n"
    assert ms.added_text({}, prior + big, prior).strip() == big                   # appended
    assert ms.added_text({}, "wholly rewritten body " + big, prior) is None       # reconsolidated

    # ★ THE HARD CASE, and the reason the containment test alone is not the fix:
    # the prior body is IN the text and is already in the corpus, so scoring the
    # whole text self-matches at 1.00 and names the file being written on every
    # append. Only the delta may reach the scorer.
    docs3 = dict(docs, t=prior)
    # `hook_decision` returns (message, blocking); these cases assert on the
    # message, and the blocking branch has its own case at the end.
    hd = lambda *a, **k: ms.hook_decision(*a, **k)[0]
    novel = ("Turbine hot-section inspections fall due on calendar time as well as hours, "
             "and an owner flying below the annual utilisation assumption pays the "
             "reserve twice over the life of the engine program regardless. " * 2)
    assert hd("t", {}, prior + novel, prior, lambda: docs3) is None, \
        "an append whose DELTA is novel must not fire on its own prior body"
    restated = ("The default execution path is the dossier pipeline run interactively "
                "for every one of these. " * 4)
    assert hd("t", {}, prior + restated, prior, lambda: docs3), \
        "an append whose delta restates another file MUST fire"
    assert hd("MEMORY.md", {}, prior + restated, prior, lambda: docs3) is None
    assert hd("CHECKPOINTS.md", {}, prior + restated, prior, lambda: docs3) is None

    # ★ THE SECOND HARD CASE: an in-place rewrite that EXTENDS what it replaces.
    # A markdown paragraph is one physical line, so the extended paragraph is
    # never byte-identical and survives `added_text`'s anchor subtraction; the
    # replaced span must therefore leave the corpus before scoring.
    # `marked` lives ONLY in doc `t`, so any hit on it is the file matching
    # itself; a retiring sentence another doc also asserts would be a true
    # positive and prove nothing. ⚠ It carries MARKDOWN deliberately, because the
    # corpus holds stripped bodies and this case fails without the strip.
    marked = "The `hangar` rate structure is **quote-gated** on this airframe entirely.\n"
    docs_md = dict(docs, t=ms.strip(marked))
    assert hd("t", {"old_string": marked}, marked + novel, marked,
                         lambda: docs_md) is None, \
        "a rewrite must not be scored against the passage it is retiring"
    # and the guard must still bite: the same rewrite restating ANOTHER file fires.
    # `restated * 2` because an Edit must also clear the 400-character delta gate
    # in `added_text`; a single `restated` sits just under it and would pass this
    # assertion for the wrong reason.
    assert hd("t", {"old_string": prior}, prior + restated * 2, prior,
                         lambda: docs3), \
        "a rewrite whose new material restates another file MUST still fire"

    # a holding space is scored against ITSELF: a repeated entry fires, and an entry
    # restating a fact that lives elsewhere in the corpus does not, because turning
    # over something already known is the surface's job.
    held = ("The incubator is holding the observation that autonomous cycles skip "
            "reconsolidation whenever the surface is already large. ")
    stem = "Promotion target if this holds across further instances: "
    body = held + stem + "Ramp availability shifts seasonally with transient traffic volumes. "
    assert hd("SHORT_TERM.md", {}, body + held * 2, body,
                         lambda: docs3), "a repeated entry in a holding space must fire"
    assert hd("SHORT_TERM.md", {}, body + restated, body,
                         lambda: docs3) is None, "a holding space must not score against the corpus"
    # the file's own furniture must not fire: a stem shared by three entries says
    # nothing about whether a fourth entry is new.
    assert hd("SHORT_TERM.md", {}, body + stem * 3 + stem + novel, body + stem * 3,
                         lambda: docs3) is None, "a shared stem is not a duplicate entry"

    # a dated measurement series is chronology, not repetition: the same sentence
    # carrying a later date and a different figure must be allowed to land, while
    # the undated twin of the same sentence must still fire.
    # ⚠ Each reading must clear the floor (`gate.MIN_ADDED`) in `hook_decision` on its
    # own, or both assertions below pass for the wrong reason: a short `added`
    # never reaches the scorer, so the carve-out would look correct untested.
    tail = ("bytes on this reading of the ledger growth series, measured against the "
            "previous reading and recorded here so the trend stays visible.\n")
    r1 = ("Measurement update, 2026-08-13 (conscious cycle 312): the checkpoints ledger "
          "reached 300232 " + tail)
    r2 = ("Measurement update, 2026-08-16 (weekly-synthesis cycle 10): the checkpoints ledger "
          "reached 341539 " + tail)
    assert len(r1) > 200 and len(r2) > 200, "readings must clear the hook's length floor"
    assert ms.dated_series_pair(r1, r2) and not ms.dated_series_pair(r1, r1)
    assert hd("OPEN_QUESTIONS.md", {}, body + r1 + r2, body + r1,
                         lambda: docs3) is None, "a later reading in a dated series must not fire"
    undated = r2.replace("2026-08-16", "").replace("341539", "300232")
    assert hd("OPEN_QUESTIONS.md", {}, body + r1 + undated, body + r1,
                         lambda: docs3), "an undated restatement is still a duplicate entry"

    # ★ PROVENANCE: text already filed under archive/ is a MOVE, not a fork, and
    # must not be scored against the source it is moving away from. `moved` lives
    # in doc `t` and is being written into a new file while the source is still
    # live, which is exactly the ordering 4 prescribes and the shape that
    # blocked four legitimate operations before this subtraction existed.
    moved = ("The conscious cycle runs at 09:00 Pacific on the Opus model under a "
             "three dollar ceiling, and the four subconscious cycles follow it at "
             "three-hourly intervals on Sonnet. " * 3)
    docs_mv = dict(docs, t=moved)
    assert hd("new.md", {}, moved, "", lambda: docs_mv) is not None, \
        "control: without provenance the relocation fires against its own source"
    arch = [ts for _, ts in ms.claims(ms.strip(moved))]
    assert hd("new.md", {}, moved, "", lambda: docs_mv, lambda: arch) is None, \
        "a relocation of ARCHIVED text must not fire against its still-live source"
    # ⚠ the archive is read STRIPPED: a filed copy carrying markdown furniture
    # must still match, or the patch no-ops silently while looking like it works.
    arch_md = [ts for _, ts in ms.claims(ms.strip(
        "The **conscious cycle** runs at `09:00` Pacific on the [[opus]] model under a "
        "three dollar ceiling, and the four subconscious cycles follow it at "
        "three-hourly intervals on Sonnet.\n"))]
    assert hd("new.md", {}, moved, "", lambda: docs_mv, lambda: arch_md) is None, \
        "archived text carrying markdown must still subtract"
    # ★ and the guard must not over-subtract: a move carrying one NEW claim that
    # restates another file still fires on that claim alone.
    assert hd("new.md", {}, moved + restated * 2, "", lambda: docs3, lambda: arch), \
        "a mixed write must still fire on the material that is not archived"
    # an unarchived duplicate is untouched by provenance
    assert hd("t", {}, prior + restated, prior, lambda: docs3, lambda: arch), \
        "provenance must not excuse a duplicate that was never archived"

    # ★ a dated history clause on a durable surface fires on the shape alone, warns
    # supervised, blocks a cycle, and never fires on a holding space or a ledger.
    stack = ("Some durable entry about hangar rates and the quote-gated structure of them.\n"
             "⚠ UPDATE (2026-08-08): the endpoint moved to the API repo and the rates changed.\n") * 2
    _m, _b = ms.hook_decision("entry.md", {}, stack, "", lambda: {})
    assert _m and "dated history clause" in _m and _b is False, _m
    _m, _b = ms.hook_decision("entry.md", {}, stack, "", lambda: {}, autonomous=True)
    assert _m and _b is True, "a cycle stacking a dated clause is blocked"
    scoped = ("Rebrand SHIPPED (v1.1.3), superseding the prior contractor iteration entirely.\n"
              "It moved into the API repo on 2026-08-02 and is no longer at the workspace root.\n") * 2
    _m, _b = ms.hook_decision("entry.md", {}, scoped, "", lambda: {})
    assert _m and "dated history clause" not in _m and "writes a date" in _m and _b is False, \
        f"a scoped date is not a history clause, and is still asked about: {_m}"
    _m, _b = ms.hook_decision("PROMOTIONS.md", {}, stack, "", lambda: {})
    assert not (_m and "dated history clause" in _m), "a holding space takes its dated transition"
    _m, _b = ms.hook_decision("HARNESS.md", {}, stack, "", lambda: {})
    assert _m is None, "a ledger is never checked"
    # ★ any other date on a durable surface is asked about: warns supervised, never blocks a
    # cycle, skips a dated heading, and is silent on a holding space or a ledger.
    attributed = ("The trim keeps only rules that generalize to any store (Peer ruling 2026-09-20).\n"
                  "Its apparatus clauses move to a Prime-only rule and the file ships to the Partial.\n") * 2
    _m, _b = ms.hook_decision("entry.md", {}, attributed, "", lambda: {})
    assert _m and "writes a date" in _m and _b is False, _m
    _m, _b = ms.hook_decision("entry.md", {}, attributed, "", lambda: {}, autonomous=True)
    assert _m and "writes a date" in _m and _b is False, "a bare date never blocks a cycle"
    headed = ("## Session 33, 2026-09-20\n"
              "The build passes and the widget target is unchanged since the convention audit.\n") * 2
    _m, _b = ms.hook_decision("entry.md", {}, headed, "", lambda: {})
    assert _m is None, f"a dated heading is convention: {_m}"
    _m, _b = ms.hook_decision("PROMOTIONS.md", {}, attributed, "", lambda: {})
    assert not (_m and "writes a date" in _m), "a holding space dates its transitions"
    _m, _b = ms.hook_decision("HARNESS.md", {}, attributed, "", lambda: {})
    assert _m is None, "a ledger is never checked for dates"

    # ★ the supervised/autonomous branch: same finding, different force. A cycle
    # cannot read a warning, so its guard stays hard; a session can, so it warns.
    _m, blocking = ms.hook_decision("t", {}, prior + restated, prior, lambda: docs3)
    assert _m and blocking is False, "a supervised nudge must warn, not block"
    _m, blocking = ms.hook_decision("t", {}, prior + restated, prior, lambda: docs3,
                                 autonomous=True)
    assert _m and blocking is True, "an autonomous nudge must still block"
    assert "about to be written" in _m and "landed" not in _m, _m
    _m, _b = ms.hook_decision("t", {}, prior + restated, prior, lambda: docs3)
    assert _m.startswith(ms.LEAD_WARNING), "the warn channel must not claim the write was stopped"
    # ⚠ `main` re-leads a warning into the blocking form when the 2e gate refuses
    # the same write. That swap is a literal replace, so it silently no-ops if the
    # two leads ever drift; this is the assertion that keeps them in step.
    assert _m.replace(ms.LEAD_WARNING, ms.LEAD_BLOCKING, 1).startswith(ms.LEAD_BLOCKING), \
        "the blocking re-lead must actually match what the warn channel emits"

    # within-file pairs are the uniqueness check and are invisible to `audit`
    dup = {"h.md": held * 2 + "\nUnrelated line about ramp availability and transient traffic.\n" + held}
    assert ms.cross_file_pairs(dup, same_file=True), "repeated claim inside one file must pair"
    assert not ms.cross_file_pairs(dup), "the same file must not pair against itself cross-file"

    # the walker and the write-time gate must cover the same files, or a
    # subdirectory is nudged against a corpus it is not a member of
    for d in hosts.EXCLUDED_DIRS:
        assert ms.corpus_for(COUNCIL / f"{d}/x.md") is None, d

    # --- the §2e question gate -------------------------------------------------
    # Driveable without ollama: an unreachable embedder degrades the message and
    # never changes the decision, which is the property that lets it block at all.
    d = Path(tempfile.mkdtemp())
    (d / "asked.jsonl").write_bytes(
        b'{"message":{"content":[{"type":"tool_use","name":"Bash",'
        b'"input":{"command":"python3 ~/.claude/scripts/memfind.py \\"x\\" council"}}]}}\n')
    (d / "discussed.jsonl").write_bytes(
        b'{"message":{"content":[{"type":"text","text":"memfind.py ranks candidate owners"}]}}\n')
    assert ms.memfind_asked(str(d / "asked.jsonl")), "a Bash invocation must count as asked"
    assert not ms.memfind_asked(str(d / "discussed.jsonl")), \
        "prose naming the script must NOT count; a session doing memory work names it constantly"
    assert ms.memfind_asked(str(d / "absent.jsonl")), "an unreadable transcript must not block"
    assert ms.memfind_asked(None)

    # ★ POSITION, not presence. Every string below contains the script name; only
    # the first group runs it. The `printf` case is the one that was live: it wrote
    # a test fixture containing the name and discharged a whole session's gate.
    for cmd in ('python3 ~/.claude/scripts/memfind.py "q" council',
                '~/.claude/scripts/memfind.py "q"',
                'locket find "q" council',
                'python3 ~/.claude/scripts/locket.py find "q"',
                'cd /tmp && locket find "q" | head',
                'cd /tmp && python3 scripts/memfind.py "q" memory',
                'python3 -X faulthandler memfind.py "q"',
                'PYTHONPATH=. python3 memfind.py "q"',
                'memfind.py "q" | head -20'):
        assert ms.runs_memfind(cmd), cmd
    for cmd in ('grep -n memfind.py scripts/*.py',
                'locket index council',
                'locket audit',
                'echo "locket find q"',
                'python3 ~/.claude/scripts/locket.py --index notes',
                'cat ~/.claude/scripts/memfind.py',
                'ls -l memfind.py',
                'sed -n "1,40p" memfind.py',
                'wc -l memfind.py memscan.py',
                'printf \'{"command":"python3 memfind.py x"}\' > fixture.jsonl',
                'echo "run memfind.py next"',
                '\'~/.claude/scripts/memfind.py "q"\',',
                'python3 - <<\'PY\'\npython3 scripts/memfind.py "q" x\nPY',
                'python3 -c "import re; print(\'memfind.py\')"'):
        assert not ms.runs_memfind(cmd), cmd

    (d / "mentioned.jsonl").write_bytes(
        b'{"message":{"content":[{"type":"tool_use","name":"Bash","input":'
        b'{"command":"grep -n memfind.py ~/.claude/scripts/memscan.py"}}]}}\n')
    assert not ms.memfind_asked(str(d / "mentioned.jsonl")), \
        "a Bash command that only NAMES the script must not discharge the gate"

    # ⚠ fail-open on an unrecognised schema. A false negative here cannot be
    # cleared by asking, because Write and Edit carry no MEMWRITE-OK equivalent,
    # so a transcript shape this function cannot read must never block.
    (d / "future-schema.jsonl").write_bytes(
        b'{"envelope":{"parts":[{"tool":"Bash","args":'
        b'{"cmd":"python3 memfind.py q"}}]}}\n')
    assert ms.memfind_asked(str(d / "future-schema.jsonl")), \
        "a marker this parser cannot structure must fail open, not block"

    unasked = {"transcript_path": str(d / "discussed.jsonl")}
    root = COUNCIL
    assert ms.memfind_gate(unasked, "t.md", {}, novel * 3, "", root), \
        "net-new material with the question unasked must fire"
    assert ms.memfind_gate({"transcript_path": str(d / "asked.jsonl")}, "t.md", {}, novel * 3, "",
                        root) is None, "asking the question once must silence the gate"
    for exempt in ("CHECKPOINTS.md", "SHORT_TERM.md"):
        assert ms.memfind_gate(unasked, exempt, {}, novel * 3, "", root) is None, exempt
    assert ms.memfind_gate(unasked, "MEMORY.md", {}, novel * 3, "", root), \
        "an index file is exempt from the lexical nudge, never from the question"
    assert ms.memfind_gate(unasked, "t.md", {}, novel, novel, root) is None, \
        "a reconsolidation adds nothing and must not fire"
    os.environ["CLAUDE_AUTONOMOUS"] = "1"
    assert ms.memfind_gate(unasked, "t.md", {}, novel * 3, "", root) is None, \
        "a cycle cannot act on a nudge and must never be blocked by one"
    del os.environ["CLAUDE_AUTONOMOUS"]
    shutil.rmtree(d, ignore_errors=True)
    # --- bashguard: fires on the thing, silent on every near-miss -------------
    # a command naming a scratch path is exempt by design, so under a HOME in
    # /tmp (Homebrew's test sandbox) every absolute-path case would pass silently
    # Claude Code's home is the real one for this block, under HOME, so every
    # home-relative spelling names it; the council store moves back after.
    paths.PATHS.host_stores["council"] = saved_council; ms.settings.cache_clear()
    if not any(m in str(hosts.claude()) + "/" for m in ms.SCRATCH_MARKERS):
        cmem = str(hosts.claude() / "projects" / "-Users-x-proj" / "memory")
        def bw(cmd, cwd=cmem, auto=False):
            return ms.bash_write_decision(cmd, cwd, autonomous=auto)
        assert bw("cat > MEMORY.md <<EOF")[1], "a redirect into a corpus file must block"
        assert bw("sed -i '' 's/a/b/' MEMORY.md")[1], "sed -i must block"
        assert bw(f"echo x >> {cmem}/topic_note.md")[1], "an absolute corpus path must block"
        rel = Path(cmem).relative_to(paths.HOME)
        for spelt in (f"~/{rel}", f"$HOME/{rel}", f"${{HOME}}/{rel}", f'"$HOME/{rel}"', f"'~/{rel}'"):
            assert bw(f"echo x >> {spelt}/topic_note.md")[1], f"a home-relative path must block: {spelt}"
        assert bw(f"cd ~/{rel} && echo x >> topic_note.md")[1], "a cd through ~ resolves a bare name"
        assert bw(f"cd ~nosuchuser9/x && echo x >> {cmem}/topic_note.md")[1], \
            "an unknown ~user elsewhere in the command must not disable the check"
        assert bw(f"cd /Users/x/elsewhere && cd ~/{rel} && echo x >> topic_note.md")[1], "the last cd wins"
        assert bw("cd .. && cd memory && echo x >> topic_note.md")[1], "a relative cd chains from the cwd"
        assert bw("echo x >> topic_note.md && cd /Users/x/elsewhere")[1], "a cd after the write does not move it"
        for sed in ("sed -Ei 's/a/b/' MEMORY.md", "sed --in-place 's/a/b/' MEMORY.md",
                    "sed -ibak 's/a/b/' MEMORY.md", "sed -in 's/a/b/p' MEMORY.md", "sed -i.bak 's/a/b/' MEMORY.md"):
            assert bw(sed)[1], f"an in-place sed must block: {sed}"
        for inplace in ("perl -pi -e 's/a/b/' MEMORY.md", f"perl -pi -e 's/a/b/' {cmem}/x.md",
                        "perl -i.bak -pe 's/a/b/' MEMORY.md", "perl -0pi -e 's/a/b/' MEMORY.md",
                        "ruby -pi -e 'gsub(/a/, \"b\")' MEMORY.md", "gawk -i inplace '{print}' MEMORY.md",
                        "gsed -i 's/a/b/' MEMORY.md"):
            assert bw(inplace)[1], f"an in-place edit must block: {inplace}"
        # a command outside the readers warns, so the next unlisted writer announces itself
        for unlisted in ("sponge MEMORY.md", "python3 fix.py MEMORY.md",
                         "find . -name x | xargs perl -pe 's/a/b/' MEMORY.md", "ex -sc 'wq' MEMORY.md"):
            msg, blk = bw(unlisted)
            assert msg and not blk and "read-only" in msg, f"an unlisted command stayed silent: {unlisted}"
        for semi in ("sed -i '' 's/a/b/; s/c/d/' MEMORY.md", "perl -pi -e 's/a/b/; s/c/d/' MEMORY.md",
                     "gawk -i inplace '{a=1; print}' MEMORY.md"):
            assert bw(semi)[1], f"a `;` in the program hid an in-place edit: {semi}"
        assert bw("gawk --include inplace 1 MEMORY.md")[1], "awk's spaced --include inplace"
        for glued in ("perl -ne'print' MEMORY.md", "perl -e'print 1' MEMORY.md", "perl -CSDio -e 1 MEMORY.md"):
            assert not bw(glued)[1], f"an `i` inside a glued argument read as in-place: {glued}"
        # a read-only flag's value is an input, scoped to the command that owns the flag
        for reads in ("claude --bare -p x --append-system-prompt-file MEMORY.md",
                      "claude -p x --system-prompt-file=MEMORY.md",
                      "ollama launch claude --append-system-prompt-file MEMORY.md"):
            assert bw(reads) == (None, False), f"a read-only flag's value warned: {reads}"
        for writes in ("claude -p x --debug-file MEMORY.md", "other --append-system-prompt-file MEMORY.md",
                       "ollama launch claude -p x MEMORY.md"):
            msg, blk = bw(writes)
            assert msg and not blk, f"a file outside a read-only flag stayed silent: {writes}"
        msg, _ = bw("python3 x.py > /Users/x/elsewhere/log.txt MEMORY.md")
        assert msg and "`python3`" in msg, f"a redirect target was read as the command: {msg}"
        for gitw in ("git checkout -- MEMORY.md", "git restore MEMORY.md", "git rm MEMORY.md",
                     "git -C /Users/x rm MEMORY.md", "git --git-dir=.git rm MEMORY.md",
                     "git clean -f MEMORY.md"):
            msg, blk = bw(gitw)
            assert msg and not blk and "`git " in msg, f"a writing git subcommand stayed silent: {gitw}"
        msg, blk = bw("""awk '{print > "MEMORY.md"}' /Users/x/elsewhere/in.txt""")
        assert msg and blk, "awk writing from its program text (the redirect scan reads it) passed"
        for nested, word in (("timeout 5 python3 fix.py MEMORY.md", "python3"),
                             ("diff <(python3 fix.py MEMORY.md) x", "python3"),
                             ("find . -exec perl -pe 1 {} MEMORY.md \\;", "perl"),
                             ("echo `python3 fix.py MEMORY.md`", "python3")):
            msg, blk = bw(nested)
            assert msg and not blk and f"`{word}`" in msg, f"a nested command was misread: {nested} -> {msg}"
        # a `cd` that is not a command moves nothing: each of these still writes into the cwd's corpus
        for loud in (
            'echo "cd /Users/x/elsewhere to look at notes" >> topic_note.md',
            'grep -rn "cd /Users/x" README.md >> topic_note.md',
            "# cd /Users/x/elsewhere if this breaks\necho x >> topic_note.md",
            'echo "notes; cd /Users/x/elsewhere" >> topic_note.md',          # a separator inside quotes
            "true  # then; cd /Users/x/elsewhere\necho y >> topic_note.md",   # a separator inside a comment
            "echo see cd /Users/x/elsewhere >> topic_note.md",               # cd as an argument
            "cat <<'EOF' > /Users/x/script.sh\ncd /Users/x/elsewhere\nEOF\necho x >> topic_note.md",
            "pushd /Users/x/elsewhere && popd && cd ~/" + str(rel) + " && echo x >> topic_note.md",
        ):
            assert bw(loud)[1], f"a real corpus write passed: {loud!r}"
        assert bw("pushd /Users/x/elsewhere && echo x >> topic_note.md") == (None, False), "pushd moves like cd"
        # the false positives that taught the resolver: each is silent
        for quiet in (
            "cd /Users/x/elsewhere && sed -i '' 's/a/b/' make-it-yours.md",   # a cd away from the corpus
            "cd /Users/x/elsewhere && echo x > topic_note.md",
            "sed -n '/a/,/b/p' make-it-yours.md",                             # `-i` inside a file name
            "sed -e 's/-i/x/' notes.txt > /Users/x/out.txt",
            'cd "$(mktemp -d)" && echo x > a.md',                              # a cd computed at run time
            "cd - && echo x > topic_note.md",                                  # the previous dir, untracked
        ):
            assert bw(quiet) == (None, False), f"decoy fired: {quiet}"
        # a run-time variable is undecidable from the command: expanded, it may warn, never block
        saved = {k: os.environ.get(k) for k in ("LOCKET_T_OUT", "LOCKET_T_MEM", "LOCKET_T_UNSET")}
        os.environ.update(LOCKET_T_OUT="/Users/x/elsewhere/", LOCKET_T_MEM=cmem)
        os.environ.pop("LOCKET_T_UNSET", None)
        for quiet in ('cat > "$LOCKET_T_OUT/pr.md" <<EOF', "cat > ${LOCKET_T_OUT}pr.md",
                      "echo x >> $LOCKET_T_UNSET/topic_note.md", 'echo x > "$(mktemp -d)/n.md"'):
            assert bw(quiet) == (None, False), f"a variable outside the corpus fired: {quiet}"
        msg, blk = bw("echo x >> $LOCKET_T_MEM/topic_note.md")
        assert msg and not blk, "a variable expanding into a corpus warns rather than blocks"
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        msg, blk = bw("python3 - <<'EOF'\npathlib.Path('MEMORY.md').write_text(s)\nEOF")
        assert msg and not blk, "a scripted write warns rather than blocks"
        # a Python program's targets are read off the program: names inside its data are not
        # destinations, and a directory joined in another token still decides membership
        out = "/Users/x/elsewhere"
        for quiet in (
            # CMPCanon: rule names as replacement strings in an edit of a file outside the corpus
            f"python3 - <<'EOF'\nimport pathlib\np=pathlib.Path('{out}/lint.py')\n"
            "p.write_text(p.read_text().replace('memory-discipline.md', 'prime-memory-discipline.md'))\nEOF",
            # CMPCurator: a directory outside the corpus joined to bare names, one an f-string
            f"cd {out} && python3 - <<'EOF'\nimport pathlib\nFD=pathlib.Path('research/fd')\n"
            "rows=(FD/'SERVICE-INDEX.md').read_text()\nfor f in ('CT','F2'):\n"
            "    (FD/f'SERVICE-INDEX-{f}-US.md').write_text(rows)\nEOF",
            f"python3 - <<'EOF'\nfor n in ['a.md', 'b.md']:\n    open('{out}/' + n, 'w').write('x')\nEOF",
            f"python3 -c 'import pathlib; pathlib.Path(\"{out}/n.md\").write_text(\"MEMORY.md\")'",
            "python3 - <<'EOF'\nopen('/Users/x/out.txt', 'w').write(open('MEMORY.md').read())\nEOF",  # a read
            "python3 - <<'EOF'\nimport json, re\nd = json.loads(s)\nre.sub('a', 'b', d)\nprint('MEMORY.md')\nEOF",  # known calls, no writer
            f"cd {out} && python3 - <<'EOF'\nimport json, re\nd = json.loads(s)\nre.sub('a', 'b', d)\nEOF",
            "python3 - <<'EOF'\n" + "s = 'a'" + " + 'a'" * 3000 + "\nEOF",       # too deep to fold: no crash
            # a routine transform: read the corpus, call unlisted helpers with no `.md` in hand,
            # write outside it; neutral calls must not reopen the token scan over the read
            "python3 - <<'EOF'\nimport argparse, datetime, sys, json, re\nfrom collections import Counter\n"
            "from pathlib import Path\nt = Path('MEMORY.md').read_text()\nnow = datetime.datetime.now()\n"
            "top = Counter(re.findall('x', t)).most_common(3)\nsys.stdout.write(str(top))\n"
            "d = dict.fromkeys(['a'])\nargparse.ArgumentParser()\nimport subprocess\n"
            "subprocess.run(['git', 'status'])\n"
            "Path('/Users/x/out.json').write_text(json.dumps(d))\nEOF",
        ):
            assert bw(quiet) == (None, False), f"a Python program's data read as a target: {quiet!r}"
        for loud in (
            "python3 - <<'EOF'\nimport pathlib\nFD=pathlib.Path('.')\n(FD/f'NOTE-{x}.md').write_text(s)\nEOF",
            f"cd {out} && python3 - <<'EOF'\nFD='{cmem}'\nopen(FD + '/fresh_note.md', 'a').write('x')\nEOF",
            "python3 - <<'EOF'\nfor n in ['a.py', 'topic_note.md']:\n    open(n, 'w')\nEOF",
            "python3 -c 'open(\"fresh_note.md\", \"w\")'",
            "python3 - <<'EOF'\nimport shutil\nshutil.copy('/Users/x/in.txt', 'fresh_note.md')\nEOF",
            # a target the program computes is undecidable, so the token scan of old still answers
            "python3 - <<'EOF'\nimport sys, pathlib\np=pathlib.Path(sys.argv[1])\np.write_text('MEMORY.md')\nEOF",
            "python3 - <<'EOF'\nimport os, pathlib\nos.chdir('/Users/x')\npathlib.Path('MEMORY.md').write_text(s)\nEOF",
            "python3 - <<'EOF'\nthis does not parse (\nopen('MEMORY.md', 'w')\nEOF",
            # the convention audit's silent passes: each is a write the first cut missed
            "python3 - <<'EOF'\nfrom shutil import copy\ncopy('/Users/x/a', 'MEMORY.md')\nEOF",
            "python3 - <<'EOF'\nimport shutil as sh\nsh.copy('/Users/x/a', 'fresh_note.md')\nEOF",
            "python3 - <<'EOF'\no = open\no('fresh_note.md', 'w')\nEOF",
            "python3 - <<'EOF'\nfrom pathlib import Path\nw = Path('MEMORY.md').write_text\nw(s)\nEOF",
            "python3 - <<'EOF'\nfrom pathlib import Path\ngetattr(Path('MEMORY.md'), 'write_text')(s)\nEOF",
            "python3 - <<'EOF'\nfrom pathlib import Path\nPath('/Users/x/z').rename('MEMORY.md')\nEOF",
            "python3 - <<'EOF'\nfrom pathlib import Path\nPath('/Users/x/z').replace('fresh_note.md')\nEOF",
            "python3 - <<'EOF'\nimport os\nos.open('fresh_note.md', os.O_WRONLY | os.O_CREAT)\nEOF",
            "python3 - <<'EOF'\nfrom pathlib import Path\nPath('fresh_note.md').touch()\nEOF",
            "python3 - <<'EOF'\nimport subprocess\nsubprocess.run(['cp', '/Users/x/a', 'MEMORY.md'])\nEOF",
            "python3 - <<'EOF'\nopen('/Users/x/o.txt', 'wb').write(b'x')\nopen('fresh_note.md', 'wb')\nEOF",
            "python3 - <<'EOF'\np = '/ok.txt'\ntry:\n    pass\nexcept OSError as p:\n    pass\nopen(p, 'w').write('MEMORY.md')\nEOF",
            f"cd {out} && python3 - <<'EOF'\nimport os\nos.chdir('{cmem}')\nopen('fresh_note.md', 'w')\nEOF",
            f"cd {out} && python3 - <<'EOF'\nfrom os import chdir\nchdir('{cmem}')\nopen('fresh_note.md', 'w')\nEOF",
            "python3 - <<'EOF'\nimport shutil\nthis does not parse (\nshutil.copy(a, 'fresh_note.md')\nEOF",
            "python3 - <<'EOF'\nimport sys\nd = sys.argv[1]\nopen(f'{d}/MEMORY.md', 'w')\nEOF",
            "python3 - <<'EOF'\nimport sys\nopen(sys.argv[1] + '/MEMORY.md', 'w')\nEOF",
            # the second pass: an unknown call handed a `.md` path may write it
            "python3 - <<'EOF'\nimport mylib\nmylib.save('MEMORY.md', 'x')\nEOF",
            "python3 - <<'EOF'\nimport os\nos.truncate('MEMORY.md', 0)\nEOF",
            "python3 - <<'EOF'\nimport fileinput\nfor l in fileinput.input('MEMORY.md', inplace=True):\n    print(l)\nEOF",
            "python3 - <<'EOF'\nfrom shutil import *\ncopy('/Users/x/a', 'MEMORY.md')\nEOF",
            f"python3 - <<'EOF'\nimport shutil\nshutil.copy('/Users/x/draft.md', '{cmem}')\nEOF",
            f"python3 - <<'EOF'\nimport shutil\nshutil.move('/Users/x/draft.md', '{cmem}/')\nEOF",
            "python3 - <<'EOF'\nfrom pathlib import Path\nPath('/Users/x/z').rename(target='MEMORY.md')\nEOF",
            "python3 - <<'EOF'\nimport codecs\ncodecs.open(filename='fresh_note.md', mode='w')\nEOF",
        ):
            msg, blk = bw(loud)
            assert msg and not blk, f"a scripted corpus write stayed silent: {loud!r}"
        for quiet in (
            "cat MEMORY.md",                               # a read
            "grep -n foo MEMORY.md",                       # a read
            "cp MEMORY.md /private/tmp/scratchpad/x.bak",  # backup OUT of the corpus
            "cat > CHECKPOINTS.md",                        # ledger, exempt by 2d
            "cat > SHORT_TERM.md",                         # holding space, exempt
            "cat > archive/ARCHIVE-old.md",                # archived, exempt
            "git add MEMORY.md && git diff MEMORY.md",     # readers, and git is one
            "git log -p MEMORY.md && git show HEAD:MEMORY.md",
            "sed -n 's/a/b/; p' MEMORY.md",                # no in-place flag
            "awk '{print}' MEMORY.md > /Users/x/elsewhere/o.txt",
            "wc -l MEMORY.md | sort -n",
            "for f in *.md; do wc -l \"$f\"; done",      # a loop header and a glob
            'hermes chat -q "tidy MEMORY.md please"',      # prose in a quoted argument
            "grep x MEMORY.md # then perl over it",        # a comment
            'grep -nE "scripts/(a|b)\\.py" MEMORY.md topic_note.md',   # parens inside a quoted pattern
            "sed -fimport.sed MEMORY.md",                                # an `i` inside a glued file name
            "grep x > /Users/x/elsewhere/res.txt MEMORY.md",           # a redirect is not the command
            "perl -e 'print 1' > /Users/x/elsewhere/o.txt",
            "diff <(sort MEMORY.md) <(sort topic_note.md)",      # readers inside substitutions
            "curl -s https://example.com/README.md | head",   # a URL is not a path
            "find . \\( -iname 'A*.md' -o -name 'NOTES.md' \\) -print",  # escaped parens group
        ):
            assert bw(quiet) == (None, False), f"decoy fired: {quiet}"
        assert bw("cat > MEMORY.md", auto=True) == (None, False), \
            "an autonomous cycle must never be blocked (2d: it cannot report the failure)"
        assert bw("cat > MEMORY.md  # MEMWRITE-OK: scripted multi-file patch") == (None, False), \
            "a declared escape must pass"
        # Resolving rather than name-matching: the SAME command is a corpus write from
        # inside a corpus and nothing at all from outside one.
        assert bw("cat > README.md")[1], "a .md written into a corpus dir is a corpus write"
        assert bw("cat > README.md", cwd="/Users/x/some-repo") == (None, False), \
            "the same command outside a corpus must stay silent"
    paths.PATHS.host_stores["council"] = COUNCIL; ms.settings.cache_clear()

    # --- grepassist: only a question, never a regex ---------------------------
    assert ms.prose_like("workspace root is a git repository")
    assert not ms.prose_like("waive|prototype"), "an alternation is not a question"
    assert not ms.prose_like("v_ne\\b"), "a metacharacter disqualifies"
    assert not ms.prose_like("repo"), "one content word is not a question"
    assert not ms.prose_like(""), "an empty pattern must not fire"
    for ident in ("angle-scout", "cmp-fanout-is-a-dossier", "settings.json",
                  "REFERENCE-CMP-Rulings.csv", "spec_provenance tcds_ref", "~/brain/fact.md"):
        assert not ms.prose_like(ident), f"a joined identifier read as a phrase: {ident}"
    assert ms.prose_like("source of truth for prices"), "a real phrase must still fire"
    # --- grepassist's decision log: never raises, trims atomically, records every branch
    import subprocess
    real_paths, logdir = paths.PATHS, Path(tempfile.mkdtemp())
    try:
        paths.PATHS = real_paths.replace(grepassist_log=logdir / "ro" / "ga.csv")
        (logdir / "ro").mkdir(); (logdir / "ro").chmod(0o500)
        assert grepassist.log("Bash", "declined", "x", "p") == 0, "an unwritable log must not raise"
        (logdir / "ro").chmod(0o700)
        paths.PATHS = real_paths.replace(grepassist_log=logdir / "ga.csv")
        paths.PATHS.grepassist_log.write_text("".join(f"2026-01-01T00:00:00Z,Bash,declined,x,row {i}\n" for i in range(4000)))
        grepassist.trim_log(paths.PATHS.grepassist_log, 1000)
        kept = paths.PATHS.grepassist_log.read_text().splitlines()
        assert kept and kept[-1].endswith("row 3999") and len(kept) < 4000 and all(r.count(",") == 4 for r in kept), \
            "a trim must keep whole newest rows"
        assert grepassist.log("Bash", "declined", "x", "=cmd") == 0 and \
            paths.PATHS.grepassist_log.read_text().splitlines()[-1].endswith(",'=cmd"), "a formula-led pattern must be quoted"
        assert paths.PATHS.grepassist_log.stat().st_mode & 0o777 in (0o600, 0o644), paths.PATHS.grepassist_log.stat().st_mode
        fresh = logdir / "hook.csv"
        nowhere = Path(tempfile.mkdtemp())
        for cmd, reason in (("grep -rn 'a fact here' .", "no-corpus"), ("echo hi", None)):
            payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Bash", "cwd": str(nowhere),
                                  "tool_input": {"command": cmd}})
            r = subprocess.run([sys.executable, str(SRC / "memscan.py"), "grepassist"], input=payload,
                               capture_output=True, text=True, timeout=60, cwd=str(nowhere),
                               env={**os.environ, "LOCKET_GREPASSIST_LOG": str(fresh)})
            assert r.returncode == 0, (cmd, r.stderr)
        rows = fresh.read_text().splitlines() if fresh.exists() else []
        assert len(rows) == 1 and ",declined,no-corpus," in rows[0], rows
        assert fresh.stat().st_mode & 0o777 == 0o600, "a new log must be created 0600"
    finally:
        paths.PATHS = real_paths
        shutil.rmtree(logdir, ignore_errors=True)

    # --- portability: a manifest makes any directory a corpus -----------------
    d = Path(tempfile.mkdtemp())
    store = d / "brain"
    (store / "old").mkdir(parents=True)
    (store / "LOG.md").write_text("entry one\n")
    (store / "facts.md").write_text("A fact.\n")
    assert ms.corpus_for(store / "facts.md") is None, "no manifest, no host store: not a corpus"
    (store / ms.MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "ledger_surfaces": ["LOG.md"],
        "holding_spaces": ["INBOX.md"], "index_files": [],
        "code_roots": [str(d)]}))
    ms.settings.cache_clear()
    assert ms.corpus_for(store / "facts.md") == store, "a manifest names the corpus"
    assert ms.corpus_for(store / "old" / "x.md") is None, "the manifest's excluded dir holds"
    assert ms.corpus_for(store / "archive" / "x.md") == store, \
        "a manifest REPLACES the excluded set; `archive` is ours, not theirs"
    s = ms.settings(store)
    assert s["ledger_surfaces"] == {"LOG.md"} and "INBOX.md" in s["append_surfaces"], s
    assert ".memfind" in s["excluded_dirs"], "the cache dir is excluded whatever the manifest says"
    assert [r for r, _ in ms.code_roots(store)] == [d], "declared code roots win outright"
    assert ms.md_files(store) == [store / "LOG.md", store / "facts.md"], ms.md_files(store)
    assert ms.bash_write_decision("cat > LOG.md", str(store)) == (None, False), \
        "a ledger named by the manifest is exempt in bashguard too"
    assert ms.bash_write_decision("cat > facts.md", str(store))[1], \
        "a plain file in a manifest corpus blocks"
    # grepassist places a search by its path words as the shell expands them, from a cwd in no store
    outside, real_home = Path(tempfile.mkdtemp()), os.environ["HOME"]
    assert ms.corpus_here(outside) is None, "the cwd fallback has nothing here, so the path must place it"
    os.environ["HOME"] = str(d)
    try:
        for cmd in ("grep -rn 'a fact here' ~/brain", "rg 'a fact here' $HOME/brain/*.md",
                    "grep 'a fact here' ~/brain/facts.md", "grep 'a fact here' ~/brain/fact*.md"):
            assert ms.search_root(shlex.split(cmd), outside) == store, cmd
        assert ms.search_root(["grep", "-rn", "../facts.md"], store / "old") == store, \
            "a relative path resolves against the session's cwd"
        assert ms.search_root(["grep", "~/nowhere"], outside) is None, "a path that does not exist places nothing"
        here = d / "cwd-store"; here.mkdir(); (here / ms.MANIFEST).write_text("{}"); ms.settings.cache_clear()
        assert ms.search_root(["grep", "-rn", "~/brain"], here) == store, \
            "a missing word (the verb) resolved against a cwd in another store must not place the search there"
        q = "a fact here"
        for cmd in (f"rg --glob '*.md' '{q}' ~/brain", f"cd /tmp && grep -rn '{q}' ~/brain 2>/dev/null | head",
                    f"grep -rne '{q}' ~/brain", f"grep --regexp='{q}' ~/brain", f"grep -e'{q}' ~/brain",
                    f"grep -rn '{q}' ~/brain 2>&1 | head", f"grep -rn -- '{q}' ~/brain",
                    f"ls | xargs grep -l '{q}' ~/brain", f"git grep '{q}' -- ~/brain",
                    f"which grep; grep '{q}' ~/brain", f"bash -c \"grep -rn '{q}' ~/brain\""):
            assert ms.grep_words(cmd, outside) == (q, ["~/brain"]), (cmd, ms.grep_words(cmd, outside))
        assert ms.grep_words(f"echo '{q}' | wc -l", outside) is None, "no grep verb, no search"
        assert ms.grep_words(f"grep -rn foo {store}", outside) == ("", []), \
            "an existing path is never the pattern, however prose-like its words"
        assert ms.grep_words("grep -rn foo ~/not-there/memory-store", outside) == ("", []), \
            "a path-shaped word is never the pattern, existing or not"
        assert ms.grep_words(f"rg -e '{q}' -e 'other words here' ~/brain", outside) == (q, ["~/brain"]), \
            "a second -e value leaked into the path words"
        assert ms.grep_words(f"grep '{q} ~/brain", outside) is not None, "unbalanced quotes must still parse"
        assert ms.grep_words(f"git commit -m \"which grep; grep '{q}' ~/brain\"", outside) is None, \
            "a quoted word that only mentions grep is not a command"
        pat, words = ms.grep_words(f"grep '{q}' --exclude-dir cwd-store ~/brain", d)
        assert pat == q and ms.search_root(words, d) == store, \
            "a flag value naming a store under the cwd shadowed the path after it"
        # the first pattern word decides: a regex there means no question, never a later word
        for regex in ("grep -rln 'NOT-CURATION\\|dossier' hooks/ settings.json",
                      "grep -c '\\.py' REFERENCE-CMP-Rulings.csv REFERENCE-CMP-Claims.csv",
                      "grep 'a|b' 'notes about fact here'"):
            assert ms.grep_words(regex, outside) == ("", []), (regex, ms.grep_words(regex, outside))
        for valued in (f"grep -A 12 '{q}' ~/brain", f"grep --exclude-dir node_modules -rn '{q}' ~/brain",
                       f"grep -m 5 --include '*.md' '{q}' ~/brain", f"rg -g '*.md' '{q}' ~/brain",
                       f"rg -t py '{q}' ~/brain", f"rg -g '!vendor' '{q}' ~/brain",
                       f"grep -d skip '{q}' ~/brain", f"grep -f pats.txt '{q}' ~/brain"):
            assert ms.grep_words(valued, outside) == (q, ["~/brain"]), (valued, ms.grep_words(valued, outside))
    finally:
        os.environ["HOME"] = real_home
    # --- a project store answers to its manifest's name, and to its slug still
    import contextlib, io
    real_claude, fake = hosts.claude(), Path(tempfile.mkdtemp())
    proj = fake / "projects" / "-Users-x-devdocs" / "memory"
    proj.mkdir(parents=True); (proj / "a.md").write_text("A note about the project.\n")
    paths.PATHS.host_stores["council"] = fake
    try:
        assert "-Users-x-devdocs" in ms.all_corpora(), "an unnamed project store is keyed by its slug"
        (proj / ms.MANIFEST).write_text(json.dumps({"name": "devdocs-main"})); ms.settings.cache_clear()
        assert ms.all_corpora().get("devdocs-main") == proj, "a named project store is keyed by its name"
        assert ms.resolve("devdocs-main") == proj and ms.resolve("-Users-x-devdocs") == proj, \
            "the name resolves, and the slug still does"
        persona = fake / "projects" / "-Users-x-devdocs-Persona" / "memory"
        persona.mkdir(parents=True); (persona / "p.md").write_text("A persona note.\n")
        ms.settings.cache_clear()
        assert ms.resolve("-Users-x-devdocs") == proj, \
            "a named store's exact slug beats a longer unnamed slug containing it"
        assert ms.resolve("-Users-x-devdocs-Persona") == persona and ms.resolve("devdocs-main") == proj
        assert ms.resolve("-USERS-X-DEVDOCS") == proj, "an exact slug matches without case"
        shutil.rmtree(persona.parent); ms.settings.cache_clear()
        loose = fake / "zq-loose-vault" / "memory"; loose.mkdir(parents=True)
        (loose / "v.md").write_text("A vault note.\n")
        (loose / ms.MANIFEST).write_text(json.dumps({"name": "zq-named"}))
        assert ms.register(loose); ms.settings.cache_clear()
        with contextlib.redirect_stderr(io.StringIO()):
            assert ms.resolve("zq-loose-vault") is None, "only a projects/ store answers to its parent's name"
        assert ms.forget(loose); ms.settings.cache_clear()
        kid = fake / "projects" / "-Users-x-kid" / "memory"; kid.mkdir(parents=True)
        (kid / "b.md").write_text("Another note.\n")
        (kid / ms.MANIFEST).write_text(json.dumps({"belongs_to": "-Users-x-devdocs"})); ms.settings.cache_clear()
        assert ms.parent_of(kid) == proj, "a belongs_to written by slug still joins the named store"
        (proj / ms.MANIFEST).write_text(json.dumps({"name": "council"})); ms.settings.cache_clear()
        assert ms.raw_corpora().get("council") != proj and "-Users-x-devdocs" in ms.raw_corpora(), \
            "a project name never takes a host key"
        (proj / ms.MANIFEST).write_bytes(b'{"name": "\xff"}'); ms.settings.cache_clear()
        assert ms.read_manifest(proj) == {} and "-Users-x-devdocs" in ms.raw_corpora(), \
            "a manifest that is not UTF-8 degrades instead of raising"
    finally:
        paths.PATHS.host_stores["council"] = real_claude
        shutil.rmtree(fake, ignore_errors=True); ms.settings.cache_clear()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        hygiene.cmd_budget(store).show()
    assert "Largest" not in out.getvalue() and "bytes" not in out.getvalue(), \
        f"a store under budget printed an empty table: {out.getvalue()}"
    # --- init --ledgers: the built-in ledgers, header only, two registered as sources
    man_before = (store / ms.MANIFEST).read_text()
    with contextlib.redirect_stdout(io.StringIO()):
        assert ms.scaffold_ledgers(store, "Brain") == 0
    for pattern, glob, _ in ms.LEDGER_FILES:
        f = store / pattern.format("Brain")
        assert f.read_text() == ",".join(ms.LEDGER_SCHEMAS[glob]["columns"]) + "\n", f.name
        assert not ms.check_ledger_text(ms.ledger_schema_for(f)[0], f.read_text()), f"a fresh {f.name} fails its schema"
    srcs = ms.read_manifest(store)["sources"]
    assert sorted(e["path"] for e in srcs) == ["CLAIMS-Brain.csv", "RULINGS-Brain.csv"], srcs
    assert ms.read_manifest(store)["excluded_dirs"] == ["old"], "the manifest's other keys are kept"
    (store / "RULINGS-Brain.csv").write_text("Ruling,Category,Area,Date\nKeep it.,x,y,2026-09-29\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert ms.scaffold_ledgers(store, "Brain") == 0
    assert "Keep it." in (store / "RULINGS-Brain.csv").read_text(), "an existing ledger is never touched"
    assert len(ms.read_manifest(store)["sources"]) == 2, "a source is never listed twice"
    assert ms.session_ledger(store).name == "HISTORY-Brain-Sessions.csv"
    with contextlib.redirect_stderr(io.StringIO()):
        assert ms.scaffold_ledgers(store, "a/b") == 1 and ms.scaffold_ledgers(store, "*") == 1
    _m, _b = ms.hook_decision("entry.md", {}, stack, "", lambda: {}, root=store)
    assert _m and "a row in HISTORY-Brain-Sessions.csv" in _m and not _b, _m
    _m, _ = ms.hook_decision("entry.md", {}, stack, "", lambda: {}, root=d)
    assert _m and "HISTORY-" not in _m, "no ledger, no pointer at one"
    for pattern, _g, _ in ms.LEDGER_FILES:
        (store / pattern.format("Brain")).unlink()
    # a ledger already present under another name is the store's own, registered not doubled
    (store / "RULINGS-Old.csv").write_text("Ruling,Category,Area,Date\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert ms.scaffold_ledgers(store, "Brain") == 0
    assert not (store / "RULINGS-Brain.csv").exists(), "a second RULINGS ledger split the store's rows"
    assert "RULINGS-Old.csv" in {e["path"] for e in ms.read_manifest(store)["sources"]}
    for pattern, _g, _ in ms.LEDGER_FILES:
        for f in store.glob(_g):
            f.unlink()
    # a manifest it cannot parse, or `sources` that is not a list, is refused and never rewritten
    for bad in ('{"name": "a", "excluded_dirs": ["old"], "sources": [', '{"sources": {"k": 1}}'):
        (store / ms.MANIFEST).write_text(bad); ms.settings.cache_clear()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert ms.scaffold_ledgers(store, "Brain") == 1, bad
        assert (store / ms.MANIFEST).read_text() == bad, f"a bad manifest was rewritten: {bad}"
        assert not list(store.glob("*.csv")), "no ledger is written beside a manifest it refused"
    (store / ms.MANIFEST).write_text(man_before)
    ms.settings.cache_clear()
    spaced = d / "my notes"; spaced.mkdir(); (spaced / "a.md").write_text("A note.\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert ms.cmd_init(spaced, index=False, ledgers="") == 0, "a folder name with a space still takes ledgers"
    assert (spaced / "HISTORY-my-notes-Sessions.csv").is_file()
    with contextlib.redirect_stderr(io.StringIO()):
        assert ms.cmd_init(spaced, index=False, ledgers="a b") == 1, "an explicit bad name is refused"
    ms.forget(spaced)
    assert "old" in ms.settings(store)["excluded_dirs"] and "backups" in ms.settings(store)["excluded_dirs"], \
        "a backup directory is excluded whatever the manifest says"
    (store / ".hidden").mkdir(); (store / ".hidden" / "h.md").write_text("A hidden note.\n")
    assert store / ".hidden" / "h.md" not in ms.md_files(store), "dot-directories are never corpus"
    # an index in a member subdirectory points beside itself, not at the root
    (store / "sub").mkdir(); (store / "sub" / "t.md").write_text("A target.\n")
    (store / "sub" / "MEMORY.md").write_text("- [T](t.md) the target\n- [G](gone.md) removed\n")
    (store / ms.MANIFEST).write_text(json.dumps({"excluded_dirs": ["old"], "index_files": ["MEMORY.md"]}))
    ms.settings.cache_clear()
    ih = {h[1]: h[4].exists() for h in hygiene.index_hooks(store)}
    assert ih == {"sub/t.md": True, "sub/gone.md": False}, ih
    shutil.rmtree(store / "sub")

    # --- the CSV boundary: registered rows are IN the corpus --------------------
    (store / "RULES.csv").write_text(
        "Ruling,Area,Date\n"
        '"Serve the estimate bottom-up from priced components and never from an aggregate.",cost,2026-01-01\n'
        '"A rebase fights the watcher, so build the sheet and commit it before rebasing.",dev-loop,2026-01-02\n')
    (store / ms.MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "index_files": [], "ledger_surfaces": ["LOG.md"],
        "holding_spaces": ["INBOX.md"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    ms.settings.cache_clear()
    c = ms.corpus(store)
    assert "RULES.csv#cost" in c and "RULES.csv#dev-loop" in c, sorted(c)
    assert "RULES.csv#cost" not in ms.corpus(store, csv=False), "memfind takes rows as its own units"
    (store / "notes.md").write_text(
        "Serve the estimate bottom-up from priced components and never from an aggregate.\n")
    pairs = ms.cross_file_pairs(ms.corpus(store))
    assert any({fa, fb} == {"notes.md", "RULES.csv#cost"} for _, fa, fb, _, _ in pairs), \
        "a CSV row copied into prose is a fork the audit must see"
    # and the reverse at write time: a new row restating prose warns, never blocks
    rules_prior = (store / "RULES.csv").read_text()
    row = '"Serve the estimate bottom-up from priced components, never from an aggregate.",pricing,2026-01-03\n'
    msg, blocking = ms.source_row_decision(str(store / "RULES.csv"), {"new_string": row}, rules_prior, append=True)
    assert msg and "notes.md" in msg and not blocking, msg
    assert ms.source_row_decision(str(store / "RULES.csv"), {"content": rules_prior}, rules_prior) == (None, False), \
        "a rewrite adding no row draws nothing"
    edit = {"old_string": "never from an aggregate", "new_string": "not from an aggregate"}
    msg = ms.source_row_decision(str(store / "RULES.csv"), edit, rules_prior)[0] or ""
    assert "- RULES.csv " not in msg, "an edited row never matches the version it replaces"
    msg, _ = ms.hook_decision("new.md", {}, "A rebase fights the watcher, so build the sheet and commit it "
                           "before rebasing. " * 3, "", lambda: ms.corpus(store), lambda: [], sets=ms.settings(store))
    assert msg and "RULES.csv#dev-loop" in msg, msg
    # query pointers: the [[wikilink]] of a ruling, checked against the rows
    (store / "guide.md").write_text(
        "Query `RULES.csv` on `Area=cost` before pricing anything.\n"
        "The rebase ruling (`RULES.csv`, query `Area=dev-loop|frontend`) governs the sheet.\n"
        "See `OTHER.csv` (query `Area=cost`) for the rest.\n"
        "Never query `RULES.csv` on `Zone=cost`, which is the old column.\n"
        "The file `RULES.csv` holds `Area=cost` rows, but this sentence is not a pointer.\n")
    st = {(x[1], x[2], x[3]): x[4] for x in hygiene.csv_pointers(store)}
    assert st[("RULES.csv", "Area", "cost")] == "ok", st
    assert st[("RULES.csv", "Area", "dev-loop|frontend")] == "DANGLING frontend", st
    assert st[("OTHER.csv", "Area", "cost")] == "UNREGISTERED", st
    assert st[("RULES.csv", "Zone", "cost")] == "NO COLUMN", st
    assert len(hygiene.csv_pointers(store)) == 4, "a mention without the word query is not a pointer"

    # --- graduated: a holding-space claim that a durable source now holds -------
    import contextlib, io
    (store / ms.MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "index_files": [], "ledger_surfaces": ["LOG.md"],
        "holding_spaces": ["INBOX.md", "QUEUE.md"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    ms.settings.cache_clear()
    (store / "INBOX.md").write_text(
        "A rebase fights the watcher, so build the sheet and commit it before rebasing.\n"
        "Gate: a third instance where the watcher ordering bites on a freshly built sheet.\n")
    (store / "QUEUE.md").write_text(
        "Gate: a third instance where the watcher ordering bites on a freshly built sheet.\n")
    (store / "LOG.md").write_text(
        "entry one\nGate: a third instance where the watcher ordering bites on a freshly built sheet.\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = _shown(overlap.cmd_graduated(store))
    txt = buf.getvalue()
    assert n == 1 and "INBOX.md" in txt and "RULES.csv#dev-loop" in txt, txt
    body = txt.split("\n\n", 1)[1]
    assert "QUEUE.md" not in body, "a holding-to-holding pair is a ledger restating a gate, not a graduation"
    assert "LOG.md" not in body, "a ledger entry quoting a gate is chronology, not a graduation"
    assert "2 holding files (INBOX.md, QUEUE.md) against" in txt, txt
    (store / "INBOX.md").unlink(); (store / "QUEUE.md").unlink()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert _shown(overlap.cmd_graduated(store)) == 0
    assert "none of the configured holding spaces" in buf.getvalue(), \
        "a clean run still says what it was scored against"
    (store / ms.MANIFEST).write_text("{not json")
    ms.settings.cache_clear()
    assert ms.settings(store)["ledger_surfaces"] == ms.LEDGER_SURFACES, \
        "a broken manifest degrades to the defaults rather than taking the hook down"
    (store / ms.MANIFEST).unlink()
    ms.settings.cache_clear()
    assert ms.resolve(str(store)).is_absolute(), "a literal path resolves absolute"

    # --- excluded_files: one file ignored at both ends --------------------------
    (store / "notes").mkdir()
    (store / "notes" / "draft.md").write_text("A draft fact nobody should score against.\n")
    (store / "export.md").write_text("An exported artifact that is not memory.\n")
    (store / ms.MANIFEST).write_text(json.dumps({
        "excluded_dirs": ["old"], "index_files": ["facts.md"], "ledger_surfaces": ["LOG.md"],
        "excluded_files": ["notes/draft.md", "export*"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    ms.settings.cache_clear()
    got = ms.md_files(store)
    assert store / "notes" / "draft.md" not in got, "a relative path excludes that file"
    assert store / "export.md" not in got, "a glob excludes by basename"
    assert store / "facts.md" in got and store / "guide.md" in got, got
    assert ms.corpus_for(store / "notes" / "draft.md") is None, "the gate skips an excluded file too"
    assert ms.corpus_for(store / "facts.md") == store
    assert ms.csv_sources(store), "a registered CSV that no pattern names stays a source"
    (store / ms.MANIFEST).write_text(json.dumps({
        "excluded_files": ["*.csv"],
        "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
    ms.settings.cache_clear()
    assert ms.csv_sources(store) == [], "an excluded CSV is not a source even when registered"
    assert "RULES.csv#cost" not in ms.corpus(store)
    (store / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()

    # --- init and the registry: a manifest directory the listing can find -----
    saved = paths.PATHS; paths.PATHS = saved.replace(registry=d / "locket" / "corpora.json")
    try:
        vault = d / "vault"; (vault / "sub").mkdir(parents=True)
        vault = vault.resolve()             # the registry stores resolved paths; /var is /private/var here
        (vault / "one.md").write_text("A note in a folder nobody has registered.\n")
        assert ms.registry() == [] and vault not in ms.all_corpora().values()
        assert ms.corpus_here(vault) is None, "no manifest, no auto-memory: the default has nothing"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ms.cmd_init(vault, index=False)
        assert rc == 0 and json.loads((vault / ms.MANIFEST).read_text()) == {"name": "vault"}, \
            "a standalone manifest carries the folder's name and no default"
        assert ms.registry() == [vault] and vault in ms.all_corpora().values(), \
            "init makes the listing see a manifest directory"
        assert ms.corpus_here(vault) == vault and ms.corpus_here(vault / "sub") == vault, \
            "the cwd default finds a manifest directory from anywhere inside it"
        assert ms.corpus_for(vault / "sub" / "x.md") == vault
        assert not ms.register(vault), "registering twice is a no-op"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert ms.cmd_init(vault, index=False) == 0
        assert "(kept)" in buf.getvalue() and "already listed" in buf.getvalue(), buf.getvalue()
        # --name without --parent: a standalone label, set at init or renamed later
        named = (d / "notes-2").resolve(); named.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(named, index=False, name="field-notes") == 0
        assert ms.read_manifest(named) == {"name": "field-notes"} and ms.all_corpora().get("field-notes") == named
        with contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(vault, index=False, name="vault-renamed") == 0
        assert ms.read_manifest(vault)["name"] == "vault-renamed", "--name on a kept manifest renames it"
        clash = (d / "clash").resolve(); clash.mkdir()
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(clash, index=False, name="field-notes") == 1
        assert "is taken by" in err.getvalue() and not (clash / ms.MANIFEST).exists(), err.getvalue()
        twin = (d / "other" / "field-notes").resolve(); twin.mkdir(parents=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(twin, index=False) == 1, "a folder name another store holds is refused, not keyed by path"
        assert "--name" in err.getvalue(), err.getvalue()
        assert ms.forget(named), "the named store leaves the registry as it came"
        with contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(vault, index=False, name="vault") == 0
        with contextlib.redirect_stderr(io.StringIO()):
            assert ms.cmd_init(vault / "sub", index=False) == 1, "a directory inside a corpus is refused"
        # and so is a directory CONTAINING one, whether the inner store is listed
        # or only carries a manifest by hand; the parent is left untouched.
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert ms.cmd_init(d, index=False) == 1, "a directory containing a corpus is refused"
        assert "contains the corpus" in err.getvalue() and not (d / ms.MANIFEST).exists(), err.getvalue()
        hand = d / "hand"; hand.mkdir(); (hand / ms.MANIFEST).write_text("{}\n")
        assert ms.forget(vault)
        with contextlib.redirect_stderr(io.StringIO()):
            assert ms.cmd_init(d, index=False) == 1, "an unlisted manifest below is found too"
        assert not (d / ms.MANIFEST).exists()
        (hand / ms.MANIFEST).unlink(); hand.rmdir()
        assert ms.register(vault)
        (vault / ms.MANIFEST).write_text(json.dumps({"name": "vault-of-one"})); ms.settings.cache_clear()
        assert ms.all_corpora().get("vault-of-one") == vault, "the manifest's name is the listing key"
        assert ms.resolve("vault-of") == vault, "and a substring of it resolves"
        (d / "vault-of-one-persona").mkdir()
        (d / "vault-of-one-persona" / ms.MANIFEST).write_text(json.dumps({"name": "vault-of-one-persona"}))
        assert ms.register(d / "vault-of-one-persona"); ms.settings.cache_clear()
        assert ms.resolve("vault-of-one") == vault, "an exact name wins over a longer name containing it"
        with contextlib.redirect_stderr(io.StringIO()):
            assert ms.resolve("vault-of") is None, "a substring of both is still ambiguous"
        assert ms.forget(d / "vault-of-one-persona")
        shutil.rmtree(d / "vault-of-one-persona")
        (vault / ms.MANIFEST).unlink(); ms.settings.cache_clear()
        assert vault not in ms.all_corpora().values() and ms.registry_stale() == [vault], \
            "a registered path whose manifest is gone is reported, never enumerated"
        assert any("is gone" in l for l in stores.corpora_lines()), stores.corpora_lines()
        assert ms.forget(vault) and not ms.forget(vault) and ms.registry() == [], "forget drops the row once"
        assert not any("is gone" in l for l in stores.corpora_lines())
        with contextlib.redirect_stderr(io.StringIO()):
            assert ms.cmd_init(vault / "one.md", index=False) == 1, "a file is not a corpus"
        hostlike = (d / "hostlike").resolve(); hostlike.mkdir()
        (hostlike / "a.md").write_text("A host store's note.\n")
        paths.PATHS.host_stores["_selftest"] = hostlike        # a store listed on sight, for this block only
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert ms.cmd_init(hostlike, index=False) == 0
            assert "by convention" in buf.getvalue() and ms.registry() == [] \
                and not (hostlike / ms.MANIFEST).exists(), \
                "a host store is listed on sight, so init writes it no manifest and no row"
        finally:
            del paths.PATHS.host_stores["_selftest"]
    finally:
        paths.PATHS = saved

    # --- belongs_to: two folders in two places, one corpus ------------------
    saved = paths.PATHS; paths.PATHS = saved.replace(registry=d / "locket2" / "corpora.json")
    try:
        par = (d / "par").resolve(); (par / "docs").mkdir(parents=True)
        (par / "IMPL.md").write_text("The impl points at [[topic_one]] and [[topic_two]].\n"
                                     "See `RULES.csv` query `Area=cost` for the ruling.\n")
        (par / "docs" / "d.md").write_text("A doc under the parent's member directory.\n")
        (par / "RULES.csv").write_text("Area,Ruling\ncost,The hangar rate is quote-gated on every airframe.\n")
        (par / ms.MANIFEST).write_text(json.dumps({
            "name": "par", "members": ["docs"], "ledger_surfaces": ["LOG.md"],
            "whole_read_exempt": ["IMPL.md"],
            "sources": [{"path": "RULES.csv", "text": "Ruling", "label": "Area"}]}))
        ms.settings.cache_clear(); ms.register(par)
        kid = (d / "elsewhere" / "memory").resolve(); (kid / "archive").mkdir(parents=True)
        (kid / "topic_one.md").write_text("Topic one holds a durable fact about hangars.\n")
        (kid / "SHORT.md").write_text("- an entry\n")
        (kid / "LOG.md").write_text("2026-01-01 a ledger line\n")
        (kid / "archive" / "old.md").write_text("A superseded sentence kept as provenance here.\n")
        (kid / ms.LEGACY_MANIFEST).write_text(json.dumps(
            [{"path": str(par / "RULES.csv"), "text": "Ruling", "label": "Area"}]))
        (kid / ms.MANIFEST).write_text(json.dumps({"name": "kid", "holding_spaces": ["SHORT.md"]}))
        ms.settings.cache_clear(); ms.register(kid)
        assert {"par", "kid"} <= set(ms.all_corpora()), "before joining, both are listed"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert ms.cmd_init(kid, index=False, parent="par") == 0, buf.getvalue()
        ms.settings.cache_clear()
        assert ms.read_manifest(kid)["belongs_to"] == "par" and ms.read_manifest(kid)["name"] == "kid", \
            "init --parent adds belongs_to and keeps the rest of the manifest"
        assert "kid" not in ms.all_corpora() and ms.all_corpora()["par"] == par, \
            "a joined store is listed through its parent alone"
        assert ms.parent_of(kid) == par and ms.top_of(kid) == par and ms.top_of(par) == par
        assert ms.roots_of(par) == [par, kid]
        assert ms.corpus_for(kid / "topic_one.md") == par, "a write into the joined store is scored at the parent"
        assert ms.corpus_for(kid / "archive" / "old.md") is None, "the joined store's own exclusions hold"
        assert ms.corpus_here(kid) == par and ms.resolve(str(kid)) == par and ms.resolve("par") == par
        got = {ms.corpus_key(par, q) for q in ms.md_files(par)}
        assert got == {"IMPL.md", "docs/d.md", "kid/LOG.md", "kid/SHORT.md", "kid/topic_one.md"}, got
        assert set(ms.corpus(par)) >= {"kid/topic_one.md", "RULES.csv#cost"}, "keys are composite; one CSV, once"
        assert len(ms.csv_sources(par)) == 1, "a CSV registered by both stores is one source"
        st = ms.settings(par)
        assert "SHORT.md" in st["holding_spaces"] and "LOG.md" in st["ledger_surfaces"] \
            and "SHORT.md" in st["append_surfaces"], "name-sets union upward"
        assert "SHORT.md" not in ms.settings(kid)["ledger_surfaces"], "and never downward"
        assert st["whole_read_exempt"] == {"IMPL.md"} and ms.settings(kid)["whole_read_exempt"] == set(), \
            "the size-sweep exemption is a name-set like the others: read, unioned upward, never down"
        assert not hygiene.dangling(par).get("topic_one") and hygiene.dangling(par).get("topic_two"), \
            "a wikilink into the joined store resolves; a missing one still dangles"
        (kid / "HISTORY-Kid-Sessions.csv").write_text("date,title,summary,area,repo,commit_hash,issue,pr\n")
        assert ms.session_ledger(par) == kid / "HISTORY-Kid-Sessions.csv" == ms.session_ledger(kid), \
            "a joined store's session ledger is the store's, found from either folder"
        (kid / "HISTORY-Kid-Sessions.csv").unlink()
        (par / "feedback_named_one.md").write_text(
            "---\nname: named-one\ndescription: x\n---\n\nA body line with name: not-this.\n"
            "Links to [[named-one]] and [[not-this]].\n")
        dl = hygiene.dangling(par)
        assert "named-one" not in dl and "not-this" in dl, \
            "a link resolves by the frontmatter name, never by a `name:` in the body"
        (par / "feedback_named_one.md").unlink()
        assert any("superseded" in " ".join(ts) for _, _, ts in ms.archived_claims(par).items), \
            "provenance is read from the joined store's archive too"
        assert any(s[4] == "ok" for s in hygiene.csv_pointers(par)), hygiene.csv_pointers(par)
        assert any("[1 joined store]" in l for l in stores.corpora_lines()), stores.corpora_lines()
        # refusals: unknown parent, self, a store nesting on disk, a cycle
        for bad in ("nobody", "kid"):
            with contextlib.redirect_stderr(io.StringIO()):
                assert ms.cmd_init(kid, index=False, parent=bad) == 1, bad
        sub = par / "docs"; (sub / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()
        with contextlib.redirect_stderr(io.StringIO()):
            assert ms.cmd_init(sub, index=False, parent="par") == 1, "a nested directory is a member, not a join"
        (sub / ms.MANIFEST).unlink(); ms.settings.cache_clear()
        (par / ms.MANIFEST).write_text(json.dumps({"name": "par", "belongs_to": "kid"}))
        ms.settings.cache_clear()
        assert ms.parent_of(par) is None and ms.parent_of(kid) is None, "a cycle leaves both standing alone"
        assert {"par", "kid"} <= set(ms.all_corpora())
        (par / ms.MANIFEST).write_text(json.dumps({"name": "par"})); ms.settings.cache_clear()
        assert ms.parent_of(kid) == par
        # nested on disk but outside the parent's members: a host home's own
        # auto-memory under projects/ is the case, and it joins
        (par / ms.MANIFEST).write_text(json.dumps({"name": "par", "members": ["docs"]})); ms.settings.cache_clear()
        out = par / "projects" / "m"; out.mkdir(parents=True)
        (out / "o.md").write_text("A nested store outside the parent's members joins it.\n")
        (out / ms.MANIFEST).write_text(json.dumps({"name": "outm"})); ms.settings.cache_clear()
        ms.register(out)                          # a real auto-memory is listed on sight
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(out, index=False, parent="par") == 0, "a store outside the members joins"
        assert ms.parent_of(out) == par and ms.corpus_for(out / "o.md") == par
        assert [q.name for q in ms.md_files(par)].count("o.md") == 1, "counted once, as a joined store"
        # a joining folder called like one of the parent's directories needs a label
        am = (d / "host" / "projects" / "slug" / "memory").resolve(); am.mkdir(parents=True)
        (am / "a.md").write_text("An auto-memory fact about the council store here.\n")
        (am / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear(); ms.register(am)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(am, index=False, parent="par", name="docs") == 1, "a label naming a parent dir"
            assert ms.cmd_init(am, index=False, parent="par", name="a/b") == 1, "a label with a slash"
        assert "--name <label>" in err.getvalue(), err.getvalue()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert ms.cmd_init(am, index=False, parent="par", name="auto-memory") == 0
        assert ms.read_manifest(am) == {"name": "auto-memory", "belongs_to": "par"}, ms.read_manifest(am)
        assert ms.corpus_key(par, am / "a.md") == "auto-memory/a.md"
        assert ms.forget(am)
        assert ms.forget(out)
        assert ms.forget(par) and ms.forget(kid)
    finally:
        paths.PATHS = saved
        ms.settings.cache_clear()

    # --- the manifest schema: hints for the hand that writes one ---------------
    sd = (d / "schemadir").resolve(); sd.mkdir()
    assert ms.validate_manifest(sd) == [], "no manifest, nothing to say"
    (sd / ms.MANIFEST).write_text(json.dumps({
        "name": "s", "members": ["docs"], "belongs_to": "x",
        "sources": [{"path": "R.csv", "text": "Ruling", "label": "Area"}],
        "ledgers": [{"path": "L.csv", "caps": {"note": 10}, "enum": {"k": ["a", ""]}}]}))
    assert ms.validate_manifest(sd) == [], ms.validate_manifest(sd)
    (sd / ms.MANIFEST).write_text(json.dumps({
        "belongs-to": "x", "members": "docs", "sources": [{"text": "Ruling"}],
        "ledgers": [{"path": "L.csv", "caps": {"note": "ten"}, "extra": 1}]}))
    f = ms.validate_manifest(sd)
    assert any("unknown key 'belongs-to', did you mean 'belongs_to'?" in x for x in f), f
    assert any("members: expected array" in x for x in f), f
    assert any("sources[0]: missing required key 'path'" in x for x in f), f
    assert any("ledgers[0].caps.note: expected integer" in x for x in f), f
    assert any("ledgers[0]: unknown key 'extra'" in x for x in f), f
    (sd / ms.MANIFEST).write_text("{not json")
    assert ms.validate_manifest(sd) and "does not parse" in ms.validate_manifest(sd)[0]
    (sd / ms.MANIFEST).unlink()
    assert json.loads(json.dumps(ms.MANIFEST_SCHEMA))["properties"].keys() >= {
        "belongs_to", "ledgers", "sources", "members"}, "the schema names the keys the reader takes"

    # --- ledger schemas: the one decidable refusal ---------------------------
    led = (d / "proj").resolve(); led.mkdir()
    hist = led / "HISTORY-Proj-Sessions.csv"
    hdr = "date,title,summary,area,repo,commit_hash,issue,pr\n"
    good = '2026-09-20,Composite corpus,"Two stores joined into one, links resolved.",api,grp/sub/proj,5ab427a,ENG-42,#155\n'
    hist.write_text(hdr + good)
    assert ms.ledger_schema_for(hist) == (None, None), "no schema governs a CSV outside every store"
    (led / ms.MANIFEST).write_text("{}\n")
    sch, glob = ms.ledger_schema_for(hist)
    assert sch is not None and glob == "HISTORY-*-Sessions.csv", "the built-in governs the file inside a store"
    assert ms.ledger_schema_for(led / "notes.csv") == (None, None)
    assert ms.ledger_decision(str(hist), {"content": hdr + good}) == (None, False), "a clean whole-file write passes"
    assert ms.ledger_decision(str(hist), {"content": hdr}) == (None, False), "a header alone passes"
    long_row = f'2026-09-20,{"t" * 91},{"s" * 241},ios,a b,zz,two words,155\n'
    msg, blocking = ms.ledger_decision(str(hist), {"content": hdr + good + long_row})
    assert blocking and "row 3: title is 91 characters, cap 90" in msg \
        and "summary is 241 characters, cap 240" in msg and "commit_hash is 'zz'" in msg and "issue is 'two words'" in msg and "pr is '155'" in msg \
        and "repo is 'a b'" in msg, msg
    mr = good.replace("#155", "!155")
    assert ms.ledger_decision(str(hist), {"content": hdr + mr}) == (None, False), "a GitLab MR is a pr"
    msg, blocking = ms.ledger_decision(str(hist), {"content": "date,title\n2026-01-01,x\n"})
    assert blocking and "header is date,title" in msg, "a header off the schema is one finding"
    msg, blocking = ms.ledger_decision(str(hist), {"old_string": "x", "new_string": long_row})
    assert blocking and "row 1: title is 91" in msg, "an Edit fragment is checked against the header on disk"
    assert ms.ledger_decision(str(hist), {"old_string": "x", "new_string": "Composite"}) == (None, False), \
        "a fragment that is not a row is not judged"
    assert ms.ledger_decision(str(hist), {"old_string": "x", "new_string": ",,,,,,,\n"}) == (None, False), \
        "an empty row is a blank line"
    msg, blocking = ms.ledger_decision(str(hist), {"old_string": "x", "new_string": "2026-09-20,,ok,,,,,\n"})
    assert blocking and "title is required" in msg, msg
    for dt in ("2026-09-23T03:12Z", "2026-09-23T03:12:45Z"):
        assert ms.ledger_decision(str(hist), {"old_string": "x", "new_string": f"{dt},t,s,,,,,\n"}) == (None, False), \
            f"a UTC instant is a date: {dt}"
    for dt in ("2026-09-23T03:12", "2026-09-23T03:12:45-07:00", "2026-09-23 03:12Z"):
        msg, blocking = ms.ledger_decision(str(hist), {"old_string": "x", "new_string": f"{dt},t,s,,,,,\n"})
        assert blocking and "date" in msg, f"only a Z-suffixed instant passes: {dt}"
    # bashguard: heredoc rows and echo rows are read; an opaque append warns
    msg, blocking = ms.ledger_bash_decision(f"cat >> {hist} <<'EOF'\n{long_row}EOF\n", None)
    assert blocking and "bashguard" in msg and "cap 90" in msg, msg
    msg, blocking = ms.ledger_bash_decision(f'echo "{good.strip()}" >> {hist}', None)
    assert (msg, blocking) == (None, False), msg
    msg, blocking = ms.ledger_bash_decision(f"python3 gen.py >> {hist}", None)
    assert msg and not blocking and "cannot read" in msg, msg
    assert ms.ledger_bash_decision(f"cat {hist}", None) == (None, False), "a read is nothing"
    assert ms.ledger_bash_decision("echo hi >> other.csv", str(led)) == (None, False)
    # a body counts only against the ledger its own pipeline writes
    msg, blocking = ms.ledger_bash_decision(f"printf '{good.strip()}\\n' >> {hist} && echo '# t' > a.md", None)
    assert (msg, blocking) == (None, False), f"a second redirect is not the ledger's content: {msg}"
    msg, blocking = ms.ledger_bash_decision(f"printf 'not-a-date,t,s,,,,,\\n' | tee -a {hist}; echo '# t' > a.md", None)
    assert blocking and "date" in msg and "header" not in msg, msg
    msg, blocking = ms.ledger_bash_decision(f"echo '# t' > a.md; python3 gen.py >> {hist}", None)
    assert msg and not blocking and "cannot read" in msg, "an unrelated echo does not vouch for an opaque append"
    msg, blocking = ms.ledger_bash_decision(f"cat <<'EOF' > {hist}\n{long_row}EOF\n", None)
    assert blocking and "cap 90" in msg, "a redirect after the heredoc opener is the same pipeline"
    hist2 = led / "HISTORY-Other-Sessions.csv"; hist2.write_text(hdr)
    msg, blocking = ms.ledger_bash_decision(
        f"python3 gen.py >> {hist} && printf 'not-a-date,t,s,,,,,\\n' >> {hist2}", None)
    assert blocking and "date" in msg, "an opaque write first never excuses a readable bad row after it"
    msg, blocking = ms.ledger_bash_decision(f"cat <<'EOF' \\\n  >> {hist}\n{long_row}EOF\n", None)
    assert blocking and "cap 90" in msg, "a redirect on a continued opener line is the same pipeline"
    for before in ("cat <<<x\n", "echo 'a <<b'\n", "cat <<'E-F'\nx\nE-F\n"):
        msg, blocking = ms.ledger_bash_decision(f"{before}cat >> {hist} <<'EOF'\n{long_row}EOF\n", None)
        assert blocking and "cap 90" in msg, f"the ledger heredoc after {before!r} is read: {msg}"
    hist2.unlink()
    # a manifest overrides the built-in for its store, and governs another CSV
    (led / ms.MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": "HISTORY-*-Sessions.csv", "caps": {"summary": 20}},
        {"path": "LEDGER-Closures.csv", "required": ["slug"], "caps": {"note": 10}}]}))
    ms.settings.cache_clear()
    msg, blocking = ms.ledger_decision(str(hist), {"content": hdr + good})
    assert blocking and "characters, cap 20" in msg and ms.MANIFEST in msg, msg
    clos = led / "LEDGER-Closures.csv"; clos.write_text("slug,note\n")
    msg, blocking = ms.ledger_decision(str(clos), {"old_string": "x", "new_string": ",a very long note here\n"})
    assert blocking and "slug is required" in msg and "note is 21 characters" in msg, msg
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = _shown(hygiene.cmd_ledgers(led))
    assert n == 1 and "2 ledger(s) checked, 1 finding(s)" in buf.getvalue(), buf.getvalue()
    (led / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()
    assert "@@" not in help_text.usage() and "caps title ≤90" in help_text.usage(), "the help renders the built-ins from the constant"
    # an entry naming a built-in's glob MERGES onto it: only the keys it names change
    (led / ms.MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": "HISTORY-*-Sessions.csv", "enum": {"area": ["docs", ""]}}]}))
    ms.settings.cache_clear()
    ok = '2026-09-20,Docs pass,"A row under the store\'s own areas.",docs,,,,\n'
    assert ms.ledger_decision(str(hist), {"content": hdr + ok}) == (None, False), "a store's own area passes"
    msg, blocking = ms.ledger_decision(str(hist), {"content": hdr + ok.replace(",docs,", ",ui,")})
    assert blocking and "area is 'ui'" in msg, "the store's enum replaces the built-in's for that column"
    msg, blocking = ms.ledger_decision(str(hist), {"content": hdr + ok.replace("2026-09-20", "20 Sep")})
    assert blocking and "date" in msg, "the built-in date pattern survives an enum override"
    msg, blocking = ms.ledger_decision(str(hist), {"content": hdr + f'2026-09-20,{"t" * 91},s,docs,,,,\n'})
    assert blocking and "cap 90" in msg, "the built-in caps survive an enum override"
    (led / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()
    # `extends`: a file named outside the glob inherits a built-in by declaration
    odd = led / "HISTORY-session-summary.csv"
    assert ms.ledger_schema_for(odd) == (None, None), "outside the glob, nothing governs it"
    (led / ms.MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": odd.name, "extends": "HISTORY-*-Sessions.csv", "enum": {"area": ["docs", ""]}}]}))
    ms.settings.cache_clear()
    assert ms.validate_manifest(led) == [], ms.validate_manifest(led)
    assert ms.ledger_decision(str(odd), {"content": hdr + ok}) == (None, False), "extends inherits the built-in"
    msg, blocking = ms.ledger_decision(str(odd), {"content": hdr + ok.replace("2026-09-20", "20 Sep")})
    assert blocking and "date" in msg, "the inherited date pattern holds"
    (led / ms.MANIFEST).write_text(json.dumps({"ledgers": [{"path": odd.name, "extends": "nope"}]}))
    ms.settings.cache_clear()
    assert any("is no built-in" in f for f in ms.validate_manifest(led)), ms.validate_manifest(led)
    (led / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()
    # an entry naming one file inside the built-in's glob governs the sweep alone
    (led / ms.MANIFEST).write_text(json.dumps({"ledgers": [
        {"path": hist.name, "enum": {"area": ["docs", ""]}}]}))
    ms.settings.cache_clear(); hist.write_text(hdr + ok)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = _shown(hygiene.cmd_ledgers(led))
    assert n == 0 and "1 ledger(s) checked, 0 finding(s)" in buf.getvalue(), buf.getvalue()
    (led / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear(); hist.write_text(hdr + good)
    # rulings and claims: one glob covers the single-store and the project name
    rh, ch = "Ruling,Category,Area,Date\n", "Claim,Category,Area,Source,Date\n"
    rul = '"Pro unlocks by a one-time purchase, never a subscription.",product,pricing,2026-09-26\n'
    for name in ("RULINGS.csv", "RULINGS-Proj.csv"):
        assert ms.ledger_schema_for(led / name)[1] == "RULINGS*.csv", name
        assert ms.ledger_decision(str(led / name), {"content": rh + rul}) == (None, False), name
    assert ms.ledger_schema_for(led / "CLAIMS-Proj.csv")[1] == "CLAIMS*.csv"
    assert ms.ledger_schema_for(led / "REFERENCE-Proj-Rulings.csv") == (None, None), \
        "a name outside the glob is governed only through `extends`"
    rp = led / "RULINGS-Proj.csv"; rp.write_text(rh)
    assert ms.ledger_decision(str(rp), {"old_string": "x", "new_string": rul.replace(",pricing,", ",pricing|release,")}) \
        == (None, False), "a pipe-joined Area is several keys"
    msg, blocking = ms.ledger_decision(str(rp), {"old_string": "x", "new_string": rul.replace(",pricing,", ",Pricing,")})
    assert blocking and "Area is 'Pricing'" in msg, "an Area is a lowercase key, because pointers query it"
    msg, blocking = ms.ledger_decision(str(rp), {"old_string": "x", "new_string": rul.replace(",2026-09-26", ",")})
    assert blocking and "Date is required" in msg, msg
    cp = led / "CLAIMS-Proj.csv"; cp.write_text(ch)
    msg, blocking = ms.ledger_decision(str(cp), {"old_string": "x", "new_string": '"An empty product list means the agreement is unsigned.",code,iap,,2026-09-26\n'})
    assert blocking and "Source is required" in msg, "a claim without a source is a finding"
    rp.unlink(); cp.unlink()
    # the size reading: a declared CSV past the limit is a finding, never split or capped
    assert ms.large_csvs(led) == [], "a small ledger reported past 5 MB"
    assert [p.name for p, _n in ms.large_csvs(led, limit=1)] == [hist.name], ms.large_csvs(led, limit=1)
    saved, ledger.CSV_REPORT_BYTES = ledger.CSV_REPORT_BYTES, 1
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = _shown(hygiene.cmd_ledgers(led))
        assert "declared CSV(s) past the" in buf.getvalue() and "never split it" in buf.getvalue(), buf.getvalue()
        assert n >= 1, "a ledger past the size reading counted as clean"
    finally:
        ledger.CSV_REPORT_BYTES = saved
    # the sweep walks the store: a ledger under a subdirectory is checked, one
    # under an excluded directory is not
    (led / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()
    (led / "memory").mkdir(exist_ok=True)
    (led / "memory" / "RULINGS.csv").write_text(rh + rul.replace(",pricing,", ",Pricing,"))
    (led / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (led / "node_modules" / "pkg" / "README.md").write_text("A dependency's readme is never prose of this store.\n")
    assert not any("node_modules" in str(p) for p in ms.md_files(led)), "a dependency tree is not walked"
    (led / ms.MANIFEST).write_text(json.dumps({"excluded_dirs": ["drafts"]})); ms.settings.cache_clear()
    assert not any("node_modules" in str(p) for p in ms.md_files(led)), "a store's own exclusions do not re-admit it"
    (led / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()
    (led / "archive").mkdir(exist_ok=True)
    (led / "archive" / "RULINGS.csv").write_text(rh + rul.replace(",pricing,", ",Pricing,"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = _shown(hygiene.cmd_ledgers(led))
    assert n == 1 and "memory/RULINGS.csv" in buf.getvalue() and "archive/" not in buf.getvalue(), buf.getvalue()
    (led / "memory" / "RULINGS.csv").unlink(); (led / "archive" / "RULINGS.csv").unlink()
    (led / ms.MANIFEST).write_text("{}\n"); ms.settings.cache_clear()

    # --- host adapters: Hermes speaks the same payload in its own vocabulary ---
    h = ms.normalise({"hook_event_name": "pre_tool_call", "tool_name": "write_file",
                   "tool_input": {"path": "/x/MEMORY.md", "content": "c"},
                   "session_id": "s1"})
    assert h["_host"] == "hermes" and h["tool_name"] == "Write", h
    assert h["tool_input"]["file_path"] == "/x/MEMORY.md", h
    h = ms.normalise({"hook_event_name": "pre_tool_call", "tool_name": "terminal",
                   "args": {"command": "ls"}})
    assert h["tool_name"] == "Bash" and h["tool_input"]["command"] == "ls", \
        "Hermes may carry the arguments under `args`"
    c = ms.normalise({"hook_event_name": "PreToolUse", "tool_name": "Edit",
                   "tool_input": {"file_path": "/x/a.md", "new_string": "n"}})
    assert c["_host"] == "claude-code" and c["tool_name"] == "Edit", c
    m = ms.normalise({"hook_event_name": "pre_tool_call", "tool_name": "memory",
                   "tool_input": {"action": "add", "target": "user", "content": "likes tea"}})
    assert m["tool_name"] == "Edit" and m["tool_input"]["file_path"].endswith("memories/USER.md"), m
    assert m["tool_input"]["content"] == "likes tea" and m["tool_input"]["_append"], m
    assert ms.added_text({}, "old body\nlikes tea", "old body\n") == " \nlikes tea", \
        "an append reaches the scorer as the entry alone"
    m = ms.normalise({"hook_event_name": "pre_tool_call", "tool_name": "memory",
                   "tool_input": {"operations": [
                       {"action": "replace", "target": "memory", "old_text": "a", "content": "b"},
                       {"action": "remove", "target": "memory", "old_text": "z"},
                       {"action": "add", "target": "memory", "new_text": "c"}]}})
    assert m["tool_input"]["file_path"].endswith("memories/MEMORY.md"), m
    assert m["tool_input"]["new_string"] == "b\nc" and m["tool_input"]["old_string"] == "a\nz", m
    assert "MEMORY.md" not in ms.settings(hosts.hermes())["index_files"], \
        "Hermes's MEMORY.md is its whole memory and must not inherit the index exemption"
    r = ms.normalise({"hook_event_name": "pre_tool_call", "tool_name": "write_file",
                   "tool_input": {"path": "MEMORY.md", "content": "c"},
                   "cwd": str(hosts.claude() / "memory")})
    assert r["tool_input"]["file_path"] == str(hosts.claude() / "memory" / "MEMORY.md"), \
        "a relative path resolves against the payload cwd, or the write escapes the gate"
    assert ms.session_asked({"_host": "hermes"}), "no session source at all fails open"

    # a Hermes nudge refuses the call once, and the repeat passes once state.db shows it
    saved = paths.PATHS; paths.PATHS = saved.replace(pending_dir=d / "pending")
    try:
        import contextlib, io, sqlite3
        db = d / "shown.db"
        con = sqlite3.connect(db)
        con.execute("create table messages (session_id text, role text, content text, timestamp real)")
        con.commit()
        store = lambda text: (con.execute("insert into messages values ('s1','tool',?,?)",
                                          (json.dumps({"error": text}), time.time())), con.commit())
        call = lambda cmd: {"_host": "hermes", "session_id": "s1", "tool_name": "Bash",
                            "tool_input": {"command": cmd}}
        first = ms.refuse_once(call("a"), "nudge A", "hook", db)
        assert first and first.startswith(ms.ONCE_NOTE) and first.endswith("nudge A"), first
        assert ms.refuse_once(call("a"), "nudge A", "hook", db) == first, "an unseen refusal passed"
        assert ms.refuse_once(call("a"), "nudge A", "hook", db) is None, "the refusal cap did not hold"
        assert ms.refuse_once(call("a"), "nudge A", "hook", db) == first, "a passed call stayed passed"
        store(first)
        assert ms.refuse_once(call("a"), "nudge A", "hook", db) is None, "a delivered refusal refused again"
        # two modes refuse one call; Hermes shows only the first block
        b_hook = ms.refuse_once(call("b"), "from hook", "hook", db)
        b_trig = ms.refuse_once(call("b"), "from trigger", "trigger", db)
        assert b_hook and b_trig, "a second mode's refusal was swallowed"
        store(b_hook)
        assert ms.refuse_once(call("b"), "from hook", "hook", db) is None, "the shown mode refused again"
        assert ms.refuse_once(call("b"), "from trigger", "trigger", db) == b_trig, \
            "a mode whose message Hermes never showed let the retry pass"
        assert ms.refuse_once(call("f"), "m1", "hook", db) and ms.refuse_once(call("f"), "m2", "trigger", db) and \
            ms.refuse_once(call("f"), "m2", "trigger", db), "one mode's refusal counted against another's cap"
        assert ms.refuse_once(dict(call("c"), session_id="s2"), "n", "hook", db) and \
            ms.refuse_once(dict(call("c"), session_id="s2"), "n", "hook", db), "another session's result counted"
        nodb = d / "none.db"
        assert ms.refuse_once(call("d"), "n", "hook", nodb) and ms.refuse_once(call("d"), "n", "hook", nodb) and \
            ms.refuse_once(call("d"), "n", "hook", nodb) is None, "a state.db that cannot say must fall to the cap"
        con.close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = ms.emit(call("e"), ms.LEAD_WARNING + " more", False, db=db)
        sent = json.loads(out.getvalue())
        assert rc == 2 and sent["action"] == "block", "a Hermes nudge did not refuse the call"
        assert ms.LEAD_BLOCKING in sent["message"] and ms.LEAD_WARNING not in sent["message"], \
            "a refused write was described as landed"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert ms.emit({"_host": "claude-code"}, "advice", False) == 0
        assert "additionalContext" in out.getvalue(), "Claude Code lost its non-blocking channel"
    finally:
        paths.PATHS = saved

    # the 2e gate reads Hermes's state.db the way it reads Claude Code's transcript
    import sqlite3
    db = d / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (session_id text, role text, tool_calls text)")
    call = lambda cmd: json.dumps([{"function": {"name": "terminal",
                                                 "arguments": json.dumps({"command": cmd})}}])
    con.execute("insert into messages values ('asked','assistant',?)",
                (call("python3 ~/.claude/scripts/memfind.py 'q' council"),))
    con.execute("insert into messages values ('named','assistant',?)",
                (call("grep -n memfind.py memscan.py"),))
    con.commit(); con.close()
    assert hermes.memfind_asked_db(db, "asked"), "a terminal call that RUNS memfind discharges"
    assert not hermes.memfind_asked_db(db, "named"), "one that only names it does not"
    assert not hermes.memfind_asked_db(db, "silent"), "a session with no such call has not asked"
    assert hermes.memfind_asked_db(d / "missing.db", "asked"), "no database fails open"

    # --- Codex: Claude Code's payload, a patch for every edit, a rollout transcript ---
    rollout = str(d / "rollout-2026-09-25T18-00-40-x.jsonl")
    codex = lambda tool, ti, **kw: ms.normalise({"hook_event_name": "PreToolUse", "tool_name": tool,
                                              "tool_input": ti, "turn_id": "t1", "cwd": str(COUNCIL),
                                              "transcript_path": rollout, **kw})
    add = "*** Begin Patch\n*** Add File: notes.md\n+first line\n+second line\n*** End Patch"
    upd = ("*** Begin Patch\n*** Update File: /w/a.py\n@@\n-x = 1\n+x = 2\n"
           "*** Update File: MEMORY.md\n@@\n old\n-gone\n+new fact\n*** End Patch")
    w = codex("apply_patch", {"command": add})
    assert w["_host"] == "codex" and w["tool_name"] == "Write", w
    assert w["tool_input"] == {"file_path": str(COUNCIL / "notes.md"),
                               "content": "first line\nsecond line"}, w
    e = codex("Bash", {"command": f"apply_patch <<'PATCH'\n{upd}\nPATCH"})
    assert e["tool_name"] == "Edit", "a patch sent through the shell (no native tool) went unseen"
    assert e["tool_input"]["file_path"] == str(COUNCIL / "MEMORY.md"), \
        "the file inside a store is the one judged, whatever its place in the patch"
    assert e["tool_input"]["new_string"] == "new fact" and e["tool_input"]["old_string"] == "gone", e
    assert ms.parse_patch(add.replace("\n", "\r\n")) == [("add", "notes.md", "first line\nsecond line", "")], \
        "a CRLF patch parsed to nothing, which passes the gate silently"
    both = "*** Begin Patch\n*** Add File: MEMORY.md\n+m\n*** Add File: /w/l.csv\n+r\n*** End Patch"
    assert codex("apply_patch", {"command": both})["tool_input"]["file_path"] == "/w/l.csv", \
        "a ledger in the patch was not the file judged"
    assert ms.parse_patch("*** Begin Patch\n*** Update File: a.md\n*** Move to: b.md\n+x\n*** End Patch") \
        == [("update", "b.md", "x", "")], "a move is judged at its destination"
    assert codex("Bash", {"command": "echo apply_patch"})["tool_name"] == "Bash", \
        "a shell call that only names apply_patch stays a shell call"
    assert codex("apply_patch", {"command": "*** Begin Patch\n*** Delete File: a.md\n*** End Patch"}
                 )["tool_name"] == "apply_patch", "a patch that adds nothing stays opaque"
    assert ms.normalise({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {
        "command": f"apply_patch <<'P'\n{add}\nP"}})["tool_name"] == "Bash", \
        "Claude Code's shell is never read as a patch"
    assert ms.host_of({"hook_event_name": "PreToolUse", "transcript_path": rollout}) == "codex"
    import contextlib, io
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = ms.emit({"_host": "codex"}, "refused", True)
    deny = json.loads(out.getvalue())["hookSpecificOutput"]
    assert rc == 0 and deny["permissionDecision"] == "deny" and deny["permissionDecisionReason"] == "refused", \
        "a Codex block must be the deny shape Codex reads"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert ms.emit({"_host": "codex"}, "advice", False) == 0
    assert json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] == "advice"
    # the rollout: a call is a function_call whose `arguments` is a JSON string
    fc = lambda args: json.dumps({"type": "response_item", "payload": {
        "type": "function_call", "name": "exec_command", "arguments": json.dumps(args)}})
    out_line = json.dumps({"type": "response_item", "payload": {
        "type": "function_call_output", "output": "locket find 'q' council"}})
    asked, named, argv = (d / f"rollout-{n}.jsonl" for n in ("asked", "named", "argv"))
    asked.write_text(fc({"cmd": "locket find 'a fact' council", "workdir": "/w"}) + "\n")
    named.write_text(fc({"cmd": "grep -n locket README.md"}) + "\n" + out_line + "\n")
    argv.write_text(fc({"command": ["bash", "-lc", "locket find 'a fact' council"]}) + "\n")
    assert ms.memfind_asked(str(asked)), "a Codex shell call that RUNS locket find discharges"
    assert not ms.memfind_asked(str(named)), "a call that names it, or a result quoting it, does not"
    assert ms.memfind_asked(str(argv)), "the older shell tool's argv form is read through bash -lc"
    shutil.rmtree(d, ignore_errors=True)

    paths.PATHS.host_stores["council"] = saved_council
    ms.settings.cache_clear()
    shutil.rmtree(fixture, ignore_errors=True)
def hook_payload_from(docs, text, top=3):
    """selftest seam: the FIRE_FLOOR gate against an in-memory corpus."""
    ranked = [r for r in ms.score_against(text, docs)[:top] if r[1] >= ms.FIRE_FLOOR]
    return None if not ranked else "hit"


class MemscanBehaviour(unittest.TestCase):
    def test_behaviour(self):
        with fixture_env():
            _selftest_inner()

    def test_memfind_loads_no_memscan(self):
        """memfind reads the layers below memscan and never loads it, so a script run of
        memscan, which is `__main__`, cannot meet a second copy with caches of its own."""
        import subprocess
        prog = (f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
                "import memfind\nprint('memscan' in sys.modules)\n")
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "False", r.stderr)


if __name__ == "__main__":
    unittest.main()
