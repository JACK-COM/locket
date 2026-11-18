"""gate: what a write adds, how the decision scores it against the store, and the 2e
question gate, which reads the archive only once the question is known to be unasked."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.harness import DOCS, StoreCase, fixture_env

import claims  # noqa: E402
import corpus  # noqa: E402
import gate  # noqa: E402
import ledger  # noqa: E402
import manifest  # noqa: E402

NOVEL = ("Cold-soaked turbines need a longer motoring cycle before light-off on the "
         "northern routes, because the oil viscosity at the scavenge pump stalls the "
         "starter below minus thirty. ")
UNASKED = {"_host": "mcp", "_asked": False}
ASKED = {"_host": "mcp", "_asked": True}


# `hook_decision` returns (message, blocking); most cases below assert on the message, and
# the blocking branch has its own case.
def hd(*a, **k):
    return gate.hook_decision(*a, **k)[0]


PRIOR = "The hangar rate structure is quote-gated on this airframe entirely.\n"
NOVEL_RESERVE = ("Turbine hot-section inspections fall due on calendar time as well as hours, "
                 "and an owner flying below the annual utilisation assumption pays the "
                 "reserve twice over the life of the engine program regardless. " * 2)
RESTATED = ("The default execution path is the dossier pipeline run interactively "
            "for every one of these. " * 4)
HELD = ("The incubator is holding the observation that autonomous cycles skip "
        "reconsolidation whenever the surface is already large. ")
STEM = "Promotion target if this holds across further instances: "
BODY = HELD + STEM + "Ramp availability shifts seasonally with transient traffic volumes. "


class ArchiveReads(unittest.TestCase):
    def setUp(self):
        env = fixture_env()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / manifest.MANIFEST).write_text(json.dumps({"index_files": [], "ledger_surfaces": []}))
        (self.root / "facts.md").write_text("Silhouette glyphs scale by wingspan across the comparison cards.\n")
        self.reads = 0
        real = corpus.archived_claims

        def counted(root):
            self.reads += 1
            return real(root)
        patch = mock.patch.object(corpus, "archived_claims", counted)
        patch.start()
        self.addCleanup(patch.stop)

    def test_one_write_reads_the_archive_once(self):
        d = gate.evaluate_write(str(self.root / "new.md"), {"content": NOVEL * 2}, payload=UNASKED)
        self.assertTrue(d.blocking and d.checked == "store", d)
        self.assertEqual(self.reads, 1)

    def test_an_asked_session_never_reads_it_for_the_2e_gate(self):
        msg = gate.memfind_gate(ASKED, "new.md", {}, NOVEL * 2, "", self.root)
        self.assertIsNone(msg)
        self.assertEqual(self.reads, 0)

    def test_a_short_write_reads_nothing(self):
        d = gate.evaluate_write(str(self.root / "new.md"), {"content": NOVEL[:120]}, payload=UNASKED)
        self.assertEqual((d.message, d.blocking, self.reads), (None, False, 0))


class Decisions(unittest.TestCase):
    def setUp(self):
        env = fixture_env()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / manifest.MANIFEST).write_text(json.dumps({"index_files": [], "ledger_surfaces": []}))

    def test_a_check_that_raises_lets_the_write_through(self):
        d = gate.evaluate_write(str(self.root / "a\0b.md"), {"content": NOVEL * 2}, payload=UNASKED)
        self.assertEqual(d, gate.Decision(None, False, "error"))

    def test_an_autonomous_write_is_never_asked_the_question(self):
        d = gate.evaluate_write(str(self.root / "new.md"), {"content": NOVEL * 2}, autonomous=True,
                                payload=UNASKED)
        self.assertEqual((d.message, d.blocking), (None, False))

    def test_an_edit_in_a_subdirectory_never_stands_in_for_a_root_file_of_its_name(self):
        (self.root / "MEMORY.md").write_text(NOVEL * 2)
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "MEMORY.md").write_text("A line about the hangar door schedule, kept short.\n")
        d = gate.evaluate_write(str(sub / "MEMORY.md"), {"old_string": "A line about the hangar door",
                                                         "new_string": NOVEL * 3}, payload=ASKED)
        self.assertTrue(d.message and "MEMORY.md" in d.message, d)

    def test_an_append_is_checked_against_the_ledger_header(self):
        hist = self.root / "HISTORY-Proj-Sessions.csv"
        hist.write_text("date,title,summary,area,repo,commit_hash,issue,pr\n2026-09-20,Ok,Fine.,api,x,5ab427a,,\n")
        good = gate.evaluate_write(str(hist), {"content": "2026-09-21,Two,Fine.,api,x,6bc5380,,\n"}, append=True)
        self.assertEqual((good.blocking, good.checked), (False, "ledger"), good)
        bad = gate.evaluate_write(str(hist), {"content": "not-a-date,Two,Fine.,api,x,6bc5380,,\n"}, append=True)
        self.assertTrue(bad.blocking and "date" in bad.message, bad)


class AddedText(unittest.TestCase):
    def test_only_added_material_is_scored(self):
        big = "x" * 900
        assert gate.added_text({}, big, "") == big                                     # new or empty file
        assert gate.added_text({"old_string": "y" * 800}, big, "y" * 800) is None       # in-place rewrite
        assert gate.added_text({"old_string": "y" * 20}, big, "z") == big               # inserted block
        # an Edit that inserts BETWEEN two anchor lines: both anchors must be subtracted
        anchored = gate.added_text({"old_string": "TOP LINE\nBOTTOM LINE"},
                              "TOP LINE\n" + big + "\nBOTTOM LINE", "z")
        assert "TOP LINE" not in anchored and "BOTTOM LINE" not in anchored, anchored
        # a whole-file Write: an append LEAVES THE OLD BODY INTACT, a reconsolidation
        # does not. This is the dominant append shape at session close.
        prior = "The hangar rate structure is quote-gated on this airframe entirely.\n"
        assert gate.added_text({}, prior + big, prior).strip() == big                   # appended
        assert gate.added_text({}, "wholly rewritten body " + big, prior) is None       # reconsolidated


class HookDecision(unittest.TestCase):
    def setUp(self):
        self.ctx = (DOCS, PRIOR, NOVEL_RESERVE, RESTATED, dict(DOCS, t=PRIOR), HELD, STEM, BODY)

    def test_an_append_is_scored_by_its_delta(self):
        docs, prior, novel, restated, docs3, held, stem, body = self.ctx
        # ★ THE HARD CASE, and the reason the containment test alone is not the fix:
        # the prior body is IN the text and is already in the corpus, so scoring the
        # whole text self-matches at 1.00 and names the file being written on every
        # append. Only the delta may reach the scorer.
        assert hd("t", {}, prior + novel, prior, lambda: docs3) is None, \
            "an append whose DELTA is novel must not fire on its own prior body"
        assert hd("t", {}, prior + restated, prior, lambda: docs3), \
            "an append whose delta restates another file MUST fire"
        assert hd("MEMORY.md", {}, prior + restated, prior, lambda: docs3) is None
        assert hd("CHECKPOINTS.md", {}, prior + restated, prior, lambda: docs3) is None

    def test_a_rewrite_is_not_scored_against_what_it_retires(self):
        docs, prior, novel, restated, docs3, held, stem, body = self.ctx
        # ★ THE SECOND HARD CASE: an in-place rewrite that EXTENDS what it replaces.
        # A markdown paragraph is one physical line, so the extended paragraph is
        # never byte-identical and survives `added_text`'s anchor subtraction; the
        # replaced span must therefore leave the corpus before scoring.
        # `marked` lives ONLY in doc `t`, so any hit on it is the file matching
        # itself; a retiring sentence another doc also asserts would be a true
        # positive and prove nothing. ⚠ It carries MARKDOWN deliberately, because the
        # corpus holds stripped bodies and this case fails without the strip.
        marked = "The `hangar` rate structure is **quote-gated** on this airframe entirely.\n"
        docs_md = dict(docs, t=claims.strip(marked))
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

    def test_a_holding_space_scores_itself(self):
        docs, prior, novel, restated, docs3, held, stem, body = self.ctx
        # a holding space is scored against ITSELF: a repeated entry fires, and an entry
        # restating a fact that lives elsewhere in the corpus does not, because turning
        # over something already known is the surface's job.
        assert hd("SHORT_TERM.md", {}, body + held * 2, body,
                             lambda: docs3), "a repeated entry in a holding space must fire"
        assert hd("SHORT_TERM.md", {}, body + restated, body,
                             lambda: docs3) is None, "a holding space must not score against the corpus"
        # the file's own furniture must not fire: a stem shared by three entries says
        # nothing about whether a fourth entry is new.
        assert hd("SHORT_TERM.md", {}, body + stem * 3 + stem + novel, body + stem * 3,
                             lambda: docs3) is None, "a shared stem is not a duplicate entry"

    def test_a_dated_series_is_chronology(self):
        docs, prior, novel, restated, docs3, held, stem, body = self.ctx
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
        assert gate.dated_series_pair(r1, r2) and not gate.dated_series_pair(r1, r1)
        assert hd("OPEN_QUESTIONS.md", {}, body + r1 + r2, body + r1,
                             lambda: docs3) is None, "a later reading in a dated series must not fire"
        undated = r2.replace("2026-08-16", "").replace("341539", "300232")
        assert hd("OPEN_QUESTIONS.md", {}, body + r1 + undated, body + r1,
                             lambda: docs3), "an undated restatement is still a duplicate entry"

    def test_archived_text_is_a_move(self):
        docs, prior, novel, restated, docs3, held, stem, body = self.ctx
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
        arch = [ts for _, ts in claims.claims(claims.strip(moved))]
        assert hd("new.md", {}, moved, "", lambda: docs_mv, lambda: arch) is None, \
            "a relocation of ARCHIVED text must not fire against its still-live source"
        # ⚠ the archive is read STRIPPED: a filed copy carrying markdown furniture
        # must still match, or the patch no-ops silently while looking like it works.
        arch_md = [ts for _, ts in claims.claims(claims.strip(
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

    def test_dates_on_a_durable_surface(self):
        # ★ a dated history clause on a durable surface fires on the shape alone, warns
        # supervised, blocks a cycle, and never fires on a holding space or a ledger.
        stack = ("Some durable entry about hangar rates and the quote-gated structure of them.\n"
                 "⚠ UPDATE (2026-08-08): the endpoint moved to the API repo and the rates changed.\n") * 2
        _m, _b = gate.hook_decision("entry.md", {}, stack, "", lambda: {})
        assert _m and "dated history clause" in _m and _b is False, _m
        _m, _b = gate.hook_decision("entry.md", {}, stack, "", lambda: {}, autonomous=True)
        assert _m and _b is True, "a cycle stacking a dated clause is blocked"
        scoped = ("Rebrand SHIPPED (v1.1.3), superseding the prior contractor iteration entirely.\n"
                  "It moved into the API repo on 2026-08-02 and is no longer at the workspace root.\n") * 2
        _m, _b = gate.hook_decision("entry.md", {}, scoped, "", lambda: {})
        assert _m and "dated history clause" not in _m and "writes a date" in _m and _b is False, \
            f"a scoped date is not a history clause, and is still asked about: {_m}"
        _m, _b = gate.hook_decision("PROMOTIONS.md", {}, stack, "", lambda: {})
        assert not (_m and "dated history clause" in _m), "a holding space takes its dated transition"
        _m, _b = gate.hook_decision("HARNESS.md", {}, stack, "", lambda: {})
        assert _m is None, "a ledger is never checked"
        # ★ any other date on a durable surface is asked about: warns supervised, never blocks a
        # cycle, skips a dated heading, and is silent on a holding space or a ledger.
        attributed = ("The trim keeps only rules that generalize to any store (Peer ruling 2026-09-20).\n"
                      "Its apparatus clauses move to a Prime-only rule and the file ships to the Partial.\n") * 2
        _m, _b = gate.hook_decision("entry.md", {}, attributed, "", lambda: {})
        assert _m and "writes a date" in _m and _b is False, _m
        _m, _b = gate.hook_decision("entry.md", {}, attributed, "", lambda: {}, autonomous=True)
        assert _m and "writes a date" in _m and _b is False, "a bare date never blocks a cycle"
        headed = ("## Session 33, 2026-09-20\n"
                  "The build passes and the widget target is unchanged since the convention audit.\n") * 2
        _m, _b = gate.hook_decision("entry.md", {}, headed, "", lambda: {})
        assert _m is None, f"a dated heading is convention: {_m}"
        _m, _b = gate.hook_decision("PROMOTIONS.md", {}, attributed, "", lambda: {})
        assert not (_m and "writes a date" in _m), "a holding space dates its transitions"
        _m, _b = gate.hook_decision("HARNESS.md", {}, attributed, "", lambda: {})
        assert _m is None, "a ledger is never checked for dates"

    def test_supervised_warns_and_a_cycle_blocks(self):
        docs, prior, novel, restated, docs3, held, stem, body = self.ctx
        # ★ the supervised/autonomous branch: same finding, different force. A cycle
        # cannot read a warning, so its guard stays hard; a session can, so it warns.
        _m, blocking = gate.hook_decision("t", {}, prior + restated, prior, lambda: docs3)
        assert _m and blocking is False, "a supervised nudge must warn, not block"
        _m, blocking = gate.hook_decision("t", {}, prior + restated, prior, lambda: docs3,
                                     autonomous=True)
        assert _m and blocking is True, "an autonomous nudge must still block"
        assert "about to be written" in _m and "landed" not in _m, _m
        _m, _b = gate.hook_decision("t", {}, prior + restated, prior, lambda: docs3)
        assert _m.startswith(ledger.LEAD_WARNING), "the warn channel must not claim the write was stopped"
        # ⚠ `main` re-leads a warning into the blocking form when the 2e gate refuses
        # the same write. That swap is a literal replace, so it silently no-ops if the
        # two leads ever drift; this is the assertion that keeps them in step.
        assert _m.replace(ledger.LEAD_WARNING, ledger.LEAD_BLOCKING, 1).startswith(ledger.LEAD_BLOCKING), \
            "the blocking re-lead must actually match what the warn channel emits"


class QuestionGate(StoreCase):
    def test_the_question_gate(self):
        novel, COUNCIL = NOVEL_RESERVE, self.council
        # --- the §2e question gate -------------------------------------------------
        # Driveable without ollama: an unreachable embedder degrades the message and
        # never changes the decision, which is the property that lets it block at all.
        d = Path(tempfile.mkdtemp())
        (d / "asked.jsonl").write_bytes(
            b'{"message":{"content":[{"type":"tool_use","name":"Bash",'
            b'"input":{"command":"python3 ~/.claude/scripts/memfind.py \\"x\\" council"}}]}}\n')
        (d / "discussed.jsonl").write_bytes(
            b'{"message":{"content":[{"type":"text","text":"memfind.py ranks candidate owners"}]}}\n')
        assert gate.memfind_asked(str(d / "asked.jsonl")), "a Bash invocation must count as asked"
        assert not gate.memfind_asked(str(d / "discussed.jsonl")), \
            "prose naming the script must NOT count; a session doing memory work names it constantly"
        assert gate.memfind_asked(str(d / "absent.jsonl")), "an unreadable transcript must not block"
        assert gate.memfind_asked(None)

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
            assert gate.runs_memfind(cmd), cmd
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
            assert not gate.runs_memfind(cmd), cmd

        (d / "mentioned.jsonl").write_bytes(
            b'{"message":{"content":[{"type":"tool_use","name":"Bash","input":'
            b'{"command":"grep -n memfind.py ~/.claude/scripts/memscan.py"}}]}}\n')
        assert not gate.memfind_asked(str(d / "mentioned.jsonl")), \
            "a Bash command that only NAMES the script must not discharge the gate"

        # ⚠ fail-open on an unrecognised schema. A false negative here cannot be
        # cleared by asking, because Write and Edit carry no MEMWRITE-OK equivalent,
        # so a transcript shape this function cannot read must never block.
        (d / "future-schema.jsonl").write_bytes(
            b'{"envelope":{"parts":[{"tool":"Bash","args":'
            b'{"cmd":"python3 memfind.py q"}}]}}\n')
        assert gate.memfind_asked(str(d / "future-schema.jsonl")), \
            "a marker this parser cannot structure must fail open, not block"

        unasked = {"transcript_path": str(d / "discussed.jsonl")}
        root = COUNCIL
        assert gate.memfind_gate(unasked, "t.md", {}, novel * 3, "", root), \
            "net-new material with the question unasked must fire"
        assert gate.memfind_gate({"transcript_path": str(d / "asked.jsonl")}, "t.md", {}, novel * 3, "",
                            root) is None, "asking the question once must silence the gate"
        for exempt in ("CHECKPOINTS.md", "SHORT_TERM.md"):
            assert gate.memfind_gate(unasked, exempt, {}, novel * 3, "", root) is None, exempt
        assert gate.memfind_gate(unasked, "MEMORY.md", {}, novel * 3, "", root), \
            "an index file is exempt from the lexical nudge, never from the question"
        assert gate.memfind_gate(unasked, "t.md", {}, novel, novel, root) is None, \
            "a reconsolidation adds nothing and must not fire"
        with mock.patch.dict(os.environ, CLAUDE_AUTONOMOUS="1"):
            assert gate.memfind_gate(unasked, "t.md", {}, novel * 3, "", root) is None, \
                "a cycle cannot act on a nudge and must never be blocked by one"
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
