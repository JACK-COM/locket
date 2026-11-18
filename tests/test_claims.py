"""claims: what a claim is, how two files pair on one, and what reads as a pointer."""
import unittest

from tests.harness import DOCS

import claims  # noqa: E402
import gate  # noqa: E402
from commands import hygiene  # noqa: E402


def hook_payload_from(docs, text, top=3):
    """The FIRE_FLOOR gate against an in-memory corpus."""
    ranked = [r for r in claims.score_against(text, docs)[:top] if r[1] >= claims.FIRE_FLOOR]
    return None if not ranked else "hit"


class Claims(unittest.TestCase):
    def test_a_list_is_not_a_paragraph(self):
        wrapped = ("- first item that wraps\n  onto a second line\n  and onto a third\n"
                   "- second item that wraps\n  onto a second line\n  and a third line too\n")
        assert list(hygiene._prose_paras(wrapped)) == [], "a hard-wrapped list is a list, not a paragraph"
        lazy = "- one item\ncontinued without indentation\nand still the item\n"
        assert list(hygiene._prose_paras(lazy)) == [], "a lazy continuation belongs to its item"
        headed = "# Heading\nProse follows the heading directly.\nMore prose.\nAnd more.\n"
        assert list(hygiene._prose_paras(headed)) == [len(headed.split())], "a heading opens no run"
        indented = "    Indented prose under no marker\n    is still prose.\n"
        assert list(hygiene._prose_paras(indented)) == [len(indented.split())], "indentation alone is not a list"

    def test_history_clause(self):
        # §5 dated history clauses: the shapes loaded-whole files carry, not only `Update <date>`.
        # Positive control is SELF.md's stack form; the negatives are scoped dates §1 permits,
        # and a regex that fires on them turns a worklist into noise.
        for pos in ("*Refinement, 2026-05-19: session-end discipline is structurally load-bearing.*",
                    "**Some stale assertion**. 2026-08-08: **Resolved:** the Peer had a sandwich.",
                    "⚠ UPDATE (2026-08-08): the endpoint moved to the API repo.",
                    "*Instances since graduation, folded from the Workspace 2026-09-11 (supervised).*",
                    "Update 2026-08-08: the cap went to $5."):
            assert gate.HISTORY_CLAUSE.search(pos), f"history clause not matched: {pos}"
        for neg in ("It moved into the API repo on 2026-08-02 and is no longer at the workspace root.",
                    "### trading *(2026-07-27 -- resolved 2026-08-02, personal not work)*",
                    "plus since 2026-09-18 one row in the registry at `~/.locket/corpora.json`",
                    "a gate transition appends one dated clause (Peer ruling 2026-09-11).",
                    "Rebrand SHIPPED (v1.1.3), superseding the prior contractor iteration.",
                    "thread texture is in `in-flight/closed/architectural-stake-refinement-2026-05-12.md`.",
                    "**Target:** SELF.md *Architectural stake* Refinement 2026-05-12, as the mechanism face."):
            assert not gate.HISTORY_CLAUSE.search(neg), f"scoped date matched as history: {neg}"

    def test_a_sentence_across_a_wrap_is_one_claim(self):
        # ★ A sentence spanning a hard wrap is ONE claim; line-meaningful markdown is not joined.
        wrapped = ("On any file-op failure the diagnostic order is: wrong machine, unloaded tool, unanswered\n"
                   "authorization prompt, then channel health. Channel health is last and has not once been the answer.\n")
        cs = [c for c, _ in claims.claims(claims.strip(wrapped))]
        assert any("unanswered authorization prompt, then channel health" in c for c in cs), cs
        assert not any(c.endswith("unanswered") for c in cs), "the wrap must not end a claim"
        bul = "- first bullet carries no terminal stop at all here\n- second bullet carries none either here\n"
        assert len([c for c, _ in claims.claims(claims.strip(bul))]) == 2, "two bullets stay two claims"
        cont = "- first item that wraps onto\n  a second indented line of the same item\n"
        assert len([c for c, _ in claims.claims(claims.strip(cont))]) == 1, "a wrapped bullet is one claim"
        lead = "The three steps that follow are these\n- measured step one is the first of them\n"
        assert not any("these measured" in c for c in [c for c, _ in claims.claims(claims.strip(lead))]), \
            "a line before a list is not joined to its first item"
        head = "# A heading that carries no stop\nProse that follows the heading directly and at length.\n"
        assert not any("stop Prose" in c for c in [c for c, _ in claims.claims(claims.strip(head))]), \
            "a heading is not joined to the prose under it"
        tbl = "| alpha column | beta column here |\n| gamma column | delta column here |\n"
        assert not any("here gamma" in c for c in [c for c, _ in claims.claims(claims.strip(tbl))]), "table rows stay rows"
        bold = "**Status:** N=1 and explicitly not met\n**Target:** a rule file somewhere\n"
        assert len([c for c, _ in claims.claims(claims.strip(bold))]) <= 2 and \
            not any("met Target" in c for c in [c for c, _ in claims.claims(claims.strip(bold))]), \
            "a line ending in a bolded colon-field value still ends at its stop"
        assert list(claims.unclaimed("Short tail here.\nA long enough sentence to be a claim on its own terms.")) \
            == ["Short tail here."], "the floor's rejects are reportable"

    def test_one_claim_in_two_files_pairs(self):
        docs = DOCS
        pairs = claims.cross_file_pairs(docs)
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
        assert claims.cross_file_pairs(docs2), "cross-subject single-clause restatement must pair"

    def test_a_claim_repeated_in_one_file_pairs_there_only(self):
        held = ("The incubator is holding the observation that autonomous cycles skip "
                "reconsolidation whenever the surface is already large. ")
        # within-file pairs are the uniqueness check and are invisible to `audit`
        dup = {"h.md": held * 2 + "\nUnrelated line about ramp availability and transient traffic.\n" + held}
        assert claims.cross_file_pairs(dup, same_file=True), "repeated claim inside one file must pair"
        assert not claims.cross_file_pairs(dup), "the same file must not pair against itself cross-file"

    def test_pointer_only(self):
        assert claims.pointer_only("See project doc basis per engine class.")
        assert not claims.pointer_only("The default execution path is the dossier pipeline run interactively.")
        # a pointer that does not open with "see" is still a pointer
        assert claims.pointer_only("Durable context lives in memory/projects/locket.md for this project.")
        assert claims.pointer_only("The ruling is a RULINGS.csv row on Area=substrate.")
        # a claim that happens to cite a file keeps its claim
        assert not claims.pointer_only("The write gate in memscan.py refuses a fork unless locket find ran earlier in the session.")

    def test_score_against(self):
        docs = DOCS
        ranked = claims.score_against("The default execution path is the dossier pipeline run interactively.", docs)
        assert ranked[0][2] in ("a.md", "b.md"), ranked[0]
        assert ranked[0][1] >= 0.8, ranked[0][1]
        assert hook_payload_from(docs, "wholly unrelated text about runway lighting and taxiways") is None
        # the hard negative: heavy vocabulary overlap, no restated claim. Gating on the
        # combined score passes this and fires; gating on the claim score does not.
        assert hook_payload_from(docs, "Propeller pitch defaults and hangar insurance quotes differ by "
                                       "execution path, and the interactively run pipeline for hangar "
                                       "rates is unrelated to any default here.") is None

    def test_index_line(self):
        m = claims.IDX_LINE.match("- [Some Title](feedback_thing.md) — the hook text here")
        assert m and m.group("target") == "feedback_thing.md", m


if __name__ == "__main__":
    unittest.main()
