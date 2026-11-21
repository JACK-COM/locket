"""overlap: one claim across stores, and a holding-space claim a durable file now holds."""
import contextlib
import io
import json
import unittest

from tests.harness import DOCS, StoreCase, shown as _shown

import corpora  # noqa: E402
import corpus  # noqa: E402
import manifest  # noqa: E402
from commands import overlap  # noqa: E402


class Across(StoreCase):
    def test_across(self):
        fixture, docs = self.tmp, DOCS
        # across: one claim held by two stores pairs exactly once, and a pair inside
        # one store is audit's, not this mode's.
        two = fixture / "two"
        for store, files in (("alpha", {"m.md": docs["a.md"], "n.md": docs["b.md"]}),
                             ("beta", {"o.md": docs["a.md"] + "\nA sentence only beta holds here today."})):
            (two / store).mkdir(parents=True)
            (two / store / manifest.MANIFEST).write_text("{}\n")
            for f, body in files.items():
                (two / store / f).write_text(body)
        _, _, _, cross = overlap.across_pairs({"alpha": two / "alpha", "beta": two / "beta"})
        assert {(a.split("::")[0], b.split("::")[0]) for _, a, b, _, _ in cross} == {("alpha", "beta")}, cross
        # alpha/m holds both of beta/o's shared sentences, alpha/n holds one: three pairs
        assert len(cross) == 3, [(a, b) for _, a, b, _, _ in cross]
        # a sentence propagated to three stores, one file each, is the finding; the
        # same sentence in three files of ONE store is furniture and drops the pair.
        spread = "Carry cost structure in curated prose and never the cost level itself."
        (two / "gamma").mkdir(); (two / "gamma" / manifest.MANIFEST).write_text("{}\n")
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
        (two / "alpha" / manifest.MANIFEST).write_text(reg)
        (two / "beta" / manifest.MANIFEST).write_text(reg)
        (two / "beta" / "o.md").write_text(docs["a.md"] + "\n" + row)
        (two / "alpha" / "SHORT_TERM.md").write_text(docs["a.md"].split("\n")[0])
        corpora.clear_settings()
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
        (inner / manifest.MANIFEST).write_text("{}\n")
        (inner / "p.md").write_text(docs["a.md"])
        assert not [f for f in corpus.md_files(two / "alpha") if "inner" in str(f)], "nested store must not be walked"
        assert corpora._root_of(inner / "p.md") == inner, "a write inside the nested store belongs to it"


class Graduated(StoreCase):
    def test_graduated(self):
        store = self.tmp / "brain"
        store.mkdir()
        (store / "RULES.csv").write_text(
            "Ruling,Area,Date\n"
            '"Serve the estimate bottom-up from priced components and never from an aggregate.",cost,2026-01-01\n'
            '"A rebase fights the watcher, so build the sheet and commit it before rebasing.",dev-loop,2026-01-02\n')
        # --- graduated: a holding-space claim that a durable source now holds -------
        (store / manifest.MANIFEST).write_text(json.dumps({
            "excluded_dirs": ["old"], "index_files": [], "ledger_surfaces": ["LOG.md"],
            "holding_spaces": ["INBOX.md", "QUEUE.md"],
            "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
        corpora.clear_settings()
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


class Dupes(StoreCase):
    def test_a_holding_space_below_the_store_root_is_found_by_name(self):
        store = self.tmp / "brain"
        (store / "notes").mkdir(parents=True)
        (store / manifest.MANIFEST).write_text(json.dumps({"holding_spaces": ["INBOX.md"]}))
        corpora.clear_settings()
        entry = "Gate: a third instance where the watcher ordering bites on a freshly built sheet."
        (store / "notes" / "INBOX.md").write_text(f"{entry}\n\n{entry}\n")
        for files in (None, ["INBOX.md"], ["notes/INBOX.md"], [str(store / "notes" / "INBOX.md")]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                n = _shown(overlap.cmd_dupes(store, files=files))
            assert n == 1 and "notes/INBOX.md (1)" in buf.getvalue(), (files, buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert _shown(overlap.cmd_dupes(store, files=["NOPE.md"])) == 0
        assert "no file named NOPE.md" in buf.getvalue(), buf.getvalue()

    def _table_store(self):
        store = self.tmp / "tables"
        store.mkdir()
        (store / manifest.MANIFEST).write_text(json.dumps(
            {"sources": [{"path": "RULINGS.csv", "text": "Ruling", "label": "Area"}]}))
        corpora.clear_settings()
        row = "Grep every consumer of an overlapping field before calling the field dead."
        (store / "RULINGS.csv").write_text(
            f'Ruling,Area\n"{row}",schema\n"{row} Always.",schema\n"Paint the hangar doors on alternate days.",x\n')
        (store / "note.md").write_text("A markdown file so the store has content.\n")
        return store

    def test_a_registered_csv_is_ranked_by_meaning_under_any_spelling_of_its_name(self):
        from unittest import mock
        import rows
        store = self._table_store()
        canned = ([(0.9, "RULINGS.csv#schema", "a", "RULINGS.csv#schema", "b")], None)
        for name in ("RULINGS.csv", "./RULINGS.csv", str(store / "RULINGS.csv")):
            with mock.patch.object(rows, "row_pairs", return_value=canned) as rp:
                rep = overlap.cmd_dupes(store, files=[name], top=7)
            rp.assert_called_once()
            assert rp.call_args.args[1] == "RULINGS.csv" and rp.call_args.kwargs["top"] == 7, name
            assert rep.count == 1 and "by meaning" in rep.lines[0], (name, rep.lines)

    def test_a_csv_and_a_markdown_file_in_one_call_each_take_their_own_path(self):
        from unittest import mock
        import rows
        store = self._table_store()
        entry = "Gate: a third instance where the watcher ordering bites on a freshly built sheet."
        (store / "INBOX.md").write_text(f"{entry}\n\n{entry}\n")
        canned = ([(0.9, "RULINGS.csv#schema", "a", "RULINGS.csv#schema", "b")], None)
        with mock.patch.object(rows, "row_pairs", return_value=canned):
            rep = overlap.cmd_dupes(store, files=["RULINGS.csv", "INBOX.md"])
        assert rep.count == 2, rep.lines
        assert any("by meaning" in l for l in rep.lines) and any("INBOX.md (1)" in l for l in rep.lines)

    def test_with_no_embedder_a_csv_falls_back_to_word_overlap_and_says_so(self):
        from unittest import mock
        import rows
        store = self._table_store()
        down = rows.NoEmbedder("no embedder")
        with mock.patch.object(rows, "row_pairs", side_effect=down):
            rep = overlap.cmd_dupes(store, files=["RULINGS.csv"])
        assert "WORD OVERLAP ONLY" in rep.lines[0] and rep.count == 1, rep.lines
        assert "Grep every consumer" in rep.lines[1] and "Always." in rep.lines[1], rep.lines
        with mock.patch.object(rows, "row_pairs", side_effect=down):
            rep = overlap.cmd_dupes(store, files=["RULINGS.csv"], as_json=True)
        assert json.loads(rep.lines[0])[0]["verdict"] is None, "the fallback worklist still parses"
        with mock.patch.object(rows, "incoming_rows", side_effect=down):
            rep = overlap.cmd_dupes(store, files=["RULINGS.csv"], incoming=str(store / "RULINGS.csv"))
        assert "needs an embedder" in rep.lines[0], "incoming rows are never scored by word overlap"
        # any other RuntimeError is a fault, never "no embedder"
        with mock.patch.object(rows, "row_pairs", side_effect=RuntimeError("a real fault")):
            with self.assertRaises(RuntimeError):
                overlap.cmd_dupes(store, files=["RULINGS.csv"])

    def test_json_from_the_command_line_parses(self):
        from unittest import mock
        import rows
        from commands import overlap as ov
        store = self._table_store()
        canned = ([(0.9, "RULINGS.csv#schema", "a", "RULINGS.csv#schema", "b")], None)
        verb = next(v for v in ov.VERBS if v.name == "dupes")
        buf = io.StringIO()
        with mock.patch.object(rows, "row_pairs", return_value=canned), contextlib.redirect_stdout(buf):
            verb.run([str(store), "RULINGS.csv", "--json"])
        assert json.loads(buf.getvalue())[0]["a"] == "a", buf.getvalue()

    def test_an_unreadable_incoming_file_names_its_real_fault(self):
        store = self._table_store()
        rep = overlap.cmd_dupes(store, files=["RULINGS.csv"], incoming=str(store / "absent.csv"))
        assert "holds no" not in rep.lines[0] and "absent.csv" in rep.lines[0], rep.lines
        (store / "wrong.csv").write_text("Claim\n\"A row under some other header entirely.\"\n")
        rep = overlap.cmd_dupes(store, files=["RULINGS.csv"], incoming=str(store / "wrong.csv"))
        assert "holds no `Ruling` rows" in rep.lines[0], rep.lines

    def test_against_reads_a_file_outside_the_store(self):
        from unittest import mock
        import rows
        store = self._table_store()
        rule = self.tmp / "rule.md"
        rule.write_text("Grep every consumer of a field before you call it dead, always.\n")
        with mock.patch.object(rows, "covered_rows", return_value=[(0.8, "RULINGS.csv#schema", "r", "s")]) as cr:
            rep = overlap.cmd_dupes(store, files=["RULINGS.csv"], against=str(rule))
        assert cr.call_args.args[2] == rule.read_text() and rep.count == 1, rep.lines

    def test_the_command_line_refuses_what_it_cannot_apply(self):
        from unittest import mock
        store = self._table_store()
        with mock.patch.object(overlap, "cmd_dupes") as cd:
            overlap._dupes(store, ["RULINGS.csv", "-n", "5", "--json"])
        cd.assert_called_once_with(store, files=["RULINGS.csv"], top=5, incoming=None,
                                   against=None, as_json=True)
        for argv, says in ((["RULINGS.csv", "-n", "five"], "whole number"),
                           (["RULINGS.csv", "-n", "0"], "1 or more"),
                           (["RULINGS.csv", "--against"], "needs a value"),
                           (["RULINGS.csv", "--incomming", "x.csv"], "does not take"),
                           (["RULINGS.csv", "--incoming", "a.csv", "--against", "b.md"], "one at a time"),
                           (["RULINGS.csv", "--json", "--against", "b.md"], "worklist only"),
                           (["note.md", "--json"], "registered under")):
            with mock.patch.object(overlap, "cmd_dupes") as cd:
                rep = overlap._dupes(store, argv)
            cd.assert_not_called()
            assert says in rep.lines[0], (argv, rep.lines)


class IndexPairs(StoreCase):
    def test_an_index_restating_its_own_target_is_not_listed(self):
        """An index line restating the file it links to is the index doing its job; one
        restating a file it does not link to is a finding."""
        store = self.tmp / "s"
        (store / "results").mkdir(parents=True)
        (store / manifest.MANIFEST).write_text(json.dumps({"index_files": ["INDEX.md"]}))
        linked = "Edge reveal measured three hundred maps per row from the centre chunk"
        other = "Every study keeps its tables and limits in its own results file here"
        (store / "INDEX.md").write_text(f"- [{linked}](results/edge.md)\n\n{other}.\n")
        (store / "results" / "edge.md").write_text(f"{linked}.\n")
        (store / "README.md").write_text(f"{other}.\n")
        corpora.clear_settings()
        r = overlap.cmd_audit(store)
        self.assertEqual(r.count, 1, r.text)
        self.assertIn("README.md", r.text)
        self.assertNotIn("results/edge.md", r.text.split("\n", 1)[1])
        self.assertIn("1 as an index line restating the file it links to", r.text)


if __name__ == "__main__":
    unittest.main()
