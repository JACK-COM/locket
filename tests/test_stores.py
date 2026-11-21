"""Stores: a manifest makes a corpus; names, init and the registry; joins; the schema."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tests.harness import StoreCase

import bashguard  # noqa: E402
import cites  # noqa: E402
import claims  # noqa: E402
import corpora  # noqa: E402
import corpus  # noqa: E402
import gate  # noqa: E402
import hosts  # noqa: E402
import ledger  # noqa: E402
import manifest  # noqa: E402
import paths  # noqa: E402
import scaffold  # noqa: E402
import trust  # noqa: E402
from commands import hygiene, stores  # noqa: E402


class Stores(StoreCase):
    def setUp(self):
        super().setUp()
        self.d = self.tmp / "d"
        self.d.mkdir()

    def brain(self):
        """The store every case below builds on: a ledger, a fact, an excluded `old/`."""
        d = self.d
        store = d / "brain"
        (store / "old").mkdir(parents=True)
        (store / "LOG.md").write_text("entry one\n")
        (store / "facts.md").write_text("A fact.\n")
        (store / manifest.MANIFEST).write_text(json.dumps({
            "excluded_dirs": ["old"], "ledger_surfaces": ["LOG.md"],
            "holding_spaces": ["INBOX.md"], "index_files": [],
            "code_roots": [str(d)]}))
        corpora.clear_settings()
        return store

    def test_a_manifest_makes_a_corpus(self):
        d = self.d
        store = d / "brain"
        (store / "old").mkdir(parents=True)
        (store / "LOG.md").write_text("entry one\n")
        (store / "facts.md").write_text("A fact.\n")
        assert corpora.corpus_for(store / "facts.md") is None, "no manifest, no host store: not a corpus"
        (store / manifest.MANIFEST).write_text(json.dumps({
            "excluded_dirs": ["old"], "ledger_surfaces": ["LOG.md"],
            "holding_spaces": ["INBOX.md"], "index_files": [],
            "code_roots": [str(d)]}))
        corpora.clear_settings()
        assert corpora.corpus_for(store / "facts.md") == store, "a manifest names the corpus"
        assert corpora.corpus_for(store / "old" / "x.md") is None, "the manifest's excluded dir holds"
        assert corpora.corpus_for(store / "archive" / "x.md") == store, \
            "a manifest REPLACES the excluded set; `archive` is ours, not theirs"
        s = corpora.settings(store)
        assert s["ledger_surfaces"] == {"LOG.md"} and "INBOX.md" in s["append_surfaces"], s
        assert ".memfind" in s["excluded_dirs"], "the cache dir is excluded whatever the manifest says"
        assert [r for r, _ in cites.code_roots(store)] == [d], "declared code roots win outright"
        assert corpus.md_files(store) == [store / "LOG.md", store / "facts.md"], corpus.md_files(store)
        assert bashguard.bash_write_decision("cat > LOG.md", str(store)) == (None, False), \
            "a ledger named by the manifest is exempt in bashguard too"
        assert bashguard.bash_write_decision("cat > facts.md", str(store))[1], \
            "a plain file in a manifest corpus blocks"

    def test_a_project_store_answers_to_its_name(self):
        # --- a project store answers to its manifest's name, and to its slug still
        real_claude, fake = hosts.claude(), Path(tempfile.mkdtemp(dir=self.tmp))
        proj = fake / "projects" / "-Users-x-devdocs" / "memory"
        proj.mkdir(parents=True); (proj / "a.md").write_text("A note about the project.\n")
        paths.PATHS.host_stores["council"] = fake
        try:
            assert "-Users-x-devdocs" in corpora.all_corpora(), "an unnamed project store is keyed by its slug"
            (proj / manifest.MANIFEST).write_text(json.dumps({"name": "devdocs-main"})); corpora.clear_settings()
            assert corpora.all_corpora().get("devdocs-main") == proj, "a named project store is keyed by its name"
            assert corpus.resolve("devdocs-main") == proj and corpus.resolve("-Users-x-devdocs") == proj, \
                "the name resolves, and the slug still does"
            persona = fake / "projects" / "-Users-x-devdocs-Persona" / "memory"
            persona.mkdir(parents=True); (persona / "p.md").write_text("A persona note.\n")
            corpora.clear_settings()
            assert corpus.resolve("-Users-x-devdocs") == proj, \
                "a named store's exact slug beats a longer unnamed slug containing it"
            assert corpus.resolve("-Users-x-devdocs-Persona") == persona and corpus.resolve("devdocs-main") == proj
            assert corpus.resolve("-USERS-X-DEVDOCS") == proj, "an exact slug matches without case"
            shutil.rmtree(persona.parent); corpora.clear_settings()
            loose = fake / "zq-loose-vault" / "memory"; loose.mkdir(parents=True)
            (loose / "v.md").write_text("A vault note.\n")
            (loose / manifest.MANIFEST).write_text(json.dumps({"name": "zq-named"}))
            assert trust.register(loose); corpora.clear_settings()
            with contextlib.redirect_stderr(io.StringIO()):
                assert corpus.resolve("zq-loose-vault") is None, "only a projects/ store answers to its parent's name"
            assert trust.forget(loose); corpora.clear_settings()
            kid = fake / "projects" / "-Users-x-kid" / "memory"; kid.mkdir(parents=True)
            (kid / "b.md").write_text("Another note.\n")
            (kid / manifest.MANIFEST).write_text(json.dumps({"belongs_to": "-Users-x-devdocs"})); corpora.clear_settings()
            assert corpora.parent_of(kid) == proj, "a belongs_to written by slug still joins the named store"
            (proj / manifest.MANIFEST).write_text(json.dumps({"name": "council"})); corpora.clear_settings()
            assert corpora.raw_corpora().get("council") != proj and "-Users-x-devdocs" in corpora.raw_corpora(), \
                "a project name never takes a host key"
            (proj / manifest.MANIFEST).write_bytes(b'{"name": "\xff"}'); corpora.clear_settings()
            assert trust.read_manifest(proj) == {} and "-Users-x-devdocs" in corpora.raw_corpora(), \
                "a manifest that is not UTF-8 degrades instead of raising"
        finally:
            paths.PATHS.host_stores["council"] = real_claude
            shutil.rmtree(fake, ignore_errors=True); corpora.clear_settings()

    def test_budget_under_the_limit_prints_no_table(self):
        store = self.brain()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            hygiene.cmd_budget(store).show()
        assert "Largest" not in out.getvalue() and "bytes" not in out.getvalue(), \
            f"a store under budget printed an empty table: {out.getvalue()}"

    def test_init_ledgers(self):
        d, store = self.d, self.brain()
        stack = ("Some durable entry about hangar rates and the quote-gated structure of them.\n"
                 "⚠ UPDATE (2026-08-08): the endpoint moved to the API repo and the rates changed.\n") * 2
        # --- init --ledgers: the built-in ledgers, header only, two registered as sources
        man_before = (store / manifest.MANIFEST).read_text()
        with contextlib.redirect_stdout(io.StringIO()):
            assert scaffold.scaffold_ledgers(store, "Brain") == 0
        for pattern, glob, _ in scaffold.LEDGER_FILES:
            f = store / pattern.format("Brain")
            assert f.read_text() == ",".join(manifest.LEDGER_SCHEMAS[glob]["columns"]) + "\n", f.name
            assert not ledger.check_ledger_text(ledger.ledger_schema_for(f)[0], f.read_text()), f"a fresh {f.name} fails its schema"
        srcs = trust.read_manifest(store)["sources"]
        assert sorted(e["path"] for e in srcs) == ["CLAIMS-Brain.csv", "RULINGS-Brain.csv"], srcs
        assert trust.read_manifest(store)["excluded_dirs"] == ["old"], "the manifest's other keys are kept"
        (store / "RULINGS-Brain.csv").write_text("Ruling,Category,Area,Date\nKeep it.,x,y,2026-09-29\n")
        with contextlib.redirect_stdout(io.StringIO()):
            assert scaffold.scaffold_ledgers(store, "Brain") == 0
        assert "Keep it." in (store / "RULINGS-Brain.csv").read_text(), "an existing ledger is never touched"
        assert len(trust.read_manifest(store)["sources"]) == 2, "a source is never listed twice"
        assert gate.session_ledger(store).name == "HISTORY-Brain-Sessions.csv"
        with contextlib.redirect_stderr(io.StringIO()):
            assert scaffold.scaffold_ledgers(store, "a/b") == 1 and scaffold.scaffold_ledgers(store, "*") == 1
        _m, _b = gate.hook_decision("entry.md", {}, stack, "", lambda: {}, root=store)
        assert _m and "a row in HISTORY-Brain-Sessions.csv" in _m and not _b, _m
        _m, _ = gate.hook_decision("entry.md", {}, stack, "", lambda: {}, root=d)
        assert _m and "HISTORY-" not in _m, "no ledger, no pointer at one"
        for pattern, _g, _ in scaffold.LEDGER_FILES:
            (store / pattern.format("Brain")).unlink()
        # a ledger already present under another name is the store's own, registered not doubled
        (store / "RULINGS-Old.csv").write_text("Ruling,Category,Area,Date\n")
        with contextlib.redirect_stdout(io.StringIO()):
            assert scaffold.scaffold_ledgers(store, "Brain") == 0
        assert not (store / "RULINGS-Brain.csv").exists(), "a second RULINGS ledger split the store's rows"
        assert "RULINGS-Old.csv" in {e["path"] for e in trust.read_manifest(store)["sources"]}
        for pattern, _g, _ in scaffold.LEDGER_FILES:
            for f in store.glob(_g):
                f.unlink()
        # a manifest it cannot parse, or `sources` that is not a list, is refused and never rewritten
        for bad in ('{"name": "a", "excluded_dirs": ["old"], "sources": [', '{"sources": {"k": 1}}'):
            (store / manifest.MANIFEST).write_text(bad); corpora.clear_settings()
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.scaffold_ledgers(store, "Brain") == 1, bad
            assert (store / manifest.MANIFEST).read_text() == bad, f"a bad manifest was rewritten: {bad}"
            assert not list(store.glob("*.csv")), "no ledger is written beside a manifest it refused"
        (store / manifest.MANIFEST).write_text(man_before)
        corpora.clear_settings()
        spaced = d / "my notes"; spaced.mkdir(); (spaced / "a.md").write_text("A note.\n")
        with contextlib.redirect_stdout(io.StringIO()):
            assert scaffold.cmd_init(spaced, index=False, ledgers="") == 0, "a folder name with a space still takes ledgers"
        assert (spaced / "HISTORY-my-notes-Sessions.csv").is_file()
        with contextlib.redirect_stderr(io.StringIO()):
            assert scaffold.cmd_init(spaced, index=False, ledgers="a b") == 1, "an explicit bad name is refused"
        trust.forget(spaced)
        assert "old" in corpora.settings(store)["excluded_dirs"] and "backups" in corpora.settings(store)["excluded_dirs"], \
            "a backup directory is excluded whatever the manifest says"
        (store / ".hidden").mkdir(); (store / ".hidden" / "h.md").write_text("A hidden note.\n")
        assert store / ".hidden" / "h.md" not in corpus.md_files(store), "dot-directories are never corpus"
        # an index in a member subdirectory points beside itself, not at the root
        (store / "sub").mkdir(); (store / "sub" / "t.md").write_text("A target.\n")
        (store / "sub" / "MEMORY.md").write_text("- [T](t.md) the target\n- [G](gone.md) removed\n")
        (store / manifest.MANIFEST).write_text(json.dumps({"excluded_dirs": ["old"], "index_files": ["MEMORY.md"]}))
        corpora.clear_settings()
        ih = {h[1]: h[4].exists() for h in hygiene.index_hooks(store)}
        assert ih == {"sub/t.md": True, "sub/gone.md": False}, ih
        shutil.rmtree(store / "sub")

    def test_an_edited_row_below_the_root_never_matches_itself(self):
        """The write hook hides the file's own rows by the key their labels carry. For a
        CSV below the root that key is its path, and the bare name missed every row."""
        store = self.brain()
        (store / "memory").mkdir()
        rules = store / "memory" / "RULES.csv"
        rules.write_text("Ruling,Area,Date\n"
                         '"Serve the estimate bottom-up from priced components and never from an aggregate.",cost,2026-01-01\n')
        (store / manifest.MANIFEST).write_text(json.dumps({
            "sources": [{"path": str(rules), "text": "Ruling", "label": "Area"}]}))
        corpora.clear_settings()
        assert "memory/RULES.csv#cost" in corpus.corpus(store), sorted(corpus.corpus(store))
        prior = rules.read_text()
        edit = {"old_string": "never from an aggregate", "new_string": "not from an aggregate"}
        msg = ledger.source_row_decision(str(rules), edit, prior)[0] or ""
        assert "- memory/RULES.csv" not in msg, msg
        # and a new row restating a row the file keeps is named by the file's key
        again = '"Serve the estimate bottom-up from priced components and never from an aggregate.",pricing,2026-01-02\n'
        msg = ledger.source_row_decision(str(rules), {"new_string": again}, prior, append=True)[0] or ""
        assert "- memory/RULES.csv" in msg, msg
        # the positive control: an edit that matches OTHER prose still warns, so the
        # silence above is the hide working, never the hook deciding nothing
        (store / "note.md").write_text("Serve the estimate top-down from one aggregate and never bottom-up.\n")
        edit = {"old_string": "bottom-up from priced components and never from an aggregate",
                "new_string": "top-down from one aggregate and never bottom-up"}
        msg = ledger.source_row_decision(str(rules), edit, prior)[0] or ""
        assert "- note.md" in msg and "- memory/RULES.csv" not in msg, msg

    def test_a_root_csv_edit_still_sees_a_same_named_sibling(self):
        """Hiding a root CSV's own rows must not hide `memory/<same name>`'s."""
        store = self.brain()
        (store / "memory").mkdir()
        root, sib = store / "RULES.csv", store / "memory" / "RULES.csv"
        root.write_text('Ruling,Area,Date\n"Quote every hangar rate per airframe and never as a flat fee.",cost,2026-01-01\n')
        sib.write_text('Ruling,Area,Date\n"Quote the hangar rate per airframe, never as one flat fee for all.",cost,2026-01-01\n')
        (store / manifest.MANIFEST).write_text(json.dumps({"sources": [
            {"path": str(f), "text": "Ruling", "label": "Area"} for f in (root, sib)]}))
        corpora.clear_settings()
        prior = root.read_text()
        edit = {"old_string": "and never as a flat fee", "new_string": "and never as a flat fee for all"}
        msg = ledger.source_row_decision(str(root), edit, prior)[0] or ""
        assert "- memory/RULES.csv" in msg, msg
        # and the other way: editing the one below the root still sees the root one
        prior = sib.read_text()
        edit = {"old_string": "never as one flat fee for all", "new_string": "and never as a flat fee"}
        msg = ledger.source_row_decision(str(sib), edit, prior)[0] or ""
        assert "- RULES.csv#" in msg and "- memory/RULES.csv" not in msg, msg

    def test_the_claim_cache_never_serves_rows_under_an_older_label(self):
        """Claims cached while rows were labelled by file name must not come back
        under the path labels: the cache key carries the label prefix."""
        from unittest import mock
        import os, time
        store = self.brain()
        (store / "memory").mkdir()
        rules = store / "memory" / "RULES.csv"
        rules.write_text("Ruling,Area,Date\n"
                         '"Serve the estimate bottom-up from priced components and never from an aggregate.",cost,2026-01-01\n')
        old = time.time() - 60                    # past the racy window, so the cache keeps it
        os.utime(rules, (old, old))
        (store / manifest.MANIFEST).write_text(json.dumps({
            "sources": [{"path": str(rules), "text": "Ruling", "label": "Area"}]}))
        corpora.clear_settings()
        real = corpus.csv_keys
        name_only = lambda root, sources=None, roots=None: {p: p.name for p in real(root, sources, roots)}
        with mock.patch.object(corpus, "csv_keys", name_only):
            assert "RULES.csv#cost" in corpus.claim_index(store), "the older scheme, cached"
        assert "memory/RULES.csv#cost" in corpus.claim_index(store), "served the cached older label"

    def test_registered_rows_are_in_the_corpus(self):
        store = self.brain()
        # --- the CSV boundary: registered rows are IN the corpus --------------------
        (store / "RULES.csv").write_text(
            "Ruling,Area,Date\n"
            '"Serve the estimate bottom-up from priced components and never from an aggregate.",cost,2026-01-01\n'
            '"A rebase fights the watcher, so build the sheet and commit it before rebasing.",dev-loop,2026-01-02\n')
        (store / manifest.MANIFEST).write_text(json.dumps({
            "excluded_dirs": ["old"], "index_files": [], "ledger_surfaces": ["LOG.md"],
            "holding_spaces": ["INBOX.md"],
            "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
        corpora.clear_settings()
        c = corpus.corpus(store)
        assert "RULES.csv#cost" in c and "RULES.csv#dev-loop" in c, sorted(c)
        assert "RULES.csv#cost" not in corpus.corpus(store, csv=False), "memfind takes rows as its own units"
        (store / "notes.md").write_text(
            "Serve the estimate bottom-up from priced components and never from an aggregate.\n")
        pairs = claims.cross_file_pairs(corpus.corpus(store))
        assert any({fa, fb} == {"notes.md", "RULES.csv#cost"} for _, fa, fb, _, _ in pairs), \
            "a CSV row copied into prose is a fork the audit must see"
        # and the reverse at write time: a new row restating prose warns, never blocks
        rules_prior = (store / "RULES.csv").read_text()
        row = '"Serve the estimate bottom-up from priced components, never from an aggregate.",pricing,2026-01-03\n'
        msg, blocking = ledger.source_row_decision(str(store / "RULES.csv"), {"new_string": row}, rules_prior, append=True)
        assert msg and "notes.md" in msg and not blocking, msg
        assert ledger.source_row_decision(str(store / "RULES.csv"), {"content": rules_prior}, rules_prior) == (None, False), \
            "a rewrite adding no row draws nothing"
        edit = {"old_string": "never from an aggregate", "new_string": "not from an aggregate"}
        msg = ledger.source_row_decision(str(store / "RULES.csv"), edit, rules_prior)[0] or ""
        assert "- RULES.csv " not in msg, "an edited row never matches the version it replaces"
        msg, _ = gate.hook_decision("new.md", {}, "A rebase fights the watcher, so build the sheet and commit it "
                               "before rebasing. " * 3, "", lambda: corpus.corpus(store), lambda: [], sets=corpora.settings(store))
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

    def test_a_broken_manifest_degrades(self):
        store = self.brain()
        (store / manifest.MANIFEST).write_text("{not json")
        corpora.clear_settings()
        assert corpora.settings(store)["ledger_surfaces"] == manifest.LEDGER_SURFACES, \
            "a broken manifest degrades to the defaults rather than taking the hook down"
        (store / manifest.MANIFEST).unlink()
        corpora.clear_settings()
        assert corpus.resolve(str(store)).is_absolute(), "a literal path resolves absolute"

    def test_excluded_files(self):
        store = self.brain()
        (store / "RULES.csv").write_text(
            "Ruling,Area,Date\n"
            '"Serve the estimate bottom-up from priced components and never from an aggregate.",cost,2026-01-01\n'
            '"A rebase fights the watcher, so build the sheet and commit it before rebasing.",dev-loop,2026-01-02\n')
        (store / "guide.md").write_text("A guide to the store and where its rulings live.\n")
        # --- excluded_files: one file ignored at both ends --------------------------
        (store / "notes").mkdir()
        (store / "notes" / "draft.md").write_text("A draft fact nobody should score against.\n")
        (store / "export.md").write_text("An exported artifact that is not memory.\n")
        (store / manifest.MANIFEST).write_text(json.dumps({
            "excluded_dirs": ["old"], "index_files": ["facts.md"], "ledger_surfaces": ["LOG.md"],
            "excluded_files": ["notes/draft.md", "export*"],
            "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
        corpora.clear_settings()
        got = corpus.md_files(store)
        assert store / "notes" / "draft.md" not in got, "a relative path excludes that file"
        assert store / "export.md" not in got, "a glob excludes by basename"
        assert store / "facts.md" in got and store / "guide.md" in got, got
        assert corpora.corpus_for(store / "notes" / "draft.md") is None, "the gate skips an excluded file too"
        assert corpora.corpus_for(store / "facts.md") == store
        assert corpus.csv_sources(store), "a registered CSV that no pattern names stays a source"
        (store / manifest.MANIFEST).write_text(json.dumps({
            "excluded_files": ["*.csv"],
            "sources": [{"path": str(store / "RULES.csv"), "text": "Ruling", "label": "Area"}]}))
        corpora.clear_settings()
        assert corpus.csv_sources(store) == [], "an excluded CSV is not a source even when registered"
        assert "RULES.csv#cost" not in corpus.corpus(store)
        (store / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()

    def test_init_and_the_registry(self):
        d = self.d
        # --- init and the registry: a manifest directory the listing can find -----
        saved = paths.PATHS; paths.PATHS = saved.replace(registry=d / "locket" / "corpora.json")
        try:
            vault = d / "vault"; (vault / "sub").mkdir(parents=True)
            vault = vault.resolve()             # the registry stores resolved paths; /var is /private/var here
            (vault / "one.md").write_text("A note in a folder nobody has registered.\n")
            assert trust.registry() == [] and vault not in corpora.all_corpora().values()
            assert corpora.corpus_here(vault) is None, "no manifest, no auto-memory: the default has nothing"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = scaffold.cmd_init(vault, index=False)
            assert rc == 0 and json.loads((vault / manifest.MANIFEST).read_text()) == {"name": "vault"}, \
                "a standalone manifest carries the folder's name and no default"
            assert trust.registry() == [vault] and vault in corpora.all_corpora().values(), \
                "init makes the listing see a manifest directory"
            assert corpora.corpus_here(vault) == vault and corpora.corpus_here(vault / "sub") == vault, \
                "the cwd default finds a manifest directory from anywhere inside it"
            assert corpora.corpus_for(vault / "sub" / "x.md") == vault
            assert not trust.register(vault), "registering twice is a no-op"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert scaffold.cmd_init(vault, index=False) == 0
            assert "(kept)" in buf.getvalue() and "already listed" in buf.getvalue(), buf.getvalue()
            # --name without --parent: a standalone label, set at init or renamed later
            named = (d / "notes-2").resolve(); named.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(named, index=False, name="field-notes") == 0
            assert trust.read_manifest(named) == {"name": "field-notes"} and corpora.all_corpora().get("field-notes") == named
            with contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(vault, index=False, name="vault-renamed") == 0
            assert trust.read_manifest(vault)["name"] == "vault-renamed", "--name on a kept manifest renames it"
            clash = (d / "clash").resolve(); clash.mkdir()
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(clash, index=False, name="field-notes") == 1
            assert "is taken by" in err.getvalue() and not (clash / manifest.MANIFEST).exists(), err.getvalue()
            twin = (d / "other" / "field-notes").resolve(); twin.mkdir(parents=True)
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(twin, index=False) == 1, "a folder name another store holds is refused, not keyed by path"
            assert "--name" in err.getvalue(), err.getvalue()
            assert trust.forget(named), "the named store leaves the registry as it came"
            with contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(vault, index=False, name="vault") == 0
            with contextlib.redirect_stderr(io.StringIO()):
                assert scaffold.cmd_init(vault / "sub", index=False) == 1, "a directory inside a corpus is refused"
            # and so is a directory CONTAINING one, whether the inner store is listed
            # or only carries a manifest by hand; the parent is left untouched.
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                assert scaffold.cmd_init(d, index=False) == 1, "a directory containing a corpus is refused"
            assert "contains the corpus" in err.getvalue() and not (d / manifest.MANIFEST).exists(), err.getvalue()
            hand = d / "hand"; hand.mkdir(); (hand / manifest.MANIFEST).write_text("{}\n")
            assert trust.forget(vault)
            with contextlib.redirect_stderr(io.StringIO()):
                assert scaffold.cmd_init(d, index=False) == 1, "an unlisted manifest below is found too"
            assert not (d / manifest.MANIFEST).exists()
            (hand / manifest.MANIFEST).unlink(); hand.rmdir()
            assert trust.register(vault)
            (vault / manifest.MANIFEST).write_text(json.dumps({"name": "vault-of-one"})); corpora.clear_settings()
            assert corpora.all_corpora().get("vault-of-one") == vault, "the manifest's name is the listing key"
            assert corpus.resolve("vault-of") == vault, "and a substring of it resolves"
            (d / "vault-of-one-persona").mkdir()
            (d / "vault-of-one-persona" / manifest.MANIFEST).write_text(json.dumps({"name": "vault-of-one-persona"}))
            assert trust.register(d / "vault-of-one-persona"); corpora.clear_settings()
            assert corpus.resolve("vault-of-one") == vault, "an exact name wins over a longer name containing it"
            with contextlib.redirect_stderr(io.StringIO()):
                assert corpus.resolve("vault-of") is None, "a substring of both is still ambiguous"
            assert trust.forget(d / "vault-of-one-persona")
            shutil.rmtree(d / "vault-of-one-persona")
            (vault / manifest.MANIFEST).unlink(); corpora.clear_settings()
            assert vault not in corpora.all_corpora().values() and trust.registry_stale() == [vault], \
                "a registered path whose manifest is gone is reported, never enumerated"
            assert any("is gone" in l for l in stores.corpora_lines()), stores.corpora_lines()
            assert trust.forget(vault) and not trust.forget(vault) and trust.registry() == [], "forget drops the row once"
            assert not any("is gone" in l for l in stores.corpora_lines())
            with contextlib.redirect_stderr(io.StringIO()):
                assert scaffold.cmd_init(vault / "one.md", index=False) == 1, "a file is not a corpus"
            hostlike = (d / "hostlike").resolve(); hostlike.mkdir()
            (hostlike / "a.md").write_text("A host store's note.\n")
            paths.PATHS.host_stores["_selftest"] = hostlike        # a store listed on sight, for this block only
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    assert scaffold.cmd_init(hostlike, index=False) == 0
                assert "by convention" in buf.getvalue() and trust.registry() == [] \
                    and not (hostlike / manifest.MANIFEST).exists(), \
                    "a host store is listed on sight, so init writes it no manifest and no row"
            finally:
                del paths.PATHS.host_stores["_selftest"]
        finally:
            paths.PATHS = saved

    def test_belongs_to(self):
        d = self.d
        # --- belongs_to: two folders in two places, one corpus ------------------
        saved = paths.PATHS; paths.PATHS = saved.replace(registry=d / "locket2" / "corpora.json")
        try:
            par = (d / "par").resolve(); (par / "docs").mkdir(parents=True)
            (par / "IMPL.md").write_text("The impl points at [[topic_one]] and [[topic_two]].\n"
                                         "See `RULES.csv` query `Area=cost` for the ruling.\n")
            (par / "docs" / "d.md").write_text("A doc under the parent's member directory.\n")
            (par / "RULES.csv").write_text("Area,Ruling\ncost,The hangar rate is quote-gated on every airframe.\n")
            (par / manifest.MANIFEST).write_text(json.dumps({
                "name": "par", "members": ["docs"], "ledger_surfaces": ["LOG.md"],
                "whole_read_exempt": ["IMPL.md"],
                "sources": [{"path": "RULES.csv", "text": "Ruling", "label": "Area"}]}))
            corpora.clear_settings(); trust.register(par)
            kid = (d / "elsewhere" / "memory").resolve(); (kid / "archive").mkdir(parents=True)
            (kid / "topic_one.md").write_text("Topic one holds a durable fact about hangars.\n")
            (kid / "SHORT.md").write_text("- an entry\n")
            (kid / "LOG.md").write_text("2026-01-01 a ledger line\n")
            (kid / "archive" / "old.md").write_text("A superseded sentence kept as provenance here.\n")
            (kid / manifest.LEGACY_MANIFEST).write_text(json.dumps(
                [{"path": str(par / "RULES.csv"), "text": "Ruling", "label": "Area"}]))
            (kid / manifest.MANIFEST).write_text(json.dumps({"name": "kid", "holding_spaces": ["SHORT.md"]}))
            corpora.clear_settings(); trust.register(kid)
            assert {"par", "kid"} <= set(corpora.all_corpora()), "before joining, both are listed"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert scaffold.cmd_init(kid, index=False, parent="par") == 0, buf.getvalue()
            corpora.clear_settings()
            assert trust.read_manifest(kid)["belongs_to"] == "par" and trust.read_manifest(kid)["name"] == "kid", \
                "init --parent adds belongs_to and keeps the rest of the manifest"
            assert "kid" not in corpora.all_corpora() and corpora.all_corpora()["par"] == par, \
                "a joined store is listed through its parent alone"
            assert corpora.parent_of(kid) == par and corpora.top_of(kid) == par and corpora.top_of(par) == par
            assert corpora.roots_of(par) == [par, kid]
            assert corpora.corpus_for(kid / "topic_one.md") == par, "a write into the joined store is scored at the parent"
            assert corpora.corpus_for(kid / "archive" / "old.md") is None, "the joined store's own exclusions hold"
            assert corpora.corpus_here(kid) == par and corpus.resolve(str(kid)) == par and corpus.resolve("par") == par
            got = {corpora.corpus_key(par, q) for q in corpus.md_files(par)}
            assert got == {"IMPL.md", "docs/d.md", "kid/LOG.md", "kid/SHORT.md", "kid/topic_one.md"}, got
            assert set(corpus.corpus(par)) >= {"kid/topic_one.md", "RULES.csv#cost"}, "keys are composite; one CSV, once"
            assert len(corpus.csv_sources(par)) == 1, "a CSV registered by both stores is one source"
            # A CSV living in the joined store is keyed as its markdown is, under the
            # store's label, so a same-named CSV at the parent's root stays apart from it.
            legacy = (kid / manifest.LEGACY_MANIFEST).read_text()
            (kid / "RULES.csv").write_text("Area,Ruling\ncost,The joined store's own ruling on hangar rates.\n")
            (kid / manifest.LEGACY_MANIFEST).write_text(json.dumps(
                json.loads(legacy) + [{"path": str(kid / "RULES.csv"), "text": "Ruling", "label": "Area"}]))
            corpora.clear_settings(); trust.register(kid)
            kid_key = f"{corpora.store_label(kid)}/RULES.csv"
            assert sorted(corpus.csv_keys(par).values()) == sorted(["RULES.csv", kid_key]), corpus.csv_keys(par)
            assert {"RULES.csv#cost", f"{kid_key}#cost"} <= set(corpus.corpus(par)), sorted(corpus.corpus(par))
            # a joined store's key is no path on disk, so only the exact-key rung finds it
            assert [k for _p, _t, k in corpus.csv_named(par, kid_key)] == [kid_key]
            (kid / manifest.LEGACY_MANIFEST).write_text(legacy); (kid / "RULES.csv").unlink()
            corpora.clear_settings(); trust.register(kid)
            st = corpora.settings(par)
            assert "SHORT.md" in st["holding_spaces"] and "LOG.md" in st["ledger_surfaces"] \
                and "SHORT.md" in st["append_surfaces"], "name-sets union upward"
            assert "SHORT.md" not in corpora.settings(kid)["ledger_surfaces"], "and never downward"
            assert st["whole_read_exempt"] == {"IMPL.md"} and corpora.settings(kid)["whole_read_exempt"] == set(), \
                "the size-sweep exemption is a name-set like the others: read, unioned upward, never down"
            assert not hygiene.dangling(par).get("topic_one") and hygiene.dangling(par).get("topic_two"), \
                "a wikilink into the joined store resolves; a missing one still dangles"
            (kid / "HISTORY-Kid-Sessions.csv").write_text("date,title,summary,area,repo,commit_hash,issue,pr\n")
            assert gate.session_ledger(par) == kid / "HISTORY-Kid-Sessions.csv" == gate.session_ledger(kid), \
                "a joined store's session ledger is the store's, found from either folder"
            (kid / "HISTORY-Kid-Sessions.csv").unlink()
            (par / "feedback_named_one.md").write_text(
                "---\nname: named-one\ndescription: x\n---\n\nA body line with name: not-this.\n"
                "Links to [[named-one]] and [[not-this]].\n")
            dl = hygiene.dangling(par)
            assert "named-one" not in dl and "not-this" in dl, \
                "a link resolves by the frontmatter name, never by a `name:` in the body"
            (par / "feedback_named_one.md").unlink()
            assert any("superseded" in " ".join(ts) for _, _, ts in corpus.archived_claims(par).items), \
                "provenance is read from the joined store's archive too"
            assert any(s[4] == "ok" for s in hygiene.csv_pointers(par)), hygiene.csv_pointers(par)
            assert any("[1 joined store]" in l for l in stores.corpora_lines()), stores.corpora_lines()
            # refusals: unknown parent, self, a store nesting on disk, a cycle
            for bad in ("nobody", "kid"):
                with contextlib.redirect_stderr(io.StringIO()):
                    assert scaffold.cmd_init(kid, index=False, parent=bad) == 1, bad
            sub = par / "docs"; (sub / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()
            with contextlib.redirect_stderr(io.StringIO()):
                assert scaffold.cmd_init(sub, index=False, parent="par") == 1, "a nested directory is a member, not a join"
            (sub / manifest.MANIFEST).unlink(); corpora.clear_settings()
            (par / manifest.MANIFEST).write_text(json.dumps({"name": "par", "belongs_to": "kid"}))
            corpora.clear_settings()
            assert corpora.parent_of(par) is None and corpora.parent_of(kid) is None, "a cycle leaves both standing alone"
            assert {"par", "kid"} <= set(corpora.all_corpora())
            (par / manifest.MANIFEST).write_text(json.dumps({"name": "par"})); corpora.clear_settings()
            assert corpora.parent_of(kid) == par
            # nested on disk but outside the parent's members: a host home's own
            # auto-memory under projects/ is the case, and it joins
            (par / manifest.MANIFEST).write_text(json.dumps({"name": "par", "members": ["docs"]})); corpora.clear_settings()
            out = par / "projects" / "m"; out.mkdir(parents=True)
            (out / "o.md").write_text("A nested store outside the parent's members joins it.\n")
            (out / manifest.MANIFEST).write_text(json.dumps({"name": "outm"})); corpora.clear_settings()
            trust.register(out)                          # a real auto-memory is listed on sight
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(out, index=False, parent="par") == 0, "a store outside the members joins"
            assert corpora.parent_of(out) == par and corpora.corpus_for(out / "o.md") == par
            assert [q.name for q in corpus.md_files(par)].count("o.md") == 1, "counted once, as a joined store"
            # a joining folder called like one of the parent's directories needs a label
            am = (d / "host" / "projects" / "slug" / "memory").resolve(); am.mkdir(parents=True)
            (am / "a.md").write_text("An auto-memory fact about the council store here.\n")
            (am / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings(); trust.register(am)
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(am, index=False, parent="par", name="docs") == 1, "a label naming a parent dir"
                assert scaffold.cmd_init(am, index=False, parent="par", name="a/b") == 1, "a label with a slash"
            assert "--name <label>" in err.getvalue(), err.getvalue()
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                assert scaffold.cmd_init(am, index=False, parent="par", name="auto-memory") == 0
            assert trust.read_manifest(am) == {"name": "auto-memory", "belongs_to": "par"}, trust.read_manifest(am)
            assert corpora.corpus_key(par, am / "a.md") == "auto-memory/a.md"
            assert trust.forget(am)
            assert trust.forget(out)
            assert trust.forget(par) and trust.forget(kid)
        finally:
            paths.PATHS = saved
            corpora.clear_settings()

    def test_manifest_schema(self):
        d = self.d
        # --- the manifest schema: hints for the hand that writes one ---------------
        sd = (d / "schemadir").resolve(); sd.mkdir()
        assert manifest.validate_manifest(sd) == [], "no manifest, nothing to say"
        (sd / manifest.MANIFEST).write_text(json.dumps({
            "name": "s", "members": ["docs"], "belongs_to": "x",
            "sources": [{"path": "R.csv", "text": "Ruling", "label": "Area"}],
            "ledgers": [{"path": "L.csv", "caps": {"note": 10}, "enum": {"k": ["a", ""]}}]}))
        assert manifest.validate_manifest(sd) == [], manifest.validate_manifest(sd)
        (sd / manifest.MANIFEST).write_text(json.dumps({
            "belongs-to": "x", "members": "docs", "sources": [{"text": "Ruling"}],
            "ledgers": [{"path": "L.csv", "caps": {"note": "ten"}, "extra": 1}]}))
        f = manifest.validate_manifest(sd)
        assert any("unknown key 'belongs-to', did you mean 'belongs_to'?" in x for x in f), f
        assert any("members: expected array" in x for x in f), f
        assert any("sources[0]: missing required key 'path'" in x for x in f), f
        assert any("ledgers[0].caps.note: expected integer" in x for x in f), f
        assert any("ledgers[0]: unknown key 'extra'" in x for x in f), f
        (sd / manifest.MANIFEST).write_text("{not json")
        assert manifest.validate_manifest(sd) and "does not parse" in manifest.validate_manifest(sd)[0]
        (sd / manifest.MANIFEST).unlink()
        assert json.loads(json.dumps(manifest.MANIFEST_SCHEMA))["properties"].keys() >= {
            "belongs_to", "ledgers", "sources", "members"}, "the schema names the keys the reader takes"


class UntrustedNote(StoreCase):
    def test_a_verb_says_why_it_passed_over_a_store(self):
        """A folder whose manifest changed since a person trusted it acts as no store; a
        verb run inside it names that folder and the command that restores it, where it
        fell back to another store in silence."""
        work = (self.tmp / "work").resolve()
        (work / "sub").mkdir(parents=True)
        (work / manifest.MANIFEST).write_text("{}\n")
        with mock.patch.object(trust, "trusted", lambda d: False):
            note = trust.untrusted_note(work / "sub")
        self.assertIn(str(work), note or "")
        self.assertIn("not trusted", note or "")
        self.assertIsNone(trust.untrusted_note(self.tmp / "elsewhere"))


if __name__ == "__main__":
    unittest.main()
