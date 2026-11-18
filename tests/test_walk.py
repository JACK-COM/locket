"""The corpus walk: which files a store holds, and which store a written path belongs to."""
import contextlib
import io
import json
import os
import unittest
from pathlib import Path

from tests.harness import StoreCase

import corpora  # noqa: E402
import corpus  # noqa: E402
import hosts  # noqa: E402
import manifest  # noqa: E402
import scaffold  # noqa: E402


class HostHome(StoreCase):
    def test_a_host_home_holds_its_members(self):
        fixture, COUNCIL = self.tmp, self.council
        # corpus resolution reaches EVERY project, not a named list
        assert corpora.corpus_for(COUNCIL / "x.md") == COUNCIL
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
        self.hosts(stores={"home": home}, members={"home": {"memory", "rules"}})
        corpora.clear_settings()
        assert corpora.corpus_for(home / "rules/r.md") == home, "a member directory resolves to the home"
        assert corpora.corpus_for(home / "CLAUDE.md") == home, "root-level markdown is corpus"
        assert corpora.corpus_for(home / "plugins/p.md") is None, "a non-member directory is host state"
        assert corpora.corpus_for(home / "memory/archive/old.md") is None, "archive stays provenance"
        assert corpora.corpus_for(home / "memory/vault/v.md") == home / "memory/vault", "a nested store is its own"
        got = {str(q.relative_to(home)) for q in corpus.md_files(home)}
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
            corpus.md_files(home); list(corpus._archive_md(home)); list(corpus._walk_files(home, "*.csv"))
        finally:
            os.scandir, Path.rglob, Path.glob = real_scandir, real_rglob, real_glob
        strays = sorted({str(p.relative_to(home)) for p in scanned
                         if (p == home / "**") or (home in p.parents
                                                   and p.relative_to(home).parts[0] not in ("memory", "rules"))})
        assert not strays, f"the corpus walk entered host state: {strays}"
        # A store holds content when any markdown or registered row exists; an empty one does not.
        assert corpus.has_content(home)
        (fixture / "hollow").mkdir(exist_ok=True)
        assert not corpus.has_content(fixture / "hollow"), "an empty folder read as a store with memory"
        # An unlistable home keeps its members: the root listing failing must not empty
        # the store, or every write into it passes unscored.
        if hasattr(os, "getuid") and os.getuid() != 0:
            home.chmod(0o311)
            try:
                got = {str(q.relative_to(home)) for q in corpus.md_files(home)}
            finally:
                home.chmod(0o755)
            assert got == {"memory/a.md", "rules/r.md"}, f"an unlistable root dropped its members: {got}"
        # A member path is entered only if every component would be: `memory/archive`
        # stays provenance even when a manifest names it as a member.
        self.hosts(members={"home": {"memory", "rules", "memory/archive"}})
        assert not any("archive" in str(q) for q in corpus.md_files(home)), "a member path entered an excluded dir"
        self.hosts(members={"home": {"memory", "rules"}})
        # One unreadable file is skipped and the rest still score: raised, it emptied the
        # whole read inside the hook and the catch-all let the write through.
        if hasattr(os, "getuid") and os.getuid() != 0:
            (home / "memory/locked.md").write_text("A sentence nobody can read here.\n")
            (home / "memory/locked.md").chmod(0)
            try:
                got = corpus.corpus(home)
            finally:
                (home / "memory/locked.md").chmod(0o644); (home / "memory/locked.md").unlink()
            assert "memory/a.md" in got and "memory/locked.md" not in got, sorted(got)
        # A dangling symlink is not a file: reading it raised inside the hook.
        (home / "memory/gone.md").symlink_to(home / "memory/never-there.md")
        assert "memory/gone.md" not in {str(q.relative_to(home)) for q in corpus.md_files(home)}
        (home / "memory/gone.md").unlink()
        assert any("superseded" in " ".join(ts) for _, _, ts in corpus.archived_claims(home).items), \
            "provenance is read from the archive under a member, not root/archive alone"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = scaffold.cmd_init(home, index=False)
        assert rc == 0 and "by convention" in buf.getvalue() and not (home / manifest.MANIFEST).exists(), \
            "init on a host home is a no-op that does not refuse itself for the stores under it"
        (home / manifest.MANIFEST).write_text(json.dumps({"members": ["memory"]}))
        corpora.clear_settings()
        assert corpora.corpus_for(home / "rules/r.md") is None, "a manifest's members replaces the host default"
        assert corpora.corpus_for(home / "memory/a.md") == home
        (home / manifest.MANIFEST).unlink(); corpora.clear_settings()
        corpora.clear_settings()


class Placement(StoreCase):
    def test_auto_memory_and_exclusions(self):
        COUNCIL = self.council
        for slug in ("-a-notes", "-b-myapp", "-c-brand-new-project"):
            p = hosts.claude() / f"projects/{slug}/memory/x.md"
            assert corpora.corpus_for(p) == p.parent, slug
        assert corpora.corpus_for(COUNCIL / "archive/x.md") is None                       # verbatim by design
        assert corpora.corpus_for(COUNCIL / "in-flight/x.md") is None                     # appends until it closes
        assert corpora.corpus_for(hosts.claude() / "projects/-a/memory/archive/x.md") is None
        assert corpora.corpus_for(COUNCIL / "x.txt") is None                              # not markdown
        assert corpora.corpus_for("/Users/x/Software/notes/x.md") is None                 # outside the corpus
        assert corpora.corpus_for(hosts.claude() / "projects/-a/x.md") is None                    # project root, not memory
        # the walker and the write-time gate must cover the same files, or a
        # subdirectory is nudged against a corpus it is not a member of
        for d in hosts.EXCLUDED_DIRS:
            assert corpora.corpus_for(COUNCIL / f"{d}/x.md") is None, d


if __name__ == "__main__":
    unittest.main()
