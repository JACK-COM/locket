"""grepassist: only a question, never a regex; its decision log; where a search lands."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.harness import SRC, StoreCase

import corpora  # noqa: E402
import grepassist  # noqa: E402
import manifest  # noqa: E402
import paths  # noqa: E402


class GrepAssist(StoreCase):
    COUNCIL = False

    def test_only_a_question(self):
        assert grepassist.prose_like("workspace root is a git repository")
        assert not grepassist.prose_like("waive|prototype"), "an alternation is not a question"
        assert not grepassist.prose_like("v_ne\\b"), "a metacharacter disqualifies"
        assert not grepassist.prose_like("repo"), "one content word is not a question"
        assert not grepassist.prose_like(""), "an empty pattern must not fire"
        for ident in ("angle-scout", "cmp-fanout-is-a-dossier", "settings.json",
                      "REFERENCE-CMP-Rulings.csv", "spec_provenance tcds_ref", "~/brain/fact.md"):
            assert not grepassist.prose_like(ident), f"a joined identifier read as a phrase: {ident}"
        assert grepassist.prose_like("source of truth for prices"), "a real phrase must still fire"

    def test_decision_log(self):
        # --- grepassist's decision log: never raises, trims atomically, records every branch
        real_paths, logdir = paths.PATHS, Path(tempfile.mkdtemp(dir=self.tmp))
        try:
            paths.PATHS = real_paths.replace(grepassist_log=logdir / "ro" / "ga.csv")
            (logdir / "ro").mkdir(); (logdir / "ro").chmod(0o500)
            ro = logdir / "ro"                  # writable again, or the cleanup cannot remove it
            self.addCleanup(lambda: ro.exists() and ro.chmod(0o700))
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
            nowhere = Path(tempfile.mkdtemp(dir=self.tmp))
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

    def test_where_a_search_lands(self):
        d = self.tmp
        store = d / "brain"
        (store / "old").mkdir(parents=True)
        (store / "LOG.md").write_text("entry one\n")
        (store / "facts.md").write_text("A fact.\n")
        (store / manifest.MANIFEST).write_text(json.dumps({
            "excluded_dirs": ["old"], "ledger_surfaces": ["LOG.md"],
            "holding_spaces": ["INBOX.md"], "index_files": [],
            "code_roots": [str(d)]}))
        corpora.clear_settings()
        # grepassist places a search by its path words as the shell expands them, from a cwd in no store
        outside, real_home = Path(tempfile.mkdtemp(dir=self.tmp)), os.environ["HOME"]
        assert corpora.corpus_here(outside) is None, "the cwd fallback has nothing here, so the path must place it"
        os.environ["HOME"] = str(d)
        try:
            for cmd in ("grep -rn 'a fact here' ~/brain", "rg 'a fact here' $HOME/brain/*.md",
                        "grep 'a fact here' ~/brain/facts.md", "grep 'a fact here' ~/brain/fact*.md"):
                assert corpora.search_root(shlex.split(cmd), outside) == store, cmd
            assert corpora.search_root(["grep", "-rn", "../facts.md"], store / "old") == store, \
                "a relative path resolves against the session's cwd"
            assert corpora.search_root(["grep", "~/nowhere"], outside) is None, "a path that does not exist places nothing"
            here = d / "cwd-store"; here.mkdir(); (here / manifest.MANIFEST).write_text("{}"); corpora.clear_settings()
            assert corpora.search_root(["grep", "-rn", "~/brain"], here) == store, \
                "a missing word (the verb) resolved against a cwd in another store must not place the search there"
            q = "a fact here"
            for cmd in (f"rg --glob '*.md' '{q}' ~/brain", f"cd /tmp && grep -rn '{q}' ~/brain 2>/dev/null | head",
                        f"grep -rne '{q}' ~/brain", f"grep --regexp='{q}' ~/brain", f"grep -e'{q}' ~/brain",
                        f"grep -rn '{q}' ~/brain 2>&1 | head", f"grep -rn -- '{q}' ~/brain",
                        f"ls | xargs grep -l '{q}' ~/brain", f"git grep '{q}' -- ~/brain",
                        f"which grep; grep '{q}' ~/brain", f"bash -c \"grep -rn '{q}' ~/brain\""):
                assert grepassist.grep_words(cmd, outside) == (q, ["~/brain"]), (cmd, grepassist.grep_words(cmd, outside))
            assert grepassist.grep_words(f"echo '{q}' | wc -l", outside) is None, "no grep verb, no search"
            assert grepassist.grep_words(f"grep -rn foo {store}", outside) == ("", []), \
                "an existing path is never the pattern, however prose-like its words"
            assert grepassist.grep_words("grep -rn foo ~/not-there/memory-store", outside) == ("", []), \
                "a path-shaped word is never the pattern, existing or not"
            assert grepassist.grep_words(f"rg -e '{q}' -e 'other words here' ~/brain", outside) == (q, ["~/brain"]), \
                "a second -e value leaked into the path words"
            assert grepassist.grep_words(f"grep '{q} ~/brain", outside) is not None, "unbalanced quotes must still parse"
            assert grepassist.grep_words(f"git commit -m \"which grep; grep '{q}' ~/brain\"", outside) is None, \
                "a quoted word that only mentions grep is not a command"
            pat, words = grepassist.grep_words(f"grep '{q}' --exclude-dir cwd-store ~/brain", d)
            assert pat == q and corpora.search_root(words, d) == store, \
                "a flag value naming a store under the cwd shadowed the path after it"
            # the first pattern word decides: a regex there means no question, never a later word
            for regex in ("grep -rln 'NOT-CURATION\\|dossier' hooks/ settings.json",
                          "grep -c '\\.py' REFERENCE-CMP-Rulings.csv REFERENCE-CMP-Claims.csv",
                          "grep 'a|b' 'notes about fact here'"):
                assert grepassist.grep_words(regex, outside) == ("", []), (regex, grepassist.grep_words(regex, outside))
            for valued in (f"grep -A 12 '{q}' ~/brain", f"grep --exclude-dir node_modules -rn '{q}' ~/brain",
                           f"grep -m 5 --include '*.md' '{q}' ~/brain", f"rg -g '*.md' '{q}' ~/brain",
                           f"rg -t py '{q}' ~/brain", f"rg -g '!vendor' '{q}' ~/brain",
                           f"grep -d skip '{q}' ~/brain", f"grep -f pats.txt '{q}' ~/brain"):
                assert grepassist.grep_words(valued, outside) == (q, ["~/brain"]), (valued, grepassist.grep_words(valued, outside))
        finally:
            os.environ["HOME"] = real_home


if __name__ == "__main__":
    unittest.main()
