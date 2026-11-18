"""bashguard: a shell command that writes into a store, and every near-miss that does not."""
import os
import unittest
from pathlib import Path
from unittest import mock

from tests.harness import StoreCase

import bashguard  # noqa: E402
import hosts  # noqa: E402
import manifest  # noqa: E402
import paths  # noqa: E402


class BashGuard(StoreCase):
    COUNCIL = False

    def test_fires_on_the_thing_and_no_near_miss(self):
        # The council at its real place under HOME, so every home-relative spelling names
        # it; the store inside is a slug that exists nowhere, so nothing live is read.
        self.hosts(stores={"council": paths.HOME / ".claude"})
        # a command naming a scratch path is exempt by design, so under a HOME in
        # /tmp (Homebrew's test sandbox) every absolute-path case would pass silently
        # Claude Code's home is the real one for this block, under HOME, so every
        # home-relative spelling names it; the council store moves back after.
        if not any(m in str(hosts.claude()) + "/" for m in bashguard.SCRATCH_MARKERS):
            cmem = str(hosts.claude() / "projects" / "-Users-x-proj" / "memory")
            def bw(cmd, cwd=cmem, auto=False):
                return bashguard.bash_write_decision(cmd, cwd, autonomous=auto)
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
            env = mock.patch.dict(os.environ, LOCKET_T_OUT="/Users/x/elsewhere/", LOCKET_T_MEM=cmem)
            env.start()
            self.addCleanup(env.stop)
            os.environ.pop("LOCKET_T_UNSET", None)
            for quiet in ('cat > "$LOCKET_T_OUT/pr.md" <<EOF', "cat > ${LOCKET_T_OUT}pr.md",
                          "echo x >> $LOCKET_T_UNSET/topic_note.md", 'echo x > "$(mktemp -d)/n.md"'):
                assert bw(quiet) == (None, False), f"a variable outside the corpus fired: {quiet}"
            msg, blk = bw("echo x >> $LOCKET_T_MEM/topic_note.md")
            assert msg and not blk, "a variable expanding into a corpus warns rather than blocks"
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


class Registry(StoreCase):
    COUNCIL = False

    def test_a_moved_settings_file_is_still_refused(self):
        """`PANOPLY_CONFIG` moves the embedder settings; the shell refusal follows it,
        as the write tools' does."""
        moved = self.tmp / "elsewhere" / "panoply-settings.json"
        with mock.patch.dict(os.environ, PANOPLY_CONFIG=str(moved)):
            msg, blocking = bashguard.decide(f"cat {moved}", str(self.tmp))
        self.assertTrue(blocking and msg, msg)

    def test_a_sibling_pieces_settings_are_refused(self):
        """Augur's and Grille's settings say what runs and where text goes, so a shell
        command naming either, by home-relative form or by an `*_HOME` move, is refused
        on the same terms as Locket's registry, and the refusal names the file."""
        for cmd in ("cat ~/.augur/augur.json", "echo '{}' > ~/.grille/grille.json"):
            msg, blocking = bashguard.decide(cmd, str(self.tmp))
            self.assertTrue(blocking and msg, cmd)
        moved = self.tmp / "ah"
        with mock.patch.dict(os.environ, AUGUR_HOME=str(moved)):
            msg, blocking = bashguard.decide(f"sed -i '' s/x/y/ {moved}/augur.json", str(self.tmp))
        self.assertTrue(blocking and str(moved) in (msg or ""), msg)
        msg, _ = bashguard.decide("cat notes/augur.md", str(self.tmp))
        self.assertIsNone(msg)

    def test_a_config_name_in_any_case_is_read(self):
        store = (self.tmp / "s").resolve(); store.mkdir()
        (store / manifest.MANIFEST).write_text("{}\n")
        msg, blocking = bashguard.decide(f"echo x > {store / manifest.MANIFEST.upper()}", str(self.tmp))
        self.assertTrue(blocking, msg)


class ReadingConfig(StoreCase):
    COUNCIL = False

    def test_a_refused_read_names_the_way_to_read(self):
        """A command naming a store's manifest is refused, reading or not; the refusal
        says so and names the file tool, so it never reads as a bug to work around."""
        store = (self.tmp / "s").resolve(); store.mkdir()
        (store / manifest.MANIFEST).write_text("{}\n")
        msg, blocking = bashguard.decide(f"cat {store / manifest.MANIFEST}", str(self.tmp))
        self.assertTrue(blocking, msg)
        self.assertIn("file-reading tool", msg)
        self.assertNotIn("This write", msg)


if __name__ == "__main__":
    unittest.main()
