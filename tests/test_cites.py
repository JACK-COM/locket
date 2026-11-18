"""cites: what a path:line citation is, the scope it lands in, and whether it resolves."""
import unittest
from unittest import mock

from tests.harness import StoreCase

import cites  # noqa: E402
import corpora  # noqa: E402
from commands import hygiene  # noqa: E402


class Cites(StoreCase):
    def test_what_a_citation_is(self):
        found = {(a, int(b)) for a, b in cites.CITE.findall(
            "see src/models/aircraft.py:2493 and results.html:109 and app.js:7-9, "
            "shipped in 4.2.31 at 2026-08-17T03:23:17Z, cost 11.88:2 not a cite")}
        assert found == {("src/models/aircraft.py", 2493), ("results.html", 109),
                         ("app.js", 7)}, found

        src = ["class Thing:", "    def method(self):", "        return 1",
               "", "def top():", "    pass", "", "CONSTANT = 3"]
        assert cites.enclosing_scope(src, 3) == "def method", cites.enclosing_scope(src, 3)
        assert cites.enclosing_scope(src, 6) == "def top", cites.enclosing_scope(src, 6)
        assert cites.enclosing_scope(src, 5) == "def top", cites.enclosing_scope(src, 5)
        assert cites.enclosing_scope(src, 1) == "class Thing", cites.enclosing_scope(src, 1)
        assert cites.enclosing_scope(src, 99) is None
        # the regression: a module-level constant BELOW a def is not inside it.
        assert cites.enclosing_scope(src, 8) is None, cites.enclosing_scope(src, 8)

    def test_project_slug_round_trips(self):
        # The slug is lossy (dots, dashes and underscores all arrive as `-`), so the
        # decode is only trustworthy if it round-trips through the encoder.
        for name, c in sorted(corpora.all_corpora().items()):
            if name in ("council", "home"):
                continue
            d = cites.project_dir_for(c)
            if d is not None:
                assert corpora.corpus_for_cwd(d) == c, (name, d)

    def test_resolution(self):
        home = self.tmp / "home"
        (home / "rules").mkdir(parents=True)
        # An ambiguous citation is PAST EOF only when no same-named file reaches the line.
        (home / "rules/a").mkdir(); (home / "rules/a/x.py").write_text("a\n")
        (home / "rules/b").mkdir(); (home / "rules/b/x.py").write_text("a\nb\nc\n")
        assert cites.code_index([(home / "rules", frozenset())])["x.py"][0].endswith("a/x.py"), \
            "the short candidate must sort first, or this test passes on first-match resolution"
        cidx = cites.code_index([(home / "rules", frozenset())])
        assert cites.resolve_citation("x.py", 3, cidx)[0] == "ok", "a short first candidate hid a long one"
        assert cites.resolve_citation("x.py", 9, cidx)[0] == "eof"
        # A citation naming a directory never resolves to a same-named file in another one.
        assert cites.resolve_citation("b/x.py", 3, cidx)[0] == "ok"
        assert cites.resolve_citation("c/x.py", 1, cidx) == ("missing", 2), "a path resolved to a stranger"
        assert cites.resolve_citation("./b//x.py", 3, cidx)[0] == "ok", "a normalisable path read as missing"
        (home / "rules/repo-ui/app").mkdir(parents=True); (home / "rules/repo-ui/app/y.py").write_text("y\n")
        cidx = cites.code_index([(home / "rules", frozenset())])
        assert cites.resolve_citation("ui/app/y.py", 1, cidx)[0] == "ok", "an abbreviated repository root read as missing"
        assert cites.resolve_citation("ui/y.py", 1, cidx)[0] == "missing", "a two-part path matched a stranger's folder"
        # The mode end to end: defects counted, every citation listed, the scope named.
        cited = {("x.py", 3): {"m.md"}, ("x.py", 9): {"m.md"}, ("gone.py", 1): {"n.md"}}
        with mock.patch.object(cites, "citations", return_value=cited), \
                mock.patch.object(cites, "code_roots", return_value=[(home / "rules", frozenset())]):
            r = hygiene.cmd_cites(home)
        assert r.count == 2 and "gone.py:1  FILE NOT FOUND" in r.text and "x.py:9  PAST EOF" in r.text \
            and "3 distinct path:line citations across 2 memory files" in r.text, r.text


if __name__ == "__main__":
    unittest.main()
