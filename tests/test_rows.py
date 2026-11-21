"""rows: `dupes` on a registered CSV, over synthetic indexes, so it runs with no embedder."""
import unittest
from pathlib import Path

from tests.harness import fixture_env  # noqa: F401  (puts src/locket on the path)

import memfind as mf  # noqa: E402
import rows  # noqa: E402


class Rows(unittest.TestCase):
    """`dupes` on a CSV, over a synthetic index: no embedder, so it runs everywhere.

    Every vector carries one large shared direction, the anisotropy a real embedding
    space has, so raw cosine puts every pair near 1 and only centring on the table's
    own mean leaves the planted paraphrase on top."""

    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(1)
        shared = rng.normal(size=16) * 6
        self.basis = {k: rng.normal(size=16) for k in "ABCDEFGH"}
        planted = [("R.csv#x", f"row {k}", shared + v) for k, v in self.basis.items()]
        planted.insert(1, ("R.csv#y", "row A reworded", shared + self.basis["A"] + rng.normal(size=16) * 0.6))
        # The decoy: two unrelated rows lying almost wholly along the shared direction.
        # Raw cosine ranks them above the paraphrase; centring is what puts them back.
        planted += [("R.csv#z", f"decoy {k}", shared + rng.normal(size=16) * 0.2) for k in "YZ"]
        # a markdown claim and another table's row: neither may enter R.csv's pairs or its mean
        planted.append(("notes.md", "row A as prose", shared + self.basis["A"]))
        planted.append(("S.csv#x", "row A in another table", shared + self.basis["A"]))
        self.items = [(label, text) for label, text, _ in planted]
        self.shared = shared
        self.idx = (self.items, mf._embed.normalise(np.stack([v for _, _, v in planted])), 16)

    def test_the_planted_paraphrase_ranks_first_and_clears_chance(self):
        pairs, base = rows.row_pairs(Path("."), "R.csv", top=3, idx=self.idx)
        top = pairs[0]
        assert {top[2], top[4]} == {"row A", "row A reworded"}, pairs
        assert base is None, "nine rows are too few pairs for a chance baseline"
        assert all(p[1].startswith("R.csv#") and p[3].startswith("R.csv#") for p in pairs), pairs

    def test_a_table_big_enough_gets_a_chance_line_below_its_duplicate(self):
        import numpy as np
        rng = np.random.default_rng(2)
        shared = rng.normal(size=32) * 6
        vecs = [shared + rng.normal(size=32) for _ in range(80)]
        vecs.append(vecs[0] + rng.normal(size=32) * 0.2)
        items = [("T.csv#x", f"row {i}") for i in range(len(vecs))]
        idx = (items, mf._embed.normalise(np.stack(vecs)), 32)
        pairs, base = rows.row_pairs(Path("."), "T.csv", top=1, idx=idx)
        assert base is not None and pairs[0][0] > base, (pairs, base)
        assert {pairs[0][2], pairs[0][4]} == {"row 0", "row 80"}, pairs

    def test_a_table_with_no_indexed_rows_says_so(self):
        with self.assertRaises(LookupError):
            rows.row_pairs(Path("."), "NOPE.csv", idx=self.idx)

    def test_incoming_rows_find_their_owner_and_filter_nothing(self):
        from unittest import mock
        q = [self.shared + self.basis["C"] + 0.05, self.shared - self.basis["G"]]
        with mock.patch.object(mf, "embed", return_value=q), \
                mock.patch.object(mf, "index_backend", return_value="ollama"):
            ranked = rows.incoming_rows(Path("."), "R.csv", ["near C", "unlike anything", "  "], idx=self.idx)
        assert len(ranked) == 2, "every non-blank incoming row is reported, however low"
        assert ranked[0][1] == "near C" and ranked[0][3] == "row C", ranked

    def test_covered_rows_name_the_row_a_file_restates(self):
        from unittest import mock
        text = "Row E restated as a rule sentence with enough words in it."
        with mock.patch.object(mf, "embed", return_value=[self.shared + self.basis["E"]]), \
                mock.patch.object(mf, "index_backend", return_value="ollama"):
            ranked = rows.covered_rows(Path("."), "R.csv", text, top=2, idx=self.idx)
        assert ranked[0][2] == "row E" and ranked[0][3] == text, ranked
        with self.assertRaises(LookupError):
            rows.covered_rows(Path("."), "R.csv", "Too short.", idx=self.idx)


if __name__ == "__main__":
    unittest.main()
