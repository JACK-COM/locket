"""scripts/score-verdicts.py: band edges, and a verdict it does not know."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "score_verdicts", Path(__file__).resolve().parent.parent / "scripts" / "score-verdicts.py")
sv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sv)  # type: ignore[union-attr]


class ScoreVerdicts(unittest.TestCase):
    def _run(self, pairs, *extra):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(pairs, fh)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = sv.main(["score-verdicts.py", fh.name, *extra])
        Path(fh.name).unlink()
        return rc, out.getvalue(), err.getvalue()

    def test_an_edge_score_lands_in_its_own_band(self):
        for score, lo in ((0.70, 0.70), (0.60, 0.60), (0.95, 0.95), (0.699, 0.65)):
            self.assertAlmostEqual(sv.band(score, 0.05), lo, msg=score)

    def test_a_cut_is_blockable_only_above_every_distinct_pair(self):
        rc, out, _ = self._run([{"score": 0.86, "verdict": "dup-a"}, {"score": 0.81, "verdict": "distinct"},
                                {"score": 0.83, "verdict": "merge"}, {"score": 0.7, "verdict": None}])
        assert rc == 0 and "3 of 4 pairs adjudicated" in out, out
        lines = out.splitlines()
        assert "blockable" in next(l for l in lines if l.startswith("0.85")), out
        assert "1 wrongly refused" in next(l for l in lines if l.startswith("0.80")), out

    def test_an_unknown_verdict_is_refused_rather_than_counted_real(self):
        rc, _, err = self._run([{"score": 0.9, "verdict": "dupe"}])
        assert rc == 1 and "unknown verdict(s) dupe" in err, err

    def test_a_width_that_is_not_a_band_is_refused(self):
        for width in ("0", "-0.1", "wide"):
            rc, _, err = self._run([{"score": 0.9, "verdict": "dup-a"}], width)
            assert rc == 1 and "band width" in err, (width, err)


if __name__ == "__main__":
    unittest.main()
