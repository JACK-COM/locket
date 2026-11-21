"""memfind's suite: the lexical rung offline; the ranking and CSV rows where an embedder answers."""
import json
import math
import sys
import unittest
from pathlib import Path

from tests.harness import fixture_env

import _embed  # noqa: E402
import memfind as mf  # noqa: E402
import memscan  # noqa: E402
import corpora  # noqa: E402
import manifest  # noqa: E402

def _ranked():
    # A synthetic store, never the live one. Reading whatever this machine holds
    # made the assertions depend on its shape: a store of three files, one of them
    # named as an index by a fixed list, has one pair and returned nothing. The
    # fixture carries a manifest naming its own index file, so that path is
    # exercised too, and it needs an embedder: without one it says so and stops.
    import tempfile, shutil
    d = Path(tempfile.mkdtemp(prefix="memfind-selftest-"))
    root = d / "store"
    root.mkdir()
    (root / "engines.md").write_text(
        "Turbine overhaul programs cover the engine reserve entirely for the operator.\n"
        "A hot-section inspection falls due at half the overhaul interval on most turboprops.\n")
    (root / "airframes.md").write_text(
        "Corrosion inspections on a seaplane hull are due every hundred hours in salt water.\n"
        "Fabric-covered wings need a punch test whenever the finish shows crazing or chalking.\n")
    (root / "costs.md").write_text(
        "The engine reserve is the largest single line in a turboprop's hourly operating cost.\n"
        "Hangar rent varies more with the airport than with the aircraft parked inside it.\n")
    (root / "INDEX.md").write_text(
        "- [engines](engines.md) — turbine overhaul programs cover the engine reserve\n"
        "- [costs](costs.md) — the engine reserve is the largest hourly line\n")
    (root / "fleet.md").write_text(               # a table: one claim per row, never one per table
        "| aircraft | engine | note |\n|---|---|---|\n"
        + "".join(f"| Type {i} | Engine {i} | a row about airframe number {i} in the fleet table |\n"
                  for i in range(40)))
    (root / manifest.MANIFEST).write_text(json.dumps({"index_files": ["INDEX.md"]}))
    corpora.clear_settings()
    items, flat, dim = mf.load_index(root, quiet=True)
    assert flat.shape == (len(items), dim), "vector file does not match item count"
    assert sum(1 for f, _ in items if f == "fleet.md") == 40, "table rows must index one claim each"
    for i in (0, len(items) // 2, len(items) - 1):        # stored normalised
        n = math.sqrt(sum(float(x) ** 2 for x in flat[i]))
        assert abs(n - 1.0) < 1e-3, f"vector {i} not normalised: {n}"
    r = mf.rank(items[0][1], root, top=1, quiet=True)         # a claim finds itself
    assert r[0][0] > 0.95, f"self-similarity too low: {r[0][0]:.3f}"
    assert mf.LAST_RANK_MODE == "semantic"
    # the last rung: with no embedder at all the same claim still finds itself by
    # word overlap, the caller is told which mode answered, and a hook refuses
    # rather than dressing overlap up as meaning
    lx = mf.lexical_rank(items[0][1], root, top=1, items=items)
    assert lx and lx[0][1] == items[0][0] and lx[0][0] > 0.99, lx
    with mf.settings(_ACTIVE=None, AUTOSTART=False, OLLAMA="http://127.0.0.1:1"):
        if not _embed.have_onnx():
            r = mf.rank(items[0][1], root, top=1, quiet=True, idx=(items, flat, dim))
            assert mf.LAST_RANK_MODE == "lexical" and r[0][1] == items[0][0], r
            try:
                mf.rank(items[0][1], root, top=1, quiet=True, idx=(items, flat, dim), lexical=False)
                assert False, "a hook-path rank with no embedder must raise, never fall back"
            except RuntimeError:
                pass
    # the rebinding trap is closed: an embed setting assigned through memfind raises
    # rather than rebinding a name the embedder never reads
    try:
        sys.modules[mf.__name__].TIMEOUT = 1
        assert False, "memfind.TIMEOUT = ... must raise, not silently miss the embedder"
    except AttributeError as e:
        assert "settings(" in str(e), e
    with mf.settings(TIMEOUT=7):
        assert sys.modules[mf.__name__].TIMEOUT == 7 == _embed.TIMEOUT, "a setting read through memfind is stale"
    for name in mf._EMBED_SETTINGS:           # every refusal names a call that works
        try:
            setattr(sys.modules[mf.__name__], name, None)
        except AttributeError as e:
            key = str(e).split("settings(")[1].split("=")[0]
            with mf.settings(**{key: getattr(_embed, key)}):
                pass
    s = mf.siblings(root, top=3, quiet=True, idx=(items, flat, dim))
    assert len(s) == 3, f"three non-index files make three pairs, got {len(s)}"
    assert all(-1.001 <= c <= 1.001 for c, _, _, _ in s), "subject score out of range"
    assert [c for c, _, _, _ in s] == sorted((c for c, _, _, _ in s), reverse=True), \
        "siblings not ranked descending"
    assert all(f != "INDEX.md" for _, _, a, b in s for f in (a, b)), \
        "the manifest's index file leaked into the sibling pairs"
    # No assertion on WHICH pair ranks first: centroids are mean-centred, and on
    # three files the three residuals sum to zero, so every subject score is
    # negative and their order says little. Siblings is a maintenance tool for a
    # corpus of dozens of files; what is checked here is shape, not judgment.
    corpora.clear_settings()
    shutil.rmtree(d, ignore_errors=True)
def _lexical():
    """The no-embedder rung, which needs no embedder to test, so it runs on the
    hosts that depend on it. The fixture is the case BM25 replaced the cosine for:
    one claim shares three words that most claims carry, another shares the one
    word nothing else does, and the rare word has to win. The unweighted cosine
    ranked the common-word claim first on this fixture."""
    common = [("chatter.md", f"session file notes number {i} appear in the daily log")
              for i in range(20)]
    items = common + [
        ("common.md", "session file notes appear together every single day"),
        ("hermes.md", "hermes runs overnight beside ollama locally"),
    ]
    r = mf.lexical_rank("hermes session file notes", None, top=2, items=items)
    assert r[0][1] == "hermes.md", f"the rare word must outrank three common ones: {r}"
    r = mf.lexical_rank(items[-2][1], None, top=1, items=items)
    assert r[0][1] == "common.md" and r[0][0] > 0.99, f"a claim must find itself at 1.000: {r}"
    assert mf.lexical_rank("nothing here matches", None, items=items) == []
    assert all(0 < c <= 1.0 for c, _, _ in mf.lexical_rank("session notes", None, top=30, items=items))
def _rows():
    """The CSV-row path, on a corpus built for it.

    Built rather than borrowed: few standing corpora register a reference
    CSV, so asserting against whatever this machine happens to hold would pass
    vacuously everywhere else and hide a broken row path until it mattered.
    """
    import tempfile, shutil, csv as _csv
    d = Path(tempfile.mkdtemp(prefix="memfind-selftest-"))
    try:
        (d / "a.md").write_text("The hangar door opens northward onto the taxiway apron.\n")
        (d / "b.md").write_text("Fuel is drawn from the south pump before every departure.\n")
        c = d / "R.csv"
        with open(c, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["Ruling", "Area"])
            w.writerow(["Never taxi past the yellow line without a marshaller present.", "ops"])
            w.writerow(["Chock both mains before disconnecting the tug from the nosewheel.", "ops"])
        (d / mf.MANIFEST).write_text(json.dumps(
            [{"path": str(c), "text": "Ruling", "label": "Area"}]))

        items, flat, dim = mf.build_index(d, quiet=True)
        rows = [i for i in items if mf._is_row(i[0])]
        assert len(rows) == 2, f"expected 2 row items, got {len(rows)}"
        assert all(l == "R.csv#ops" for l, _ in rows), f"bad row labels: {rows}"
        assert flat.shape == (len(items), dim), "vector file does not match item count"

        r = mf.rank("chock the mains before the tug comes off", d, top=3, quiet=True)
        assert mf._is_row(r[0][1]), f"a row should outrank the markdown here: {r[0][1]}"

        # Both rows share one label, so grouping by label would collapse them.
        r = mf.rank("marshaller and chocks on the apron", d, top=4, quiet=True)
        assert len([x for x in r if mf._is_row(x[1])]) == 2, \
            f"rows collapsed under their shared label: {r}"

        sib = mf.siblings(d, top=5, quiet=True)
        assert all(not mf._is_row(f) for _, _, a, b in sib for f in (a, b)), \
            "a CSV row leaked into the sibling pairs"

        # An edited CSV must reuse the untouched row rather than re-embedding it.
        with open(c, "a", newline="") as fh:
            _csv.writer(fh).writerow(["Tow only with the nose gear pin installed.", "ops"])
        items2, flat2, dim2 = mf.build_index(d, quiet=True)
        assert len([i for i in items2 if mf._is_row(i[0])]) == 3, "appended row not indexed"
        old_vec = [i for i, x in enumerate(items) if x[1].startswith("Never taxi")][0]
        new_vec = [i for i, x in enumerate(items2) if x[1].startswith("Never taxi")][0]
        assert (flat[old_vec] == flat2[new_vec]).all(), \
            "an unchanged row was re-embedded instead of reused"
    finally:
        meta_p, vec_p = mf.cache_paths(d)
        for f in (meta_p, vec_p):
            if f.exists():
                f.unlink()
        shutil.rmtree(d, ignore_errors=True)


class MemfindBehaviour(unittest.TestCase):
    def test_lexical(self):
        with fixture_env():
            _lexical()

    def test_rows(self):
        self._needs_embedder()
        with fixture_env():
            _rows()

    def test_ranked(self):
        self._needs_embedder()
        with fixture_env():
            _ranked()

    def _needs_embedder(self):
        try:
            mf.resolve_backend()
        except RuntimeError as e:
            self.skipTest(f"no embedder here: {e}")


class Fingerprint(unittest.TestCase):
    def test_two_csvs_of_one_name_are_two_entries(self):
        """Joined stores can each register a RULINGS.csv; keyed by name, an edit to one
        left the index looking fresh."""
        import tempfile
        from unittest import mock
        with fixture_env(), tempfile.TemporaryDirectory() as t:
            root = Path(t) / "s"; root.mkdir()
            (root / manifest.MANIFEST).write_text("{}\n")
            a, b = Path(t) / "a" / "RULINGS.csv", Path(t) / "b" / "RULINGS.csv"
            for p in (a, b):
                p.parent.mkdir(); p.write_text("Ruling,Area\n")
            with mock.patch.object(mf, "csv_sources", return_value=[(a, "", ""), (b, "", "")]):
                before = mf.corpus_fingerprint(root)
                a.write_text("Ruling,Area\nA new row that changes the file size.,x\n")
                after = mf.corpus_fingerprint(root)
        self.assertNotEqual(before, after)


class BuildLock(unittest.TestCase):
    """One build per store at a time, and a build that waited embeds nothing new."""

    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory(prefix="memfind-lock-")
        self.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name)
        (self.d / "a.md").write_text("Turbine overhaul programs cover the engine reserve for the operator.\n")

    def test_concurrent_builds_embed_once(self):
        import threading
        import time
        from unittest import mock
        calls, done, active, peak = [], [], [0], [0]
        gate, barrier = threading.Lock(), threading.Barrier(5, timeout=10)

        def fake_build(root, quiet=False):
            with gate:
                calls.append(root)
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with gate:
                active[0] -= 1
                done.append(root)                 # the cache is current only once a build finishes
            return ("built",)

        built = []

        def run():
            barrier.wait()                        # every caller asks at the same moment
            built.append(mf.build_index(self.d, quiet=True))

        with fixture_env(), mock.patch.object(mf, "_build_index", side_effect=fake_build), \
                mock.patch.object(mf, "_current", side_effect=lambda root: ("cached",) if done else None):
            ts = [threading.Thread(target=run) for _ in range(5)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=20)
        self.assertEqual(len(calls), 1, "a waiting build re-embedded instead of reusing the first")
        self.assertEqual(peak[0], 1, "two builds of one store ran at once")
        self.assertEqual(sorted(built), [("built",)] + [("cached",)] * 4)

    def test_a_failed_build_releases_the_lock(self):
        import threading
        from unittest import mock
        with fixture_env(), mock.patch.object(mf, "_current", return_value=None):
            with mock.patch.object(mf, "_build_index", side_effect=RuntimeError("embedder down")):
                with self.assertRaises(RuntimeError):
                    mf.build_index(self.d, quiet=True)
            out = []
            with mock.patch.object(mf, "_build_index", return_value=("built",)):
                t = threading.Thread(target=lambda: out.append(mf.build_index(self.d, quiet=True)))
                t.start()
                t.join(timeout=10)
        self.assertEqual(out, [("built",)], "the lock outlived the build that raised")

    def test_no_lock_still_builds(self):
        from unittest import mock
        import corpus
        with fixture_env(), mock.patch.object(mf, "_current", return_value=None), \
                mock.patch.object(corpus, "ready_cache_dir", side_effect=RuntimeError("cache dir is a link")), \
                mock.patch.object(mf, "_build_index", return_value=("built",)) as b:
            self.assertEqual(mf.build_index(self.d, quiet=True), ("built",))
        b.assert_called_once()


if __name__ == "__main__":
    unittest.main()
