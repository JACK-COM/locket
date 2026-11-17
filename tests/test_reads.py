"""What a store's walk reads, and where the index lives: a trusted store is read only
through plain files it holds itself, and its cache sits outside it. Each probe runs the
source named by LOCKET_SRC (default: this checkout) in a subprocess with a scratch HOME.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_trust import SRC

CLAIM = "The harbour gauge at the north pier is read at dawn and at dusk by the night watchman.\n"


class Reads(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.home = Path(self._t.name).resolve()
        (self.home / ".claude").mkdir()
        (self.home / ".locket").mkdir()
        self.store = self.home / "store"
        self.store.mkdir()
        (self.store / "locket.json").write_text("{}\n")
        (self.store / "own.md").write_text("A sentence the store itself holds about propeller pitch.\n")
        self.outside = self.home / "outside"
        self.outside.mkdir()
        (self.outside / "secret.md").write_text(CLAIM)
        reg = self.home / ".locket" / "corpora.json"
        reg.write_text(json.dumps({"corpora": [str(self.store)]}))
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith("LOCKET_") and k not in ("CODEX_HOME", "HERMES_HOME")}
        self.env.update(HOME=str(self.home), LOCKET_REGISTRY=str(reg),
                        LOCKET_HEALTH_DIR=str(self.home / "health"))

    def tearDown(self):
        self._t.cleanup()

    def probe(self, body):
        code = (f"import sys, json; sys.path.insert(0, {str(SRC)!r}); import memscan\n"
                f"from pathlib import Path\nroot = Path({str(self.store)!r})\n{body}")
        r = subprocess.run([sys.executable, "-c", code], env=self.env, capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def read_keys(self):
        return json.loads(self.probe("print(json.dumps(sorted(memscan.corpus(root, csv=False))))"))

    def test_symlinked_file_is_not_read(self):
        (self.store / "link.md").symlink_to(self.outside / "secret.md")
        self.assertNotIn("link.md", self.read_keys())

    def test_symlinked_member_directory_is_not_walked(self):
        (self.store / "locket.json").write_text(json.dumps({"members": ["notes"]}))
        (self.store / "notes").symlink_to(self.outside)
        self.assertFalse(any("secret" in k for k in self.read_keys()))

    def test_hardlinked_file_is_not_read(self):
        os.link(self.outside / "secret.md", self.store / "hard.md")
        self.assertNotIn("hard.md", self.read_keys())

    def test_fifo_is_not_read(self):
        os.mkfifo(self.store / "pipe.md")
        self.assertNotIn("pipe.md", self.read_keys())     # returns at all: a FIFO read would hang

    def test_plain_file_is_read(self):
        self.assertIn("own.md", self.read_keys(), "the control: the walk still reads the store")

    def test_planted_in_store_cache_is_never_read(self):
        planted = self.store / ".memfind"
        planted.mkdir()
        model = self.probe("import memfind\nprint(memfind.tag('ollama'))").strip()
        (planted / "index.json").write_text(json.dumps({     # a cache 0.11 would have served
            "root": str(self.store), "model": model, "dim": 1,
            "fingerprint": {}, "items": [["own.md", "PLANTED INSTRUCTION"]]}))
        (planted / "index.vec").write_bytes(b"\0\0\x80\x3f")
        out = self.probe("import memfind\nprint(memfind.load_cached(root))")
        self.assertNotIn("PLANTED", out)

    def test_cache_lives_outside_the_store(self):
        out = self.probe("import memfind\nprint(memfind.cache_dir(root))")
        cache = Path(out.strip())
        self.assertNotIn(str(self.store), str(cache))
        self.assertIn(str(self.home / ".locket"), str(cache))

    def test_symlinked_cache_dir_is_refused(self):
        out = self.probe("import memfind\nprint(memfind.cache_dir(root))")
        cache = Path(out.strip())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.symlink_to(self.outside)
        r = self.probe("import memfind, array\n"
                       "try:\n    memfind._write_cache(root, array.array('f', [1.0]), {'items': []})\n"
                       "    print('WROTE')\nexcept RuntimeError as e:\n    print('REFUSED')")
        self.assertIn("REFUSED", r)
        self.assertEqual(sorted(p.name for p in self.outside.iterdir()), ["secret.md"])


if __name__ == "__main__":
    unittest.main()
