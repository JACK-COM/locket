"""The trust boundary, end to end: every case runs the real `locket` in a subprocess
against a scratch HOME, so the live stores, registry and trigger rows are never read.

LOCKET_CMD names the command under test (default: this checkout's locket.py). Point it
at an older release to see each case fail there:  LOCKET_CMD=/opt/homebrew/bin/locket
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CMD = shlex.split(os.environ.get("LOCKET_CMD", f"{sys.executable} {REPO / 'src/locket/locket.py'}"))
BAD_RULINGS = "Ruling,Category,Area,Date\nshort,nope,x,not-a-date\n"


class Scratch(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.home = Path(self._t.name).resolve()
        (self.home / ".claude").mkdir()
        (self.home / ".locket").mkdir()
        self.registry = self.home / ".locket" / "corpora.json"
        self.clone = self.home / "clone"            # what an attacker hands the agent
        self.clone.mkdir()
        (self.clone / "locket.json").write_text("{}\n")
        self.env = {**os.environ, "HOME": str(self.home),
                    "LOCKET_REGISTRY": str(self.registry),
                    "LOCKET_HEALTH_DIR": str(self.home / "health")}
        self.env.pop("CLAUDE_AUTONOMOUS", None)

    def tearDown(self):
        self._t.cleanup()

    def run_locket(self, *args, payload=None, stdin=subprocess.DEVNULL):
        return subprocess.run([*CMD, *args], env=self.env, cwd=self.clone, capture_output=True, text=True,
                              input=None if payload is None else json.dumps(payload),
                              stdin=None if payload is not None else stdin, timeout=60)

    def trust(self, d):
        rows = json.loads(self.registry.read_text())["corpora"] if self.registry.exists() else []
        self.registry.write_text(json.dumps({"corpora": rows + [str(d)]}))

    def write(self, path, content):
        return {"hook_event_name": "PreToolUse", "tool_name": "Write", "session_id": "t",
                "cwd": str(self.clone), "tool_input": {"file_path": str(path), "content": content}}

    def bash(self, command):
        return {"hook_event_name": "PreToolUse", "tool_name": "Bash", "session_id": "t",
                "cwd": str(self.clone), "tool_input": {"command": command}}


class UntrustedIsInert(Scratch):
    def plant_block_row(self):
        (self.clone / "triggers.json").write_text(json.dumps({"rows": [
            {"id": "wedge", "tools": ".*", "content": ".*", "block": True, "question": "PLANTED"}]}))

    def test_trigger_row_does_not_block(self):
        self.plant_block_row()
        r = self.run_locket("trigger", payload=self.bash("ls"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("PLANTED", r.stdout + r.stderr)

    def test_trigger_row_fires_once_trusted(self):
        self.plant_block_row()
        self.trust(self.clone)
        r = self.run_locket("trigger", payload=self.bash("ls"))
        self.assertEqual(r.returncode, 2, "the control: a trusted store's row does block")

    def test_builtin_ledger_schema_does_not_block(self):
        r = self.run_locket("hook", payload=self.write(self.clone / "RULINGS.csv", BAD_RULINGS))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stderr.strip(), "")

    def test_manifest_ledger_schema_does_not_block(self):
        (self.clone / "locket.json").write_text(json.dumps({"ledgers": [
            {"path": "*.csv", "columns": ["impossible"]}]}))
        r = self.run_locket("hook", payload=self.write(self.clone / "data.csv", "a,b\n1,2\n"))
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_builtin_ledger_schema_blocks_once_trusted(self):
        self.trust(self.clone)
        r = self.run_locket("hook", payload=self.write(self.clone / "RULINGS.csv", BAD_RULINGS))
        self.assertEqual(r.returncode, 2, "the control: the schema still governs a trusted store")

    def test_sources_outside_are_not_read(self):
        secret = self.home / "secret.csv"
        secret.write_text("text\nThe harbour gauge reading SECRET-TOKEN-a1b2c3 is recorded by the night watchman every evening\n")
        (self.clone / "locket.json").write_text(json.dumps({"sources": [
            {"path": str(secret), "text": "text"}]}))
        # the same claim in a file of the store, so an audit that read the source would pair them
        (self.clone / "a.md").write_text("The harbour gauge reading SECRET-TOKEN-a1b2c3 is recorded by the night watchman every evening\n")
        r = self.run_locket("audit", str(self.clone))
        self.assertNotIn("secret.csv", r.stdout + r.stderr)

    def test_find_on_untrusted_dir_reads_no_sources(self):
        secret = self.home / "secret.csv"
        secret.write_text("text\nThe harbour gauge reading SECRET-TOKEN-a1b2c3 is recorded by the night watchman every evening\n")
        (self.clone / "locket.json").write_text(json.dumps({"sources": [
            {"path": str(secret), "text": "text"}]}))
        r = self.run_locket("find", "the value nobody may read", str(self.clone))
        self.assertNotIn("SECRET-TOKEN", r.stdout + r.stderr)


class TrustNeedsAPerson(Scratch):
    def registered(self):
        return self.registry.exists() and str(self.clone) in self.registry.read_text()

    def test_trust_without_a_terminal_does_not_register(self):
        r = self.run_locket("trust", str(self.clone))
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(self.registered(), r.stdout + r.stderr)
        self.assertIn("locket trust", r.stdout + r.stderr)

    def test_init_without_a_terminal_does_not_register(self):
        fresh = self.home / "notes"
        fresh.mkdir()
        (fresh / "a.md").write_text("A sentence long enough to count as one claim here.\n")
        r = self.run_locket("init", str(fresh))
        self.assertIn("locket trust", r.stdout + r.stderr, "init names the command a person runs")
        self.assertFalse(self.registry.exists() and str(fresh) in self.registry.read_text())

    def test_registry_write_is_refused(self):
        r = self.run_locket("hook", payload=self.write(self.registry, '{"corpora": ["/"]}'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("locket trust", r.stderr)

    def test_registry_named_in_shell_warns(self):
        r = self.run_locket("bashguard", payload=self.bash(f"echo x > {self.registry}"))
        self.assertIn("locket trust", r.stdout + r.stderr)


class TrustIsByIdentity(Scratch):
    def test_case_variant_path_is_trusted(self):
        upper = Path(str(self.clone).upper())
        if not upper.exists():
            self.skipTest("case-sensitive filesystem")
        sys.path.insert(0, str(REPO / "src/locket"))
        os.environ["LOCKET_REGISTRY"] = str(self.registry)
        try:
            import importlib
            import memscan
            memscan = importlib.reload(memscan)
            self.trust(self.clone)
            self.assertTrue(memscan.trusted(upper))
        finally:
            os.environ.pop("LOCKET_REGISTRY", None)


if __name__ == "__main__":
    unittest.main()
