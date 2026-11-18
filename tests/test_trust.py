"""The trust boundary, end to end: every case runs the real `locket` in a subprocess
against a scratch HOME, so the live stores, registry and trigger rows are never read.

LOCKET_CMD names the command under test (default: this checkout's locket.py), and
LOCKET_SRC the source the in-process probes import. Point both at an older release to
see each case fail there:  LOCKET_CMD=/opt/homebrew/bin/locket LOCKET_SRC=<its src/locket>
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
SRC = Path(os.environ.get("LOCKET_SRC", REPO / "src/locket"))     # the in-process probes' source
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
        for k in [k for k in self.env if k.startswith("LOCKET_") and k not in
                  ("LOCKET_REGISTRY", "LOCKET_HEALTH_DIR")] + ["CLAUDE_AUTONOMOUS", "CODEX_HOME", "HERMES_HOME"]:
            self.env.pop(k, None)

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
    def registered(self):
        return self.registry.exists() and str(self.clone) in self.registry.read_text()

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


    def test_trust_at_a_terminal_registers(self):
        """The one path that grants trust, run for real: a pty stands in for the person."""
        import pty
        import select
        pid, fd = pty.fork()
        if pid == 0:                                    # child: the person's terminal session
            try:
                os.chdir(self.clone)
                os.execvpe(CMD[0], [*CMD, "trust", str(self.clone)], self.env)
            finally:
                os._exit(127)                           # never fall back into the test runner
        out = b""
        sent = False
        while True:
            ready, _, _ = select.select([fd], [], [], 30)
            if not ready:
                break
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            if not sent and b"Type yes" in out:
                os.write(fd, b"yes\n")
                sent = True
        os.waitpid(pid, 0)
        self.assertTrue(sent, out.decode(errors="replace"))
        self.assertTrue(self.registered(), out.decode(errors="replace"))

    def test_untrust_withdraws_by_identity(self):
        variant = str(self.clone).upper()
        if not Path(variant).exists():
            self.skipTest("case-sensitive filesystem")
        self.trust(self.clone)
        r = self.run_locket("untrust", variant)
        self.assertIn("row removed", r.stdout, r.stderr)
        self.assertFalse(self.registered())

    def test_registry_write_by_case_variant_is_refused(self):
        self.registry.write_text('{"corpora": []}')
        variant = Path(str(self.registry).upper())
        if not variant.exists():
            self.skipTest("case-sensitive filesystem")
        r = self.run_locket("hook", payload=self.write(variant, '{"corpora": ["/"]}'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)


class ConfigIsAPersonsToWrite(Scratch):
    def test_manifest_write_into_a_trusted_store_is_refused(self):
        mem = self.home / ".claude" / "projects" / "-x" / "memory"     # trusted by convention
        mem.mkdir(parents=True)
        r = self.run_locket("hook", payload=self.write(mem / "locket.json",
                                                       json.dumps({"triggers": str(self.clone / "t.json")})))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_host_triggers_write_is_refused(self):
        r = self.run_locket("hook", payload=self.write(self.home / ".claude" / "triggers.json", '{"rows": []}'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_named_rows_file_write_is_refused(self):
        self.trust(self.clone)
        (self.clone / "locket.json").write_text(json.dumps({"triggers": "rules/rows.json"}))
        (self.clone / "rules").mkdir()
        (self.clone / "rules" / "rows.json").write_text('{"rows": []}')
        r = self.run_locket("hook", payload=self.write(self.clone / "rules" / "rows.json", '{"rows": []}'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_json_elsewhere_is_not_refused(self):
        self.trust(self.clone)
        r = self.run_locket("hook", payload=self.write(self.clone / "package.json", "{}"))
        self.assertEqual(r.returncode, 0, "the control: an ordinary JSON file passes")

    def test_remote_embedder_host_needs_a_person(self):
        r = self.run_locket("configure", "embedder", "--ollama-host", "http://attacker.example:11434", "--no-check")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        cfg = self.home / ".panoply" / "config.json"
        self.assertFalse(cfg.exists() and "attacker.example" in cfg.read_text())

    def test_loopback_embedder_host_is_set(self):
        r = self.run_locket("configure", "embedder", "--ollama-host", "http://127.0.0.1:11434", "--no-check")
        self.assertEqual(r.returncode, 0, "the control: this machine's own host needs no one: " + r.stderr)


class TriggerPadding(Scratch):
    def plant(self):
        (self.home / ".claude" / "triggers.json").write_text(json.dumps({"rows": [
            {"id": "rm", "tools": "Bash", "content": "rm -rf", "block": True, "question": "Refused."}]}))

    def test_padding_before_the_match_still_fires(self):
        self.plant()
        r = self.run_locket("trigger", payload=self.bash("x" * 70000 + "; rm -rf ~"))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_padding_around_the_match_still_fires(self):
        self.plant()
        r = self.run_locket("trigger", payload=self.bash("x" * 40000 + " rm -rf ~ " + "x" * 40000))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)


class TrustIsByIdentity(Scratch):
    def probe(self, expr):
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import memscan; "
                f"from pathlib import Path; print({expr})")
        r = subprocess.run([sys.executable, "-c", code], env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def test_case_variant_path_is_trusted(self):
        upper = str(self.clone).upper()
        if not Path(upper).exists():
            self.skipTest("case-sensitive filesystem")
        self.trust(self.clone)
        self.assertEqual(self.probe(f"memscan.trusted(Path({upper!r}))"), "True")

    def test_untrusted_nested_manifest_is_no_store(self):
        outer = self.home / "outer"
        (outer / "inner").mkdir(parents=True)
        (outer / "locket.json").write_text("{}\n")
        (outer / "inner" / "locket.json").write_text("{}\n")
        self.trust(outer)
        self.assertEqual(self.probe(f"memscan.corpus_for(Path({str(outer / 'inner' / 'a.md')!r}))"), "None")
        self.assertEqual(self.probe(f"memscan.corpus_for(Path({str(outer / 'a.md')!r}))"), str(outer))

    def test_legacy_sources_file_is_inert_when_untrusted(self):
        secret = self.home / "secret.csv"
        secret.write_text("text\nA row from outside the folder that nobody trusted it to read\n")
        (self.clone / "reference-sources.json").write_text(json.dumps([{"path": str(secret), "text": "text"}]))
        self.assertEqual(self.probe(f"memscan.csv_sources(Path({str(self.clone)!r}))"), "[]")

    def test_project_memory_is_trusted_by_convention(self):
        mem = self.home / ".claude" / "projects" / "-x" / "memory"
        mem.mkdir(parents=True)
        self.assertEqual(self.probe(f"memscan.trusted(Path({str(mem)!r}))"), "True")

    def test_untrusted_store_is_found_from_a_session_record(self):
        import re
        proj = self.home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(self.clone))
        proj.mkdir(parents=True)
        found = self.probe("[str(p) for p in memscan.untrusted_stores()]")
        self.assertIn(str(self.clone), found)


if __name__ == "__main__":
    unittest.main()
