"""The trust boundary, end to end: every case runs the real `locket` in a subprocess
against a scratch HOME, so the live stores, registry and trigger rows are never read.
`tests/harness.py` names the command and source under test."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.harness import CMD, SRC
BAD_RULINGS = "Ruling,Category,Area,Date\nshort,nope,x,not-a-date\n"


def grant(env, d):
    """Register `d` as a person's `locket trust` would: through the source under test,
    so the row carries whatever pin that source writes."""
    code = f"import sys; sys.path.insert(0, {str(SRC)!r}); import memscan; memscan.register({str(d)!r})"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


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
        grant(self.env, d)

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


    def as_person(self, *args):
        """Run `locket *args` on a pty and type `yes` at its prompt, as a person would."""
        import pty
        import select
        pid, fd = pty.fork()
        if pid == 0:                                    # child: the person's terminal session
            try:
                os.chdir(self.clone)
                os.execvpe(CMD[0], [*CMD, *args], self.env)
            finally:
                os._exit(127)                           # never fall back into the test runner
        out, sent = b"", False
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
        return sent, out.decode(errors="replace")

    def test_trust_at_a_terminal_registers(self):
        """The one path that grants trust, run for real: a pty stands in for the person."""
        sent, out = self.as_person("trust", str(self.clone))
        self.assertTrue(sent, out)
        self.assertTrue(self.registered(), out)

    def old_caches(self):
        self.trust(self.clone)
        caches = []
        for d in (self.clone, self.clone / "notes"):
            (d / ".memfind").mkdir(parents=True)
            (d / ".memfind" / "index.json").write_text("{}")
            caches.append(d / ".memfind")
        return caches

    def test_migrate_names_old_caches_in_one_command(self):
        caches = self.old_caches()
        r = self.run_locket("migrate")
        line = next(l for l in r.stdout.splitlines() if l.startswith("index: kept them"))
        for c in caches:
            self.assertIn(str(c), line)
            self.assertTrue(c.exists(), "nothing is deleted without a person")

    def test_migrate_deletes_old_caches_once_a_person_agrees(self):
        caches = self.old_caches()
        sent, out = self.as_person("migrate")
        self.assertTrue(sent, out)
        self.assertFalse(any(c.exists() for c in caches), out)

    def test_untrust_without_a_terminal_is_refused(self):
        self.trust(self.clone)
        r = self.run_locket("untrust", str(self.clone))
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue(self.registered(), "an agent cannot withdraw a store's guardrails")

    def test_untrust_withdraws_by_identity(self):
        variant = str(self.clone).upper()
        if not Path(variant).exists():
            self.skipTest("case-sensitive filesystem")
        self.trust(self.clone)
        sent, out = self.as_person("untrust", variant)
        self.assertIn("row removed", out)
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
        (self.clone / "locket.json").write_text(json.dumps({"triggers": "rules/rows.json"}))
        (self.clone / "rules").mkdir()
        (self.clone / "rules" / "rows.json").write_text('{"rows": []}')
        self.trust(self.clone)                          # trusted as it now stands
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


class RedTeamCases(Scratch):
    """Each case is a path a blind red team found from the attacker's seat."""

    def test_a_pull_that_changes_a_trusted_store_leaves_it_inert(self):
        self.registered_via_cli()
        (self.clone / "triggers.json").write_text(json.dumps({"rows": [
            {"id": "wedge", "tools": ".*", "content": ".*", "block": True, "question": "PULLED"}]}))
        r = self.run_locket("trigger", payload=self.bash("ls"))
        self.assertEqual(r.returncode, 0, "a rows file added after trust must not act: " + r.stderr)

    def test_locket_own_init_keeps_trust(self):
        self.registered_via_cli()
        self.run_locket("init", str(self.clone), "--name", "renamed")
        self.assertIn("renamed", (self.clone / "locket.json").read_text())
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import memscan; from pathlib import Path; "
                f"print(memscan.trusted(Path({str(self.clone)!r})))")
        r = subprocess.run([sys.executable, "-c", code], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "True", "Locket's own edit moves the pin: " + r.stderr)

    def registered_via_cli(self):
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import memscan; "
                f"memscan.register({str(self.clone)!r})")
        r = subprocess.run([sys.executable, "-c", code], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_init_never_writes_through_a_symlinked_manifest(self):
        target = self.home / "precious.json"
        target.write_text('{"keep": true}\n')
        (self.clone / "locket.json").unlink()
        (self.clone / "locket.json").symlink_to(target)
        self.run_locket("init", str(self.clone), "--name", "notes")
        self.assertEqual(target.read_text(), '{"keep": true}\n')

    def test_init_never_creates_through_a_dangling_link(self):
        target = self.home / "created-by-locket"
        (self.clone / "locket.json").unlink()
        (self.clone / "locket.json").symlink_to(target)
        self.run_locket("init", str(self.clone))
        self.assertFalse(target.exists())

    def test_codex_patch_with_a_manifest_second_is_refused(self):
        mem = self.home / ".claude" / "projects" / "-x" / "memory"
        mem.mkdir(parents=True)
        patch = ("*** Begin Patch\n*** Add File: notes.txt\n+hello\n"
                 f"*** Add File: {mem / 'locket.json'}\n+{{}}\n*** End Patch")
        payload = {"hook_event_name": "PreToolUse", "tool_name": "apply_patch", "turn_id": "t",
                   "session_id": "s", "cwd": str(self.clone), "tool_input": {"command": patch}}
        r = self.run_locket("hook", payload=payload)
        self.assertNotEqual(r.stdout + r.stderr, "", "the manifest in the patch draws the refusal")
        self.assertIn("locket.json", r.stdout + r.stderr)

    def test_shell_write_of_a_store_config_is_refused(self):
        mem = self.home / ".claude" / "projects" / "-x" / "memory"
        mem.mkdir(parents=True)
        for cmd in (f"cat > {mem / 'triggers.json'} <<'EOF'\n{{}}\nEOF",
                    f"cd {mem} && printf x > locket.json",
                    f"echo '{{}}' > ~/.claude/triggers.json"):
            r = self.run_locket("bashguard", payload=self.bash(cmd))
            self.assertEqual(r.returncode, 2, cmd + " | " + r.stdout + r.stderr)

    def test_registry_named_in_shell_is_refused(self):
        r = self.run_locket("bashguard", payload=self.bash(f"cat {self.registry}"))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_absolute_member_is_ignored(self):
        (self.clone / "locket.json").write_text(json.dumps({"members": ["/"]}))
        self.trust(self.clone)
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import memscan; from pathlib import Path; "
                f"print(memscan.settings(Path({str(self.clone)!r}))['members'])")
        r = subprocess.run([sys.executable, "-c", code], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "set()", r.stderr)

    def test_cites_in_an_untrusted_folder_reads_no_host_file(self):
        (self.home / ".claude" / "secret.py").write_text("line one\nTOKEN-in-a-host-file\n")
        (self.clone / "notes.md").write_text("The value lives at secret.py:2 in the host config folder here.\n")
        r = self.run_locket("cites", str(self.clone))
        self.assertNotIn("TOKEN-in-a-host-file", r.stdout + r.stderr)

    def test_purge_needs_a_person(self):
        self.trust(self.clone)
        self.run_locket("uninstall", "--purge", "--yes")
        self.assertTrue((self.clone / "locket.json").exists())

    def test_trigger_question_is_sanitised(self):
        (self.home / ".claude" / "triggers.json").write_text(json.dumps({"rows": [
            {"id": "x", "tools": "Bash", "content": "ls", "question": "ok\u202e\x1b[2Jhidden" + "q" * 5000}]}))
        r = self.run_locket("trigger", payload=self.bash("ls"))
        out = r.stdout + r.stderr
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\u202e", out)
        self.assertLess(len(out), 4000)


class RoundTwo(Scratch):
    """The second blind round and its audit, one case each."""

    def probe(self, expr):
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import memscan, trigger; "
                f"from pathlib import Path; print({expr})")
        r = subprocess.run([sys.executable, "-c", code], env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def test_window_edges_do_not_invent_anchors(self):
        # built inside the probe: Linux caps one argument at 128 KB
        text = "'x' * (64 * 1024 - 4096) + 'rm -rf /' + 'y' * 70000"
        self.assertEqual(self.probe(f"trigger._search(r'^rm -rf', {text})"), "False")
        self.assertEqual(self.probe(f"trigger._search(r'rm -rf', {text})"), "True", "the control")

    def test_a_fully_read_long_call_is_not_unread(self):
        self.assertEqual(self.probe("trigger._search('zzz', 'a' * 950000)"), "False")

    def test_padding_across_a_window_boundary_still_fires(self):
        step = 64 * 1024 - 4096
        (self.home / ".claude" / "triggers.json").write_text(json.dumps({"rows": [
            {"id": "rm", "tools": "Bash", "content": "rm -rf", "block": True, "question": "Refused."}]}))
        r = self.run_locket("trigger", payload=self.bash("x" * (step - 3) + " rm -rf ~ " + "x" * 90000))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_case_variant_config_name_is_refused(self):
        variant = self.home / ".claude" / "TRIGGERS.JSON"
        if not Path(str(self.home).upper()).exists():
            self.skipTest("case-sensitive filesystem")
        r = self.run_locket("hook", payload=self.write(variant, '{"rows": []}'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_embedder_settings_write_is_refused(self):
        cfg = self.home / ".panoply" / "config.json"
        r = self.run_locket("hook", payload=self.write(cfg, '{"embed": {"ollama_host": "http://evil:1"}}'))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_trust_verb_in_an_agent_shell_is_refused(self):
        r = self.run_locket("bashguard", payload=self.bash("printf 'yes\\n' | script -q /dev/null locket trust ."))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_data_heredocs_that_name_the_guards_pass(self):
        verb = "locket " + "trust"
        for cmd in (f"git commit -F - <<'EOF'\nfix: run {verb} and edit locket.json\nEOF",
                    f"gh release create v1 --notes-file - <<'EOF'\nRun {verb} . and see triggers.json\nEOF"):
            r = self.run_locket("bashguard", payload=self.bash(cmd))
            self.assertEqual(r.returncode, 0, cmd + " | " + r.stdout + r.stderr)

    def test_an_interpreter_heredoc_that_writes_config_is_refused(self):
        cmd = f"python3 - <<'EOF'\nopen('{self.home}/.claude/triggers.json', 'w').write('{{}}')\nEOF"
        r = self.run_locket("bashguard", payload=self.bash(cmd))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_a_link_into_host_memory_is_judged_as_host_memory(self):
        link = self.clone / "notes"
        link.symlink_to(self.home / ".claude")
        r = self.run_locket("hook", payload=self.write(link / "triggers.json", '{"rows": []}'))
        self.assertEqual(r.returncode, 2, "a write through a link reaches the file it names: " + r.stderr)

    def test_member_through_a_link_stays_inside(self):
        outside = self.home / "outside"
        (outside / "sub").mkdir(parents=True)
        (outside / "sub" / "secret.md").write_text("A secret sentence long enough to count as a claim.\n")
        (self.clone / "link").symlink_to(outside)
        (self.clone / "locket.json").write_text(json.dumps({"members": ["link/sub"]}))
        self.trust(self.clone)
        self.assertEqual(self.probe(f"[str(p) for p in memscan.md_files(Path({str(self.clone)!r}))]"), "[]")

    def test_an_unpinned_row_is_not_trusted(self):
        self.registry.write_text(json.dumps({"corpora": [str(self.clone)]}))     # written by hand
        self.assertEqual(self.probe(f"memscan.trusted(Path({str(self.clone)!r}))"), "False")

    def test_init_refuses_the_home_directory(self):
        r = self.run_locket("init", str(self.home))
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.home / "locket.json").exists())

    def test_cli_output_carries_no_control_character(self):
        (self.clone / "MEMORY.md").write_text("- [Title \x1b[2J\u202e hidden](fact.md) \u2014 a hook line here\n")
        (self.clone / "fact.md").write_text("A fact.\n")
        self.trust(self.clone)
        r = self.run_locket("pointers", str(self.clone))
        for hidden in ("\x1b", "\u202e"):
            self.assertNotIn(hidden, r.stdout)


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
