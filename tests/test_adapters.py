"""Host adapters: Hermes and Codex payloads in Claude Code's vocabulary, and each host's reply."""
import contextlib
import io
import json
import os
import sqlite3
import time
import unittest
from unittest import mock

from tests.harness import StoreCase

import adapters  # noqa: E402
import corpora  # noqa: E402
import gate  # noqa: E402
import hosts  # noqa: E402
import ledger  # noqa: E402
import paths  # noqa: E402
from adapters import codex, hermes  # noqa: E402


class Hermes(StoreCase):
    def test_vocabulary(self):
        # --- host adapters: Hermes speaks the same payload in its own vocabulary ---
        h = adapters.normalise({"hook_event_name": "pre_tool_call", "tool_name": "write_file",
                       "tool_input": {"path": "/x/MEMORY.md", "content": "c"},
                       "session_id": "s1"})
        assert h["_host"] == "hermes" and h["tool_name"] == "Write", h
        assert h["tool_input"]["file_path"] == "/x/MEMORY.md", h
        h = adapters.normalise({"hook_event_name": "pre_tool_call", "tool_name": "terminal",
                       "args": {"command": "ls"}})
        assert h["tool_name"] == "Bash" and h["tool_input"]["command"] == "ls", \
            "Hermes may carry the arguments under `args`"
        c = adapters.normalise({"hook_event_name": "PreToolUse", "tool_name": "Edit",
                       "tool_input": {"file_path": "/x/a.md", "new_string": "n"}})
        assert c["_host"] == "claude-code" and c["tool_name"] == "Edit", c
        m = adapters.normalise({"hook_event_name": "pre_tool_call", "tool_name": "memory",
                       "tool_input": {"action": "add", "target": "user", "content": "likes tea"}})
        assert m["tool_name"] == "Edit" and m["tool_input"]["file_path"].endswith("memories/USER.md"), m
        assert m["tool_input"]["content"] == "likes tea" and m["tool_input"]["_append"], m
        assert gate.added_text({}, "old body\nlikes tea", "old body\n") == " \nlikes tea", \
            "an append reaches the scorer as the entry alone"
        m = adapters.normalise({"hook_event_name": "pre_tool_call", "tool_name": "memory",
                       "tool_input": {"operations": [
                           {"action": "replace", "target": "memory", "old_text": "a", "content": "b"},
                           {"action": "remove", "target": "memory", "old_text": "z"},
                           {"action": "add", "target": "memory", "new_text": "c"}]}})
        assert m["tool_input"]["file_path"].endswith("memories/MEMORY.md"), m
        assert m["tool_input"]["new_string"] == "b\nc" and m["tool_input"]["old_string"] == "a\nz", m
        assert "MEMORY.md" not in corpora.settings(hosts.hermes())["index_files"], \
            "Hermes's MEMORY.md is its whole memory and must not inherit the index exemption"
        r = adapters.normalise({"hook_event_name": "pre_tool_call", "tool_name": "write_file",
                       "tool_input": {"path": "MEMORY.md", "content": "c"},
                       "cwd": str(hosts.claude() / "memory")})
        assert r["tool_input"]["file_path"] == str(hosts.claude() / "memory" / "MEMORY.md"), \
            "a relative path resolves against the payload cwd, or the write escapes the gate"
        assert gate.session_asked({"_host": "hermes"}), "no session source at all fails open"

    def test_a_nudge_refuses_once(self):
        d = self.tmp
        # a Hermes nudge refuses the call once, and the repeat passes once state.db shows it
        saved = paths.PATHS; paths.PATHS = saved.replace(pending_dir=d / "pending")
        try:
            db = d / "shown.db"
            con = sqlite3.connect(db)
            con.execute("create table messages (session_id text, role text, content text, timestamp real)")
            con.commit()
            store = lambda text: (con.execute("insert into messages values ('s1','tool',?,?)",
                                              (json.dumps({"error": text}), time.time())), con.commit())
            call = lambda cmd: {"_host": "hermes", "session_id": "s1", "tool_name": "Bash",
                                "tool_input": {"command": cmd}}
            first = hermes.refuse_once(call("a"), "nudge A", "hook", db)
            assert first and first.startswith(hermes.ONCE_NOTE) and first.endswith("nudge A"), first
            assert hermes.refuse_once(call("a"), "nudge A", "hook", db) == first, "an unseen refusal passed"
            assert hermes.refuse_once(call("a"), "nudge A", "hook", db) is None, "the refusal cap did not hold"
            assert hermes.refuse_once(call("a"), "nudge A", "hook", db) == first, "a passed call stayed passed"
            store(first)
            assert hermes.refuse_once(call("a"), "nudge A", "hook", db) is None, "a delivered refusal refused again"
            # two modes refuse one call; Hermes shows only the first block
            b_hook = hermes.refuse_once(call("b"), "from hook", "hook", db)
            b_trig = hermes.refuse_once(call("b"), "from trigger", "trigger", db)
            assert b_hook and b_trig, "a second mode's refusal was swallowed"
            store(b_hook)
            assert hermes.refuse_once(call("b"), "from hook", "hook", db) is None, "the shown mode refused again"
            assert hermes.refuse_once(call("b"), "from trigger", "trigger", db) == b_trig, \
                "a mode whose message Hermes never showed let the retry pass"
            assert hermes.refuse_once(call("f"), "m1", "hook", db) and hermes.refuse_once(call("f"), "m2", "trigger", db) and \
                hermes.refuse_once(call("f"), "m2", "trigger", db), "one mode's refusal counted against another's cap"
            assert hermes.refuse_once(dict(call("c"), session_id="s2"), "n", "hook", db) and \
                hermes.refuse_once(dict(call("c"), session_id="s2"), "n", "hook", db), "another session's result counted"
            nodb = d / "none.db"
            assert hermes.refuse_once(call("d"), "n", "hook", nodb) and hermes.refuse_once(call("d"), "n", "hook", nodb) and \
                hermes.refuse_once(call("d"), "n", "hook", nodb) is None, "a state.db that cannot say must fall to the cap"
            con.close()
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                rc = adapters.emit(call("e"), ledger.LEAD_WARNING + " more", False, db=db)
            sent = json.loads(out.getvalue())
            assert rc == 2 and sent["action"] == "block", "a Hermes nudge did not refuse the call"
            assert ledger.LEAD_BLOCKING in sent["message"] and ledger.LEAD_WARNING not in sent["message"], \
                "a refused write was described as landed"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert adapters.emit({"_host": "claude-code"}, "advice", False) == 0
            assert "additionalContext" in out.getvalue(), "Claude Code lost its non-blocking channel"
        finally:
            paths.PATHS = saved

    def test_the_question_gate_reads_state_db(self):
        d = self.tmp
        # the 2e gate reads Hermes's state.db the way it reads Claude Code's transcript
        db = d / "state.db"
        con = sqlite3.connect(db)
        con.execute("create table messages (session_id text, role text, tool_calls text)")
        call = lambda cmd: json.dumps([{"function": {"name": "terminal",
                                                     "arguments": json.dumps({"command": cmd})}}])
        con.execute("insert into messages values ('asked','assistant',?)",
                    (call("python3 ~/.claude/scripts/memfind.py 'q' council"),))
        con.execute("insert into messages values ('named','assistant',?)",
                    (call("grep -n memfind.py memscan.py"),))
        con.commit(); con.close()
        assert hermes.memfind_asked_db(db, "asked"), "a terminal call that RUNS memfind discharges"
        assert not hermes.memfind_asked_db(db, "named"), "one that only names it does not"
        assert not hermes.memfind_asked_db(db, "silent"), "a session with no such call has not asked"
        assert hermes.memfind_asked_db(d / "missing.db", "asked"), "no database fails open"


class Codex(StoreCase):
    def test_codex(self):
        d, COUNCIL = self.tmp, self.council
        # --- Codex: Claude Code's payload, a patch for every edit, a rollout transcript ---
        rollout = str(d / "rollout-2026-09-25T18-00-40-x.jsonl")
        via_codex = lambda tool, ti, **kw: adapters.normalise({"hook_event_name": "PreToolUse", "tool_name": tool,
                                                  "tool_input": ti, "turn_id": "t1", "cwd": str(COUNCIL),
                                                  "transcript_path": rollout, **kw})
        add = "*** Begin Patch\n*** Add File: notes.md\n+first line\n+second line\n*** End Patch"
        upd = ("*** Begin Patch\n*** Update File: /w/a.py\n@@\n-x = 1\n+x = 2\n"
               "*** Update File: MEMORY.md\n@@\n old\n-gone\n+new fact\n*** End Patch")
        w = via_codex("apply_patch", {"command": add})
        assert w["_host"] == "codex" and w["tool_name"] == "Write", w
        assert w["tool_input"] == {"file_path": str(COUNCIL / "notes.md"),
                                   "content": "first line\nsecond line"}, w
        e = via_codex("Bash", {"command": f"apply_patch <<'PATCH'\n{upd}\nPATCH"})
        assert e["tool_name"] == "Edit", "a patch sent through the shell (no native tool) went unseen"
        assert e["tool_input"]["file_path"] == str(COUNCIL / "MEMORY.md"), \
            "the file inside a store is the one judged, whatever its place in the patch"
        assert e["tool_input"]["new_string"] == "new fact" and e["tool_input"]["old_string"] == "gone", e
        assert codex.parse_patch(add.replace("\n", "\r\n")) == [("add", "notes.md", "first line\nsecond line", "")], \
            "a CRLF patch parsed to nothing, which passes the gate silently"
        both = "*** Begin Patch\n*** Add File: MEMORY.md\n+m\n*** Add File: /w/l.csv\n+r\n*** End Patch"
        assert via_codex("apply_patch", {"command": both})["tool_input"]["file_path"] == "/w/l.csv", \
            "a ledger in the patch was not the file judged"
        assert codex.parse_patch("*** Begin Patch\n*** Update File: a.md\n*** Move to: b.md\n+x\n*** End Patch") \
            == [("update", "b.md", "x", "")], "a move is judged at its destination"
        assert via_codex("Bash", {"command": "echo apply_patch"})["tool_name"] == "Bash", \
            "a shell call that only names apply_patch stays a shell call"
        assert via_codex("apply_patch", {"command": "*** Begin Patch\n*** Delete File: a.md\n*** End Patch"}
                     )["tool_name"] == "apply_patch", "a patch that adds nothing stays opaque"
        assert adapters.normalise({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {
            "command": f"apply_patch <<'P'\n{add}\nP"}})["tool_name"] == "Bash", \
            "Claude Code's shell is never read as a patch"
        assert adapters.host_of({"hook_event_name": "PreToolUse", "transcript_path": rollout}) == "codex"
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = adapters.emit({"_host": "codex"}, "refused", True)
        deny = json.loads(out.getvalue())["hookSpecificOutput"]
        assert rc == 0 and deny["permissionDecision"] == "deny" and deny["permissionDecisionReason"] == "refused", \
            "a Codex block must be the deny shape Codex reads"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert adapters.emit({"_host": "codex"}, "advice", False) == 0
        assert json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] == "advice"
        # the rollout: a call is a function_call whose `arguments` is a JSON string
        fc = lambda args: json.dumps({"type": "response_item", "payload": {
            "type": "function_call", "name": "exec_command", "arguments": json.dumps(args)}})
        out_line = json.dumps({"type": "response_item", "payload": {
            "type": "function_call_output", "output": "locket find 'q' council"}})
        asked, named, argv = (d / f"rollout-{n}.jsonl" for n in ("asked", "named", "argv"))
        asked.write_text(fc({"cmd": "locket find 'a fact' council", "workdir": "/w"}) + "\n")
        named.write_text(fc({"cmd": "grep -n locket README.md"}) + "\n" + out_line + "\n")
        argv.write_text(fc({"command": ["bash", "-lc", "locket find 'a fact' council"]}) + "\n")
        assert gate.memfind_asked(str(asked)), "a Codex shell call that RUNS locket find discharges"
        assert not gate.memfind_asked(str(named)), "a call that names it, or a result quoting it, does not"
        assert gate.memfind_asked(str(argv)), "the older shell tool's argv form is read through bash -lc"


class Uninstall(StoreCase):
    COUNCIL = False

    def test_desktop_keeps_an_entry_it_did_not_write(self):
        from adapters import desktop
        cfg = self.tmp / "claude_desktop_config.json"
        theirs = {"mcpServers": {"locket": {"command": "/opt/x/run", "args": ["--mine"]}}, "note": "café"}
        cfg.write_text(json.dumps(theirs, ensure_ascii=False))
        with mock.patch.object(desktop, "CONFIG", cfg):
            r = desktop.removal()
        self.assertEqual((r.shown, r.apply), ([], None))
        self.assertTrue(r.left and "left as it is" in r.left[0], r.left)

    def test_desktop_removes_its_own_entry_whole(self):
        from adapters import desktop
        cfg = self.tmp / "claude_desktop_config.json"
        ours = {"mcpServers": {"locket": {"command": "/opt/homebrew/bin/locket", "args": ["mcp"]},
                               "other": {"command": "x"}}, "note": "café"}
        cfg.write_text(json.dumps(ours, ensure_ascii=False))
        with mock.patch.object(desktop, "CONFIG", cfg), contextlib.redirect_stdout(io.StringIO()):
            r = desktop.removal()
            r.apply()
        self.assertEqual(json.loads(cfg.read_text()), {"mcpServers": {"other": {"command": "x"}}, "note": "café"})
        self.assertIn("café", cfg.read_text(), "non-ASCII escaped on the way out")

    def test_install_leaves_another_tools_link(self):
        import install
        bin_ = self.tmp / "bin"; bin_.mkdir()
        other = self.tmp / "other-tool"; other.write_text("#!/bin/sh\n")
        (bin_ / "locket").symlink_to(other)
        with mock.patch.object(install, "BIN", bin_), contextlib.redirect_stdout(io.StringIO()) as out:
            rc = install._install_link()
        self.assertEqual(rc, 1, out.getvalue())
        self.assertEqual(os.readlink(bin_ / "locket"), str(other))


if __name__ == "__main__":
    unittest.main()
