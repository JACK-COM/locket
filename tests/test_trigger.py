"""trigger's behaviour suite: rows, windows, caps, firing and the ask-once record."""
import json
import sys
import time
import unittest
from pathlib import Path

from tests.harness import fixture_env

import memscan  # noqa: E402
import paths  # noqa: E402
import trigger as tg  # noqa: E402

def _suite():
    """Offline, in a throwaway home: firing, silencing, blocking, the cap, the quiet
    directories and the lint."""
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        host, proj = Path(t) / "host", Path(t) / "proj"
        (host / "memory").mkdir(parents=True)
        (host / "memory/owner.md").write_text("The fact lives here.\n")
        (host / "RULINGS.csv").write_text("Ruling,Area\nNever do the thing.,practice\n")
        proj.mkdir()
        (proj / memscan.MANIFEST).write_text('{"triggers": "rules.json"}\n')
        transcript = Path(t) / "t.jsonl"
        transcript.write_text('{"skill": "loaded"}\n')
        (host / tg.TRIGGERS).write_text(json.dumps({
            "quiet_under": [str(Path(t) / "quiet")],
            "rows": [
                {"id": "git-reset", "tools": "Bash", "content": r"git\s+reset\s+--hard",
                 "question": "This discards uncommitted work.", "owner": "memory/owner.md"},
                {"id": "blocker", "tools": "Bash", "content": "rm -rf /", "unless": "RM-OK:",
                 "block": True, "everywhere": True, "question": "Refused; add RM-OK: <reason>."},
                {"id": "once", "tools": "Read", "content": r"\.pdf$", "unless_transcript": '"skill": "loaded"',
                 "question": "Sift it first."},
                {"id": "prompt", "tools": "UserPromptSubmit", "content": "hermes", "ignorecase": True,
                 "block": True, "question": "Load the skill."},
                {"id": "hermes-raw", "tools": "browser_navigate|patch", "content": "https|zzz", "question": "Raw name."},
            ]}))
        (proj / "rules.json").write_text(json.dumps({"cap": 1, "rows": [
            {"id": "proj-row", "tools": "Write", "content": "enum", "question": "Name, not value.",
             "owner": str(host / "RULINGS.csv#practice")},
            {"id": "second", "tools": "Write", "content": "x", "question": "Never runs."}]}))

        other = Path(t) / "elsewhere"       # in no store but the host's
        other.mkdir()

        def run(tool, inp, cwd=other, **kw):
            return tg.fire({"tool_name": tool, "tool_input": inp, "cwd": str(cwd), **kw}, host=host)

        msg, block = run("Bash", {"command": "git reset --hard"})
        assert "[git-reset]" in msg and not block, msg
        assert f"-> {tg._show(host / 'memory/owner.md')}" in msg, "the owner was not shown"
        assert run("Bash", {"command": "git status"})[0] == "", "fired on a non-matching command"
        assert run("Bash", {"command": "rm -rf / now"}) == ("[blocker] Refused; add RM-OK: <reason>.", True)
        assert run("Bash", {"command": "rm -rf / RM-OK: test"})[0] == "", "`unless` did not silence"
        assert "[once]" in run("Read", {"file_path": "a.pdf"})[0], "a transcript-gated row did not fire"
        assert run("Read", {"file_path": "a.pdf"}, transcript_path=str(transcript))[0] == "", \
            "a transcript-gated row fired after its pattern was on record"
        msg, block = run("Write", {"file_path": "x.py", "content": "enum"}, cwd=proj)
        assert "[proj-row]" in msg and "[over-cap]" in msg and "second" in msg and "[second]" not in msg, msg
        assert "RULINGS.csv#practice" in msg, "a CSV owner lost its label"
        assert "[over-cap]" in run("Bash", {"command": "ls"}, cwd=proj)[0], "the over-cap notice went quiet"
        assert "[proj-row]" not in run("Write", {"content": "enum"})[0], "a project row fired outside its store"
        quiet = Path(t) / "quiet/sub"
        quiet.mkdir(parents=True)
        assert run("Bash", {"command": "git reset --hard"}, cwd=quiet)[0] == "", "a row fired under quiet_under"
        assert run("Bash", {"command": "rm -rf /"}, cwd=quiet)[1], "an `everywhere` row stayed quiet"
        msg, block = tg.fire({"hook_event_name": "UserPromptSubmit", "prompt": "Ask Hermes", "cwd": str(other)},
                          host=host)
        assert "[prompt]" in msg and not block, "a prompt row matched wrongly, or blocked a prompt"

        # Hermes: its tool names map onto the rows', a prompt arrives on pre_llm_call
        def herm(tool, inp, **kw):
            return tg.fire({"hook_event_name": "pre_tool_call", "tool_name": tool, "tool_input": inp,
                         "cwd": str(other), "session_id": "s1", **kw}, host=host)
        assert "[git-reset]" in herm("terminal", {"command": "git reset --hard"})[0], "terminal is not Bash"
        assert herm("terminal", {"command": "rm -rf /"})[1], "a block row did not block on Hermes"
        assert "[once]" in herm("read_file", {"path": "a.pdf"})[0], "read_file is not Read, or path was lost"
        assert "[proj-row]" in tg.fire({"hook_event_name": "pre_tool_call", "tool_name": "write_file",
                                     "tool_input": {"path": "x.py", "content": "enum"}, "cwd": str(proj)},
                                    host=host)[0], "write_file is not Write"
        assert "[hermes-raw]" in herm("browser_navigate", {"url": "https://x"})[0], \
            "a Hermes tool with no Claude Code twin was not matched by its own name"
        assert "[hermes-raw]" in herm("patch", {"path": "x.md", "new_string": "zzz"})[0], \
            "a row naming a mapped Hermes tool by its own name did not fire"
        saved_homes = memscan.CLAUDE, memscan.HERMES
        memscan.CLAUDE, memscan.HERMES = other, host
        try:
            assert "[git-reset]" in tg.fire({"hook_event_name": "pre_tool_call", "tool_name": "terminal",
                                          "tool_input": {"command": "git reset --hard"}, "cwd": str(other)})[0], \
                "a Hermes call did not read Hermes's own rows"
        finally:
            memscan.CLAUDE, memscan.HERMES = saved_homes
        import sqlite3
        con = sqlite3.connect(host / "state.db")
        con.execute("create table messages (session_id text, role text, content text, tool_calls text)")
        con.execute("""insert into messages values ('s1', 'tool', '{"skill": "loaded"}', null)""")
        con.commit()
        con.close()
        assert herm("read_file", {"path": "a.pdf"})[0] == "", "unless_transcript did not read state.db"
        assert "[once]" in herm("read_file", {"path": "a.pdf"}, session_id="s2")[0], \
            "another session's messages silenced a row"
        llm = lambda m: {"hook_event_name": "pre_llm_call", "session_id": "s1", "cwd": str(other),
                         "extra": {"user_message": m}}
        msg, block = tg.fire(llm("Ask Hermes"), host=host)
        assert "[prompt]" in msg and not block, "a Hermes prompt row did not fire, or blocked"
        assert "[prompt]" in tg.fire(llm([{"type": "text", "text": "hermes?"}, {"type": "image_url"}]), host=host)[0], \
            "a multimodal prompt's text was not read"

        def hook_out(payload):
            out, err = io.StringIO(), io.StringIO()
            saved_out, saved_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = out, err
            try:
                rc = tg.hook(io.StringIO(json.dumps(payload)), host=host)
            finally:
                sys.stdout, sys.stderr = saved_out, saved_err
            return rc, out.getvalue()
        rc, out = hook_out(llm("hermes"))
        assert rc == 0 and "[prompt]" in json.loads(out)["context"], "a prompt row's context was not stdout JSON"
        saved = paths.PATHS; paths.PATHS = saved.replace(pending_dir=Path(t) / "pending")
        try:
            advise = {"hook_event_name": "pre_tool_call", "tool_name": "terminal", "session_id": "s9",
                      "tool_input": {"command": "git reset --hard"}, "cwd": str(other)}
            rc, out = hook_out(advise)
            sent = json.loads(out)
            assert rc == 2 and sent["action"] == "block" and sent["message"].startswith(memscan.ONCE_NOTE) \
                and "[git-reset]" in sent["message"], "an advisory row did not refuse the Hermes call once"
            other_call = dict(advise, tool_input={"command": "git reset --hard HEAD"})
            assert hook_out(other_call)[0] == 2 and hook_out(other_call)[0] == 2, \
                "the repeat passed before state.db showed the question"
            con = sqlite3.connect(host / "state.db")
            con.execute("alter table messages add column timestamp real")
            con.execute("insert into messages values ('s9', 'tool', ?, null, ?)",
                        (json.dumps({"error": json.loads(out)["message"]}), time.time()))
            con.commit()
            con.close()
            assert hook_out(dict(advise, tool_input={"command": "git reset --hard ."}))[0] == 2, \
                "a different call passed on another call's record"
            assert hook_out(advise) == (0, ""), "the host's state.db showing the question did not lift it"
            rc, out = hook_out(dict(advise, tool_input={"command": "rm -rf /"}))
            assert rc == 2 and memscan.ONCE_NOTE not in json.loads(out)["message"], \
                "a block row was worded as refuse-once"
        finally:
            paths.PATHS = saved

        # Codex: Claude Code's payload with a turn id; an edit is a patch, read as the Write it amounts to
        cx = lambda tool, inp, **kw: {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": inp,
                                      "turn_id": "t1", "cwd": str(proj), **kw}
        patch = "*** Begin Patch\n*** Add File: x.py\n+enum\n*** End Patch"
        assert "[proj-row]" in tg.fire(cx("apply_patch", {"command": patch}), host=host)[0], \
            "a native Codex patch was not read as a Write"
        assert "[proj-row]" in tg.fire(cx("Bash", {"command": f"apply_patch <<'P'\n{patch}\nP"}), host=host)[0], \
            "a Codex patch through the shell was not read as a Write"
        rc, out = hook_out(cx("Bash", {"command": "rm -rf /"}, cwd=str(other)))
        deny = json.loads(out)["hookSpecificOutput"]
        assert rc == 0 and deny["permissionDecision"] == "deny" and "[blocker]" in deny["permissionDecisionReason"], \
            "a block row did not refuse the Codex call in the deny shape"
        rc, out = hook_out(cx("Bash", {"command": "git reset --hard"}, cwd=str(other)))
        assert rc == 0 and "[git-reset]" in json.loads(out)["hookSpecificOutput"]["additionalContext"], \
            "an advisory row did not reach Codex as context"

        assert tg.findings(host) == [] and tg.findings(other) == [], tg.findings(host)
        bad = json.loads((host / tg.TRIGGERS).read_text())
        bad["rows"] += [{"id": "git-reset", "tools": "(", "content": "x", "question": "q",
                         "owner": "memory/gone.md", "blok": True},
                        {"id": "csv", "tools": "x", "content": "x", "question": "q", "owner": "RULINGS.csv#nope"}]
        (host / tg.TRIGGERS).write_text(json.dumps(bad))
        found = "\n".join(tg.findings(host))
        for want in ("unknown key 'blok'", "appears twice", "tools is not a regex", "missing file: memory/gone.md",
                     "no row of RULINGS.csv carries: nope"):
            assert want in found, f"the lint missed {want!r}:\n{found}"
        assert "over-cap" not in found and "cap of 1" in "\n".join(tg.findings(proj)), "the cap went unlinted"
        (proj / "rules.json").unlink()
        assert "named by the manifest but missing" in "\n".join(tg.findings(proj))

        out = io.StringIO()
        saved = sys.stdout
        sys.stdout = out
        try:
            rc = tg.hook(io.StringIO("not json"))
        finally:
            sys.stdout = saved
        assert rc == 0 and out.getvalue() == "", "a bad payload was not ignored"
    return 0


class TriggerBehaviour(unittest.TestCase):
    def test_behaviour(self):
        with fixture_env():
            _suite()


if __name__ == "__main__":
    unittest.main()
