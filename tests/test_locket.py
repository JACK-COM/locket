"""The locket front end: doctor's probes, dead-hook detection, help routing, install and uninstall."""
import json
import os
import shlex
import sys
import unittest
from pathlib import Path

from tests.harness import fixture_env

import locket as lk  # noqa: E402
import install  # noqa: E402
import probes  # noqa: E402
import cli  # noqa: E402
from adapters import codex, desktop, hermes  # noqa: E402
from adapters import claude_code  # noqa: E402
import memscan  # noqa: E402
import corpora  # noqa: E402
import health  # noqa: E402
import manifest  # noqa: E402

def _uninstall():
    """`uninstall --yes` in throwaway homes, in a child process because Locket's paths are
    fixed from HOME at import: the shared venv goes only with the last piece on PATH, at
    either location, and an older ~/.locket/venv beside ~/.panoply/venv goes with ~/.locket."""
    import subprocess, tempfile
    cases = (  # venvs made, another piece on PATH, venvs that must survive
        ((".panoply/venv",), False, ()),
        ((".panoply/venv",), True, (".panoply/venv",)),
        ((".locket/venv",), False, ()),
        ((".locket/venv",), True, (".locket/venv",)),
        ((".panoply/venv", ".locket/venv"), True, (".panoply/venv",)),
    )
    for made, other, kept in cases:
        with tempfile.TemporaryDirectory() as t:
            h = Path(t)
            for v in made:
                (h / v).mkdir(parents=True)
            (h / "bin").mkdir()
            if other:
                (h / "bin/grille").write_text("#!/bin/sh\n")
                (h / "bin/grille").chmod(0o755)
            r = subprocess.run([sys.executable, str(lk.HERE / "locket.py"), "uninstall", "--yes"],
                               env={"HOME": t, "PATH": str(h / "bin")}, capture_output=True, text=True)
            left = tuple(v for v in (".panoply/venv", ".locket/venv") if (h / v).is_dir())
            assert r.returncode == 0 and left == kept, \
                f"made {made}, grille on PATH {other}: left {left}, expected {kept}\n{r.stdout}{r.stderr}"
            assert kept or not (h / ".panoply").exists(), "an empty ~/.panoply outlived its venv"
    with tempfile.TemporaryDirectory() as t:     # doctor runs to its summary, rows file and all
        (Path(t) / ".claude").mkdir()
        (Path(t) / ".claude/triggers.json").write_text('{"rows": []}\n')
        r = subprocess.run([sys.executable, str(lk.HERE / "locket.py"), "doctor"], capture_output=True, text=True,
                           env={"HOME": t, "PATH": "/usr/bin:/bin", "MEMFIND_NO_AUTOSTART": "1"})
        assert " checks: " in r.stdout and "Traceback" not in r.stderr, f"doctor did not finish:\n{r.stderr[-600:]}"
        assert "triggers" in r.stdout, "doctor skipped a host rows file"
        assert "usage" in r.stdout and "never recorded" in r.stdout, "doctor skipped the usage row"
    run = lambda t, *a: subprocess.run([sys.executable, str(lk.HERE / "locket.py"), *a], capture_output=True, text=True,
                                       env={"HOME": t, "PATH": "/usr/bin:/bin", "MEMFIND_NO_AUTOSTART": "1"})
    for purge in (False, True):     # the ledger survives uninstall, the SessionEnd hook does not
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / ".claude").mkdir()
            (Path(t) / ".locket").mkdir()
            (Path(t) / ".locket/usage.csv").write_text("date\n")
            (Path(t) / ".panoply").mkdir()
            (Path(t) / ".panoply/config.json").write_text('{"locket": {"embed": {"model": "m"}}}')
            r = run(t, "install", "--hooks")
            hooks = json.loads((Path(t) / ".claude/settings.json").read_text())["hooks"]
            assert claude_code.event_command(hooks.get("SessionEnd"), "usage hook"), f"install registered no usage hook:\n{r.stdout}"
            r = run(t, "uninstall", "--yes", *(["--purge"] if purge else []))
            hooks = json.loads((Path(t) / ".claude/settings.json").read_text())["hooks"]
            assert not claude_code.event_command(hooks.get("SessionEnd"), "usage hook"), "uninstall left the usage hook"
            assert (Path(t) / ".locket/usage.csv").is_file() != purge, \
                f"uninstall {'kept' if purge else 'removed'} the usage ledger with purge={purge}\n{r.stdout}{r.stderr}"
            assert 'the "locket" section of' in r.stdout and (Path(t) / ".panoply/config.json").is_file(), \
                f"uninstall did not name, or removed, Locket's settings section\n{r.stdout}"
def _suite():
    # doctor's probe: fires on a live hook, silent on a dead one
    live = f"{shlex.quote(sys.executable)} {shlex.quote(str(lk.HERE / 'memscan.py'))} hook"
    assert probes.probe_hook(live) is True, "doctor's probe does not fire through this memscan.py"
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        cfg = Path(t) / "settings.json"
        py = shlex.quote(sys.executable)
        cfg.write_text(json.dumps({"hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
                "command": f"{py} {shlex.quote(str(lk.HERE / 'locket.py'))} bashguard"}]}],
            "SessionEnd": [{"hooks": [{"type": "command", "command": "/nonexistent/bin/locket usage hook"}]}],
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": f"{py} /nonexistent/memscan.py trigger"}]}],
        }}))
        dead = {ev for ev, _c, _w in claude_code.dead_hooks(cfg)}
        assert dead == {"SessionEnd", "UserPromptSubmit"}, \
            f"doctor misread which entries are dead: {claude_code.dead_hooks(cfg)}"
    assert probes.probe_hook("true") is False, "doctor's probe reads a silent command as firing"
    # doctor's latency check: each entry's own timeout, the default where it names none,
    # a timeout reported as a fail, an empty store skipped, the grepassist log kept out
    with tempfile.TemporaryDirectory() as t:
        cfg = Path(t) / "settings.json"
        cfg.write_text(json.dumps({"hooks": {"PreToolUse": [
            {"matcher": "Write|Edit", "hooks": [{"type": "command", "timeout": 7.5,
                                                 "command": "/opt/x/bin/locket hook"}]},
            {"matcher": "Grep|Bash", "hooks": [{"type": "command", "command": "/opt/x/bin/locket grepassist"}]},
            {"matcher": "Bash", "hooks": [{"type": "command", "timeout": True,
                                           "command": "/opt/x/bin/locket bashguard"}]}]}}))
        assert claude_code.hook_timeout(cfg, "hook") == 7.5, claude_code.hook_entries(cfg)
        assert claude_code.hook_timeout(cfg, "grepassist") == claude_code.DEFAULT_TIMEOUT
        assert claude_code.hook_timeout(cfg, "bashguard") == claude_code.DEFAULT_TIMEOUT, "a bool is not a timeout"
        for bad in (0, -5, float("inf"), 10 ** 400, "10"):
            assert claude_code.usable_timeout(bad) == claude_code.DEFAULT_TIMEOUT, f"{bad!r} read as a timeout"
        assert claude_code.usable_timeout(1e9) == 600.0, "a huge timeout leaves doctor unbounded"
        assert claude_code.registered_hooks(cfg)["hook"] == "/opt/x/bin/locket hook"
        cfg.write_text("{not json")
        assert claude_code.registered_hooks(cfg) is None and claude_code.hook_timeout(cfg, "hook") == claude_code.DEFAULT_TIMEOUT
        full, empty = Path(t) / "full", Path(t) / "empty"
        full.mkdir(); empty.mkdir()
        (full / "a.md").write_text("A fact the latency probe never reads.\n")
        # the csv probe: sizes list each declared CSV once with its text column (None for a
        # schema-only ledger, which doctor never times); the probe is a schema-clean Edit whose
        # old_string is the whole last record; crashes, blocks and unfaithful files are named
        rh = "Ruling,Category,Area,Date\n"
        ref = full / "REF.csv"
        ref.write_text(rh + '"A row the probe copies.",product,pricing,2026-09-26\n'
                       + '"A ruling quoted\nover two lines.",product,pricing,2026-09-27\n')
        (full / manifest.MANIFEST).write_text(json.dumps(
            {"sources": [{"path": "REF.csv", "text": "Ruling", "label": "Area"}]}))
        (full / "RULINGS.csv").write_text(rh)
        sizes = probes.declared_sizes({"empty": empty, "full": full, "again": full}, ["empty", "full", "again"])
        assert sorted((x[1].name, x[2]) for x in sizes) == [("REF.csv", "Ruling"), ("RULINGS.csv", None)], sizes
        target = [x for x in sizes if x[2]][0]
        seen = Path(t) / "payload.json"
        secs, problem, why = probes.csv_hook_latency(f"cat > {shlex.quote(str(seen))}", target, 5)
        assert secs < 5 and problem is None and why is None, (secs, problem, why)
        ti = json.loads(seen.read_text())["tool_input"]
        assert ti["old_string"] == '"A ruling quoted\nover two lines.",product,pricing,2026-09-27', \
            f"old_string is not the whole multiline last record: {ti['old_string']!r}"
        assert "latency probe" in ti["new_string"] and ",pricing,2026-09-27" in ti["new_string"], ti
        assert probes.csv_hook_latency("sleep 3", target, 1)[0] == 1.0, "a csv probe past the timeout is not at the limit"
        _s, problem, _w = probes.csv_hook_latency("exit 1", target, 5)
        assert problem and "crashed" in problem, f"a hook that crashed fast read as a fast hook: {problem}"
        _s, problem, _w = probes.csv_hook_latency("exit 2", target, 5)
        assert problem and "blocked" in problem, f"a hook that blocked the probe read as a fast hook: {problem}"
        dup = '"Same row.",product,pricing,2026-09-26\n'
        ref.write_text(rh + dup + dup)
        assert probes.csv_hook_latency("true", target, 5)[2] == "its last row also appears earlier in the file"
        ref.write_text(rh)
        assert probes.csv_hook_latency("true", target, 5)[2] == "it holds no row to model the probe on"
        ref.write_text(rh + '"Short row.",product\n')
        assert "field count" in probes.csv_hook_latency("true", target, 5)[2], "a ragged last row was probed"
        for f in (ref, full / "RULINGS.csv", full / manifest.MANIFEST):
            f.unlink()
        corpora.clear_settings()
        assert probes.hook_latency("true", {"empty": empty}, 5) == (None, None), "an empty store is timed"
        fast, _o = probes.hook_latency("true", {"empty": empty, "full": full}, 5)
        assert fast and fast[0] == "full" and fast[1] < 5, fast
        slow, _o = probes.hook_latency("sleep 3", {"full": full}, 1)
        assert slow == ("full", 1.0), f"a hook past its timeout is not reported at the limit: {slow}"
        once = Path(t) / "warmed"
        cold, _o = probes.hook_latency(f"[ -e {shlex.quote(str(once))} ] || {{ touch {shlex.quote(str(once))}; sleep 2.4; }}",
                                 {"full": full}, 4)
        assert cold[1] < 2, f"a one-time cold start was reported as the store's latency: {cold}"
        steady, _o = probes.hook_latency("sleep 2.4", {"full": full}, 4)
        assert 2.3 < steady[1] < 4, f"a store slow on every run was hidden by the retry: {steady}"
        once.unlink()
        worse, _o = probes.hook_latency(f"[ -e {shlex.quote(str(once))} ] && sleep 9 || {{ touch {shlex.quote(str(once))}; sleep 2.4; }}",
                                  {"full": full}, 4)
        assert worse[1] == 4.0, f"a retry that timed out was hidden by the faster first run: {worse}"
        quiet, _o = probes.hook_latency(f'[ "$LOCKET_GREPASSIST_LOG" = {shlex.quote(os.devnull)} ] || sleep 3',
                              {"full": full}, 1, "grepassist")
        assert quiet[1] < 1, "the grepassist probe writes to the real log"
        swallow = (f"{py} -c " + shlex.quote(f"import sys; sys.path.insert(0, {str(lk.HERE)!r}); import memscan, claims, corpora, corpus, fsio, gate, health, manifest, trust; "
                                             "health.health_error('hook', 'lexical', ValueError())"))
        _w, opened = probes.hook_latency(swallow, {"full": full}, 5)
        assert opened and opened[0] == "full" and "ValueError in lexical" in opened[1], \
            f"a hook that fails open quickly passed doctor's latency check: {opened}"
        crash = f"{py} -c " + shlex.quote("raise SystemExit(1)")
        _w, opened = probes.hook_latency(crash, {"full": full}, 5)
        assert opened and "crashed" in opened[1], f"a crashing hook passed: {opened}"
        hang = (f"{py} -c " + shlex.quote(f"import sys, time; sys.path.insert(0, {str(lk.HERE)!r}); import memscan, claims, corpora, corpus, fsio, gate, health, manifest, trust\n"
                                          "with health.HookRun('hook'): time.sleep(5)"))
        worst, opened = probes.hook_latency(hang, {"full": full}, 1)
        assert worst == ("full", 1.0) and opened is None, \
            f"a probe killed at its timeout read as failing open: {worst} {opened}"
    assert probes.probe_hook(live, codex=True) is True, \
        "doctor's Codex probe does not fire through a patch into this memscan.py"
    merged = codex.merge_hooks({})
    assert "usage (SessionEnd)" not in merged and {"hook", "bashguard", "grepassist"} <= set(merged), merged
    data = {}
    codex.merge_hooks(data)
    assert codex.merge_hooks(data) == [], "a second --codex added its hooks again"
    assert "SessionEnd" not in data["hooks"], "Codex got the usage recorder, which cannot read its sessions"
    hk = lambda key, cmd, st: {"key": key, "command": cmd, "trustStatus": st}
    reply = {"id": 2, "result": {"data": [{"cwd": "/x", "warnings": [], "errors": [], "hooks": [
        hk("/c/hooks.json:pre_tool_use:0:0", "/opt/homebrew/bin/locket hook", "trusted"),
        hk("/c/hooks.json:pre_tool_use:1:0", "/opt/homebrew/bin/locket bashguard", "modified"),
        hk("/c/hooks.json:pre_tool_use:2:0", "/usr/bin/other-tool", "untrusted")]}]}}
    assert codex.trust_from(reply) == {"/c/hooks.json:pre_tool_use:0:0": "trusted",
                                        "/c/hooks.json:pre_tool_use:1:0": "modified"}, \
        "doctor misread hooks/list, or counted a hook that is not Locket's"
    assert codex.trust_from({"id": 2, "error": {"code": -32601}}) is None, "an error reply reads as trust"
    with tempfile.TemporaryDirectory() as t:
        home = Path(t); k0, k1 = (f"{home}/hooks.json:pre_tool_use:{i}:0" for i in (0, 1))
        tbl = lambda k: f'[hooks.state."{k}"]\ntrusted_hash = "sha256:00"\n\n'
        (home / "config.toml").write_text('model = "x"\n\n[hooks.state]\n\n' + tbl(k0))
        (home / "ollama-launch.config.toml").write_text(tbl(k0) + tbl(k1) + tbl("/elsewhere/hooks.json:x:0:0"))
        assert codex.profile_trust_gap(home, [k1]) == {home / "ollama-launch.config.toml": [k1]}, \
            "the profile trap's gap is misread: only the keys Codex reported lapsed count"
        assert codex.profile_trust_gap(home, [k0 + "9"]) == {}, "a lapsed key no profile holds reads as a gap"
        # _codex_trust against stub servers: one that answers, one that quits on EOF, one that hangs
        real_path, stub = os.environ["PATH"], home / "codex"
        os.environ["PATH"] = f"{home}{os.pathsep}{real_path}"
        answer = {"result": {"data": [{"hooks": [{"key": k0, "command": "locket hook", "trustStatus": "trusted"}]}]}}
        try:
            for body, want in (
                ("import json,sys\nfor l in sys.stdin:\n m=json.loads(l)\n if m.get('id')==1: print(json.dumps({'id':1,'result':{}}),flush=True)\n"
                 f" if m.get('id')==2: print(json.dumps({{'id':2,'method':'x'}}),flush=True); print(json.dumps(dict(id=2,**{answer!r})),flush=True)\n",
                 {k0: "trusted"}),
                ("import sys\nsys.stdin.readline()\n", None),
                ("import time\ntime.sleep(30)\n", None)):
                stub.write_text(f"#!{sys.executable}\n{body}"); stub.chmod(0o755)
                got = codex.trust_status(timeout=2)
                assert got == want, f"_codex_trust read {got!r} from a stub that should give {want!r}"
        finally:
            os.environ["PATH"] = real_path
    live_trigger = f"{shlex.quote(sys.executable)} {shlex.quote(str(lk.HERE / 'locket.py'))} trigger"
    assert probes.probe_trigger(live_trigger) is True, "doctor's trigger probe does not fire through this locket.py"
    assert probes.probe_trigger("true") is False, "doctor's trigger probe reads a silent command as firing"
    assert probes.probe_trigger(live_trigger, hermes=True) is True, \
        "doctor's trigger probe does not fire through this locket.py on Hermes"
    assert probes.probe_trigger("echo '[doctor-probe] probe'", hermes=True) is not True, \
        "doctor's Hermes trigger probe reads a call that was not refused as firing"
    assert probes.probe_trigger("exit 2", hermes=True) is False, \
        "doctor's Hermes trigger probe reads a bare refusal as firing"
    yaml = ('model: x\nhooks:\n  pre_tool_call:\n    - command: "/opt/homebrew/bin/locket hook"\n'
            '      matcher:\n        nested: x\n    - command: /opt/homebrew/bin/locket trigger\n'
            '  pre_llm_call:\n    - command: "/opt/homebrew/bin/locket trigger"\n'
            'other:\n  pre_tool_call:\n    - command: "locket bashguard"\n')
    assert hermes.config_hooks(yaml) == {"pre_tool_call": ["/opt/homebrew/bin/locket hook",
                                                     "/opt/homebrew/bin/locket trigger"],
                                   "pre_llm_call": ["/opt/homebrew/bin/locket trigger"]}, hermes.config_hooks(yaml)
    assert isinstance(probes.probe_hook(f"{shlex.quote(sys.executable)} /nonexistent/memscan.py hook"), str), \
        "doctor's probe reads a missing script as firing"
    import contextlib
    import io
    for form in (["audit", "council", "-h"], ["audit", "help"], ["help", "audit"], ["help", "pointers"],
                 ["dupes", "x", "--help"], ["find", "a statement", "help"]):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = lk.main(["locket", *form])
        assert code == 0 and out.getvalue().startswith(f"usage: locket {form[1] if form[0] == 'help' else form[0]}"), \
            f"`locket {' '.join(form)}` did not print that command's help"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = lk.main(["locket", "help", "manifest"])
    assert code == 0 and out.getvalue().startswith("MANIFEST\n") and "\nMODES" not in out.getvalue() \
        and "belongs_to" in out.getvalue(), "`locket help manifest` is not the MANIFEST section alone"
    import tempfile
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as t:
        os.chdir(t)
        try:
            assert cli.wants_help("init", ["help"]), "a bare help with nothing named help is not help"
            os.mkdir("help")
            assert not cli.wants_help("init", ["help"]), "a directory named help was read as a help request"
            assert not cli.wants_help("hook", ["-h"]) and not cli.wants_help("find", ["help"]), \
                "a hook verb or find's statement was read as a help request"
        finally:
            os.chdir(here)
    _uninstall()


class LocketFrontEnd(unittest.TestCase):
    def test_scripts_removal_names_the_package(self):
        """From the package directory, uninstall names the directory, so no module is left."""
        self.assertEqual(install.scripts_removal(), f"rm -r {shlex.quote(str(install.HERE))}")

    def test_front_end(self):
        with fixture_env():
            _suite()

    def test_verb_registry(self):
        """One declaration per verb: every spelling names one verb, every verb runs, every
        listed verb has its `-h`, and the hook entries a host config names are the hidden
        verbs memscan dispatches without the CLI."""
        from commands import hooks
        import health
        import locket_mcp
        spellings = [n for v in cli.VERBS for n in (v.name, *v.aliases)]
        self.assertEqual(len(spellings), len(set(spellings)), "two verbs share a name or alias")
        self.assertEqual([v.name for v in cli.VERBS if v.run is None], [], "a verb with nothing to run")
        listed = {v.name for v in cli.VERBS if not v.hidden}
        self.assertEqual(set(cli.parser()[1].choices), listed)
        self.assertEqual(set(install.HOOK_VERBS), set(hooks.ENTRIES))
        # the trigger hook is a listed verb whose `hook` counts its own runs
        self.assertTrue(set(health.HEALTH_VERBS) <= set(hooks.ENTRIES) | {"trigger"})
        mcp_names = [v.mcp.name or f"locket_{v.name}" for v in cli.VERBS if v.mcp]
        self.assertTrue(set(mcp_names) <= set(locket_mcp.TOOLS), "a verb's tool is not served")


if __name__ == "__main__":
    unittest.main()
