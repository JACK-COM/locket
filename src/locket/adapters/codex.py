"""Codex: Claude Code's event names, told apart by the fields only Codex sends, with its
file edits arriving as patches and a refusal sent as a deny decision on stdout."""
import json
import os
import re
import shutil
from pathlib import Path

import corpora
import gate
import hosts
import install
from adapters import claude_code

NAME = "codex"

# A Codex patch, as the native `apply_patch` tool's `command` or as the body of an
# `apply_patch` heredoc when the model has no native tool (every model Codex has no
# metadata for, glm through Ollama among them). Same grammar either way.
PATCH_BODY = re.compile(r"\*\*\* Begin Patch\n(.*?)\n?\*\*\* End Patch", re.S)
PATCH_FILE = re.compile(r"\*\*\* (Add|Update|Delete) File: (.+)")
APPLY_PATCH_CMD = re.compile(r"\s*apply_patch\b")


def detects(payload):
    """Codex shares Claude Code's event names and is told apart by the fields only it
    sends (`turn_id`) and by its rollout transcript's file name."""
    return bool(payload.get("turn_id")
                or Path(str(payload.get("transcript_path") or "")).name.startswith("rollout-"))


def parse_patch(text):
    """[(op, path, added, removed)] for each file a Codex patch touches, op one of
    add | update | delete; `added` and `removed` are the `+` and `-` lines joined.
    A `*** Move to:` retargets the update to its destination. [] where the text
    holds no patch."""
    # CRLF first: the body match needs a bare newline, and a stray \r would ride into
    # every added line; a patch that fails to parse passes the gate silently
    m = PATCH_BODY.search((text or "").replace("\r\n", "\n"))
    if not m:
        return []
    out, cur = [], None
    for line in m.group(1).split("\n"):
        f = PATCH_FILE.fullmatch(line.rstrip())
        if f:
            cur = [f.group(1).lower(), f.group(2).strip(), [], []]
            out.append(cur)
        elif cur is None:
            continue
        elif line.startswith("*** Move to: "):
            cur[1] = line[len("*** Move to: "):].strip()
        elif line.startswith("+"):
            cur[2].append(line[1:])
        elif line.startswith("-"):
            cur[3].append(line[1:])
    return [(op, path, "\n".join(a), "\n".join(r)) for op, path, a, r in out]


def patch_input(patch, cwd):
    """The Write or Edit a Codex patch amounts to, as (tool, tool_input), or None
    when it adds nothing. A patch can touch several files and a hook judges one, so
    the most decidable is judged first: the trust registry or a store's configuration
    (refused outright, so a benign first file cannot carry a manifest past the gate),
    then a CSV (whose ledger gate may block), then the first file inside a store."""
    files = [f for f in parse_patch(patch) if f[0] != "delete"]
    at = lambda f: gate.absolute_path(f[1], cwd)
    refused = lambda f: gate.is_registry(at(f)) or gate.store_config(at(f)) is not None
    pick = (next((f for f in files if refused(f)), None)
            or next((f for f in files if f[1].lower().endswith(".csv")), None)
            or next((f for f in files if corpora.corpus_for(at(f)) is not None), None)
            or (files[0] if files else None))
    if pick is None:
        return None
    op, path, added, removed = pick
    if op == "add":
        return "Write", {"file_path": path, "content": added}
    return "Edit", {"file_path": path, "old_string": removed, "new_string": added}


def native(payload, tool, ti):
    """A patch, native or through the shell, becomes the Write or Edit it amounts to."""
    if tool == "apply_patch" or (tool == "Bash" and APPLY_PATCH_CMD.match(str(ti.get("command", "")))):
        edit = patch_input(str(ti.get("command", "")), payload.get("cwd"))
        if edit:
            return edit
    return tool, ti


def deliver(payload, msg, blocking, source, db):
    """The same `additionalContext` as Claude Code advises; a block is a
    `permissionDecision: deny` on stdout, which reaches the model as `Command blocked
    by PreToolUse hook: <reason>` and the write does not land (measured live on
    codex-cli 0.157.0; openai/codex#27833 reports older builds ignoring it). Exit 0,
    since exit 2 on a tool event was not measured."""
    if not blocking:
        return claude_code.deliver(payload, msg, blocking, source, db)
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": msg}}))
    return 0


def status():
    if not hosts.codex().is_dir():
        return None
    return ("2e gate on Codex: armed (reads the rollout transcript Codex hands the hook); nudges "
            "reach the model on the write, once /hooks has trusted them")


def home():
    return hosts.codex()


def trigger_event(payload):
    """A tool event normalised as every Locket hook normalises it, keeping Codex's own
    tool name; a prompt event marked as Codex's."""
    if payload.get("hook_event_name") == "PreToolUse":
        import adapters
        return dict(adapters.normalise(payload), _host_tool=payload.get("tool_name"))
    return dict(payload, _host=NAME)


seen = claude_code.seen
asked = claude_code.asked


def trigger_reply(payload, message, block, home):
    """A blocked tool call is refused by deny JSON; everything else as Claude Code."""
    if block and payload.get("hook_event_name") == "PreToolUse":
        import adapters
        return adapters.emit(payload, message, True, "trigger")
    return claude_code.trigger_reply(payload, message, block, home)


def merge_hooks(data):
    return claude_code.merge_hooks(data, usage=False)


def trust_from(reply):
    """{key: trustStatus} for the Locket entries in a `hooks/list` reply, or None when it is not one."""
    try:
        return {h["key"]: h["trustStatus"] for e in reply["result"]["data"] for h in e["hooks"]
                if install.is_locket_hook(str(h.get("command", "")))}
    except (KeyError, TypeError, AttributeError):
        return None


def trust_status(timeout=20, version=""):
    """Codex's own reading of Locket hook trust, from `codex app-server`'s `hooks/list`, or None
    when Codex is absent or the call fails (the protocol is marked experimental). Codex computes
    the hash; Locket never does (RULINGS.csv on Area=substrate). The app-server takes no
    --profile, so this is config.toml's view, which is the one Codex honours. ⚠ The server exits
    on EOF before answering, so the requests go in one at a time and stdin stays open."""
    import subprocess, threading
    exe = shutil.which("codex")
    if not exe:
        return None
    try:
        p = subprocess.Popen([exe, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "CODEX_HOME": str(hosts.codex())},
                             start_new_session=True)    # an npm launcher's native child dies with it
    except OSError:
        return None

    def kill():
        try:
            os.killpg(p.pid, 9) if hasattr(os, "killpg") else p.kill()
        except OSError:
            pass
    timer = threading.Timer(timeout, kill)
    timer.start()

    def reply_to(i):
        for line in p.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if isinstance(m, dict) and m.get("id") == i and ("result" in m or "error" in m):
                return m
        return None
    try:
        for m in ({"id": 1, "method": "initialize", "params": {"clientInfo": {
                       "name": "locket-doctor", "title": None, "version": version}, "capabilities": None}},
                  {"method": "initialized"},
                  {"id": 2, "method": "hooks/list", "params": {"cwds": [str(hosts.codex())]}}):
            p.stdin.write(json.dumps(m) + "\n"); p.stdin.flush()
            if m.get("id") == 1 and reply_to(1) is None:
                return None
        reply = reply_to(2)
        return trust_from(reply) if reply else None
    except (OSError, ValueError):
        return None
    finally:
        timer.cancel()
        kill()
        for f in (p.stdin, p.stdout):
            try:
                f.close()
            except OSError:
                pass
        p.wait()


def profile_trust_gap(codex_home, lapsed):
    """{profile config: [keys]} for the entries Codex reported untrusted (`lapsed`, its own keys)
    that a profile config (`<name>.config.toml` beside config.toml) holds trust for. codex-cli
    0.160.0 writes trust given under `--profile` into the profile's file and honours it only from
    config.toml, so this is trust that lapses on restart. Keys are matched, never hashes."""
    head = re.compile(r'^\s*\[hooks\.state\."([^"]+)"\]', re.M)
    gap = {}
    for f in sorted(codex_home.glob("*.config.toml")):
        try:
            held = sorted(set(head.findall(f.read_text())) & set(lapsed))
        except (OSError, ValueError):
            continue
        if held:
            gap[f] = held
    return gap


# --- install face ------------------------------------------------------------------------

FLAG = "--codex"


def config_file():
    return hosts.codex() / "hooks.json"


def register():
    """Merge Locket's hooks into hooks.json, and say what Codex still needs: trust given in
    /hooks, and copied out of a profile config that Codex never reads back."""
    rc = install.merge_json(config_file(), merge_hooks)
    print("Codex runs a hook only once it is trusted: start `codex`, run /hooks, and trust "
          "each Locket entry. A later change to an entry asks again; an upgrade does not")
    profiles = sorted(f.name for f in hosts.codex().glob("*.config.toml"))
    if profiles:
        print(f"Codex profile config found ({', '.join(profiles)}): trust given in /hooks under --profile "
              "(as `ollama launch codex` runs it) is written there and lapses on restart; copy its "
              "[hooks.state] tables into config.toml, which `locket doctor` checks")
    return rc


def removal():
    return claude_code.file_removal(config_file())


def checks(m, row):
    """Doctor's rows for Codex: its hooks registered, firing on a patch, and trusted by Codex itself."""
    import probes
    codex_cfg = hosts.codex() / "hooks.json"
    if hosts.codex().is_dir():
        hooks = claude_code.registered_hooks(codex_cfg) if codex_cfg.is_file() else {}
        missing = [v for _, v, _t in claude_code.PRE_TOOL if v not in hooks] if hooks is not None else []
        if hooks is None:
            row("fail", "codex hooks", f"{codex_cfg} is not valid JSON", f"fix {codex_cfg}, then locket install --codex")
        elif missing:
            row("warn", "codex hooks", f"not registered: {', '.join(missing)}", "locket install --codex")
        else:
            probe = probes.probe_hook(hooks["hook"], codex=True)
            if probe is True:
                # the probe runs the command directly, so it proves the hook, not Codex's dispatch;
                # whether Codex will dispatch it is trust, which only Codex's own hooks/list may say
                granted = trust_status(version=m.version)
                lapsed = [k for k, s in (granted or {}).items() if s != "trusted"]
                gap = profile_trust_gap(hosts.codex(), lapsed) if lapsed else {}
                if not granted:               # {} too: a reply naming no Locket entry proves nothing
                    row("warn", "codex hooks", "registered, and the write hook fires on a known fork in a patch; "
                        "Codex could not report trust, and skips a hook until it is trusted",
                        "in codex, run /hooks and trust each Locket entry")
                elif not lapsed:
                    row("ok", "codex hooks", f"registered, firing on a known fork in a patch, and trusted ({len(granted)} entries)")
                elif gap:
                    prof = ", ".join(f.name for f in gap)
                    row("warn", "codex hooks", f"{len(lapsed)} entries untrusted in config.toml, but trusted in {prof}, "
                        "which Codex writes under --profile and never reads back",
                        f"copy the [hooks.state.\"{codex_cfg}:…\"] tables from {prof} into config.toml; "
                        "recopy after any change to hooks.json")
                else:
                    row("warn", "codex hooks", "registered and firing, but Codex reports " +
                        ", ".join(f"{k.split(':', 1)[1]} {granted[k]}" for k in lapsed),
                        "in codex, run /hooks and trust each Locket entry")
            else:
                row("fail", "codex hooks", "registered, but the write hook " +
                    ("stayed silent on a known fork" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {hooks['hook']}")
    claude_code.check_entries(row, "codex", codex_cfg, "--codex")
