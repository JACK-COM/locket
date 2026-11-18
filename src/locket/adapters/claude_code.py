"""Claude Code: the shape every hook mode is written against, and the fallback host, so
it detects every payload the hosts before it in `adapters.HOSTS` did not claim."""
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import hosts
import install
import paths

NAME = "claude-code"


def detects(payload):
    return True


def native(payload, tool, ti):
    """(tool, tool_input) in Claude Code's shape: already so."""
    return tool, ti


def deliver(payload, msg, blocking, source, db):
    """Exit 2 with the message on stderr blocks; stdout JSON carrying
    `additionalContext` reaches the model without blocking."""
    if blocking:
        print(msg, file=sys.stderr)
        return 2
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "additionalContext": msg}}))
    return 0


def status():
    if not hosts.claude().is_dir():
        return None
    return ("2e gate on Claude Code: armed (reads the session transcript Claude Code hands the "
            "hook); nudges reach the model on the write")


def home():
    return hosts.claude()


def trigger_event(payload):
    """The payload in the shape the trigger hook reads: already so."""
    return payload


def seen(payload, home, pattern):
    """Whether the session carries `pattern` past its transcript, which is read first."""
    return False


def asked(payload):
    """The 2e answer with no transcript to read: yes, failing open."""
    return True


def trigger_reply(payload, message, block, home):
    """The question on stderr with exit 2 for a block; context on stdout otherwise."""
    if block:
        print(message, file=sys.stderr)
        return 2
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": payload.get("hook_event_name") or "PreToolUse", "additionalContext": message}}))
    return 0


# Claude Code's PreToolUse entries, the three `locket install --hooks` merges.
PRE_TOOL = (("Write|Edit", "hook", 10), ("Bash", "bashguard", 5), ("Grep|Bash", "grepassist", 10))


# `locket trigger` reads every tool call and every prompt, since a row can name any tool.
TRIGGER = (("PreToolUse", "*"), ("UserPromptSubmit", None))


# `locket usage hook` records the ledger as each session ends, before its transcript can age out.
USAGE = ("SessionEnd", install.USAGE_HOOK, 30)


def merge_hooks(data, usage=True):
    """Merge Locket's hooks into a Claude Code settings.json, or into a Codex
    hooks.json, which takes the same schema; `usage` adds the SessionEnd recorder,
    which reads Claude Code's and Hermes's sessions and not Codex's."""
    pre = data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    present = [str(h.get("command", "")) for e in pre if isinstance(e, dict)
               for h in e.get("hooks", []) if isinstance(h, dict)]
    added = []
    for matcher, verb, timeout in PRE_TOOL:
        if any(install.is_locket_hook(c, verb) for c in present):
            continue
        pre.append({"matcher": matcher, "hooks": [{"type": "command", "timeout": timeout,
                    "command": install.hook_command(verb)}]})
        added.append(verb)
    for event, on in TRIGGER:
        entries = data["hooks"].setdefault(event, [])
        if trigger_command(entries):
            continue
        entry = {"hooks": [{"type": "command", "timeout": 5, "command": install.hook_command("trigger")}]}
        entries.append({"matcher": on, **entry} if on else entry)
        added.append(f"trigger ({event})")
    event, verb, timeout = USAGE
    if not usage:
        return added
    entries = data["hooks"].setdefault(event, [])
    if not event_command(entries, verb):
        entries.append({"hooks": [{"type": "command", "timeout": timeout, "command": install.hook_command(verb)}]})
        added.append(f"usage ({event})")
    return added


def event_command(entries, verb):
    """The registered Locket `verb` command among one event's hook entries, or None."""
    for e in entries if isinstance(entries, list) else []:
        for h in (e.get("hooks") or []) if isinstance(e, dict) else []:
            cmd = str(h.get("command", "")) if isinstance(h, dict) else ""
            if install.is_locket_hook(cmd, verb):
                return cmd
    return None


def trigger_command(entries):
    return event_command(entries, "trigger")


def locket_entries(path):
    """(entries to drop, remaining) for a Claude Code settings.json: every hook
    whose command runs memscan.py."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return [], None
    hooks = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(hooks, dict):
        return [], None
    drop = []
    for event, entries in list(hooks.items()):
        if not isinstance(entries, list):
            continue
        keep = []
        for e in entries:
            hs = e.get("hooks", []) if isinstance(e, dict) else []
            kept = [h for h in hs if not install.is_locket_hook(str(h.get("command", "")))]
            drop += [f"{event} {e.get('matcher', '')}: {h.get('command')}" for h in hs if h not in kept]
            if kept:
                keep.append({**e, "hooks": kept})
            elif not hs:
                keep.append(e)
        hooks[event] = keep
    return drop, data


# Claude Code's timeout for a command hook whose entry names none, in seconds.
DEFAULT_TIMEOUT = 60


def usable_timeout(t):
    """A hook entry's timeout in seconds when it is a finite positive number, else the
    default; capped, so a huge value cannot leave doctor's probe unbounded."""
    import math
    try:
        t = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else None
    except OverflowError:
        t = None
    if t is None or not math.isfinite(t) or t <= 0:
        return DEFAULT_TIMEOUT
    return min(t, 600.0)


def hook_entries(settings):
    """{verb: (command, timeout seconds)} for the Locket PreToolUse hooks in a Claude Code
    settings.json, or None when the file is not readable JSON. An entry naming no usable
    timeout gets Claude Code's default."""
    try:
        data = json.loads(settings.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    hooks = data.get("hooks") if isinstance(data, dict) else None
    pre = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
    out: dict[str, tuple[str, int]] = {}
    for e in pre if isinstance(pre, list) else []:
        for h in (e.get("hooks") or []) if isinstance(e, dict) else []:
            cmd = str(h.get("command", "")) if isinstance(h, dict) else ""
            t = usable_timeout(h.get("timeout") if isinstance(h, dict) else None)
            for _, verb, _t in PRE_TOOL:
                if install.is_locket_hook(cmd, verb):
                    out.setdefault(verb, (cmd, t))
    return out


def registered_hooks(settings):
    """{verb: command} for the Locket hooks in a Claude Code settings.json."""
    entries = hook_entries(settings)
    return None if entries is None else {v: c for v, (c, _t) in entries.items()}


def dead_hooks(config):
    """[(event, command, why)] for every Locket entry in a Claude Code settings.json or a
    Codex hooks.json that cannot run. Each command's program must exist, and each is run on a
    payload that matches nothing, which must exit 0 or 2 without a traceback. `usage hook`
    is checked for its program only, because running it would write the ledger. ⚠ Checking only the verbs
    `install` registers let a dead SessionEnd entry report ok."""
    import subprocess, tempfile
    try:
        events = json.loads(config.read_text()).get("hooks") or {}
    except (OSError, ValueError, AttributeError):
        return []
    out = []
    for event, entries in (events.items() if isinstance(events, dict) else []):
        for e in entries if isinstance(entries, list) else []:
            for h in (e.get("hooks") or []) if isinstance(e, dict) else []:
                cmd = str(h.get("command", "")) if isinstance(h, dict) else ""
                if not install.is_locket_hook(cmd):
                    continue
                try:
                    words = shlex.split(cmd)
                except ValueError:
                    out.append((event, cmd, "the command does not parse")); continue
                script = install.script_in(cmd)
                exe = next((w for w in words if not re.fullmatch(r"\w+=.*", w)), "")   # past `VAR=1`
                found = (script.is_file() if script else
                         Path(os.path.expanduser(exe)).is_file() if "/" in exe else shutil.which(exe))
                if not found:
                    out.append((event, cmd, f"{script or exe} does not exist")); continue
                if install.is_locket_hook(cmd, USAGE[1]):   # `usage hook` writes the ledger
                    continue
                with tempfile.TemporaryDirectory() as d:
                    payload = json.dumps({"hook_event_name": event, "cwd": d, "prompt": "",
                                          "tool_name": "Read", "tool_input": {"file_path": f"{d}/x.txt"}})
                    try:
                        r = subprocess.run(cmd, shell=True, input=payload, capture_output=True,
                                           text=True, timeout=30, cwd=d,
                                           env=dict(os.environ, LOCKET_HEALTH_DIR=f"{d}/.health"))
                    except subprocess.TimeoutExpired:
                        out.append((event, cmd, "timed out after 30s")); continue
                if r.returncode not in (0, 2) or "Traceback" in r.stderr:
                    last = (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
                    out.append((event, cmd, last[:120]))
    return out


def hook_timeout(settings, verb):
    """The timeout, in seconds, Claude Code gives the registered `verb` hook."""
    entry = (hook_entries(settings) or {}).get(verb)
    return entry[1] if entry else DEFAULT_TIMEOUT


# --- install face: what `locket install`, `uninstall` and `doctor` do on this host -----

FLAG = "--hooks"


def register():
    """Merge Locket's hooks into settings.json; the exit code."""
    return install.merge_json(settings_file(), merge_hooks)


def file_removal(path):
    """The Locket entries a Claude-Code-shaped hooks file holds, as an `install.Removal`."""
    found, rest = locket_entries(path) if path.is_file() else ([], None)

    def apply():
        install.rewrite_json(path, rest, f"{len(found)} hook entries removed")
    return install.Removal([("hook", f"{path}: {h}") for h in found],
                           apply if found and rest is not None else None, [])


def removal():
    return file_removal(settings_file())


def settings_file():
    return hosts.claude() / "settings.json"


def check_entries(row, label, cfg, flag):
    """Doctor's row on whether every Locket entry in a Claude-Code-shaped hooks file runs."""
    if cfg.is_file():
        dead = dead_hooks(cfg)
        if dead:
            ev, cmd, why = dead[0]
            more = f" (and {len(dead) - 1} more)" if len(dead) > 1 else ""
            row("fail", f"{label} entries", f"{ev}: `{cmd}` cannot run: {why}{more}",
                f"remove the dead entries from {cfg}, then locket install {flag}")
        else:
            row("ok", f"{label} entries", "every Locket entry, under every event, runs")


def checks(m, row):
    """Doctor's rows for Claude Code: its hooks registered and firing, their latency on the real stores, every entry runnable, the trigger rows, the usage recorder."""
    import probes
    # a store trusted by convention has no pin, so a rows file appearing in one is named
    for d in sorted((hosts.claude() / "projects").glob("*/memory")):
        if (d / "triggers.json").is_file():
            row("warn", "config", f"{d / 'triggers.json'} adds trigger rows to a project's auto-memory, "
                "which is trusted without a pin; read it, and delete it if you did not write it")
    settings = settings_file()
    if hosts.claude().is_dir():
        hooks = registered_hooks(settings) if settings.is_file() else {}
        if hooks is None:
            row("fail", "claude hooks", f"{settings} is not valid JSON", f"fix {settings}, then locket install --hooks")
        else:
            missing = [v for _, v, _t in PRE_TOOL if v not in hooks]
            stale = [v for v, c in hooks.items() if (p := install.script_in(c)) and not p.is_file()]
            if missing:
                row("fail", "claude hooks", f"not registered: {', '.join(missing)}", "locket install --hooks")
            elif stale:
                row("fail", "claude hooks", f"point at a missing script: {', '.join(stale)}",
                    "locket uninstall --dry-run, then remove the stale entries and locket install --hooks")
            else:
                probe = probes.probe_hook(hooks["hook"])
                if probe is True:
                    row("ok", "claude hooks", "registered, and the write hook fires on a known fork")
                    for verb, what, lost in (("hook", "the write hook", "the write lands unchecked"),
                                             ("grepassist", "grepassist", "the search runs without its neighbours")):
                        if verb not in hooks:
                            continue
                        limit = hook_timeout(settings, verb)
                        slow, opened = probes.hook_latency(hooks[verb], m.stores, limit, verb)
                        label = f"{verb} latency"
                        if opened is not None:
                            row("fail", label, f"{what} failed open on {opened[0]}: {opened[1]}",
                                "find the cause in that store (an unreadable file, a broken manifest)")
                            continue
                        if slow is None:
                            continue
                        name, secs = slow
                        if secs >= limit:
                            row("fail", label, f"{what} takes {secs:.0f}s or more on {name}, past its {limit:g}s "
                                f"timeout: Claude Code abandons it and {lost}",
                                "brew upgrade locket; if current, the store's walk is entering host state")
                        elif secs > limit / 2:
                            row("warn", label, f"{what} takes {secs:.1f}s on {name}, over half its {limit:g}s "
                                f"timeout; past it, {lost}", "find what the store's walk enters that it should not")
                        else:
                            row("ok", label, f"{secs:.1f}s on the slowest store ({name}), inside its {limit:g}s timeout")
                    # Only a registered source is scored row by row, so only sources are timed; the
                    # largest three, because the slowest file need not be the largest.
                    limit = hook_timeout(settings, "hook")
                    timed, skipped, failed = [], [], None
                    for target in [s for s in m.sizes if s[2] is not None][:3]:
                        secs, problem, why = probes.csv_hook_latency(hooks["hook"], target, limit)
                        if problem:
                            failed = (target, problem)
                            break
                        if why:
                            skipped.append(f"{target[1].name} ({why})")
                        else:
                            timed.append((secs, target))
                            if secs >= limit:
                                break
                    fix = "choose a remedy against this reading (a faster scoring path, an index); never split the file"
                    if failed:
                        row("fail", "csv latency", f"the write hook failed on a row into {failed[0][1].name} in "
                            f"{failed[0][0]}: {failed[1]}", "find the cause in that store")
                    elif not timed:
                        row("warn" if skipped else "ok", "csv latency",
                            ("not timed: " + "; ".join(skipped)) if skipped else "no registered source CSV to time")
                    if timed and not failed:
                        secs, big = max(timed, key=lambda x: x[0])
                        where = f"{big[1].name} in {big[0]} ({big[3] / 1048576:.1f} MB, slowest of {len(timed)} timed)"
                        if secs >= limit:
                            row("fail", "csv latency", f"the write hook takes {secs:.0f}s or more on a row into {where}, "
                                f"past its {limit:g}s timeout: rows land unchecked", fix)
                        elif secs > limit / 2:
                            row("warn", "csv latency", f"the write hook takes {secs:.1f}s on a row into {where}, "
                                f"over half its {limit:g}s timeout", fix)
                        else:
                            row("ok", "csv latency", f"{secs:.1f}s on a row into {where}, inside its {limit:g}s timeout")
                else:
                    row("fail", "claude hooks", "registered, but the write hook " +
                        ("stayed silent on a known fork" if probe is False else f"failed: {probe}"),
                        f"run it by hand: {hooks['hook']}")
    check_entries(row, "claude", settings, "--hooks")
    import trigger
    rows_path = trigger.rows_file(hosts.claude())
    if rows_path.is_file():
        try:
            events = (json.loads(settings.read_text()).get("hooks") or {}) if settings.is_file() else {}
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            events = None
        unregistered = [e for e, _m in TRIGGER
                        if events is not None and not trigger_command(events.get(e) if isinstance(events, dict) else None)]
        problems = trigger.findings(hosts.claude())
        if unregistered:
            row("fail", "triggers", f"{rows_path} has rows, but the hook is not registered for "
                f"{' or '.join(unregistered)}", "locket install --hooks")
        elif problems:
            row("fail", "triggers", f"{len(problems)} problem(s) in {rows_path}: {problems[0]}", "locket trigger check")
        elif events is not None:
            probe = probes.probe_trigger(trigger_command(events.get("PreToolUse")))
            if probe is True:
                row("ok", "triggers", f"{rows_path} lints clean; the hook is registered and fires on a known row")
            else:
                row("fail", "triggers", "registered, but the hook " +
                    ("stayed silent on a known row" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {trigger_command(events.get('PreToolUse'))}")
    import time
    if hosts.claude().is_dir():
        try:
            events = (json.loads(settings.read_text()).get("hooks") or {}) if settings.is_file() else {}
            registered = event_command(events.get(USAGE[0]), USAGE[1])
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            registered = None
        age = (time.time() - paths.PATHS.usage_ledger.stat().st_mtime) / 86400 if paths.PATHS.usage_ledger.is_file() else None
        seen = "never recorded" if age is None else f"last recorded {age:.1f} days ago"
        dangling = bool(registered and (p := install.script_in(registered)) and not p.is_file())
        if dangling:
            row("fail", "usage", f"the SessionEnd hook names a missing script: {p}",
                "locket uninstall --dry-run, then remove the stale entry and locket install --hooks")
        elif not registered:
            row("warn", "usage", f"{seen}; nothing records it as a session ends, so a day older than "
                "Claude Code keeps its transcripts is lost unless `locket usage` ran within it", "locket install --hooks")
        else:
            row("ok", "usage", f"recorded as each session ends; {seen}")
