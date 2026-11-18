"""Live runs of this install's hooks on scratch stores: whether each fires, how long it
takes on the real stores, and what the hosts' own configs say about it.
"""
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import corpus
import health
import hosts
import install
import ledger
import manifest
import memscan
import paths
import trust


# A paragraph long enough to clear the gate's floor (`gate.MIN_ADDED`), written into a
# throwaway store under ~/.locket so the probe never reads or writes a user's memory.
_PROBE = ("The doctor probe paragraph states one durable fact about a harbour: the tide "
          "gauge at the north pier is read at dawn and dusk, the readings are logged in "
          "the harbourmaster's ledger, and a reading missed twice in a row is escalated "
          "to the port authority before the next high water.")


# Claude Code's timeout for a command hook whose entry names none, in seconds.
CLAUDE_HOOK_DEFAULT_TIMEOUT = 60


def _usable_timeout(t):
    """A hook entry's timeout in seconds when it is a finite positive number, else the
    default; capped, so a huge value cannot leave doctor's probe unbounded."""
    import math
    try:
        t = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else None
    except OverflowError:
        t = None
    if t is None or not math.isfinite(t) or t <= 0:
        return CLAUDE_HOOK_DEFAULT_TIMEOUT
    return min(t, 600.0)


def _hook_entries(settings):
    """{verb: (command, timeout seconds)} for the Locket PreToolUse hooks in a Claude Code
    settings.json, or None when the file is not readable JSON. An entry naming no usable
    timeout gets Claude Code's default."""
    try:
        data = json.loads(settings.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    hooks = data.get("hooks") if isinstance(data, dict) else None
    pre = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
    out = {}
    for e in pre if isinstance(pre, list) else []:
        for h in (e.get("hooks") or []) if isinstance(e, dict) else []:
            cmd = str(h.get("command", "")) if isinstance(h, dict) else ""
            t = _usable_timeout(h.get("timeout") if isinstance(h, dict) else None)
            for _, verb, _t in install.CLAUDE_HOOKS:
                if install._is_locket_hook(cmd, verb):
                    out.setdefault(verb, (cmd, t))
    return out


def _registered_hooks(settings):
    """{verb: command} for the Locket hooks in a Claude Code settings.json."""
    entries = _hook_entries(settings)
    return None if entries is None else {v: c for v, (c, _t) in entries.items()}


def _dead_hooks(config):
    """[(event, command, why)] for every Locket entry in a Claude Code settings.json or a
    Codex hooks.json that cannot run. Each command's program must exist, and each is run on a
    payload that matches nothing, which must exit 0 or 2 without a traceback. `usage hook`
    is checked for its program only, because running it would write the ledger. ⚠ Checking only the verbs
    `install` registers let a dead SessionEnd entry report ok (Tercius letter 2026-09-23)."""
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
                if not install._is_locket_hook(cmd):
                    continue
                try:
                    words = shlex.split(cmd)
                except ValueError:
                    out.append((event, cmd, "the command does not parse")); continue
                script = _script_in(cmd)
                exe = next((w for w in words if not re.fullmatch(r"\w+=.*", w)), "")   # past `VAR=1`
                found = (script.is_file() if script else
                         Path(os.path.expanduser(exe)).is_file() if "/" in exe else shutil.which(exe))
                if not found:
                    out.append((event, cmd, f"{script or exe} does not exist")); continue
                if install._is_locket_hook(cmd, install.CLAUDE_USAGE[1]):   # `usage hook` writes the ledger
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


def _script_in(cmd):
    """The first path in a hook command that names a .py file, or None."""
    for tok in cmd.split():
        if tok.endswith(".py"):
            return Path(os.path.expanduser(tok))
    return None


def _smoke():
    """This install, end to end on scratch stores: the write hook fires through this
    memscan.py on a write repeating a stored paragraph, and an audit finds a repeated
    pair. None when both hold, else what broke. The behaviour suites are the
    repository's tests/, which a package does not ship."""
    import tempfile
    live = f"{shlex.quote(sys.executable)} {shlex.quote(str(install.HERE / 'memscan.py'))} hook"
    fired = _probe_hook(live)
    if fired is not True:
        return f"the write hook did not fire through {install.HERE / 'memscan.py'}: {fired or 'it was silent'}"
    with tempfile.TemporaryDirectory() as t:
        store = Path(t) / "store"
        store.mkdir()
        for name in ("one.md", "two.md"):
            (store / name).write_text(_PROBE + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rc = memscan.main(["memscan.py", "audit", str(store)])
            except (Exception, SystemExit) as e:   # a crash or an exit is the finding, named
                return f"audit raised {type(e).__name__}: {e}"
        if rc:
            return f"audit exited {rc}: {(buf.getvalue().strip().splitlines() or ['no output'])[-1]}"
        if "two.md" not in buf.getvalue():
            return "an audit of two files holding one paragraph found no pair"
    return None


def _probe_hook(cmd, codex=False):
    """Run a registered `hook` command on a write that must fire: a new file in a
    throwaway store repeating a paragraph the store already holds. True when it
    answered, False when it was silent, a string when it failed to run. For Codex
    the write arrives as the patch Codex sends, so the probe crosses the parser."""
    import tempfile
    base = paths.PATHS.state
    base.mkdir(parents=True, exist_ok=True)
    store = Path(tempfile.mkdtemp(prefix="doctor-probe-", dir=base))
    try:
        (store / manifest.MANIFEST).write_text("{}\n")
        (store / "harbour.md").write_text(_PROBE + "\n")
        payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Write",
                              "tool_input": {"file_path": str(store / "zz-probe.md"), "content": _PROBE}}
                             if not codex else
                             {"hook_event_name": "PreToolUse", "tool_name": "apply_patch", "turn_id": "doctor",
                              "cwd": str(store), "tool_input": {"command": "*** Begin Patch\n*** Add File: "
                              "zz-probe.md\n+" + _PROBE + "\n*** End Patch"}})
        # supervised, so a firing hook nudges on stdout and exits 0; exit 2 would
        # also be what python prints for a script path that is gone
        # the probe store is trusted through a registry of its own, so the probe never
        # adds a row to the live one and an untrusted manifest stays inert as it should
        reg = store / ".registry.json"
        reg.write_text(json.dumps({"corpora": [str(store.resolve())],
                                   "pins": {str(store.resolve()): trust.config_digest(store.resolve())}}))
        r, _secs = _run_hook(cmd, payload, 30, LOCKET_HEALTH_DIR=str(store / ".health"),
                             LOCKET_REGISTRY=str(reg))
        if r is None:
            return "timed out after 30s"
        if r.returncode != 0:
            return (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
        return bool(r.stdout.strip())
    finally:
        shutil.rmtree(store, ignore_errors=True)


def _run_hook(cmd, payload, timeout, **env_extra):
    """(CompletedProcess or None on timeout, seconds) for one hook run as Claude Code runs
    it: the command through a shell, the payload on stdin. CLAUDE_AUTONOMOUS is dropped,
    so the run is supervised and a nudge warns on stdout instead of blocking with exit 2.
    `env_extra` overrides the environment, so a probe can point a hook's log elsewhere."""
    import subprocess, time as _time
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_AUTONOMOUS"}
    env.update(env_extra)
    t0 = _time.monotonic()
    try:
        r = subprocess.run(cmd, shell=True, input=payload, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, float(timeout)
    return r, _time.monotonic() - t0


def _hook_timeout(settings, verb):
    """The timeout, in seconds, Claude Code gives the registered `verb` hook."""
    entry = (_hook_entries(settings) or {}).get(verb)
    return entry[1] if entry else CLAUDE_HOOK_DEFAULT_TIMEOUT


def _probe_problem(r, record):
    """The first way a probe run failed open, or None: a crash (an exit other than 0 or
    2, or a traceback) or an error the hook's catch-all swallowed into its health record.
    A fast run that failed open is not a fast hook, so every latency probe asks this."""
    _rec, problems = health.health_report(record, fold=False, kinds=("errors",))
    if r is not None and (r.returncode not in (0, 2) or "Traceback" in r.stderr):
        last = (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][:120]
        problems = [f"the hook crashed ({last}), so the call ran unchecked"] + problems
    return problems[0] if problems else None


def _hook_latency(cmd, stores, limit, verb="hook"):
    """(worst, opened): `worst` is (store, seconds) for the slowest REAL store, `opened` is
    (store, problem) for the first store whose probe failed open, and either may be None.
    It times a registered hook against its
    timeout: `hook` on a new, novel file at each store's root, with an empty transcript so
    the run reaches the meaning-ranked question as well as lexical scoring, or `grepassist`
    on a prose search of the store. A store with neither markdown nor registered rows is not
    timed, except a host store, whose walk is the cost being measured. A hook that answers
    quickly by failing open is caught by
    `opened`: each probe runs with its own health record, read for swallowed errors after it
    exits, and a probe that crashes outright counts too. A probe killed at the limit is a
    timeout, reported by `worst`, never a failure to open.

    ⚠ `_probe_hook` proves the hook fires on a throwaway store of one file, and a store that
    small cannot show a walk that scales with the host tree: the hook once took 25-35 s on
    the council store against a 10 s timeout while the probe reported ok and every real
    write landed unchecked. Only the real stores can show this.

    Nothing lands in a store: the hook is PreToolUse and decides on the payload, and the
    grepassist log is pointed at the null device and hook health at a throwaway directory,
    so the probe leaves no row and no count. A host store
    is always timed, since an empty corpus under a large home still pays the walk; any
    other store is skipped when its walk finds no markdown, checked by the first file."""
    import tempfile
    body = ("Locket doctor latency probe " + os.urandom(6).hex() + ": a novel paragraph that "
            "names no fact any store holds, long enough to pass the net-new floor of the "
            "write gate and reach its meaning-ranked question. ") * 2
    host_roots = {Path(p) for p in paths.PATHS.host_stores.values()}
    worst = opened = None
    # A probe killed at its timeout must not count as a live run the host abandoned, so each
    # probe keeps its own health record; an error in it is a hook that failed open quickly.
    with tempfile.TemporaryDirectory() as scratch:
        transcript = Path(scratch) / "transcript.jsonl"
        transcript.write_text("")
        for i, (name, root) in enumerate(sorted(stores.items())):
            root = Path(root)
            try:
                if root not in host_roots and not corpus.has_content(root):
                    continue                # nothing to score: neither markdown nor registered rows
            except OSError:
                continue
            ti = ({"file_path": str(root / "zz-locket-doctor-probe.md"), "content": body}
                  if verb == "hook" else {"pattern": "where does this store say the gate fails open",
                                          "path": str(root)})
            payload = json.dumps({"hook_event_name": "PreToolUse",
                                  "tool_name": "Write" if verb == "hook" else "Grep",
                                  "session_id": "locket-doctor-latency", "cwd": str(root),
                                  "transcript_path": str(transcript), "tool_input": ti})
            record = Path(scratch) / f"health-{i}"
            r, secs = _run_hook(cmd, payload, limit, LOCKET_GREPASSIST_LOG=os.devnull,
                                LOCKET_HEALTH_DIR=str(record))
            crashed = r is not None and (r.returncode not in (0, 2) or "Traceback" in r.stderr)
            if limit / 2 < secs < limit and not crashed:
                # A first run after an upgrade, or on a store whose index the OS has not cached,
                # pays a one-time cost; the second run is what each later write pays, slower or
                # faster, so it is the one reported, a timeout included.
                r, secs = _run_hook(cmd, payload, limit, LOCKET_GREPASSIST_LOG=os.devnull,
                                    LOCKET_HEALTH_DIR=str(record))
            problem = _probe_problem(r, record)
            if problem and opened is None:
                opened = (name, problem)
            if worst is None or secs > worst[1]:
                worst = (name, secs)
            if secs >= limit:           # one store past the timeout is the finding; stop paying
                break
    return worst, opened


def _declared_sizes(stores, names):
    """[(store name, path, text column or None, bytes)] for every declared CSV in the
    named stores, largest first. One walk serves both doctor rows, and a CSV that a
    parent store and its member both declare is listed once."""
    out, seen = [], set()
    for n in names:
        try:
            declared = ledger.declared_csvs(Path(stores[n]))
        except OSError:
            continue
        for p, text in declared:
            try:
                key, size = p.resolve(), p.stat().st_size
            except OSError:
                continue
            if key not in seen:
                seen.add(key)
                out.append((n, p, text, size))
    return sorted(out, key=lambda x: -x[3])


def _csv_probe_edit(path, textcol):
    """(tool_input, None) for an Edit appending one novel row to `path`, or (None, why)
    when no faithful probe can be built. The row copies the file's last record with its
    text column replaced, so it passes the schema and reaches the source scoring.

    The file is read as the hook reads it (universal newlines), and `old_string` is the
    whole last record, multiline quoting included, because the hook substitutes at the
    FIRST occurrence: a fragment that also appears mid-file would land the probe inside
    another record, and the run would time a parse failure, never the scoring."""
    import csv as _csv, io as _io
    try:
        text = path.read_text(errors="ignore")
        lines = text.splitlines(keepends=True)
        reader, records, start = _csv.reader(lines), [], 0
        for rec in reader:
            if any(x.strip() for x in rec):
                records.append((rec, start, reader.line_num))
            start = reader.line_num
    except (OSError, _csv.Error, UnicodeDecodeError) as e:
        return None, f"it does not read as CSV ({type(e).__name__})"
    if len(records) < 2:
        return None, "it holds no row to model the probe on"
    header, (rec, s, e) = records[0][0], records[-1]
    if textcol not in header:
        return None, f"its header names no {textcol!r} column"
    if len(rec) != len(header):
        return None, "its last row has a different field count from the header"
    old = "".join(lines[s:e]).rstrip("\r\n")
    if text.count(old) != 1:
        return None, "its last row also appears earlier in the file"
    row = list(rec)
    row[header.index(textcol)] = "Locket doctor latency probe " + os.urandom(6).hex() + " names no fact a store holds."
    buf = _io.StringIO()
    _csv.writer(buf, lineterminator="").writerow(row)
    ti = {"file_path": str(path), "old_string": old, "new_string": old + "\n" + buf.getvalue()}
    if ledger.ledger_decision(str(path), ti) != (None, False):
        return None, "a row modelled on its last row breaks its schema"
    return ti, None


def _csv_hook_latency(cmd, target, limit):
    """(seconds, problem, why not timed) for the write hook on one novel row appended to
    `target`, a `_declared_sizes` entry that is a registered source. Exactly one of
    `seconds` and `why` is None; `problem` names a run that failed open or was blocked,
    because a fast run that never reached the scoring is not a fast hook. Nothing is
    written: the hook is PreToolUse and decides on the payload alone.

    ⚠ Growth breaks a declared CSV here and nowhere else: grep and row lookups scale,
    but the hook scores each new row against the whole file and the store, and past
    its timeout the host abandons it and the row lands unchecked. A split was rejected
    as the remedy because it hides rows from every reader that names the table."""
    import tempfile
    _name, path, textcol, _n = target
    ti, why = _csv_probe_edit(path, textcol)
    if ti is None:
        return None, None, why
    with tempfile.TemporaryDirectory() as scratch:
        tr = Path(scratch) / "transcript.jsonl"
        tr.write_text("")
        payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Edit",
                              "session_id": "locket-doctor-latency", "cwd": str(path.parent),
                              "transcript_path": str(tr), "tool_input": ti})
        record = Path(scratch) / "health"
        env = {"LOCKET_GREPASSIST_LOG": os.devnull, "LOCKET_HEALTH_DIR": str(record)}
        r, secs = _run_hook(cmd, payload, limit, **env)
        if limit / 2 < secs < limit:       # the second run is what each later write pays
            r, secs = _run_hook(cmd, payload, limit, **env)
        problem = _probe_problem(r, record)
        if problem is None and r is not None and r.returncode == 2:
            problem = "the hook blocked the probe row, so the scoring was never reached"
    return secs, problem, None


def _codex_trust_from(reply):
    """{key: trustStatus} for the Locket entries in a `hooks/list` reply, or None when it is not one."""
    try:
        return {h["key"]: h["trustStatus"] for e in reply["result"]["data"] for h in e["hooks"]
                if install._is_locket_hook(str(h.get("command", "")))}
    except (KeyError, TypeError, AttributeError):
        return None


def _codex_trust(timeout=20, version=""):
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
        return _codex_trust_from(reply) if reply else None
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


def _profile_trust_gap(codex_home, lapsed):
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


def _probe_trigger(cmd, hermes=False):
    """Run a registered `trigger` command against a throwaway home whose host store holds
    one row the payload matches. True when the row's question came back, False when the
    command was silent, a string when it failed to run. On Hermes the row refuses the call
    once, so the answer is exit 2 with the question in the block message."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as t:
        home = Path(t) / (".hermes" if hermes else ".claude")
        home.mkdir()
        (home / "triggers.json").write_text(json.dumps({"rows": [
            {"id": "doctor-probe", "tools": "Bash", "content": "locket-doctor-probe", "question": "probe"}]}))
        payload = json.dumps({"hook_event_name": "pre_tool_call" if hermes else "PreToolUse",
                              "tool_name": "terminal" if hermes else "Bash", "cwd": t, "session_id": "",
                              "tool_input": {"command": "echo locket-doctor-probe"}})
        try:        # TMPDIR too: the refusal record must not outlive the probe
            r = subprocess.run(cmd, shell=True, input=payload, capture_output=True, text=True, timeout=30,
                               env={**os.environ, "HOME": t, "TMPDIR": t})
        except subprocess.TimeoutExpired:
            return "timed out after 30s"
        if r.returncode != (2 if hermes else 0):
            return (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
        return "[doctor-probe] probe" in r.stdout
