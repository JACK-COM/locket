"""Hermes: its own event and tool names, its memory tool, and no same-turn channel for
advice, so a non-blocking message refuses the call once."""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fsio
import hosts
import install
import ledger
import paths

NAME = "hermes"
DISPLAY = "Hermes"
BASE = False
TRIGGER_EXIT = 2            # Hermes has no context channel: a matched row refuses the call once
INSTALL = "takes its hooks from its own config.yaml, by hand"
INSTALL_EXAMPLE = ""
ROW = hosts.row(__name__)

# Hermes tool name -> the Claude Code tool the hook modes are written against.
TOOLS = {"write_file": "Write", "patch": "Edit", "terminal": "Bash",
         "search_files": "Grep", "read_file": "Read"}

PENDING_TTL = 3600      # seconds; a refusal record older than this is stale
ONCE_NOTE = "Locket asks this once; to proceed, repeat the same call unchanged."
ONCE_MAX = 2            # refusals of one call before it passes whatever state.db says


def detects(payload):
    return str(payload.get("hook_event_name") or "") in ("pre_tool_call", "post_tool_call", "pre_llm_call")


def native(payload, tool, ti):
    """Hermes's tool names mapped to Claude Code's, its `path` copied into `file_path`,
    and its memory tool read as the Edit it amounts to. A `patch` call in Hermes's
    unadvertised V4A mode carries a whole patch string and no `new_string`; it stays
    opaque and the write modes see an empty text, which is silence rather than a
    wrong nudge."""
    if tool != "memory":
        tool = TOOLS.get(tool, tool)
        if "file_path" not in ti and "path" in ti:
            ti["file_path"] = ti["path"]
        return tool, ti
    # Hermes's own memory writes go through this tool, never write_file, so
    # without this mapping the store it exists for is the one it never sees.
    # add = append (an Edit with no anchor), replace = Edit with old_text,
    # remove = nothing added; `operations` batches several of those.
    ops = ti.get("operations") if isinstance(ti.get("operations"), list) else [ti]
    added, old = [], []
    for op in ops:
        if not isinstance(op, dict):
            continue
        if isinstance(op.get("old_text"), str):
            old.append(op["old_text"])   # a removed span leaves the corpus too
        if op.get("action") == "remove":
            continue
        txt = op.get("content") or op.get("new_text")
        if isinstance(txt, str):
            added.append(txt)
    target = ti.get("target") or "memory"   # top-level in both shapes
    ti = {"file_path": str(home() / "memories" / ("USER.md" if target == "user" else "MEMORY.md"))}
    if old:                              # replace: an Edit with its anchor
        ti.update(old_string="\n".join(old), new_string="\n".join(added))
    else:                                # add: an append, so the hook sees
        ti.update(content="\n".join(added), _append=True)   # prior + entry
    return "Edit", ti


def deliver(payload, msg, blocking, source, db):
    """Exit 2 blocks, and the message travels as stdout JSON so it is not cut at the
    400 characters Hermes keeps of stderr. ⚠ Hermes has NO same-turn channel for a
    non-blocking message on a tool event: `pre_tool_call` honours only block, modify
    and approve, a `post_tool_call` context is discarded, and `pre_llm_call` runs once
    per user turn, so advice parked for it arrives after the agent has finished.
    Advice is therefore refused once (`refuse_once`)."""
    if not blocking:
        msg = refuse_once(payload, msg.replace(ledger.LEAD_WARNING, ledger.LEAD_BLOCKING, 1), source, db)
        if msg is None:
            return 0
    print(json.dumps({"action": "block", "message": msg}))
    print(msg, file=sys.stderr)
    return 2


def _pending_ours():
    """The record directory exists, is no symlink, and is this user's. A directory
    someone else made could forge or erase the records that let a call through."""
    try:
        return (paths.PATHS.pending_dir.is_dir() and not paths.PATHS.pending_dir.is_symlink()
                and (not hasattr(os, "getuid") or paths.PATHS.pending_dir.stat().st_uid == os.getuid()))
    except OSError:
        return False


def _strings(value):
    """Every string inside a parsed JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def shown(session_id, message, since, db=None):
    """Whether a tool result Hermes stored for this session at or after `since`
    carries the start of `message`: True, False, or None when state.db cannot say.
    Hermes stores a blocked call's message as that call's result, JSON-encoded,
    before the next tool round runs."""
    db = home() / "state.db" if db is None else Path(db)
    if not session_id or not db.is_file():
        return None
    head = message[:300]            # stored JSON-encoded, so it is read back through json
    rows = fsio.sqlite_rows(db, "select content from messages where session_id=? and role='tool' "
                                "and timestamp>=?", (session_id, since - 1))
    if rows is None:                         # locked, missing table, schema drift
        return None
    for (content,) in rows:
        if not isinstance(content, str):
            continue
        try:
            if any(head in s for s in _strings(json.loads(content))):
                return True
        except (json.JSONDecodeError, ValueError):
            continue
    return False


def refuse_once(payload, msg, source, db=None):
    """The message to refuse a Hermes tool call with, or None to let it pass.
    The same call from the same mode is refused until state.db shows the message
    reached the model, and at most ONCE_MAX times. Hermes runs every pre_tool_call
    hook but shows the model only the first block, so a record alone would let a
    retry pass a second mode's message nobody saw; the cap keeps a result Hermes
    moved to a file, or a state.db that cannot be read, from refusing forever. The
    note leads so the model reads how to proceed before the advice. `source` names
    the mode, so two modes refusing one call keep separate records.
    """
    note = f"{ONCE_NOTE}\n\n{msg}"
    try:
        paths.PATHS.pending_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not _pending_ours():
            return None                      # no record means no retry could pass
        now = time.time()
        for p in paths.PATHS.pending_dir.iterdir():
            try:
                if now - p.stat().st_mtime > PENDING_TTL:
                    p.unlink()
            except OSError:
                continue
        call = json.dumps([source, payload.get("tool_name"), payload.get("tool_input")],
                          sort_keys=True, default=str)
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(payload.get("session_id") or "any"))
        record = paths.PATHS.pending_dir / f"once-{sid}-{hashlib.sha256(call.encode()).hexdigest()[:32]}"
        count = int(record.read_text() or 0) if record.is_file() else 0
        if count and (count >= ONCE_MAX or shown(payload.get("session_id"), note,
                                                 record.stat().st_mtime, db) is True):
            record.unlink()
            return None
        record.write_text(str(count + 1))
        return note
    except Exception:                        # a guardrail never breaks the call it watched
        return None


def status():
    if not home().is_dir():
        return None
    armed = (home() / "state.db").is_file()
    return ("2e gate on Hermes: " + ("armed (reads state.db by session_id)" if armed else
                                     "UNARMED, state.db not found, so every write passes the question silently")
            + "; a nudge refuses the call once and the same call repeated passes")


def probe_write(store, name, text):
    """The write a doctor probe sends as Hermes does."""
    return {"hook_event_name": "pre_tool_call", "tool_name": "write_file", "session_id": "",
            "tool_input": {"path": str(store / name), "content": text}}


def probe_call(cwd, command):
    """The shell call a doctor trigger probe sends as Hermes does."""
    return {"hook_event_name": "pre_tool_call", "tool_name": "terminal", "cwd": cwd, "session_id": "",
            "tool_input": {"command": command}}


def session_cwds():
    """Working directories Hermes has opened a session in, from its state.db."""
    db = home() / "state.db"
    if not db.is_file():
        return set()
    return {r[0] for r in fsio.sqlite_rows(
        db, "SELECT DISTINCT cwd FROM sessions WHERE cwd IS NOT NULL AND cwd != ''") or ()}


def home():
    return hosts.home(ROW.name)


def trigger_event(payload):
    """The payload in the shape the trigger hook reads: `pre_llm_call` carries the user's
    message under `extra` and is read as a prompt; a tool call is normalised as every
    Locket hook normalises it, keeping Hermes's own tool name."""
    event = payload.get("hook_event_name")
    if event == "pre_llm_call":
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        msg = extra.get("user_message")
        if isinstance(msg, list):           # multimodal: the text parts
            msg = " ".join(p["text"] for p in msg if isinstance(p, dict) and isinstance(p.get("text"), str))
        return dict(payload, hook_event_name="UserPromptSubmit", _host=NAME,
                    prompt=msg if isinstance(msg, str) else "")
    if event == "pre_tool_call":
        import adapters
        return dict(adapters.normalise(payload), _host_tool=payload.get("tool_name"))
    return payload


def session_has(db, session_id, pattern):
    """True when a message or tool call this session stored in state.db carries
    `pattern`; False ("not yet") on any failure."""
    if not session_id or not pattern or not Path(db).is_file():
        return False
    try:
        rx = re.compile(pattern)
    except (re.error, TypeError):
        return False
    rows = fsio.sqlite_rows(db, "select content, tool_calls from messages where session_id=?", (session_id,))
    if rows is None:
        return False
    return any(isinstance(v, str) and rx.search(v) for r in rows for v in r)


def seen(payload, home, pattern):
    """Whether the session carries `pattern`, from the state.db Hermes keeps it in."""
    return session_has(Path(home) / "state.db", payload.get("session_id"), pattern)


def memfind_asked_db(db_path, session_id):
    """Whether this session ran memfind, from state.db, whose `messages` table holds one
    row per assistant turn with `tool_calls` as JSON. Fails open on a missing or locked
    database and on any shape it does not recognise, as the transcript reader does: a
    wrong no here cannot be cleared by asking."""
    import gate
    if not session_id or not Path(db_path).is_file():
        return True
    rows = fsio.sqlite_rows(db_path, "select tool_calls from messages where session_id=? "
                                     "and (tool_calls like ? or tool_calls like ?)",
                            (session_id, "%memfind%", "%locket%"))
    if rows is None:                         # locked, missing table, schema drift
        return True
    for (raw,) in rows:
        try:
            calls = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, ValueError):
            continue
        for call in calls if isinstance(calls, list) else []:
            fn = (call or {}).get("function") or {}
            if fn.get("name") not in ("terminal", "Bash"):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    continue
            cmd = (args or {}).get("command")
            if isinstance(cmd, str) and gate.runs_memfind(cmd):
                return True
    return False


def asked(payload):
    """The 2e answer, read from state.db by session_id."""
    return memfind_asked_db(home() / "state.db", payload.get("session_id"))


def trigger_reply(payload, message, block, home):
    """A prompt's rows as stdout JSON context; a tool call answered through
    `adapters.emit`, which refuses it once for a row that does not block."""
    if payload["hook_event_name"] == "UserPromptSubmit":
        # Hermes moves a context over 10,000 characters to a file; keep the rows inline.
        print(json.dumps({"context": message[:9000]}))
        return 0
    try:
        import adapters
        return adapters.emit(payload, message, block, "trigger", Path(home or hosts.home(ROW.name)) / "state.db")
    except Exception:
        return 0


# The (event, verb) entries Hermes's config.yaml needs. `deliver` is retired and does nothing.
HOOKS = (("pre_tool_call", "hook"), ("pre_tool_call", "bashguard"), ("pre_tool_call", "grepassist"),
                ("pre_tool_call", "trigger"), ("pre_llm_call", "trigger"))


def locket_lines(path):
    """`path:line` for every Hermes hook line that runs Locket, by script or command."""
    try:
        return [f"{path}:{i}" for i, l in enumerate(path.read_text().splitlines(), 1)
                if (c := re.search(r"command:\s*(.+?)\s*$", l)) and install.is_locket_hook(c.group(1))]
    except OSError:
        return []


def config_hooks(text):
    """{event: [command]} from config.yaml's top-level `hooks:` block, read by indentation
    in the shape Step 4 prints, so doctor needs no YAML parser."""
    out: dict[str, list[str]] = {}
    inside, event, depth = False, None, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            inside, event, depth = line.rstrip() == "hooks:", None, None
            continue
        if not inside:
            continue
        indent = len(line) - len(line.lstrip())
        depth = indent if depth is None else depth
        if indent == depth and (m := re.fullmatch(r"\s+([\w-]+):\s*", line)):
            event = m.group(1)
        elif (c := re.search(r"command:\s*(.+?)\s*$", line)) and event:
            out.setdefault(event, []).append(c.group(1).strip("'\""))
    return out


# --- install face: config.yaml is edited by hand (`locket help install`, Step 4) ----------

FLAG = None


def register():
    """Nothing to merge: config.yaml is edited by hand."""
    return 0


def removal():
    """Nothing Locket edits; the YAML lines that run it, named for a person to remove."""
    cfg = home() / "config.yaml"
    lines = locket_lines(cfg) if cfg.is_file() else []
    return install.Removal([], None, [f"Hermes hook line {hl}  (YAML; remove the entry by hand)" for hl in lines])


def checks(m, row):
    """Doctor's rows for Hermes: the five config.yaml entries, and its trigger rows firing."""
    import probes
    import trigger
    hermes_cfg = home() / "config.yaml"
    if hermes_cfg.is_file():
        events = config_hooks(hermes_cfg.read_text(errors="replace"))
        missing = [f"{v} on {e}" for e, v in HOOKS
                   if not any(install.is_locket_hook(c, v) for c in events.get(e, []))]
        retired = any(install.is_locket_hook(c, "deliver") for cmds in events.values() for c in cmds)
        if missing:
            row("warn", "hermes hooks", f"not in config.yaml: {', '.join(missing)}"
                + ("; `locket deliver` is retired, remove it" if retired else ""), "`locket help install`, Step 4")
        elif retired:
            row("warn", "hermes hooks", "`locket deliver` is retired and does nothing", "remove its pre_llm_call entry")
        else:
            row("ok", "hermes hooks", "all five registered")
        cmd = next((c for c in events.get("pre_tool_call", []) if install.is_locket_hook(c, "trigger")), None)
        hermes_rows = trigger.rows_file(home())
        problems = trigger.findings(home())
        if problems:
            row("fail", "hermes triggers", f"{len(problems)} problem(s) in {hermes_rows}: {problems[0]}",
                "locket trigger check hermes")
        elif hermes_rows.is_file() and cmd:
            probe = probes.probe_trigger(cmd, sys.modules[__name__])
            if probe is True:
                row("ok", "hermes triggers", f"{hermes_rows} lints clean; the hook refuses a call on a known row")
            else:
                row("fail", "hermes triggers", "registered, but the hook " +
                    ("stayed silent on a known row" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {cmd}")


if TYPE_CHECKING:                           # the faces this module has, checked by mypy
    from adapters import HookHost, Target, hermes as _self
    _hookhost: HookHost = _self
    _target: Target = _self
