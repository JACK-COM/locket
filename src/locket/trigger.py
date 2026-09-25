"""trigger: ask a question at the tool boundary, for a fact that has already bitten.

Every other Locket mode answers a question someone asked: `audit` compares claims someone
wrote, `find` ranks a store someone queried. None reaches the agent that never searched,
because each runs only after the decision to look. A row here reaches that agent: when a
tool call or a prompt matches it, the row's question is put in front of the agent before
the call runs, and a `block` row refuses the call until its escape hatch is used.

Rows live in a store's `triggers.json`, or the file its manifest names under `triggers`.
The host's own store (Claude Code's `~/.claude`) fires everywhere; any other store fires
when the working directory is inside it. `locket trigger check` lints every such file, and
`locket trigger schema` prints every row key with what it does.

A rows file's cheapness is its failure mode: a nudge the agent has learned to skim is no
nudge. So a file holds at most `cap` rows (default 12) and the hook names any beyond it
instead of running them; a row needs `tools` AND `content`, so none fires on a tool name
alone; and a row should point at the file owning its fact (`owner`), never restate it.
Nothing detects a row that has stopped firing; read the file against what has bitten since.
"""
import json
import os
import re
import sys
from pathlib import Path

import memscan

TRIGGERS = "triggers.json"
CAP = 12        # deliberately low: adding a row should mean arguing another one out

_STR_LIST = {"type": "array", "items": {"type": "string"}}
_REGEX_KEYS = ("tools", "content", "cwd", "unless", "unless_transcript")
ROW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "tools", "content", "question"],
    "properties": {
        "id": {"type": "string", "description": "Short name, printed in brackets before the question."},
        "tools": {"type": "string", "description": "Regex that must match the whole tool name; "
                                                   "`UserPromptSubmit` matches the user's prompt instead."},
        "content": {"type": "string", "description": "Regex searched over the tool input's strings, or the prompt."},
        "question": {"type": "string", "description": "What the agent is asked at that moment."},
        "cwd": {"type": "string", "description": "Regex the working directory must contain."},
        "fields": {**_STR_LIST, "description": "Tool-input keys to search; default every top-level string."},
        "unless": {"type": "string", "description": "Regex over the same text that silences the row: the "
                                                    "right form, or a stated decision, is already there."},
        "unless_transcript": {"type": "string", "description": "Regex that silences the row for the rest of "
                                                               "the session once it appears in the transcript."},
        "ignorecase": {"type": "boolean", "description": "Match `content` without case."},
        "block": {"type": "boolean", "description": "Refuse the tool call (exit 2) rather than add context; "
                                                   "the question must name the escape hatch. Ignored on a prompt."},
        "everywhere": {"type": "boolean", "description": "Fire under `quiet_under` too."},
        "owner": {"type": "string", "description": "The file owning this row's fact, store-relative or "
                                                   "absolute, or `<file>.csv#<label>` for rows of a reference "
                                                   "CSV; `locket trigger check` confirms it exists."},
        "_comment": {"type": "string", "description": "Why the row is shaped as it is, for its maintainer; "
                                                      "never shown to the agent."},
        "pointer": {"type": "string", "description": "Free text shown under the question."},
    },
}
SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Locket trigger rows (triggers.json)",
    "type": "object", "additionalProperties": False,
    "properties": {
        "$schema": {"type": "string", "description": "Editor hint only; ignored by the tool."},
        "_comment": {"type": ["string", "array"], "items": {"type": "string"},
                     "description": "Free text for the human reader, such as the admission test; ignored."},
        "quiet_under": {**_STR_LIST, "description": "Directories (`~` allowed) where rows stay silent "
                                                    "unless they set `everywhere`."},
        "cap": {"type": "integer", "description": f"Rows run at most; default {CAP}. Raising it is the "
                                                  "decision the number exists to force."},
        "rows": {"type": "array", "items": ROW_SCHEMA},
    },
}


def rows_file(root):
    """The rows file for a store: the manifest's `triggers`, else `triggers.json`."""
    name = memscan._read_manifest(root).get("triggers")
    p = Path(name).expanduser() if isinstance(name, str) and name else Path(TRIGGERS)
    return p if p.is_absolute() else Path(root) / p


def stores_for(cwd, host=None):
    """The stores whose rows can fire here: the host's, then the one the directory is in."""
    host = memscan.CLAUDE if host is None else host
    out = [host] if Path(host).is_dir() else []
    here = memscan.corpus_here(cwd)
    if here is not None and Path(here) not in [Path(h) for h in out]:
        out.append(Path(here))
    return out


def load(path):
    """The rows file as a dict; raises OSError or ValueError on an unreadable one."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("top level is not an object")
    return data


def _haystack(tool_input, fields):
    if fields:
        return " ".join(str(tool_input.get(f, "")) for f in fields)
    return " ".join(v for v in tool_input.values() if isinstance(v, str))


def _transcript_has(path, pattern):
    """True when the transcript already carries `pattern`. An unreadable or absent one
    reads as "not yet", so the row fires rather than silently never firing."""
    if not path or not pattern:
        return False
    try:
        rx = re.compile(pattern)
        with open(path, encoding="utf-8", errors="replace") as fh:
            return any(rx.search(line) for line in fh)
    except (OSError, re.error):
        return False


def matches(row, tool, cwd, tool_input, transcript=None):
    """Conjunctive over whatever the row specifies; the transcript is read last, only
    after the cheap matches pass."""
    if not re.fullmatch(row["tools"], tool):
        return False
    if row.get("cwd") and not re.search(row["cwd"], cwd):
        return False
    hay = _haystack(tool_input, row.get("fields"))
    if not re.search(row["content"], hay, re.IGNORECASE if row.get("ignorecase") else 0):
        return False
    if row.get("unless") and re.search(row["unless"], hay):
        return False
    return not _transcript_has(transcript, row.get("unless_transcript"))


def _quiet(cwd, dirs):
    cwd = Path(cwd)
    for d in dirs if isinstance(dirs, list) else []:
        d = Path(str(d)).expanduser()
        if cwd == d or d in cwd.parents:
            return True
    return False


def _owner(root, owner):
    """(path, label) for a row's owner; label is the CSV row label after `#`, or None."""
    name, _, label = str(owner).partition("#")
    p = Path(name).expanduser()
    return (p if p.is_absolute() else Path(root) / p), (label or None)


def _owner_missing(root, owner):
    """Why an owner does not resolve, or None when it does."""
    path, label = _owner(root, owner)
    if not path.exists():
        return f"names a missing file: {owner}"
    if label is None:
        return None
    import csv
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rows = csv.reader(fh)
            next(rows, None)                # a header naming the label column is not a row carrying it
            if any(label in row for row in rows):
                return None
    except OSError as e:
        return f"cannot read {owner} ({e})"
    return f"names a label no row of {path.name} carries: {label}"


def _show(path):
    home = str(Path.home())
    s = str(path)
    return "~" + s[len(home):] if s.startswith(home + os.sep) else s


def fire(payload, host=None):
    """(message, block) for a hook payload; message is "" when nothing fired."""
    cwd = payload.get("cwd") or os.getcwd()
    event = payload.get("hook_event_name") or "PreToolUse"
    if event == "UserPromptSubmit":
        tool, tool_input = "UserPromptSubmit", {"prompt": payload.get("prompt") or ""}
    else:
        tool, tool_input = payload.get("tool_name") or "", payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return "", False
    parts, block = [], False
    for root in stores_for(cwd, host):
        path = rows_file(root)
        if not path.is_file():
            continue
        try:
            data = load(path)
        except (OSError, ValueError):
            continue        # a broken file never takes the tool down; `doctor` and `check` name it
        rows = [r for r in data.get("rows") or [] if isinstance(r, dict)
                and all(isinstance(r.get(k), str) and r[k] for k in ("tools", "content", "question"))]
        cap = data.get("cap") if isinstance(data.get("cap"), int) and data.get("cap") > 0 else CAP
        quiet = _quiet(cwd, data.get("quiet_under"))
        for row in rows[:cap]:
            if quiet and not row.get("everywhere"):
                continue
            try:
                if not matches(row, tool, cwd, tool_input, payload.get("transcript_path")):
                    continue
            except (re.error, TypeError):
                continue    # a malformed row disables itself, never the rest of the file
            line = f"[{row.get('id', 'trigger')}] {row['question']}"
            if row.get("owner"):
                path, label = _owner(root, row["owner"])
                line += f"\n  -> {_show(path)}" + (f"#{label}" if label else "")
            if row.get("pointer"):
                line += f"\n  -> {row['pointer']}"
            parts.append(line)
            block |= bool(row.get("block")) and event != "UserPromptSubmit"
        if len(rows) > cap:
            parts.append(f"[over-cap] {_show(path)} holds {len(rows)} rows and runs {cap}; these are not "
                         f"running: {', '.join(str(r.get('id', '?')) for r in rows[cap:])}. Argue one out or "
                         f"raise `cap` deliberately.")
    return "\n".join(parts), block


def hook(stdin=None):
    """The hook entry: JSON on stdin; context on stdout, or the question on stderr and
    exit 2 for a block. Any failure exits 0, because a hook that breaks the tool it
    guards is worse than no hook."""
    try:
        payload = json.load(stdin or sys.stdin)
        if not isinstance(payload, dict):
            return 0
        message, block = fire(payload)
    except Exception:
        return 0
    if not message:
        return 0
    if block:
        print(message, file=sys.stderr)
        return 2
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": payload.get("hook_event_name") or "PreToolUse", "additionalContext": message}}))
    return 0


def findings(root):
    """Every problem in a store's rows file, as lines; [] when clean or absent."""
    path = rows_file(root)
    named = memscan._read_manifest(root).get("triggers")
    if not path.is_file():
        return [f"{_show(path)}: named by the manifest but missing"] if named else []
    try:
        data = load(path)
    except (OSError, ValueError) as e:
        return [f"{_show(path)}: unreadable ({e})"]
    import _manifest_schema
    out = [f"{_show(path)}: {f}" for f in _manifest_schema.check_value(data, SCHEMA, "")]
    rows = [r for r in data.get("rows") or [] if isinstance(r, dict)]
    cap = data.get("cap") if isinstance(data.get("cap"), int) and data.get("cap") > 0 else CAP
    if len(rows) > cap:
        out.append(f"{_show(path)}: {len(rows)} rows against a cap of {cap}; the last {len(rows) - cap} never run")
    seen = set()
    for i, r in enumerate(rows):
        rid = r.get("id") or f"rows[{i}]"
        if rid in seen:
            out.append(f"{_show(path)}: id {rid!r} appears twice")
        seen.add(rid)
        for k in _REGEX_KEYS:
            if isinstance(r.get(k), str):
                try:
                    re.compile(r[k])
                except re.error as e:
                    out.append(f"{_show(path)}: {rid}.{k} is not a regex ({e})")
        if isinstance(r.get("owner"), str):
            why = _owner_missing(root, r["owner"])
            if why:
                out.append(f"{_show(path)}: {rid}.owner {why}")
    return out


def check(argv):
    """`locket trigger check [corpus]`: lint the rows files that can fire from here, or one store's."""
    if argv:
        root = memscan.resolve(argv[0])
        if root is None:
            return 1
        roots = [root]
    else:
        roots = stores_for(os.getcwd())
    problems, files = [], 0
    for root in roots:
        path = rows_file(root)
        if path.is_file():
            files += 1
            try:
                n = len(load(path).get("rows") or [])
            except (OSError, ValueError, AttributeError):
                n = "?"
            print(f"{_show(path)}: {n} rows")
        problems += findings(root)
    for p in problems:
        print(p)
    print(f"checked {files} rows file(s) in {', '.join(_show(r) for r in roots) or 'no store'}: "
          + (f"{len(problems)} problem(s)" if problems else "clean"))
    return 1 if problems else 0


def main(argv):
    if argv[:1] == ["check"]:
        return check(argv[1:])
    if argv[:1] == ["schema"]:
        print(json.dumps(SCHEMA, indent=2))
        return 0
    if argv or sys.stdin.isatty():
        print("usage: locket trigger [check [corpus] | schema]; with no argument it is the hook, "
              "JSON on stdin", file=sys.stderr)
        return 1
    return hook()


def selftest():
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
        (host / TRIGGERS).write_text(json.dumps({
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
            ]}))
        (proj / "rules.json").write_text(json.dumps({"cap": 1, "rows": [
            {"id": "proj-row", "tools": "Write", "content": "enum", "question": "Name, not value.",
             "owner": str(host / "RULINGS.csv#practice")},
            {"id": "second", "tools": "Write", "content": "x", "question": "Never runs."}]}))

        other = Path(t) / "elsewhere"       # in no store but the host's
        other.mkdir()

        def run(tool, inp, cwd=other, **kw):
            return fire({"tool_name": tool, "tool_input": inp, "cwd": str(cwd), **kw}, host=host)

        msg, block = run("Bash", {"command": "git reset --hard"})
        assert "[git-reset]" in msg and not block, msg
        assert f"-> {_show(host / 'memory/owner.md')}" in msg, "the owner was not shown"
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
        msg, block = fire({"hook_event_name": "UserPromptSubmit", "prompt": "Ask Hermes", "cwd": str(other)},
                          host=host)
        assert "[prompt]" in msg and not block, "a prompt row matched wrongly, or blocked a prompt"

        assert findings(host) == [] and findings(other) == [], findings(host)
        bad = json.loads((host / TRIGGERS).read_text())
        bad["rows"] += [{"id": "git-reset", "tools": "(", "content": "x", "question": "q",
                         "owner": "memory/gone.md", "blok": True},
                        {"id": "csv", "tools": "x", "content": "x", "question": "q", "owner": "RULINGS.csv#nope"}]
        (host / TRIGGERS).write_text(json.dumps(bad))
        found = "\n".join(findings(host))
        for want in ("unknown key 'blok'", "appears twice", "tools is not a regex", "missing file: memory/gone.md",
                     "no row of RULINGS.csv carries: nope"):
            assert want in found, f"the lint missed {want!r}:\n{found}"
        assert "over-cap" not in found and "cap of 1" in "\n".join(findings(proj)), "the cap went unlinted"
        (proj / "rules.json").unlink()
        assert "named by the manifest but missing" in "\n".join(findings(proj))

        out = io.StringIO()
        saved = sys.stdout
        sys.stdout = out
        try:
            rc = hook(io.StringIO("not json"))
        finally:
            sys.stdout = saved
        assert rc == 0 and out.getvalue() == "", "a bad payload was not ignored"
    print("trigger selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["--selftest"] else main(sys.argv[1:]))
