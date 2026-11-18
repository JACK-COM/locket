"""trigger: ask a question at the tool boundary, for a fact that has already bitten.

Every other Locket mode answers a question someone asked: `audit` compares claims someone
wrote, `find` ranks a store someone queried. None reaches the agent that never searched,
because each runs only after the decision to look. A row here reaches that agent: when a
tool call or a prompt matches it, the row's question is put in front of the agent before
the call runs, and a `block` row refuses the call until its escape hatch is used.

Rows live in a store's `triggers.json`, or the file its manifest names under `triggers`.
The host's own store (Claude Code's `~/.claude`, Codex's `~/.codex`, Hermes's `~/.hermes`) fires everywhere;
any other store fires when the working directory is inside it. `locket trigger check`
lints every such file, and `locket trigger schema` prints every row key with what it does.

A row names Claude Code's tools, and Hermes's are mapped onto them (`terminal` is `Bash`,
`write_file` and `patch` are `Write` and `Edit`, `read_file` is `Read`); a Hermes tool
with no Claude Code twin is matched by its own name. Hermes has no way to add advice to a
tool call, so there a row that does not block refuses the call once instead: the agent
reads the question, and the same call repeated passes. A prompt row answers `pre_llm_call`
and reaches the model with the message it matched. Codex speaks Claude Code's payload, and
its file edits are patches, which are read as the Write or Edit they amount to; a row may
also name `apply_patch` itself.

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

import hosts
import fsio
import health
import manifest
import adapters
import corpora
import corpus
import trust

CAP = 12        # deliberately low: adding a row should mean arguing another one out

_STR_LIST = {"type": "array", "items": {"type": "string"}}
_REGEX_KEYS = ("tools", "content", "cwd", "unless", "unless_transcript")
ROW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "tools", "content", "question"],
    "properties": {
        "id": {"type": "string", "description": "Short name, printed in brackets before the question."},
        "tools": {"type": "string", "description": "Regex that must match the whole tool name, in Claude "
                                                   "Code's names, which Hermes's tools are mapped onto; a Hermes "
                                                   "name also matches. `UserPromptSubmit` matches the user's "
                                                   "prompt instead."},
        "content": {"type": "string", "description": "Regex searched over the tool input's strings, or the prompt."},
        "question": {"type": "string", "description": "What the agent is asked at that moment."},
        "cwd": {"type": "string", "description": "Regex the working directory must contain."},
        "fields": {**_STR_LIST, "description": "Tool-input keys to search; default every top-level string."},
        "unless": {"type": "string", "description": "Regex over the same text that silences the row: the "
                                                    "right form, or a stated decision, is already there."},
        "unless_transcript": {"type": "string", "description": "Regex that silences the row for the rest of "
                                                               "the session once it appears in the transcript "
                                                               "(on Hermes, in a stored message or tool call)."},
        "ignorecase": {"type": "boolean", "description": "Match `content` without case."},
        "block": {"type": "boolean", "description": "Refuse the tool call (exit 2) rather than add context; "
                                                   "the question must name the escape hatch. Ignored on a prompt. "
                                                   "On Hermes a row without it refuses the call once."},
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
    return manifest.triggers_file(root, trust.read_manifest(root).get("triggers"))


def stores_for(cwd, host=None):
    """The stores whose rows can fire here: the host's, then the one the directory is in."""
    host = hosts.claude() if host is None else host
    out = [host] if Path(host).is_dir() else []
    here = corpora.corpus_here(cwd)
    if here is not None and Path(here) not in [Path(h) for h in out]:
        out.append(Path(here))
    return out


def load(path):
    """The rows file as a dict; raises OSError or ValueError on an unreadable one."""
    data = json.loads(fsio.read_safe(path))     # a plain file under the size cap, never a FIFO
    if not isinstance(data, dict):
        raise ValueError("top level is not an object")
    return data


def _haystack(tool_input, fields):
    if fields:
        return " ".join(str(tool_input.get(f, "")) for f in fields)
    return " ".join(v for v in tool_input.values() if isinstance(v, str))


TRANSCRIPT_TAIL = 8 * 1024 * 1024
OVERLAP = 4096          # a match no longer than this is found across a window boundary
MAX_WINDOWS = 16        # 1 MB of a call read, at most


def _search(pattern, text, flags=0, unread=True):
    """`re.search` over `text`, never over more than fsio.MAX_SCAN at once: a long
    call is read in overlapping windows, so padding cannot push the matching part out
    of view. Past MAX_WINDOWS the rest is unread and answers `unread`: True for a
    row's content, so an oversized call fires the row, False for its escape hatch,
    so padding never stands in for the escape."""
    if len(text) <= fsio.MAX_SCAN:
        return re.search(pattern, text, flags) is not None
    rx = re.compile(pattern, flags)
    step = fsio.MAX_SCAN - OVERLAP
    for i in range(MAX_WINDOWS):
        start = i * step
        if start >= len(text):
            return False
        end = min(start + fsio.MAX_SCAN, len(text))
        # pos/endpos keep `^`, `\b` and lookbehind judged against the whole text; a match
        # ending at an inner window's edge may owe that to the edge (`$`, a lookahead), so
        # it is set aside: the next window, OVERLAP back, sees it with what follows it
        for m in rx.finditer(text, start, end):
            if m.end() < end or end == len(text):
                return True
        if end == len(text):
            return False
    return unread


def _transcript_has(path, pattern):
    """True when the transcript already carries `pattern`. An unreadable or absent one
    reads as "not yet", so the row fires rather than silently never firing."""
    if not path or not pattern:
        return False
    try:
        rx = re.compile(pattern)
        with open(path, "rb") as fh:            # the last TRANSCRIPT_TAIL bytes, where a recent
            fh.seek(0, os.SEEK_END)               # load or answer sits; never the whole log
            fh.seek(max(0, fh.tell() - TRANSCRIPT_TAIL))
            tail = fh.read().decode("utf-8", errors="replace")
        return any(rx.search(line[:fsio.MAX_SCAN]) for line in tail.splitlines())
    except (OSError, re.error, TypeError):     # TypeError: a row's pattern that is no string
        return False


def _seen(payload, host, pattern):
    """Whether this session already carries `pattern`: its transcript where the host hands
    one, else whichever record the sending host keeps."""
    if payload.get("transcript_path"):
        return _transcript_has(payload["transcript_path"], pattern)
    return adapters.sender(payload).seen(payload, host, pattern)


def _host_payload(payload):
    """Any host's payload in the shape `fire` reads, by the host that sent it."""
    if "_host" in payload:
        return payload
    return adapters.BY_NAME[adapters.host_of(payload)].trigger_event(payload)


def matches(row, tools, cwd, tool_input, seen=lambda pattern: False):
    """Conjunctive over whatever the row specifies; `tools` is every name the call goes by,
    and `seen` reads the session last, only after the cheap matches pass."""
    if not any(re.fullmatch(row["tools"], t) for t in tools):
        return False
    if row.get("cwd") and not re.search(row["cwd"], cwd):
        return False
    hay = _haystack(tool_input, row.get("fields"))
    if not _search(row["content"], hay, re.IGNORECASE if row.get("ignorecase") else 0):
        return False
    if row.get("unless") and _search(row["unless"], hay, unread=False):
        return False
    return not (row.get("unless_transcript") and seen(row["unless_transcript"]))


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
        with fsio.open_safe(path, newline="") as fh:
            rows = csv.reader(fh)
            next(rows, None)                # a header naming the label column is not a row carrying it
            if any(label in row for row in rows):
                return None
    except OSError as e:
        return f"cannot read {owner} ({e})"
    return f"names a label no row of {path.name} carries: {label}"


MAX_CAP = 40            # rows a file may run, whatever its `cap` says: each costs every call


def _cap(data):
    c = data.get("cap")
    return min(c, MAX_CAP) if isinstance(c, int) and not isinstance(c, bool) and c > 0 else CAP


def _show(path):
    home = str(Path.home())
    s = str(path)
    return fsio.relay("~" + s[len(home):] if s.startswith(home + os.sep) else s, 300)


def fire(payload, host=None):
    """(message, block) for a hook payload from either host; message is "" when nothing fired."""
    payload = _host_payload(payload)
    if host is None:
        host = adapters.sender(payload).home()
    cwd = payload.get("cwd") or os.getcwd()
    event = payload.get("hook_event_name") or "PreToolUse"
    if event == "UserPromptSubmit":
        tools, tool_input = ["UserPromptSubmit"], {"prompt": payload.get("prompt") or ""}
    else:
        tools = [payload.get("tool_name") or "", payload.get("_host_tool") or ""]
        tool_input = payload.get("tool_input") or {}
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
        cap = _cap(data)
        quiet = _quiet(cwd, data.get("quiet_under"))
        for row in rows[:cap]:
            if quiet and not row.get("everywhere"):
                continue
            try:
                if not matches(row, [t for t in tools if t], cwd, tool_input,
                               lambda pattern: _seen(payload, host, pattern)):
                    continue
            except (re.error, TypeError):
                continue    # a malformed row disables itself, never the rest of the file
            line = (f"[{fsio.relay(row.get('id', 'trigger'), 60)}] "
                    f"{fsio.relay(row['question'], 2000, lines=True)}")
            if row.get("owner"):
                opath, label = _owner(root, row["owner"])
                line += f"\n  -> {_show(opath)}" + (f"#{label}" if label else "")
            if row.get("pointer"):
                line += f"\n  -> {fsio.relay(row['pointer'], 300)}"
            parts.append(line)
            block |= bool(row.get("block")) and event != "UserPromptSubmit"
        if len(rows) > cap:
            parts.append(f"[over-cap] {_show(path)} holds {len(rows)} rows and runs {cap}; these are not "
                         f"running: {fsio.relay(', '.join(str(r.get('id', '?')) for r in rows[cap:]), 400)}. Argue one out or "
                         f"raise `cap` deliberately.")
    return "\n".join(parts), block


def hook(stdin=None, host=None):
    """The hook entry: JSON on stdin; context on stdout, or the question on stderr and
    exit 2 for a block. On Hermes a prompt's context is stdout JSON, and a tool call is
    answered through `adapters.emit`, which refuses it once for a row that does not block.
    Any failure exits 0, because a hook that breaks the tool it guards is worse than none."""
    with health.HookRun("trigger"):
        try:
            payload = json.load(stdin or sys.stdin)
            if not isinstance(payload, dict):
                return 0
            payload = _host_payload(payload)
            message, block = fire(payload, host)
        except Exception as e:
            health.health_error("trigger", "fire", e)
            return 0
        if not message:
            return 0
        return adapters.sender(payload).trigger_reply(payload, message, block, host)


def findings(root):
    """Every problem in a store's rows file, as lines; [] when clean or absent."""
    path = rows_file(root)
    named = trust.read_manifest(root).get("triggers")
    if not path.is_file():
        return [f"{_show(path)}: named by the manifest but missing"] if named else []
    try:
        data = load(path)
    except (OSError, ValueError) as e:
        return [f"{_show(path)}: unreadable ({e})"]
    import _manifest_schema
    out = [f"{_show(path)}: {f}" for f in _manifest_schema.check_value(data, SCHEMA, "")]
    rows = [r for r in data.get("rows") or [] if isinstance(r, dict)]
    cap = _cap(data)
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
        root = corpus.resolve(argv[0])
        if root is None:
            return 1
        if not trust.trusted(root):
            print(f"{root} is not trusted, so its rows file is not read; a person trusts it with  "
                  f"locket trust {root}", file=sys.stderr)
            return 1
        roots = [root]
    else:
        roots = stores_for(os.getcwd())
        for h in adapters.HOSTS:           # every other host's own store, after this one's
            other = h.home()
            if other.is_dir() and other not in roots:
                roots.insert(1, other)
    problems, files = [], 0
    for root in roots:
        path = rows_file(root)
        if path.is_file():
            files += 1
            try:
                n = str(len(load(path).get("rows") or []))
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
