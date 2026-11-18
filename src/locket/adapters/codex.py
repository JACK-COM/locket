"""Codex: Claude Code's event names, told apart by the fields only Codex sends, with its
file edits arriving as patches and a refusal sent as a deny decision on stdout."""
import json
import re
from pathlib import Path

import corpora
import gate
import hosts
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
