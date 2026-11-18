"""What a host calls at its boundaries: the write-time hooks (one JSON payload on stdin
each, never for manual use), the trigger hook, and the usage ledger's SessionEnd hook,
with the reports a person reads off the last two.
"""
import os
import sys
from pathlib import Path

import adapters
import bashguard
import gate
import grepassist
import health
import ledger
from commands import Tool, Verb


def hook_mode(payload):
    """The write gate on one normalised Write or Edit, delivered to the host. Reached
    from `hook`, and from `bashguard` when a shell call turned out to be a Codex patch."""
    ti = payload["tool_input"]
    d = gate.evaluate_write(str(ti.get("file_path", "")), ti, append=bool(ti.get("_append")),
                            autonomous=os.environ.get("CLAUDE_AUTONOMOUS") == "1", payload=payload)
    return adapters.emit(payload, d.message, d.blocking, "hook")


def _write():
    payload = adapters.read_payload()
    return 0 if payload is None else hook_mode(payload)


def _shell():
    """PreToolUse on Bash: the writes the write gate never sees."""
    payload = adapters.read_payload()
    if payload is not None and payload["tool_name"] in ("Write", "Edit"):
        return hook_mode(payload)           # a Codex patch sent through the shell
    if payload is None or payload["tool_name"] != "Bash":
        return 0
    try:
        msg, blocking = bashguard.decide(str(payload["tool_input"].get("command", "")), payload.get("cwd"),
                                         autonomous=os.environ.get("CLAUDE_AUTONOMOUS") == "1")
    except Exception as e:                  # a guardrail must never break the shell
        health.health_error("bashguard", "decision", e)
        return 0
    return adapters.emit(payload, msg, blocking, "bashguard")


def _search():
    """PreToolUse on Grep or Bash: the store's ranked neighbours, before the search runs."""
    try:
        payload = adapters.read_payload()
    except Exception as e:                  # a payload normalise cannot read
        health.health_error("grepassist", "rank", e)
        return grepassist.log("?", "declined", f"error:{type(e).__name__}", "")
    if payload is None:
        return 0
    return adapters.emit(payload, grepassist.advise(payload), False, "grepassist")


def _entry(verb, fn):
    """`run` for a write-time hook: its run recorded, so one a host abandoned or whose
    catch-all swallowed an error shows in `locket doctor`."""
    def run(rest):
        with health.HookRun(verb):
            return fn()
    return run


def _retired(rest):
    sys.stdin.read()                        # an older Hermes config still calls `deliver`
    return 0


def _check_tool(args, session):
    """The write-time hook, on request, through the hook's own gate: the registry and
    config refusals, a CSV's schema and source rows, the one-authority nudge and the 2e
    question, over the text about to be written. It cannot stop the write, so what the
    hook would refuse comes back as advice saying the write should not land."""
    text = str(args.get("text") or "")
    path = str(args.get("path") or "")
    old = str(args.get("old_text") or "")
    if not text or not path:
        raise ValueError("text and path are required")
    p = Path(path).expanduser()
    # Three shapes, and the caller has to say which, because a filesystem connector
    # writes whole files and the hook cannot see the operation behind them:
    # `old_text` is an edit; `append` is an entry going on the end; otherwise `text`
    # is the whole new file, so a rewrite in place scores as nothing added, which
    # is the discipline being followed and must not draw a nudge naming the file.
    ti = {"old_string": old, "new_string": text} if old else {"content": text}
    d = gate.evaluate_write(str(p), ti, append=bool(args.get("append")) and not old,
                            payload={"_host": "mcp", "_asked": session.asked})
    if d.message:
        return (d.message.replace("the write is refused", "the write should not land")
                .replace(ledger.LEAD_WARNING, "About to write:")
                .replace(ledger.LEAD_BLOCKING, "About to write:"))
    return {
        "outside": (f"{p} is not inside a memory store this tool knows, so nothing to check. Call "
                    f"locket_corpora; a folder becomes a store once a person runs  locket trust <folder>."),
        "ledger": "No concerns: every row fits the ledger's schema.",
        "csv": (f"No concerns: {p.name} has no ledger schema, and no row it adds restates "
                f"a claim a registered store already carries."),
        "error": "The check failed inside Locket, so this text is unchecked; `locket doctor` reports the error.",
    }.get(d.checked, "No concerns: nothing here already owns this text, and the session has asked the store.")


def _doc_guide(module):
    """`locket help <verb>`: the verb's `-h`, then its module docstring past the title."""
    def guide(own):
        import importlib
        own()
        print("\n" + (importlib.import_module(module).__doc__ or "").split("\n\n", 1)[-1].rstrip())
        return 0
    return guide


def _trigger(rest):
    import trigger
    return trigger.main(rest)


def _usage(rest):
    import usage
    return usage.hook() if rest == ["hook"] else usage.main(rest)


VERBS = (
    Verb("trigger", "[check [corpus] | schema]",
         "the tool-boundary hook: when a tool call or prompt matches a row in a store's triggers.json, "
         "put the row's question to the agent; `check` lints the rows files, `schema` prints their schema",
         "locket trigger check                  the rows that can fire from here\n"
         "locket trigger check council          one store's rows\n"
         "locket trigger schema                 the JSON Schema a triggers.json row follows\n"
         "locket help trigger                   how rows are written and when they fire",
         run=_trigger, guide=_doc_guide("trigger")),
    Verb("usage", "[today | week | month | record] [--by project|model|day] [--json]",
         "tokens Claude Code and Hermes spent, per source and project, against a typical day; "
         "keeps a ledger that outlives the transcripts",
         "locket usage                          the last 7 days, per project\n"
         "locket usage today                    today; `month` is the last 30 days\n"
         "locket usage month --by model         30 days, per model; `--by day` is one row a day\n"
         "locket usage --json                   the same summary as JSON\n"
         "locket usage record                   copy new sessions into the ledger now (the\n"
         "                                      SessionEnd hook does this on its own)",
         run=_usage, guide=_doc_guide("usage")),
    Verb("hook", hidden=True, run=_entry("hook", _write),
         mcp=Tool("Run the write-time checks on text you are about to write into a "
                  "memory file: which file already owns it, and whether this session "
                  "has asked the store yet. Advisory; read it, then write.", _check_tool,
                  {"text": {"type": "string", "description":
                            "The text about to be written: the whole new file, or with "
                            "`append` true just the entry being added to the end."},
                   "path": {"type": "string", "description": "Absolute path of the memory file."},
                   "old_text": {"type": "string", "description":
                                "For an edit, the text being replaced; `text` is then its replacement."},
                   "append": {"type": "boolean", "description":
                              "True when `text` is an entry appended to the file rather than "
                              "the whole new file (default false)."}},
                  name="locket_check_before_write")),
    Verb("bashguard", hidden=True, run=_entry("bashguard", _shell)),
    Verb("grepassist", hidden=True, run=_entry("grepassist", _search)),
    Verb("deliver", hidden=True, run=_retired),
)

# The PreToolUse entry points, which `memscan.py <verb>` reaches without the CLI.
ENTRIES = {v.name: v for v in VERBS if v.hidden}
