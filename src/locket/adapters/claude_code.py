"""Claude Code: the shape every hook mode is written against, and the fallback host, so
it detects every payload the hosts before it in `adapters.HOSTS` did not claim."""
import json
import sys

import hosts

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
