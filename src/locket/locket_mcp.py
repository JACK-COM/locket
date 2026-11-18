#!/usr/bin/env python3
"""locket_mcp: Locket's memory checks as MCP tools, for a host with no hooks.

Claude Desktop with a filesystem connector can read and write memory files and
cannot run a command or intercept a write, so the hooks in `memscan.py` never
reach it. What it can do is spawn a local MCP server. This is that server: a
stdio JSON-RPC process wrapping the same functions the hooks and the CLI use, so
a Desktop session can ask the same questions by tool call.

Where a host has hooks, this is a convenience and never the gate. Where it does
not, the gate is advisory by construction: nothing fires unless the agent calls
`check_before_write` before writing, so the discipline lives in the project's
instructions and this server answers when asked. The server process lives as
long as the client keeps it, which is what lets `check_before_write` know whether
`find` was asked without a transcript to read. ⚠ Claude Desktop spawns a server
once per app launch and shares it across every conversation, so "asked" there
means asked in ANY conversation since the last restart; MCP hands the server no
conversation identity, so this is disclosed rather than fixed.

The standard library plus numpy, which memfind's ranking needs. Framing is newline-delimited JSON on
stdin/stdout, written through the binary buffers so Windows never turns a `\\n`
into `\\r\\n` and breaks the frame.

    locket_mcp.py            serve on stdio (what the client config runs)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli
from commands import Session, Tool

PROTOCOL = "2025-03-26"
SESSION = Session()     # has this server process run `find`? The 2e question, in-process.


def _tools():
    """Every verb's MCP face under its tool name, and each older alias under its own."""
    out = {}
    for v in cli.VERBS:
        if v.mcp is None:
            continue
        name = v.mcp.name or f"locket_{v.name}"
        out[name] = v.mcp
        for a in v.aliases:
            out[f"locket_{a}"] = Tool(f"Older name of {name}.", v.mcp.call, v.mcp.props)
    return out


TOOLS = _tools()


def tool_list():
    return [{"name": n, "description": t.description,
             "inputSchema": {"type": "object", "properties": t.props,
                             "required": [k for k in t.props if k in ("statement", "text", "path")]}}
            for n, t in TOOLS.items()]


# ---------------------------------------------------------------- JSON-RPC

def handle(msg):
    """One request or notification in, one response out (None for a notification)."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "locket", "version": "1"}}}
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tool_list()}}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": f"unknown tool {name!r}"}}
        try:
            text = TOOLS[name].call(args, SESSION)
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": text}], "isError": False}}
        except Exception as e:                      # a tool error is a result, not a crash
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                               "isError": True}}
    if mid is None:
        return None                                  # an unknown notification is ignored
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    for raw in inp:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = handle(msg) if isinstance(msg, dict) else None
        if resp is not None:
            out.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            out.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__); sys.exit(0)
    try:
        serve()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
