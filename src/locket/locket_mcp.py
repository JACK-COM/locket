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

Stdlib only, like everything else here. Framing is newline-delimited JSON on
stdin/stdout, written through the binary buffers so Windows never turns a `\\n`
into `\\r\\n` and breaks the frame.

    locket_mcp.py            serve on stdio (what the client config runs)
"""
import contextlib
import io
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memscan
import memfind

PROTOCOL = "2025-03-26"
ASKED = False           # has this session run `find`? The 2e question, in-process.


# ---------------------------------------------------------------- stores

def _store(name):
    """A corpus path from a name, or the only store on the machine when no name
    is given, else a readable error the model can act on."""
    if name:
        root = memscan.resolve(str(name))
        if root is None:
            raise ValueError(f"no memory store matches {name!r}; call locket_corpora")
        return root
    stores = {k: v for k, v in memscan.all_corpora().items() if memscan.md_files(v)}
    if len(stores) == 1:
        return next(iter(stores.values()))
    raise ValueError("more than one memory store here; name one: "
                     + ", ".join(sorted(stores)) + ", or 'all'")


def _capture(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(memscan.SafeOut(buf)):
        fn(*a, **kw)
    return buf.getvalue()


def _over_stores(name, fn):
    """Run a CLI-shaped mode over one store or, for 'all', every store with files."""
    if name == "all":
        out = []
        for k, v in sorted(memscan.all_corpora().items()):
            if memscan.md_files(v):
                out.append(f"== {k}\n" + _capture(fn, v))
        return "\n".join(out) or "no memory stores with markdown files found"
    return _capture(fn, _store(name))


# ---------------------------------------------------------------- tools

def t_corpora(args):
    # The same counts `audit` reports, from the same function the CLI prints:
    # a listing that counted markdown alone read as a half-indexed store.
    return "\n".join(memscan.corpora_lines(60)) or "no memory stores found on this machine"


def t_init(args):
    """Write a folder's manifest. Never trusts it: trust is a person's act at a
    terminal (`locket trust`), so the reply names that command instead."""
    d = str(args.get("path") or "").strip()
    if not d:
        raise ValueError("path is required: the absolute path of the folder to make a store")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = memscan.cmd_init(d, confirm=lambda _d: False)
    return (out.getvalue() + err.getvalue()).strip() or f"init returned {rc}"


def t_forget(args):
    d = str(args.get("path") or "").strip()
    if not d:
        raise ValueError("path is required")
    if any(memscan._same(Path(d).expanduser(), r) for r in memscan.registry()):
        return (f"{d} is trusted, and withdrawing a store's guardrails is a person's act: ask the user "
                f"to run  locket untrust {d}  at a terminal")
    return f"{d} is not in the registry"


def t_find(args):
    global ASKED
    statement = str(args.get("statement") or "").strip()
    if not statement:
        raise ValueError("statement is required")
    top = int(args.get("top") or 5)
    name = args.get("store")
    roots = ([v for k, v in sorted(memscan.all_corpora().items()) if memscan.md_files(v)]
             if name == "all" else [_store(name)])
    out = []
    for root in roots:
        rows = memfind.rank(statement, root, top=top, quiet=True)
        out.append(f"store: {root}")
        out.extend(f"[{c:.3f}] {memscan.relay(f, 160)}\n        {memscan.relay(s, 190)}" for c, f, s in rows)
        if memfind.LAST_RANK_MODE == "lexical":
            out.append("WORD OVERLAP ONLY: no embedder could be reached, so this cannot see a "
                       "paraphrase and a silent result is not evidence the store is silent.")
    ASKED = True
    out.append("Ranked candidates, not a verdict. Read the top file before writing; "
               "this tool cannot tell you a fact is new.")
    return "\n".join(out)


def t_across(args):
    """Claim pairs whose two sides sit in different stores, over every store on
    the machine at once. The one check no single store can run on itself."""
    return _capture(memscan.cmd_across) or "no memory stores with markdown files found"


def t_check_before_write(args):
    """The write-time hook, on request, through the hook's own gate: the registry and
    config refusals, a CSV's schema and source rows, the one-authority nudge and the 2e
    question, over the text about to be written. Advisory; nothing here blocks."""
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
    d = memscan.evaluate_write(str(p), ti, append=bool(args.get("append")) and not old,
                               payload={"_host": "mcp", "_asked": ASKED})
    if d.message:
        return (d.message.replace("the write is refused", "the write should not land")
                .replace(memscan.LEAD_WARNING, "About to write:")
                .replace(memscan.LEAD_BLOCKING, "About to write:"))
    return {
        "outside": (f"{p} is not inside a memory store this tool knows, so nothing to check. Call "
                    f"locket_corpora; a folder becomes a store once a person runs  locket trust <folder>."),
        "ledger": "No concerns: every row fits the ledger's schema.",
        "csv": (f"No concerns: {p.name} has no ledger schema, and no row it adds restates "
                f"a claim a registered store already carries."),
        "error": "The check failed inside Locket, so this text is unchecked; `locket doctor` reports the error.",
    }.get(d.checked, "No concerns: nothing here already owns this text, and the session has asked the store.")


def t_audit(args):
    return _over_stores(args.get("store"), memscan.cmd_audit)


def t_dupes(args):
    files = args.get("files") or None
    if isinstance(files, str):              # a bare string would be matched as a substring
        files = [files]
    return _over_stores(args.get("store"), lambda r: memscan.cmd_dupes(r, files=files))


def t_budget(args):
    return _over_stores(args.get("store"), memscan.cmd_budget)


def t_graduated(args):
    return _over_stores(args.get("store"), memscan.cmd_graduated)


def t_links(args):
    return _over_stores(args.get("store"), memscan.cmd_links)


def t_pointers(args):
    return _over_stores(args.get("store"), memscan.cmd_pointers)


def t_siblings(args):
    root = _store(args.get("store"))
    rows = memfind.siblings(root, top=int(args.get("top") or 20), quiet=True)
    out = [f"[subject {c:.3f} | max claim {m:.3f}] {fa}  ::  {fb}" for c, m, fa, fb in rows]
    out.append("High subject with a LOW max claim is one subject written twice, a merge "
               "candidate; low subject with a max near 1.00 is shared boilerplate. Neither "
               "is a verdict, and contradictions are not detected: this is the shortlist a "
               "reader adjudicates.")
    return "\n".join(out)


def t_status(args):
    try:
        name, model = memfind.resolve_backend()
        emb = f"{name} ({model})"
    except RuntimeError as e:
        emb = "none; " + str(e).splitlines()[0]
    return "\n".join([
        f"locket on {platform.system()} {platform.release()}, python {platform.python_version()} "
        f"at {sys.executable}",
        f"embedder: {emb}",
        f"asked this session: {ASKED}  (a session here is this server process, which the app "
        f"shares across its conversations until it restarts)",
        memscan.gate_status(),
    ])


STORE_ARG = {"type": "string", "description":
             "A memory store: a shorthand or path from locket_corpora, or 'all'. "
             "Omit it when the machine has one store."}
TOOLS = {
    "locket_corpora": (t_corpora, "Every memory store on this machine, with markdown file and "
                                  "registered CSV-row counts, the sources every check reads.", {}),
    "locket_init": (t_init, "Make a folder of markdown a memory store every other tool can find: "
                            "writes locket.json if absent. It does not trust the folder: a person must "
                            "run `locket trust <path>` at a terminal before the store acts, and "
                            "the reply names that command for you to pass on. Idempotent.",
                    {"path": {"type": "string", "description": "Absolute path of the folder."}}),
    "locket_forget": (t_forget, "Drop a store from the registry. The folder's locket.json and "
                                "stays in the folder, inert; its index under ~/.locket/cache stays until uninstall.",
                      {"path": {"type": "string", "description": "Absolute path of the folder."}}),
    "locket_find": (t_find,
                    "Which memory files already hold this idea, ranked by meaning. Call it "
                    "BEFORE writing a durable fact and read the top file; a paraphrase shares "
                    "no words with the claim it restates, so a grep finding nothing proves nothing.",
                    {"statement": {"type": "string", "description": "The fact, in one sentence."},
                     "store": STORE_ARG,
                     "top": {"type": "integer", "description": "Rows to return (default 5)."}}),
    "locket_check_before_write": (t_check_before_write,
                                  "Run the write-time checks on text you are about to write into a "
                                  "memory file: which file already owns it, and whether this session "
                                  "has asked the store yet. Advisory; read it, then write.",
                                  {"text": {"type": "string", "description":
                                            "The text about to be written: the whole new file, or with "
                                            "`append` true just the entry being added to the end."},
                                   "path": {"type": "string", "description": "Absolute path of the memory file."},
                                   "old_text": {"type": "string", "description":
                                                "For an edit, the text being replaced; `text` is then its replacement."},
                                   "append": {"type": "boolean", "description":
                                              "True when `text` is an entry appended to the file rather than "
                                              "the whole new file (default false)."}}),
    "locket_audit": (t_audit, "Claims that appear in two files: fork risk, not proof.",
                     {"store": STORE_ARG}),
    "locket_across": (t_across, "Claims held by two different stores, over every store on "
                                "this machine at once. Each store's own checks score it "
                                "against itself, so a fact written into two stores is "
                                "invisible to both until this runs. Fork risk, not proof.", {}),
    "locket_dupes": (t_dupes, "Claims repeated inside one file (holding spaces by default).",
                     {"store": STORE_ARG,
                      "files": {"type": "array", "items": {"type": "string"},
                                "description": "File names to check instead of the defaults."}}),
    "locket_budget": (t_budget, "Files past a size, update-clause or paragraph budget.",
                      {"store": STORE_ARG}),
    "locket_graduated": (t_graduated, "Holding-space claims a durable file or CSV row now holds: "
                                      "entries promoted and never retired. Prints its scope even "
                                      "when clean. Run with locket_links at session close.",
                         {"store": STORE_ARG}),
    "locket_links": (t_links, "Wikilinks that resolve to no file.", {"store": STORE_ARG}),
    "locket_pointers": (t_pointers, "Index lines paired with the files they point at, a "
                                    "missing target first. Whether the write-time gate is "
                                    "armed is locket_status.", {"store": STORE_ARG}),
    "locket_hooks": (t_pointers, "Older name of locket_pointers.", {"store": STORE_ARG}),
    "locket_siblings": (t_siblings, "File pairs sharing one subject: the shortlist for merges "
                                    "and for a reader looking for contradictions.",
                        {"store": STORE_ARG,
                         "top": {"type": "integer", "description": "Pairs to return (default 20)."}}),
    "locket_status": (t_status, "Host, interpreter, embedder rung, and whether this session "
                                "has asked the store yet.", {}),
}


def tool_list():
    return [{"name": n, "description": d,
             "inputSchema": {"type": "object", "properties": props,
                             "required": [k for k in props if k in ("statement", "text", "path")]}}
            for n, (_, d, props) in TOOLS.items()]


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
            text = TOOLS[name][0](args)
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
