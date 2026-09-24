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
    locket_mcp.py --selftest spawn itself and drive one full exchange
"""
import contextlib
import io
import json
import os
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
    with contextlib.redirect_stdout(buf):
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
    """The one call for a folder that already holds markdown: manifest, registry
    row, index. `memscan.cmd_init` prints its report; stderr carries a refusal."""
    d = str(args.get("path") or "").strip()
    if not d:
        raise ValueError("path is required: the absolute path of the folder to make a store")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = memscan.cmd_init(d)
    return (out.getvalue() + err.getvalue()).strip() or f"init returned {rc}"


def t_forget(args):
    d = str(args.get("path") or "").strip()
    if not d:
        raise ValueError("path is required")
    return (f"{d}: registry row removed. Its locket.json and .memfind/ are yours to delete."
            if memscan.forget(d) else f"{d} is not in the registry")


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
        out.extend(f"[{c:.3f}] {f}\n        {s[:190]}" for c, f, s in rows)
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
    """The write-time hook, on request: the one-authority nudge and the 2e
    question, over the text about to be written. Advisory; nothing here blocks."""
    text = str(args.get("text") or "")
    path = str(args.get("path") or "")
    old = str(args.get("old_text") or "")
    if not text or not path:
        raise ValueError("text and path are required")
    p = Path(path).expanduser()
    if p.suffix.lower() == ".csv":
        # What the hook does with a CSV: the ledger schema, which may block, then
        # a registered source's new rows scored against the store, which warns.
        ti = ({"old_string": old, "new_string": text} if old
              else {"new_string": text} if args.get("append") else {"content": text})
        msg, _ = memscan.ledger_decision(str(p), ti)
        if msg:
            return msg.replace("the write is refused", "the write should not land")
        try:
            prior = p.read_text(errors="ignore")
        except OSError:
            prior = ""
        msg, _ = memscan.source_row_decision(str(p), ti, prior, append=bool(args.get("append")))
        if msg:
            return msg.replace(memscan.LEAD_WARNING, "About to write:")
        if memscan.ledger_schema_for(p)[0] is not None:
            return "No concerns: every row fits the ledger's schema."
        return (f"No concerns: {p.name} has no ledger schema, and no row it adds restates "
                f"a claim a registered store already carries.")
    root = memscan.corpus_for(p)
    if root is None:
        return (f"{p} is not inside a memory store this tool knows, so nothing to check. "
                f"Call locket_corpora, or put a locket.json in the store's directory.")
    try:
        prior = p.read_text(errors="ignore")
    except OSError:
        prior = ""
    # Three shapes, and the caller has to say which, because a filesystem connector
    # writes whole files and the hook cannot see the operation behind them:
    # `old_text` is an edit; `append` is an entry going on the end; otherwise `text`
    # is the whole new file, so a rewrite in place scores as nothing added, which
    # is the discipline being followed and must not draw a nudge naming the file.
    ti = {"file_path": str(p)}
    if old:
        ti.update(old_string=old, new_string=text)
    elif args.get("append") and prior:
        text = prior.rstrip("\n") + "\n" + text
    msg, _ = memscan.hook_decision(
        p.name, ti, text, prior, lambda: memscan.corpus(root),
        lambda: memscan.archived_claims(root), autonomous=False,
        sets=memscan.settings(root))
    gate = memscan.memfind_gate({"_host": "mcp", "_asked": ASKED}, p.name, ti, text, prior, root)
    parts = [x for x in (msg, gate) if x]
    if not parts:
        return "No concerns: nothing here already owns this text, and the session has asked the store."
    return "\n\n".join(parts).replace(memscan.LEAD_WARNING, "About to write:")


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
                            "writes locket.json if absent, registers the path, builds the index. "
                            "Idempotent. Call it when someone points you at a folder and says it "
                            "is their memory, vault or notes.",
                    {"path": {"type": "string", "description": "Absolute path of the folder."}}),
    "locket_forget": (t_forget, "Drop a store from the registry. The folder's locket.json and "
                                ".memfind/ are left for the user to delete.",
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


# ---------------------------------------------------------------- selftest

def selftest():
    import subprocess, tempfile
    scratch = Path(tempfile.mkdtemp())
    # a scratch registry, so `locket_init` below never touches ~/.locket
    p = subprocess.Popen([sys.executable, __file__], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env={**os.environ, "LOCKET_REGISTRY": str(scratch / "corpora.json")})

    def call(msg):
        p.stdin.write((json.dumps(msg) + "\n").encode()); p.stdin.flush()
        if "id" not in msg:
            return None
        line = p.stdout.readline()
        assert line, f"server closed after {msg}: {p.stderr.read()[:400]}"
        return json.loads(line)

    r = call({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "t"}}})
    assert r["result"]["serverInfo"]["name"] == "locket", r
    call({"jsonrpc": "2.0", "method": "notifications/initialized"})
    r = call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in r["result"]["tools"]}
    assert names == set(TOOLS), names
    r = call({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "locket_status", "arguments": {}}})
    assert not r["result"]["isError"] and "asked this session: False" in r["result"]["content"][0]["text"], r
    r = call({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
              "params": {"name": "locket_find", "arguments": {}}})
    assert r["result"]["isError"] and "statement is required" in r["result"]["content"][0]["text"], r
    # A synthetic store: a manifest makes any directory a corpus, so the live
    # memory is never read. Reading the first real store made this test depend
    # on whatever shape that store happened to have on the machine running it.
    root = scratch / "store"
    root.mkdir()
    (root / memscan.MANIFEST).write_text(json.dumps({
        "index_files": [], "ledger_surfaces": [], "holding_spaces": ["INBOX.md"]}))
    (root / "facts.md").write_text(
        "The default execution path is the dossier pipeline run interactively.\n"
        "Silhouette glyphs scale by wingspan across the comparison cards.\n")
    (root / "notes.md").write_text(
        "Turbine overhaul programs cover the engine reserve entirely here.\n")
    (root / "INBOX.md").write_text(
        "Silhouette glyphs scale by wingspan across the comparison cards.\n")
    claim = "The default execution path is the dossier pipeline run interactively. "
    r = call({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
              "params": {"name": "locket_check_before_write",
                         "arguments": {"text": claim * 3, "path": str(root / "zz-selftest.md")}}})
    txt = r["result"]["content"][0]["text"]
    assert not r["result"]["isError"] and "not asked" in txt.replace("has not asked", "not asked"), txt
    # a rewrite in place of an existing file adds nothing and must draw no nudge,
    # or the tool fires on exactly the operation the discipline asks for
    f0 = memscan.md_files(root)[0]
    body = f0.read_text(errors="ignore")
    r = call({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
              "params": {"name": "locket_check_before_write",
                         "arguments": {"text": body.replace("the", "The", 1), "path": str(f0)}}})
    assert r["result"]["content"][0]["text"].startswith("No concerns"), r["result"]["content"][0]["text"][:200]
    # the same pasted sentence as an APPEND to that file is scored as the entry alone
    r = call({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
              "params": {"name": "locket_check_before_write",
                         "arguments": {"text": claim * 3, "path": str(f0), "append": True}}})
    assert "already own" in r["result"]["content"][0]["text"] or "not asked" in \
        r["result"]["content"][0]["text"].replace("has not asked", "not asked"), r
    # a CSV takes the hook's CSV question, the ledger schema, not "outside any store"
    hist = scratch / "HISTORY-Proj-Sessions.csv"
    hist.write_text("date,title,summary,area,commit_hash\n2026-09-20,Ok,Fine.,api,5ab427a\n")
    r = call({"jsonrpc": "2.0", "id": 15, "method": "tools/call",
              "params": {"name": "locket_check_before_write",
                         "arguments": {"text": f"2026-09-20,{'t' * 91},s,ios,zz\n", "path": str(hist),
                                       "append": True}}})
    assert "breaks its schema" in r["result"]["content"][0]["text"], r
    r = call({"jsonrpc": "2.0", "id": 16, "method": "tools/call",
              "params": {"name": "locket_check_before_write",
                         "arguments": {"text": "a,b\n", "path": str(scratch / "R.csv"), "append": True}}})
    assert "No concerns" in r["result"]["content"][0]["text"], r
    r = call({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
              "params": {"name": "locket_find", "arguments": {"statement": claim, "store": str(root)}}})
    assert not r["result"]["isError"] and "[" in r["result"]["content"][0]["text"], r
    r = call({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
              "params": {"name": "locket_status", "arguments": {}}})
    assert "asked this session: True" in r["result"]["content"][0]["text"], r
    r = call({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
              "params": {"name": "locket_graduated", "arguments": {"store": str(root)}}})
    txt = r["result"]["content"][0]["text"]
    assert not r["result"]["isError"] and txt.startswith("1 holding-space claim") \
        and "INBOX.md" in txt and "facts.md" in txt, txt
    # init on a folder of markdown nobody registered: listed afterwards by its
    # manifest name, and the index is built by the same call
    vault = scratch / "vault"; vault.mkdir()
    (vault / "one.md").write_text("A schema change ships as its own release because the deploy runs no migrations.\n")
    r = call({"jsonrpc": "2.0", "id": 12, "method": "tools/call",
              "params": {"name": "locket_init", "arguments": {"path": str(vault)}}})
    txt = r["result"]["content"][0]["text"]
    assert not r["result"]["isError"] and "(written)" in txt and "(added)" in txt, txt
    r = call({"jsonrpc": "2.0", "id": 13, "method": "tools/call",
              "params": {"name": "locket_corpora", "arguments": {}}})
    assert "vault" in r["result"]["content"][0]["text"], r["result"]["content"][0]["text"]
    r = call({"jsonrpc": "2.0", "id": 14, "method": "tools/call",
              "params": {"name": "locket_forget", "arguments": {"path": str(vault)}}})
    assert "row removed" in r["result"]["content"][0]["text"], r
    r = call({"jsonrpc": "2.0", "id": 8, "method": "nope"})
    assert r["error"]["code"] == -32601, r
    p.stdin.close(); p.wait(timeout=10)
    assert p.returncode == 0, p.stderr.read()[:400]
    print("selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest(); sys.exit(0)
    try:
        serve()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
