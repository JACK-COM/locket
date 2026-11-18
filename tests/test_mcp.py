"""The MCP server end to end: spawned on stdio with a scratch registry, one full exchange."""
import json
import os
import sys
import unittest
from pathlib import Path

from tests.harness import SRC, fixture_env

import locket_mcp as mcp  # noqa: E402
import memscan  # noqa: E402

def _suite():
    import subprocess, tempfile
    scratch = Path(tempfile.mkdtemp())
    # a scratch registry, so `locket_init` below never touches ~/.locket
    p = subprocess.Popen([sys.executable, str(SRC / "locket_mcp.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
              "params": {"protocolVersion": mcp.PROTOCOL, "capabilities": {}, "clientInfo": {"name": "t"}}})
    assert r["result"]["serverInfo"]["name"] == "locket", r
    call({"jsonrpc": "2.0", "method": "notifications/initialized"})
    r = call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in r["result"]["tools"]}
    assert names == set(mcp.TOOLS), names
    r = call({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "locket_status", "arguments": {}}})
    assert not r["result"]["isError"] and "asked this session: False" in r["result"]["content"][0]["text"], r
    r = call({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
              "params": {"name": "locket_find", "arguments": {}}})
    assert r["result"]["isError"] and "statement is required" in r["result"]["content"][0]["text"], r
    # A synthetic store: a trusted manifest makes a directory a corpus, so the live
    # memory is never read. Reading the first real store made this test depend
    # on whatever shape that store happened to have on the machine running it.
    root = scratch / "store"
    # the store a person trusted: registered in the scratch registry the server reads
    root.mkdir()
    (root / memscan.MANIFEST).write_text(json.dumps({
        "index_files": [], "ledger_surfaces": [], "holding_spaces": ["INBOX.md"]}))
    # the store a person trusted, pinned to its manifest, in the scratch registry the server reads
    (scratch / "corpora.json").write_text(json.dumps({
        "corpora": [str(root.resolve())], "pins": {str(root.resolve()): memscan.config_digest(root.resolve())}}))
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
    hist = root / "HISTORY-Proj-Sessions.csv"
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
    # the hook's own gate: the registry and a store's manifest are refused, and a link
    # into a store is judged as the file it reaches
    for i, (target, said) in enumerate(((scratch / "corpora.json", "trust registry"),
                                        (root / memscan.MANIFEST, "configuration"))):
        r = call({"jsonrpc": "2.0", "id": 30 + i, "method": "tools/call",
                  "params": {"name": "locket_check_before_write",
                             "arguments": {"text": "{}", "path": str(target)}}})
        assert said in r["result"]["content"][0]["text"], r
    link = scratch / "elsewhere.md"
    link.symlink_to(root / "facts.md")
    r = call({"jsonrpc": "2.0", "id": 32, "method": "tools/call",
              "params": {"name": "locket_check_before_write",
                         "arguments": {"text": claim * 3, "path": str(link), "append": True}}})
    assert "not inside a memory store" not in r["result"]["content"][0]["text"], r
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
    # the manifest is written but the folder is not trusted: an agent cannot grant it
    assert "(written, name " in txt and "locket trust " in txt and "(added)" not in txt, txt
    assert str(vault.resolve()) not in (scratch / "corpora.json").read_text()
    r = call({"jsonrpc": "2.0", "id": 13, "method": "tools/call",
              "params": {"name": "locket_corpora", "arguments": {}}})
    assert "vault" not in r["result"]["content"][0]["text"], r["result"]["content"][0]["text"]
    r = call({"jsonrpc": "2.0", "id": 14, "method": "tools/call",
              "params": {"name": "locket_forget", "arguments": {"path": str(root)}}})
    assert "person's act" in r["result"]["content"][0]["text"], r      # an agent cannot withdraw trust
    r = call({"jsonrpc": "2.0", "id": 8, "method": "nope"})
    assert r["error"]["code"] == -32601, r
    p.stdin.close(); p.wait(timeout=10); p.stdout.close(); p.stderr.close()
    assert p.returncode == 0, p.stderr.read()[:400]


class McpExchange(unittest.TestCase):
    def test_exchange(self):
        with fixture_env():
            _suite()


if __name__ == "__main__":
    unittest.main()
