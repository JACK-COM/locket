"""This machine's install: putting Locket on PATH and the hosts' configs and
taking it off again, what it reaches, whether every step works, and the MCP server.
"""
import platform
import sys

import adapters
import commands
from commands import Tool, Verb


def _install(rest):
    import install
    return install.cmd_install(rest)


def _uninstall(rest):
    import install
    return install.cmd_uninstall(rest)


def _install_guide(own):
    """The agent's half of the install, shipped beside the scripts."""
    import install
    doc = install.HERE / "INSTALL-locket.md"
    if not doc.is_file():
        print(f"{doc} is missing; reinstall Locket", file=sys.stderr)
        return 1
    print(doc.read_text())
    return 0


def _status(rest):
    import doctor
    return doctor.cmd_status()


def _status_tool(args, session):
    import install
    try:
        name, model = install.load_memfind().resolve_backend()
        emb = f"{name} ({model})"
    except RuntimeError as e:
        emb = "none; " + str(e).splitlines()[0]
    return "\n".join([
        f"locket on {platform.system()} {platform.release()}, python {platform.python_version()} "
        f"at {sys.executable}",
        f"embedder: {emb}",
        f"asked this session: {session.asked}  (a session here is this server process, which the app "
        f"shares across its conversations until it restarts)",
        adapters.gate_status(),
    ])


def _doctor(rest):
    import doctor
    import health
    rc = doctor.cmd_doctor(commands.VERSION)    # shown first, so an ack never hides what it marks seen
    if "--ack" in rest:
        print("hook health: " + ("the counts above are marked seen" if health.health_ack()
                                 else "could not mark the counts seen (the record was busy); run it again"))
    return rc


def _mcp(rest):
    """The Desktop server, reached through the command."""
    import install
    import runpy
    sys.argv = [str(install.HERE / "locket_mcp.py"), *rest]
    runpy.run_path(str(install.HERE / "locket_mcp.py"), run_name="__main__")
    return 0


def _selftest(rest):
    import probes
    broken = probes.smoke()
    print(f"selftest: {broken}" if broken else "selftest ok", file=sys.stderr if broken else sys.stdout)
    return 1 if broken else 0


def _version(rest):
    print(f"locket {commands.VERSION}")
    return 0


def _bin():
    import install
    return install.BIN


VERBS = (
    Verb("install", "[--hooks] [--codex] [--desktop]",
         "put `locket` on PATH (%s); --hooks merges the write-time hooks into Claude Code's "
         "settings.json, --codex into Codex's hooks.json, --desktop registers the MCP server "
         "with Claude Desktop" % _bin(),
         "locket install                        the command only\n"
         "locket install --hooks                plus the Claude Code hooks, keeping a copy of settings.json\n"
         "locket install --codex                plus the Codex hooks; trust them with /hooks in Codex\n"
         "locket install --desktop              plus the Claude Desktop server; restart the app after",
         run=_install, guide=_install_guide),
    Verb("uninstall", "[--yes] [--purge] [--dry-run]",
         "remove the command, ~/.locket, every cache, and the shared venv when no other piece uses it; "
         "with --purge every manifest and the usage ledger, confirmed by a person at a terminal; print what is left",
         "locket uninstall --dry-run            print what would go, remove nothing\n"
         "locket uninstall                      remove, after asking once\n"
         "locket uninstall --yes                remove without asking\n"
         "locket uninstall --purge              also every store's locket.json and the usage ledger",
         run=_uninstall),
    Verb("status", "", "where the command points and what it can reach", None, run=_status,
         mcp=Tool("Host, interpreter, embedder rung, and whether this session "
                  "has asked the store yet.", _status_tool)),
    Verb("doctor", "[--ack]", "check every install step a machine can check, with the fix for each failure",
         "locket doctor                        check, including live hook runs abandoned or failed open\n"
         "locket doctor --ack                  check, then mark the hook-health counts shown as seen",
         run=_doctor),
    Verb("mcp", "", "serve the checks over MCP on stdio, for a host without hooks", None, run=_mcp),
    Verb("selftest", "", "check this install end to end on scratch stores", None, run=_selftest),
    Verb("version", "", "print the version (also --version, -V, -v)", None, run=_version),
)
