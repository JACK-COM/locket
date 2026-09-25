#!/usr/bin/env python3
"""locket: the one command for Locket.

Locket is two scripts that sit beside this one: `memscan.py` (structure checks
over a memory corpus, the write-time hooks) and `memfind.py` (which file already
holds an idea, by meaning). Every mode of both is reached from here, so a user
never has to remember which script owns which mode, and the install and removal
of the command itself are verbs here too. The hooks keep calling the two scripts
by absolute path, which is why this file adds nothing they depend on.
"""

import json
import os
import shlex
import shutil
import sys

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memscan  # noqa: E402

__version__ = "0.1.6"                          # the one home: pyproject.toml and the formula's test read it

BIN = Path(os.environ.get("LOCKET_BIN", str(Path.home() / ".local" / "bin")))
# A package manager owns the command and moves the scripts on every upgrade: Homebrew's
# wrapper sets LOCKET_PACKAGED, a `uv tool` or pip install enters through `cli()`. A
# packaged install registers hooks and the Desktop server through the command, whose
# path survives an upgrade, and leaves the link in BIN to the package manager.
PACKAGED = os.environ.get("LOCKET_PACKAGED") == "1"
LINKS = ("locket", "memscan", "memfind")      # the last two are older spellings this replaces
SCRIPTS = ("memscan.py", "memfind.py", "locket.py", "locket_mcp.py")

# memfind verbs, in memfind's own spelling; `find` is the bare statement form.
FIND = {"find": None, "siblings": "--siblings", "index": "--index", "embedder": "--embedder"}
SCAN = ("audit", "across", "dupes", "graduated", "ledgers", "budget", "pointers", "hooks", "links", "cites",
        "corpora", "init", "schema", "migrate", "forget", "against")

# The verb table: (name, argument metavar, one-line help, examples). argparse renders
# `locket -h` and `locket <verb> -h` from it; dispatch below stays by hand, so
# every flag after the verb reaches memscan or memfind untouched.
VERBS = [
    ("find", '"<statement>" [corpus] [-n N]', "files most likely to own the statement, by meaning",
     'locket find "a fact you are about to write" council\n'
     'locket find "the same fact" all             every store, one list per store'),
    ("siblings", "[corpus] [-n N]", "file pairs sharing one subject", "locket siblings notes -n 10"),
    ("index", "[corpus]", "build or refresh the semantic cache", "locket index council"),
    ("embedder", "", "which embedder answers on this machine", None),
    ("audit", "[corpus]", "claim pairs across files that agree: fork risk", "locket audit all"),
    ("across", "", "claim pairs split across two stores; takes no corpus", None),
    ("dupes", "[corpus] [file ...]", "repeated claims inside one file; use on the holding spaces",
     "locket dupes council SHORT_TERM.md"),
    ("graduated", "[corpus] [floor]", "holding-space entries a durable file already states", "locket graduated"),
    ("ledgers", "[corpus]", "schema-governed CSV rows off their schema", "locket ledgers notes"),
    ("budget", "[corpus]", "files past a byte, history-clause or paragraph budget", "locket budget council"),
    ("pointers", "[corpus]", "index lines beside the files they point at, missing targets first; "
     "needs a reader", "locket pointers council"),
    ("links", "[corpus]", "wikilink targets resolving to no file", "locket links council"),
    ("cites", "[corpus] [path ...]", "path:line citations resolved against the code",
     "locket cites myapp models.py"),
    ("against", "[corpus] < text", "rank files by how much of stdin they already assert",
     'echo "a sentence" | locket against council'),
    ("corpora", "", "every store on this machine", None),
    ("init", "<dir> [--parent <corpus> [--name <label>]]", "make a folder of markdown a store", "locket init ~/notes"),
    ("migrate", "[<dir>|all]", "rename an older memfind.json manifest to locket.json", "locket migrate all"),
    ("schema", "[<dir>]", "write the manifest schema, or check a manifest", "locket schema ~/notes"),
    ("forget", "<dir>", "drop a store's registry row", None),
    ("install", "[--hooks] [--desktop]",
     "put `locket` on PATH (%s); --hooks merges the write-time hooks into Claude Code's "
     "settings.json, --desktop registers the MCP server with Claude Desktop" % BIN,
     "locket install --hooks"),
    ("uninstall", "[--yes] [--purge] [--dry-run]",
     "remove the command, ~/.locket and every cache; with --purge every manifest; print what is left",
     "locket uninstall --dry-run"),
    ("status", "", "where the command points and what it can reach", None),
    ("doctor", "", "check every install step a machine can check, with the fix for each failure", None),
    ("mcp", "", "serve the checks over MCP on stdio, for a host without hooks", None),
    ("selftest", "", "both scripts' internal checks", None),
    ("version", "", "print the version (also --version, -V, -v)", None),
    ("help", "find | scan | install", "the full help of either script, or the agent's install steps", None),
]
VERB_NAMES = frozenset(v[0] for v in VERBS)
HOOK_VERBS = ("hook", "bashguard", "grepassist", "deliver")   # PreToolUse entry points, JSON on stdin

EPILOG = """examples:
  locket find "the fact you are about to write" council
  locket find "the same fact" all           every store, one list per store
  locket index notes                        refresh one cache; `find` builds it on demand too
  locket graduated && locket links && locket across
                                            the session-close trio
  locket init ~/notes                       any folder of markdown becomes a store

The corpus argument takes a shorthand (council, hermes) or registered name, a project slug
substring, a literal path, or `all`; omitted, it is the store the current
directory sits in. `--index`, `--siblings`, `--embedder` and `--selftest` are
accepted as verbs too. `locket help scan` explains what a store is and how a
manifest shapes it; `locket help find` explains how to read a ranking.
Needs nothing but Python 3 for the checks; `find` needs an embedder (ollama with
nomic-embed-text, or fastembed in ~/.locket/venv) and says so when none answers.
"""


def _parser():
    import argparse
    fmt = dict(formatter_class=argparse.RawDescriptionHelpFormatter)
    p = argparse.ArgumentParser(
        prog="locket", epilog=EPILOG, **fmt,
        description="locket: which memory file already holds this idea, and where a corpus repeats itself.")
    sub = p.add_subparsers(title="commands", metavar="<command>")
    for name, args, help, examples in VERBS:
        script = "memfind" if name in FIND else "memscan" if name in SCAN else None
        tail = (f"\n`locket help {'find' if script == 'memfind' else 'scan'}` has {script}.py's full help."
                if script else "")
        shown = "\n".join("  " + ln for ln in examples.split("\n")) if examples else ""
        c = sub.add_parser(name, help=help, description=help, **fmt,
                           usage=f"locket {name} {args}".rstrip(),
                           epilog=(f"examples:\n{shown}\n" if shown else "") + tail)
        c.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return p, sub


def _memfind():
    import memfind
    return memfind


# --- install / uninstall ------------------------------------------------------

def _link_target(link):
    try:
        return Path(os.readlink(link)) if link.is_symlink() else None
    except OSError:
        return None


def _ours(link):
    """A link in BIN that points into this directory, or dangles there."""
    t = _link_target(link)
    return t is not None and (t.parent == HERE or not t.exists())


# Claude Code's PreToolUse entries, the three `locket install --hooks` merges.
CLAUDE_HOOKS = (("Write|Edit", "hook", 10), ("Bash", "bashguard", 5), ("Grep|Bash", "grepassist", 10))
# Where Claude Desktop keeps its config: macOS and Windows only. Elsewhere
# it is None, so `--desktop` refuses rather than write a file Desktop never reads.
DESKTOP_CONFIG = (Path.home() / "Library/Application Support/Claude/claude_desktop_config.json" if sys.platform == "darwin"
                  else Path(os.environ["APPDATA"]) / "Claude/claude_desktop_config.json" if os.environ.get("APPDATA")
                  else None)


def _merge_json(path, merge):
    """Apply `merge(data) -> [what was added]` to a JSON config, keeping a copy
    beside it. A file that does not parse is left untouched: rewriting it would
    drop whatever the parse missed."""
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"{path}: not valid JSON ({e}); left untouched. Fix it and rerun")
        return 1
    if not isinstance(data, dict):
        print(f"{path}: top level is not an object; left untouched"); return 1
    added = merge(data)
    if not added:
        print(f"{path}: nothing to add (already registered)"); return 0
    if path.is_file():
        bak = path.with_suffix(".json.locket-install.bak")
        shutil.copy2(path, bak)
    else:
        bak = None
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"edited {path}: added {', '.join(added)}" + (f"; copy at {bak}" if bak else ""))
    return 0


def _is_locket_hook(cmd, verb=None):
    """A hook command that runs Locket: memscan.py by path, or the `locket` command."""
    verbs = [verb] if verb else HOOK_VERBS
    cmd = cmd.replace("'", "").replace('"', "").rstrip()   # a quoted path, as _hook_command writes one
    return any(f"memscan.py {v}" in cmd or cmd.endswith(f"locket {v}") for v in verbs)


def _command():
    """The `locket` command's absolute path for a packaged install, or None."""
    w = shutil.which("locket") if PACKAGED else None
    return str(Path(w).absolute()) if w else None


def _hook_command(verb):
    cmd = _command()
    return f"{shlex.quote(cmd)} {verb}" if cmd else f"python3 {HERE / 'memscan.py'} {verb}"


def _add_claude_hooks(data):
    pre = data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    present = [str(h.get("command", "")) for e in pre if isinstance(e, dict)
               for h in e.get("hooks", []) if isinstance(h, dict)]
    added = []
    for matcher, verb, timeout in CLAUDE_HOOKS:
        if any(_is_locket_hook(c, verb) for c in present):
            continue
        pre.append({"matcher": matcher, "hooks": [{"type": "command", "timeout": timeout,
                    "command": _hook_command(verb)}]})
        added.append(verb)
    return added


# Claude Code runs a hook through the user's shell, so `python3` survives an interpreter
# upgrade; Claude Desktop spawns a server with a minimal PATH, where a bare `python3` can
# be the macOS installer stub and hang, so the server takes the absolute path.
def _add_desktop_server(data):
    servers = data.setdefault("mcpServers", {})
    if "locket" in servers:
        return []
    cmd = _command()
    servers["locket"] = ({"command": cmd, "args": ["mcp"]} if cmd else
                         {"command": sys.executable, "args": [str(HERE / "locket_mcp.py")]})
    return ["mcpServers.locket"]


def cmd_install(argv=()):
    if PACKAGED:
        print(f"command: {_command() or 'locket, managed by the package manager'}")
    rc = 0 if PACKAGED else _install_link()
    if rc:
        return rc
    try:
        memscan.cmd_schema(quiet=True); print(f"schema: {memscan.SCHEMA_FILE}")
    except OSError as e:
        print(f"schema: not written ({e})")
    stale = memscan.SCHEMA_FILE.parent / "memfind.schema.json"   # the file's name before the rename
    if stale.is_file():
        stale.unlink(); print(f"schema: {stale} removed (older name)")
    for cfg in _editor_settings():
        if "memfind.schema.json" in cfg.read_text(errors="replace"):
            print(f"editor: {cfg} still maps the manifest to memfind.schema.json; change that entry to\n"
                  f'        {{"fileMatch": ["locket.json", "memfind.json"], "url": "file://{memscan.SCHEMA_FILE}"}}')
    print(_embedder_line())
    if "--hooks" in argv:
        rc |= _merge_json(memscan.CLAUDE / "settings.json", _add_claude_hooks)
    if "--desktop" in argv and DESKTOP_CONFIG is None:
        print("--desktop: no known Claude Desktop config path on this platform; add the "
              "mcpServers entry by hand (`locket help install`, Step 4)", file=sys.stderr)
        rc |= 1
    elif "--desktop" in argv:
        rc |= _merge_json(DESKTOP_CONFIG, _add_desktop_server)
        print("restart Claude Desktop once for the server to load")
    print("verify from another directory:  locket doctor")
    on_path = PACKAGED or str(BIN) in os.environ.get("PATH", "").split(os.pathsep)
    return rc or (0 if on_path else 2)


def _install_link():
    """Link `locket` into BIN for an install from the scripts; 0, or 1 on a
    file in the way. Prints whether BIN is on PATH."""
    me = HERE / "locket.py"
    me.chmod(me.stat().st_mode | 0o111)
    BIN.mkdir(parents=True, exist_ok=True)
    link = BIN / "locket"
    if link.is_symlink() or link.exists():
        if _link_target(link) == me:
            print(f"link: {link} -> {me} (already)")
        elif _ours(link) or link.is_symlink():
            link.unlink(); link.symlink_to(me); print(f"link: {link} -> {me} (replaced)")
        else:
            print(f"link: {link} exists and is not a symlink; leaving it. Remove it and rerun.")
            return 1
    else:
        link.symlink_to(me); print(f"link: {link} -> {me}")
    for old in LINKS[1:]:                   # the two names this command replaces
        l = BIN / old
        if l.is_symlink() and _ours(l):
            l.unlink(); print(f"link: {l} removed (use `locket {'find' if old == 'memfind' else '<mode>'}`)")
    on_path = str(BIN) in os.environ.get("PATH", "").split(os.pathsep)
    if on_path:
        print(f"PATH: yes ({BIN})")
    else:
        print(f"PATH: {BIN} is not on it; add  export PATH=\"{BIN}:$PATH\"  to the shell profile")
    return 0


def _editor_settings():
    """VS Code user settings files present on this machine, for the one entry
    `locket help install` names. Read only; the file allows comments."""
    home = Path.home()
    cands = [home / "Library/Application Support/Code/User/settings.json",
             home / ".config/Code/User/settings.json",
             Path(os.environ.get("APPDATA", "")) / "Code/User/settings.json" if os.environ.get("APPDATA") else None]
    return [c for c in cands if c and c.is_file()]


def _embedder_line():
    try:
        name, model = _memfind().resolve_backend()
        return f"embedder: {name} ({model})"
    except RuntimeError as e:
        return f"embedder: none ({str(e).splitlines()[0]}); `locket find` will fall back to word overlap"


def _caches():
    """Every `.memfind/` cache Locket may have written: at each store root, in
    each member of a host store (older layout), and in every registered path."""
    roots = set(memscan._raw_corpora().values()) | set(memscan.registry())
    out = set()
    for r in roots:
        if not r.is_dir():
            continue
        out.add(r / ".memfind")
        for d in r.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                out.add(d / ".memfind")
    return sorted(p for p in out if p.is_dir())


def _manifests():
    roots = set(memscan._raw_corpora().values()) | set(memscan.registry())
    out = []
    for r in roots:
        for name in (memscan.MANIFEST, memscan.OLD_MANIFEST):
            if (r / name).is_file():
                out.append(r / name)
    return sorted(out)


def _settings_hooks(path):
    """(entries to drop, remaining) for a Claude Code settings.json: every hook
    whose command runs memscan.py."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return [], None
    hooks = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(hooks, dict):
        return [], None
    drop = []
    for event, entries in list(hooks.items()):
        if not isinstance(entries, list):
            continue
        keep = []
        for e in entries:
            hs = e.get("hooks", []) if isinstance(e, dict) else []
            kept = [h for h in hs if not _is_locket_hook(str(h.get("command", "")))]
            drop += [f"{event} {e.get('matcher', '')}: {h.get('command')}" for h in hs if h not in kept]
            if kept:
                keep.append({**e, "hooks": kept})
            elif not hs:
                keep.append(e)
        hooks[event] = keep
    return drop, data


def _grep_lines(path, needle):
    try:
        return [f"{path}:{i}" for i, l in enumerate(path.read_text().splitlines(), 1) if needle in l]
    except OSError:
        return []


def cmd_uninstall(argv):
    yes, purge, dry = "--yes" in argv, "--purge" in argv, "--dry-run" in argv
    links = [BIN / n for n in LINKS if (BIN / n).is_symlink() and _ours(BIN / n)]
    locket_dir = memscan.REGISTRY.parent
    caches = _caches()
    manifests = _manifests()
    settings = memscan.CLAUDE / "settings.json"
    hook_lines, new_settings = _settings_hooks(settings) if settings.is_file() else ([], None)
    hermes_cfg = memscan.HERMES / "config.yaml"
    hermes_lines = _grep_lines(hermes_cfg, "memscan.py") if hermes_cfg.is_file() else []
    desktop = DESKTOP_CONFIG
    mcp = False
    if desktop and desktop.is_file():
        try:
            mcp = "locket" in (json.loads(desktop.read_text()).get("mcpServers") or {})
        except (OSError, json.JSONDecodeError, ValueError):
            mcp = False

    print("uninstall will remove:")
    for l in links:
        print(f"  link      {l}")
    if locket_dir.is_dir():
        print(f"  directory {locket_dir}  (registry, schema{', venv' if (locket_dir / 'venv').is_dir() else ''})")
    for c in caches:
        print(f"  cache     {c}")
    if purge:
        for m in manifests:
            print(f"  manifest  {m}")
    for h in hook_lines:
        print(f"  hook      {settings}: {h}")
    if mcp:
        print(f"  mcp       {desktop}: mcpServers.locket")
    if not any((links, locket_dir.is_dir(), caches, hook_lines, mcp, purge and manifests)):
        print("  nothing; Locket is not installed here")
    if dry:
        return 0
    if not yes:
        if sys.stdin.isatty():
            if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("nothing removed"); return 1
        else:
            print("nothing removed; rerun with --yes"); return 1

    for l in links:
        l.unlink(); print(f"removed {l}")
    if locket_dir.is_dir():
        shutil.rmtree(locket_dir); print(f"removed {locket_dir}")
    for c in caches:
        shutil.rmtree(c, ignore_errors=True); print(f"removed {c}")
    if purge:
        for m in manifests:
            m.unlink(); print(f"removed {m}")
    if hook_lines and new_settings is not None:
        bak = settings.with_suffix(".json.locket-uninstall.bak")
        shutil.copy2(settings, bak)
        settings.write_text(json.dumps(new_settings, indent=2) + "\n")
        print(f"edited {settings} ({len(hook_lines)} hook entries removed; copy at {bak})")
    if mcp:
        bak = desktop.with_suffix(".json.locket-uninstall.bak")
        shutil.copy2(desktop, bak)
        data = json.loads(desktop.read_text()); data["mcpServers"].pop("locket", None)
        desktop.write_text(json.dumps(data, indent=2) + "\n")
        print(f"edited {desktop} (mcpServers.locket removed; copy at {bak})")

    print("\nleft for you, if you want it gone completely:")
    if manifests and not purge:
        print("  manifests, inert without the tool and often under version control:")
        for m in manifests:
            print(f"    {m}")
        print("    (rerun with --purge to delete them)")
    for hl in hermes_lines:
        print(f"  Hermes hook line {hl}  (YAML; remove the entry by hand)")
    print(f"  the scripts:  rm {' '.join(str(HERE / s) for s in SCRIPTS if (HERE / s).exists())} "
          f"{HERE / 'INSTALL-locket.md' if (HERE / 'INSTALL-locket.md').exists() else ''}".rstrip())
    print("  an editor `json.schemas` entry pointing at ~/.locket/locket.schema.json, if you added one")
    return 0


def cmd_status():
    link = BIN / "locket"
    t = _link_target(link)
    if PACKAGED:
        print(f"command: {_command() or 'locket (packaged, not on PATH)'}")
    else:
        print(f"command: {link} -> {t}" if t else f"command: {link} not installed (`locket install`)")
    print(f"scripts: {HERE}")
    print(f"on PATH: {'yes' if str(BIN) in os.environ.get('PATH', '').split(os.pathsep) else 'no'}")
    print(f"registry: {memscan.REGISTRY} ({len(memscan.registry())} registered)")
    print(_embedder_line())
    n = len(memscan.all_corpora())
    print(f"corpora: {n}  (`locket corpora` lists them)")
    print(memscan.gate_status())
    return 0


# --- doctor -------------------------------------------------------------------

# A paragraph long enough to clear the gate's 200-character floor, written into a
# throwaway store under ~/.locket so the probe never reads or writes a user's memory.
_PROBE = ("The doctor probe paragraph states one durable fact about a harbour: the tide "
          "gauge at the north pier is read at dawn and dusk, the readings are logged in "
          "the harbourmaster's ledger, and a reading missed twice in a row is escalated "
          "to the port authority before the next high water.")


def _registered_hooks(settings):
    """{verb: command} for the Locket hooks in a Claude Code settings.json."""
    try:
        data = json.loads(settings.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    hooks = data.get("hooks") if isinstance(data, dict) else None
    pre = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
    out = {}
    for e in pre if isinstance(pre, list) else []:
        for h in (e.get("hooks") or []) if isinstance(e, dict) else []:
            cmd = str(h.get("command", "")) if isinstance(h, dict) else ""
            for _, verb, _t in CLAUDE_HOOKS:
                if _is_locket_hook(cmd, verb):
                    out.setdefault(verb, cmd)
    return out


def _script_in(cmd):
    """The first path in a hook command that names a .py file, or None."""
    for tok in cmd.split():
        if tok.endswith(".py"):
            return Path(os.path.expanduser(tok))
    return None


def _probe_hook(cmd):
    """Run a registered `hook` command on a write that must fire: a new file in a
    throwaway store repeating a paragraph the store already holds. True when it
    answered, False when it was silent, a string when it failed to run."""
    import subprocess, tempfile
    base = memscan.REGISTRY.parent
    base.mkdir(parents=True, exist_ok=True)
    store = Path(tempfile.mkdtemp(prefix="doctor-probe-", dir=base))
    try:
        (store / memscan.MANIFEST).write_text("{}\n")
        (store / "harbour.md").write_text(_PROBE + "\n")
        payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Write",
                              "tool_input": {"file_path": str(store / "zz-probe.md"), "content": _PROBE}})
        # supervised, so a firing hook nudges on stdout and exits 0; exit 2 would
        # also be what python prints for a script path that is gone
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_AUTONOMOUS"}
        try:
            r = subprocess.run(cmd, shell=True, input=payload, capture_output=True, text=True,
                               timeout=30, env=env)
        except subprocess.TimeoutExpired:
            return "timed out after 30s"
        if r.returncode != 0:
            return (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
        return bool(r.stdout.strip())
    finally:
        shutil.rmtree(store, ignore_errors=True)


def cmd_doctor():
    """Every install step a machine can check, one line each, with the command
    that fixes a failure. Exit 1 when anything fails; a warning is a rung
    missing that Locket works without."""
    import contextlib, io
    rows = []

    def row(level, what, detail, fix=None):
        rows.append((level, what, detail, fix))

    w = shutil.which("locket")
    row("ok", "command", w) if w else row("fail", "command", "`locket` is not on PATH", "locket install")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rc = memscan.main(["memscan.py", "selftest"])
        except Exception as e:              # an assertion names the broken check
            rc, _ = 1, buf.write(f"{type(e).__name__}: {e}")
    last = (buf.getvalue().strip().splitlines() or ["no output"])[-1]
    row("ok", "selftest", last) if rc == 0 else row("fail", "selftest", last[:160], "locket selftest")

    stores = memscan.all_corpora()
    row("ok", "stores", f"{len(stores)} ({', '.join(sorted(stores)[:6])}{', ...' if len(stores) > 6 else ''})") \
        if stores else row("fail", "stores", "none found", "locket init <folder of markdown>")

    try:
        name, model = _memfind().resolve_backend()
        row("ok", "embedder", f"{name} ({model})")
    except RuntimeError:
        row("warn", "embedder", "none; `find` falls back to word overlap and misses paraphrases",
            "ollama pull nomic-embed-text  (or fastembed: `locket help install`)")

    # an empty store (an auto-memory folder no session has written to) has nothing to
    # index and `index all` skips it, so it never counts against this check
    def _has_md(r):
        try:
            return bool(memscan.md_files(r))
        except OSError:                     # an unreadable store (pathlib raises before 3.13)
            return False
    unindexed = [n for n, r in stores.items()
                 if not (r / ".memfind").is_dir() and _has_md(r)]
    if unindexed:
        row("warn", "index", f"not built for {', '.join(sorted(unindexed))}", "locket index all")
    elif stores:
        row("ok", "index", "built for every store")

    settings = memscan.CLAUDE / "settings.json"
    if memscan.CLAUDE.is_dir():
        hooks = _registered_hooks(settings) if settings.is_file() else {}
        if hooks is None:
            row("fail", "claude hooks", f"{settings} is not valid JSON", f"fix {settings}, then locket install --hooks")
        else:
            missing = [v for _, v, _t in CLAUDE_HOOKS if v not in hooks]
            stale = [v for v, c in hooks.items() if (p := _script_in(c)) and not p.is_file()]
            if missing:
                row("fail", "claude hooks", f"not registered: {', '.join(missing)}", "locket install --hooks")
            elif stale:
                row("fail", "claude hooks", f"point at a missing script: {', '.join(stale)}",
                    "locket uninstall --dry-run, then remove the stale entries and locket install --hooks")
            else:
                probe = _probe_hook(hooks["hook"])
                if probe is True:
                    row("ok", "claude hooks", "registered, and the write hook fires on a known fork")
                else:
                    row("fail", "claude hooks", "registered, but the write hook " +
                        ("stayed silent on a known fork" if probe is False else f"failed: {probe}"),
                        f"run it by hand: {hooks['hook']}")

    hermes_cfg = memscan.HERMES / "config.yaml"
    if hermes_cfg.is_file():
        text = hermes_cfg.read_text(errors="replace")
        lines = text.splitlines()
        missing = [v for v in HOOK_VERBS if not any(_is_locket_hook(ln, v) for ln in lines)]
        row("warn", "hermes hooks", f"not in config.yaml: {', '.join(missing)}", "`locket help install`, Step 4") \
            if missing else row("ok", "hermes hooks", "all four registered")

    if DESKTOP_CONFIG is not None and DESKTOP_CONFIG.is_file():
        try:
            srv = (json.loads(DESKTOP_CONFIG.read_text()).get("mcpServers") or {}).get("locket")
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            srv = None
        if srv:
            parts = [Path(srv.get("command", ""))] + [Path(a) for a in srv.get("args", []) if str(a).endswith(".py")]
            gone = [str(p) for p in parts if not p.is_file()]
            row("fail", "desktop server", f"names a missing file: {', '.join(gone)}", "locket install --desktop") \
                if gone else row("ok", "desktop server", "registered; restart Claude Desktop after a change")

    if not memscan.SCHEMA_FILE.is_file():
        row("warn", "schema", f"{memscan.SCHEMA_FILE} not written; editors get no manifest hints", "locket schema")

    old = [n for n in LINKS[1:] if (BIN / n).is_symlink() and _ours(BIN / n)]
    if old:
        row("warn", "old links", f"{', '.join(old)} still linked in {BIN}", "locket install")

    width = max(len(r[1]) for r in rows)
    for level, what, detail, fix in rows:
        print(f"{level:<4}  {what:<{width}}  {detail}")
        if fix and level != "ok":
            print(f"{'':<4}  {'':<{width}}  fix: {fix}")
    fails = sum(r[0] == "fail" for r in rows)
    warns = sum(r[0] == "warn" for r in rows)
    print(f"{len(rows)} checks: {len(rows) - fails - warns} ok, {warns} warning(s), {fails} failure(s)")
    return 1 if fails else 0


# --- dispatch -----------------------------------------------------------------

def _wants_help(verb, rest):
    """A help request anywhere after a documented verb, trailing included, so none reaches
    a script as a corpus or path: `-h` or `--help`, or a bare `help` unless something
    named `help` exists here to be the directory or file it names. find's statement may be
    any word, so it never counts. Cheap on purpose: the hook verbs dispatch through here
    on every tool call and never build the parser."""
    if verb not in VERB_NAMES:
        return False
    if {"-h", "--help"} & set(rest):
        return True
    return "help" in (rest[1:] if verb == "find" else rest) and not Path("help").exists()


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        _parser()[0].print_help()
        return 0 if len(argv) > 1 else 1
    verb, rest = argv[1], argv[2:]
    if verb in ("--version", "-V", "-v", "version"):
        print(f"locket {__version__}"); return 0
    if verb == "help":
        which = rest[0] if rest else ""
        if which in _parser()[1].choices and which not in ("find", "install"):
            _parser()[1].choices[which].print_help(); return 0
        if which in ("find", "memfind"):
            return _memfind().main(["memfind.py", "--help"])
        if which in ("scan", "memscan"):
            return memscan.main(["memscan.py", "--help"])
        if which == "install":              # the agent's half of the install, shipped beside the scripts
            doc = HERE / "INSTALL-locket.md"
            if not doc.is_file():
                print(f"{doc} is missing; reinstall Locket", file=sys.stderr); return 1
            print(doc.read_text()); return 0
        _parser()[0].print_help(); return 0
    if verb.startswith("-"):
        verb = verb.lstrip("-")               # memfind's own spelling: --index, --siblings, --selftest
    if _wants_help(verb, rest):
        _parser()[1].choices[verb].print_help(); return 0
    if verb == "find":
        if not rest or rest[0].startswith("-"):
            print('find needs a statement:  locket find "<statement>" [corpus] [-n N]', file=sys.stderr)
            return 1
        return _memfind().main(["memfind.py", *rest])
    if verb in FIND:
        return _memfind().main(["memfind.py", FIND[verb], *rest])
    if verb == "selftest":
        rc = memscan.main(["memscan.py", "selftest"])
        if not rc:                          # doctor's probe: fires on a live hook, silent on a dead one
            live = f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'memscan.py'))} hook"
            assert _probe_hook(live) is True, "doctor's probe does not fire through this memscan.py"
            assert _probe_hook("true") is False, "doctor's probe reads a silent command as firing"
            assert isinstance(_probe_hook(f"{shlex.quote(sys.executable)} /nonexistent/memscan.py hook"), str), \
                "doctor's probe reads a missing script as firing"
            import contextlib
            import io
            for form in (["audit", "council", "-h"], ["audit", "help"], ["help", "audit"],
                         ["dupes", "x", "--help"], ["find", "a statement", "help"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    code = main(["locket", *form])
                assert code == 0 and out.getvalue().startswith(f"usage: locket {form[1] if form[0] == 'help' else form[0]}"), \
                    f"`locket {' '.join(form)}` did not print that command's help"
            import tempfile
            here = os.getcwd()
            with tempfile.TemporaryDirectory() as t:
                os.chdir(t)
                try:
                    assert _wants_help("init", ["help"]), "a bare help with nothing named help is not help"
                    os.mkdir("help")
                    assert not _wants_help("init", ["help"]), "a directory named help was read as a help request"
                    assert not _wants_help("hook", ["-h"]) and not _wants_help("find", ["help"]), \
                        "a hook verb or find's statement was read as a help request"
                finally:
                    os.chdir(here)
        return rc or _memfind().main(["memfind.py", "--selftest"])
    if verb == "install":
        return cmd_install(rest)
    if verb == "uninstall":
        return cmd_uninstall(rest)
    if verb == "status":
        return cmd_status()
    if verb == "doctor":
        return cmd_doctor()
    if verb == "mcp":                       # the Desktop server, reached through the command
        import runpy
        sys.argv = [str(HERE / "locket_mcp.py"), *rest]
        runpy.run_path(str(HERE / "locket_mcp.py"), run_name="__main__")
        return 0
    if verb in SCAN:
        return memscan.main(["memscan.py", verb, *rest])
    if verb in HOOK_VERBS:
        return memscan.main(["memscan.py", verb, *rest])
    _parser()[0].print_usage(sys.stderr)
    print(f"locket: unknown command {verb!r}; `locket -h` lists them", file=sys.stderr)
    return 1


def cli():
    """The entry point a `uv tool` or pip install puts on PATH."""
    global PACKAGED
    PACKAGED = True
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:
        sys.exit(0)
