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
import re
import shlex
import shutil
import sys

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memscan  # noqa: E402

__version__ = "0.6.0"                          # the one home: pyproject.toml and the formula's test read it

BIN = Path(os.environ.get("LOCKET_BIN", str(Path.home() / ".local" / "bin")))
# A package manager owns the command and moves the scripts on every upgrade: Homebrew's
# wrapper sets LOCKET_PACKAGED, a `uv tool` or pip install enters through `cli()`. A
# packaged install registers hooks and the Desktop server through the command, whose
# path survives an upgrade, and leaves the link in BIN to the package manager.
PACKAGED = os.environ.get("LOCKET_PACKAGED") == "1"
LINKS = ("locket", "memscan", "memfind")      # the last two are older spellings this replaces
SCRIPTS = ("memscan.py", "memfind.py", "locket.py", "locket_mcp.py", "trigger.py", "usage.py")

# memfind verbs, in memfind's own spelling; `find` is the bare statement form.
FIND = {"find": None, "siblings": "--siblings", "index": "--index", "embedder": "--embedder"}
SCAN = ("audit", "across", "dupes", "graduated", "ledgers", "budget", "pointers", "links", "cites",
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
    ("trigger", "[check [corpus] | schema]",
     "the tool-boundary hook: when a tool call or prompt matches a row in a store's triggers.json, "
     "put the row's question to the agent; `check` lints the rows files, `schema` prints their schema",
     "locket trigger check                  the rows that can fire from here\n"
     "locket trigger check council          one store's rows"),
    ("usage", "[today | week | month | record] [--by project|model|day] [--json]",
     "tokens Claude Code and Hermes spent, per source and project, against a typical day; "
     "keeps a ledger that outlives the transcripts",
     "locket usage                          the last 7 days\n"
     "locket usage month --by model         30 days, per model"),
    ("corpora", "", "every store on this machine", None),
    ("init", "<dir> [--parent <corpus> [--name <label>]]", "make a folder of markdown a store", "locket init ~/notes"),
    ("migrate", "[<dir>|all]", "rename an older memfind.json manifest to locket.json", "locket migrate all"),
    ("schema", "[<dir>]", "write the manifest schema, or check a manifest", "locket schema ~/notes"),
    ("forget", "<dir>", "drop a store's registry row", None),
    ("install", "[--hooks] [--codex] [--desktop]",
     "put `locket` on PATH (%s); --hooks merges the write-time hooks into Claude Code's "
     "settings.json, --codex into Codex's hooks.json, --desktop registers the MCP server "
     "with Claude Desktop" % BIN,
     "locket install --hooks"),
    ("uninstall", "[--yes] [--purge] [--dry-run]",
     "remove the command, ~/.locket, every cache, and the shared venv when no other piece uses it; "
     "with --purge every manifest and the usage ledger; print what is left",
     "locket uninstall --dry-run"),
    ("status", "", "where the command points and what it can reach", None),
    ("doctor", "", "check every install step a machine can check, with the fix for each failure", None),
    ("mcp", "", "serve the checks over MCP on stdio, for a host without hooks", None),
    ("selftest", "", "both scripts' internal checks", None),
    ("version", "", "print the version (also --version, -V, -v)", None),
    ("help", "find | scan | trigger | usage | install",
     "the full help of either script, how trigger rows or usage work, or the agent's install steps", None),
]
VERB_NAMES = frozenset(v[0] for v in VERBS)
HOOK_VERBS = ("hook", "bashguard", "grepassist", "deliver")   # PreToolUse entry points, JSON on stdin
# The (event, verb) entries Hermes's config.yaml needs. `deliver` is retired and does nothing.
HERMES_HOOKS = (("pre_tool_call", "hook"), ("pre_tool_call", "bashguard"), ("pre_tool_call", "grepassist"),
                ("pre_tool_call", "trigger"), ("pre_llm_call", "trigger"))

EPILOG = """examples:
  locket find "the fact you are about to write" council
  locket find "the same fact" all           every store, one list per store
  locket index notes                        refresh one cache; `find` builds it on demand too
  locket graduated && locket links && locket across
                                            the session-close trio
  locket init ~/notes                       any folder of markdown becomes a store

The corpus argument takes a shorthand (council, hermes, codex) or registered name, a project slug
substring, a literal path, or `all`; omitted, it is the store the current
directory sits in. `--index`, `--siblings`, `--embedder` and `--selftest` are
accepted as verbs too. `locket help scan` explains what a store is and how a
manifest shapes it; `locket help find` explains how to read a ranking.
Needs nothing but Python 3 for the checks; `find` needs an embedder (ollama with
nomic-embed-text, or fastembed in the venv the Panoply pieces share) and says so when none answers.
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
# `locket trigger` reads every tool call and every prompt, since a row can name any tool.
CLAUDE_TRIGGER = (("PreToolUse", "*"), ("UserPromptSubmit", None))
# `locket usage hook` records the ledger as each session ends, before its transcript can age out.
CLAUDE_USAGE = ("SessionEnd", "usage hook", 30)
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
    verbs = [verb] if verb else (*HOOK_VERBS, "trigger", CLAUDE_USAGE[1])
    cmd = cmd.replace("'", "").replace('"', "").rstrip()   # a quoted path, as _hook_command writes one
    return any(f"memscan.py {v}" in cmd or cmd.endswith((f"locket {v}", f"locket.py {v}")) for v in verbs)


def _command():
    """The `locket` command's absolute path for a packaged install, or None."""
    w = shutil.which("locket") if PACKAGED else None
    return str(Path(w).absolute()) if w else None


def _hook_command(verb):
    cmd = _command()
    script = HERE / ("locket.py" if verb in ("trigger", CLAUDE_USAGE[1]) else "memscan.py")
    return f"{shlex.quote(cmd)} {verb}" if cmd else f"python3 {script} {verb}"


def _add_claude_hooks(data, usage=True):
    """Merge Locket's hooks into a Claude Code settings.json, or into a Codex
    hooks.json, which takes the same schema; `usage` adds the SessionEnd recorder,
    which reads Claude Code's and Hermes's sessions and not Codex's."""
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
    for event, matcher in CLAUDE_TRIGGER:
        entries = data["hooks"].setdefault(event, [])
        if _trigger_command(entries):
            continue
        entry = {"hooks": [{"type": "command", "timeout": 5, "command": _hook_command("trigger")}]}
        entries.append({"matcher": matcher, **entry} if matcher else entry)
        added.append(f"trigger ({event})")
    event, verb, timeout = CLAUDE_USAGE
    if not usage:
        return added
    entries = data["hooks"].setdefault(event, [])
    if not _event_command(entries, verb):
        entries.append({"hooks": [{"type": "command", "timeout": timeout, "command": _hook_command(verb)}]})
        added.append(f"usage ({event})")
    return added


def _add_codex_hooks(data):
    return _add_claude_hooks(data, usage=False)


def _event_command(entries, verb):
    """The registered Locket `verb` command among one event's hook entries, or None."""
    for e in entries if isinstance(entries, list) else []:
        for h in (e.get("hooks") or []) if isinstance(e, dict) else []:
            cmd = str(h.get("command", "")) if isinstance(h, dict) else ""
            if _is_locket_hook(cmd, verb):
                return cmd
    return None


def _trigger_command(entries):
    return _event_command(entries, "trigger")


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
    if "--codex" in argv:
        rc |= _merge_json(memscan.CODEX / "hooks.json", _add_codex_hooks)
        print("Codex runs a hook only once it is trusted: start `codex`, run /hooks, and trust "
              "each Locket entry. A later change to an entry asks again; an upgrade does not")
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


def _hermes_lines(path):
    """`path:line` for every Hermes hook line that runs Locket, by script or command."""
    try:
        return [f"{path}:{i}" for i, l in enumerate(path.read_text().splitlines(), 1)
                if (c := re.search(r"command:\s*(.+?)\s*$", l)) and _is_locket_hook(c.group(1))]
    except OSError:
        return []


def _hermes_hooks(text):
    """{event: [command]} from config.yaml's top-level `hooks:` block, read by indentation
    in the shape Step 4 prints, so doctor needs no YAML parser."""
    out, inside, event, depth = {}, False, None, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            inside, event, depth = line.rstrip() == "hooks:", None, None
            continue
        if not inside:
            continue
        indent = len(line) - len(line.lstrip())
        depth = indent if depth is None else depth
        if indent == depth and (m := re.fullmatch(r"\s+([\w-]+):\s*", line)):
            event = m.group(1)
        elif (c := re.search(r"command:\s*(.+?)\s*$", line)) and event:
            out.setdefault(event, []).append(c.group(1).strip("'\""))
    return out


def cmd_uninstall(argv):
    yes, purge, dry = "--yes" in argv, "--purge" in argv, "--dry-run" in argv
    links = [BIN / n for n in LINKS if (BIN / n).is_symlink() and _ours(BIN / n)]
    locket_dir = memscan.REGISTRY.parent
    caches = _caches()
    manifests = _manifests()
    settings = memscan.CLAUDE / "settings.json"
    hook_lines, new_settings = _settings_hooks(settings) if settings.is_file() else ([], None)
    codex_cfg = memscan.CODEX / "hooks.json"
    codex_lines, new_codex = _settings_hooks(codex_cfg) if codex_cfg.is_file() else ([], None)
    hermes_cfg = memscan.HERMES / "config.yaml"
    hermes_lines = _hermes_lines(hermes_cfg) if hermes_cfg.is_file() else []
    desktop = DESKTOP_CONFIG
    mcp = False
    if desktop and desktop.is_file():
        try:
            mcp = "locket" in (json.loads(desktop.read_text()).get("mcpServers") or {})
        except (OSError, json.JSONDecodeError, ValueError):
            mcp = False

    # the fastembed venv every Panoply piece shares goes only with the last piece on PATH;
    # an older install made it inside ~/.locket, the rest in ~/.panoply
    import contextlib
    import _embed
    shared, others = _embed.shared_venv("locket")
    venv = locket_dir / "venv"
    # an older install's venv stays only while it is the one in use; beside ~/.panoply/venv
    # nothing reads it, and it goes with ~/.locket
    keep = venv.is_dir() and venv == _embed.VENV and shared != venv
    # the usage ledger holds days no source can give back, so only --purge takes it
    import usage
    ledger = usage.LEDGER if usage.LEDGER.is_file() and usage.LEDGER.parent == locket_dir and not purge else None
    spare = [p for p in (venv if keep else None, ledger) if p]
    print("uninstall will remove:")
    for l in links:
        print(f"  link      {l}")
    if locket_dir.is_dir():
        stays = " and ".join(n for n, p in (("its venv", keep), ("the usage ledger", ledger)) if p)
        print(f"  directory {locket_dir}  (registry, schema{f'; {stays} stays' if stays else ''})")
    if shared:
        print(f"  directory {shared}  (the fastembed venv; no other Panoply piece is on PATH)")
    for c in caches:
        print(f"  cache     {c}")
    if purge:
        for m in manifests:
            print(f"  manifest  {m}")
    for h in hook_lines:
        print(f"  hook      {settings}: {h}")
    for h in codex_lines:
        print(f"  hook      {codex_cfg}: {h}")
    if mcp:
        print(f"  mcp       {desktop}: mcpServers.locket")
    if not any((links, locket_dir.is_dir(), shared, caches, hook_lines, codex_lines, mcp, purge and manifests)):
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
    if locket_dir.is_dir() and spare:
        for child in locket_dir.iterdir():
            if child not in spare:
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
        print(f"removed {locket_dir}, all but {', '.join(map(str, spare))}")
    elif locket_dir.is_dir():
        shutil.rmtree(locket_dir); print(f"removed {locket_dir}")
    if shared and shared.is_dir():
        shutil.rmtree(shared); print(f"removed {shared}")
        with contextlib.suppress(OSError):
            shared.parent.rmdir()           # ~/.panoply, once nothing else is in it
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
    if codex_lines and new_codex is not None:
        bak = codex_cfg.with_suffix(".json.locket-uninstall.bak")
        shutil.copy2(codex_cfg, bak)
        codex_cfg.write_text(json.dumps(new_codex, indent=2) + "\n")
        print(f"edited {codex_cfg} ({len(codex_lines)} hook entries removed; copy at {bak})")
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
    if others:
        print(f"  the fastembed venv {_embed.VENV}, which {' and '.join(others)} still use{'s' if len(others) == 1 else ''}")
    elif _embed.VENV.is_dir() and not shared:
        print(f"  the fastembed venv you named: {_embed.VENV}")
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


def _probe_hook(cmd, codex=False):
    """Run a registered `hook` command on a write that must fire: a new file in a
    throwaway store repeating a paragraph the store already holds. True when it
    answered, False when it was silent, a string when it failed to run. For Codex
    the write arrives as the patch Codex sends, so the probe crosses the parser."""
    import subprocess, tempfile
    base = memscan.REGISTRY.parent
    base.mkdir(parents=True, exist_ok=True)
    store = Path(tempfile.mkdtemp(prefix="doctor-probe-", dir=base))
    try:
        (store / memscan.MANIFEST).write_text("{}\n")
        (store / "harbour.md").write_text(_PROBE + "\n")
        payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Write",
                              "tool_input": {"file_path": str(store / "zz-probe.md"), "content": _PROBE}}
                             if not codex else
                             {"hook_event_name": "PreToolUse", "tool_name": "apply_patch", "turn_id": "doctor",
                              "cwd": str(store), "tool_input": {"command": "*** Begin Patch\n*** Add File: "
                              "zz-probe.md\n+" + _PROBE + "\n*** End Patch"}})
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


def _selftest_uninstall():
    """`uninstall --yes` in throwaway homes, in a child process because Locket's paths are
    fixed from HOME at import: the shared venv goes only with the last piece on PATH, at
    either location, and an older ~/.locket/venv beside ~/.panoply/venv goes with ~/.locket."""
    import subprocess, tempfile
    cases = (  # venvs made, another piece on PATH, venvs that must survive
        ((".panoply/venv",), False, ()),
        ((".panoply/venv",), True, (".panoply/venv",)),
        ((".locket/venv",), False, ()),
        ((".locket/venv",), True, (".locket/venv",)),
        ((".panoply/venv", ".locket/venv"), True, (".panoply/venv",)),
    )
    for made, other, kept in cases:
        with tempfile.TemporaryDirectory() as t:
            h = Path(t)
            for v in made:
                (h / v).mkdir(parents=True)
            (h / "bin").mkdir()
            if other:
                (h / "bin/grille").write_text("#!/bin/sh\n")
                (h / "bin/grille").chmod(0o755)
            r = subprocess.run([sys.executable, str(HERE / "locket.py"), "uninstall", "--yes"],
                               env={"HOME": t, "PATH": str(h / "bin")}, capture_output=True, text=True)
            left = tuple(v for v in (".panoply/venv", ".locket/venv") if (h / v).is_dir())
            assert r.returncode == 0 and left == kept, \
                f"made {made}, grille on PATH {other}: left {left}, expected {kept}\n{r.stdout}{r.stderr}"
            assert kept or not (h / ".panoply").exists(), "an empty ~/.panoply outlived its venv"
    with tempfile.TemporaryDirectory() as t:     # doctor runs to its summary, rows file and all
        (Path(t) / ".claude").mkdir()
        (Path(t) / ".claude/triggers.json").write_text('{"rows": []}\n')
        r = subprocess.run([sys.executable, str(HERE / "locket.py"), "doctor"], capture_output=True, text=True,
                           env={"HOME": t, "PATH": "/usr/bin:/bin", "MEMFIND_NO_AUTOSTART": "1"})
        assert " checks: " in r.stdout and "Traceback" not in r.stderr, f"doctor did not finish:\n{r.stderr[-600:]}"
        assert "triggers" in r.stdout, "doctor skipped a host rows file"
        assert "usage" in r.stdout and "never recorded" in r.stdout, "doctor skipped the usage row"
    run = lambda t, *a: subprocess.run([sys.executable, str(HERE / "locket.py"), *a], capture_output=True, text=True,
                                       env={"HOME": t, "PATH": "/usr/bin:/bin", "MEMFIND_NO_AUTOSTART": "1"})
    for purge in (False, True):     # the ledger survives uninstall, the SessionEnd hook does not
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / ".claude").mkdir()
            (Path(t) / ".locket").mkdir()
            (Path(t) / ".locket/usage.csv").write_text("date\n")
            r = run(t, "install", "--hooks")
            hooks = json.loads((Path(t) / ".claude/settings.json").read_text())["hooks"]
            assert _event_command(hooks.get("SessionEnd"), "usage hook"), f"install registered no usage hook:\n{r.stdout}"
            r = run(t, "uninstall", "--yes", *(["--purge"] if purge else []))
            hooks = json.loads((Path(t) / ".claude/settings.json").read_text())["hooks"]
            assert not _event_command(hooks.get("SessionEnd"), "usage hook"), "uninstall left the usage hook"
            assert (Path(t) / ".locket/usage.csv").is_file() != purge, \
                f"uninstall {'kept' if purge else 'removed'} the usage ledger with purge={purge}\n{r.stdout}{r.stderr}"


def _probe_trigger(cmd, hermes=False):
    """Run a registered `trigger` command against a throwaway home whose host store holds
    one row the payload matches. True when the row's question came back, False when the
    command was silent, a string when it failed to run. On Hermes the row refuses the call
    once, so the answer is exit 2 with the question in the block message."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as t:
        home = Path(t) / (".hermes" if hermes else ".claude")
        home.mkdir()
        (home / "triggers.json").write_text(json.dumps({"rows": [
            {"id": "doctor-probe", "tools": "Bash", "content": "locket-doctor-probe", "question": "probe"}]}))
        payload = json.dumps({"hook_event_name": "pre_tool_call" if hermes else "PreToolUse",
                              "tool_name": "terminal" if hermes else "Bash", "cwd": t, "session_id": "",
                              "tool_input": {"command": "echo locket-doctor-probe"}})
        try:        # TMPDIR too: the refusal record must not outlive the probe
            r = subprocess.run(cmd, shell=True, input=payload, capture_output=True, text=True, timeout=30,
                               env={**os.environ, "HOME": t, "TMPDIR": t})
        except subprocess.TimeoutExpired:
            return "timed out after 30s"
        if r.returncode != (2 if hermes else 0):
            return (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
        return "[doctor-probe] probe" in r.stdout


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

    codex_cfg = memscan.CODEX / "hooks.json"
    if memscan.CODEX.is_dir():
        hooks = _registered_hooks(codex_cfg) if codex_cfg.is_file() else {}
        missing = [v for _, v, _t in CLAUDE_HOOKS if v not in hooks] if hooks is not None else []
        if hooks is None:
            row("fail", "codex hooks", f"{codex_cfg} is not valid JSON", f"fix {codex_cfg}, then locket install --codex")
        elif missing:
            row("warn", "codex hooks", f"not registered: {', '.join(missing)}", "locket install --codex")
        else:
            probe = _probe_hook(hooks["hook"], codex=True)
            if probe is True:
                # where Codex records hook trust is not documented, so the trust step is named, never
                # checked; the probe runs the command directly, so it proves the hook, not Codex's dispatch
                row("warn", "codex hooks", "registered, and the write hook fires on a known fork in a patch; "
                    "Codex skips them until trusted", "in codex, run /hooks and trust each Locket entry")
            else:
                row("fail", "codex hooks", "registered, but the write hook " +
                    ("stayed silent on a known fork" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {hooks['hook']}")

    import trigger
    rows_path = trigger.rows_file(memscan.CLAUDE)
    if rows_path.is_file():
        try:
            events = (json.loads(settings.read_text()).get("hooks") or {}) if settings.is_file() else {}
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            events = None
        unregistered = [e for e, _m in CLAUDE_TRIGGER
                        if events is not None and not _trigger_command(events.get(e) if isinstance(events, dict) else None)]
        problems = trigger.findings(memscan.CLAUDE)
        if unregistered:
            row("fail", "triggers", f"{rows_path} has rows, but the hook is not registered for "
                f"{' or '.join(unregistered)}", "locket install --hooks")
        elif problems:
            row("fail", "triggers", f"{len(problems)} problem(s) in {rows_path}: {problems[0]}", "locket trigger check")
        elif events is not None:
            probe = _probe_trigger(_trigger_command(events.get("PreToolUse")))
            if probe is True:
                row("ok", "triggers", f"{rows_path} lints clean; the hook is registered and fires on a known row")
            else:
                row("fail", "triggers", "registered, but the hook " +
                    ("stayed silent on a known row" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {_trigger_command(events.get('PreToolUse'))}")

    import time
    import usage
    if memscan.CLAUDE.is_dir():
        try:
            events = (json.loads(settings.read_text()).get("hooks") or {}) if settings.is_file() else {}
            registered = _event_command(events.get(CLAUDE_USAGE[0]), CLAUDE_USAGE[1])
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            registered = None
        age = (time.time() - usage.LEDGER.stat().st_mtime) / 86400 if usage.LEDGER.is_file() else None
        seen = "never recorded" if age is None else f"last recorded {age:.1f} days ago"
        stale = registered and (p := _script_in(registered)) and not p.is_file()
        if stale:
            row("fail", "usage", f"the SessionEnd hook names a missing script: {p}",
                "locket uninstall --dry-run, then remove the stale entry and locket install --hooks")
        elif not registered:
            row("warn", "usage", f"{seen}; nothing records it as a session ends, so a day older than "
                "Claude Code keeps its transcripts is lost unless `locket usage` ran within it", "locket install --hooks")
        else:
            row("ok", "usage", f"recorded as each session ends; {seen}")

    hermes_cfg = memscan.HERMES / "config.yaml"
    if hermes_cfg.is_file():
        events = _hermes_hooks(hermes_cfg.read_text(errors="replace"))
        missing = [f"{v} on {e}" for e, v in HERMES_HOOKS
                   if not any(_is_locket_hook(c, v) for c in events.get(e, []))]
        retired = any(_is_locket_hook(c, "deliver") for cmds in events.values() for c in cmds)
        if missing:
            row("warn", "hermes hooks", f"not in config.yaml: {', '.join(missing)}"
                + ("; `locket deliver` is retired, remove it" if retired else ""), "`locket help install`, Step 4")
        elif retired:
            row("warn", "hermes hooks", "`locket deliver` is retired and does nothing", "remove its pre_llm_call entry")
        else:
            row("ok", "hermes hooks", "all five registered")
        cmd = next((c for c in events.get("pre_tool_call", []) if _is_locket_hook(c, "trigger")), None)
        hermes_rows = trigger.rows_file(memscan.HERMES)
        problems = trigger.findings(memscan.HERMES)
        if problems:
            row("fail", "hermes triggers", f"{len(problems)} problem(s) in {hermes_rows}: {problems[0]}",
                "locket trigger check hermes")
        elif hermes_rows.is_file() and cmd:
            probe = _probe_trigger(cmd, hermes=True)
            if probe is True:
                row("ok", "hermes triggers", f"{hermes_rows} lints clean; the hook refuses a call on a known row")
            else:
                row("fail", "hermes triggers", "registered, but the hook " +
                    ("stayed silent on a known row" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {cmd}")

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
    if verb == "hooks" or (verb == "help" and rest[:1] == ["hooks"]):   # pointers' older name
        verb, rest = ("pointers", rest) if verb == "hooks" else ("help", ["pointers", *rest[1:]])
    if verb in ("--version", "-V", "-v", "version"):
        print(f"locket {__version__}"); return 0
    if verb == "help":
        which = rest[0] if rest else ""
        if which in _parser()[1].choices and which not in ("find", "install", "trigger", "usage"):
            _parser()[1].choices[which].print_help(); return 0
        if which in ("find", "memfind"):
            return _memfind().main(["memfind.py", "--help"])
        if which in ("scan", "memscan"):
            return memscan.main(["memscan.py", "--help"])
        if which == "trigger":
            import trigger
            _parser()[1].choices["trigger"].print_help()
            print("\n" + trigger.__doc__.split("\n\n", 1)[1].rstrip()); return 0
        if which == "usage":
            import usage
            _parser()[1].choices["usage"].print_help()
            print("\n" + usage.__doc__.split("\n\n", 1)[1].rstrip()); return 0
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
            assert _probe_hook(live, codex=True) is True, \
                "doctor's Codex probe does not fire through a patch into this memscan.py"
            merged = _add_codex_hooks({})
            assert "usage (SessionEnd)" not in merged and {"hook", "bashguard", "grepassist"} <= set(merged), merged
            data = {}
            _add_codex_hooks(data)
            assert _add_codex_hooks(data) == [], "a second --codex added its hooks again"
            assert "SessionEnd" not in data["hooks"], "Codex got the usage recorder, which cannot read its sessions"
            live_trigger = f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'locket.py'))} trigger"
            assert _probe_trigger(live_trigger) is True, "doctor's trigger probe does not fire through this locket.py"
            assert _probe_trigger("true") is False, "doctor's trigger probe reads a silent command as firing"
            assert _probe_trigger(live_trigger, hermes=True) is True, \
                "doctor's trigger probe does not fire through this locket.py on Hermes"
            assert _probe_trigger("echo '[doctor-probe] probe'", hermes=True) is not True, \
                "doctor's Hermes trigger probe reads a call that was not refused as firing"
            assert _probe_trigger("exit 2", hermes=True) is False, \
                "doctor's Hermes trigger probe reads a bare refusal as firing"
            yaml = ('model: x\nhooks:\n  pre_tool_call:\n    - command: "/opt/homebrew/bin/locket hook"\n'
                    '      matcher:\n        nested: x\n    - command: /opt/homebrew/bin/locket trigger\n'
                    '  pre_llm_call:\n    - command: "/opt/homebrew/bin/locket trigger"\n'
                    'other:\n  pre_tool_call:\n    - command: "locket bashguard"\n')
            assert _hermes_hooks(yaml) == {"pre_tool_call": ["/opt/homebrew/bin/locket hook",
                                                             "/opt/homebrew/bin/locket trigger"],
                                           "pre_llm_call": ["/opt/homebrew/bin/locket trigger"]}, _hermes_hooks(yaml)
            assert isinstance(_probe_hook(f"{shlex.quote(sys.executable)} /nonexistent/memscan.py hook"), str), \
                "doctor's probe reads a missing script as firing"
            import contextlib
            import io
            for form in (["audit", "council", "-h"], ["audit", "help"], ["help", "audit"], ["help", "pointers"],
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
            _selftest_uninstall()
            import trigger
            import usage
            rc = trigger.selftest() or usage.selftest()
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
    if verb == "trigger":
        import trigger
        return trigger.main(rest)
    if verb == "usage":
        import usage
        return usage.hook() if rest == ["hook"] else usage.main(rest)
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
