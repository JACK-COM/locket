"""How Locket is installed on this machine: the command and its link, the hook entries each
host's config carries, and their removal.
"""
import contextlib
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import corpora
import hosts
import manifest
import paths
import scaffold
import _settings
import trust


HERE = Path(__file__).resolve().parent
BIN = Path(os.environ.get("LOCKET_BIN", str(Path.home() / ".local" / "bin")))
# A package manager owns the command and moves the scripts on every upgrade: Homebrew's
# wrapper sets LOCKET_PACKAGED, a `uv tool` or pip install enters through `cli()`. A
# packaged install registers hooks and the Desktop server through the command, whose
# path survives an upgrade, and leaves the link in BIN to the package manager.
PACKAGED = os.environ.get("LOCKET_PACKAGED") == "1"
LINKS = ("locket", "memscan", "memfind")      # the last two are older spellings this replaces
SCRIPTS = ("memscan.py", "memfind.py", "locket.py", "locket_mcp.py", "trigger.py", "usage.py", "configure.py")
HOOK_VERBS = ("hook", "bashguard", "grepassist", "deliver")   # PreToolUse entry points, JSON on stdin
# The (event, verb) entries Hermes's config.yaml needs. `deliver` is retired and does nothing.
HERMES_HOOKS = (("pre_tool_call", "hook"), ("pre_tool_call", "bashguard"), ("pre_tool_call", "grepassist"),
                ("pre_tool_call", "trigger"), ("pre_llm_call", "trigger"))


def _memfind():
    import memfind
    return memfind


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
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
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
    _settings.write_json(path, data)       # whole or not at all, and a symlinked file keeps its link
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
        scaffold.cmd_schema(quiet=True); print(f"schema: {paths.PATHS.schema_file}")
    except OSError as e:
        print(f"schema: not written ({e})")
    stale = paths.PATHS.schema_file.parent / "memfind.schema.json"   # the file's name before the rename
    if stale.is_file():
        stale.unlink(); print(f"schema: {stale} removed (older name)")
    for cfg in _editor_settings():
        if "memfind.schema.json" in cfg.read_text(errors="replace"):
            print(f"editor: {cfg} still maps the manifest to memfind.schema.json; change that entry to\n"
                  f'        {{"fileMatch": ["locket.json", "memfind.json"], "url": "file://{paths.PATHS.schema_file}"}}')
    print(_embedder_line())
    if "--hooks" in argv:
        rc |= _merge_json(hosts.claude() / "settings.json", _add_claude_hooks)
    if "--codex" in argv:
        rc |= _merge_json(hosts.codex() / "hooks.json", _add_codex_hooks)
        print("Codex runs a hook only once it is trusted: start `codex`, run /hooks, and trust "
              "each Locket entry. A later change to an entry asks again; an upgrade does not")
        profiles = sorted(f.name for f in hosts.codex().glob("*.config.toml"))
        if profiles:
            print(f"Codex profile config found ({', '.join(profiles)}): trust given in /hooks under --profile "
                  "(as `ollama launch codex` runs it) is written there and lapses on restart; copy its "
                  "[hooks.state] tables into config.toml, which `locket doctor` checks")
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
    """Every in-store `.memfind/` cache a release before 0.12 may have written: at each
    store root, in each member of a host store (older layout), and in every registered
    path. Nothing reads them now; `migrate` names them and `uninstall` removes them."""
    roots = set(corpora._raw_corpora().values()) | set(trust.registry())
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
    roots = set(corpora._raw_corpora().values()) | set(trust.registry())
    out = []
    for r in roots:
        for name in (manifest.MANIFEST, manifest.OLD_MANIFEST):
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
    locket_dir = paths.PATHS.state
    caches = _caches()
    manifests = _manifests()
    settings = hosts.claude() / "settings.json"
    hook_lines, new_settings = _settings_hooks(settings) if settings.is_file() else ([], None)
    codex_cfg = hosts.codex() / "hooks.json"
    codex_lines, new_codex = _settings_hooks(codex_cfg) if codex_cfg.is_file() else ([], None)
    hermes_cfg = hosts.hermes() / "config.yaml"
    hermes_lines = _hermes_lines(hermes_cfg) if hermes_cfg.is_file() else []
    desktop = DESKTOP_CONFIG
    mcp = False
    if desktop and desktop.is_file():
        try:
            mcp = "locket" in (json.loads(desktop.read_text()).get("mcpServers") or {})
        except (OSError, json.JSONDecodeError, ValueError):
            mcp = False

    # the embedder venv every Panoply piece shares goes only with the last piece on PATH;
    # an older install made it inside ~/.locket, the rest in ~/.panoply
    import _embed
    import configure
    configure.apply()                       # a configured venv is the one in use
    shared, others = _embed.shared_venv("locket")
    venv = locket_dir / "venv"
    # an older install's venv stays only while it is the one in use; beside ~/.panoply/venv
    # nothing reads it, and it goes with ~/.locket
    keep = venv.is_dir() and venv == _embed.VENV and shared != venv
    # the usage ledger holds days no source can give back, so only --purge takes it
    ledger = paths.PATHS.usage_ledger if paths.PATHS.usage_ledger.is_file() and paths.PATHS.usage_ledger.parent == locket_dir and not purge else None
    spare = [p for p in (venv if keep else None, ledger) if p]
    print("uninstall will remove:")
    for l in links:
        print(f"  link      {l}")
    if locket_dir.is_dir():
        stays = " and ".join(n for n, p in (("its venv", keep), ("the usage ledger", ledger)) if p)
        print(f"  directory {locket_dir}  (registry, schema, settings{f'; {stays} stays' if stays else ''})")
    if shared:
        print(f"  directory {shared}  (the embedder venv and its model; no other Panoply piece is on PATH)")
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
    if purge and manifests and not trust.person_confirms(
            f"Delete the locket.json in {len(manifests)} stores, and the usage ledger?",
            [str(m) for m in manifests[:8]], "delete them"):
        # every store's configuration at once is a person's call, --yes or not
        print("nothing removed: --purge deletes every store's locket.json, so a person confirms it "
              "at a terminal. Run it yourself, or uninstall without --purge"); return 1
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
    if isinstance(_settings.load(check=False)[0].get("locket"), dict):
        print(f"  Locket's embedder overrides: the \"locket\" section of {_settings.path()}")
    if others:
        print(f"  the embedder venv {_embed.VENV}, which {' and '.join(others)} still use{'s' if len(others) == 1 else ''}")
    elif _embed.VENV.is_dir() and not shared:
        print(f"  the embedder venv you named: {_embed.VENV}")
    return 0
