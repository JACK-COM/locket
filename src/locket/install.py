"""How Locket is installed on this machine: the command and its link, the hook entries each
host's config carries, and their removal.
"""
import contextlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import corpora
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
# The hook commands run through locket.py rather than memscan.py: the trigger, and the
# usage recorder a SessionEnd hook runs.
USAGE_HOOK = "usage hook"
LOCKET_VERBS = ("trigger", USAGE_HOOK)


def load_memfind():
    import memfind
    return memfind


def link_target(link):
    try:
        return Path(os.readlink(link)) if link.is_symlink() else None
    except OSError:
        return None


def ours(link):
    """A link in BIN that points into this directory, or dangles there."""
    t = link_target(link)
    return t is not None and (t.parent == HERE or not t.exists())


def rewrite_json(path, data, label):
    """Replace a host's JSON config with `data` after uninstall took Locket's entries out:
    a copy beside it first, then written whole or not at all, non-ASCII kept as written."""
    bak = path.with_suffix(".json.locket-uninstall.bak")
    shutil.copy2(path, bak)
    _settings.write_json(path, data)
    print(f"edited {path} ({label}; copy at {bak})")


class Removal(NamedTuple):
    """What uninstall does to one host's config: `shown` as (kind, what) lines before it
    asks, `apply` to run once it may (None when there is nothing to edit), `left` the
    lines a person removes by hand."""
    shown: list
    apply: Optional[Callable[[], None]]
    left: list


def merge_json(path, merge):
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


def is_locket_hook(cmd, verb=None):
    """A hook command that runs Locket: memscan.py by path, or the `locket` command."""
    from commands import hooks          # the hidden verbs memscan dispatches; imported here, since they import us
    verbs = [verb] if verb else (*hooks.ENTRIES, *LOCKET_VERBS)
    cmd = cmd.replace("'", "").replace('"', "").rstrip()   # a quoted path, as hook_command writes one
    return any(f"memscan.py {v}" in cmd or cmd.endswith((f"locket {v}", f"locket.py {v}")) for v in verbs)


def command():
    """The `locket` command's absolute path for a packaged install, or None."""
    w = shutil.which("locket") if PACKAGED else None
    return str(Path(w).absolute()) if w else None


def hook_command(verb):
    cmd = command()
    script = HERE / ("locket.py" if verb in LOCKET_VERBS else "memscan.py")
    return f"{shlex.quote(cmd)} {verb}" if cmd else f"python3 {script} {verb}"


def cmd_install(argv=()):
    if PACKAGED:
        print(f"command: {command() or 'locket, managed by the package manager'}")
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
    print(embedder_line())
    from adapters import TARGETS
    for target in TARGETS:                  # each host's own flag: --hooks, --codex, --desktop
        if target.FLAG and target.FLAG in argv:
            rc |= target.register()
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
        if link_target(link) == me:
            print(f"link: {link} -> {me} (already)")
        elif ours(link) or (link_target(link) or Path()).name == "locket.py":
            link.unlink(); link.symlink_to(me); print(f"link: {link} -> {me} (replaced)")
        else:
            print(f"link: {link} exists and is not Locket's; leaving it. Remove it and rerun.")
            return 1
    else:
        link.symlink_to(me); print(f"link: {link} -> {me}")
    for old in LINKS[1:]:                   # the two names this command replaces
        l = BIN / old
        if l.is_symlink() and ours(l):
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


def embedder_line():
    try:
        name, model = load_memfind().resolve_backend()
        return f"embedder: {name} ({model})"
    except RuntimeError as e:
        return f"embedder: none ({str(e).splitlines()[0]}); `locket find` will fall back to word overlap"


def caches():
    """Every in-store `.memfind/` cache a release before 0.12 may have written: at each
    store root, in each member of a host store (older layout), and in every registered
    path. Nothing reads them now; `migrate` names them and `uninstall` removes them."""
    roots = set(corpora.raw_corpora().values()) | set(trust.registry())
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
    roots = set(corpora.raw_corpora().values()) | set(trust.registry())
    out = []
    for r in roots:
        for name in (manifest.MANIFEST, manifest.OLD_MANIFEST):
            if (r / name).is_file():
                out.append(r / name)
    return sorted(out)


def scripts_removal():
    """The command that removes Locket's own files. Where HERE is the package directory
    (it holds the package's `__init__.py`) that is the whole directory, so a module added
    later is never left behind; a loose copy among other scripts names Locket's entry
    scripts only, since its other files cannot be told from a neighbour's."""
    if (HERE / "__init__.py").is_file():
        return f"rm -r {shlex.quote(str(HERE))}"
    named = [HERE / s for s in (*SCRIPTS, "INSTALL-locket.md") if (HERE / s).exists()]
    return "rm " + " ".join(shlex.quote(str(p)) for p in named)


def cmd_uninstall(argv):
    yes, purge, dry = "--yes" in argv, "--purge" in argv, "--dry-run" in argv
    links = [BIN / n for n in LINKS if (BIN / n).is_symlink() and ours(BIN / n)]
    locket_dir = paths.PATHS.state
    stale = caches()
    manifests = _manifests()
    from adapters import TARGETS
    removals = [t.removal() for t in TARGETS]
    shown = [line for r in removals for line in r.shown]

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
    for c in stale:
        print(f"  cache     {c}")
    if purge:
        for m in manifests:
            print(f"  manifest  {m}")
    for kind, what in shown:
        print(f"  {kind:<9} {what}")
    if not any((links, locket_dir.is_dir(), shared, stale, shown, purge and manifests)):
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
    for c in stale:
        shutil.rmtree(c, ignore_errors=True); print(f"removed {c}")
    if purge:
        for m in manifests:
            m.unlink(); print(f"removed {m}")
    for r in removals:
        if r.apply:
            r.apply()
    print("\nleft for you, if you want it gone completely:")
    if manifests and not purge:
        print("  manifests, inert without the tool and often under version control:")
        for m in manifests:
            print(f"    {m}")
        print("    (rerun with --purge to delete them)")
    for r in removals:
        for line in r.left:
            print(f"  {line}")
    print(f"  the scripts:  {scripts_removal()}")
    print("  an editor `json.schemas` entry pointing at ~/.locket/locket.schema.json, if you added one")
    if isinstance(_settings.load(check=False)[0].get("locket"), dict):
        print(f"  Locket's embedder overrides: the \"locket\" section of {_settings.path()}")
    if others:
        print(f"  the embedder venv {_embed.VENV}, which {' and '.join(others)} still use{'s' if len(others) == 1 else ''}")
    elif _embed.VENV.is_dir() and not shared:
        print(f"  the embedder venv you named: {_embed.VENV}")
    return 0


def script_in(cmd):
    """The first path in a hook command that names a .py file, or None."""
    for tok in cmd.split():
        if tok.endswith(".py"):
            return Path(os.path.expanduser(tok))
    return None
