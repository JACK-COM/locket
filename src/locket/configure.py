#!/usr/bin/env python3
"""configure: Locket's embedder settings, held in the Panoply's one settings file.

The file is ~/.panoply/config.json (PANOPLY_CONFIG names another), shared by every
Panoply piece. Its top-level `embed` section is the embedder every piece uses, and a
section named for a piece holds that piece's own overrides:

    {"embed":  {"model": "embeddinggemma-2:270m", "venv": "~/.panoply/venv"},
     "locket": {"embed": {"model": "nomic-embed-text"}}}

`locket configure embedder` writes Locket's own section; `--global` writes the top level,
which reaches every piece that does not override that key. Every way Locket runs, the
command line, the Claude Code, Codex and Hermes hooks, the MCP server, reads the file, so
one change reaches all of them.

Each setting resolves in this order, first found wins:
  1. an environment variable (MEMFIND_MODEL, OLLAMA_HOST, MEMFIND_NO_AUTOSTART=1,
     PANOPLY_VENV or LOCKET_VENV), for one shell or one test;
  2. Locket's own section;
  3. the top level;
  4. the shipped default.
A variable set in one place and not another is how two processes come to disagree, so
`locket configure` and `locket doctor` name every variable that is overriding the file.

An index is built in one embedder's space and tagged with it. After a model change every
store's index belongs to the old model: `find` and `index` rebuild a store when run, and
the hooks stay silent for that store until it is rebuilt, because a hook never rebuilds.
`--index` rebuilds every store as part of the change; without it, run `locket index all`
when convenient. `locket configure` lists which stores are behind.

Locket 0.10 kept these settings in ~/.locket/config.json. While that file remains, Locket
reads it below the top level; the next `locket configure` moves its settings into the
top level and renames it config.json.migrated.

Write the file with `configure`, never by hand while another configure runs: it holds a
lock, writes atomically and checks the result against the schema (`locket configure
schema` prints it).
"""
import argparse
import contextlib
import json
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _embed  # noqa: E402
import _settings  # noqa: E402

PIECE = "locket"
LEGACY = None           # Locket 0.10's file; resolved on use, so an unset HOME cannot break an import


def _legacy_path():
    return LEGACY or Path.home() / ".locket" / "config.json"


def _legacy():
    """The usable embed settings of Locket 0.10's file while it remains, else None."""
    try:
        data = json.loads(_legacy_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError):
        return None
    e = data.get("embed") if isinstance(data, dict) else None
    return {k: v for k, v in e.items() if k in _embed.CONFIG_KEYS and _embed._usable(k, e)} \
        if isinstance(e, dict) else None


def _layers(data):
    """Locket's view of the embed settings in `data`, highest first: its own section, the
    top level, then an unmigrated ~/.locket/config.json."""
    out = _settings.layers(PIECE, "embed", data)
    legacy = _legacy()
    return out + [("~/.locket/config.json", legacy)] if legacy else out


def apply():
    """Put Locket's embed settings into the embedder, under the environment. Every module
    that ranks calls this once at import; it is cheap and changes nothing without a file."""
    try:
        return _embed.apply_config(_layers(_settings.load(check=False)[0]))
    except Exception:                       # nothing in a settings file may stop a hook
        return _embed.apply_config(None)


def migrate():
    """Move ~/.locket/config.json's settings into the shared file's top level, where a key
    already set keeps its value, and rename the old file, both under the shared file's lock.
    Leaves everything in place when either file cannot be read. Returns a line, or None."""
    old = _legacy_path()
    if not old.exists():
        return None
    legacy = _legacy()
    if legacy is None:
        return f"note: {old} has no readable embed section; left in place"
    with _settings.lock():
        data, findings = _settings.load()
        if _settings.unreadable(findings) or not isinstance(data.get("embed", {}), dict):
            return f"note: {_settings.path()} cannot be read, so {old} was left in place; fix it and rerun"
        top = data.get("embed", {})
        add = {k: v for k, v in legacy.items() if k not in top}
        if add:
            _settings.write_json(_settings.path(), _settings.edit(data, "embed", add))
        done = old.with_name("config.json.migrated")
        if done.exists():               # an earlier migration's copy is kept, never overwritten
            done = old.with_name(f"config.json.migrated-{int(time.time())}")
        try:
            old.rename(done)
        except FileNotFoundError:       # another process moved it first
            return None
    return f"moved {old} into {_settings.path()}; the old file is now {done.name}"


def _ollama_check(model):
    """(ok, message) for whether ollama serves `model` as an embedding model. Unreachable
    ollama is ok with a warning: the file may be written ahead of the server."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(f"{_embed.OLLAMA}/api/show", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with _embed._open(req, 5) as r:
            info = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False, (f"ollama at {_embed.OLLAMA} does not have {model!r}. Pull it first:\n"
                           f"  ollama pull {model}\n(or pass --no-check to write the setting anyway)")
        return True, f"warning: ollama answered {e.code} on /api/show; {model!r} not verified"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return True, f"warning: ollama not answering at {_embed.OLLAMA} ({getattr(e, 'reason', e)}); {model!r} not verified"
    caps = info.get("capabilities")
    if isinstance(caps, list) and "embedding" not in caps:
        return False, (f"{model!r} is not an embedding model (ollama lists {', '.join(caps) or 'nothing'}); "
                       "pick one whose `ollama show` lists embedding")
    return True, ""


def _index_state():
    """[(store, path, state)] for every store with markdown, state being "current", "absent"
    or "built with <tag>". Reads each index's tag only: no embedder is probed or started."""
    import memfind
    import memscan
    now = {_embed.tag("ollama"), _embed.tag("onnx")}
    out = []
    for name, root in sorted(memscan.all_corpora().items()):
        if not memscan.md_files(root):
            continue
        meta = memfind.cache_paths(root)[0]
        try:
            model = json.loads(meta.read_text(encoding="utf-8")).get("model")
        except (OSError, ValueError, AttributeError):   # unreadable, not JSON, or not an object
            model = None
        model = model if isinstance(model, str) else None
        out.append((name, root, "absent" if model is None else "current" if model in now else f"built with {model}"))
    return out


def show():
    data, findings = _settings.load()
    path = _settings.path()
    print(f"settings file: {path}" + ("" if path.exists() else "  (absent: every setting is its default)"))
    for f in findings:
        print(f"  problem: {f}")
    print("\nembedder, as Locket uses it   (source: env, locket's own section, global, or default)")
    resolved = apply()
    in_file = {k for _, layer in _layers(data) for k in layer}
    for key, (value, src) in resolved.items():
        shown = ("on" if value else "off") if key == "autostart" else value
        note = "  <- overrides the file" if src.startswith("env") and key in in_file else ""
        print(f"  {key:12s} {str(shown):40s} {src}{note}")
    print(f"  {'in-process':12s} {_embed.ONNX_MODEL}@{_embed.ONNX_REVISION[:12]}  (fixed; answers when ollama "
          f"does not{', a different model from the one above' if resolved['model'][0] != _embed.SHIPPED['model'] else ''})")
    others = {p: sorted(data[p]["embed"]) for p in _settings.PIECES if p != PIECE and isinstance(data.get(p), dict)
              and isinstance(data[p].get("embed"), dict) and data[p]["embed"]}
    for p, keys in others.items():
        print(f"  {p} overrides for itself: {', '.join(keys)}")
    state = _index_state()
    behind = [s for s in state if s[2] != "current"]
    print(f"\nindexes: {len(state)} store(s), {len(state) - len(behind)} current")
    for name, root, st in behind:
        print(f"  {name:20s} {st}")
    if behind:
        print("  rebuild with:  locket index all   (hooks stay silent for these until then)")
    return 1 if findings else 0


def _set_embed(args):
    keys = ("model", "ollama_host", "venv", "autostart")
    clearing = {k for k in keys if getattr(args, k) == "default"}
    changes = {k: getattr(args, k) for k in keys if getattr(args, k) not in (None, "default")}
    if not (changes or clearing or args.reset or args.index):
        print("nothing to change: name a setting (see `locket configure embedder -h`)", file=sys.stderr)
        return 2
    if "venv" in changes:                   # absolute: a hook resolves a relative path against its own cwd
        changes["venv"] = os.path.abspath(Path(changes["venv"]).expanduser())
    if "ollama_host" in changes and not changes["ollama_host"].startswith(("http://", "https://")):
        print(f"--ollama-host needs a scheme, e.g. http://{changes['ollama_host']}", file=sys.stderr)
        return 2
    if "model" in changes and not args.no_check:
        # ask ollama before taking the lock, against the host the new settings would use
        new = _settings.edit(_settings.load(check=False)[0], "embed", changes, clearing, args.reset,
                             None if args.shared else PIECE)
        view = [("global", new.get("embed"))] if args.shared else _layers(new)
        with _embed.settings(**{n: getattr(_embed, n) for n in ("MODEL", "OLLAMA", "AUTOSTART", "VENV")}):
            _embed.apply_config(view)
            ok, msg = _ollama_check(changes["model"])
        if msg:
            print(msg, file=sys.stderr)
        if not ok:
            return 1
    outcome = {}
    change = lambda data, findings: _change(args, changes, clearing, data, findings, outcome)
    try:
        if args.dry_run:
            change(*_settings.load())
        else:
            written = _settings.update(change)
            if "rc" not in outcome:
                print(f"wrote {_settings.path()}" if written else "no change to write")
    except TimeoutError as e:
        print(f"not written: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"cannot write {_settings.path()}: {e}", file=sys.stderr)
        return 1
    if "rc" in outcome:
        return outcome["rc"]
    if args.shared:
        keys = set(changes) | clearing | (set(_embed.CONFIG_KEYS) if args.reset else set())
        data = _settings.load()[0]
        for k in sorted(keys):
            for p in _settings.overriding(data, "embed", k):
                print(f"note: {p} sets its own {k}, so this change does not reach {p}; "
                      f"`{p} configure embedder --{k.replace('_', '-')} default` makes it follow the global one")
    show()
    if args.index:
        import memfind
        print("\nre-indexing every store with the new settings")
        rc = memfind.main(["memfind.py", "--index", "all"])
        name = _embed.resolve_backend()[0]
        if name != "ollama":
            print(f"warning: ollama did not answer, so the stores were indexed in-process with "
                  f"{_embed.ONNX_MODEL}, not {_embed.MODEL}; run `locket index all` once ollama is up",
                  file=sys.stderr)
        return rc
    return 0


def _change(args, changes, clearing, data, findings, outcome):
    """The new settings for `_settings.update`, or None to write nothing, with the exit
    code to stop with left in `outcome["rc"]`. Runs under the lock unless a dry run. Only
    an unreadable file blocks, since writing over it loses every piece's settings; --reset
    writes anyway, after `update` copies the old file to config.json.bak."""
    if _settings.unreadable(findings) and not args.reset:
        print(f"{findings[0]}\nfix it, or pass --reset to start the file over (the old one is kept "
              "as config.json.bak)", file=sys.stderr)
        outcome["rc"] = 1
        return None
    new = _settings.edit(data, "embed", changes, clearing, args.reset, None if args.shared else PIECE)
    if args.dry_run:
        print(json.dumps(new, indent=2))
        outcome["rc"] = 0
        return None
    return new


def _wrap(text):
    return textwrap.fill(text, 78)


def _parser():
    fmt = dict(formatter_class=argparse.RawDescriptionHelpFormatter)
    p = argparse.ArgumentParser(
        prog="locket configure", **fmt,
        description=_wrap("Show or change Locket's embedder settings, held in ~/.panoply/config.json, the "
                    "settings file every Panoply piece and every Locket process reads (the command line, "
                    "every hook, the MCP server). Run with no arguments to see each setting, where its "
                    "value comes from, and which stores' indexes are behind."),
        epilog="examples:\n"
               "  locket configure                                   show settings, their sources and index state\n"
               "  locket configure embedder --model nomic-embed-text switch Locket's model; indexes rebuild later\n"
               "  locket configure embedder --model nomic-embed-text --global\n"
               "                                                     switch every piece that does not set its own\n"
               "  locket configure embedder --model nomic-embed-text --index\n"
               "                                                     switch model and rebuild every index now\n"
               "  locket configure embedder --model default          back to the shipped model\n"
               "  locket configure embedder --reset --index          every embedder setting back to its default\n"
               "  locket configure schema                            the file's JSON Schema, for an editor\n\n"
               "Precedence, first found wins: environment variable, then Locket's own section, then\n"
               "the file's top level (--global), then the shipped default. `locket help configure`\n"
               "explains the file and what a model change does to each store's index.")
    sub = p.add_subparsers(dest="section", metavar="<section>")
    e = sub.add_parser(
        "embedder", **fmt, help="the model and server Locket ranks by meaning with",
        description=_wrap("Change the embedder every Locket process uses. Each option writes one key of "
                    "Locket's own `embed` section, or of the top-level one every piece shares with --global; "
                    "options left out keep their current value. Pass `default` as a value to remove that "
                    "key, so the next source applies again: the global value for Locket's own section, the "
                    "shipped default for the global one."),
        epilog="what a model change costs:\n"
               "  Each store's index is built in one model's space. After a change, `find` and\n"
               "  `index` rebuild a store when they next run it, which takes minutes on a large\n"
               "  store, and the hooks stay silent for a store until it is rebuilt. --index pays\n"
               "  that cost now for every store; without it, run `locket index all` when it suits.\n\n"
               "the in-process rung:\n"
               "  With ollama down, Locket runs EmbeddingGemma 2 in onnxruntime inside the venv.\n"
               "  That model is fixed: --model changes the ollama model only, so a machine set to\n"
               "  another model ranks in a different space when ollama is down, and keeps a\n"
               "  separate index for it.\n\n"
               "examples:\n"
               "  locket configure embedder --model nomic-embed-text\n"
               "  locket configure embedder --model embeddinggemma-2:270m --index\n"
               "  locket configure embedder --ollama-host http://studio.local:11434 --no-autostart\n"
               "  locket configure embedder --venv ~/envs/panoply\n"
               "  locket configure embedder --model default --dry-run")
    e.add_argument("--model", metavar="NAME",
                   help="the ollama embedding model to rank with, as `ollama list` names it "
                        f"(default {_embed.SHIPPED['model']}). It must already be pulled: the change is "
                        "refused when ollama answers without it, or when the model is not an embedding "
                        "model. `default` removes the setting, so a model literally named `default` "
                        "cannot be set here. MEMFIND_MODEL overrides it.")
    e.add_argument("--ollama-host", metavar="URL",
                   help=f"where ollama answers, with its scheme (default {_embed.SHIPPED['ollama_host']}). "
                        "Name another machine to rank on its GPU; Locket starts ollama itself only on "
                        "this machine. `default` removes the setting. OLLAMA_HOST overrides it.")
    auto = e.add_mutually_exclusive_group()
    auto.add_argument("--autostart", dest="autostart", nargs="?", const=True, default=None, choices=["default"],
                      help="start `ollama serve` when the port is closed and the binary is on PATH "
                           "(the default); `--autostart default` removes the setting. Hooks never start "
                           "it, whatever this says.")
    auto.add_argument("--no-autostart", dest="autostart", action="store_false", default=None,
                      help="never start ollama; with it down, rank in-process or by word overlap. "
                           "MEMFIND_NO_AUTOSTART=1 does the same for one shell.")
    e.add_argument("--venv", metavar="PATH",
                   help="the virtualenv holding onnxruntime (1.23 or later) and tokenizers for the "
                        "in-process rung; the model downloads into it on first use, about 314 MB. "
                        "Default ~/.panoply/venv, shared by every Panoply piece. `default` removes the "
                        "setting. PANOPLY_VENV, then LOCKET_VENV, override it.")
    e.add_argument("--global", dest="shared", action="store_true",
                   help="write the top-level settings every Panoply piece shares, instead of Locket's "
                        "own; a piece that sets the same key itself keeps its own value, and the "
                        "command names it")
    e.add_argument("--reset", action="store_true",
                   help="remove every embedder setting from the section being written (Locket's own, "
                        "or the global one with --global) before applying any others given")
    e.add_argument("--index", action="store_true",
                   help="after writing, rebuild every store's index with the new settings (minutes "
                        "per large store). Without it the change is instant and stores rebuild when "
                        "`find` or `locket index` next runs them.")
    e.add_argument("--no-check", action="store_true",
                   help="write --model without asking ollama whether it holds that model, for a "
                        "server that is down now or a model you will pull later")
    e.add_argument("--dry-run", action="store_true",
                   help="print the file that would be written and change nothing; --index is skipped")
    sub.add_parser("schema", help="print the settings file's JSON Schema, for an editor's hints",
                   description="Print the JSON Schema ~/.panoply/config.json follows. Save it beside the "
                               "file and name it in the file's `$schema` key for editor completion.")
    return p


def main(argv):
    args = _parser().parse_args(argv)
    if args.section == "schema":
        print(json.dumps(_settings.schema(), indent=2))
        return 0
    if not getattr(args, "dry_run", False):
        try:
            moved = migrate()
        except OSError as e:
            moved = f"warning: could not move {LEGACY} into {_settings.path()} ({e}); Locket still reads it"
        if moved:
            print(moved, file=sys.stderr)
    if args.section == "embedder":
        return _set_embed(args)
    return show()


def selftest():
    """Offline, against a throwaway settings file: write, read back, precedence across the
    piece, global and legacy layers, clearing, a refused bad file, and the migration."""
    global LEGACY
    names = [v for _, env in _embed.CONFIG_KEYS.values() for v in env] + ["PANOPLY_CONFIG"]
    saved_legacy, saved_env = LEGACY, {v: os.environ.pop(v, None) for v in names}
    try:
        with tempfile.TemporaryDirectory() as t, _embed.settings():
            os.environ["PANOPLY_CONFIG"] = str(Path(t) / "panoply" / "config.json")
            LEGACY = Path(t) / "locket" / "config.json"
            cfg, load = _settings.path(), lambda: _settings.load()
            assert load() == ({}, []) and apply()["model"] == (_embed.SHIPPED["model"], "default")
            run = lambda *a: _set_embed(_parser().parse_args(["embedder", *a]))
            quiet = open(os.devnull, "w")
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                assert run("--model", "m1", "--no-check", "--no-autostart") == 0
                assert load()[0] == {"locket": {"embed": {"model": "m1", "autostart": False}}}, load()
                assert apply()["model"] == ("m1", "locket") and _embed.MODEL == "m1"
                assert run("--model", "m-all", "--ollama-host", "http://g:1", "--no-check", "--global") == 0
                got = apply()
                assert got["model"] == ("m1", "locket") and got["ollama_host"] == ("http://g:1", "global"), got
                os.environ["MEMFIND_MODEL"] = "m-env"
                assert apply()["model"] == ("m-env", "env MEMFIND_MODEL"), "the environment must beat the file"
                del os.environ["MEMFIND_MODEL"]
                assert run("--model", "default") == 0 and apply()["model"] == ("m-all", "global"), load()
                assert run("--reset") == 0 and load()[0] == {"embed": {"model": "m-all", "ollama_host": "http://g:1"}}
                assert run("--reset", "--global") == 0 and load()[0] == {}, load()
                assert run("--ollama-host", "studio:11434") == 2, "a host with no scheme was written"
                assert run() == 2, "a run naming nothing to change wrote"
                before = cfg.read_text()
                assert run("--model", "m2", "--no-check", "--dry-run") == 0 and cfg.read_text() == before
                cfg.write_text('{"embed": {"model": 3, "colour": "red"}, "grille": {"embd": {}, "embed": {"model": "g"}}}')
                assert len(load()[1]) == 3 and apply()["model"][1] == "default", f"a bad file took effect: {load()}"
                assert run("--model", "m3", "--no-check") == 0, "a structural finding blocked a write"
                assert load()[0]["locket"] == {"embed": {"model": "m3"}} and load()[0]["grille"]["embed"] == {"model": "g"}
                assert run("--no-autostart") == 0 and run("--autostart", "default") == 0
                assert load()[0]["locket"] == {"embed": {"model": "m3"}}, "--autostart default did not clear it"
                assert run("--autostart") == 0 and load()[0]["locket"]["embed"]["autostart"] is True
                cfg.write_text('{"locket": {"embed": {"model": "m4"}}}')
                assert run("--venv", "rel/v") == 0
                assert load()[0]["locket"]["embed"]["venv"] == os.path.abspath("rel/v"), "a relative --venv was stored as given"
                cfg.write_text("{not json")
                assert load()[0] == {} and _settings.unreadable(load()[1])
                assert run("--model", "m5", "--no-check") == 1, "a write over an unreadable file went ahead"
                assert run("--reset", "--model", "m5", "--no-check") == 0
                assert cfg.with_name("config.json.bak").read_text() == "{not json", "--reset kept no copy"
                # Locket 0.10's file: read below the top level, then moved into it
                LEGACY.parent.mkdir()
                LEGACY.write_text('{"embed": {"model": "m-old", "ollama_host": "http://old:1", "venv": "", "colour": "x"}}')
                cfg.write_text("[]")
                assert "left in place" in migrate() and LEGACY.exists(), "migrated over an unreadable file"
                cfg.write_text('{"embed": {"model": "m-top"}}')
                got = apply()
                assert got["model"] == ("m-top", "global") and got["ollama_host"][1] == "~/.locket/config.json", got
                assert migrate() and not LEGACY.exists() and LEGACY.with_name("config.json.migrated").exists()
                assert load() == ({"embed": {"model": "m-top", "ollama_host": "http://old:1"}}, []), load()
                assert migrate() is None, "a second migration ran"
                LEGACY.write_text('{"embed": {"model": "m-again"}}')
                assert "migrated-" in migrate() and LEGACY.with_name("config.json.migrated").exists(), \
                    "a second migration overwrote the first one's copy"
    finally:
        LEGACY = saved_legacy
        for v, val in saved_env.items():
            if val is not None:
                os.environ[v] = val
        apply()
    print("configure selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["selftest"] else main(sys.argv[1:]))
