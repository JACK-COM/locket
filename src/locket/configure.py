#!/usr/bin/env python3
"""configure: Locket's machine settings, one file every Locket process reads.

The file is ~/.locket/config.json (LOCKET_CONFIG names another). Every way Locket runs,
the command line, the Claude Code, Codex and Hermes hooks, the MCP server, reads it, so
one change reaches all of them. It holds one section today, `embed`, the embedder:

    {"embed": {"model": "embeddinggemma-2:270m", "ollama_host": "http://127.0.0.1:11434",
               "autostart": true, "venv": "~/.panoply/venv"}}

Each setting resolves in this order, first found wins:
  1. an environment variable (MEMFIND_MODEL, OLLAMA_HOST, MEMFIND_NO_AUTOSTART=1,
     PANOPLY_VENV or LOCKET_VENV), for one shell or one test;
  2. this file;
  3. the shipped default.
A variable set in one place and not another is how two processes come to disagree, so
`locket configure` and `locket doctor` name every variable that is overriding the file.

An index is built in one embedder's space and tagged with it. After a model change every
store's index belongs to the old model: `find` and `index` rebuild a store when run, and
the hooks stay silent for that store until it is rebuilt, because a hook never rebuilds.
`--index` rebuilds every store as part of the change; without it, run `locket index all`
when convenient. `locket configure` lists which stores are behind.

Write the file with `locket configure embedder`, never by hand while another configure
runs: the command holds a lock, writes atomically and checks the result against the
schema (`locket configure schema` prints it).
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
import _manifest_schema  # noqa: E402

CONFIG = Path(os.environ.get("LOCKET_CONFIG") or Path.home() / ".locket" / "config.json").expanduser()

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "locket.config.schema.json",
    "title": "Locket machine settings (~/.locket/config.json)",
    "type": "object", "additionalProperties": False,
    "properties": {
        "$schema": {"type": "string", "description": "Editor hint only; ignored by Locket."},
        "embed": {
            "type": "object", "additionalProperties": False,
            "description": "The embedder `find`, `siblings`, `index` and the hooks rank with.",
            "properties": {
                "model": {"type": "string", "description": "The ollama embedding model. Default: "
                          + _embed.SHIPPED["model"] + ". Overridden by MEMFIND_MODEL."},
                "ollama_host": {"type": "string", "description": "Where ollama answers. Default: "
                                + _embed.SHIPPED["ollama_host"] + ". Overridden by OLLAMA_HOST."},
                "autostart": {"type": "boolean", "description": "Start `ollama serve` when the port is "
                              "closed. Default: true. MEMFIND_NO_AUTOSTART=1 turns it off."},
                "venv": {"type": "string", "description": "The virtualenv holding onnxruntime and tokenizers "
                         "for the in-process rung, shared by the Panoply pieces. Default: ~/.panoply/venv, or "
                         "an older ~/.locket/venv. Overridden by PANOPLY_VENV, then LOCKET_VENV."},
            },
        },
    },
}


def load(path=None):
    """(settings, findings). A missing file is empty settings and no finding; unreadable
    JSON is empty settings and one finding, so a broken file never stops a hook."""
    p = Path(path) if path else CONFIG
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, []
    except (OSError, ValueError) as e:
        return {}, [f"{p}: unreadable ({e})"]
    return (data if isinstance(data, dict) else {}), _manifest_schema.check_value(data, SCHEMA, str(p))


def apply(path=None):
    """Put the file's embed section into the embedder, under the environment. Every module
    that ranks calls this once at import; it is cheap and changes nothing without a file."""
    try:
        return _embed.apply_config(load(path)[0].get("embed"))
    except Exception:                       # nothing in a settings file may stop a hook
        return _embed.apply_config(None)


def _write_json(path, obj):
    """Whole or not at all: a temporary file beside the target, synced, renamed over it.
    A symlinked config keeps its link (the file it points at is replaced) and its mode."""
    target = Path(os.path.realpath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        mask = os.umask(0)
        os.umask(mask)
        mode = 0o666 & ~mask
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


LOCK_WAIT = 10


@contextlib.contextmanager
def _lock():
    """Held across read-modify-write, so two configure runs at once cannot each read the old
    file and the second drop the first's change. A holder past LOCK_WAIT is stuck."""
    import fcntl
    lock = CONFIG.with_name(CONFIG.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as f:
        deadline = time.monotonic() + LOCK_WAIT
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise SystemExit(f"{lock} is held by another `locket configure` for over {LOCK_WAIT}s; "
                                     "if none is running, remove it")
                time.sleep(0.05)
        yield


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
    data, findings = load()
    print(f"config file: {CONFIG}" + ("" if CONFIG.exists() else "  (absent: every setting is its default)"))
    for f in findings:
        print(f"  problem: {f}")
    print("\nembedder")
    resolved = apply()
    for key, (value, src) in resolved.items():
        shown = ("on" if value else "off") if key == "autostart" else value
        note = "  <- overrides the file" if src.startswith("env") and key in (data.get("embed") or {}) else ""
        print(f"  {key:12s} {str(shown):40s} {src}{note}")
    print(f"  {'in-process':12s} {_embed.ONNX_MODEL}@{_embed.ONNX_REVISION[:12]}  (fixed; answers when ollama "
          f"does not{', a different model from the one above' if resolved['model'][0] != _embed.SHIPPED['model'] else ''})")
    state = _index_state()
    behind = [s for s in state if s[2] != "current"]
    print(f"\nindexes: {len(state)} store(s), {len(state) - len(behind)} current")
    for name, root, st in behind:
        print(f"  {name:20s} {st}")
    if behind:
        print("  rebuild with:  locket index all   (hooks stay silent for these until then)")
    return 1 if findings else 0


def _set_embed(args):
    clearing = {k for k in ("model", "ollama_host", "venv") if getattr(args, k) == "default"}
    changes = {k: getattr(args, k) for k in ("model", "ollama_host", "venv")
               if getattr(args, k) not in (None, "default")}
    if args.autostart is not None:
        changes["autostart"] = args.autostart
    if not (changes or clearing or args.reset or args.index):
        print("nothing to change: name a setting (see `locket configure embedder -h`)", file=sys.stderr)
        return 2
    if "venv" in changes:                   # absolute: a hook resolves a relative path against its own cwd
        changes["venv"] = os.path.abspath(Path(changes["venv"]).expanduser())
    if "ollama_host" in changes and not changes["ollama_host"].startswith(("http://", "https://")):
        print(f"--ollama-host needs a scheme, e.g. http://{changes['ollama_host']}", file=sys.stderr)
        return 2
    try:
        with (contextlib.nullcontext() if args.dry_run else _lock()):
            rc = _write_embed(args, changes, clearing)
    except OSError as e:
        print(f"cannot write {CONFIG}: {e}", file=sys.stderr)
        return 1
    if rc is not None:
        return rc
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


def _write_embed(args, changes, clearing):
    """The read-modify-write, under the lock unless a dry run. None means written; a
    number is the exit code to stop with."""
    data, findings = load()
    if findings and CONFIG.exists() and not args.reset:
        print("the config file has problems; fix them or start over with --reset:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        return 1
    if args.reset:                      # start over: only an editor's $schema hint survives
        data = {k: v for k, v in data.items() if k == "$schema"}
    embed = {} if args.reset else dict(data.get("embed") or {})
    for k in clearing:
        embed.pop(k, None)
    embed.update(changes)
    new = {**data, "embed": embed} if embed else {k: v for k, v in data.items() if k != "embed"}
    if "model" in changes and not args.no_check:
        # check against the host this change would use, then restore this process's settings
        with _embed.settings(**{n: getattr(_embed, n) for n in ("MODEL", "OLLAMA", "AUTOSTART", "VENV")}):
            _embed.apply_config(embed)
            ok, msg = _ollama_check(changes["model"])
        if msg:
            print(msg, file=sys.stderr)
        if not ok:
            return 1
    if args.dry_run:
        print(json.dumps(new, indent=2))
        return 0
    if new != load()[0] or not CONFIG.exists():
        _write_json(CONFIG, new)
        print(f"wrote {CONFIG}")
    else:
        print("no change to write")
    return None


def _wrap(text):
    return textwrap.fill(text, 78)


def _parser():
    fmt = dict(formatter_class=argparse.RawDescriptionHelpFormatter)
    p = argparse.ArgumentParser(
        prog="locket configure", **fmt,
        description=_wrap("Show or change Locket's machine settings, held in one file every Locket process "
                    "reads (the command line, every hook, the MCP server). Run with no arguments to see "
                    "each setting, where its value comes from, and which stores' indexes are behind."),
        epilog="examples:\n"
               "  locket configure                                   show settings, their sources and index state\n"
               "  locket configure embedder --model nomic-embed-text switch model; indexes rebuild later\n"
               "  locket configure embedder --model nomic-embed-text --index\n"
               "                                                     switch model and rebuild every index now\n"
               "  locket configure embedder --model default          back to the shipped model\n"
               "  locket configure embedder --reset --index          every embedder setting back to its default\n"
               "  locket configure schema                            the file's JSON Schema, for an editor\n\n"
               "Precedence, first found wins: environment variable, then the file, then the shipped\n"
               "default. `locket help configure` explains the file and what a model change does to\n"
               "each store's index.")
    sub = p.add_subparsers(dest="section", metavar="<section>")
    e = sub.add_parser(
        "embedder", **fmt, help="the model and server Locket ranks by meaning with",
        description=_wrap("Change the embedder every Locket process uses. Each option writes one key of the "
                    "file's `embed` section; options left out keep their current value. Pass `default` "
                    "as a value to remove that key, so the shipped default applies again."),
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
    auto.add_argument("--autostart", dest="autostart", action="store_true", default=None,
                      help="start `ollama serve` when the port is closed and the binary is on PATH "
                           "(the default). Hooks never start it, whatever this says.")
    auto.add_argument("--no-autostart", dest="autostart", action="store_false",
                      help="never start ollama; with it down, rank in-process or by word overlap. "
                           "MEMFIND_NO_AUTOSTART=1 does the same for one shell.")
    e.add_argument("--venv", metavar="PATH",
                   help="the virtualenv holding onnxruntime (1.23 or later) and tokenizers for the "
                        "in-process rung; the model downloads into it on first use, about 314 MB. "
                        "Default ~/.panoply/venv, shared by every Panoply piece. `default` removes the "
                        "setting. PANOPLY_VENV, then LOCKET_VENV, override it.")
    e.add_argument("--reset", action="store_true",
                   help="remove every embedder setting before applying any others given, returning "
                        "the section to the shipped defaults")
    e.add_argument("--index", action="store_true",
                   help="after writing, rebuild every store's index with the new settings (minutes "
                        "per large store). Without it the change is instant and stores rebuild when "
                        "`find` or `locket index` next runs them.")
    e.add_argument("--no-check", action="store_true",
                   help="write --model without asking ollama whether it holds that model, for a "
                        "server that is down now or a model you will pull later")
    e.add_argument("--dry-run", action="store_true",
                   help="print the file that would be written and change nothing; --index is skipped")
    sub.add_parser("schema", help="print the config file's JSON Schema, for an editor's hints",
                   description="Print the JSON Schema the config file follows. Save it beside the file "
                               "and name it in the file's `$schema` key for editor completion.")
    return p


def main(argv):
    args = _parser().parse_args(argv)
    if args.section == "schema":
        print(json.dumps(SCHEMA, indent=2))
        return 0
    if args.section == "embedder":
        return _set_embed(args)
    return show()


def selftest():
    """Offline, against a throwaway config file: write, read back, precedence, clearing,
    a refused bad file, and the lock released."""
    global CONFIG
    saved_cfg, saved_env = CONFIG, {v: os.environ.pop(v, None) for _, env in _embed.CONFIG_KEYS.values() for v in env}
    try:
        with tempfile.TemporaryDirectory() as t, _embed.settings():
            CONFIG = Path(t) / "config.json"
            assert load() == ({}, []) and apply()["model"] == (_embed.SHIPPED["model"], "default")
            run = lambda *a: _set_embed(_parser().parse_args(["embedder", *a]))
            quiet = open(os.devnull, "w")
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                assert run("--model", "m1", "--no-check", "--no-autostart") == 0
                assert load()[0] == {"embed": {"model": "m1", "autostart": False}}, load()
                assert apply()["model"] == ("m1", "file") and _embed.MODEL == "m1"
                os.environ["MEMFIND_MODEL"] = "m-env"
                assert apply()["model"] == ("m-env", "env MEMFIND_MODEL"), "the environment must beat the file"
                del os.environ["MEMFIND_MODEL"]
                assert run("--model", "default") == 0 and load()[0] == {"embed": {"autostart": False}}, load()
                assert run("--reset") == 0 and load()[0] == {}, load()
                assert run("--ollama-host", "studio:11434") == 2, "a host with no scheme was written"
                assert run() == 2, "a run naming nothing to change wrote"
                before = CONFIG.read_text()
                assert run("--model", "m2", "--no-check", "--dry-run") == 0 and CONFIG.read_text() == before
                CONFIG.write_text('{"embed": {"model": 3, "colour": "red"}}')
                found = load()[1]
                assert len(found) == 2 and apply()["model"][1] == "default", f"a bad file took effect: {found}"
                assert run("--model", "m3", "--no-check") == 1, "a write over a broken file went ahead"
                assert run("--reset", "--model", "m3", "--no-check") == 0 and load() == ({"embed": {"model": "m3"}}, [])
                CONFIG.write_text('{"colour": 1, "$schema": "s.json"}')
                assert run("--reset") == 0 and load() == ({"$schema": "s.json"}, []), "--reset kept a bad key"
                assert run("--venv", "rel/v") == 0
                assert load()[0]["embed"]["venv"] == os.path.abspath("rel/v"), "a relative --venv was stored as given"
                CONFIG.write_text("{not json")
                assert load()[0] == {} and "unreadable" in load()[1][0]
            with _lock():
                pass
            with _lock():                   # released, so a second hold does not wait
                pass
    finally:
        CONFIG = saved_cfg
        for v, val in saved_env.items():
            if val is not None:
                os.environ[v] = val
        apply()
    print("configure selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["selftest"] else main(sys.argv[1:]))
