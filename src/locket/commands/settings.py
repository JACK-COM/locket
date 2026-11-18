"""`locket configure`: shows and changes the embedder settings `configure` resolves, with
the checks a change needs first: the model is served, the host is this machine or a
person said otherwise, and which stores' indexes the change leaves behind.
"""
import argparse
import json
import os
import sys
import textwrap

from commands import Verb


def _ollama_check(model):
    """(ok, message) for whether ollama serves `model` as an embedding model. Unreachable
    ollama is ok with a warning: the file may be written ahead of the server."""
    import _embed
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
    import _embed
    import corpora
    import corpus
    import memfind
    now = {_embed.tag("ollama"), _embed.tag("onnx")}
    out = []
    for name, root in sorted(corpora.all_corpora().items()):
        if not corpus.md_files(root):
            continue
        meta = memfind.cache_paths(root)[0]
        try:
            model = memfind.read_meta(meta).get("model")
        except (OSError, ValueError, AttributeError):   # unreadable, not JSON, or not an object
            model = None
        model = model if isinstance(model, str) else None
        out.append((name, root, "absent" if model is None else "current" if model in now else f"built with {model}"))
    return out


def show():
    import _embed
    import _settings
    import configure
    data, findings = _settings.load()
    path = _settings.path()
    print(f"settings file: {path}" + ("" if path.exists() else "  (absent: every setting is its default)"))
    for f in findings:
        print(f"  problem: {f}")
    print("\nembedder, as Locket uses it   (source: env, locket's own section, global, or default)")
    resolved = configure.apply()
    in_file = {k for _, layer in configure.layers(data) for k in layer}
    for key, (value, src) in resolved.items():
        shown = ("on" if value else "off") if key == "autostart" else value
        note = "  <- overrides the file" if src.startswith("env") and key in in_file else ""
        print(f"  {key:12s} {str(shown):40s} {src}{note}")
    print(f"  {'in-process':12s} {_embed.ONNX_MODEL}@{_embed.ONNX_REVISION[:12]}  (fixed; answers when ollama "
          f"does not{', a different model from the one above' if resolved['model'][0] != _embed.SHIPPED['model'] else ''})")
    others = {p: sorted(data[p]["embed"]) for p in _settings.PIECES if p != configure.PIECE and isinstance(data.get(p), dict)
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


def _loopback(url):
    """Whether `url` names this machine, so embedding through it sends nothing away."""
    import ipaddress
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _set_embed(args):
    import _embed
    import _settings
    import configure
    keys = ("model", "ollama_host", "venv", "autostart")
    clearing = {k for k in keys if getattr(args, k) == "default"}
    changes = {k: getattr(args, k) for k in keys if getattr(args, k) not in (None, "default")}
    if not (changes or clearing or args.reset or args.index):
        print("nothing to change: name a setting (see `locket configure embedder -h`)", file=sys.stderr)
        return 2
    empty = [k for k, v in changes.items() if v == ""]
    if empty:
        print(f"--{empty[0].replace('_', '-')} is empty; pass `default` to remove the setting", file=sys.stderr)
        return 2
    if "venv" in changes:                   # absolute: a hook resolves a relative path against its own cwd
        venv = os.path.expanduser(changes["venv"])
        if venv.startswith("~"):
            print(f"--venv {changes['venv']}: no such user's home to expand", file=sys.stderr)
            return 2
        changes["venv"] = os.path.abspath(venv)
    if "ollama_host" in changes and not changes["ollama_host"].startswith(("http://", "https://")):
        print(f"--ollama-host needs a scheme, e.g. http://{changes['ollama_host']}", file=sys.stderr)
        return 2
    if "ollama_host" in changes and not args.dry_run and not _loopback(changes["ollama_host"]):
        # every text Locket embeds goes to this host, memory included: a person decides that
        import trust
        if not trust.person_confirms(
                f"Send every text Locket embeds, your memory files included, to {changes['ollama_host']}?",
                ["Only a machine you control should receive it; an agent cannot set this for you."],
                "send it there"):
            print(f"not written: an ollama host off this machine is set by a person at a terminal. "
                  f"Run it yourself:  locket configure embedder --ollama-host {changes['ollama_host']}",
                  file=sys.stderr)
            return 1
    if "model" in changes and not args.no_check:
        # ask ollama before taking the lock, against the host the new settings would use
        new = _settings.edit(_settings.load(check=False)[0], "embed", changes, clearing, args.reset,
                             None if args.shared else configure.PIECE)
        view = [("global", new.get("embed"))] if args.shared else configure.layers(new)
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
            for p in _settings.overriding(data, "embed", k, usable=_embed._usable):
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
    import _settings
    import configure
    if _settings.unreadable(findings) and not args.reset:
        print(f"{findings[0]}\nfix it, or pass --reset to start the file over (the old one is kept "
              "as config.json.bak)", file=sys.stderr)
        outcome["rc"] = 1
        return None
    new = _settings.edit(data, "embed", changes, clearing, args.reset, None if args.shared else configure.PIECE)
    if args.dry_run:
        print(json.dumps(new, indent=2))
        outcome["rc"] = 0
        return None
    return new


def _wrap(text):
    return textwrap.fill(text, 78)


def parser():
    import _embed
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
    import _settings
    import configure
    args = parser().parse_args(argv)
    if args.section == "schema":
        print(json.dumps(_settings.schema(), indent=2))
        return 0
    if not getattr(args, "dry_run", False):
        try:
            moved = configure.migrate()
        except OSError as e:
            moved = f"warning: could not move {configure.legacy_path()} into {_settings.path()} ({e}); Locket still reads it"
        if moved:
            print(moved, file=sys.stderr)
    if args.section == "embedder":
        return _set_embed(args)
    return show()


def _guide(own):
    import configure
    parser().print_help()
    print("\n" + configure.__doc__.split("\n\n", 1)[1].rstrip())
    return 0


VERBS = (
    Verb("configure", "[embedder [--model NAME] [--ollama-host URL] [--[no-]autostart] [--venv PATH] [--global] [--reset] [--index] [--no-check] [--dry-run] | schema]",
         "show or change the embedder settings every Locket process reads, Locket's own or (--global) every Panoply piece's, in ~/.panoply/config.json: the model, ollama host, autostart and venv",
         "locket configure                      each setting, where its value comes from, stores behind\n"
         "locket configure embedder --model nomic-embed-text --index\n"
         "                                      switch model and rebuild every index now\n"
         "locket configure embedder -h          every option, explained\n"
         "locket help configure                 the file, precedence, and what a model change costs",
         run=main, guide=_guide, own_help=True),
)
