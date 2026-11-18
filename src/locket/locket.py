#!/usr/bin/env python3
"""locket: the one command for Locket.

Locket is two scripts that sit beside this one: `memscan.py` (structure checks
over a memory corpus, the write-time hooks) and `memfind.py` (which file already
holds an idea, by meaning). Every mode of both is reached from here, so a user
never has to remember which script owns which mode, and the install and removal
of the command itself are verbs here too. The hooks keep calling the two scripts
by absolute path, which is why this file adds nothing they depend on.
"""

import shlex
import shutil
import sys

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memscan  # noqa: E402
import paths  # noqa: E402
import doctor  # noqa: E402
import install  # noqa: E402
import probes  # noqa: E402

__version__ = "0.12.3"                          # the one home: pyproject.toml and the formula's test read it


# memfind verbs, in memfind's own spelling; `find` is the bare statement form.
FIND = {"find": None, "siblings": "--siblings", "index": "--index", "embedder": "--embedder"}
SCAN = ("audit", "across", "dupes", "graduated", "ledgers", "budget", "pointers", "links", "cites",
        "corpora", "init", "trust", "untrust", "schema", "migrate", "forget", "against")

# The verb table: (name, argument metavar, one-line help, examples). argparse renders
# `locket -h` and `locket <verb> -h` from it; dispatch below stays by hand, so
# every flag after the verb reaches memscan or memfind untouched.
VERBS = [
    ("find", '"<statement>" [corpus] [-n N]', "files most likely to own the statement, by meaning",
     'locket find "a fact to write"          the 5 closest files in this directory\'s store\n'
     'locket find "a fact" council -n 10     the 10 closest in the store `council`\n'
     'locket find "the same fact" all        every store, one list per store'),
    ("siblings", "[corpus] [-n N]", "file pairs sharing one subject: candidates to merge",
     "locket siblings                       the 30 closest pairs in this directory's store\n"
     "locket siblings notes -n 10           the 10 closest in `notes`"),
    ("index", "[corpus]", "build or refresh the semantic cache", "locket index council"),
    ("embedder", "", "which embedder answers on this machine", None),
    ("audit", "[corpus [floor]]", "claim pairs across files that agree: fork risk",
     "locket audit                          this directory's store\n"
     "locket audit all                      every store, one after another\n"
     "locket audit notes 0.7                only pairs at least 0.7 alike (default 0.55, 1.0 is identical)"),
    ("across", "[floor]", "claim pairs split across two stores; always reads every store",
     "locket across                         pairs at least 0.55 alike\n"
     "locket across 0.8                     only near-identical pairs"),
    ("dupes", "[corpus] [file ...]", "claims repeated inside one file; use on scratch and inbox files",
     "locket dupes                          the scratch and inbox files the store names\n"
     "locket dupes council SHORT_TERM.md    one file, named as the store lists it"),
    ("graduated", "[corpus [floor]]", "scratch-file entries a permanent file already states: ready to clear",
     "locket graduated                      this directory's store\n"
     "locket graduated notes 0.7            only entries at least 0.7 alike (default 0.55)"),
    ("ledgers", "[corpus]", "schema-governed CSV rows off their schema, and declared CSVs past 5 MB",
     "locket ledgers notes"),
    ("budget", "[corpus]", "files past a byte, history-clause or paragraph budget", "locket budget council"),
    ("pointers", "[corpus]", "index lines beside the files they point at, missing targets first; "
     "needs a reader", "locket pointers council"),
    ("links", "[corpus]", "wikilink targets resolving to no file", "locket links council"),
    ("cites", "[corpus] [path ...]", "path:line citations resolved against the code",
     "locket cites myapp                    every citation in `myapp`\n"
     "locket cites myapp models.py          only citations of files whose path contains models.py"),
    ("against", "[corpus] < text", "rank files by how much of the text on stdin they already state, "
     "word by word; what the write hook runs",
     'echo "a sentence" | locket against council\n'
     "locket against notes < draft.md       a whole draft before it is written"),
    ("trigger", "[check [corpus] | schema]",
     "the tool-boundary hook: when a tool call or prompt matches a row in a store's triggers.json, "
     "put the row's question to the agent; `check` lints the rows files, `schema` prints their schema",
     "locket trigger check                  the rows that can fire from here\n"
     "locket trigger check council          one store's rows\n"
     "locket trigger schema                 the JSON Schema a triggers.json row follows\n"
     "locket help trigger                   how rows are written and when they fire"),
    ("usage", "[today | week | month | record] [--by project|model|day] [--json]",
     "tokens Claude Code and Hermes spent, per source and project, against a typical day; "
     "keeps a ledger that outlives the transcripts",
     "locket usage                          the last 7 days, per project\n"
     "locket usage today                    today; `month` is the last 30 days\n"
     "locket usage month --by model         30 days, per model; `--by day` is one row a day\n"
     "locket usage --json                   the same summary as JSON\n"
     "locket usage record                   copy new sessions into the ledger now (the\n"
     "                                      SessionEnd hook does this on its own)"),
    ("configure", "[embedder [--model NAME] [--ollama-host URL] [--[no-]autostart] [--venv PATH] [--global] [--reset] [--index] [--no-check] [--dry-run] | schema]",
     "show or change the embedder settings every Locket process reads, Locket's own or (--global) every Panoply piece's, in ~/.panoply/config.json: the model, ollama host, autostart and venv",
     "locket configure                      each setting, where its value comes from, stores behind\n"
     "locket configure embedder --model nomic-embed-text --index\n"
     "                                      switch model and rebuild every index now\n"
     "locket configure embedder -h          every option, explained\n"
     "locket help configure                 the file, precedence, and what a model change costs"),
    ("corpora", "", "every store on this machine", None),
    ("init", "<dir> [--name <label>] [--parent <corpus>] [--ledgers[=<Name>]]",
     "make a folder of markdown a store; running it again changes nothing",
     "locket init ~/notes                   the folder becomes a store, named after it\n"
     "locket init ~/notes --name work       names the store `work`; works on a Claude Code project store too\n"
     "locket init ~/docs --parent notes     ~/docs joins the store `notes`: two folders, one store\n"
     "locket init ~/notes --ledgers         adds RULINGS-notes.csv (decisions), CLAIMS-notes.csv\n"
     "                                      (facts with sources) and HISTORY-notes-Sessions.csv (one\n"
     "                                      row per session), header only; rows off their columns\n"
     "                                      are refused as they are written\n"
     "locket init ~/notes --ledgers=Work    the same files, named RULINGS-Work.csv and so on"),
    ("migrate", "[<dir>|all]", "finish an upgrade: move older settings files into their current home and rename "
     "each store's older memfind.json manifest to locket.json; safe to run any time",
     "locket migrate                        everything this machine needs after an upgrade\n"
     "locket migrate ~/notes                one store's manifest only"),
    ("schema", "[<dir>]", "write the manifest schema, or check a manifest",
     "locket schema                         write ~/.locket/locket.schema.json for editor hints\n"
     "locket schema ~/notes                 check that store's locket.json against it"),
    ("trust", "<dir>", "let a store's locket.json act (trigger rows, ledger schemas, sources); "
     "asks you to type yes at a terminal, so an agent cannot grant it",
     "locket trust ~/Software/myapp"),
    ("untrust", "<dir>", "withdraw a store's trust; its locket.json stays, inert",
     "locket untrust ~/Software/myapp"),
    ("forget", "<dir>", "stop listing a store and withdraw its trust; its files and locket.json stay",
     "locket forget ~/old-notes"),
    ("install", "[--hooks] [--codex] [--desktop]",
     "put `locket` on PATH (%s); --hooks merges the write-time hooks into Claude Code's "
     "settings.json, --codex into Codex's hooks.json, --desktop registers the MCP server "
     "with Claude Desktop" % install.BIN,
     "locket install                        the command only\n"
     "locket install --hooks                plus the Claude Code hooks, keeping a copy of settings.json\n"
     "locket install --codex                plus the Codex hooks; trust them with /hooks in Codex\n"
     "locket install --desktop              plus the Claude Desktop server; restart the app after"),
    ("uninstall", "[--yes] [--purge] [--dry-run]",
     "remove the command, ~/.locket, every cache, and the shared venv when no other piece uses it; "
     "with --purge every manifest and the usage ledger, confirmed by a person at a terminal; print what is left",
     "locket uninstall --dry-run            print what would go, remove nothing\n"
     "locket uninstall                      remove, after asking once\n"
     "locket uninstall --yes                remove without asking\n"
     "locket uninstall --purge              also every store's locket.json and the usage ledger"),
    ("status", "", "where the command points and what it can reach", None),
    ("doctor", "[--ack]", "check every install step a machine can check, with the fix for each failure",
     "locket doctor                        check, including live hook runs abandoned or failed open\n"
     "locket doctor --ack                  check, then mark the hook-health counts shown as seen"),
    ("mcp", "", "serve the checks over MCP on stdio, for a host without hooks", None),
    ("selftest", "", "check this install end to end on scratch stores", None),
    ("version", "", "print the version (also --version, -V, -v)", None),
    ("help", "find | scan | manifest | trigger | usage | install",
     "the full help of either script, the manifest keys alone, how trigger rows or usage work, "
     "or the agent's install steps",
     "locket help find                      how to read a ranking\n"
     "locket help scan                      every check, every manifest key, the ledger schemas\n"
     "locket help manifest                  the manifest keys alone, grouped by the question each answers\n"
     "locket help install                   the install steps written for an agent"),
]
VERB_NAMES = frozenset(v[0] for v in VERBS)

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
manifest shapes it, `locket help manifest` prints the manifest keys alone, and
`locket help find` explains how to read a ranking.
Needs nothing but Python 3 for the checks; `find` needs an embedder (ollama with
embeddinggemma-2:270m, or onnxruntime in the venv the Panoply pieces share) and says so when none answers.
"""


CORPUS_NOTE = """
corpus: a store's name as `locket corpora` lists it, or any part of one that
matches a single store; a folder path; or `all`. Omitted, it is the store the
current directory sits in."""


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
        if "corpus" in args:
            tail = CORPUS_NOTE + tail
        shown = "\n".join("  " + ln for ln in examples.split("\n")) if examples else ""
        c = sub.add_parser(name, help=help, description=help, **fmt,
                           usage=f"locket {name} {args}".rstrip(),
                           epilog=(f"examples:\n{shown}\n" if shown else "") + tail)
        c.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return p, sub


# --- install / uninstall ------------------------------------------------------


# --- doctor -------------------------------------------------------------------


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
    if not isinstance(sys.stdout, memscan.SafeOut):
        sys.stdout = memscan.SafeOut(sys.stdout)    # nothing a store holds reaches a terminal raw
    try:
        return _main(argv)
    except ModuleNotFoundError as e:
        if e.name != "numpy":
            raise
        print("locket: numpy is missing, and ranking, configure, doctor and uninstall need it. "
              "Reinstall (brew reinstall locket) or run: python3 -m pip install numpy", file=sys.stderr)
        return 1


def _main(argv):
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
        if which in _parser()[1].choices and which not in ("find", "install", "trigger", "usage", "configure"):
            _parser()[1].choices[which].print_help(); return 0
        if which in ("find", "memfind"):
            return install.load_memfind().main(["memfind.py", "--help"])
        if which in ("scan", "memscan"):
            return memscan.main(["memscan.py", "--help"])
        if which == "manifest":
            print(memscan.manifest_help(), end=""); return 0
        if which == "trigger":
            import trigger
            _parser()[1].choices["trigger"].print_help()
            print("\n" + trigger.__doc__.split("\n\n", 1)[1].rstrip()); return 0
        if which == "configure":
            import configure
            configure._parser().print_help()
            print("\n" + configure.__doc__.split("\n\n", 1)[1].rstrip()); return 0
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
        verb = verb.lstrip("-")               # the dashed spellings: --index, --siblings, --selftest
    if verb == "configure":                 # its own parser: every option carries its own -h text
        import configure
        return configure.main(rest)
    if _wants_help(verb, rest):
        _parser()[1].choices[verb].print_help(); return 0
    if verb == "find":
        if not rest or rest[0].startswith("-"):
            print('find needs a statement:  locket find "<statement>" [corpus] [-n N]', file=sys.stderr)
            return 1
        return install.load_memfind().main(["memfind.py", *rest])
    if verb in FIND:
        return install.load_memfind().main(["memfind.py", FIND[verb], *rest])
    if verb == "selftest":
        broken = probes.smoke()
        print(f"selftest: {broken}" if broken else "selftest ok", file=sys.stderr if broken else sys.stdout)
        return 1 if broken else 0

    if verb == "install":
        return install.cmd_install(rest)
    if verb == "uninstall":
        return install.cmd_uninstall(rest)
    if verb == "status":
        return doctor.cmd_status()
    if verb == "doctor":
        rc = doctor.cmd_doctor(__version__)                   # shown first, so an ack never hides what it marks seen
        if "--ack" in rest:
            print("hook health: " + ("the counts above are marked seen" if memscan.health_ack()
                                     else "could not mark the counts seen (the record was busy); run it again"))
        return rc
    if verb == "mcp":                       # the Desktop server, reached through the command
        import runpy
        sys.argv = [str(HERE / "locket_mcp.py"), *rest]
        runpy.run_path(str(HERE / "locket_mcp.py"), run_name="__main__")
        return 0
    if verb == "migrate" and (not rest or rest[0] == "all"):
        # every pending upgrade step, each one a no-op once done: the machine's settings first,
        # then each store's manifest. A release that needs a new step adds it here.
        import configure
        try:
            moved = configure.migrate()
        except (OSError, TimeoutError) as e:
            moved = f"warning: {configure._legacy_path()} not moved ({e}); Locket still reads it"
        if moved:
            print(moved)
        # 0.12: the index lives in ~/.locket/cache, so a cache inside a store is dead weight;
        # every one is offered for deletion at once, and named in one command otherwise
        stale = [c for c in install.caches() if c.is_dir() and not c.is_symlink()]
        if stale:
            print(f"index: {len(stale)} caches from before 0.12 are no longer read; the index now lives in "
                  f"{paths.PATHS.cache}")
            if memscan.person_confirms(f"Delete these {len(stale)} old caches?", [str(c) for c in stale],
                                       "delete them"):
                for c in stale:
                    shutil.rmtree(c, ignore_errors=True)
                print(f"index: deleted {len(stale)} old caches")
            else:
                print("index: kept them; to delete them all:  rm -r " + " ".join(shlex.quote(str(c)) for c in stale))
        unbuilt = [n for n, r in memscan.all_corpora().items()
                   if memscan.has_content(r, csv=False) and not install.load_memfind().cache_paths(r)[0].is_file()]
        if unbuilt:
            print(f"index: {len(unbuilt)} stores have no index in the new home yet; build them once:  locket index all")
        # 0.12: trust is pinned to a store's configuration; pin the rows that predate it
        pinned = memscan.pin_unpinned()
        if pinned is None:
            print("trust: stores registered before 0.12 are unpinned and act as no store until a person "
                  "runs  locket migrate  at a terminal")
        for d in pinned or []:
            print(f"trust: pinned {d} to its configuration as it stands; a later change to its "
                  f"locket.json or rows file needs  locket trust {shlex.quote(str(d))}  again")
        # 0.12: a manifest acts only once a person trusts it; name the ones that stopped
        try:
            loose = memscan.untrusted_stores()
        except Exception as e:
            loose = []
            print(f"trust: the scan for untrusted stores failed ({type(e).__name__}: {e}); run  locket doctor")
        for d in loose:
            print(f"trust: {d} holds a {memscan.MANIFEST} that is no longer acted on until a person "
                  f"trusts it:  locket trust {shlex.quote(str(d))}")
    if verb in SCAN:
        return memscan.main(["memscan.py", verb, *rest])
    if verb in install.HOOK_VERBS:
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
    install.PACKAGED = True
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:
        sys.exit(0)
