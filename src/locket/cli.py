"""The command line: `locket -h`, each verb's `-h`, `locket help <topic>`, and dispatch,
all built from the verbs each family in `commands/` declares. Dispatch hands every
argument after the verb to the verb's `run` untouched; the parser is built only to
print help, so a hook verb dispatched on every tool call never pays for it.
"""
import sys
from functools import lru_cache
from pathlib import Path

import commands
import fsio
import help_text
from commands import Verb, hooks, hygiene, machine, meaning, overlap, stores

# The order `locket -h` lists them in. A new family is one module in `commands/` and one
# entry here; a new verb is one `Verb` in its family's VERBS.
FAMILIES = (meaning, overlap, hygiene, hooks, stores, machine)
SCRIPTS = {"find": "memfind", "scan": "memscan"}      # a family's SEE, and the script it names


def _help(rest):
    which = rest[0] if rest else ""
    which = {"memfind": "find", "memscan": "scan"}.get(which, which)
    v = _BY_NAME.get(which)
    if v is not None and not v.hidden:
        own = _subparser(v.name).print_help
        if v.guide is not None:
            return v.guide(own) or 0
        own()
        return 0
    if which in TOPICS:
        return TOPICS[which]()
    parser()[0].print_help()
    return 0


def _scan_help():
    print(help_text.usage())
    return 0


def _manifest_help():
    print(help_text.manifest_help(), end="")
    return 0


# `locket help <topic>` for a topic that is no verb's name.
TOPICS = {"scan": _scan_help, "manifest": _manifest_help}

HELP = Verb("help", "find | scan | manifest | trigger | usage | install",
            "the full help of either script, the manifest keys alone, how trigger rows or usage work, "
            "or the agent's install steps",
            "locket help find                      how to read a ranking\n"
            "locket help scan                      every check, every manifest key, the ledger schemas\n"
            "locket help manifest                  the manifest keys alone, grouped by the question each answers\n"
            "locket help install                   the install steps written for an agent",
            run=_help)

VERBS = tuple(v for f in FAMILIES for v in f.VERBS) + (HELP,)
_BY_NAME = {n: v for v in VERBS for n in (v.name, *v.aliases)}
_SEE = {v.name: getattr(f, "SEE", "") for f in FAMILIES for v in f.VERBS}


@lru_cache(maxsize=None)
def parser():
    """(the top-level parser, its subparsers action), every listed verb a subparser."""
    import argparse
    fmt = dict(formatter_class=argparse.RawDescriptionHelpFormatter)
    p = argparse.ArgumentParser(
        prog="locket", epilog=help_text.EPILOG, **fmt,
        description="locket: which memory file already holds this idea, and where a corpus repeats itself.")
    sub = p.add_subparsers(title="commands", metavar="<command>")
    for v in VERBS:
        if v.hidden:
            continue
        see = _SEE.get(v.name, "")
        tail = f"\n`locket help {see}` has {SCRIPTS[see]}.py's full help." if see else ""
        if "corpus" in v.args:
            tail = help_text.CORPUS_NOTE + tail
        shown = "\n".join("  " + ln for ln in v.examples.split("\n")) if v.examples else ""
        c = sub.add_parser(v.name, help=v.help, description=v.help, **fmt,
                           usage=f"locket {v.name} {v.args}".rstrip(),
                           epilog=(f"examples:\n{shown}\n" if shown else "") + tail)
        c.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return p, sub


def _subparser(name):
    return parser()[1].choices[name]


def wants_help(name, rest):
    """A help request anywhere after a listed verb, trailing included, so none reaches a
    verb as a corpus or path: `-h` or `--help`, or a bare `help` unless something named
    `help` exists here to be the directory or file it names. A quoted first argument
    (find's statement) is free text and may be any word. Cheap on purpose: the hook verbs
    dispatch through here on every tool call and never build the parser."""
    v = _BY_NAME.get(name)
    if v is None or v.hidden:
        return False
    if {"-h", "--help"} & set(rest):
        return True
    return "help" in (rest[1:] if v.args.startswith('"') else rest) and not Path("help").exists()


def dispatch(name, rest):
    """Run one verb by any of its spellings: its name, an older alias, or a dashed form
    (`--index`, `--selftest`, `--version`, `-V`, `-v`)."""
    if name in ("-V", "-v"):
        name = "version"
    v = _BY_NAME.get(name.lstrip("-") if name.startswith("-") else name)
    if v is None:
        parser()[0].print_usage(sys.stderr)
        print(f"locket: unknown command {name.lstrip('-')!r}; `locket -h` lists them", file=sys.stderr)
        return 1
    if v is not HELP and not v.own_help and wants_help(v.name, rest):
        _subparser(v.name).print_help()
        return 0
    return v.run(rest)


def main(argv, version=""):
    commands.VERSION = version
    if not isinstance(sys.stdout, fsio.SafeOut):
        sys.stdout = fsio.SafeOut(sys.stdout)       # nothing a store holds reaches a terminal raw
    try:
        if len(argv) < 2 or argv[1] in ("-h", "--help"):
            parser()[0].print_help()
            return 0 if len(argv) > 1 else 1
        return dispatch(argv[1], argv[2:])
    except ModuleNotFoundError as e:
        if e.name != "numpy":
            raise
        print("locket: numpy is missing, and ranking, configure, doctor and uninstall need it. "
              "Reinstall (brew reinstall locket) or run: python3 -m pip install numpy", file=sys.stderr)
        return 1
