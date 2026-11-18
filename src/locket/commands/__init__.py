"""Every verb Locket answers, one module per family, each declaring its verbs in `VERBS`.

A `Verb` is the one declaration both fronts read: `cli` builds `locket -h`, each verb's
`-h` and the dispatch from it, and `locket_mcp` builds its tool list from the verbs that
carry a `Tool`. A new verb is one `Verb` in its family's `VERBS`; a new family is one
module here and one entry in `cli.FAMILIES`. A report verb returns a `Report`, which the
CLI prints and the MCP server returns as text.
"""
import os
import sys
from io import StringIO
from typing import Callable, NamedTuple, Optional

import corpora
import corpus
import fsio

VERSION = ""            # this release's, set by the front before dispatch (locket.py owns it)


class Report(NamedTuple):
    """What a report mode found: the lines to show, and how many findings they hold."""
    lines: list
    count: int = 0

    @property
    def text(self):
        return "\n".join(self.lines)

    def show(self):
        """Print the lines as the CLI shows them; nothing at all for an empty report."""
        if self.lines:
            print(self.text)


class Session:
    """What one MCP server process remembers between calls: whether `find` was asked,
    the 2e question the write check reads with no transcript to read it from."""
    asked = False


class Tool(NamedTuple):
    """A verb's MCP face. `call(args, session)` returns the reply text; `props` are the
    arguments' JSON Schema properties; `name` is the tool's name when it is not
    `locket_<verb>`."""
    description: str
    call: Callable
    props: dict = {}
    name: str = ""


class Verb(NamedTuple):
    """One verb. `run(rest)` takes the arguments after the verb and returns the exit code.
    `aliases` are older names, accepted by the CLI and served by the MCP server alike.
    A `hidden` verb is a hook entry point: dispatched, never listed, its arguments never
    read as a help request. `guide(own_help)` is `locket help <name>` when that says more
    than the verb's `-h`; `own_help` marks a verb that parses its own `-h`. A family's
    `SEE`, where set, names the script whose full help its verbs' `-h` points to."""
    name: str
    args: str = ""
    help: str = ""
    examples: Optional[str] = None
    run: Optional[Callable] = None
    mcp: Optional[Tool] = None
    aliases: tuple = ()
    hidden: bool = False
    guide: Optional[Callable] = None
    own_help: bool = False


# ---------------------------------------------------------------- the CLI's corpus argument

def corpus_verb(fn, header=True):
    """`run` for a verb over one store: `fn(root, rest)` returns its `Report`. The corpus
    argument is a name, a path or `all` (every store holding markdown, one after another);
    omitted, it is the store the current directory sits in, never a fixed one, so the
    verb run in another repo never sweeps this one and reports it as that repo's.
    `header` prints the directory resolved and whether the write hook reaches it, because
    a literal path can name a directory the hook never guards."""
    def run(rest):
        if rest[:1] == ["all"]:
            rc = 0
            for k, v in sorted(corpora.all_corpora().items()):
                if not corpus.md_files(v):          # an empty store has nothing to report
                    continue
                print(f"== {k}")
                rc |= run([str(v), *rest[1:]])
                print()
            return rc
        if rest:
            root = corpus.resolve(rest[0])
        else:
            root = corpora.corpus_here(os.getcwd())
            if root is None:
                print("no corpus for this directory; name one, or `init` it:", file=sys.stderr)
                for k, v in sorted(corpora.all_corpora().items()):
                    print(f"  {k:70s} {len(corpus.md_files(v)):4d} files", file=sys.stderr)
        if root is None:
            return 1
        if header:
            guarded = corpora.corpus_for(root / "_probe.md") is not None
            print(f"corpus: {root}"
                  f"  [write-time hook: {'ACTIVE' if guarded else 'NOT PROTECTED — audit only'}]")
        fn(root, rest[1:]).show()
        return 0
    return run


def floor_arg(rest, default):
    """An optional similarity floor as the first argument after the corpus."""
    return float(rest[0]) if rest else default


# ---------------------------------------------------------------- the MCP server's store argument

STORE_ARG = {"type": "string", "description":
             "A memory store: a shorthand or path from locket_corpora, or 'all'. "
             "Omit it when the machine has one store."}


def store_root(name):
    """A store's path from a name, or the only store on the machine when no name is
    given, else an error naming what the model can call next."""
    if name:
        root = corpus.resolve(str(name))
        if root is None:
            raise ValueError(f"no memory store matches {name!r}; call locket_corpora")
        return root
    found = {k: v for k, v in corpora.all_corpora().items() if corpus.md_files(v)}
    if len(found) == 1:
        return next(iter(found.values()))
    raise ValueError("more than one memory store here; name one: "
                     + ", ".join(sorted(found)) + ", or 'all'")


def as_text(report):
    """A report as the text the CLI prints, through the same SafeOut filter, so store
    text never reaches the model raw."""
    buf = StringIO()
    if report.lines:
        fsio.SafeOut(buf).write(report.text + "\n")
    return buf.getvalue()


def over_stores(name, fn):
    """A report over one store or, for 'all', every store with markdown."""
    if name == "all":
        out = [f"== {k}\n" + as_text(fn(v)) for k, v in sorted(corpora.all_corpora().items())
               if corpus.md_files(v)]
        return "\n".join(out) or "no memory stores with markdown files found"
    return as_text(fn(store_root(name)))


def report_tool(description, fn, **props):
    """A `Tool` running `fn(root, args)` over the store the call names, or every store."""
    return Tool(description, lambda a, s: over_stores(a.get("store"), lambda r: fn(r, a)),
                {"store": STORE_ARG, **props})
