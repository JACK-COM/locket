#!/usr/bin/env python3
"""memscan: the entry point host configs call by path for the write-time hooks
(`memscan.py hook`, `bashguard`, `grepassist`), and any other verb through the CLI. What
follows is the design behind the structure checks.

The unit is the CLAIM, not the file. Document-level similarity cannot see one
clause restated inside four files that are otherwise about different subjects,
and it scores sibling records sharing a whole vocabulary as duplicates.

Modes:

  audit    cross-file verbatim claim pairs. These are copies that currently
           AGREE, so they are fork RISK rather than proof of a fork.
  dupes    repeated claims INSIDE one file, which `audit` skips by construction.
           The uniqueness invariant on a holding space (prime-memory-discipline §2).
  budget   §5 as counting: bytes, dated history clauses (each listed by line,
           the reconsolidation worklist), longest prose paragraph. `audit` is
           blind to a file that only disagrees with the world, so nothing else
           measures this.
  pointers every index line paired to the file it points at, for adjudication
           (`hooks` is its older name). Needs a reader: a hook asserting what its topic file now refutes
           scores LOW on similarity and is invisible to `audit`.
  links    wikilink targets that resolve to no file.
  cites    every `path:line` citation resolved against the code it points at.
           The one claim class nothing else here can see, because the claim is
           ABOUT the code rather than IN it and no other mode opens the file.
  corpora  every corpus on the machine, with its file count.
  init     make a directory a corpus every mode can find: manifest, registry
           row, index. `forget` drops the row again.
  against  candidate owners for an incoming statement. Used by the write-time
           hook.

The corpus argument takes a registered name (`notes`), any substring of a project
slug (`myapp`), or a literal path. With no argument it takes THIS project's corpus.

Standard library only at import, by design: this runs inside a PreToolUse hook on
every memory write, so it pays interpreter startup and nothing else. memfind, and
numpy with it, loads only when a check asks for a ranking by meaning.
"""
import sys

import help_text
from commands import hooks
# The names outside callers import from memscan (~/.claude/hooks, older deploys); every
# other name lives in its own module and is read from there.
from corpora import corpus_for_cwd, corpus_key, roots_of, settings  # noqa: F401  (re-exported)
from gate import HISTORY_CLAUSE, ISO_DATE  # noqa: F401  (re-exported)


def main(argv):
    """`memscan.py <verb>`: the write-time hooks a host config calls by path, then any
    other verb through the CLI's dispatch."""
    if len(argv) < 2 or argv[1] in ("--help", "-h", "help"):
        print(help_text.usage())
        return 0 if len(argv) > 1 else 1
    if argv[1] in hooks.ENTRIES:
        return hooks.ENTRIES[argv[1]].run(argv[2:])
    import cli
    return cli.dispatch(argv[1], argv[2:])


if __name__ == "__main__":
    # memfind imports memscan; run as a script, this module is __main__, and without the
    # alias that import would load and run a second copy with caches of its own
    sys.modules.setdefault("memscan", sys.modules[__name__])
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:      # piping any mode into `head` is normal use
        sys.stderr.close()
        sys.exit(0)
