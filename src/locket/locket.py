#!/usr/bin/env python3
"""locket: the one command for Locket.

This file holds the release's version and the entry points. Every verb is declared in
its family under `commands/`, and `cli.py` builds the command line from them. The
hooks call this file (`trigger`, `usage hook`) and `memscan.py` by absolute path, so
both stay where an installed config points.
"""

import sys

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cli as front  # noqa: E402
import install  # noqa: E402

__version__ = "0.13.0"                          # the one home: pyproject.toml and the formula's test read it


def main(argv):
    return front.main(argv, __version__)


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
