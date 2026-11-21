<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/locket-dark.svg">
  <img src="docs/locket-light.svg" alt="Locket" width="112">
</picture>
</p>

# Locket

*Remember who you are.* One piece of [the Panoply](https://github.com/JACK-COM/homebrew-panoply).

Locket keeps an AI agent's memory from holding the same fact twice. When the agent goes to write a fact its markdown memory already holds, Locket names the file that holds it, so the agent edits that file instead of starting a second copy. It also puts a question to the agent at the moment a known mistake is about to repeat, and keeps a ledger of the tokens Claude Code and Hermes spend.

**[Read the Locket guide](https://github.com/JACK-COM/homebrew-panoply/blob/main/docs/locket/README.md)**: when to use it, the first five minutes, how to tell it is working, troubleshooting, and [how to make it yours](https://github.com/JACK-COM/homebrew-panoply/blob/main/docs/locket/make-it-yours.md).

## Install

```sh
brew tap jack-com/panoply
brew trust --formula jack-com/panoply/locket
brew install locket
```

Or with [uv](https://docs.astral.sh/uv/): `uv tool install git+https://github.com/JACK-COM/locket`.

Then ask your agent to run `locket help install` and follow it. `locket doctor` checks the install and prints the fix for anything missing.

## Requirements

Python 3.11 or later and numpy. Ranking by meaning uses [ollama](https://ollama.com) 0.36 or later serving `embeddinggemma-2:270m`, or onnxruntime in the virtualenv the Panoply pieces share; without either, Locket compares words and says so. Supported hosts are Claude Code, Codex, Hermes Agent and any MCP client (`locket mcp`). Tested on macOS and Linux (Debian, Python 3.12); on Windows, run it under WSL.

## Developing

`src/locket` holds one module per job, imported as siblings. Two declarations carry most new work:

- A host is one row in `hosts.TABLE`, naming its home, the directories in it that are memory, and its module, plus that module under `src/locket/adapters/`. The module provides the hook face and the install face declared in `adapters/__init__.py`. The hooks, `install`, `uninstall` and `doctor` all read the hosts off the table, and `make typecheck` fails while a module lacks part of either face.
- A verb is one `Verb` in its family's `VERBS` under `src/locket/commands/`. The command line, its help and the MCP server's tools are all built from it.

`make test` checks the panoply-lib copies and the types (mypy, reading the code as Python 3.11 does), then runs `tests/` (stdlib `unittest`) and `locket selftest`, the install smoke check a package can run. Modules whose names start with an underscore are copies kept in step by panoply-lib; change them there.

## Releasing

`make version` (or `version-minor`, `version-major`) computes the next version from `__version__` and hands it to `scripts/release.sh X.Y.Z`. The script refuses a module past 800 lines and runs `make test` before it stamps the version, so a failed release leaves the tree clean. It then tags and pushes, moves the formula in [the tap](https://github.com/JACK-COM/homebrew-panoply) to the new tarball, and names any guide page the release has moved past.
