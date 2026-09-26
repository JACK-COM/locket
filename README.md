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

Python 3.9 or later, standard library only. Ranking by meaning uses [ollama](https://ollama.com) serving `nomic-embed-text`, or `fastembed` in the virtualenv the Panoply pieces share; without either, Locket compares words and says so. Supported hosts are Claude Code, Codex, Hermes Agent and any MCP client (`locket mcp`). Tested on macOS and Linux (Debian, Python 3.12); on Windows, run it under WSL.

## Releasing

`make version` (or `version-minor`, `version-major`) computes the next version from `__version__` and hands it to `scripts/release.sh X.Y.Z`, which stamps it, runs the selftests, tags and pushes, then moves the formula in [the tap](https://github.com/JACK-COM/homebrew-panoply) to the new tarball and names any guide page the release has moved past. `make test` runs the selftests alone.
