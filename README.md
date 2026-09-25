<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/locket-dark.svg">
  <img src="docs/locket-light.svg" alt="Locket" width="112">
</picture>
</p>

# Locket

*Remember who you are.* One piece of [the Panoply](https://github.com/JACK-COM/homebrew-panoply).

Locket keeps an AI agent's memory from holding the same fact twice. It reads every sentence in a folder of markdown memory as a claim, finds a claim stated in two files or twice in one, and ranks which file already holds an idea before the agent writes it a second time. Hooks run the check at write time, so the agent is told which file owns a fact while it can still edit that file instead.

It checks structure. It cannot tell whether a fact is still true or whether two files contradict each other.

It also asks a question at the moment a mistake is about to repeat. A store's `triggers.json` holds rows such as "this command discards uncommitted work", each matched against a tool call or a prompt and pointing at the file that owns the fact; `locket trigger` puts the row's question to the agent before the call runs, or refuses the call when the row says to.

## Install

```
brew tap jack-com/panoply
brew trust --formula jack-com/panoply/locket
brew install locket
```

Or with [uv](https://docs.astral.sh/uv/): `uv tool install git+https://github.com/JACK-COM/locket`.

Then ask your agent to run `locket help install` and follow it. The agent finds your memory stores, registers the hooks with your consent, and reports back in under ten lines. `locket doctor` checks every step a machine can check and prints the fix for each failure.

## Use

```
locket find "a fact you are about to write"   which file already holds it, by meaning
locket audit                                  claims two files both state
locket graduated && locket links && locket across
                                              the session-close checks
locket trigger check                          lint the trigger rows that can fire from here
locket doctor                                 is the install healthy
```

`locket -h` lists every command; `locket help scan`, `locket help find` and `locket help trigger` explain what a store is, how to read a ranking and how trigger rows work.

## Requirements

Python 3.9 or later, standard library only. `locket find` ranks by meaning when an embedder answers: [ollama](https://ollama.com) serving `nomic-embed-text`, or `fastembed` in the venv the Panoply pieces share (`~/.panoply/venv`, or an older `~/.locket/venv`). Without one it falls back to word overlap and says so. Supported hosts are Claude Code, Hermes Agent and any MCP client (`locket mcp`). Tested on macOS and Linux (Debian, Python 3.12); on Windows, run it under WSL.

## Releasing

`make version` (or `version-minor`, `version-major`) computes the next version from `__version__` and hands it to `scripts/release.sh X.Y.Z`, which stamps it, runs the selftests, tags and pushes, then moves the formula in [the tap](https://github.com/JACK-COM/homebrew-panoply) to the new tarball. `make test` runs the selftests alone.
