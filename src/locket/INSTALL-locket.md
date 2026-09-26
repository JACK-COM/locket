# Install Locket

Locket is one command, `locket`, over three scripts: `memscan.py` and `memfind.py`, which keep an agent's memory from holding the same fact twice, and `locket_mcp.py`, which offers the same checks as tools to a host that has no hooks. You are reading this because your user asked you to install it, or ran `locket help install`. Read the whole file, then do every step in order, in this session, without waiting to be asked step by step. Ask the user only where a step says to.

`locket doctor` checks every step a machine can check and prints the command that fixes each failure. This file holds what it cannot check: the decisions only you can make, and the report.

## Two ways to run this, and you start in the first

**Collaborative** is the default. Every check that fires asks you to read a short list and decide; nothing is auto-fixed; each host prompts the user once before a new hook may run. Tell the user this is the mode you are starting in, and that it is the one where they see what the tools do.

**Effortless** is the pathway once the user has watched a few firings and trusts them: hooks approved without a prompt (`hooks_auto_accept: true` on Hermes, or `hermes chat --accept-hooks` at each launch; nothing to change on Claude Code), and you running the Step 5 audit on a schedule you own. Do not switch to it on your own. When the user asks for less friction, name these two changes, say what each widens, and make the ones they choose.

## What these tools do, and what they do not

`memscan.py` treats every sentence in a memory file as a claim. It finds the same claim in two files (a fork) or twice in one (a duplicate), files past a size budget or carrying dated history clauses that should have been rewrites, index lines whose target has changed, and `path:line` citations that no longer point at code. `memfind.py` ranks which memory file already holds an idea, by meaning, so a fact restated in new words is found before it is written a second time. Neither returns a verdict; both return a list for you to read.

They check structure. They cannot tell whether a stored fact is still true, and they cannot find two files that contradict each other; `locket siblings` lists file pairs sharing one subject, the shortlist a reader adjudicates. Do not tell the user contradictions are detected. Every check scores one store against itself, so a fact written into two stores is caught by neither at write time; `locket across` is the one check that reads every store at once, and it sees a copy only while the two copies still agree.

One sentence you may give the user when you are done: *"Locket is installed: I now check my own memory before I write to it, so the same fact does not end up in two places."*

## Step 1. Put the command on this machine

If you cannot run a command on this machine, you are on a hookless host: skip to Step 4 and take the MCP branch, then come back to Step 5 through the `locket_*` tools once the app has restarted. `locket_status` stands in for `doctor` there. Do not report a result for a command you could not run.

- **Homebrew (macOS, Linux).** `brew tap jack-com/panoply` and `brew trust --formula jack-com/panoply/locket` once, then `brew install locket`; Homebrew refuses a formula from an untrusted tap. Upgrades come with `brew upgrade`.
- **From the scripts.** Copy `memscan.py`, `memfind.py`, `locket.py`, `locket_mcp.py` and this file together into one stable directory (`~/.claude/scripts/` on Claude Code, `~/.codex/scripts/` on Codex, `~/.hermes/scripts/` on Hermes); the scripts import each other from their own directory. If they already sit in one of those places, use them there, because two copies drift. On macOS files in Downloads are unreadable to an agent until the user grants access; if a read there is refused, ask the user to move them. Then run `python3 <that directory>/locket.py install`, which links `locket` into `~/.local/bin` (or `LOCKET_BIN`) and prints an `export` line when that directory is not on `PATH`. On Windows the interpreter is usually `python`, and an alias stands in for the link.

Then run `locket doctor` and read it. It will fail on the hooks until Step 4; that is expected.

## Step 2. Find your memory stores

A store is a directory of markdown you write memory into. A host's core directory is its **council store**: the one place its constitution, rules, skills and memory live, treated as one corpus, so a rule restating a memory fact draws the same nudge as a second memory file would.

- **Claude Code.** `~/.claude/` is the council store, recognised without configuration; its members are `memory/`, `rules/`, `skills/`, `agents/`, `agent-memory/` and the markdown at the root. Each `~/.claude/projects/<slug>/memory/` is a store of its own.
- **Hermes Agent.** `~/.hermes/` is the council store, with `memories/` and `SOUL.md`; a Hermes that authors its own skills adds `"members": ["memories", "skills"]` to `~/.hermes/locket.json`.
- **Codex.** `~/.codex/` (or `$CODEX_HOME`) is the council store, with `memories/` and `AGENTS.md`. Codex writes `memories/` itself, from past sessions and outside any tool call, so no hook sees those writes: on Codex the write-time check covers `AGENTS.md` and any memory file you keep by hand, and the audit in Step 5 is what reads the memories Codex generated. Say so in the report rather than claim more.
- **Anything else.** Find the directory that holds your memory, rules and skills together and run `locket init <that directory>`. A hand-written `locket.json` is not enough: nothing searches the disk for manifests, so only `init` puts a store in the listing.

Two folders that are one memory, such as a project's documents beside its code and the host's auto-memory for that project, join: `locket init <folder> --parent <name>`. A host's auto-memory folder is always called `memory`, which collides with the council store's own `memory/`, so give it a label: `--name <label>`. If your memory is not markdown files on disk, stop and tell the user these tools cannot see it.

## Step 3. Give `find` an embedder

`doctor` names the embedder that answers, or warns that `find` has fallen back to word overlap, which cannot see a paraphrase. Change nothing the machine already has; stop at the first rung that answers:

1. **ollama is installed but has no model:** `ollama pull nomic-embed-text`. The server need not be running; `memfind.py` starts it.
2. **No ollama:** prefer `fastembed`, in-process with no server. A system `pip install` is refused on Homebrew and distro Pythons, so make the venv every Panoply piece shares, with the same `python3` that runs Locket: `python3 -m venv ~/.panoply/venv && ~/.panoply/venv/bin/python -m pip install fastembed` (`PANOPLY_VENV` if elsewhere). An existing `~/.locket/venv` is found and used as it stands. The model downloads on first use, about 130 MB. Offer ollama instead only if the user would rather run a server than a venv.
3. **Nothing could be installed:** say what that loses, that a fact restated in different words will not be found.

Then `locket index all`.

## Step 4. Register the hooks

The hooks make the checks fire at write time instead of only when you remember to run them.

**Claude Code.** `locket install --hooks` merges three `PreToolUse` entries into `~/.claude/settings.json`, plus `locket trigger` on every tool call and every prompt, keeping every entry already there and a copy of the file beside it. The trigger hook is silent until a store holds a `triggers.json`: rows the user writes for mistakes that have already happened, each asking the agent a question at the moment it is about to repeat one (`locket help trigger`). Mention it; do not write rows for them. It also registers `locket usage hook` on `SessionEnd`, which copies the session's token counts into `~/.locket/usage.csv` before Claude Code's 30-day cleanup can delete the transcript; `locket usage` reads it back. If `doctor` then says the write hook is registered but silent, ask the user to run `/reload-plugins`, and failing that to restart the session.

**Hermes.** Merge this into `~/.hermes/config.yaml` at the top level, keeping any `hooks:` already there, with `$LOCKET` the absolute path `locket status` names as `command`:

```yaml
hooks:
  pre_tool_call:
    - command: "$LOCKET hook"
      matcher: "write_file|patch|memory"
      timeout: 10
      fail_closed: false
    - command: "$LOCKET bashguard"
      matcher: "terminal"
      timeout: 5
      fail_closed: false
    - command: "$LOCKET grepassist"
      matcher: "search_files|terminal"
      timeout: 10
      fail_closed: false
    - command: "$LOCKET trigger"
      timeout: 5
      fail_closed: false
  pre_llm_call:
    - command: "$LOCKET trigger"
      timeout: 5
```

Hermes asks the user once, at the next launch, to approve each new hook; tell them to expect it, and do not set `hooks_auto_accept` for them. A Hermes tool hook can refuse a call but cannot add advice to one, so where Claude Code would add advice, Locket refuses the call once, leading with how to proceed: repeat the same call unchanged. `locket trigger` is silent until `~/.hermes/triggers.json` holds rows (`locket help trigger`); on `pre_llm_call` its prompt rows reach you with the message they matched. A config written for an older Locket registers `$LOCKET deliver` on `pre_llm_call`, which now does nothing; replace it with the `trigger` entry.

**Codex.** `locket install --codex` merges the same entries into `~/.codex/hooks.json`, which takes Claude Code's schema, without the usage recorder, which cannot read Codex's sessions yet. Codex runs a hook only after the user trusts it: ask them to start `codex`, run `/hooks`, and trust each Locket entry, and tell them Codex asks again only when an entry changes. `doctor` cannot see that trust and names it as a warning until then. A file edit reaches the hooks as a patch, native or through the shell on a model with no `apply_patch` tool, and Locket reads it as the write it amounts to; advice reaches you in the same turn, as on Claude Code. Trigger rows live in `~/.codex/triggers.json`.

**Claude Desktop, or any host that can attach a local MCP server.** `locket install --desktop` registers the server by absolute path, since the app spawns servers with a minimal `PATH` where a bare `python3` can be an installer stub that hangs. Where you cannot run it, add `"locket": {"command": "<the absolute path of locket>", "args": ["mcp"]}` under `mcpServers` by hand. Ask the user to restart the app once, then call `locket_init` on any directory from Step 2. Nothing fires by itself on such a host: put one line in the project's instructions, *before writing a durable fact to memory, call `locket_check_before_write`, and run `locket_audit`, `locket_graduated` and `locket_links` at session close*. Pass `append: true` when the text goes on the end of a file and `old_text` when it is an edit. The app shares one server across its conversations until it restarts.

**Anything else.** Skip this step; Step 6 is your discipline instead of a hook.

## Step 5. `doctor`, then the first audit

Run `locket doctor` until it reports no failure. A warning is a rung missing that Locket works without; say which in the report.

Then audit your own store (on a hookless host through the `locket_*` tools). `all` reaches every store on the machine, and on a machine shared with another agent that is someone else's memory: sweep it, and run `across`, only if the user says to.

```
locket audit
locket graduated
locket dupes
locket budget
locket links
```

Fix nothing yet. An `audit` pair is fork risk, not proof; an index file restating what it points at is doing its job. Pick the three findings you would act on first.

## Step 6. The change in how you work

This is the part the user cannot do for you.

- Before writing a durable fact into memory, run `locket find "<the fact in one sentence>" <store>` and read the top file. If it already holds the idea, rewrite the passage there instead of adding a sibling.
- Write memory through your file tools, never through a shell redirect, `sed -i` or a heredoc. The hooks watch the file tools.
- Give every fact one home and point at it from everywhere else. The home of a fact several stores need is the one they all read; `locket find "<the fact>" all` asks every store at once.
- Keep each entry unique in an inbox or scratch file whose entries get promoted or discarded; `dupes` checks it.
- Write each fact as a whole sentence in the body of a file. A fact only in a heading, a code fence or front matter is invisible to these tools.
- Where more than one session can hold the same store, edit surgically, and treat a *modified on disk since you read it* warning as a stop. A whole-file rewrite over another session's write loses it silently.

Hook messages cite `prime-memory-discipline.md` by section. That file holds the general rules these tools enforce where a deploy carries it; where it is absent, those references mean this section.

**Two structured surfaces worth adding, if the user agrees.** Things the user has ruled and things you found the hard way pile up in prose. Keep each in a CSV beside the store, one sentence per row: `RULINGS.csv` (`Ruling,Category,Area,Date`) for decisions that must not be relitigated, and `CLAIMS.csv` (`Claim,Category,Area,Source,Date`) for facts a future session would otherwise rediscover. A fact every session needs belongs in markdown that loads; one needed only when its area comes up belongs in a CSV. A ruling closes an option a session would otherwise re-decide, a claim states what it would otherwise re-find, so only a claim carries a `Source`. A fact in both, or in a CSV and in prose, is a fork. Register them so `find` ranks their rows:

```json
{"sources": [
  {"path": "/abs/path/RULINGS.csv", "text": "Ruling", "label": "Area"},
  {"path": "/abs/path/CLAIMS.csv", "text": "Claim", "label": "Area"}
]}
```

Point at rows by file and filter, never by copying them: *(`RULINGS.csv`, query `Area=guides`)*; `links` reports a pointer whose file, column or value matches nothing. When seeding the CSVs from prose, remove the prose copies in the same pass.

## Step 7. Report

Tell the user, in plain language and under ten lines, with no command output pasted:

1. The one sentence from *What these tools do*.
2. Which stores were found, and whether `doctor` passed.
3. The three audit findings you would act on first, one line each.
4. Which embedder rung you landed on, and what changes if it is word overlap.
5. What they may be asked to approve, and when the hooks take effect.
6. The operating system and interpreter (`locket status`), since these tools are tested on macOS and untested on Linux and Windows.

## Later

- **A new store.** `locket init <folder>`, then Step 5's audit over it. A folder inside a store, or containing one, is refused; `init` the directory you mean. `locket forget <folder>` drops one from the listing.
- **The manifest.** `locket.json` in a store holds its exceptions, all optional: `name`, `excluded_files`, `excluded_dirs`, `members`, `sources`, `ledgers`. `locket help scan` documents every key; `locket schema <folder>` checks one. For editor hints, point VS Code's `json.schemas` at `~/.locket/locket.schema.json` by `fileMatch: ["locket.json"]`, with the home directory written out in full.
- **Removal.** `locket uninstall` prints what it will remove and asks once (`--dry-run` only prints, `--purge` deletes each store's manifest too): the command link, `~/.locket/`, every cache, the Claude Code and Codex hook entries and the Desktop server, keeping a copy of each config it edits. The fastembed venv the Panoply pieces share goes too when no other piece is on PATH, and otherwise stays with the pieces named. Hermes hook lines are YAML and stay for you to remove; a Homebrew install ends with `brew uninstall locket`.
