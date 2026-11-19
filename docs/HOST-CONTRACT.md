# Running Locket in another agent host

For anyone running Locket under an agent other than Claude Code, or adding a host to Locket. It answers two questions: which of Locket's protections a given host can give you, and what it takes to get the rest.

Locket acts at two moments: when memory should reach the model before it decides, and when the model writes to memory. A host is judged here only on whether it lets Locket in at those moments. Its tools, its coding loop and its model are out of scope.

## Turn off the host's own memory

Locket can only audit the memory it sees. A host that keeps a memory of its own (a built-in memory file, skills it writes for itself, instructions it rewrites over time) runs a second store that no gate checks. Facts fork between the two, and the drift that follows has no visible cause.

Switch the host's memory off where it can be switched off. Claude Code's auto-memory is a setting. Check a host by running a session that learns a new fact, then listing what changed in the host's home outside your memory directories: any new prose there is a second store.

A host's bundled skills and instructions compete with your memory in a milder way. When a bundled skill matches the prompt, the model may answer from it and never look in the store.

## Four levels

Each level names who decides when memory acts. A host reaches the highest level whose every item it provides.

### Pull

The model reaches the store with tools it chooses to call: `locket mcp` for any MCP client, or `locket find` from a shell.

Every MCP client reaches this level, Claude Desktop included (`locket install --desktop`; setup is in [INSTALL-locket.md](../src/locket/INSTALL-locket.md)). The write check is a tool too, `locket_check_before_write`, which the model calls only if its instructions tell it to. Memory acts only when the model thinks to look, so a fact it does not suspect exists stays unread.

### Push

The host puts memory in front of the model at fixed moments, without being asked.

1. **Session start.** The host runs a command and adds its output to the context.
2. **Prompt submit.** The host runs a command with the prompt text and adds its output beside the prompt. This is where recall by meaning arrives.
3. **Instruction files.** The host loads named files whole into the system prompt, and loads a path-scoped rule the first time a matching file is read.

### Gated write

A write into memory passes Locket's gate before it lands.

1. **A pre-tool hook that can block.** Before a file write, an edit or a shell command runs, the host hands Locket the tool name and its arguments. It honours a refusal, showing the reason to the model, and passes advice through.
2. **A post-tool hook that can annotate.** After a tool runs, the host appends a command's output to the result the model sees.
3. **A transcript Locket can read.** The gate refuses a session's first sizeable new memory write until the session has run `locket find`, and it learns that from the transcript. A host whose transcript Locket cannot parse makes this check pass every write, silently. Locket reads Claude Code's and Codex's transcripts; for any other host, a bridge has to write one in Claude Code's format.
4. **Session end.** The host runs a command when the session closes, with the transcript path, so `locket usage` can record the session.

### Curation at compaction

This level is optional, and no host Locket supports reaches it today. When the context fills, the host gives the store a chance to keep what is about to leave the window, rather than only summarising it into a window that ends with the session.

1. **An extension point at compaction**, where an extension can queue a turn of its own.
2. **A curation turn that writes through the gate**, so it is checked like any other write.
3. **One curation turn per compaction.** The curation turn reads files, which can fill the window again; an extension that queues a turn on every compaction loops. Queue once, and clear the flag when the host goes idle.

## Supported hosts

| Host | Level | Notes |
|---|---|---|
| Claude Code | Gated write | Locket's native protocol. Switch off auto-memory. |
| Codex | Gated write | Claude Code's event names; edits arrive as patches, and a refusal is a deny decision on stdout. `locket install` merges the hooks into Codex's `hooks.json`. |
| Hermes Agent | Partial gated write | Its own event names, and no channel for advice: a matched trigger row refuses the call once instead of advising. Hermes writes memory and skills of its own, which Locket cannot see. Hooks go into its `config.yaml` by hand. |
| Claude Desktop, other MCP clients | Pull | `locket mcp`. |

## Adding a host

Locket's hooks read Claude Code's payloads and answer in its protocol: exit 2 with the reason on stderr refuses, and stdout JSON carrying `hookSpecificOutput.additionalContext` advises. There are two ways to bring a host in.

- **An adapter**, under `src/locket/adapters/`, one row in `hosts.TABLE`. Use it when the host has its own hook configuration. It covers Locket's hooks only.
- **A bridge**, an extension written in the host's own extension API, which turns the host's events into Claude Code payloads and runs the commands in your `settings.json` unchanged. A bridge carries every hook you register, Locket's and your own, and it can write the transcript and drive curation. Use it when the host has an extension API rich enough for the levels above.

A bridge also has to mirror how Claude Code finds settings, or project-level hooks go missing with no error. The bridge has to:

- read hooks from both `settings.json` and `settings.local.json`, in the user directory and in the project's `.claude/`
- set `CLAUDE_PROJECT_DIR` to the project root
- skip any instruction file the host already loads itself, or the model reads it twice
- give the instruction files precedence over the host's own system prompt

The last point decides whether the agent behaves like itself. A host's default prompt carries its own identity line and style rules ("You are a coding assistant…", "Be concise"). Instruction files appended after it with no stated precedence load and can be quoted, yet the model weighs them as reference material. Claude Code states that the user's instructions override its defaults; a bridge should say the same, and drop the host's competing identity and style lines where its API allows.

### Example: pi

[pi](https://github.com/earendil-works/pi) 1.5 has an extension API that reaches every level, curation included. The mapping:

| Locket needs | pi event |
|---|---|
| Session start | `session_start`; add the output to the system prompt in `before_agent_start`, so it stands before the first prompt |
| Prompt submit | `before_agent_start`, which carries the prompt and can add a message |
| Pre-tool, can block | `tool_call`, returning `{ block, reason }` |
| Post-tool, annotate | `tool_result`, returning new content |
| Transcript | none in Claude Code's format; the bridge writes one, recording each tool call after its pre-tool hooks run |
| Session end | `session_shutdown` |
| Settle | `agent_settled` |
| Compaction | `session_compact`, then `sendUserMessage(…, { deliverAs: "followUp" })` |
| Its own memory | none |

pi loads `AGENTS.md` and `CLAUDE.md` from the working directory and its parents, and its MCP tools run through its code mode under their own names. Hooks that match MCP tools by name need checking against that.

## Checking a host

Run each check in the new host with your own store.

1. **Push.** Start a session and ask what instructions and session-start context it was given. The answer should name your load list, and each instruction file should appear once.
2. **Scoped rules.** Read a file a path-scoped rule matches, then ask whether the rule is now in context.
3. **Gated write.** Ask the model to append, word for word, a sentence a memory file already holds. Locket should refuse, and the model should quote the refusal.
4. **Transcript.** In a fresh session, ask for a new memory write of a few hundred characters before any search. Locket should refuse until the session has run `locket find`.
5. **Project hooks.** In a project with its own `.claude/settings.json`, trigger one of its pre-tool hooks.

6. **Instructions in force.** Give the same open-ended task in both hosts, in a project whose records hold a discrepancy, and compare which procedures the agent loads before working and whether it raises the discrepancy before answering.

The first five checks show that memory reaches the model and writes are gated. The sixth is the one a host's own system prompt can fail with every file loaded: the same model, with the same files in context, retrieved well in pi and still reasoned less carefully over what it retrieved.
