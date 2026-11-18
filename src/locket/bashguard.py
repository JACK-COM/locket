"""The shell guard: what a Bash call would do to a memory store that the write gate never
sees, decided by an ordered chain of rules.

The write gate matches the Write and Edit tools, so a memory file rewritten by a shell
redirect or a Python heredoc never reaches it and nothing says it did not: a full
memory-maintenance session wrote every correction through heredocs and tripped the gate
zero times. This guard is a deliberate carve-out from any session directive preferring
Bash for file edits, for memory stores only.
"""
import os
import re
from pathlib import Path
from typing import NamedTuple, Optional

import corpora
import fsio
import gate
import ledger
import paths
import shellscan
import shellwrites


_QUOTED = re.compile(r"""(?:echo|printf)\s+(?:-[a-zA-Z]+\s+)*(?:"((?:[^"\\]|\\.)*)"|'([^']*)')""")


_CSV_REDIRECT = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*(?P<p>[^\s;&|<>]+\.csv)")


def ledger_bash_decision(command, cwd):
    """(message, blocking) for a shell command appending to a schema-governed CSV.

    A heredoc body or an echo/printf string is read as rows and checked, and a
    violation blocks. Each body counts only against the ledger its own pipeline
    redirects into, so a second redirect elsewhere in a compound command is never
    read as the ledger's rows. A redirect into the ledger whose rows cannot be read
    off the command (a script, a variable) warns instead, because the check cannot
    decide it, and names the sweep that can."""
    if ".csv" not in command:
        return None, False
    targets, owner = {}, {}
    for m in _CSV_REDIRECT.finditer(command):
        tok = m.group("p").strip("'\"")
        cands = [tok] if tok.startswith("/") else [str(Path(r) / tok) for r in shellwrites.cd_targets(command, cwd, m.start("p"))]
        for c in cands:
            schema, glob = ledger.ledger_schema_for(c)
            if schema is not None:
                owner[tok] = c
                targets[c] = (schema, glob)
                break
    if not targets:
        return None, False
    # keyed by resolved path, so two spellings of one ledger share one verdict
    bodies: dict[str, list[str]] = {c: [] for c in targets}

    owners = {}                             # pipeline span: the ledgers it redirects into

    def attribute(pos, body):
        span = shellscan.pipeline_span(command, pos)
        if span not in owners:
            owners[span] = {owner.get(m.group("p").strip("'\""))
                            for m in _CSV_REDIRECT.finditer(shellscan.pipeline_of(command, pos))}
        for c in owners[span]:
            if c in bodies:
                bodies[c].append(body)

    for d in shellscan.heredocs(command):
        # the opener's pipeline in the full command, so a redirect on a continued line counts
        attribute(d.start, d.body(command))
    for m in _QUOTED.finditer(command):
        attribute(m.start(),
                  (m.group(1) if m.group(1) is not None else m.group(2))
                  .replace("\\n", "\n").replace('\\"', '"'))
    # every target is checked before any warns: an opaque write earlier in the command
    # must never stand in for a readable, violating one later
    for path, (schema, glob) in targets.items():
        header = ledger.ledger_header(path)
        findings = []
        for body in bodies[path]:
            if header is None:
                findings += ledger.check_ledger_text(schema, body)
            else:
                findings += ledger.check_ledger_text(schema, body, header)
        if findings:
            return ledger.ledger_message(path, glob, schema, findings, "bashguard"), True
    for path in targets:
        if not bodies[path]:
            return (f"This appends to a schema-governed ledger ({Path(path).name}) with rows the "
                    f"guard cannot read off the command, so the caps go unchecked here. Run "
                    f"`locket ledgers` after it, or write the row through Write/Edit, "
                    f"where the check is mechanical."), False
    return None, False


SCRATCH_MARKERS = ("/scratchpad/", "/private/tmp/", "/tmp/")


MEMWRITE_ESCAPE = "MEMWRITE-OK:"


def bash_write_decision(command, cwd, autonomous=False):
    """(message, blocking) for a Bash command touching a corpus, or (None, False).

    Exempts what 2d exempts, plus the scratchpad and autonomous runs, and honours a
    declared MEMWRITE-OK escape the way the dossier hook honours NOT-CURATION: a
    reword records nothing, a declaration records the decision.
    """
    if ".md" not in command or autonomous or MEMWRITE_ESCAPE in command:
        return None, False
    if any(mark in command for mark in SCRATCH_MARKERS):
        return None, False
    blocking, opaque, unlisted = shellwrites.bash_write_targets(command, cwd)

    def live(q):                             # exempt by ITS corpus's names, not ours
        sets = corpora.settings(corpora.corpus_for(q))
        return q.name not in sets["append_surfaces"] and "archive" not in q.parts
    blocking = [q for q in blocking if live(q)]
    opaque = [q for q in opaque if live(q)]
    unlisted = [(w, q) for w, q in unlisted if live(q)]
    if blocking:
        names = ", ".join(sorted({q.name for q in blocking}))
        return (f"This writes to a memory corpus through Bash ({names}), where the write-time "
                f"gate cannot see it. That gate matches the Write and Edit tools, so a shell "
                f"redirect skips the 2c one-authority check and the 2e already-has-a-home "
                f"question, silently.\n"
                f"  - Use Write or Edit for memory files, even where a session directive "
                f"prefers Bash. Memory corpora are the carve-out.\n"
                f"  - A ledger, a holding space, anything under archive/ and the scratchpad "
                f"are exempt and never reach this message.\n"
                f"  - If Bash is genuinely right here (an atomic multi-file patch, a scripted "
                f"migration), put `{MEMWRITE_ESCAPE} <reason>` in the command and it passes."), True
    if opaque:
        names = ", ".join(sorted({q.name for q in opaque}))
        return (f"This looks like a scripted write into a memory corpus ({names}). The "
                f"write-time gate matches Write and Edit, so if it does write there the 2c and "
                f"2e checks will not fire and nothing will report that they did not. Route "
                f"the edit through Write/Edit, or put `{MEMWRITE_ESCAPE} <reason>` in the "
                f"command if Bash is genuinely right. Running `memfind` first does not "
                f"substitute: it answers 2e and leaves the 2c check unrun."), False
    if unlisted:
        what = ", ".join(sorted({f"`{w}` on {q.name}" for w, q in unlisted}))
        return (f"This hands a memory file to a command Locket does not know to be read-only "
                f"({what}). If it writes the file, the write-time gate never sees it, because "
                f"that gate matches Write and Edit. Route the edit through Write/Edit, or put "
                f"`{MEMWRITE_ESCAPE} <reason>` in the command if Bash is genuinely right; a "
                f"read-only command can be ignored."), False
    return None, False


_CONFIG_WORD = re.compile(r"[^\s;&|<>()]*(?:" + "|".join(re.escape(n) for n in gate.CONFIG_NAMES) + r")\b",
                          re.IGNORECASE)


_TRUST_VERB = re.compile(r"\b(?:locket|memscan(?:\.py)?|locket\.py)\s+(?:trust|untrust|forget)\b")


TRUST_REFUSAL = ("Granting or withdrawing a store's trust is a person's act at their own terminal, "
                 "never through an agent's shell. Tell the user the command to run.")


def shell_store_config(command, cwd):
    """(store, name) for the first store configuration file a shell command names,
    read as the shell would resolve it, else None. Naming one is enough: no agent task
    needs to touch a store's configuration from a shell, and whether a command writes a
    file cannot be read off its text, while whether it names one can."""
    for i, m in enumerate(_CONFIG_WORD.finditer(command[:fsio.MAX_SCAN])):
        if i >= 100:
            break
        tok = m.group(0)
        bases = [""] if tok.startswith(("/", "~", "$HOME", "${HOME}")) else shellwrites.cd_targets(command, cwd, m.start())
        for b in bases:
            path = shellwrites.expand_home(tok) if not b else str(Path(b) / tok)
            hit = gate.store_config(path)
            if hit is not None:
                return hit
    return None


_PATHLIKE = re.compile(r"""[^\s'"<>;&|()]+\.(?:md|csv)\b""")


def oversized_decision(command, cwd):
    """(message, blocking) for a shell command too long to scan. Its parsers are
    linear, but not run past MAX_SCAN, so padding cannot buy a slow hook; and since
    a skipped scan must not read as a clean one, a command naming a store file
    anywhere is treated as a write to it, never as undecided."""
    if MEMWRITE_ESCAPE in command:
        return None, False
    autonomous = os.environ.get("CLAUDE_AUTONOMOUS") == "1"
    for i, m in enumerate(_PATHLIKE.finditer(command)):
        if i >= 500:
            break
        tok = m.group(0)
        cands = [tok] if tok.startswith(("/", "~")) else [
            str(Path(r) / tok) for r in shellwrites.cd_targets(command, cwd, m.start())]
        for c in cands:
            c = str(Path(c).expanduser())
            if corpora.corpus_for(c) is not None or ledger.ledger_schema_for(c)[0] is not None:
                return (f"This command is {len(command)} characters, past the {fsio.MAX_SCAN} the guard "
                        f"scans, and names {fsio.relay(c, 160)} in a memory store, so it is read as a write "
                        f"to it. Split it, or add {MEMWRITE_ESCAPE} <reason> if the write is intended."), autonomous
    return None, False


# ------------------------------------------------------------- the rule chain

class Shell(NamedTuple):
    """One Bash call as every rule reads it."""
    command: str                # as sent
    bare: str                   # shell code only (heredoc bodies and comments gone), quotes removed
    cwd: Optional[str]
    autonomous: bool


def _trust_verb(sh):
    return (TRUST_REFUSAL, True) if _TRUST_VERB.search(sh.bare) else None


# The shell refuses a command that NAMES one of these files, reading or not, because what a
# command writes cannot always be read off its text; each message names the way to read.
SHELL_CONFIG_REFUSAL = (
    "This shell command names {name}, the configuration that decides what Locket does in {store}, "
    "and a shell command naming it is refused whether or not it writes. To read it, open it with "
    "your file-reading tool, which no Locket guard refuses; `locket schema {store}` checks it. A person "
    "edits it by hand: tell the user what you would change and why.")
SHELL_REGISTRY_REFUSAL = (
    "This shell command names Locket's trust registry or embedder settings ({reg}), and a shell "
    "command naming either is refused whether or not it writes. To read them, `locket corpora` lists "
    "the stores and `locket configure` shows each setting and where it comes from. Changing either is "
    "a person's act at a terminal: ask the user to run  locket trust <dir>, locket untrust <dir>  or  "
    "locket configure embedder ...  themselves.")


def _store_config(sh):
    cfg = shell_store_config(sh.bare, sh.cwd)
    return None if cfg is None else (SHELL_CONFIG_REFUSAL.format(name=cfg[1], store=cfg[0]), True)


def _registry(sh):
    # A shell command naming the registry is refused whether or not it writes: no
    # agent task needs the file, and a write to it would grant trust.
    b = sh.bare
    if (".locket" in b and ("corpora.json" in b or "config.json" in b)) \
            or (".panoply" in b and "config.json" in b) or any(str(p) in b for p in gate.registry_paths()):
        return SHELL_REGISTRY_REFUSAL.format(reg=paths.PATHS.registry), True
    return None


def _ledger_rows(sh):
    msg, blocking = ledger_bash_decision(sh.command, sh.cwd)
    return None if msg is None else (msg, blocking)


def _corpus_writes(sh):
    msg, blocking = bash_write_decision(sh.command, sh.cwd, autonomous=sh.autonomous)
    return None if msg is None else (msg, blocking)


# Each rule takes a Shell and returns (message, blocking), or None to pass the call on.
# The first rule that decides speaks, so the refusals no task needs come before the
# checks that read the command's writes. A new guard is one function placed here.
RULES = (_trust_verb, _store_config, _registry, _ledger_rows, _corpus_writes)


def decide(command, cwd, autonomous=False):
    """(message, blocking) for one Bash call; (None, False) passes it. A command past
    MAX_SCAN is read only for the memory paths it names (`oversized_decision`)."""
    if len(command) > fsio.MAX_SCAN:
        return oversized_decision(command, cwd)
    sh = Shell(command, re.sub(r"""['"\\]""", "", shellscan.command_text(command)), cwd, autonomous)
    for rule in RULES:
        d = rule(sh)
        if d is not None:
            return d
    return None, False
