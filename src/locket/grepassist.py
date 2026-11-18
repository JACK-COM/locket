"""The 2e assist on a search: whether a grep's pattern reads as a question about meaning,
and if so the files a store's index ranks closest to it, injected beside the grep's own
result.
"""
import os
import re
import shlex

import corpora
import fsio
import gate
import health
import paths


# A regex handed to an embedding ranker is noise, so the injector fires only on a
# pattern that reads as a question: two content words, no metacharacters. Words are
# separated by whitespace only: a token joined by a hyphen, dot, underscore or slash
# (`angle-scout`, `settings.json`, `spec_provenance`) is one identifier, never a phrase.
GREP_META = set("|[]()\\^$*+?{}")


def prose_like(pattern):
    if not pattern or GREP_META & set(pattern):
        return False
    words = (w.strip(",;:.!'\"") for w in pattern.split())
    return len([w for w in words if re.fullmatch(r"[A-Za-z0-9]{4,}", w)]) >= 2


GREP_VERBS = {"grep", "egrep", "rg", "ugrep"}


def grep_words(cmd, cwd=None, _nested=False):
    """(pattern, path words) for the grep in a shell command, or None when it runs none.

    Only a grep's own words count: the command is split at `;`, `&&`, `|` and redirects
    first, and every word naming a grep verb is tried until one yields a pattern, so `which
    grep; grep ...` still finds the second. A command with no verb is searched once inside
    the command string a shell takes after `-c` (`bash -c '...'`), never in any other
    quoted word, where a commit message or a heredoc that mentions grep would fire it.
    The pattern is the `-e`/`--regexp` value where one is given, otherwise the first word
    after the verb that is not a path, a bare count or a non-question value of a bare long
    option, and it counts only when shaped like a question (`prose_like`); no per-tool list
    of value-taking flags is needed. The path words are every
    other word after an `-e` pattern, else the words after the pattern, last first, so a
    value-taking flag between pattern and path cannot shadow the path."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        toks = list(lex)
    except ValueError:                      # unbalanced quotes: words are still words
        toks = cmd.split()
    verbs = [k for k, t in enumerate(toks) if os.path.basename(t) in GREP_VERBS]
    if not verbs:
        if not _nested:
            for flag, t in zip(toks, toks[1:]):
                if re.fullmatch(r"-[A-Za-z]*c", flag):
                    got = grep_words(t, cwd, True)
                    if got:
                        return got
        return None

    def pathlike(w):
        if "/" in w or w.startswith(("~", "$", ".")):
            return True
        return os.path.exists(os.path.join(cwd or os.getcwd(), w))

    for i in verbs:
        seg: list[str] = []
        for t in toks[i + 1:]:
            if t and not t.strip("();<>|&"):   # an operator ends the grep's words
                if t[0] in "<>" and seg and seg[-1].isdigit():
                    seg.pop()               # the `2` of `2>/dev/null` is a descriptor, not a word
                break
            seg.append(t)
        flagged, positional, opts, it, after_opt = None, [], True, iter(seg), False
        for w in it:
            # whether the word before this one was an option that could take it as a value
            follows_opt = after_opt
            after_opt = opts and w.startswith("-") and w not in ("-", "--") and "=" not in w
            attached = re.fullmatch(r"-[A-Za-z]*e(.+)", w, re.S) if opts else None
            if opts and w == "--":
                opts = False
            elif opts and w.startswith("--regexp="):
                flagged = flagged or w.partition("=")[2]
            elif opts and (w == "--regexp" or re.fullmatch(r"-[A-Za-z]*e", w)):
                value = next(it, None)      # consumed even when an earlier -e won
                flagged = flagged or value
            elif attached and not w.startswith("--"):
                flagged = flagged or attached.group(1)
            elif not (opts and w.startswith("-") and w != "-"):
                positional.append((w, follows_opt))
        if flagged:
            return flagged, [w for w, _ in positional][::-1]
        # The pattern is the FIRST word that can be one: a path, a bare count (`-A 12`) and a
        # non-question word right after an option (`-g '*.md'`, `-t py`, `--exclude-dir
        # node_modules`) are flag values or targets. When that first word is not a question
        # the grep has none to rank, so the scan stops there; searching further took a
        # filename (`settings.json`) as the query. ⚠ A regex right after a boolean option
        # (`grep -r 'a|b' 'two words here'`) is skipped as a value, so a later prose-shaped
        # path with spaces would be taken; per-tool flag tables were rejected as the fix.
        at = next((j for j, (w, fl) in enumerate(positional)
                   if not pathlike(w) and not w.isdigit() and not (fl and not prose_like(w))), None)
        if at is not None and prose_like(positional[at][0]):
            return positional[at][0], [w for w, _ in positional[at + 1:]][::-1]
    return "", []


GREPASSIST_LOG_CAP = 1 << 20                # past this the file keeps its newest half


def trim_log(path, cap):
    """Keep the newest half of `path` by bytes once it passes `cap`. The kept half is
    written beside the log and renamed over it, so a parallel hook never sees a truncated
    file; an append racing the rename lands on the old inode and is lost, which a
    diagnostic log tolerates."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size <= cap:
        return
    with path.open("rb") as fh:
        fh.seek(size // 2)
        fh.readline()                       # drop the partial row the seek landed in
        kept = fh.read()
    fsio.replace_atomic(path, kept, ".grepassist-")


def log(tool, decision, reason, pattern):
    """Record one grepassist decision and return 0, the hook's exit code.

    The hook runs on every grep and returns neighbours only for a question-shaped one, so
    a silent call is either a correct decline or a broken hook; this row is what tells
    the two apart. Columns: UTC time, tool, decision (`returned`/`declined`), reason,
    pattern's first 80 characters. A grep that never parsed as one is not logged; a grep
    outside every corpus is (`no-corpus`), because a broken corpus lookup would show as
    exactly that. The file is 0600, since a pattern can carry a secret. Writing must never
    break the hook, so every failure here is swallowed."""
    try:
        trim_log(paths.PATHS.grepassist_log, GREPASSIST_LOG_CAP)
        clean = re.sub(r"[\r\n\t,\"]+", " ", str(pattern))[:80]
        if clean[:1] in ("=", "+", "-", "@"):
            clean = "'" + clean             # a spreadsheet reads these as a formula
        row = f"{health.utc()},{tool},{decision},{reason},{clean}\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        try:
            fd = os.open(str(paths.PATHS.grepassist_log), flags, 0o600)
        except FileNotFoundError:
            paths.PATHS.grepassist_log.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(paths.PATHS.grepassist_log), flags, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(row)                   # one short write: atomic under O_APPEND
    except Exception:
        pass
    return 0


def grep_assist(pattern, root, top=4):
    """Ranked semantic neighbours for a grep about to run over a memory corpus.

    Reference-CSV rows rank here beside the markdown wherever the corpus registers one
    in `reference-sources.json`, because this routes through `memfind.rank` and memfind
    indexes both, rows a markdown-only index cannot see.

    The gap: a grep answers which file contains these WORDS, and a paraphrase shares
    fewer than MIN_SHARED tokens, so it scores 0.0 and reads as absent. 2e says run
    memfind BEFORE the grep, never as a second opinion after one, and that
    instruction was live and ignored twice by supervised sessions, which is why it
    is mechanical here rather than prose.
    A literal grep pattern ranks worse than a phrased question, so this is a floor
    rather than a replacement for asking properly.
    """
    got = gate.semantic_neighbours(root, pattern, top=top, timeout=3)
    if not got or not got[0]:
        return None
    rows, _stale = got
    lines = [f"  [{sc:.3f}] {fsio.relay(name, 120)}  ::  {fsio.relay(claim, 110)}" for sc, name, claim in rows]
    return ("Before the grep: these rank closest to that text by MEANING, which is what a "
            "grep cannot see (a paraphrase shares too few tokens to match at all).\n"
            + "\n".join(lines)
            + "\nRead them as candidates, never as a verdict; the scores adjudicate nothing "
              "(prime-memory-discipline 2e). A `<file>.csv#<area>` result is one ROW of a reference "
              "CSV rather than a file, so query that CSV on `Area=` for the rest.")


def advise(payload):
    """The neighbours to inject beside one normalised Grep or Bash call, or None when the
    call is no search, no store answers it, or the pattern is not a question. Every
    outcome is logged; an absent embedder or a bad payload declines, never raises."""
    tool, pattern = "?", ""
    try:
        ti = payload["tool_input"]
        tool = payload["tool_name"]
        cwd = payload.get("cwd") or os.getcwd()
        if tool == "Grep":
            pattern, words = str(ti.get("pattern", "")), [ti["path"]] if ti.get("path") else []
        elif tool == "Bash":
            got = grep_words(str(ti.get("command", "")), cwd)
            if got is None:
                return None
            pattern, words = got
        else:
            return None
        root = corpora.search_root(words, cwd) or corpora.corpus_here(cwd)
        if root is None:
            log(tool, "declined", "no-corpus", pattern)
            return None
        if not prose_like(pattern):
            log(tool, "declined", "not-a-question", pattern)
            return None
        out = grep_assist(pattern, root)
    except Exception as e:
        health.health_error("grepassist", "rank", e)
        log(tool, "declined", f"error:{type(e).__name__}", pattern)
        return None
    log(tool, "returned" if out else "declined", "ranked" if out else "no-neighbours", pattern)
    return out
