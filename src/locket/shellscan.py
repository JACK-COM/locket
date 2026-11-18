"""Shell text as bash reads it: where each heredoc's body lies, and which spans are
quoted or commented, by offset into the command.

One scanner for every guard that reads a command. A guard that takes a line bash runs
for heredoc body never sees that line, so each reader here errs toward code: a
construct the scanner does not model (a `case` pattern's `)` inside `$(...)`, a
`((` that bash re-reads as two subshells) leaves a body read as code, never the
reverse. `tests/test_shellscan.py` checks every reader against bash itself.
"""
import bisect
import re
from functools import lru_cache
from typing import NamedTuple

# What ends an unquoted word, bash's metacharacters.
_META = frozenset(" \t\n;&|()<>")
# A `#` starts a comment only at the start of a word.
_COMMENT_AFTER = frozenset(" \t\n;&|(")
# A program that runs a heredoc fed to it, so its body is code to every guard.
_INTERPRETER = re.compile(r"\b(?:python[0-9.]*|node|deno|bun|bash|sh|zsh|dash|fish|perl|ruby|php|osascript|lua)\b")


class Heredoc(NamedTuple):
    start: int          # the `<<` operator
    end: int            # past the delimiter word on the opening line
    word: str           # the delimiter, quotes removed
    dash: bool          # `<<-`: bash strips leading tabs from body and delimiter lines
    quoted: bool        # any part of the word quoted: bash expands nothing in the body
    body_start: int     # the line after the opener's
    body_end: int       # the delimiter line's start, or the command's end when unterminated
    close: int          # past the delimiter line's newline

    def body(self, command):
        """The body as bash feeds it to the program, without its last newline."""
        text = command[self.body_start:self.body_end]
        text = text[:-1] if text.endswith("\n") else text
        return "\n".join(ln.lstrip("\t") for ln in text.split("\n")) if self.dash else text


class Scan(NamedTuple):
    heredocs: tuple     # Heredoc, in the order their bodies appear
    quoted: tuple       # (start, end) inside a quote, the quote characters excluded
    comments: tuple     # (start, end) from `#` to the line's end


def _word(command, i):
    """(word, quoted, end) for the heredoc delimiter starting at `i`, quotes removed."""
    n, out, quoted = len(command), [], False
    while i < n and command[i] not in _META:
        c = command[i]
        if c == "\\" and i + 1 < n:
            out.append(command[i + 1])
            quoted, i = True, i + 2
        elif c in "'\"":
            j = command.find(c, i + 1)
            j = n if j < 0 else j
            out.append(command[i + 1:j])
            quoted, i = True, j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out), quoted, i


def _closers(command):
    """{offset of each `(`: offset of the `)` matching it}, quotes ignored, in one pass,
    so finding where an arithmetic `((` ends never rescans the command."""
    out, opened = {}, []
    for j, c in enumerate(command):
        if c == "(":
            opened.append(j)
        elif c == ")" and opened:
            out[opened.pop()] = j
    return out


@lru_cache(maxsize=8)
def scan(command):
    """The heredocs, quoted spans and comments in `command`. Cached because one hook
    call reads the same command through several guards."""
    n = len(command)
    docs, quoted, comments, pending = [], [], [], []
    # Enclosing contexts: "dq" (double quotes), "sub" (`$(` or `(`), "bt" (backticks),
    # each with its start. Inside "dq" only `\`, `"`, `$(` and backticks mean anything.
    stack = []
    closers = _closers(command)

    def arith_end(i):
        """Past the `)` closing the arithmetic `((` at `i`, or None."""
        return closers[i] + 1 if i in closers else None

    i = 0
    while i < n:
        c = command[i]
        top = stack[-1][0] if stack else None
        if top == "dq":
            if c == "\\":
                i += 2
            elif c == '"':
                quoted.append((stack.pop()[1] + 1, i))
                i += 1
            elif command.startswith("$((", i) and (e := arith_end(i + 1)):
                i = e
            elif command.startswith("$(", i):
                stack.append(("sub", i))
                i += 2
            elif c == "`":
                stack.append(("bt", i))
                i += 1
            else:
                i += 1
            continue
        if c == "\\":
            i += 2
        elif c == "'":
            j = command.find("'", i + 1)
            j = n if j < 0 else j
            quoted.append((i + 1, j))
            i = j + 1
        elif command.startswith("$'", i):           # ANSI-C quoting: backslash escapes
            j = i + 2
            while j < n and command[j] != "'":
                j += 2 if command[j] == "\\" else 1
            quoted.append((i + 2, min(j, n)))
            i = j + 1
        elif c == '"':
            stack.append(("dq", i))
            i += 1
        elif c == "`":
            if top == "bt":
                stack.pop()
            else:
                stack.append(("bt", i))
            i += 1
        elif c == "#" and (i == 0 or command[i - 1] in _COMMENT_AFTER):
            j = command.find("\n", i)
            j = n if j < 0 else j
            comments.append((i, j))
            i = j
        elif command.startswith("((", i) and (e := arith_end(i)):
            i = e                                   # `$((1<<2))` and `((x<<=1))` shift
        elif c == "(":
            stack.append(("sub", i))
            i += 1
        elif c == ")":
            if top == "sub":
                stack.pop()
            i += 1
        elif command.startswith("<<<", i):
            i += 3                                  # a herestring has no body
        elif command.startswith("<<", i):
            dash = command.startswith("<<-", i)
            j = i + (3 if dash else 2)
            while j < n and command[j] in " \t":
                j += 1
            word, wq, j = _word(command, j)
            if word:
                pending.append((i, j, word, dash, wq))
            i = max(j, i + 2)
        elif c == "\n" and pending:
            i += 1
            for start, end, word, dash, wq in pending:
                body = i
                while i < n:
                    e = command.find("\n", i)
                    e = n if e < 0 else e
                    line = command[i:e]
                    if (line.lstrip("\t") if dash else line) == word:
                        docs.append(Heredoc(start, end, word, dash, wq, body, i, min(e + 1, n)))
                        i = e + 1
                        break
                    i = e + 1
                else:
                    docs.append(Heredoc(start, end, word, dash, wq, body, n, n))
            pending = []
        else:
            i += 1
    for kind, start in stack:
        if kind == "dq":
            quoted.append((start + 1, n))           # an unclosed quote runs to the end
    for start, end, word, dash, wq in pending:
        docs.append(Heredoc(start, end, word, dash, wq, n, n, n))
    return Scan(tuple(docs), tuple(quoted), tuple(comments))


def heredocs(command):
    """Each heredoc in `command`, in body order."""
    return scan(command).heredocs


def _cut(command, spans):
    out, at = [], 0
    for s, e in sorted(spans):
        if s >= at:
            out.append(command[at:s])
            at = e
    out.append(command[at:])
    return "".join(out)


def strip_heredocs(command):
    """`command` with each heredoc's body and delimiter line removed, its opener kept.

    A heredoc body is data the shell feeds to a program, so an invocation quoted
    inside one is a fixture or a doc example rather than a command. ⚠ Measured on
    a real transcript: the two false positives surviving the position check were
    both test fixture strings inside `python3 - <<'PYEOF'` bodies, and one read as
    command-position because the opening quote attached to the path's first
    component and left the basename bare."""
    return _cut(command, [(d.body_start, d.close) for d in heredocs(command)])


def command_text(command):
    """`command` as the guards that refuse by name read it: a heredoc body fed to a
    program that only stores or prints it (`git commit -F -`, `cat > notes`, `gh ...
    --notes-file -`) is data and dropped, and one whose pipeline runs an interpreter
    (`python3 - <<EOF`, `cat <<EOF | sh`) is code and kept. Release notes that name the
    trust verb are not a trust command; a Python heredoc that opens a rows file is a
    write."""
    return _cut(command, [(d.body_end if code else d.body_start, d.close)
                          for d, code in fed_to(command, _INTERPRETER).items()])


def _blank(command, spans, newlines):
    out = list(command)
    for a, b in spans:
        for k in range(a, b):
            if newlines or out[k] != "\n":
                out[k] = " "
    return "".join(out)


def _bodies(command, s):
    """Each heredoc's body and delimiter line, the delimiter's own newline excluded."""
    return [(d.body_start, d.close - (d.close > d.body_end and command[d.close - 1] == "\n"))
            for d in s.heredocs]


def shell_code(command):
    """`command` with quoted text, heredoc bodies and delimiter lines, and comments
    blanked to spaces, every offset kept, so a scan for shell syntax sees only what the
    shell runs. Quote characters stay, so a quoted argument still reads as one."""
    s = scan(command)
    return _blank(command, list(s.quoted) + list(s.comments) + _bodies(command, s), False)


@lru_cache(maxsize=8)
def without_bodies(command):
    """`command` with heredoc bodies, delimiter lines and comments blanked and quoted
    text kept, every offset kept: the text of a segment `segments` returns."""
    s = scan(command)
    return _blank(command, list(s.comments) + _bodies(command, s), False)


# Where a new simple command starts, and where a new pipeline does: a pipe stays inside a
# pipeline, because `printf ... | tee -a x.csv` is one write, and so does a
# backslash-continued newline.
_COMMAND_SEP = re.compile(r"\|\||&&|[;|&\n()`]")
_PIPELINE_SEP = re.compile(r"\|\||&&|;|(?<!\\)\n")


@lru_cache(maxsize=16)
def segments(command, pipelines=False):
    """[(start, end)] of each simple command in `command`, or with `pipelines` each
    pipeline, split only where the shell splits: a separator inside a quote, a comment
    or a heredoc body splits nothing. Read each span's text off `without_bodies`."""
    s = scan(command)
    view = _blank(command, list(s.quoted) + list(s.comments) + _bodies(command, s), True)
    out, at = [], 0
    for m in (_PIPELINE_SEP if pipelines else _COMMAND_SEP).finditer(view):
        out.append((at, m.start()))
        at = m.end()
    out.append((at, len(command)))
    return tuple(out)


@lru_cache(maxsize=8)
def _pipelines(command):
    spans = segments(command, pipelines=True)
    return [a for a, _ in spans], spans


def pipeline_span(command, pos):
    """(start, end) of the pipeline holding offset `pos`."""
    starts, spans = _pipelines(command)
    return spans[bisect.bisect_right(starts, pos) - 1]


def pipeline_of(command, pos):
    """The text of the pipeline holding offset `pos`, heredoc bodies and comments blanked."""
    a, b = pipeline_span(command, pos)
    return without_bodies(command)[a:b]


def fed_to(command, pattern):
    """{heredoc: whether `pattern` matches the pipeline it opens in}, each pipeline
    searched once, so a line of many openers costs one search, not one per opener."""
    seen, out = {}, {}
    for d in heredocs(command):
        span = pipeline_span(command, d.start)
        if span not in seen:
            seen[span] = bool(pattern.search(without_bodies(command), *span))
        out[d] = seen[span]
    return out
