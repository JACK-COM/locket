"""The files a shell command writes: redirections, tee, in-place sed, git and the other
writers, each resolved against the directory the command has cd'd into by then, plus the
commands whose writes cannot be read off the text.
"""
import os
import re
import shlex
from functools import lru_cache
from pathlib import Path

import corpora
import fsio
import paths
import pywrites
import shellscan


# Destination-position writes. Direction is the whole point: a corpus path as the
# SOURCE of a cp is a read, and blocking that would break every
# backup-before-rewrite command this discipline actually requires.
BASH_WRITE = (
    re.compile(r">>?\s*(?P<p>[^\s;&|<>]+\.md)"),
    # the path starts a word (after a space, a quote or `=`): starting it mid-word made one
    # 32 KB argument cost seconds of backtracking inside a single match attempt
    re.compile(r"\btee\b[^;&|]*?(?<![^\s'\"=])(?P<p>[^\s;&|<>]+\.md)"),
    # `-i` in a flag cluster of its own (`-i`, `-Ei`, `-i.bak`, `-ibak`, `--in-place`), never
    # the `-i` inside `make-it-yours.md`, which read a `sed -n` as a write to `t-yours.md`
    re.compile(r"\bg?sed\b[^;&|]*?(?<!\S)-(?:[nEsuzr]*i|-in-place)[^;&|]*?(?<![^\s'\"])(?P<p>[^\s;&|<>]+\.md)"),
    # perl and ruby share sed's `-i` (`-pi`, `-i.bak`, `-0pi`); gawk spells it `-i inplace`
    # only argument-free flags before the `i`, as in _INPLACE: `-MList::Util` and `-CSDio` are not it
    re.compile(r"\b(?:perl|ruby)\b[^;&|]*?(?<!\S)-[0-9lnpaWwsSTtUuXc]*i[^;&|]*?(?<![^\s'\"])(?P<p>[^\s;&|<>]+\.md)"),
    re.compile(r"\bg?awk\b[^;&|]*?(?<!\S)-i\s*inplace\b[^;&|]*?(?<![^\s'\"])(?P<p>[^\s;&|<>]+\.md)"),
    re.compile(r"\b(?:cp|mv|rsync)\b[^;&|]+?\s(?P<p>[^\s;&|<>]+\.md)\s*(?:$|[;&|])"),
)


# Where each pattern above may start: the command word, or None for a redirect. A
# pattern is tried from its first start in a clause and nowhere after a miss there,
# because a lazy span retried from every `sed -i` in one clause cost 9.5 s on 4 KB:
# a match from a later start is also a match from the first, so one miss settles it.
BASH_WRITE_STARTS = (None, re.compile(r"\btee\b"), re.compile(r"\bg?sed\b"),
                     re.compile(r"\b(?:perl|ruby)\b"), re.compile(r"\bg?awk\b"),
                     re.compile(r"\b(?:cp|mv|rsync)\b"))


_CLAUSE = re.compile(r"[^;&|]+")


def write_matches(pattern, start, command):
    """`pattern.finditer(command)` in linear time: per clause, matched from the first
    `start` word, then from the next one after each hit, never after a miss."""
    if start is None:
        yield from pattern.finditer(command)
        return
    for clause in _CLAUSE.finditer(command):
        cur, end = clause.start(), clause.end()
        while True:
            s0 = start.search(command, cur, end)
            if not s0:
                break
            m = pattern.match(command, s0.start(), end)
            if not m:
                break
            yield m
            cur = m.end()


# Destination computed at run time, so it cannot be read off the command text.
# Warn rather than block: a false block here costs more than a missed one.
BASH_WRITE_OPAQUE = re.compile(
    r"write_text\(|open\([^)]*['\"][wa]['\"]|\.writelines\(|shutil\.(?:copy|move)")


# Commands known to leave a file argument unwritten, unless an in-place flag
# (_INPLACE) or a writing git subcommand (_GIT_WRITES) says otherwise. A corpus path handed to any
# other command warns. ⚠ BASH_WRITE alone is a closed list of writers, and
# `perl -pi` passed it silently for a week of real edits; listing the readers
# instead makes the next unlisted writer announce itself, at the cost of a
# warning on an unlisted reader.
BASH_READERS = frozenset((
    "cat head tail less more bat glow nl grep egrep fgrep rg ag ack wc diff cmp comm "
    "ls stat file find du tree test [ realpath readlink basename dirname md5 md5sum shasum "
    "sha256sum sort uniq cut column fold jq yq strings xxd od hexdump echo printf open "
    "git gh locket memscan memfind sed gsed awk gawk rm trash touch mkdir chmod "
    # directional: BASH_WRITE reads which argument is the destination
    "cp mv rsync tee").split())


# Flags whose value a command only reads, keyed by the command they belong to: the file
# after one of these is an input whatever else the command writes. Scoped per command,
# never a blanket reader entry, because the same CLI has writing flags (`claude
# --debug-file`). `ollama launch <cmd>` is judged as `<cmd>`.
_READ_FLAGS = {"claude": frozenset(("--append-system-prompt-file", "--system-prompt-file"))}


# Words that run the command after them; the word after is the one judged.
_SHELL_PREFIXES = frozenset("sudo env command nohup time exec builtin nice xargs caffeinate "
                            "timeout stdbuf ionice setsid "
                            "do then else if while until ! {".split())


def expand_home(tok):
    """A shell path with its home spelling expanded: `~`, `$HOME` and `${HOME}`,
    the forms a command writes a memory path in most often, which a literal
    `startswith("/")` test read as relative and resolved to nothing."""
    tok = tok.replace('"', "").replace("'", "")     # `"$HOME/dir"/f.md` quotes mid-token
    for pre in ("${HOME}", "$HOME"):
        if tok == pre or tok.startswith(pre + "/"):
            tok = str(paths.HOME) + tok[len(pre):]
    # os.path, not Path: an unknown `~user` stays literal, as the shell leaves it,
    # where Path.expanduser raises and silently disables the guard for the command
    return os.path.expanduser(tok) if tok.startswith("~") else tok


# `cd` or `pushd` in command position only; the argument runs to the next separator,
# so `"$(mktemp -d)"` is one argument
_CD = re.compile(r"(?:^|(?<=[;&|({]))[ \t]*(cd|pushd|popd)(?=[ \t;&|)\n]|$)(?:[ \t]+([^;&|\n)]*?))?[ \t]*(?=$|[;&|\n)])",
                 re.M)


def cd_targets(command, cwd, pos=None):
    """The directory a bare filename at `pos` resolves against, as a one-item list, or []
    when it cannot be known.

    The shell resolves against the last `cd` before the name, so that is the one
    taken: resolving against the session cwd first read `cd /elsewhere && sed -i … x.md`
    as a write to the session's own `x.md`. A `cd` whose target is computed at run time
    (`cd "$(mktemp -d)"`) makes every later bare name unknowable, and an unknowable
    target is not claimed as a corpus write. The scan runs over `_shell_code`: a
    `cd /elsewhere` inside a string, a comment or a heredoc body moved nothing, and
    reading it as a move let a real write into the cwd's corpus pass silently."""
    times, dirs = _cd_timeline(command, cwd)
    if pos is None:
        here = dirs[-1]
    else:
        import bisect
        here = dirs[bisect.bisect_right(times, pos - 1)]   # the last cd starting before pos
    return [here] if here else []


@lru_cache(maxsize=8)
def _cd_timeline(command, cwd):
    """(starts, dirs): `dirs[0]` is the cwd and `dirs[k]` the directory after the k-th
    `cd`, which starts at `starts[k - 1]`. Built once per command, because resolving
    each path by rescanning the whole command made a long one quadratic."""
    here, times, dirs = cwd, [], [cwd]
    for m in _CD.finditer(shellscan.shell_code(command)):
        if m.group(1) == "popd":
            here = None                          # the stack is not tracked
        else:
            d = command[m.start(2):m.end(2)].strip() if m.group(2) is not None else ""
            d = d or None
            if d is None or d == "~":
                here = str(paths.HOME)
            elif any(c in d for c in "$`(") and not d.strip("'\"").startswith(("$HOME", "${HOME}")):
                here = None
            elif d == "-":
                here = None                      # the previous directory; not tracked
            else:
                d = expand_home(d)
                here = d if d.startswith("/") else (str(Path(here) / d) if here else None)
            if here and len(here) > fsio.PATH_MAX:
                here = None                      # no such directory: `cd x; ` repeated, not a path
        times.append(m.start())
        dirs.append(here)
    return times, dirs


# An in-place flag read off the tokens, so a `;` inside the program text (`sed -i
# 's/a/b/; s/c/d/'`, which stops BASH_WRITE's clause scan) cannot hide it.
# Only flags that take no argument may precede the `i`: `-fimport.sed` and `-ne'print'`
# glue an argument on, and the `i` inside it is not the flag.
_INPLACE = {"sed": r"-[nEsuzr]*i.*|--in-place.*", "gsed": r"-[nEsuzr]*i.*|--in-place.*",
            "perl": r"-[0-9lnpaWwsSTtUuXc]*i.*", "ruby": r"-[0-9lnpaWwsSTtUuXc]*i.*"}


# git subcommands that write the paths they are handed; the rest of git reads
_GIT_WRITES = frozenset("checkout restore mv rm reset stash apply am merge rebase cherry-pick "
                        "revert pull switch clean checkout-index read-tree sparse-checkout "
                        "submodule worktree bisect".split())


_GIT_VALUED = frozenset(("-C", "-c", "--git-dir", "--work-tree", "--namespace"))   # take the next word


def _unlisted_commands(command):
    """[(command word, `.md` argument, offset, in_place)] for every simple command whose
    word is not in BASH_READERS, or that runs with an in-place flag, or is a git
    subcommand that writes. `in_place` marks a write decidable from the tokens. Heredoc bodies are data and are skipped, and so are a
    quoted string holding whitespace (prose, a program), a glob, a redirect target
    (BASH_WRITE reads those), a comment, and a `for`/`case` header. A command the
    tokenizer cannot split (an unbalanced quote) yields nothing: that line of shell
    would not run either."""
    text = shellscan.strip_heredocs(command).replace("\\\n", " ").replace("\n", " ; ")
    text = text.replace("\\(", " ").replace("\\)", " ")    # `find … \( -o \)` groups, not subshells
    text = text.replace("`", " ; ")                       # a backtick substitution is its own command
    lex = shlex.shlex(text, posix=True, punctuation_chars=True)
    lex.whitespace_split, lex.commenters = True, ""      # `a#b` is a word, `#` alone a comment
    try:
        toks = list(lex)
    except ValueError:
        return []
    out, word, skip, cursor, redirect, inplace, prev = [], None, False, 0, False, False, ""
    for t in toks:
        # `<(` and `$(` arrive fused: a process or command substitution opens a new command
        # only a token of shell punctuation: a quoted pattern like "(a|b)" is an argument
        # a bare `<`/`>` is a redirect, below; `<(` and `>(` arrive fused and open a command
        if t and set(t) <= set("();|&<>$") and ("<" not in t and ">" not in t or "(" in t or ")" in t):
            word, skip, redirect, inplace = None, False, False, False
            continue
        if t in ("-exec", "-execdir", "-ok", "-okdir"):    # find runs the command after it
            word, inplace = None, False
            continue
        if skip:
            continue
        if t.startswith("#"):
            skip = True                                  # to the end of the line
            continue
        if set(t) <= set("<>&0123456789") and ("<" in t or ">" in t):
            redirect = True
            continue
        if redirect:
            redirect = False
            continue
        if word is None:
            if (t in _SHELL_PREFIXES or "=" in t.split("/")[0] or t.startswith("-")
                    or re.fullmatch(r"\d+[smhd]?", t)):         # `timeout 5`, `nice -n 5`
                continue
            if t in ("for", "case", "select", "in", "function"):
                skip = True
                continue
            word, prev = os.path.basename(t), ""
            continue
        if word in _INPLACE and re.fullmatch(_INPLACE[word], t):
            inplace = True
        elif word in ("awk", "gawk") and (t in ("-iinplace", "--include=inplace")
                                          or (prev in ("-i", "--include") and t == "inplace")):
            inplace = True
        elif word == "git" and not t.startswith("-") and prev not in _GIT_VALUED:
            word = f"git {t}"                            # judged by its subcommand
        elif word == "ollama" and t == "launch":
            word = None                                  # the launched command is the one judged
            continue
        flag, prev = prev, t
        if not t.endswith(".md") or any(c in t for c in "*?[ \t") or "://" in t:
            continue
        reads = _READ_FLAGS.get(word, ())
        if flag in reads or t.partition("=")[0] in reads:
            continue                                     # the value of a read-only flag
        reader = word in BASH_READERS or (word.startswith("git ") and word[4:] not in _GIT_WRITES)
        if reader and not inplace:
            continue
        at = command.find(os.path.basename(t), cursor)
        cursor = at if at >= 0 else cursor
        out.append((word, t, cursor, inplace))
    return out


def bash_write_targets(command, cwd):
    """(blocking, opaque, unlisted) corpus files a Bash command appears to write;
    `unlisted` pairs each with the command word outside BASH_READERS that takes it.

    Resolves a bare `MEMORY.md` against the payload cwd, or the last `cd` before it
    in the command, because that is the shape a heredoc rewrite actually takes.
    Resolving rather than name-matching is why the same command is a corpus write
    from inside a corpus and nothing at all from outside one.

    A path the shell expands at run time (any variable but HOME, a substitution)
    cannot be decided from the command, so it never blocks: it is expanded against
    this process's environment and, if that lands in a corpus, warns. Read
    literally, `"$TMPDIR/x.md"` is relative and joins the cwd, which would block
    every temp-file heredoc written from inside a corpus.
    """
    def resolve(tok, pos):
        """(path, decidable) for the corpus file `tok` names, or None."""
        tok = expand_home(tok)
        decidable = not any(c in tok for c in "$`")
        if not decidable:
            tok = os.path.expandvars(tok)
            if any(c in tok for c in "$`"):          # unset, or a substitution
                return None
        cands = ([tok] if tok.startswith("/")
                 else [str(Path(r) / tok) for r in cd_targets(command, cwd, pos)])
        for c in cands:
            if corpora.corpus_for(c) is not None:
                return Path(c), decidable
        return None

    blocking: list[str] = []
    opaque: list[str] = []
    for pat, start in zip(BASH_WRITE, BASH_WRITE_STARTS):
        for m in write_matches(pat, start, command):
            hit = resolve(m.group("p"), m.start("p"))
            if hit:
                (blocking if hit[1] else opaque).append(hit[0])

    def flag(hit):
        if hit and hit[0] not in blocking and hit[0] not in opaque:
            opaque.append(hit[0])
    # A Python program that parses is read on its own, gate or no gate, so a writer the
    # gate's regex does not spell (`write_bytes`, mode `'wb'`) is still seen. Where every
    # destination folds and every call is known, its folded targets are the answer;
    # otherwise it gets the token scan below, as does all text outside a read program.
    chars = list(command)
    for src, start, end, unreadable in pywrites.programs(command):
        w = None if unreadable else pywrites.writes(src)
        if w is None:
            continue                            # unread: its tokens stay in `rest`
        chars[start:end] = " " * (end - start)
        for t in w.targets:
            # a hole at the end may still be `.md`; inside the name it stands for any text
            probe = (t + ".md" if t.endswith(pywrites.HOLE) else t).replace(pywrites.HOLE, "*")
            if not probe.endswith(".md"):
                continue
            flag(resolve(probe, start))
            if not probe.startswith(("/", "~")):
                for base in w.bases:            # after an `os.chdir` the program's cwd is that
                    flag(resolve(pywrites.join_path(base, probe), start))
        if not w.decided and (w.writes or BASH_WRITE_OPAQUE.search(src)):
            for m in re.finditer(r"[\w./~-]+\.md", src):
                # `f'{d}/MEMORY.md'` scans as `/MEMORY.md`: an unknown directory may be the cwd's
                for tok in {m.group(), os.path.basename(m.group())}:
                    flag(resolve(tok, start + m.start()))
                    for base in w.bases:
                        flag(resolve(pywrites.join_path(base, tok), start + m.start()))
    rest = "".join(chars)
    if BASH_WRITE_OPAQUE.search(rest):
        for m in re.finditer(r"[\w./~-]+\.md", rest):
            flag(resolve(m.group(), m.start()))
    unlisted = []
    for word, tok, pos, inplace in _unlisted_commands(command):
        hit = resolve(tok, pos)
        if not hit or hit[0] in blocking:
            continue
        if inplace and hit[1]:                  # a literal path, edited in place: decidable
            blocking.append(hit[0])
        elif hit[0] not in opaque:
            unlisted.append((word, hit[0]))
    return blocking, opaque, unlisted
