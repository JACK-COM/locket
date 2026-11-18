"""The files an inline Python program writes: each python -c or heredoc program in a shell
command, folded statically to the paths its open, pathlib and shutil calls reach.
"""
import ast
import os
import re

import shellscan


# A Python program the command runs: `python3 - <<'EOF'` bodies and `python3 -c '…'`.
# Its write targets are read off the program rather than off every `.md` token in it:
# a name inside `.replace("a.md", "b.md")` is data, and `FD / 'X.md'` puts the
# directory in another token, so a token scan both over-fired (two CMP sessions'
# routine edits) and could not see the directory that decides membership.
_PY_WORD = re.compile(r"(?<![\w.-])python[\d.]*(?![\w.-])")


_PY_DASH_C = re.compile(r"""(?<![\w.-])python[\d.]*(?:\s+-[A-Za-z]+)*?\s+-c\s+(?:'([^']*)'|"((?:[^"\\]|\\.)*)")""")


HOLE = "\0"                                     # an f-string value that does not fold


_FOLD_STEPS = 10_000                              # fold calls per program, the whole run


_FOLD_CAP = 64                                   # candidate strings per expression


# The program counts as read only when EVERY call in it is one of these or a known
# writer below; any other call (subprocess, an alias, getattr, an unknown module)
# leaves it unread and the token scan of old answers. ⚠ The first cut listed writers
# only, and an audit found six silent passes through calls it had not listed; this
# is BASH_READERS' inversion, so the next unlisted writer announces itself.
_PY_PURE_BUILTINS = frozenset(
    "len print sorted reversed list dict set frozenset tuple str int float bool bytes "
    "enumerate zip range any all min max sum abs round isinstance issubclass repr format "
    "map filter iter next hasattr type id hash chr ord divmod super".split())


_PY_PURE_MODULES = frozenset(
    "re json textwrap collections itertools functools math string difflib hashlib "
    "fnmatch shlex unicodedata datetime time statistics operator copy pprint typing "
    "dataclasses enum csv glob tomllib yaml os.path".split())


_PY_PURE_FUNCS = frozenset(
    "os.getcwd os.listdir os.walk os.getenv os.environ.get os.scandir os.stat os.remove "
    "os.unlink os.makedirs os.mkdir sys.exit pathlib.Path pathlib.PurePath "
    "pathlib.PosixPath pathlib.Path.home pathlib.Path.cwd Path PurePath".split())


# methods on a value (a string, a list, a path, a match, a handle) that write nothing
# outside a handle; a handle's own destination is read where `open` made it. Deleting
# (`unlink`, `os.remove`) is not writing here, as `rm` is a reader to BASH_READERS.
_PY_PURE_METHODS = frozenset(
    {m for m in dir(str) if not m.startswith("_")} | set(
        "append extend insert pop remove clear sort reverse index count get items keys "
        "values update setdefault add discard union intersection difference copy "
        "read_text read_bytes read readline readlines exists is_file is_dir glob rglob "
        "iterdir stat resolve expanduser absolute with_suffix with_name with_stem "
        "joinpath relative_to as_posix samefile group groups groupdict start end span "
        "sub subn search match fullmatch findall finditer write writelines close flush seek "
        "tell home cwd decode mkdir unlink is_symlink".split()))


# writers: canonical name -> index of the destination argument
_PY_WRITE_FUNCS = {"open": 0, "io.open": 0, "codecs.open": 0, "os.open": 0,
                   "shutil.copy": 1, "shutil.copy2": 1, "shutil.copyfile": 1,
                   "shutil.move": 1, "shutil.copytree": 1, "os.rename": 1,
                   "os.replace": 1, "os.link": 1, "os.symlink": 1}


_PY_ESCAPES = frozenset(("getattr", "eval", "exec", "compile", "__import__", "os.system",
                         "os.popen", "pty.spawn"))


_PY_INTO_DIR = frozenset(("shutil.copy", "shutil.copy2", "shutil.move", "shutil.copytree"))


_PY_WRITE_METHODS = frozenset(("write_text", "write_bytes", "touch", "symlink_to", "hardlink_to"))


_PY_PATH_FUNCS = frozenset(("pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath",
                            "os.path.join", "Path"))


_PY_SAME_FUNCS = frozenset(("str", "os.fspath", "os.path.expanduser", "os.path.abspath",
                            "os.path.realpath", "os.path.normpath"))


def programs(command):
    """[(source, start, end, unreadable)] for each Python program in `command`;
    offsets span the program text. `unreadable` marks a body whose literals are not
    the program's: one the shell rewrites first (an unquoted heredoc or a
    double-quoted `-c` holding `$` or a backtick), or a `-c` body the shell splices
    on with `'\\''`, which the match sees only the first piece of."""
    out = []
    for d, python in shellscan.fed_to(command, _PY_WORD).items():
        if python:
            src = d.body(command)
            raw = command[d.body_start:d.body_end]
            end = d.body_start + len(raw) - raw.endswith("\n")
            out.append((src, d.body_start, end, not d.quoted and any(c in src for c in "$`")))
    for m in _PY_DASH_C.finditer(command):
        g = 1 if m.group(1) is not None else 2
        src = m.group(g)
        spliced = command[m.end():m.end() + 1] not in ("", " ", "\t", "\n", ";", "&", "|", ")")
        out.append((src, m.start(g), m.end(g), spliced or (g == 2 and any(c in src for c in "$`"))))
    return out


class _PyWrites:
    """What a parsed Python program writes: `targets`, the folded destination of every
    writing call (`HOLE` where an f-string value does not fold); `writes`, whether it
    makes any such call; `decided`, whether every destination folded and every call
    is known. Relative targets are joined to each folded `os.chdir` base as well as
    returned bare, since the walk is flow-insensitive.

    A name's values are the union of every binding it gets anywhere in the program;
    any binding that does not fold (a parameter, a call result, an `except … as`, a
    tuple target) leaves the name unknown. Membership turns on the directory and the
    file pattern, so a target whose directory folds is decided even with a hole in
    its file name, and one with a hole in its directory is not."""

    def __init__(self, tree):
        self.steps = 0                      # fold calls this run, against _FOLD_STEPS
        self.binds, self.unknown, self.aliases, self.defs = {}, set(), {}, set()
        self.targets, self.writes, self.decided, self.bases = [], False, True, []
        accepted = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    self._bind(t, n.value, accepted)
            elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and n.value is not None:
                self._bind(n.target, n.value, accepted)
            elif isinstance(n, (ast.For, ast.comprehension)) and isinstance(
                    n.iter, (ast.List, ast.Tuple, ast.Set)) and isinstance(n.target, ast.Name):
                for e in n.iter.elts:
                    self._bind(n.target, e, accepted)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs.add(n.name)
            elif isinstance(n, ast.Import):
                for al in n.names:                        # `import a.b` binds `a`; `as x` binds all of it
                    if al.asname:
                        self.aliases[al.asname] = al.name
                    else:
                        self.aliases[al.name.split(".")[0]] = al.name.split(".")[0]
            elif isinstance(n, ast.ImportFrom) and n.module:
                for al in n.names:
                    self.aliases[al.asname or al.name] = f"{n.module}.{al.name}"
        # every other binding site leaves its name unknown, whatever construct made it
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load) and id(n) not in accepted:
                self.unknown.add(n.id)
            elif isinstance(n, ast.arg):
                self.unknown.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                self.unknown.add(n.name)
            for attr in ("name", "rest"):                 # match captures (3.10+)
                if type(n).__name__ in ("MatchAs", "MatchStar", "MatchMapping") and getattr(n, attr, None):
                    self.unknown.add(getattr(n, attr))
        self._calls(tree)

    def _bind(self, target, value, accepted):
        if isinstance(target, ast.Name):
            self.binds.setdefault(target.id, []).append(value)
            accepted.add(id(target))

    def canon(self, f):
        """The dotted name a callee resolves to through imports (`sh.copy` ->
        `shutil.copy`), `.method` for a method on a value, or None when the callee
        is itself computed."""
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if not isinstance(f, ast.Name):
            return "." + parts[0] if parts else None
        root = f.id
        if root in self.aliases:
            root = self.aliases[root]
        elif parts:
            return "." + parts[0]                         # a method on a variable
        return ".".join([root] + list(reversed(parts)))

    def fold(self, e, seen=()):
        """Set of strings `e` can be, or None when it cannot be known. A name bound
        to itself twice (`x1 = x0 + x0`) doubles the walk per line, so the run stops
        at _FOLD_STEPS and the rest is unknown. No memo: a fold's answer depends on
        `seen`, so one cached per node would answer for the wrong path."""
        self.steps += 1
        if self.steps > _FOLD_STEPS:
            return None
        if isinstance(e, ast.Constant):
            return {e.value} if isinstance(e.value, str) else None
        if isinstance(e, ast.JoinedStr):
            parts = [{str(v.value)} if isinstance(v, ast.Constant) else (self.fold(v.value, seen) or {HOLE})
                     for v in e.values]
            return _cross(parts, lambda a, b: a + b) if parts else {""}
        if isinstance(e, ast.Name):
            if e.id in self.unknown or e.id not in self.binds or e.id in seen:
                return None
            out = set()
            for v in self.binds[e.id]:
                f = self.fold(v, seen + (e.id,))
                if f is None:
                    return None
                out |= f
            return out if len(out) <= _FOLD_CAP else None
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Div, ast.Add)):
            return _cross([self.fold(e.left, seen), self.fold(e.right, seen)],
                          join_path if isinstance(e.op, ast.Div) else (lambda a, b: a + b))
        if isinstance(e, ast.Attribute) and e.attr == "parent":
            v = self.fold(e.value, seen)
            return None if v is None else {os.path.dirname(x.rstrip("/")) for x in v}
        if isinstance(e, ast.Call) and not e.keywords:
            name = self.canon(e.func) or ""
            if name in _PY_PATH_FUNCS:
                return _cross([self.fold(a, seen) for a in e.args], join_path) if e.args else {"."}
            if name == "pathlib.Path.home" or (name == ".home" and not e.args):
                return {"~"}
            if name in _PY_SAME_FUNCS and len(e.args) == 1:
                return self.fold(e.args[0], seen)
            if name in (".expanduser", ".resolve", ".absolute") and not e.args:
                return self.fold(e.func.value, seen)
        return None

    def _calls(self, tree):
        callees = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
        for n in ast.walk(tree):
            # a writer handed on as a value (`o = open`, `map(open, …)`) is called unseen
            if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Load) \
                    and id(n) not in callees:
                name = self.canon(n) if isinstance(n, ast.Attribute) else self.aliases.get(n.id, n.id)
                if name in _PY_WRITE_FUNCS or (name or "").lstrip(".") in _PY_WRITE_METHODS:
                    self.decided, self.writes = False, True
            if not isinstance(n, ast.Call):
                continue
            name = self.canon(n.func)
            dest = None
            if name is None:
                self._unknown(n, n.func)                  # `x[0]()`, `f()()`
                continue
            if isinstance(n.func, ast.Name) and n.func.id in self.defs:
                continue                                  # its body is walked on its own
            if name in _PY_ESCAPES or name.startswith(("os.exec", "os.spawn", "subprocess.")):
                self._unknown(n)                          # a shell or dynamic call: write if it names one
                continue
            if isinstance(n.func, ast.Name) and (n.func.id in self.binds or n.func.id in self.unknown) \
                    and n.func.id not in self.aliases:
                self._unknown(n)                          # a variable called: unknown callee
                continue
            if name in ("os.chdir",):
                v = self.fold(n.args[0]) if n.args else None
                if v is None or any(HOLE in x for x in v):
                    self.decided = False
                else:
                    self.bases.extend(v)
                continue
            method = name[1:] if name.startswith(".") else None
            if method in _PY_WRITE_METHODS:
                dest = n.func.value
            elif method in ("rename", "replace") and len(n.args) + len(n.keywords) == 1 \
                    and (n.args or n.keywords[0].arg == "target"):
                dest = self._arg(n, 0, "target")          # Path.rename(dst); str.replace takes two
            elif method == "open" or name in ("open", "io.open", "codecs.open"):
                builtin = method is None
                path = (self._arg(n, 0, "file") or self._arg(n, 0, "filename")) if builtin else n.func.value
                mode = self._arg(n, 1 if builtin else 0, "mode")
                m = {"r"} if mode is None else self.fold(mode)
                if m is None or any(set(x) & set("wax+") for x in m):
                    dest = path
            elif name in _PY_WRITE_FUNCS:
                dest = self._arg(n, _PY_WRITE_FUNCS[name], "dst" if _PY_WRITE_FUNCS[name] else "path")
                if name in _PY_INTO_DIR:                  # `copy(src, dir)` lands at dir/basename(src)
                    self._into_dir(dest, self._arg(n, 0, "src"), tree=name.endswith("copytree"))
            elif (method == "copy" and len(n.args) >= 2) or not (
                    method in _PY_PURE_METHODS or name in _PY_PURE_BUILTINS
                    or name in _PY_PURE_FUNCS or ("." in name.lstrip(".") and name.rsplit(".", 1)[0] in _PY_PURE_MODULES)):
                self._unknown(n)                          # `x.copy(a, b)` is not dict.copy()
                continue
            if dest is None:
                continue
            self.writes = True
            v = self.fold(dest)
            if v is None:
                self.decided = False
                continue
            for x in v:
                head = x.rsplit("/", 1)[0] if "/" in x else ""
                if x.startswith(HOLE) or HOLE in head:
                    self.decided = False                  # the directory itself is unknown
                else:
                    self.targets.append(x)

    def _mdish(self, e, seen=()):
        """Whether expression `e` can evaluate to a `.md` path: a literal or f-string
        ending in `.md`, a join or concatenation ending in one, a path constructor over
        one, a collection or subscript holding one, or a name any binding of which can.
        A call's RESULT is not its argument: `Path('MEMORY.md').read_text()` is the
        file's text, and reading it as a path re-fired the routine transform."""
        if isinstance(e, ast.Constant):
            return isinstance(e.value, str) and e.value.rstrip().endswith(".md")
        if isinstance(e, ast.JoinedStr):
            return bool(e.values) and self._mdish(e.values[-1], seen)
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Div)):
            return self._mdish(e.right, seen)
        if isinstance(e, (ast.List, ast.Tuple, ast.Set)):
            return any(self._mdish(x, seen) for x in e.elts)
        if isinstance(e, ast.Dict):
            return any(self._mdish(x, seen) for x in e.values if x is not None)
        if isinstance(e, ast.Subscript):
            return self._mdish(e.value, seen)
        if isinstance(e, ast.Starred):
            return self._mdish(e.value, seen)
        if isinstance(e, ast.Name):
            return e.id in self.binds and e.id not in seen and any(
                self._mdish(v, seen + (e.id,)) for v in self.binds[e.id])
        if isinstance(e, ast.Call):
            name = self.canon(e.func) or ""
            if name in _PY_PATH_FUNCS or name in _PY_SAME_FUNCS:
                return bool(e.args) and self._mdish(e.args[-1], seen)
            if name in (".expanduser", ".resolve", ".absolute", ".with_name", ".with_suffix"):
                return True if name == ".with_suffix" and e.args and self._mdish(e.args[0], seen) \
                    else self._mdish(e.func.value, seen)
        return False

    def _unknown(self, call, *extra):
        """An unknown call: a write only where its arguments (or receiver) reach a `.md`
        string. One that never does is neutral whatever its name, so `datetime.now()`,
        `sys.stdout.write` and `argparse` stay quiet while `mylib.save('MEMORY.md')`,
        `os.truncate('MEMORY.md', 0)` and `fileinput.input(..., inplace=True)` warn.
        A write computed with no `.md` text in sight is unseen here and at the token
        scan alike."""
        parts = [*call.args, *(k.value for k in call.keywords), *extra]
        if isinstance(call.func, ast.Attribute):
            parts.append(call.func.value)
        if any(self._mdish(p) for p in parts):
            self.decided, self.writes = False, True

    def _into_dir(self, dst, src, tree=False):
        """Targets of a copy or move whose destination may be a directory."""
        d, s = (self.fold(dst) if dst is not None else None), (self.fold(src) if src is not None else None)
        if d is None:
            return
        for x in d:
            if x.endswith(".md") or HOLE in x:
                continue                                  # a file destination, or a directory unknown
            if tree:
                self.targets.append(join_path(x, HOLE + ".md"))
            elif s is None:
                self.targets.append(join_path(x, HOLE + ".md"))
            else:
                self.targets.extend(join_path(x, os.path.basename(y)) for y in s)

    @staticmethod
    def _arg(call, i, kw):
        for k in call.keywords:
            if k.arg == kw:
                return k.value
        return call.args[i] if len(call.args) > i else None


def _cross(sets, op):
    """Every `op`-combination of one string from each set, or None past the cap."""
    if any(s is None for s in sets):
        return None
    out = set(sets[0])
    for s in sets[1:]:
        out = {op(x, y) for x in out for y in s}
        if len(out) > _FOLD_CAP:
            return None
    return out


def join_path(a, b):
    return b if b.startswith(("/", "~")) else (a.rstrip("/") + "/" + b if a else b)


def writes(source):
    """`_PyWrites` for a program, or None when it does not parse or is too deep to
    walk (a RecursionError from a long `+` chain is undecided, never a crash)."""
    try:
        return _PyWrites(ast.parse(source))
    except (SyntaxError, ValueError, RecursionError):
        return None
