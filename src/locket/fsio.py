"""File and stream I/O that a store's content cannot subvert: bounded reads that refuse
links, FIFOs and hard links, atomic replacement, and output stripped of control and
invisible characters.
"""
import os
import stat
import sys
import contextlib
import tempfile
import urllib.parse
from pathlib import Path
from typing import IO, Any, Iterator, Optional, TextIO, Union

import _guard

# What every path parameter here takes: a string or anything `os.fspath` reads.
StrPath = Union[str, "os.PathLike[str]"]


def inode(p: StrPath) -> Optional[tuple[int, int]]:
    """(st_dev, st_ino) of a path, or None. Identity for trust: `Path` equality
    calls `/users` and `/Users` two directories on a case-insensitive disk."""
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        return None
    return st.st_dev, st.st_ino


# Text made safe to print lives in panoply-lib's guard, shared with Augur and Grille.
_INVISIBLE = _guard.INVISIBLE
relay = _guard.relay
safe = _guard.safe


# argparse colours help on a terminal from Python 3.14; SafeOut strips the ESC of each
# colour code and leaves `[1;34m` on screen. Every parser takes this, and its subparsers inherit it.
PLAIN_HELP: dict = {"color": False} if sys.version_info >= (3, 14) else {}


class SafeOut:
    """A text stream that passes every line through `relay(lines=True)`'s character
    rule: whatever a mode prints (a title, a file name, a history line from a store)
    reaches a terminal or an agent with no control, bidi or invisible character."""

    def __init__(self, stream: TextIO):
        self._s = stream

    def write(self, text: object) -> int:
        return self._s.write(safe(text))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._s, name)


def same(a: StrPath, b: StrPath) -> bool:
    """Whether two paths name one directory: by inode where both exist, else by
    resolved path. Spelling is not identity on a case-insensitive disk."""
    ia, ib = inode(a), inode(b)
    if ia is not None and ib is not None:
        return ia == ib
    pa, pb = Path(a).expanduser().resolve(), Path(b).expanduser().resolve()
    if pa == pb:
        return True
    # not yet on disk: the same name under one directory, case aside, is taken as one
    # file, since on a case-insensitive disk it is, and a gate errs toward refusing
    return (pa.name.casefold() == pb.name.casefold() and pa.parent != pa
            and same(pa.parent, pb.parent))


MAX_FILE_BYTES = 8 * 1024 * 1024      # past the 5 MB a declared CSV is reported at


MAX_SCAN = 64 * 1024                  # model-written text a pattern is run over, at most


PATH_MAX = 4096                       # Linux's; longer names no file on any host we serve


MAX_STORE_FILES = 4000


MAX_STORE_BYTES = 64 * 1024 * 1024


def regular(p: StrPath) -> bool:
    """A plain file, by `lstat`: no symlink, no FIFO, no device, one hard link. A
    store is read only through files it holds itself, so a link planted in a cloned
    repository cannot make the walk read a key or a config from elsewhere on disk."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_nlink == 1


def open_safe(path: StrPath, binary: bool = False, newline: Optional[str] = None,
              max_bytes: Optional[int] = MAX_FILE_BYTES) -> IO[Any]:
    """`open` for a file a store controls, checked on the descriptor that is read:
    refused unless it is a regular file of one link under `max_bytes` (None for no
    cap), so a swap between the walk's check and this read gains nothing, and a FIFO
    never blocks it. `newline=""` for a CSV; the default translates CRLF as
    `read_text` did. Raises OSError on refusal."""
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1
                or (max_bytes is not None and st.st_size > max_bytes)):
            raise OSError(f"{path}: not a plain file" + (f" under {max_bytes} bytes" if max_bytes else ""))
        return (os.fdopen(fd, "rb") if binary
                else os.fdopen(fd, "r", newline=newline, errors="ignore", encoding="utf-8"))
    except BaseException:
        os.close(fd)
        raise


def read_safe(path: StrPath) -> str:
    """A corpus file's text by `open_safe`; raises OSError on refusal."""
    with open_safe(path) as f:
        return f.read()


def read_or_empty(path: StrPath) -> str:
    """`read_safe`, or "" for a file that is gone, unreadable or refused."""
    try:
        return read_safe(path)
    except OSError:
        return ""


def replace_atomic(path: StrPath, data: Union[str, bytes], prefix: str, mode: Optional[int] = None) -> None:
    """Write `data` (str or bytes) beside `path` and rename it over, so a reader never sees
    a partial file. The temp is removed when the write fails; a writer killed mid-way can
    still leave one, which each surface's own sweep removes."""
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), prefix=prefix)
    try:
        with os.fdopen(fd, "wb" if isinstance(data, bytes) else "w") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def sqlite_ro(db: StrPath, timeout: float = 1.0) -> Iterator[Any]:
    """A read-only connection to the SQLite file `db`, closed on exit. The path is
    percent-quoted into the URI, so a `?` or `#` in it cannot pass as a URI parameter."""
    import sqlite3
    con = sqlite3.connect("file:" + urllib.parse.quote(str(db)) + "?mode=ro", uri=True, timeout=timeout)
    try:
        yield con
    finally:
        con.close()


def sqlite_rows(db: StrPath, sql: str, params: tuple = (), timeout: float = 1.0) -> Optional[list[tuple]]:
    """Every row `sql` returns from `db` opened read-only, or None on any failure: no
    sqlite3 in this Python, no file, a lock, a missing table, a drifted schema. Never
    raises, because its callers run inside hooks, where a raise lets a write through."""
    try:
        with sqlite_ro(db, timeout) as con:
            return con.execute(sql, params).fetchall()
    except Exception:
        return None
