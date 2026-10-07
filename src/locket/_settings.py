# GENERATED from panoply-lib/settings.py (3d0e3b4) by sync.sh: edit the source and rerun sync.sh, never this copy.
"""settings: the Panoply's one settings file, ~/.panoply/config.json.

Holds the settings two or more pieces share at the top level, and each piece's overrides
of them in a section named for the piece:

    {"embed":  {"model": "embeddinggemma-2:270m", "venv": "~/.panoply/venv"},
     "grille": {"embed": {"model": "nomic-embed-text"}}}

A piece reads a key from its own section first, then the top level; the environment beats
both and the shipped default comes last, which is the caller's business (embed.apply_config
takes `layers(...)` as it stands). Settings only one piece uses stay in that piece's own
file. PANOPLY_CONFIG names another file, for a test or a second setup.

Reads never fail: a missing file is no settings, an unreadable one is no settings and a
finding, because a hook must not stop over a settings file. A structural finding (a wrong
type, a key this version does not know, which a newer piece may have written) informs and
never blocks a write; only an unreadable file does, since writing over it would lose every
other piece's settings. Writes go through `update`, which holds a lock across the
read-modify-write and replaces the file whole. POSIX only: the lock is fcntl's.
"""
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

try:                    # a panoply-lib copy, imported as a package member, flat, or from panoply-lib itself
    from . import _manifest_schema as _schema
except ImportError:
    try:
        import _manifest_schema as _schema
    except ImportError:
        import manifest_schema as _schema

# Every piece that may hold a section of overrides. A new piece adds its name here, or its
# section reads as unchecked to the others.
PIECES = ("locket", "grille")
LOCK_WAIT = 10

# The shared sections, which may appear at the top level or inside a piece's section.
SECTIONS = {
    "embed": {
        "type": "object", "additionalProperties": False,
        "description": "The embedder the pieces rank by meaning with. An environment variable beats it.",
        "properties": {
            "model": {"type": "string", "description": "The ollama embedding model, as `ollama list` names it. "
                      "Default: embeddinggemma-2:270m. Overridden by MEMFIND_MODEL."},
            "ollama_host": {"type": "string", "description": "Where ollama answers, with its scheme. "
                            "Default: http://127.0.0.1:11434. Overridden by OLLAMA_HOST."},
            "autostart": {"type": "boolean", "description": "Start `ollama serve` when the port is closed. "
                          "Default: true. MEMFIND_NO_AUTOSTART=1 turns it off."},
            "venv": {"type": "string", "description": "An absolute path to the virtualenv holding onnxruntime "
                     "and tokenizers for the in-process rung. Default: ~/.panoply/venv, or an older "
                     "~/.locket/venv. Overridden by PANOPLY_VENV, then LOCKET_VENV."},
        },
    },
}


def path():
    """The settings file, read from the environment on every call so a test can move it."""
    return Path(os.environ.get("PANOPLY_CONFIG") or Path.home() / ".panoply" / "config.json").expanduser()


def schema():
    piece = {"type": "object", "additionalProperties": False, "properties": SECTIONS}
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "panoply.config.schema.json",
        "title": "Panoply settings (~/.panoply/config.json)",
        "description": "Settings the Panoply's pieces share, and each piece's overrides of them in a "
                       "section named for the piece. A piece's own key beats the shared one.",
        "type": "object",
        # an object this version does not know is a newer piece's section, never a problem
        "additionalProperties": {"type": "object"},
        "properties": {
            "$schema": {"type": "string", "description": "Editor hint only; ignored."},
            **SECTIONS,
            **{p: {**piece, "description": f"{p}'s own overrides of the shared settings."} for p in PIECES},
        },
    }


def load(p=None, check=True):
    """(settings, findings). A missing file is empty settings and no finding; an unreadable
    one is empty settings and one finding that `unreadable` recognises. A file with
    structural problems still returns its settings, since a reader takes each key on its
    own merits. `check=False` skips the schema, for a hook that discards the findings."""
    p = Path(p) if p else path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, []
    except (OSError, ValueError) as e:
        return {}, [f"{p}: unreadable ({e})"]
    if not isinstance(data, dict):
        return {}, [f"{p}: unreadable (expected an object, got {type(data).__name__})"]
    return data, (_schema.check_value(data, schema(), str(p)) if check else [])


def unreadable(findings):
    """Whether `load` found the file unreadable, which is the one finding that blocks a write."""
    return any(": unreadable (" in f for f in findings)


def layers(piece, section, data=None):
    """[(label, dict)] for one shared section as `piece` reads it, highest first: its own
    overrides, labelled with its name, then the top level, labelled "global". A layer
    that is not an object is left out."""
    data = load(check=False)[0] if data is None else data
    own = data.get(piece)
    out = [(piece, own.get(section) if isinstance(own, dict) else None), ("global", data.get(section))]
    return [(label, s) for label, s in out if isinstance(s, dict)]


def overriding(data, section, key, usable=None):
    """The pieces whose own section sets `key`, so a global change does not reach them.
    `usable(key, section)` says whether a value takes effect; one that does not falls
    through to the global value, so it overrides nothing."""
    return [p for p, s in data.items() if p in PIECES and isinstance(s, dict)
            and isinstance(s.get(section), dict) and key in s[section]
            and (usable is None or usable(key, s[section]))]


def edit(data, section, changes=None, clear=(), reset=False, piece=None):
    """A copy of `data` with `section` changed at the top level, or inside `piece`'s section
    when one is named: `reset` empties it first, `clear` removes keys, `changes` sets them.
    A section or piece left empty is removed, so the file holds only what was set."""
    new = json.loads(json.dumps(data))
    holder = new
    if piece:
        holder = new[piece] if isinstance(new.get(piece), dict) else {}
    sec = {} if reset or not isinstance(holder.get(section), dict) else dict(holder[section])
    for k in clear:
        sec.pop(k, None)
    sec.update(changes or {})
    if sec:
        holder[section] = sec
    else:
        holder.pop(section, None)
    if piece:
        if holder:
            new[piece] = holder
        else:
            new.pop(piece, None)
    return new


def write_json(p, obj):
    """Whole or not at all: a temporary file beside the target, synced, renamed over it.
    A symlinked file keeps its link (the file it points at is replaced) and its mode."""
    target = Path(os.path.realpath(p))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        mask = os.umask(0)
        os.umask(mask)
        mode = 0o666 & ~mask
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def lock(p=None):
    """Held across a read-modify-write, so two pieces configuring at once cannot each read
    the old file and the second drop the first's change. Keyed on the file's real path, so
    two links to one file share it. Raises TimeoutError past LOCK_WAIT. The lock file is
    never removed: flock releases with its holder, and deleting a held lock file would let
    a second process lock a fresh one."""
    import fcntl
    p = Path(os.path.realpath(Path(p) if p else path()))
    held = p.with_name(p.name + ".lock")
    held.parent.mkdir(parents=True, exist_ok=True)
    with open(held, "w") as f:
        deadline = time.monotonic() + LOCK_WAIT
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"another process has held {held} for over {LOCK_WAIT}s; "
                                       "try again when it finishes")
                time.sleep(0.05)
        yield


def update(change, p=None):
    """Read, change and write under the lock. `change(data, findings)` returns the new
    settings, or None to write nothing; a return equal to what was read is not written,
    so a no-op never creates the file. Before replacing an unreadable file, which only a
    change that chose to may do, it is copied to config.json.bak, or config.json.bak-<time>
    when an earlier copy is there. Returns whether the file was written."""
    p = Path(p) if p else path()
    with lock(p):
        data, findings = load(p)
        new = change(data, findings)
        if new is None or (new == data and (p.exists() or not new)):
            return False
        if unreadable(findings) and p.exists():
            bak = p.with_name(p.name + ".bak")
            if bak.exists():                # an earlier copy is kept, never overwritten
                bak = p.with_name(f"{p.name}.bak-{int(time.time())}")
            bak.write_bytes(p.read_bytes())
        write_json(p, new)
        return True


def _selftest():
    """Offline, against a throwaway file: layering, editing, checking, and the lock."""
    saved = os.environ.get("PANOPLY_CONFIG")
    try:
        with tempfile.TemporaryDirectory() as t:
            os.environ["PANOPLY_CONFIG"] = str(Path(t) / "sub" / "config.json")
            assert load() == ({}, []) and layers("grille", "embed") == []
            d = edit({}, "embed", {"model": "m-all", "venv": "/v"})
            d = edit(d, "embed", {"model": "m-own"}, piece="grille")
            assert d == {"embed": {"model": "m-all", "venv": "/v"}, "grille": {"embed": {"model": "m-own"}}}, d
            assert layers("grille", "embed", d) == [("grille", {"model": "m-own"}), ("global", d["embed"])]
            assert layers("locket", "embed", d) == [("global", d["embed"])]
            assert overriding(d, "embed", "model") == ["grille"] and overriding(d, "embed", "venv") == []
            assert overriding(d, "embed", "model", usable=lambda k, sec: sec[k] != "m-own") == [], \
                "a value that does not take effect counted as an override"
            assert edit(d, "embed", clear=["model"], piece="grille") == {"embed": d["embed"]}, "an empty piece section stayed"
            assert edit(d, "embed", reset=True) == {"grille": d["grille"]}, "a reset reached a piece's section"
            assert update(lambda data, f: d) and load() == (d, []), load()
            assert not update(lambda data, f: dict(data)), "an unchanged file was rewritten"
            assert not update(lambda data, f: None)
            other = Path(t) / "other.json"
            assert not update(lambda data, f: {}, other) and not other.exists(), "a no-op created the file"
            path().write_text(json.dumps({"embed": {"model": 3}, "grille": {"embd": {}}, "augur": {"embed": {}},
                                          "colour": "red"}))
            found = load()[1]
            assert len(found) == 3 and not any("augur" in f for f in found), found   # an unknown piece is no problem
            assert not unreadable(found), "a structural finding read as unreadable"
            path().write_text("{not json")
            assert load()[0] == {} and unreadable(load()[1])
            assert update(lambda data, f: {"embed": {}}) and path().with_name("config.json.bak").read_text() == "{not json"
            path().write_text("{again")
            assert update(lambda data, f: {"embed": {"model": "m"}}) and path().with_name("config.json.bak").read_text() == "{not json", \
                "a second replace overwrote the first copy"
            path().write_text("[]")
            assert load()[0] == {} and unreadable(load()[1]) and "expected an object" in load()[1][0]
            with lock():
                pass
            with lock():                    # released, so a second hold does not wait
                pass
    finally:
        if saved is None:
            os.environ.pop("PANOPLY_CONFIG", None)
        else:
            os.environ["PANOPLY_CONFIG"] = saved
    print("settings selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
