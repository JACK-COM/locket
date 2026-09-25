# GENERATED from panoply-lib/embed.py (5e1713d) by sync.sh: edit the source and rerun sync.sh, never this copy.
"""embed: the embedder ladder the Panoply's pieces share.

Ranks text by meaning on whatever this machine can serve, in order: ollama as it
stands, ollama started by us, fastembed in-process. A piece that finds none falls
back to its own word-overlap ranking and says so; nothing that must succeed may
depend on an embedder answering.

Settings are module attributes, read on every call, so a caller changes them on
this module (or inside `settings(...)`, which restores them), never on a copy it
imported by name: after `from embed import TIMEOUT`, or through a module that
re-exports it, an assignment rebinds only the importer's name and is silently ignored
here. Use `settings(...)` or assign on this module itself. The MEMFIND_ environment variables keep that spelling because
hooks and user profiles already set them.

`nomic-embed-text` was chosen by benchmark: Apple's NLEmbedding separated held-out
paraphrases from register-matched negatives at 75% against its 90%. Do not rebuild
that comparison without register-matched negatives; easy ones measure topic.
"""
import contextlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("MEMFIND_MODEL", "nomic-embed-text")
# The same model family served in-process by fastembed (ONNX, no server). A different
# runtime is a different vector space, so its cache tag differs and an index built by
# one is rebuilt by the other.
FE_MODEL = os.environ.get("MEMFIND_FASTEMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBED_CHARS = 2000      # a text is cut here before embedding, on either backend
FE_BATCH = 32           # fastembed's default is 256, and it pads a batch to its longest member
# Start `ollama serve` ourselves when the binary is present and the port is closed:
# the weights are usually on disk and only the server is missing. A hook turns this
# off, because a spawn does not fit a ten-second budget.
AUTOSTART = os.environ.get("MEMFIND_NO_AUTOSTART") != "1"
BATCH = 64
TIMEOUT = 600           # a full index build is minutes; callers on a clock lower it
def _find_venv():
    """fastembed's venv, one per machine and shared by every piece: PANOPLY_VENV, then
    Locket's older LOCKET_VENV, then whichever of ~/.panoply/venv and ~/.locket/venv
    exists (a Locket installed before the shared venv made the second), else the first,
    which is where the setup message tells a user to make it. PEP 668 refuses `pip install`
    into a Homebrew or distro Python, so the extra lives in a venv, and its site-packages
    import only under the Python minor version that made it: every piece's formula
    depends on the same python@3.x, which is what makes sharing it work."""
    for var in ("PANOPLY_VENV", "LOCKET_VENV"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    home = Path.home()
    for p in (home / ".panoply/venv", home / ".locket/venv"):
        if p.is_dir():
            return p
    return home / ".panoply/venv"


VENV = _find_venv()

# Every piece that ranks through the venv, by its command. A new piece that does adds
# its name here, or another piece's uninstall will take the venv from under it.
VENV_USERS = ("locket", "grille")


def shared_venv(me):
    """What `<me> uninstall` may do with the shared venv: (path, others). `path` is the
    venv when it exists at a default location, None when there is none or the user named
    it (PANOPLY_VENV, LOCKET_VENV), which leaves it theirs to remove. `others` are the
    pieces whose command is still on PATH, and any one of them means keep it. PATH is
    probed rather than a list of users kept, because `brew uninstall` runs none of a
    piece's code and such a list would go stale; the miss is a piece on a PATH this
    shell lacks, whose ranking then falls back down the ladder and says so."""
    others = [n for n in VENV_USERS if n != me and shutil.which(n)]
    named = any(os.environ.get(v) for v in ("PANOPLY_VENV", "LOCKET_VENV"))
    home = Path.home()
    ours = VENV in (home / ".panoply/venv", home / ".locket/venv")
    return (VENV if VENV.is_dir() and ours and not named and not others else None), others

_ACTIVE = None          # ("ollama", MODEL) | ("fastembed", FE_MODEL), once resolved
_FE = None              # the fastembed model object, loaded once per process


SETTINGS = ("OLLAMA", "MODEL", "FE_MODEL", "AUTOSTART", "TIMEOUT", "VENV", "_ACTIVE",
            "EMBED_CHARS", "BATCH", "FE_BATCH")


@contextlib.contextmanager
def settings(**kw):
    """Set module settings for a block and restore them after, resolved backend
    included: `with embed.settings(TIMEOUT=8, AUTOSTART=False): ...`."""
    g = globals()
    unknown = [k for k in kw if k not in SETTINGS]
    if unknown:
        raise KeyError(f"not an embed setting: {', '.join(unknown)}")
    saved = {k: g[k] for k in dict.fromkeys((*kw, "_ACTIVE"))}
    g.update(kw)
    try:
        yield
    finally:
        g.update(saved)


def tag(backend):
    """The cache tag a backend writes, so an index says which space it is in."""
    return MODEL if backend == "ollama" else f"fastembed:{FE_MODEL}"


def _open(req, timeout):
    """urllib with proxies from the environment only. The default also asks macOS's
    SystemConfiguration, and after that lookup on a remote plain-http request every child
    this process starts died of SIGSEGV before exec (an `ollama serve` spawn, a piece's
    pdftotext). Measured under Python 3.9 and 3.14; HTTP_PROXY and NO_PROXY still apply."""
    proxies = urllib.request.ProxyHandler(urllib.request.getproxies_environment())
    return urllib.request.build_opener(proxies).open(req, timeout=timeout)


def _ollama_up(timeout=1.5):
    try:
        with _open(f"{OLLAMA}/api/tags", timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _ollama_autostart():
    """Spawn `ollama serve` detached and wait for the port; True if it came up."""
    import shutil
    import subprocess
    exe = shutil.which("ollama")
    if not exe or not OLLAMA.startswith(("http://127.0.0.1", "http://localhost")):
        return False
    try:
        subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return False
    for _ in range(16):                      # up to ~8s
        time.sleep(0.5)
        if _ollama_up():
            return True
    return False


def _embed_ollama(texts, quiet):
    out = []
    for i in range(0, len(texts), BATCH):
        chunk = [t.replace("\n", " ")[:EMBED_CHARS] for t in texts[i:i + BATCH]]
        req = urllib.request.Request(
            f"{OLLAMA}/api/embed",
            data=json.dumps({"model": MODEL, "input": chunk}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with _open(req, TIMEOUT) as r:
                out.extend(json.load(r)["embeddings"])
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach ollama at {OLLAMA} ({e.reason})") from None
        except KeyError:
            raise RuntimeError(f"model {MODEL!r} not available; run: ollama pull {MODEL}") from None
        if not quiet and len(texts) > BATCH:
            print(f"\r  embedding {min(i + BATCH, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    if not quiet and len(texts) > BATCH:
        print(file=sys.stderr)
    return out


def _add_venv():
    """Put VENV's site-packages on sys.path, so a system interpreter can import
    fastembed. site-packages is version-pinned and a compiled wheel from another
    version will not import. Silent where the venv is absent."""
    if os.name == "nt":                      # venv layout differs on Windows
        sp = VENV / "Lib" / "site-packages"
    else:
        sp = VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if sp.is_dir() and str(sp) not in sys.path:
        import site
        site.addsitedir(str(sp))


def _import_fastembed():
    try:
        import fastembed
        return fastembed
    except ImportError:
        _add_venv()
        import fastembed                    # raises ImportError again if truly absent
        return fastembed


def _embed_fastembed(texts, quiet):
    global _FE
    if _FE is None:
        try:
            TextEmbedding = _import_fastembed().TextEmbedding
        except ImportError:
            raise RuntimeError("fastembed is not installed") from None
        _FE = TextEmbedding(model_name=FE_MODEL)
    # fastembed pads every text in a batch to its longest member and attention is
    # quadratic in that length, so one long text taxes every short one beside it:
    # a 2000-character claim among short ones took a 16 GB machine to 16.2 GB and
    # twelve minutes of swapping, 3.8 GB and 82 s without it. Embedding in length
    # order means a batch pads only to its own longest, and a small batch bounds
    # the worst case. Order is restored on the way out.
    cut = [t.replace("\n", " ")[:EMBED_CHARS] for t in texts]
    order = sorted(range(len(cut)), key=lambda i: len(cut[i]))
    out = [None] * len(cut)
    for i, v in zip(order, _FE.embed([cut[i] for i in order], batch_size=FE_BATCH)):
        out[i] = list(map(float, v))
    return out


def have_fastembed():
    try:
        _import_fastembed()
        return True
    except ImportError:
        return False


def resolve_backend(want=None):
    """Which embedder this process uses, resolved once: ollama as it stands, then
    ollama started by us, then fastembed in-process. `want` pins a rung, which is
    how a query is embedded in the same space as its index."""
    global _ACTIVE
    if _ACTIVE and (want is None or _ACTIVE[0] == want):
        return _ACTIVE
    if want in (None, "ollama"):
        if _ollama_up() or (AUTOSTART and _ollama_autostart()):
            _ACTIVE = ("ollama", MODEL)
            return _ACTIVE
    if want in (None, "fastembed") and have_fastembed():
        _ACTIVE = ("fastembed", FE_MODEL)
        return _ACTIVE
    raise RuntimeError(
        "no embedder available. Three ways to get one:\n"
        f"  1. ollama:   install it, then  ollama pull {MODEL}   (the server is started automatically)\n"
        f"  2. in-process, no server, in its own venv (PEP 668 refuses a system pip install):\n"
        f"       {sys.executable} -m venv {VENV} && {VENV}/bin/python -m pip install fastembed\n"
        f"     (model {FE_MODEL} downloads on first use; the venv is found automatically)\n"
        "  3. nothing: ranking falls back to word overlap, which cannot see a paraphrase")


def embed(texts, quiet=False, backend=None):
    """Embed a list of strings on the resolved backend, or on `backend` if given.
    Raises RuntimeError with a readable message."""
    name, _ = resolve_backend(backend)
    return _embed_ollama(texts, quiet) if name == "ollama" else _embed_fastembed(texts, quiet)


def _selftest():
    """Offline: the ladder with no server, the settings block, the refusal message."""
    with settings(_ACTIVE=None, AUTOSTART=False, OLLAMA="http://127.0.0.1:1", VENV=Path("/nonexistent")):
        assert not _ollama_up(timeout=0.2), "a closed port answered"
        if not have_fastembed():
            try:
                resolve_backend()
                raise AssertionError("no rung available, yet a backend resolved")
            except RuntimeError as e:
                assert "no embedder available" in str(e), e
    assert OLLAMA != "http://127.0.0.1:1", "settings() did not restore OLLAMA"
    saved_proxies = urllib.request.getproxies
    urllib.request.getproxies = lambda: (_ for _ in ()).throw(AssertionError("the default opener was used"))
    try:
        with settings(OLLAMA="http://127.0.0.1:1"):
            assert not _ollama_up(timeout=0.2), "a closed port answered"
    finally:
        urllib.request.getproxies = saved_proxies
    try:
        with settings(TIMEOUT=1, NOPE=2):
            pass
        raise AssertionError("an unknown setting was accepted")
    except KeyError:
        pass
    with settings(EMBED_CHARS=5, BATCH=2, FE_BATCH=2):
        assert EMBED_CHARS == 5, "a size setting was refused or not applied"
    assert EMBED_CHARS == 2000, "settings() did not restore EMBED_CHARS"
    assert tag("ollama") == MODEL and tag("fastembed").startswith("fastembed:")
    import tempfile
    saved = {k: os.environ.pop(k, None) for k in ("PANOPLY_VENV", "LOCKET_VENV", "HOME")}
    try:
        with tempfile.TemporaryDirectory() as h:
            os.environ["HOME"] = h
            assert _find_venv() == Path(h) / ".panoply/venv", "no venv: not the shared default"
            (Path(h) / ".locket/venv").mkdir(parents=True)
            assert _find_venv() == Path(h) / ".locket/venv", "Locket's existing venv not shared"
            (Path(h) / ".panoply/venv").mkdir(parents=True)
            assert _find_venv() == Path(h) / ".panoply/venv", "the shared venv lost to Locket's"
            saved_path, os.environ["PATH"] = os.environ.get("PATH", ""), h
            try:
                with settings(VENV=_find_venv()):
                    assert shared_venv("locket") == (Path(h) / ".panoply/venv", []), \
                        "the last piece was not offered the shared venv"
                    (Path(h) / "grille").write_text("#!/bin/sh\n")
                    (Path(h) / "grille").chmod(0o755)
                    assert shared_venv("locket") == (None, ["grille"]), \
                        "the venv was offered while another piece still uses it"
                    (Path(h) / "grille").unlink()
                    os.environ["LOCKET_VENV"] = str(Path(h) / ".panoply/venv")
                    assert shared_venv("locket")[0] is None, "a venv the user named was offered"
            finally:
                os.environ["PATH"] = saved_path
            os.environ["LOCKET_VENV"] = "/x/locket"
            assert _find_venv() == Path("/x/locket"), "LOCKET_VENV ignored"
            os.environ["PANOPLY_VENV"] = "/x/panoply"
            assert _find_venv() == Path("/x/panoply"), "PANOPLY_VENV does not win"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / "grille"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        saved_path = os.environ.get("PATH", "")
        os.environ["PATH"] = d
        try:
            with settings(VENV=Path(d)):
                path, others = shared_venv("locket")
                assert others == ["grille"], f"a piece on PATH not seen: {others}"
                assert path is None, "a venv outside the default locations offered for removal"
                assert shared_venv("grille")[1] == [], "a piece counted itself as another user"
        finally:
            os.environ["PATH"] = saved_path
    print("embed selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
