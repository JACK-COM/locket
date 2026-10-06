# GENERATED from panoply-lib/embed.py (205ab20) by sync.sh: edit the source and rerun sync.sh, never this copy.
"""embed: the embedder ladder the Panoply's pieces share.

Ranks text by meaning on whatever this machine can serve, in order: ollama as it
stands, ollama started by us, onnxruntime in-process. A piece that finds none falls
back to its own word-overlap ranking and says so; nothing that must succeed may
depend on an embedder answering.

Settings are module attributes, read on every call, so a caller changes them on
this module (or inside `settings(...)`, which restores them), never on a copy it
imported by name: after `from embed import TIMEOUT`, or through a module that
re-exports it, an assignment rebinds only the importer's name and is silently ignored
here. Use `settings(...)` or assign on this module itself. The MEMFIND_ environment variables keep that spelling because
hooks and user profiles already set them.

EmbeddingGemma 2 was chosen by benchmark over `nomic-embed-text`, which had beaten
Apple's NLEmbedding: on 80 paraphrase queries over a memory store it ranked the owning
file first 0.86 against 0.68, and separated paraphrases from register-matched
negatives at 0.945 against 0.906. Its task prompts and Matryoshka truncation both
scored lower, so text goes in bare at full width. Do not rebuild that comparison
without register-matched negatives; easy ones measure topic.
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
MODEL = os.environ.get("MEMFIND_MODEL", "embeddinggemma-2:270m")   # needs ollama 0.36+
# The same model in-process: its ONNX export run by onnxruntime, no server. It matches
# ollama's bf16 tag at cosine 0.9999 and the default 4-bit tag at about 0.99, which
# ranks the same; the cache tags differ, so an index built on one rung is rebuilt by
# the other. Pinned to a revision, because a re-export moves
# every vector. fastembed cannot run it: the graph also requires image, video and audio
# feature inputs, which its text pipeline never supplies.
ONNX_MODEL = os.environ.get("MEMFIND_ONNX_MODEL", "onnx-community/embeddinggemma-2-ONNX")
ONNX_REVISION = os.environ.get("MEMFIND_ONNX_REVISION", "daa72c51243991dfcaf9f9137d2c573d8f7790c0")
ONNX_FILE = "onnx/model_quantized.onnx"   # 314MB with its _data; fp16 is 542MB for the same ranking
# sha256 of each file at ONNX_REVISION. A file is kept only when it matches, so a
# download cut short is fetched again rather than read as weights. A new revision
# needs new digests: the repo's API lists them (`?blobs=true`, `lfs.sha256`).
ONNX_SHA256 = {
    "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    "onnx/model_quantized.onnx": "d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953",
    "onnx/model_quantized.onnx_data": "278a7ff1248c3618e4bd11a607fc54f7bdc7778854230f3956d3f86bd9db4f3b",
}
# The model's own prompts for a question searched against passages, which is Grille's
# task. Locket compares a claim with claims and sends text bare, which scored higher there;
# whether the prompts beat bare text on Grille's task has not been measured.
QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: none | text: "
EMBED_CHARS = 2000      # a text is cut here before embedding, on either backend
ONNX_BATCH = 32         # a batch pads to its longest member
# Start `ollama serve` ourselves when the binary is present and the port is closed:
# the weights are usually on disk and only the server is missing. A hook turns this
# off, because a spawn does not fit a ten-second budget.
AUTOSTART = os.environ.get("MEMFIND_NO_AUTOSTART") != "1"
BATCH = 64
TIMEOUT = 600           # a full index build is minutes; callers on a clock lower it
def _find_venv():
    """The in-process rung's venv, one per machine and shared by every piece: PANOPLY_VENV, then
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

_ACTIVE = None          # ("ollama", MODEL) | ("onnx", ONNX_MODEL), once resolved
_ONNX = None            # (session, tokenizer, key), loaded once per process


SETTINGS = ("OLLAMA", "MODEL", "ONNX_MODEL", "ONNX_REVISION", "AUTOSTART", "TIMEOUT", "VENV",
            "_ACTIVE", "EMBED_CHARS", "BATCH", "ONNX_BATCH")


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
    return MODEL if backend == "ollama" else f"onnx:{ONNX_MODEL}@{ONNX_REVISION[:12]}"


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
        except urllib.error.HTTPError as e:
            # ollama answers a model it does not hold with 404
            if e.code == 404:
                raise RuntimeError(f"model {MODEL!r} not available; run: ollama pull {MODEL}  "
                                   "(an ollama older than 0.36 refuses the pull: upgrade it first)") from None
            raise RuntimeError(f"ollama at {OLLAMA} answered {e.code} ({e.reason})") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach ollama at {OLLAMA} ({e.reason})") from None
        except KeyError:
            raise RuntimeError(f"ollama at {OLLAMA} returned no embeddings for {MODEL!r}") from None
        if not quiet and len(texts) > BATCH:
            print(f"\r  embedding {min(i + BATCH, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    if not quiet and len(texts) > BATCH:
        print(file=sys.stderr)
    return out


def _add_venv():
    """Put VENV's site-packages on sys.path, so a system interpreter can import
    onnxruntime and tokenizers. site-packages is version-pinned and a compiled wheel from another
    version will not import. Silent where the venv is absent."""
    if os.name == "nt":                      # venv layout differs on Windows
        sp = VENV / "Lib" / "site-packages"
    else:
        sp = VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if sp.is_dir() and str(sp) not in sys.path:
        import site
        site.addsitedir(str(sp))


def _import_onnx():
    """(onnxruntime, tokenizers, numpy), from the interpreter or the shared venv. A
    fastembed install made for an older ladder already holds all three."""
    try:
        import numpy, onnxruntime, tokenizers
    except ImportError:
        _add_venv()
        import numpy, onnxruntime, tokenizers   # raises ImportError again if truly absent
    return onnxruntime, tokenizers, numpy


def model_dir():
    """Where the ONNX model lives: inside the venv, so the venv's removal takes it too."""
    return VENV / "models" / f"{ONNX_MODEL.replace('/', '--')}@{ONNX_REVISION[:12]}"


def _fetch_model(quiet):
    """Download the pinned model once, keeping each file only when its sha256 matches,
    so a download cut short is fetched again rather than read as weights. Each process
    writes its own part file and `os.replace` lands it, so two first uses at once (hooks
    fire in parallel) neither truncate each other nor fail on a vanished name. Files go
    to a plain directory, never Hugging Face's symlinked cache, because onnxruntime
    refuses external weights that resolve outside the model file's own directory."""
    import hashlib
    d = model_dir()
    for rel in ("tokenizer.json", ONNX_FILE, ONNX_FILE + "_data"):
        dst = d / rel
        if dst.exists():
            continue
        url = f"https://huggingface.co/{ONNX_MODEL}/resolve/{ONNX_REVISION}/{rel}"
        part = dst.with_name(f"{dst.name}.part.{os.getpid()}")
        if not quiet:
            print(f"  fetching {rel} from {ONNX_MODEL} (once)", file=sys.stderr)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with _open(url, TIMEOUT) as r, open(part, "wb") as f:
                for block in iter(lambda: r.read(1 << 20), b""):
                    h.update(block)
                    f.write(block)
            want = ONNX_SHA256.get(rel)
            if want and h.hexdigest() != want:
                raise RuntimeError(f"{rel} arrived incomplete or altered (sha256 {h.hexdigest()[:12]}, "
                                   f"expected {want[:12]}); the next use fetches it again")
            os.replace(part, dst)
        except RuntimeError:
            part.unlink(missing_ok=True)
            raise
        except Exception as e:              # URLError, OSError, http.client's IncompleteRead
            part.unlink(missing_ok=True)
            raise RuntimeError(f"cannot fetch {url} ({getattr(e, 'reason', e)})") from None
    return d


def _load_onnx(quiet):
    global _ONNX
    key = (ONNX_MODEL, ONNX_REVISION, str(VENV))
    if _ONNX is None or _ONNX[2] != key:
        try:
            ort, tokenizers, _ = _import_onnx()
        except ImportError:
            raise RuntimeError("onnxruntime and tokenizers are not installed") from None
        d = _fetch_model(quiet)
        try:
            tok = tokenizers.Tokenizer.from_file(str(d / "tokenizer.json"))
            tok.enable_padding(pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
            sess = ort.InferenceSession(str(d / ONNX_FILE), providers=["CPUExecutionProvider"])
        except Exception as e:
            # A file that verified can still be refused (a runtime too old for the graph);
            # every failure leaves this rung as RuntimeError, so a caller falls back.
            raise RuntimeError(f"cannot load the in-process model from {d} ({e})") from None
        _ONNX = (sess, tok, key)
    return _ONNX[0], _ONNX[1]


def _embed_onnx(texts, quiet):
    if not texts:
        return []
    sess, tok = _load_onnx(quiet)
    np = _import_onnx()[2]
    # The export is multimodal and requires image, video and audio inputs even for
    # text; an empty tensor of each leaves the text embedding exactly as ollama's.
    empty = {i.name: np.zeros([0 if not isinstance(n, int) else n for n in i.shape],
                              dtype=np.float16 if "float16" in i.type else np.float32)
             for i in sess.get_inputs() if i.name.endswith("_features")}
    # A batch pads to its longest member and attention is quadratic in that length, so
    # one long text taxes every short one beside it: under fastembed a 2000-character
    # claim among short ones took a 16 GB machine to 16.2 GB and twelve minutes of
    # swapping, 3.8 GB and 82 s without it. Embedding in length order means a batch pads
    # only to its own longest. Order is restored on the way out.
    cut = [t.replace("\n", " ")[:EMBED_CHARS] for t in texts]
    order = sorted(range(len(cut)), key=lambda i: len(cut[i]))
    out = [None] * len(cut)
    for b in range(0, len(order), ONNX_BATCH):
        idx = order[b:b + ONNX_BATCH]
        enc = tok.encode_batch([cut[i] for i in idx])
        feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                 "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64), **empty}
        for i, v in zip(idx, sess.run(["sentence_embedding"], feeds)[0]):
            out[i] = [float(x) for x in v]
        if not quiet and len(texts) > ONNX_BATCH:
            print(f"\r  embedding {min(b + ONNX_BATCH, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    if not quiet and len(texts) > ONNX_BATCH:
        print(file=sys.stderr)
    return out


def have_onnx():
    try:
        _import_onnx()
        return True
    except ImportError:
        return False


def resolve_backend(want=None):
    """Which embedder this process uses, resolved once: ollama as it stands, then
    ollama started by us, then onnxruntime in-process. `want` pins a rung, which is
    how a query is embedded in the same space as its index."""
    global _ACTIVE
    if _ACTIVE and (want is None or _ACTIVE[0] == want):
        return _ACTIVE
    if want in (None, "ollama"):
        if _ollama_up() or (AUTOSTART and _ollama_autostart()):
            _ACTIVE = ("ollama", MODEL)
            return _ACTIVE
    if want in (None, "onnx") and have_onnx():
        _ACTIVE = ("onnx", ONNX_MODEL)
        return _ACTIVE
    raise RuntimeError(
        "no embedder available. Three ways to get one:\n"
        f"  1. ollama 0.36 or later:   install it, then  ollama pull {MODEL}   (the server is started automatically)\n"
        f"  2. in-process, no server, in its own venv (PEP 668 refuses a system pip install):\n"
        f"       {sys.executable} -m venv {VENV} && {VENV}/bin/python -m pip install onnxruntime tokenizers\n"
        f"     (model {ONNX_MODEL}, 314MB, downloads on first use; the venv is found automatically)\n"
        "  3. nothing: ranking falls back to word overlap, which cannot see a paraphrase")


def embed(texts, quiet=False, backend=None):
    """Embed a list of strings on the resolved backend, or on `backend` if given.
    Raises RuntimeError with a readable message."""
    name, _ = resolve_backend(backend)
    return _embed_ollama(texts, quiet) if name == "ollama" else _embed_onnx(texts, quiet)


def _selftest_fetch():
    """Offline: a body whose digest is wrong is refused and leaves no file, a right one
    lands, and a failed request reaches the caller as RuntimeError."""
    import hashlib
    import io
    import tempfile
    global _open
    real_open, saved = _open, dict(ONNX_SHA256)
    body = {"tokenizer.json": b"tok", ONNX_FILE: b"graph", ONNX_FILE + "_data": b"weights"}
    served = {}
    _open = lambda url, timeout: io.BytesIO(served[url.rsplit(f"{ONNX_REVISION}/", 1)[1]])
    try:
        ONNX_SHA256.update({k: hashlib.sha256(v).hexdigest() for k, v in body.items()})
        with tempfile.TemporaryDirectory() as v, settings(VENV=Path(v)):
            served.update(body, **{ONNX_FILE + "_data": b"weig"})     # cut short
            try:
                _fetch_model(quiet=True)
                raise AssertionError("a short download was kept")
            except RuntimeError as e:
                assert "incomplete" in str(e), e
            data = model_dir() / (ONNX_FILE + "_data")
            assert not data.exists() and not list(data.parent.glob("*.part.*")), "a short file was left"
            served.update(body)
            assert (_fetch_model(quiet=True) / ONNX_FILE).read_bytes() == b"graph"
            assert data.read_bytes() == b"weights", "a verified file did not land"
        def refuse(url, timeout):
            raise urllib.error.URLError("offline")
        _open = refuse
        with tempfile.TemporaryDirectory() as v, settings(VENV=Path(v)):
            try:
                _fetch_model(quiet=True)
                raise AssertionError("a failed fetch did not raise")
            except RuntimeError as e:
                assert "cannot fetch" in str(e), e
    finally:
        _open = real_open
        ONNX_SHA256.clear()
        ONNX_SHA256.update(saved)


def _selftest():
    """Offline: the ladder with no server, the settings block, the refusal message."""
    with settings(_ACTIVE=None, AUTOSTART=False, OLLAMA="http://127.0.0.1:1", VENV=Path("/nonexistent")):
        assert not _ollama_up(timeout=0.2), "a closed port answered"
        if not have_onnx():
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
    with settings(EMBED_CHARS=5, BATCH=2, ONNX_BATCH=2):
        assert EMBED_CHARS == 5, "a size setting was refused or not applied"
    assert EMBED_CHARS == 2000, "settings() did not restore EMBED_CHARS"
    assert tag("ollama") == MODEL and tag("onnx") == f"onnx:{ONNX_MODEL}@{ONNX_REVISION[:12]}"
    with settings(ONNX_REVISION="abc"):
        assert tag("onnx").endswith("@abc"), "a new revision must be a new cache tag"
    with settings(VENV=Path("/v")):
        assert model_dir().parent == Path("/v/models"), "the model must live inside the venv"
    _selftest_fetch()
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
