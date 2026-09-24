#!/usr/bin/env python3
"""memfind: semantic finding tool over a memory corpus.

`memscan` answers *is this sentence already written down somewhere* by lexical
overlap, which catches COPYING and is blind to PARAPHRASE: a restatement sharing
fewer than five content tokens scores exactly 0.0, the same as a novel fact. This
answers the neighbouring question, *which file would already own this*, by
semantic similarity, so a paraphrase surfaces its owner.

It is a FINDING tool, not a decider: it returns a ranked list of files and the
closest sentence in each, and you read. It never blocks a write, is wired to no
hook, and returns no verdict about whether a fact is new. Keep it that way. The
score distributions for "paraphrase of a fact we hold" and "novel fact in the
same register" overlap on this corpus, so any automatic verdict is wrong often
enough to be worse than no verdict; ranking survives that overlap, gating does
not.

Embeddings come from `nomic-embed-text` served by local ollama: weights on disk,
inference on this machine, nothing leaves it. `/api/ps` listing the model is the
positive check, because ollama's cloud-backed models never appear there.

The model was chosen by benchmark and the free on-device candidate lost: Apple's
`NLEmbedding` separated held-out paraphrases from register-matched negatives at
75% against `nomic-embed-text`'s 90%. Register-matched negatives are what make
the benchmark mean anything; easy negatives measure topic detection and would
pass a tool useless here. Do not rebuild the comparison without them.

A corpus may also keep durable facts in a queryable CSV outside the memory
directory. Where the corpus manifest (`locket.json`, key `sources`; the older
standalone `reference-sources.json` is still read) registers one, every row is
indexed alongside the markdown and ranked as its own unit, labelled
`<file>.csv#<area>`. A store's reference CSVs can hold thousands of such rows,
which a markdown-only index cannot see at all.

    locket find "<statement>" [corpus] [-n N]   rank candidate owners
    locket siblings [corpus] [-n N]             rank file pairs by shared SUBJECT
    locket index [corpus]                       build/refresh the cache
    locket selftest

Falls back with a clear message when ollama is not running; it is never a
dependency of anything that must succeed.
"""
import sys, os, json, math, array, time, urllib.request, urllib.error
from operator import mul
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memscan

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("MEMFIND_MODEL", "nomic-embed-text")
# The same model family served in-process by fastembed (ONNX, no server), the
# second rung of the embedder ladder. A different runtime is a different vector
# space, so its cache tag differs and an index built by one is rebuilt by the other.
FE_MODEL = os.environ.get("MEMFIND_FASTEMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBED_CHARS = 2000      # a claim is cut here before embedding, on either backend
FE_BATCH = 32           # fastembed's default is 256, and it pads a batch to its longest member
# Start `ollama serve` ourselves when the binary is present and the port is
# closed: the weights are usually already on disk and only the server is missing.
# A hook turns this off, because a spawn does not fit a ten-second budget.
AUTOSTART = os.environ.get("MEMFIND_NO_AUTOSTART") != "1"
BATCH = 64
CACHE_SUBDIR = ".memfind"   # inside the corpus it indexes; with the manifest, the only thing owned there
TIMEOUT = 600           # a full index build is minutes; callers on a clock lower it


# ---------------------------------------------------------------- embedding

_ACTIVE = None          # ("ollama", MODEL) | ("fastembed", FE_MODEL), once resolved
_FE = None              # the fastembed model object, loaded once per process
LAST_RANK_MODE = "semantic"


def tag(backend):
    """The cache tag a backend writes, so an index says which space it is in."""
    return MODEL if backend == "ollama" else f"fastembed:{FE_MODEL}"


def _ollama_up(timeout=1.5):
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _ollama_autostart():
    """Spawn `ollama serve` detached and wait for the port; True if it came up."""
    import shutil, subprocess
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
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                out.extend(json.load(r)["embeddings"])
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach ollama at {OLLAMA} ({e.reason})") from None
        except KeyError:
            raise RuntimeError(f"model {MODEL!r} not available; run: ollama pull {MODEL}") from None
        if not quiet and len(texts) > BATCH:
            print(f"\r  embedding {min(i+BATCH, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    if not quiet and len(texts) > BATCH:
        print(file=sys.stderr)
    return out


LOCKET_VENV = Path(os.environ.get("LOCKET_VENV", str(Path.home() / ".locket/venv")))


def _add_locket_venv():
    """Put Locket's own venv on sys.path, so `python3 memfind.py` under the system
    interpreter can import fastembed. PEP 668 refuses `pip install` into a
    Homebrew or distro Python, so the extra lives in a venv created by that SAME
    interpreter: site-packages is version-pinned and a compiled wheel from
    another version will not import. Silent where the venv is absent."""
    if os.name == "nt":                      # venv layout differs on Windows
        sp = LOCKET_VENV / "Lib" / "site-packages"
    else:
        sp = LOCKET_VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if sp.is_dir() and str(sp) not in sys.path:
        import site
        site.addsitedir(str(sp))


def _import_fastembed():
    try:
        import fastembed
        return fastembed
    except ImportError:
        _add_locket_venv()
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
    # quadratic in that length, so one long claim taxes every short one beside it:
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


def resolve_backend(want=None):
    """Which embedder this process uses, resolved once: the ladder is ollama as
    it stands, then ollama started by us, then fastembed in-process. `want` pins
    a rung, which is how a query is embedded in the same space as its index."""
    global _ACTIVE
    if _ACTIVE and (want is None or _ACTIVE[0] == want):
        return _ACTIVE
    if want in (None, "ollama"):
        if _ollama_up() or (AUTOSTART and _ollama_autostart()):
            _ACTIVE = ("ollama", MODEL)
            return _ACTIVE
    if want in (None, "fastembed"):
        try:
            _import_fastembed()
            _ACTIVE = ("fastembed", FE_MODEL)
            return _ACTIVE
        except ImportError:
            pass
    raise RuntimeError(
        "no embedder available. Three ways to get one:\n"
        f"  1. ollama:   install it, then  ollama pull {MODEL}   (memfind starts the server itself)\n"
        f"  2. in-process, no server, in Locket's own venv (PEP 668 refuses a system pip install):\n"
        f"       {sys.executable} -m venv {LOCKET_VENV} && {LOCKET_VENV}/bin/python -m pip install fastembed\n"
        f"     (model {FE_MODEL} downloads on first use; memfind finds the venv itself)\n"
        "  3. nothing: memfind falls back to word overlap, which cannot see a paraphrase")


def embed(texts, quiet=False, backend=None):
    """Embed a list of strings on the resolved backend, or on `backend` if
    given. Raises RuntimeError with a readable message."""
    name, _ = resolve_backend(backend)
    return _embed_ollama(texts, quiet) if name == "ollama" else _embed_fastembed(texts, quiet)


# ---------------------------------------------------------------- index

MANIFEST = memscan.LEGACY_MANIFEST
# One reader for the registered CSVs, and it lives in memscan so that the audit
# and the write-time gate see the same rows this index does.
csv_sources = memscan.csv_sources
csv_rows = memscan.csv_rows


def _is_row(label):
    """Is this index item a CSV row rather than a markdown file?

    A row's label is `<file>.csv#<area>` and a markdown file's is its relative path,
    so the label alone decides and nothing has to be threaded through `load_index`,
    whose return signature three hooks already depend on.
    """
    return ".csv#" in label


def corpus_fingerprint(root):
    """{filename: [mtime_ns, size]} per file, so a stale index is detectable."""
    # Keyed by corpus-relative path, never basename: a store with depth holds
    # many SKILL.md and MEMORY.md, and a basename key silently overwrote them,
    # leaving stale detection blind to every file but the last of each name.
    fp = {memscan.corpus_key(root, p): [p.stat().st_mtime_ns, p.stat().st_size]
          for p in sorted(memscan.md_files(root))}
    # The registered CSVs and the manifest naming them, so editing either one, or
    # adding a source, invalidates the index exactly as editing a memory file does.
    for pth, _, _ in csv_sources(root):
        fp[pth.name] = [pth.stat().st_mtime_ns, pth.stat().st_size]
    for r in memscan.roots_of(root):            # a joined store's manifest counts too
        for name in (MANIFEST, memscan.MANIFEST, memscan.OLD_MANIFEST):
            man = Path(r) / name
            if man.exists():
                fp[memscan.corpus_key(root, man)] = [man.stat().st_mtime_ns, man.stat().st_size]
    # The splitter's version: a change to how a file becomes claims re-embeds
    # every file once, where the per-file reuse below would keep stale claims
    # under an unchanged file forever.
    fp["__claims__"] = memscan.CLAIMS_VERSION
    return fp


def cache_paths(root):
    """The index lives INSIDE the corpus, so a corpus copied or moved carries its
    cache, and removing the tool from one is deleting a directory it can see."""
    d = Path(root) / CACHE_SUBDIR
    return d / "index.json", d / "index.vec"


def _normalise(vecs):
    flat = array.array("f")
    for v in vecs:
        n = math.sqrt(sum(x * x for x in v)) or 1.0     # store L2-normalised:
        flat.extend([x / n for x in v])                 # cosine becomes a dot product
    return flat


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return array.array("f", [x / n for x in v])


def build_index(root, quiet=False):
    """Rebuild only the files that changed. Returns (items, flat, dim).

    Per-file reuse is load-bearing, not an optimisation: a full rebuild costs
    ~50s and a cycle writes to this corpus every three hours, so without reuse
    nearly every query pays that. One edited file takes a couple of seconds.
    """
    # `corpus` returns bodies already stripped. Stripping again is not idempotent:
    # with the `|` markers gone, `unwrap` joins every table row into one line,
    # a 4000-row table becomes one claim (and the join is quadratic), so the
    # table vanishes from the index and a large one hangs the build.
    docs = memscan.corpus(root, csv=False)   # rows are indexed below as their own units
    per_file = {name: [s for s, _ in memscan.claims(body)]
                for name, body in sorted(docs.items())}
    if not any(per_file.values()):
        raise RuntimeError(f"no claims found in {root}")

    meta_p, vec_p = cache_paths(root)
    active = tag(resolve_backend()[0])
    old_items, old_flat, old_dim, old_fp = [], None, None, {}
    if meta_p.exists() and vec_p.exists():
        try:
            m = json.loads(meta_p.read_text())
            if m.get("model") == active:
                old_items = [tuple(i) for i in m["items"]]
                old_dim = m["dim"]
                old_fp = m.get("fingerprint", {})
                old_flat = _read_vectors(vec_p, len(old_items), old_dim)
        except (json.JSONDecodeError, KeyError, ValueError):
            old_items, old_flat = [], None            # unreadable cache: start over

    # Two reuse structures, because the two sources change differently. A markdown
    # file's sentences move together, so it reuses a contiguous SLICE. A CSV row is
    # independent of its neighbours and every row below an edit shifts, so a row
    # reuses by its own TEXT. Keying a row on its label would never hit: labels are
    # `<file>.csv#<area>` and the fingerprint is per basename.
    old_slice = {}                                     # file -> (start, count)
    old_row = {}                                       # row text -> position
    if old_items:
        i = 0
        while i < len(old_items):
            f = old_items[i][0]
            if _is_row(f):
                old_row[old_items[i][1]] = i
                i += 1
                continue
            j = i
            while j < len(old_items) and old_items[j][0] == f and not _is_row(old_items[j][0]):
                j += 1
            old_slice[f] = (i, j - i)
            i = j

    fp = corpus_fingerprint(root)
    t = time.time()
    items, flat, dim = [], array.array("f"), old_dim
    reused = fresh_files = 0
    for name, sents in per_file.items():
        if not sents:
            continue
        unchanged = (old_flat is not None and name in old_slice
                     and old_fp.get(name) == fp.get(name)
                     and old_fp.get("__claims__") == fp["__claims__"]
                     and old_slice[name][1] == len(sents))
        if unchanged:
            start, count = old_slice[name]
            flat.extend(old_flat[start * old_dim:(start + count) * old_dim])
            reused += count
        else:
            vs = embed(sents, quiet=quiet)
            dim = len(vs[0])
            if old_dim and dim != old_dim:              # model changed under us
                return _full_rebuild(root, per_file, fp, quiet)
            flat.extend(_normalise(vs))
            fresh_files += 1
        items.extend((name, s) for s in sents)

    rows = csv_rows(root)
    if rows:
        need = [t for _, t in rows if not (old_flat is not None and t in old_row)]
        fresh = {}
        if need:
            uniq = list(dict.fromkeys(need))
            vs = embed(uniq, quiet=quiet)
            dim = dim or len(vs[0])
            if len(vs[0]) != dim:                       # model changed under us
                return _full_rebuild(root, per_file, fp, quiet)
            fresh = {t: _unit(v) for t, v in zip(uniq, vs)}
        for label, text in rows:
            if text in fresh:
                flat.extend(fresh[text])
            else:
                pos = old_row[text]
                flat.extend(old_flat[pos * old_dim:(pos + 1) * old_dim])
                reused += 1
            items.append((label, text))

    meta_p.parent.mkdir(parents=True, exist_ok=True)
    vec_p.write_bytes(flat.tobytes())
    meta_p.write_text(json.dumps({
        "root": str(root), "model": active, "dim": dim,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fingerprint": fp, "items": items,
    }))
    if not quiet:
        under = sum(1 for body in docs.values() for _ in memscan.unclaimed(body))
        # A claim past the cap is embedded on its first EMBED_CHARS characters and
        # is nearly always a strip artifact (a fused table, a list without stops),
        # so the count is printed with the file: what the index cannot see.
        capped = [(name, s) for name, s in items if len(s) > EMBED_CHARS]
        print(f"indexed {len(items)} claims from {len(per_file)} files in "
              f"{time.time()-t:.0f}s  (dim {dim}; {reused} reused, "
              f"{fresh_files} file(s) re-embedded; {under} sentences under the "
              f"{memscan.MIN_TOKENS}-token floor formed no claim)", file=sys.stderr)
        if capped:
            print(f"  {len(capped)} claim(s) over {EMBED_CHARS} characters, embedded on their "
                  f"first {EMBED_CHARS}: " + ", ".join(sorted({n for n, _ in capped})), file=sys.stderr)
    return items, flat, dim


def _full_rebuild(root, per_file, fp, quiet):
    items = [(n, s) for n, ss in per_file.items() for s in ss] + csv_rows(root)
    vs = embed([s for _, s in items], quiet=quiet)
    flat, dim = _normalise(vs), len(vs[0])
    meta_p, vec_p = cache_paths(root)
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    vec_p.write_bytes(flat.tobytes())
    meta_p.write_text(json.dumps({
        "root": str(root), "model": tag(resolve_backend()[0]), "dim": dim,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fingerprint": fp, "items": items,
    }))
    return items, flat, dim


def load_index(root, quiet=False):
    """Return a CURRENT index, rebuilding whatever changed. There is no
    serve-stale option on purpose: an answer drawn from a superseded corpus
    reads as current, and the refresh costs about a second. A caller that
    cannot pay that takes `load_cached` instead."""
    meta, vec = cache_paths(root)
    if meta.exists() and vec.exists():
        m = json.loads(meta.read_text())
        if (m.get("model") == tag(resolve_backend()[0])
                and m.get("fingerprint") == corpus_fingerprint(root)):
            try:
                return [tuple(i) for i in m["items"]], _read_vectors(vec, len(m["items"]), m["dim"]), m["dim"]
            except (KeyError, ValueError):
                pass                                   # damaged cache: rebuild below
    return build_index(root, quiet=quiet)


def _read_vectors(path, n, dim):
    """The cached vectors, refused unless there are exactly n of length dim: a write cut
    short by a kill leaves a file that frombytes reads without complaint, and every
    score after the cut would be off misaligned floats."""
    flat = array.array("f")
    flat.frombytes(path.read_bytes())
    if not isinstance(dim, int) or dim <= 0 or len(flat) != n * dim:
        raise ValueError(f"vector cache holds {len(flat)} floats, expected {n} x {dim}")
    return flat


def load_cached(root):
    """The cache as it stands, stale or not: (items, flat, dim, stale), or None
    where no readable cache exists. Use it from a caller on a clock: it cannot
    afford the rebuild `load_index` would run, and a slightly stale neighbour
    list beats silence. Whoever calls this owes the reader the staleness.
    """
    meta, vec = cache_paths(root)
    if not (meta.exists() and vec.exists()):
        return None
    try:
        m = json.loads(meta.read_text())
        # Whichever backend built it: the query is embedded in that same space
        # (`rank` pins the backend from the tag), so no embedder is probed here.
        if m.get("model") not in (tag("ollama"), tag("fastembed")):
            return None
        flat = _read_vectors(vec, len(m["items"]), m["dim"])
        stale = m.get("fingerprint") != corpus_fingerprint(root)
        return [tuple(i) for i in m["items"]], flat, m["dim"], stale
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


# ---------------------------------------------------------------- query

def lexical_rank(statement, root, top=5, per_file=True, items=None):
    """`rank` with no embedder: word overlap over the same claims, scored as the
    cosine of two binary bags. Blind to a paraphrase by construction, which is
    the gap the semantic rung exists to close; better than refusing to answer.
    Same output shape as `rank`, so every caller reads it unchanged."""
    if items is None:
        items = [(name, s) for name, body in sorted(memscan.corpus(root, csv=False).items())
                 for s, _ in memscan.claims(body)]              # already stripped
        items += csv_rows(root)
    q = memscan.tokens(statement)
    if not q:
        return []
    best = {}
    for i, (f, s) in enumerate(items):
        t = memscan.tokens(s)
        shared = len(q & t)
        if not shared:
            continue
        c = shared / math.sqrt(len(q) * len(t))
        key = i if (_is_row(f) or not per_file) else f
        if c > best.get(key, (-2, -1))[0]:
            best[key] = (c, i)
    rows = sorted(best.items(), key=lambda kv: -kv[1][0])[:top]
    return [(c, items[i][0], items[i][1]) for _, (c, i) in rows]


def rank(statement, root, top=5, per_file=True, quiet=False, idx=None, lexical=True):
    """Ranked (score, file-or-row, sentence). Semantic where an embedder can be
    had; with `lexical` true it falls back to word overlap and sets
    LAST_RANK_MODE so the caller can say so. A hook passes `lexical=False`,
    because a nudge worded "by meaning" must not be produced by word overlap."""
    global LAST_RANK_MODE
    LAST_RANK_MODE = "semantic"
    try:
        items, flat, dim = idx or load_index(root, quiet=quiet)
        which = "ollama"                     # a caller's idx was read from the cache
        if idx is None:
            which = resolve_backend()[0]
        else:
            m = json.loads(cache_paths(root)[0].read_text())
            which = "fastembed" if str(m.get("model", "")).startswith("fastembed:") else "ollama"
        qv = embed([statement], quiet=True, backend=which)[0]
    except RuntimeError:
        if not lexical:
            raise
        LAST_RANK_MODE = "lexical"
        return lexical_rank(statement, root, top=top, per_file=per_file,
                            items=idx[0] if idx else None)
    qn = math.sqrt(sum(x * x for x in qv)) or 1.0
    qv = [x / qn for x in qv]
    best = {}
    scored = []
    for i in range(len(items)):
        off = i * dim
        # vectors are stored L2-normalised, so the dot product IS the cosine.
        # sum(map(mul, ...)) over an array slice runs ~2.4x an indexed loop:
        # 0.52s -> 0.22s over 12.6k claims, and grepassist pays this per search.
        c = sum(map(mul, qv, flat[off:off + dim]))
        f = items[i][0]
        if not per_file:
            scored.append((c, i))
            continue
        # A CSV row is its own unit, so it never groups: the reader wants the ROW.
        # Grouping 2,918 rulings under one filename would return the single best
        # row and hide every other match in the file.
        key = i if _is_row(f) else f
        if c > best.get(key, (-2, -1))[0]:
            best[key] = (c, i)
    if per_file:
        rows = sorted(best.items(), key=lambda kv: -kv[1][0])[:top]
        # Name from the ITEM, never from the group key: a row's key is its integer
        # position, so returning the key printed a bare index where the label belongs.
        return [(c, items[i][0], items[i][1]) for _, (c, i) in rows]
    scored.sort(reverse=True)
    return [(c, items[i][0], items[i][1]) for c, i in scored[:top]]


# ---------------------------------------------------------------- siblings

def siblings(root, top=30, quiet=False, idx=None):
    """Rank file PAIRS by shared subject, for finding merge candidates.

    `memscan audit` takes the max over claim pairs, so two files sharing one
    verbatim boilerplate line score 1.00 whatever they are about: four unrelated
    files scored 1.00 on "RELEASED to main and LIVE" plus a build-provenance
    line. A file's centroid averages that furniture down and leaves the subject
    standing.

    Mean-centring is what makes the centroid readable, not an optimisation:
    nomic's space is anisotropic, so raw centroid cosine puts every pair between
    0.97 and 0.99, and subtracting the corpus mean centroid recovers a 0.68-0.84
    spread over the same 345 files. It is recorded as having FAILED against a
    novel-versus-paraphrase threshold, which is a different problem: ordering two
    CLASSES, where here the shared corpus direction is common-mode noise between
    two vectors compared only with each other.

    Read the two columns together. A high subject score with a low max is one
    subject written twice independently, which is the merge candidate. A low
    subject score with a max near 1.00 is shared boilerplate across different
    subjects, which wants the boilerplate deleted and the files left alone.

    Cost is one dot product per file pair, quadratic in files, a couple of minutes
    on a few hundred. A maintenance-pass tool, not a per-write one.
    """
    items, flat, dim = idx or load_index(root, quiet=quiet)
    # Rows are excluded outright. A CSV row is one claim rather than a file with a
    # subject, so a per-row centroid is the claim itself, and pairing thousands of
    # them is quadratic in rows rather than in files.
    per_file = {}
    for i, (name, _) in enumerate(items):
        if _is_row(name):
            continue
        per_file.setdefault(name, []).append(i)

    raw = {}
    for name, idxs in per_file.items():
        v = [0.0] * dim
        for i in idxs:
            off = i * dim
            for k in range(dim):
                v[k] += flat[off + k]
        raw[name] = [x / len(idxs) for x in v]

    mean = [0.0] * dim
    for v in raw.values():
        for k in range(dim):
            mean[k] += v[k]
    mean = [x / len(raw) for x in mean]

    cent = {}
    for name, v in raw.items():
        w = [v[k] - mean[k] for k in range(dim)]
        n = math.sqrt(sum(x * x for x in w)) or 1.0
        cent[name] = array.array("f", [x / n for x in w])

    # An index file restates every fact it points at, so it is a subject-mate of
    # everything it lists. True, and useless as a merge signal. Which files are
    # indexes is the store's own manifest's to say: on a store where MEMORY.md is
    # the whole continuity record, a fixed name list threw out a third of it.
    index_files = memscan.settings(root)["index_files"]
    names = sorted(f for f in cent if Path(f).name not in index_files)
    pairs = []
    for a in range(len(names)):
        va = cent[names[a]]
        if not quiet and a and a % 50 == 0:
            print(f"\r  pairing {a}/{len(names)}", end="", file=sys.stderr)
        for b in range(a + 1, len(names)):
            # vectors are stored L2-normalised, so the dot product IS the cosine.
            # sum(map(mul, ...)) over array slices runs ~6x an indexed loop.
            pairs.append((sum(map(mul, va, cent[names[b]])), names[a], names[b]))
    if not quiet:
        print(file=sys.stderr)
    pairs.sort(reverse=True)

    out = []
    for c, fa, fb in pairs[:top]:
        best = -2.0
        for i in per_file[fa]:
            va = flat[i * dim:(i + 1) * dim]
            for j in per_file[fb]:
                s = sum(map(mul, va, flat[j * dim:(j + 1) * dim]))
                if s > best:
                    best = s
        out.append((c, best, fa, fb))
    return out


# ---------------------------------------------------------------- cli

def selftest():
    # A synthetic store, never the live one. Reading whatever this machine holds
    # made the assertions depend on its shape: a store of three files, one of them
    # named as an index by a fixed list, has one pair and returned nothing. The
    # fixture carries a manifest naming its own index file, so that path is
    # exercised too, and it needs an embedder: without one it says so and stops.
    import tempfile, shutil
    try:
        resolve_backend()
    except RuntimeError as e:
        print(f"selftest: no embedder here, so nothing to test yet.\n{e}", file=sys.stderr)
        return
    d = Path(tempfile.mkdtemp(prefix="memfind-selftest-"))
    root = d / "store"
    root.mkdir()
    (root / "engines.md").write_text(
        "Turbine overhaul programs cover the engine reserve entirely for the operator.\n"
        "A hot-section inspection falls due at half the overhaul interval on most turboprops.\n")
    (root / "airframes.md").write_text(
        "Corrosion inspections on a seaplane hull are due every hundred hours in salt water.\n"
        "Fabric-covered wings need a punch test whenever the finish shows crazing or chalking.\n")
    (root / "costs.md").write_text(
        "The engine reserve is the largest single line in a turboprop's hourly operating cost.\n"
        "Hangar rent varies more with the airport than with the aircraft parked inside it.\n")
    (root / "INDEX.md").write_text(
        "- [engines](engines.md) — turbine overhaul programs cover the engine reserve\n"
        "- [costs](costs.md) — the engine reserve is the largest hourly line\n")
    (root / "fleet.md").write_text(               # a table: one claim per row, never one per table
        "| aircraft | engine | note |\n|---|---|---|\n"
        + "".join(f"| Type {i} | Engine {i} | a row about airframe number {i} in the fleet table |\n"
                  for i in range(40)))
    (root / memscan.MANIFEST).write_text(json.dumps({"index_files": ["INDEX.md"]}))
    memscan.settings.cache_clear()
    items, flat, dim = load_index(root, quiet=True)
    assert len(flat) == len(items) * dim, "vector file does not match item count"
    assert sum(1 for f, _ in items if f == "fleet.md") == 40, "table rows must index one claim each"
    for i in (0, len(items) // 2, len(items) - 1):        # stored normalised
        n = math.sqrt(sum(flat[i * dim + k] ** 2 for k in range(dim)))
        assert abs(n - 1.0) < 1e-3, f"vector {i} not normalised: {n}"
    r = rank(items[0][1], root, top=1, quiet=True)         # a claim finds itself
    assert r[0][0] > 0.95, f"self-similarity too low: {r[0][0]:.3f}"
    assert LAST_RANK_MODE == "semantic"
    # the last rung: with no embedder at all the same claim still finds itself by
    # word overlap, the caller is told which mode answered, and a hook refuses
    # rather than dressing overlap up as meaning
    lx = lexical_rank(items[0][1], root, top=1, items=items)
    assert lx and lx[0][1] == items[0][0] and lx[0][0] > 0.99, lx
    global _ACTIVE, AUTOSTART, OLLAMA
    saved = (_ACTIVE, AUTOSTART, OLLAMA)
    _ACTIVE, AUTOSTART, OLLAMA = None, False, "http://127.0.0.1:1"
    try:
        try:
            _import_fastembed()
            have_fe = True
        except ImportError:
            have_fe = False
        if not have_fe:
            r = rank(items[0][1], root, top=1, quiet=True, idx=(items, flat, dim))
            assert LAST_RANK_MODE == "lexical" and r[0][1] == items[0][0], r
            try:
                rank(items[0][1], root, top=1, quiet=True, idx=(items, flat, dim), lexical=False)
                assert False, "a hook-path rank with no embedder must raise, never fall back"
            except RuntimeError:
                pass
    finally:
        _ACTIVE, AUTOSTART, OLLAMA = saved
    s = siblings(root, top=3, quiet=True, idx=(items, flat, dim))
    assert len(s) == 3, f"three non-index files make three pairs, got {len(s)}"
    assert all(-1.001 <= c <= 1.001 for c, _, _, _ in s), "subject score out of range"
    assert [c for c, _, _, _ in s] == sorted((c for c, _, _, _ in s), reverse=True), \
        "siblings not ranked descending"
    assert all(f != "INDEX.md" for _, _, a, b in s for f in (a, b)), \
        "the manifest's index file leaked into the sibling pairs"
    # No assertion on WHICH pair ranks first: centroids are mean-centred, and on
    # three files the three residuals sum to zero, so every subject score is
    # negative and their order says little. Siblings is a maintenance tool for a
    # corpus of dozens of files; what is checked here is shape, not judgment.
    memscan.settings.cache_clear()
    shutil.rmtree(d, ignore_errors=True)
    _selftest_rows()
    print("selftest ok")


def _selftest_rows():
    """The CSV-row path, on a corpus built for it.

    Built rather than borrowed: few standing corpora register a reference
    CSV, so asserting against whatever this machine happens to hold would pass
    vacuously everywhere else and hide a broken row path until it mattered.
    """
    import tempfile, shutil, csv as _csv
    d = Path(tempfile.mkdtemp(prefix="memfind-selftest-"))
    try:
        (d / "a.md").write_text("The hangar door opens northward onto the taxiway apron.\n")
        (d / "b.md").write_text("Fuel is drawn from the south pump before every departure.\n")
        c = d / "R.csv"
        with open(c, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["Ruling", "Area"])
            w.writerow(["Never taxi past the yellow line without a marshaller present.", "ops"])
            w.writerow(["Chock both mains before disconnecting the tug from the nosewheel.", "ops"])
        (d / MANIFEST).write_text(json.dumps(
            [{"path": str(c), "text": "Ruling", "label": "Area"}]))

        items, flat, dim = build_index(d, quiet=True)
        rows = [i for i in items if _is_row(i[0])]
        assert len(rows) == 2, f"expected 2 row items, got {len(rows)}"
        assert all(l == "R.csv#ops" for l, _ in rows), f"bad row labels: {rows}"
        assert len(flat) == len(items) * dim, "vector file does not match item count"

        r = rank("chock the mains before the tug comes off", d, top=3, quiet=True)
        assert _is_row(r[0][1]), f"a row should outrank the markdown here: {r[0][1]}"

        # Both rows share one label, so grouping by label would collapse them.
        r = rank("marshaller and chocks on the apron", d, top=4, quiet=True)
        assert len([x for x in r if _is_row(x[1])]) == 2, \
            f"rows collapsed under their shared label: {r}"

        sib = siblings(d, top=5, quiet=True)
        assert all(not _is_row(f) for _, _, a, b in sib for f in (a, b)), \
            "a CSV row leaked into the sibling pairs"

        # An edited CSV must reuse the untouched row rather than re-embedding it.
        with open(c, "a", newline="") as fh:
            _csv.writer(fh).writerow(["Tow only with the nose gear pin installed.", "ops"])
        items2, flat2, dim2 = build_index(d, quiet=True)
        assert len([i for i in items2 if _is_row(i[0])]) == 3, "appended row not indexed"
        old_vec = [i for i, x in enumerate(items) if x[1].startswith("Never taxi")][0]
        new_vec = [i for i, x in enumerate(items2) if x[1].startswith("Never taxi")][0]
        assert flat[old_vec * dim:(old_vec + 1) * dim] == flat2[new_vec * dim2:(new_vec + 1) * dim2], \
            "an unchanged row was re-embedded instead of reused"
    finally:
        meta_p, vec_p = cache_paths(d)
        for f in (meta_p, vec_p):
            if f.exists():
                f.unlink()
        shutil.rmtree(d, ignore_errors=True)


USAGE = """memfind: which files in a memory corpus already hold this idea?

Ranks by SEMANTIC similarity, so unlike `memscan audit` it catches a paraphrase
that shares no wording. Run it BEFORE writing a durable fact, never as a second
opinion after a grep: a grep returning nothing is not evidence the corpus is
silent on a subject.

USAGE
  locket find "<statement>" [corpus] [-n N]
  locket siblings [corpus] [-n N]
  locket index [corpus]
  locket embedder                             which embedder answers here, if any
  locket selftest

  (`memfind.py "<statement>"`, `--siblings`, `--index`, `--embedder` and
  `--selftest` by path do the same; `locket` is the front for both scripts.)

  The corpus argument is optional and takes a shorthand (council, hermes)
  or registered name, any substring of a Claude Code project slug (myapp),
  a literal path, or `all` for every corpus on the machine in turn. Omit it to
  use the corpus the current directory sits inside, else the one belonging to
  it. Default -n is 5 for a statement, 30 for --siblings. What counts as a
  corpus, and `locket init <dir>`, which makes any folder of markdown one
  that every mode can find, are in `locket --help`.

  `all` is also the cross-corpus question: a statement ranked against every
  store, one list per store, is how a fact already held in ANOTHER store is
  found before it is written into this one. Nothing else asks that; each
  store's write-time gate scores that store alone. Lists are never merged
  across stores, because two caches may have been built by different
  embedders and their scores do not compare. `locket across` is the
  lexical half, over copies already made.

EXAMPLES
  locket find "gear is stored as two fields, not one"
  locket find "the deploy runs no migrations" myapp -n 15
  locket find "how we rank a source" rules
  locket find "the deploy runs no migrations" all
  locket siblings myapp -n 20
  locket index myapp

REFERENCE CSVs
  A result labelled `<file>.csv#<area>` is one ROW of a reference CSV registered
  under `sources` in `<corpus>/locket.json` (or in the older standalone
  `reference-sources.json`, still read), not a file. Rows never group, so
  several from one CSV can share the ranking; query the CSV on that `Area` for
  the rest. `sources` is a list of {path, text, label}: `text` names the column
  holding the claim, `label` the column that identifies it in output.

THE INDEX
  Lives in `<corpus>/.memfind/` and rebuilds itself, per changed file, on every
  query. It and the manifest are the only things this tool writes inside a
  corpus; delete that directory and the manifest and the tool is gone from it.

READING THE OUTPUT
  Pass -n 15 on a corpus of a few hundred files. Ranking quality does not decay
  with size, but the odds of clearing a large field in five rows do.
  Expect an index file to outrank the true owner; an index restates every fact
  it points at by design.
  --siblings answers a different question: do two files already hold ONE subject
  between them. Read its two columns together. High subject score with a low max
  claim is one subject written twice, so a merge candidate; low subject with a
  max near 1.00 is shared boilerplate across unrelated subjects, so delete the
  boilerplate and leave the files alone.

NO VERDICT, BY DESIGN
  A score never gates a write. The distributions for a paraphrase already held
  and a genuinely novel fact overlap far too much for a threshold to beat no
  threshold. Read the neighbour list; do not read the number.

Needs ollama serving `nomic-embed-text`. The module docstring carries the design
rationale."""


def main(argv):
    # A bare invocation prints help. `--help`/`-h` did not, and fell through to
    # being treated as the statement to rank, which returns neighbours for the
    # word "help" and looks exactly like a working search.
    if len(argv) < 2 or argv[1] in ("--help", "-h", "help"):
        print(USAGE)
        return 0 if len(argv) > 1 else 1
    if argv[1] == "--selftest":
        selftest(); return 0
    if argv[1] == "--embedder":
        # Which rung answers on this machine, before anything is installed. The
        # install prompt runs this first, so a host that already has an embedder
        # is left exactly as it was.
        try:
            name, model = resolve_backend()
        except RuntimeError as e:
            print(f"embedder: none\n{e}"); return 1
        where = OLLAMA if name == "ollama" else f"in-process, venv {LOCKET_VENV}"
        print(f"embedder: {name} ({model}) via {where}"); return 0
    if "all" in argv[2:]:
        # Every corpus in turn, for the user whose memory is one global store
        # and who should not need this tool's vocabulary to ask it a question.
        rc = 0
        for k, v in sorted(memscan.all_corpora().items()):
            if not memscan.md_files(v):      # an empty store cannot be indexed
                continue
            print(f"== {k}")
            try:
                rc |= main([a if a != "all" else str(v) for a in argv])
            except RuntimeError as e:
                print(f"memfind: {e}", file=sys.stderr); rc = 1
            print()
        return rc
    if argv[1] == "--index":
        root = memscan.resolve(argv[2]) if len(argv) > 2 else memscan.corpus_here(os.getcwd())
        if root is None:
            print("no corpus for this directory; name one, or make one with  locket init <dir>",
                  file=sys.stderr); return 1
        build_index(root); return 0
    if argv[1] == "--siblings":
        rest, top = argv[2:], 30
        if "-n" in rest:
            j = rest.index("-n"); top = int(rest[j + 1]); rest = rest[:j] + rest[j + 2:]
        root = memscan.resolve(rest[0]) if rest else memscan.corpus_here(os.getcwd())
        if root is None:
            print("no corpus for this directory; name one", file=sys.stderr); return 1
        print(f"corpus: {root}")
        for c, m, fa, fb in siblings(root, top=top):
            print(f"[subject {c:.3f} | max claim {m:.3f}] {fa}  ::  {fb}")
        print("\nHigh subject with a LOW max claim is one subject written twice: a merge\n"
              "candidate. Low subject with a max claim near 1.00 is shared boilerplate across\n"
              "different subjects: delete the boilerplate and leave the files alone. Neither is\n"
              "a verdict, and index files are excluded because they restate what they point at.")
        return 0

    statement = argv[1]
    rest = argv[2:]
    top = 5
    if "-n" in rest:
        j = rest.index("-n"); top = int(rest[j + 1]); rest = rest[:j] + rest[j + 2:]
    root = memscan.resolve(rest[0]) if rest else memscan.corpus_here(os.getcwd())
    if root is None:
        print("no corpus for this directory; name one", file=sys.stderr)
        return 1
    print(f"corpus: {root}")
    rows = rank(statement, root, top=top)
    for c, f, s in rows:
        print(f"[{c:.3f}] {f}")
        print(f"        {s[:190]}")
    if LAST_RANK_MODE == "lexical":
        print("\nWORD OVERLAP ONLY: no embedder could be reached, so this ranking cannot see a "
              "paraphrase and a silent result is not evidence the corpus is silent.\n"
              + "\n".join("  " + l for l in
                          f"ollama: install it, then  ollama pull {MODEL}\n"
                          f"in-process, no server:  {sys.executable} -m venv {LOCKET_VENV} && "
                          f"{LOCKET_VENV}/bin/python -m pip install fastembed".splitlines()))
    print("\nRanked candidates, not a verdict. Read the top file before writing; "
          "this tool cannot tell you a fact is new.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except RuntimeError as e:
        print(f"memfind: {e}", file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        sys.stderr.close()
        sys.exit(0)
