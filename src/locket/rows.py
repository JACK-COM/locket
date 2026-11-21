"""Rows of one registered CSV compared with each other, with incoming rows, or with
the claims of a markdown text: `locket dupes` on a CSV, reached through
`commands.meaning`. Each answers *which rows say the same thing*, and none of them
decides. memfind, and numpy with it, loads with this module, so nothing on the hook
path imports it.

Why by meaning: a reference table's uniqueness breaks by PARAPHRASE, which word
overlap cannot see. On fourteen groups of rulings a reader had adjudicated as
duplicates (70 pairs), Jaccard at `claims.PAIR_FLOOR` found 1; centred cosine at
0.55 found 49, with no random pair of the same table that high.

Why centred: the embedding space is anisotropic, so raw cosine puts a random pair
of rows near 0.70 and a duplicate near 0.88. Subtracting the table's own mean
vector puts the random pair near 0 and leaves the duplicate standing. The mean is
the TABLE's, never the whole index's: markdown claims would pull it off the rows.

Why it never decides: on a claims table half the pairs above 0.85 were different
facts, the commonest a per-variant certificated-limits row naming another model,
so a high score is no evidence the two rows share a subject. Every result is a
worklist a reader adjudicates.
"""
import numpy as np

import claims
import memfind

import _embed  # noqa: E402  (panoply-lib's embedder ladder, a generated copy beside this file)


class NoEmbedder(RuntimeError):
    """No rung of the embedder ladder answered. Its own class so a caller falling back
    to word overlap catches this and lets every other RuntimeError surface."""


def _embedded(texts, root):
    """`embed` on the rung the store's index was built with, raising NoEmbedder."""
    try:
        return memfind.embed(texts, quiet=True, backend=memfind.index_backend(root))
    except RuntimeError as e:
        raise NoEmbedder(str(e)) from e


def _source_rows(items, key):
    """Index positions of one registered CSV's rows, in file order. `key` is the CSV's
    key from `corpus.csv_keys`, its path in the store, which every row label starts with
    (`<key>#<area>`), so two CSVs sharing a file name stay two tables."""
    prefix = f"{key}#"
    return [i for i, (label, _) in enumerate(items) if label.startswith(prefix)]


def _baseline(unit, samples=20000, seed=0):
    """The 99.9th percentile of centred cosine over random pairs of rows: what
    unrelated rows of THIS table score, so a reader can see how far above chance a
    pair sits without a fixed cutoff that goes stale with the model or the table.
    None under 1,000 pairs (46 rows), where the percentile is a handful of real
    pairs, duplicates among them, rather than chance."""
    n = unit.shape[0]
    if n * (n - 1) // 2 < 1000:
        return None
    rng = np.random.default_rng(seed)
    a, b = rng.integers(n, size=samples), rng.integers(n, size=samples)
    keep = a != b
    return float(np.quantile(np.einsum("ij,ij->i", unit[a[keep]], unit[b[keep]]), 0.999))


def _into_table(vectors, mean):
    """Fresh embeddings moved into a table's centred space. The index holds unit
    vectors and the mean is theirs, so a query is normalised BEFORE the mean comes
    off; centring a raw vector puts it somewhere no row is."""
    return _embed.normalise(_embed.normalise(np.asarray(vectors, dtype=np.float64)) - mean)


def _table(root, name, quiet, idx):
    """(items, positions, centred unit rows, the table mean) for one CSV. Raises
    LookupError when the index holds none of its rows, and NoEmbedder when the index
    has to be built and nothing can embed it."""
    if idx is None:
        try:
            idx = memfind.load_index(root, quiet=quiet)
        except RuntimeError as e:
            raise NoEmbedder(str(e)) from e
    items, vecs, _dim = idx
    pos = _source_rows(items, name)
    if not pos:
        raise LookupError(f"no indexed rows for {name}; is it registered under `sources`?")
    unit, mean = memfind.centre(np.asarray(vecs[pos], dtype=np.float64))
    return items, pos, unit, mean


def row_pairs(root, name, top=30, quiet=False, idx=None):
    """(pairs, baseline): the `top` row pairs inside one CSV ranked by centred cosine,
    each (cosine, label_a, row_a, label_b, row_b), and the table's random-pair
    99.9th percentile. Quadratic in rows, run as blocked matrix products."""
    items, pos, unit, _mean = _table(root, name, quiet, idx)
    pairs = [(c, items[pos[a]][0], items[pos[a]][1], items[pos[b]][0], items[pos[b]][1])
             for c, a, b in _embed.top_pairs(unit, top)]
    return pairs, _baseline(unit)


def incoming_rows(root, name, texts, quiet=False, idx=None):
    """Every incoming row against its nearest existing row of one CSV, highest first,
    as (cosine, incoming, nearest label, nearest row). Nothing is filtered out.

    A cutoff was tried and abandoned twice: an absolute one, because a paraphrase and
    a novel row overlap on score, and a rank-1 margin, because a denser table crowds
    every neighbourhood: a deliberate reword of a live ruling put its owner at rank 1
    by a margin of only 1.15x at 2,400 rows. Ranking held where the trigger did not."""
    items, pos, unit, mean = _table(root, name, quiet, idx)
    texts = [t for t in (s.strip() for s in texts) if t]
    if not texts:
        return []
    q = _into_table(_embedded(texts, root), mean)
    sims = q @ unit.T
    best = sims.argmax(axis=1)
    out = [(float(sims[k, j]), texts[k], items[pos[j]][0], items[pos[j]][1])
           for k, j in enumerate(best)]
    return sorted(out, key=lambda t: -t[0])


def covered_rows(root, name, text, top=30, quiet=False, idx=None):
    """The rows of one CSV a markdown text already states, highest first, as
    (cosine, label, row, the text's closest claim). For a fact promoted from a table
    into a file that loads whole, such as a rule, whose row is now a separately
    editable twin: its reader already holds the fact and never queries for it. The
    text is embedded here rather than read from the index, because the files that
    load whole are often not store members. A low score can still be covered in
    substance and a high one can carry detail the file dropped, so the cut is read
    and made by hand."""
    items, pos, unit, mean = _table(root, name, quiet, idx)
    said = [s for s, _ts in claims.claims(claims.strip(text))]
    if not said:
        raise LookupError("the file holds no claim long enough to compare")
    q = _into_table(_embedded(said, root), mean)
    sims = unit @ q.T
    best = sims.argmax(axis=1)
    out = [(float(sims[k, j]), items[pos[k]][0], items[pos[k]][1], said[j])
           for k, j in enumerate(best)]
    return sorted(out, key=lambda t: -t[0])[:top]
