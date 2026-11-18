"""Claims: every sentence of a memory file with five or more content words, as a token set,
and the overlap scoring that pairs one claim with another. Text in, scores out; nothing
here reads a file.
"""
import re
from collections import Counter, defaultdict
from typing import Callable, Iterable, Iterator, Optional, Union

# A claim: the sentence as written, and the content tokens it is scored by.
Claim = tuple[str, set[str]]
# A scored file, best first: (combined score, claim score, file name, (incoming sentence, its match)).
Ranked = tuple[float, float, str, tuple[str, str]]


STOP = set("""a an the and or but if then than that this these those is are was were be been being
of in on at to for with from by as it its not no we i you they our your their my what which who
when where how why can could should would will may might must do does did done so such very more
most other some any each all both few own same too don now here there into over under again
further once about against between through during before after above below up down out off only
just also because while has have had having me him her them us""".split())


WORD = re.compile(r"[a-z0-9_./-]{3,}")


SENT = re.compile(r"(?<=[.!?;:])\s+|\n")


# A hard-wrapped sentence is one sentence. `SENT` splits on every newline, so
# before it runs a wrapped line is joined to the one before it wherever the
# earlier line does not end a sentence and the later one is not line-meaningful
# markdown (a bullet, a heading, a quote, a table row, a numbered item, a fence,
# a blank). Measured before the join on a hard-wrapped store: 30% of its claims
# stopped at the wrap, and a tail under MIN_TOKENS was in no claim at all, so a
# promotion paraphrased across a wrap was invisible to `graduated` and `across`.
# The version rides in memfind's fingerprint, so a change here re-embeds every
# store once instead of leaving stale claims under an unchanged file.
CLAIMS_VERSION = 2


_ENDS = re.compile(r"""[.!?;:]["')\]*_`]*\s*$""")


_LINE_MEANINGFUL = re.compile(r"""\s*(?:[-*+]\s|#|>|\||\d+[.)]\s|```|$)""")


_NO_JOIN_FROM = re.compile(r"""\s*(?:#|\||```|$)""")   # a bullet's continuation is still its sentence


def unwrap(txt: str) -> str:
    """`txt` with hard wraps inside a sentence joined by a space. Linear: a joined
    line is kept as its segments and joined once, and only its tail is tested for an
    ending, because rebuilding and re-searching the whole line on every wrap made one
    unpunctuated 200 KB file cost seconds inside the write hook."""
    out: list[list[str]] = []               # each line as its segments; may the last take a wrap
    joinable = False
    for ln in txt.split("\n"):
        if (joinable and not _ENDS.search(_tail(out[-1]))
                and ln.strip() and not _LINE_MEANINGFUL.match(ln)):
            out[-1][-1] = out[-1][-1].rstrip()
            out[-1].append(" " + ln.strip())
        else:
            out.append([ln])
            # decided once per line: its start never changes as wraps join onto it
            joinable = bool(ln.strip()) and not _NO_JOIN_FROM.match(ln)
    return "\n".join("".join(c) for c in out)


def _tail(segments: list[str], width: int = 256) -> str:
    """The last `width` or more characters of a line held as segments: all `_ENDS`
    needs, since an ending is a stop followed only by quotes and space."""
    parts, n = [], 0
    for seg in reversed(segments):
        parts.append(seg)
        n += len(seg)
        if n >= width:
            break
    return "".join(reversed(parts))


LINK = re.compile(r"\[\[([A-Za-z0-9_\-]+)\]\]")


# an index line: "- [Title](target.md) — hook text"
IDX_LINE = re.compile(r"^\s*[-*]\s*\[(?P<title>[^\]]+)\]\((?P<target>[a-z0-9_\-]+\.md)\)(?P<hook>.*)$")


MIN_TOKENS = 5          # content tokens a sentence needs to be a claim. Not 7: that


                        # drops short pithy rules like 'exactly two paragraphs'
PAIR_FLOOR = 0.55       # jaccard over claim tokens


BUCKET_CAP = 400        # a token in more claims than this is not a signal


MIN_SHARED = 5          # content tokens two claims must actually share


# A holding space has its own furniture: "Promotion target if this holds across
# instances:" is a stem several entries share, not an entry any of them duplicates.
# The discriminator is SUBSTANCE, not repetition: stems carry 5 to 6 content
# tokens and genuine duplicate entries carry 10 to 14, while both match the same
# number of existing claims at or above FIRE_FLOOR once MIN_SHARED is applied.
ENTRY_TOKENS = 8        # content tokens a line needs to BE an entry, not a stem


def similar(ta: set[str], tb: set[str]) -> float:
    """Jaccard over two token sets, or 0.0 where they share fewer than MIN_SHARED.

    A ratio alone cannot tell a restated fact from two short lines ending in the
    same attribution: `claims` splits on the colon, so a heading arrives as a
    six-token claim whose overlap is mostly a date and the word `ruling`, and
    scores 0.50 against an unrelated heading. A verbatim five-token claim still
    fires, because it shares all five.
    """
    shared = len(ta & tb)
    return shared / len(ta | tb) if shared >= MIN_SHARED else 0.0


def strip(txt: str) -> str:
    """Body text with frontmatter, code fences and markdown furniture removed."""
    if txt.startswith("---"):
        end = txt.find("\n---", 3)
        if end != -1:
            txt = txt[end + 4:]
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = unwrap(txt)                  # before the furniture goes: a heading is still a heading here
    # wikilinks go entirely, not just their brackets: [[a_b_c]] survives
    # bracket-stripping as the word sequence "a b c" and pairs at 1.00 against
    # every other mention of the same slug. A pointer is not a claim.
    txt = re.sub(r"\[\[[^\]]+\]\]", " ", txt)
    return re.sub(r"[*_`#>|\[\]]", " ", txt)


def tokens(s: str) -> set[str]:
    return {w for w in WORD.findall(s.lower()) if w not in STOP}


def claims(txt: str) -> Iterator[Claim]:
    """(sentence, token-set) for every sentence carrying enough to be a claim."""
    for s in SENT.split(txt):
        s = " ".join(s.split())
        ts = tokens(s)
        if len(ts) >= MIN_TOKENS:
            yield s, ts


def unclaimed(txt: str) -> Iterator[str]:
    """The sentences `claims` drops as under the floor, with at least one
    content token. What the instrument cannot see is reported beside what it
    saw, because a check cannot name the claims it never formed and a clean
    result over a corpus it half-indexed reads exactly like a clean corpus."""
    for s in SENT.split(txt):
        s = " ".join(s.split())
        ts = tokens(s)
        if 0 < len(ts) < MIN_TOKENS:
            yield s


def _inverted(items: list[tuple[str, str, set[str]]], df: Counter[str]) -> dict[str, list[int]]:
    """Bucket by the 3 rarest tokens, so candidate generation stays near-linear.

    Ties in rarity break on the token itself: broken by set order they followed
    the per-process string hash, so each run bucketed differently and a pair
    found by one `audit` was missing from the next on an unchanged store."""
    inv: defaultdict[str, list[int]] = defaultdict(list)
    for i, (_, _, ts) in enumerate(items):
        for w in sorted(ts, key=lambda w: (df[w], w))[:3]:
            inv[w].append(i)
    return inv


def cross_file_pairs(docs: dict[str, str], floor: float = PAIR_FLOOR,
                     same_file: bool = False) -> list[tuple[float, str, str, str, str]]:
    items: list[tuple[str, str, set[str]]] = []
    df: Counter[str] = Counter()
    for name, txt in docs.items():
        for s, ts in claims(txt):
            items.append((name, s, ts))
            df.update(ts)
    seen, out = set(), []
    for _, idxs in _inverted(items, df).items():
        if len(idxs) > BUCKET_CAP:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = sorted((idxs[a], idxs[b]))
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                fa, sa, ta = items[i]
                fb, sb, tb = items[j]
                if (fa == fb) != same_file:     # cross-file forks, or within-file dupes
                    continue
                jac = similar(ta, tb)
                if jac >= floor:
                    out.append((jac, fa, fb, sa, sb))
    out.sort(reverse=True)
    return out


# A reference target: a URL, a file with an extension (path included), a path
# anchored at ~, . or / or ending in /, or a ledger query (`Area=substrate`).
# A bare slash is not a path: "AD/mandate" and "CE-525 / CE-525S" are prose, so an
# unanchored extensionless path ("memory/projects") is not a target either. The
# extension list is the memory formats; a generic `.ext` would take "Node.js" and
# "e.g.". Any `k=v` counts, harmless since a claim keeps its tokens past the floor.
# Matched on the stripped sentence, so backticks are gone.
_TARGET = re.compile(r"""https?://\S+|[\w~./-]*\.(?:md|csv|jsonl?|py|sh|txt|toml|ya?ml|pdf)\b|(?<![\w/])[~.]?/[\w./-]+|\b[\w.-]+/(?=\s|$)|\b\w+=[\w.-]+""", re.I)


_REDIRECT = re.compile(r"(?:see|cf\.?|composes)\s")   # a verb whose object is the reference


def pointer_only(sentence: str) -> bool:
    """True where a sentence's payload is a cross-reference, not a claim.

    The criterion: an opening redirect ("See X", "Composes Y"), or a sentence naming a
    reference target that is no longer a claim once the target is removed, by
    the same MIN_TOKENS floor `claims` uses. These repeat across files by design
    and are the noise floor of `audit`. It replaced a closed list of opening
    phrases, under which a pointer worded any other way paired as a fork.
    """
    s = sentence.lower().lstrip("-* ")
    if _REDIRECT.match(s):
        return True
    if not _TARGET.search(s):
        return False
    return len(tokens(_TARGET.sub(" ", s))) < MIN_TOKENS


class ClaimIndex:
    """The claims of many files behind posting lists, so a scorer reads only the
    claims that can share MIN_SHARED tokens with the one it holds.

    Exact, never approximate: a claim of k tokens shares MIN_SHARED with another
    only if it shares one of its k - MIN_SHARED + 1 rarest (the prefix filter), so
    those postings hold every pair `similar` scores above 0. Comparing every pair
    was 2.8 s of a 3.1 s score for a 20 KB write at ten times the council corpus.
    """

    def __init__(self, files: Iterable[tuple[str, Iterable[Claim]]] = ()):
        self.names: list[str] = []
        self.items: list[tuple[int, str, set[str]]] = []    # (file number, sentence, tokens)
        self.postings: defaultdict[str, list[int]] = defaultdict(list)    # token: item numbers, ascending
        for name, file_claims in files:
            self.add(name, file_claims)

    @classmethod
    def from_docs(cls, docs: dict[str, str]) -> "ClaimIndex":
        """From {name: stripped body}, the shape `corpus()` returns."""
        return cls((name, claims(body)) for name, body in docs.items())

    def add(self, name: str, file_claims: Iterable[Claim]) -> None:
        f = len(self.names)
        self.names.append(name)
        items, postings = self.items, self.postings
        for s, ts in file_claims:
            i = len(items)
            items.append((f, s, ts))
            for w in ts:
                postings[w].append(i)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def candidates(self, ts: set[str]) -> list[int]:
        """Item numbers of every claim sharing MIN_SHARED tokens with `ts`, ascending."""
        if len(ts) < MIN_SHARED:
            return []
        empty: tuple[int, ...] = ()
        rare = sorted(ts, key=lambda w: (len(self.postings.get(w, empty)), w))[:len(ts) - MIN_SHARED + 1]
        found: set[int] = set()
        for w in rare:
            found.update(self.postings.get(w, empty))
        return sorted(i for i in found if len(ts & self.items[i][2]) >= MIN_SHARED)

    def holds(self, ts: set[str], floor: float) -> bool:
        """True where some claim here scores `floor` or more against `ts`."""
        return any(similar(ts, self.items[i][2]) >= floor for i in self.candidates(ts))


def _ranked(index: ClaimIndex, incoming: list[Claim], inc_all: set[str],
            hidden: Callable[[str], bool]) -> list[Ranked]:
    """`score_against`'s rows for one index, skipping the names `hidden` drops."""
    items = index.items
    overlap = Counter(items[i][0] for w in inc_all for i in index.postings.get(w, ()))
    best: dict[int, tuple] = {}                 # file number: (score, order, pair)
    for k, (isent, its) in enumerate(incoming):
        for i in index.candidates(its):
            f, s, ts = index.items[i]
            jac, cur = similar(ts, its), best.get(f)
            # the first pair at the best score wins, file order then incoming order,
            # as the all-pairs loop this replaced chose it
            if cur is None or jac > cur[0] or (jac == cur[0] and (i, k) < cur[1]):
                best[f] = (jac, (i, k), (isent, s))
    rows = []
    for f, name in enumerate(index.names):
        if hidden(name):
            continue
        jac, _, pair = best.get(f, (0.0, None, None))
        # claim-level match dominates; bulk token overlap breaks ties so a
        # paraphrase still surfaces the right topical neighbour.
        rows.append((jac + min(overlap[f] / 400.0, 0.35), jac, name, pair))
    return rows


def score_against(text: str, docs: Union[dict[str, str], ClaimIndex], hide: Optional[Callable[[str], bool]] = None,
                  extra: Optional[dict[str, str]] = None) -> list[Ranked]:
    """Rank files by how much of `text` they already assert.

    `docs` is {name: stripped body} or a ClaimIndex. `hide(name)` drops a file from
    it, and `extra` ({name: stripped body}) is scored in place of any file of the
    same name, so a caller can stand a file as a write will leave it without
    rebuilding the index. Returns (combined, claim_score, filename, (incoming,
    existing)) best first, empty where `text` carries no claim.
    """
    incoming = list(claims(strip(text)))
    if not incoming:
        return []
    index = docs if isinstance(docs, ClaimIndex) else ClaimIndex.from_docs(docs)
    inc_all = set().union(*(ts for _, ts in incoming))
    extra = extra or {}
    ranked = _ranked(index, incoming, inc_all, lambda n: n in extra or bool(hide and hide(n)))
    if extra:
        ranked += _ranked(ClaimIndex.from_docs(extra), incoming, inc_all, lambda n: False)
    ranked.sort(reverse=True)
    return ranked


# Fire only on a real claim-level restatement, which is the CLAIM score and never
# the combined one. The bulk-overlap term alone reaches 0.35, so gating on the
# combined score fires on novel text at claim scores of 0.12 to 0.25 wherever it
# shares a vocabulary with the corpus.
FIRE_FLOOR = 0.45
