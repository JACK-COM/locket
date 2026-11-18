"""The linear rewrites answer exactly what the code they replaced answered: each is run
against the 0.11.12 form, kept here as the reference, over random inputs. A speed fix
that changed an answer would be a correctness bug the bounds tests cannot see."""
import random
import unittest
from pathlib import Path

import tests.harness  # noqa: F401  (puts the source under test on sys.path)

import memscan  # noqa: E402
import paths  # noqa: E402

CASES = 20000


def unwrap_reference(txt):
    out = []
    for ln in txt.split("\n"):
        prev = out[-1] if out else None
        if (prev is not None and prev.strip() and not memscan._ENDS.search(prev)
                and not memscan._NO_JOIN_FROM.match(prev)
                and ln.strip() and not memscan._LINE_MEANINGFUL.match(ln)):
            out[-1] = prev.rstrip() + " " + ln.strip()
        else:
            out.append(ln)
    return "\n".join(out)


def cd_reference(command, cwd, pos):
    here = cwd
    for m in memscan._CD.finditer(memscan._shell_code(command)):
        if pos is not None and m.start() >= pos:
            break
        if m.group(1) == "popd":
            here = None
            continue
        d = command[m.start(2):m.end(2)].strip() if m.group(2) is not None else ""
        d = d or None
        if d is None or d == "~":
            here = str(paths.HOME)
        elif any(c in d for c in "$`(") and not d.strip("'\"").startswith(("$HOME", "${HOME}")):
            here = None
        elif d == "-":
            here = None
        else:
            d = memscan._expand_home(d)
            here = d if d.startswith("/") else (str(Path(here) / d) if here else None)
    return [here] if here else []


def score_reference(text, docs):
    """`score_against` as the all-pairs loop the claim index replaced."""
    incoming = list(memscan.claims(memscan.strip(text)))
    if not incoming:
        return []
    inc_all = set().union(*(ts for _, ts in incoming))
    ranked = []
    for name, body in docs.items():
        best, best_pair, overlap = 0.0, None, 0
        for s, ts in memscan.claims(body):
            overlap += len(ts & inc_all)
            for isent, its in incoming:
                jac = memscan.similar(ts, its)
                if jac > best:
                    best, best_pair = jac, (isent, s)
        ranked.append((best + min(overlap / 400.0, 0.35), best, name, best_pair))
    ranked.sort(reverse=True)
    return ranked


def subtract_reference(added, arch):
    if not arch:
        return memscan.strip(added)
    return "\n".join(s for s, ts in memscan.claims(memscan.strip(added))
                     if not any(memscan.similar(ts, a) >= memscan.FIRE_FLOOR for a in arch))


# a small vocabulary, so random sentences share five tokens often enough to pair
VOCAB = [f"w{i:02d}" for i in range(24)] + ["the", "and", "of"]


def _prose(rnd, lines):
    return "\n".join(" ".join(rnd.choice(VOCAB) for _ in range(rnd.randint(2, 12))) + rnd.choice([".", "", ";"])
                     for _ in range(lines))


SHELL = ["sed", "gsed", "-i", "-n", "-Ei", "--in-place", "perl", "ruby", "-pi", "-0pi", "awk", "gawk",
         "-i inplace", "tee", "-a", "cp", "mv", "rsync", "cat", "echo", "x", "a.md", "dir/b.md", "'c.md'",
         '"d.md"', ">", ">>", ";", "&&", "|", "||", "\n", "cd", "cd /tmp", "cd sub", "cd -", "popd",
         "$HOME/e.md", "~/f.md", "s/a/b/", "=g.md"]
PROSE = ["word", "end.", "q?\"", "- item", "# h", "| t", "```", "", "  ", "  indented", "1. x", "> quote",
         "a:", "x)", "\t", "b;", "'", "*", "x\r"]


class Equivalence(unittest.TestCase):
    def test_unwrap(self):
        rnd = random.Random(7)
        for _ in range(CASES):
            txt = "\n".join(" ".join(rnd.choice(PROSE) for _ in range(rnd.randint(0, 4)))
                            for _ in range(rnd.randint(0, 12)))
            self.assertEqual(memscan.unwrap(txt), unwrap_reference(txt), repr(txt))

    def test_score_against(self):
        rnd = random.Random(13)
        for _ in range(CASES // 10):
            docs = {f"f{k}.md": _prose(rnd, rnd.randint(0, 8)) for k in range(rnd.randint(0, 6))}
            text = _prose(rnd, rnd.randint(0, 5))
            want = score_reference(text, docs)
            self.assertEqual(memscan.score_against(text, docs), want, repr((text, docs)))
            index = memscan.ClaimIndex.from_docs(docs)
            self.assertEqual(memscan.score_against(text, index), want, repr((text, docs)))
            # a hidden file and one stood as a write leaves it, against the dict edited the same way
            if docs:
                gone, edited = rnd.choice(sorted(docs)), rnd.choice(sorted(docs))
                swap = {edited: _prose(rnd, 2)}
                ref = {k: v for k, v in docs.items() if k != gone or k == edited}
                ref.update(swap)
                self.assertEqual(memscan.score_against(text, index, hide=lambda n: n == gone, extra=swap),
                                 score_reference(text, ref), repr((text, docs, gone, swap)))

    def test_subtract_archived(self):
        rnd = random.Random(17)
        for _ in range(CASES // 10):
            arch = [ts for _, ts in memscan.claims(_prose(rnd, rnd.randint(0, 10)))]
            added = _prose(rnd, rnd.randint(0, 6))
            self.assertEqual(memscan.subtract_archived(added, arch), subtract_reference(added, arch))

    def test_write_matches(self):
        rnd = random.Random(11)
        for _ in range(CASES):
            cmd = " ".join(rnd.choice(SHELL) for _ in range(rnd.randint(1, 14)))
            for pat, start in zip(memscan.BASH_WRITE, memscan.BASH_WRITE_STARTS):
                got = [(m.start("p"), m.group("p")) for m in memscan.write_matches(pat, start, cmd)]
                want = [(m.start("p"), m.group("p")) for m in pat.finditer(cmd)]
                self.assertEqual(got, want, f"{pat.pattern!r} on {cmd!r}")

    def test_cd_targets(self):
        rnd = random.Random(13)
        for _ in range(CASES // 4):
            cmd = " ".join(rnd.choice(SHELL) for _ in range(rnd.randint(1, 14)))
            for pos in (None, 0, len(cmd) // 2, len(cmd)):
                self.assertEqual(memscan._cd_targets(cmd, "/w", pos), cd_reference(cmd, "/w", pos),
                                 f"{cmd!r} at {pos}")


if __name__ == "__main__":
    unittest.main()
