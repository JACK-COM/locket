"""The linear rewrites answer exactly what the code they replaced answered: each is run
against the 0.11.12 form, kept here as the reference, over random inputs. A speed fix
that changed an answer would be a correctness bug the bounds tests cannot see."""
import random
import unittest
from pathlib import Path

import tests.harness  # noqa: F401  (puts the source under test on sys.path)

import memscan  # noqa: E402

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
            here = str(memscan.HOME)
        elif any(c in d for c in "$`(") and not d.strip("'\"").startswith(("$HOME", "${HOME}")):
            here = None
        elif d == "-":
            here = None
        else:
            d = memscan._expand_home(d)
            here = d if d.startswith("/") else (str(Path(here) / d) if here else None)
    return [here] if here else []


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
