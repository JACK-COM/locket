"""Every reader that finds heredocs in a shell command agrees with bash about which
lines are code. bash is the oracle: each generated script marks its lines with
`echo L<k>`, bash prints `L<k>` for a line it runs and `echo L<k>` for one it feeds to
a heredoc reader, and each reader's answer for the same line is compared with that.

A reader that takes a code line for heredoc body hides it from every guard that reads
the command by name, which is how `cat <<<x` hid a trust-verb line from bashguard."""
import os
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import tests.harness  # noqa: F401  (puts the source under test on sys.path)

import bashguard  # noqa: E402
import memscan  # noqa: E402
import shellwrites  # noqa: E402
import pywrites  # noqa: E402
import gate  # noqa: E402
import shellscan  # noqa: E402

BASH = shutil.which("bash")
CASES = 250
WORDS = ["A", "A-B", "E.1", "EF"]
OPENERS = ["cat <<{w}", "cat <<-{w}", "cat <<'{w}'", 'cat <<"{w}"', "cat <<\\{w}", "cat << {w}",
           "cat <<{w}; cat <<{v}", "cat <<{w} # note", "cat<<{w}",
           "cat <<<{w}", "cat <<< {w}", 'echo "x <<{w}"', "echo 'x <<{w}'", "# <<{w}",
           "echo $((1<<2))", "echo ${{x//<<{w}/y}}", "echo ${{x// << {w}/y}}", "echo $[1<<2]",
           "echo $[ 1 << 2 ]", "a[1<<2]=3", "x=abcdefgh; echo ${{x:1<<1:2}}", "cat <<$'{w}'",
           "cat <<E\\\nF", 'echo "${{x:-<<{w}}}"']
CLOSERS = ["{w}", "\t{w}", "{w} ", " {w}"]
# The trust verb is built from parts: bashguard refuses a command that names it, and
# this file is read by commands.
VERB = "locket " + "tr" + "ust ~/x"


def script(rng):
    """(lines, marker line indexes) for one random script."""
    lines, marks = [], []
    for _ in range(rng.randint(2, 9)):
        w, v = rng.choice(WORDS), rng.choice(WORDS)
        kind = rng.random()
        if kind < 0.4:
            marks.append(len(lines))
            lines.append(f"echo L{len(lines)}")
        elif kind < 0.7:
            lines.append(rng.choice(OPENERS).format(w=w, v=v))
        else:
            lines.append(rng.choice(CLOSERS).format(w=w))
    return lines, marks


def bash_runs(lines):
    """The marker line indexes bash executes."""
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "s.sh"
        p.write_text("\n".join(lines) + "\n")
        out = subprocess.run([BASH, str(p)], capture_output=True, text=True, cwd=t,
                             env={"PATH": os.defpath}, timeout=10).stdout
    return {int(x[1:]) for x in out.split("\n") if re.fullmatch(r"L\d+", x)}


def offsets(lines):
    out, pos = [], 0
    for ln in lines:
        out.append(pos)
        pos += len(ln) + 1
    return out


def readers(lines, marks):
    """{reader name: the marker indexes it reads as code}."""
    cmd = "\n".join(lines)
    at = offsets(lines)
    code = shellscan.shell_code(cmd)
    kept = set(shellscan.strip_heredocs(cmd).split("\n"))
    text = set(shellscan.command_text(cmd).split("\n"))
    py_lines = [ln.replace("cat", "python3") for ln in lines]
    py_at = offsets(py_lines)
    spans = [(s, e) for _src, s, e, _u in pywrites.programs("\n".join(py_lines))]
    return {
        "_shell_code": {k for k in marks if code[at[k]:at[k] + len(lines[k])] == lines[k]},
        "_strip_heredocs": {k for k in marks if lines[k] in kept},
        "command_text": {k for k in marks if lines[k] in text},
        "pywrites.programs": {k for k in marks if not any(s <= py_at[k] < e for s, e in spans)},
    }


@unittest.skipIf(BASH is None, "bash is the oracle")
class AgainstBash(unittest.TestCase):
    def test_readers_agree_with_bash(self):
        rng = random.Random(13)
        for _ in range(CASES):
            lines, marks = script(rng)
            truth = bash_runs(lines)
            # the Python variant swaps `cat` for `python3`, which bash parses identically
            for name, got in readers(lines, marks).items():
                self.assertEqual(got, truth, f"{name} on:\n" + "\n".join(lines))


class TrustVerbHidden(unittest.TestCase):
    """bashguard's name refusal reads `command_text`; no line before the verb may hide it."""

    def seen(self, command):
        return bool(bashguard._TRUST_VERB.search(re.sub(r"""['"\\]""", "", shellscan.command_text(command))))

    def test_verb_after_each_shape_is_seen(self):
        for before in ("cat <<<x", "cat <<< x", "cat <<A-B\nx\nA-B", 'echo "a <<b"', "# <<b",
                       "echo $((1<<2))", "cat <<-T\n\tx\n\tT", "cat <<P; cat <<Q\nx\nP\ny\nQ",
                       "cat <<E\nx\nE \nE"):
            self.assertTrue(self.seen(before + "\n" + VERB), before)

    def test_verb_inside_a_data_heredoc_is_not(self):
        for body in ("cat <<'EOF'\n" + VERB + "\nEOF", "git commit -F - <<EOF\n" + VERB + "\nEOF",
                     'git commit -m "$(cat <<\'EOF\'\n' + VERB + '\nEOF\n)"'):
            self.assertFalse(self.seen(body), body)


class Bodies(unittest.TestCase):
    def test_python_dash_body_is_tab_stripped(self):
        cmd = "python3 - <<-EOF\n\timport os\n\tos.remove('x')\n\tEOF\n"
        [(src, _s, _e, unreadable)] = pywrites.programs(cmd)
        self.assertEqual(src, "import os\nos.remove('x')")
        self.assertTrue(unreadable is False)

    def test_quotes_inside_a_substituted_heredoc(self):
        # the body's own quote must not open a string that runs past the delimiter
        cmd = 'git commit -m "$(cat <<\'EOF\'\nsays "hi\nEOF\n)"\ncd /elsewhere\n'
        self.assertIn("cd /elsewhere", shellscan.shell_code(cmd))


class Comments(unittest.TestCase):
    def test_an_escaped_hash_starts_no_comment(self):
        for cmd in ("echo a\\ #b; echo L1", "echo a\\\n#b; echo L1", "echo ${x:- #}; echo L1"):
            self.assertIn("echo L1", shellscan.shell_code(cmd), cmd)

    def test_a_cd_after_an_escaped_hash_moves(self):
        self.assertEqual(shellwrites.cd_targets("echo a\\ #b; cd /x\n", "/y"), ["/x"])


class Interpreters(unittest.TestCase):
    def test_body_is_code_where_its_pipeline_runs_an_interpreter(self):
        for cmd in ("env bash <<EOF\n" + VERB + "\nEOF", "cat <<EOF | sh\n" + VERB + "\nEOF",
                    'bash -c "$(cat <<EOF\n' + VERB + '\nEOF\n)"'):
            self.assertTrue(TrustVerbHidden.seen(None, cmd), cmd)

    def test_a_script_name_is_no_interpreter(self):
        for cmd in ("cat <<EOF > run.sh\n" + VERB + "\nEOF", "cat <<EOF > node-notes.md\n" + VERB + "\nEOF"):
            self.assertFalse(TrustVerbHidden.seen(None, cmd), cmd)


class Segments(unittest.TestCase):
    """A separator splits a command only where the shell acts on it."""

    def test_memfind_named_in_a_string_does_not_run_it(self):
        for cmd in ('git commit -m "(memfind.py q)"', 'echo "a; memfind.py q"', "# x; memfind.py q",
                    "cat <<A\nmemfind.py q\nA\necho done", 'echo "x\ny; memfind.py q"'):
            self.assertFalse(gate.runs_memfind(cmd), cmd)

    def test_memfind_after_a_real_separator_runs(self):
        for cmd in ("cd x && python3 memfind.py q", 'echo "a;b"; memfind.py q', "cat <<A\nx\nA\nmemfind.py q",
                    "(memfind.py q)", "x=1 memfind.py q | head"):
            self.assertTrue(gate.runs_memfind(cmd), cmd)


if __name__ == "__main__":
    unittest.main()
