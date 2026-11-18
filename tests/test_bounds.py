"""Model-written input never buys a slow hook: each shape below cost seconds to minutes
in 0.11.12 and must now finish in well under a second. A hook past its timeout is
abandoned by the host and the call lands unchecked, so slowness here is a bypass.
Probes run the source named by LOCKET_SRC (default: this checkout) in a subprocess."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.harness import CMD, SRC

LIMIT = 3.0     # seconds, for slow shared runners; every shape took 3 s or more before the fix

SHAPES = {
    "fold, a name doubled 22 times": (
        "shellwrites.bash_write_targets",
        "\"python3 - <<'EOF'\\nx0='a'\\n\" + ''.join(f'x{i}=x{i-1}+x{i-1}\\n' for i in range(1, 23))"
        " + \"open(x22+'.md','w')\\nEOF\""),
    "sed -i repeated in one clause": ("shellwrites.bash_write_targets", "'sed -i ' * 600 + 'a'"),
    "perl -pi repeated in one clause": ("shellwrites.bash_write_targets", "'perl -pi ' * 500 + 'a'"),
    "tee before one 32 KB word": ("shellwrites.bash_write_targets", "'tee ' * 8000 + 'a' * 32000 + ' x.md'"),
    "a deep chain of cd": ("shellwrites.bash_write_targets", "'cd x; ' * 5000 + 'cp a.md b.md; ' * 1500"),
    "30,000 unclosed parens": ("shellwrites.bash_write_targets", "'((' * 15000 + ' x > a.md'"),
    "10,000 heredoc openers on one line": ("shellwrites.bash_write_targets", "'cat <<a ' * 10000 + '\\n' + 'a\\n' * 10000"),
    "15,000 nested quoted substitutions": ("shellwrites.bash_write_targets", "'cd x; ' + '\"$(' * 15000 + ')\"' * 15000 + ' > a.md'"),
    "20,000 redirects before a csv": ("bashguard.ledger_bash_decision", "'>' * 20000 + ' a.csv'"),
    "25,000 unpunctuated lines": ("claims.unwrap", "('w ' * 8 + '\\n') * 25000"),
    "a long indented line then 5,000 wraps": ("claims.unwrap", "' ' * 100000 + 'x\\n' + 'w\\n' * 5000"),
}


class Bounds(unittest.TestCase):
    def time_call(self, func, expr):
        code = (f"import sys, time; sys.path.insert(0, {str(SRC)!r}); import importlib\n"
                f"arg = {expr}\nmod, name = {func!r}.rsplit('.', 1)\nf = getattr(importlib.import_module(mod), name)\n"
                "args = (arg,) if f.__name__ == 'unwrap' else (arg, '/tmp')\n"
                "t0 = time.monotonic(); f(*args); print(time.monotonic() - t0)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        return float(r.stdout.strip())

    def test_each_shape_is_bounded(self):
        for label, (func, expr) in SHAPES.items():
            with self.subTest(label):
                try:
                    took = self.time_call(func, expr)
                except subprocess.TimeoutExpired:
                    self.fail(f"{label}: still running after 30 s")
                self.assertLess(took, LIMIT, f"{label}: {took:.2f} s")


class Oversized(unittest.TestCase):
    def test_a_padded_write_to_memory_is_still_caught(self):
        with tempfile.TemporaryDirectory() as t:
            home = Path(t).resolve()
            mem = home / ".claude" / "memory"
            mem.mkdir(parents=True)
            (mem / "MEMORY.md").write_text("A fact.\n")
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("LOCKET_") and k not in ("CODEX_HOME", "HERMES_HOME", "CLAUDE_AUTONOMOUS")}
            env.update(HOME=str(home), LOCKET_REGISTRY=str(home / ".locket" / "corpora.json"),
                       LOCKET_HEALTH_DIR=str(home / "health"))
            command = "echo " + "x " * 36000 + f"> {mem / 'MEMORY.md'}"
            payload = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "session_id": "s",
                       "cwd": str(home), "tool_input": {"command": command}}
            r = subprocess.run([*CMD, "bashguard"], env=env, input=json.dumps(payload),
                               capture_output=True, text=True, timeout=60)
            self.assertIn("MEMORY.md", r.stdout + r.stderr, "a write past the scan cap still draws the guard")


if __name__ == "__main__":
    unittest.main()
