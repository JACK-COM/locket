"""What every suite shares: the source under test, the command under test, and the
scratch environment the behaviour suites run in.

LOCKET_CMD names the command under test (default: this checkout's locket.py), and
LOCKET_SRC the source the in-process suites import. Point both at an older release to
see a case fail there:  LOCKET_CMD=/opt/homebrew/bin/locket LOCKET_SRC=<its src/locket>.
The subprocess suites (test_trust, test_bounds, test_output, test_reads) run against any
release; the in-process behaviour suites need a source that has `paths.py` (0.13+).
"""
import contextlib
import os
import shlex
import sys
import tempfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("LOCKET_SRC", REPO / "src/locket"))
CMD = shlex.split(os.environ.get("LOCKET_CMD", f"{sys.executable} {REPO / 'src/locket/locket.py'}"))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@contextlib.contextmanager
def fixture_env():
    """Every Locket path under a scratch directory, in this process and in each child it
    spawns, so the live registry, cache, hook health, grep log and refusal records are
    never read or written. Synthetic stores under the temp dir act as trusted and a
    person's `yes` is assumed, so a behaviour suite reads a store as a person's own; the
    trust boundary itself is test_trust.py's, against the real check."""
    import memscan
    import paths
    with tempfile.TemporaryDirectory(prefix="locket-test-") as t:
        p = paths.Paths.under(Path(t) / "locket", host_stores=paths.PATHS.host_stores)
        tmp = Path(tempfile.gettempdir()).resolve()
        real_trusted = memscan.trusted

        def fixture_trusted(d):
            q = Path(d).resolve()
            return q == tmp or tmp in q.parents or real_trusted(d)

        env = {"LOCKET_REGISTRY": str(p.registry), "LOCKET_HEALTH_DIR": str(p.health_dir),
               "LOCKET_GREPASSIST_LOG": str(p.grepassist_log)}
        with paths.use(p), mock.patch.dict(os.environ, env), \
                mock.patch.object(memscan, "trusted", fixture_trusted), \
                mock.patch.object(memscan, "person_confirms", lambda *_a, **_k: True):
            memscan.settings.cache_clear()
            try:
                yield p
            finally:
                memscan.settings.cache_clear()
