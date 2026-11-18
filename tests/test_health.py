"""Hook health: a run the host kills is counted, and nothing is lost under contention."""
import os
import sys
import time
import unittest
from unittest import mock

from tests.harness import SRC, StoreCase

import health  # noqa: E402
import paths  # noqa: E402


def _hook_health(fixture):
    """Hook health against a throwaway directory, in this process and in every child it
    spawns; the installed paths and the environment are both restored on the way out."""
    live_dir = health.health_dir()
    d = fixture / "hook-health"
    with paths.use(paths.PATHS.replace(health_dir=d)), mock.patch.dict(os.environ, LOCKET_HEALTH_DIR=str(d)):
        _hook_health_body(d)
    assert health.health_dir() == live_dir, "the test left hook health pointed at its fixture"


def _hook_health_body(d):
    import signal, subprocess
    marks = d / "inflight"
    ls = lambda: sorted(os.listdir(marks))
    with health.HookRun("hook"):
        assert len(ls()) == 1, "a running hook leaves its marker"
    assert not ls(), "a finished hook leaves a marker behind"
    assert health.health_report() == (None, []), "a clean run wrote counters"
    # A run the host kills never reaches its exit: its marker is the only trace. Killed for
    # real, so the test is the mechanism and not a model of it. POSIX only: elsewhere a dead
    # run is known by age alone.
    if os.name == "posix":
        prog = (f"import sys, time; sys.path.insert(0, {str(SRC)!r}); import memscan, claims, corpora, corpus, fsio, gate, health, manifest, trust\n"
                "with health.HookRun('grepassist'):\n    print('in', flush=True); time.sleep(60)\n")
        child = subprocess.Popen([sys.executable, "-c", prog], stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline().strip() == "in"
        child.send_signal(signal.SIGKILL)
        child.wait()
        child.stdout.close()
        assert len(ls()) == 1, "a killed run left no marker"
        rec, problems = health.health_report()         # the report folds: no later hook run needed
        assert rec["verbs"]["grepassist"]["killed"] == 1 and len(problems) == 1, rec
        assert not ls(), "the killed run's marker was not folded"
        assert "?" not in problems[0].split("(last ")[1][:4], problems
    health.health_ack()
    # A live run is never folded early, and a clock stepped back reads as young; past the
    # stale age it folds whoever holds it, which covers a reused PID.
    stale = health.HEALTH_STALE_POSIX if os.name == "posix" else health.HEALTH_STALE_OTHER
    live = marks / health._marker_name("hook", os.getpid(), time.time_ns())
    live.write_text("{}")
    health._fold_dead_markers(now=time.time() - 3600)
    health._fold_dead_markers()
    assert live.exists(), "a live run's marker was folded"
    health._fold_dead_markers(now=time.time() + stale + 1)
    assert not live.exists() and health.health_report()[0]["verbs"]["hook"]["killed"] == 1
    # Counted before removed: with the lock held elsewhere the count cannot land, and the
    # marker must survive for a later fold rather than vanish uncounted.
    dead = marks / health._marker_name("bashguard", 2 ** 22 + 7, time.time_ns() - 10 ** 12)
    dead.write_text("{}")
    with open(d / ".lock", "a+") as held:
        try:
            import fcntl
            fcntl.flock(held, fcntl.LOCK_EX)
            t0 = time.monotonic()
            health._fold_dead_markers(now=time.time() + stale + 1)
            assert time.monotonic() - t0 < 2, "a held lock made the fold wait"
            assert dead.exists(), "a marker was removed although its count could not land"
        except ImportError:
            pass
    health._fold_dead_markers(now=time.time() + stale + 1)
    assert not dead.exists() and health.health_report()[0]["verbs"]["bashguard"]["killed"] == 1
    # A claim or a temp a dead writer left is reclaimed or removed, never kept forever.
    orphan = marks / f".claim-999999-{health._marker_name('hook', 999999, time.time_ns())}"
    orphan.write_text("{}")
    temp = d / ".health-leftover"
    temp.write_text("{")
    old = time.time() - health.HEALTH_TEMP_AGE - 5
    os.utime(orphan, (old, old)); os.utime(temp, (old, old))
    health.health_ack()
    health._fold_dead_markers(now=time.time() + stale + 1)
    assert not ls() and not temp.exists(), (ls(), temp.exists())
    assert health.health_report()[0]["verbs"]["hook"]["killed"] == 1, "an orphaned claim was not counted"
    # The directory is bounded: past the cap the OLDEST fold, by start time across verbs.
    health.health_ack()
    base = time.time_ns()
    for i in range(health.HEALTH_MARKER_CAP + 10):
        verb = ("bashguard", "hook")[i % 2]     # mixed verbs: a name sort must still be a time sort
        (marks / health._marker_name(verb, os.getpid(), base + i)).write_text("{}")
    health._fold_dead_markers()
    kept = ls()
    assert len(kept) == health.HEALTH_MARKER_CAP, len(kept)
    assert kept[0] == health._marker_name("bashguard", os.getpid(), base + 10), "the cap did not fold the oldest"
    for n in kept:
        (marks / n).unlink()
    # A swallowed error is counted with its detail, the record stays one entry per verb,
    # an unreadable record is set aside and said so, and an ack zeroes the counters.
    health.health_ack()
    for _ in range(3):
        health.health_error("hook", "lexical", ValueError("x"))
    rec, problems = health.health_report()
    assert rec["verbs"]["hook"]["errors"] == 3 and set(rec["verbs"]) == {"hook"}, rec
    assert any("swallowed 3 error" in p and "ValueError in lexical" in p for p in problems), problems
    (d / "health.json").write_text("{not json")
    health.health_error("hook", "2e", OSError())
    rec, problems = health.health_report()
    assert (d / "health.json.corrupt").exists() and any("set aside" in p for p in problems), problems
    assert health.health_ack() and health.health_report()[1] == []
    # A record that cannot be read is a problem, never a clean reading.
    (d / "health.json").write_text("[]")
    assert any("cannot be read" in p for p in health.health_report()[1])
    health.health_ack()
    # Under contention nothing is dropped: an error waits as a marker and is counted at the
    # next fold, and a dead marker the fold could not count is reported as pending.
    dead = marks / health._marker_name("hook", 2 ** 22 + 9, time.time_ns() - 10 ** 12)
    dead.write_text("{}")
    with open(d / ".lock", "a+") as held:
        try:
            import fcntl
            fcntl.flock(held, fcntl.LOCK_EX)
            health.health_error("grepassist", "rank", KeyError())
            assert any("not yet counted" in p for p in health.health_report()[1]), "a pending abandoned run read clean"
        except ImportError:
            pass
    rec, problems = health.health_report()
    assert rec["verbs"]["grepassist"]["errors"] == 1 and rec["verbs"]["hook"]["killed"] == 1, rec
    assert any("KeyError in rank" in p for p in problems), problems
    # Only known verbs get a record, and a PID too large for the OS reads as dead.
    health.health_ack()
    (marks / health._marker_name("anything-at-all", 2 ** 70, time.time_ns())).write_text("{}")
    (marks / health._marker_name("other", 2 ** 70, time.time_ns() + 1)).write_text("{}")
    rec, _p = health.health_report()
    assert set(rec["verbs"]) == {"?"} and rec["verbs"]["?"]["killed"] == 2, rec
    assert health.health_ack() and not os.listdir(marks)


class HookHealth(StoreCase):
    COUNCIL = False

    def test_hook_health(self):
        _hook_health(self.tmp)


class TriggerHealth(StoreCase):
    COUNCIL = False

    def test_a_trigger_hook_that_raises_is_counted(self):
        import io
        import trigger
        d = self.tmp / "hook-health"
        with paths.use(paths.PATHS.replace(health_dir=d)), \
                mock.patch.object(trigger, "fire", side_effect=KeyError("x")):
            self.assertEqual(trigger.hook(io.StringIO('{"hook_event_name": "PreToolUse"}')), 0)
            rec, problems = health.health_report()
        self.assertEqual(rec["verbs"]["trigger"]["errors"], 1, rec)
        self.assertTrue(any("KeyError in fire" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
