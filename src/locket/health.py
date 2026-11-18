"""Hook health: a marker per live hook run, folded into per-verb counters when the run ends
or dies, so a hook that crashes, hangs or is killed is counted and reported by doctor
instead of passing silently.
"""
import json
import os
import time
from pathlib import Path

import fsio
import paths


# Hook health: whether a hook run was abandoned or failed open, read from live runs.
# A hook the host kills at its timeout, or one whose catch-all swallows an error, lets
# the call through and says nothing; a doctor probe sees only the runs it makes itself.
# Both surfaces are bounded by construction (RULINGS.csv on Area=substrate): a run's
# marker lives only while the run does, so the directory is empty at rest and capped,
# and the counters are one record per known hook verb, rewritten in place.
HEALTH_VERBS = ("hook", "bashguard", "grepassist")


HEALTH_MARKER_CAP = 256          # past this the oldest count as abandoned: that many live hooks is a fault


HEALTH_STALE_POSIX = 3600        # a hook alive this long is past every timeout (doctor caps them at 600 s)


HEALTH_STALE_OTHER = 600         # no liveness probe off POSIX, so age alone: past any hook timeout


HEALTH_LOCK_WAIT = 0.2           # seconds; a stuck lock holder costs a retry, never a hang


HEALTH_TEMP_AGE = 60             # seconds; a temp or claim left this long belongs to a dead writer


def health_dir():
    """Where hook health lives: `Paths.health_dir`, read at call time."""
    return paths.PATHS.health_dir


def _pid_alive(pid):
    """Whether a process exists. Off POSIX there is no harmless probe (os.kill there
    terminates), so the answer is always yes and a marker folds on age alone."""
    if os.name != "posix":
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError, ValueError):
        return False
    except OSError:                         # exists, owned by someone else
        return True
    return True


def _utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


class _HealthLock:
    """A non-blocking exclusive lock on `<dir>/.lock`, retried for HEALTH_LOCK_WAIT. `held`
    is False when it could not be taken, and the caller then drops or defers its update:
    a hook waiting here would be killed by its host and feed the record it guards."""

    def __init__(self, d):
        self.d, self.fh, self.held = d, None, False

    def __enter__(self):
        try:
            import fcntl
            take = lambda fh: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            try:
                import msvcrt
            except ImportError:
                return self                 # no lock primitive: the update is dropped, never run unlocked
            def take(fh):
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            self.fh = open(self.d / ".lock", "a+")
        except OSError:
            return self
        deadline = time.monotonic() + HEALTH_LOCK_WAIT
        while True:
            try:
                take(self.fh)
                self.held = True
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    return self
                time.sleep(0.01)

    def __exit__(self, *exc):
        if self.fh is not None:
            self.fh.close()                 # closing releases either lock
        return False


def _valid_record(rec):
    return (isinstance(rec, dict) and isinstance(rec.get("verbs"), dict)
            and all(isinstance(v, dict) for v in rec["verbs"].values()))


def _health_update(change, d=None):
    """Apply `change(record)` to the counters under the lock and replace the file whole.
    True when it landed. Never raises: telemetry must not break the hook it watches. An
    unreadable record is set aside as `health.json.corrupt` (one file, overwritten) and the
    fresh record says so, rather than zeroing the history in silence."""
    d = health_dir() if d is None else Path(d)
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _HealthLock(d) as lock:
            if not lock.held:
                return False
            f = d / "health.json"
            try:
                rec = json.loads(f.read_text())
                if not _valid_record(rec):
                    raise ValueError("not a health record")
            except FileNotFoundError:
                rec = {"since": _utc(), "verbs": {}}
            except Exception:
                try:
                    os.replace(f, d / "health.json.corrupt")
                    note = f"an unreadable record was set aside as health.json.corrupt at {_utc()}"
                except OSError:
                    note = f"an unreadable record was overwritten at {_utc()}"
                rec = {"since": _utc(), "verbs": {}, "reset": note}
            change(rec)
            fsio._replace_atomic(f, json.dumps(rec, indent=1, sort_keys=True), ".health-")
        return True
    except Exception:
        return False


def _count(rec, verb, kind, detail):
    verb = verb if verb in HEALTH_VERBS else "?"     # one record per known verb: bounded
    v = rec["verbs"].setdefault(verb, {"killed": 0, "errors": 0})
    v[kind] = int(v.get(kind, 0)) + 1
    v["last_killed" if kind == "killed" else "last_error"] = dict(detail, at=_utc())


def _marker_name(verb, pid, start_ns):
    """Start time first, so a name sort is a time sort and the cap folds the oldest."""
    return f"{start_ns:020d}-{pid}-{verb}"


def health_error(verb, where, exc):
    """Record an error a hook swallowed to let the call through. When the counters cannot
    be written now, the error is left as a marker of a finished process, which the next
    fold counts, so contention delays the count and never drops it."""
    what = type(exc).__name__
    if _health_update(lambda rec: _count(rec, verb, "errors", {"where": where, "what": what})):
        return
    try:
        marks = health_dir() / "inflight"
        marks.mkdir(mode=0o700, parents=True, exist_ok=True)
        (marks / _marker_name(f"{verb}!error", 0, time.time_ns())).write_text(
            json.dumps({"where": str(where)[:40], "what": what}))
    except Exception:
        pass


def _fold_dead_markers(d=None, now=None):
    """Count every run that ended without removing its marker, the trace of a run the host
    abandoned, and every deferred error, then remove those markers. Counted before removed:
    when the count cannot land, each marker is put back for a later fold, so evidence is
    never destroyed uncounted. Returns how many dead markers remain uncounted.

    A marker is claimed by an atomic rename first, so two folds count it once; a claim whose
    folder died is reclaimed once it has been a claim for HEALTH_TEMP_AGE. Temps a dead
    writer left are removed here too. Past HEALTH_MARKER_CAP markers the oldest fold
    whatever their state, so the directory stays bounded."""
    d = health_dir() if d is None else Path(d)
    marks = d / "inflight"
    now = time.time() if now is None else now
    stale = HEALTH_STALE_POSIX if os.name == "posix" else HEALTH_STALE_OTHER
    try:
        names = sorted(os.listdir(marks))
    except OSError:
        return 0
    plain = [n for n in names if not n.startswith(".")]
    excess = len(plain) - HEALTH_MARKER_CAP
    take = []                               # (original name, verb, age)
    for i, n in enumerate(plain):
        try:
            start_ns, pid, verb = n.split("-", 2)
            age = max(0.0, now - int(start_ns) / 1e9)   # a clock stepped back reads as young
            pid = int(pid)
        except ValueError:
            take.append((n, "?", 0.0)); continue
        if i < excess or verb.endswith("!error") or age > stale or not _pid_alive(pid):
            take.append((n, verb, age))
    for n in names:                         # claims and finished markers a dead process left
        if not n.startswith("."):
            continue
        try:
            old = now - (marks / n).stat().st_mtime > HEALTH_TEMP_AGE
        except OSError:
            continue
        if n.startswith(".claim-") and old and n.count("-") >= 2:
            orig = n.split("-", 2)[2]
            try:
                os.replace(marks / n, marks / orig)     # reclaimable like any marker
                start_ns, _pid, verb = orig.split("-", 2)
                take.append((orig, verb, max(0.0, now - int(start_ns) / 1e9)))
            except (OSError, ValueError):
                pass
        elif n.startswith(".done-") and old:
            try:
                (marks / n).unlink()
            except OSError:
                pass
    claimed = []
    for n, verb, age in take:
        c = marks / f".claim-{os.getpid()}-{n}"
        try:
            os.rename(marks / n, c)
            os.utime(c)                     # a claim's age is time since claimed, not since the run began
        except OSError:
            continue                        # another fold claimed it
        try:
            detail = json.loads(c.read_text())
            detail = detail if isinstance(detail, dict) else {}
        except Exception:
            detail = {}
        claimed.append((c, n, verb, age, detail))
    if claimed:
        def change(rec):
            for _c, _n, verb, age, detail in claimed:
                if verb.endswith("!error"):
                    _count(rec, verb[:-len("!error")], "errors",
                           {"where": str(detail.get("where", "?")), "what": str(detail.get("what", "?"))})
                else:
                    _count(rec, verb, "killed", {"cwd": str(detail.get("cwd", ""))[:200],
                                                 "seconds": round(age, 1)})
        landed = _health_update(change, d)
        for c, n, *_ in claimed:
            try:
                if landed:
                    c.unlink()
                else:
                    os.replace(c, marks / n)
            except OSError:
                pass
        if not landed:
            return len(claimed)
    for t in d.glob(".health-*"):           # temps a writer killed mid-replace left behind
        try:
            if now - t.stat().st_mtime > HEALTH_TEMP_AGE:
                t.unlink()
        except OSError:
            pass
    return 0


class HookRun:
    """`with HookRun(verb):` around one hook invocation: leaves a marker for this run,
    folds the traces of earlier abandoned runs, and removes the marker on the way out. A
    run the host kills never reaches the exit, so its marker is what a later fold counts.
    The marker is written before the fold, so a run killed while folding is still seen."""

    def __init__(self, verb):
        self.verb, self.marker = verb, None

    def __enter__(self):
        try:
            d = health_dir()
            marks = d / "inflight"
            marks.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.listdir(marks)               # unlistable: no fold could ever bound it, so write nothing
            try:
                cwd = os.getcwd()
            except OSError:
                cwd = ""
            m = marks / _marker_name(self.verb, os.getpid(), time.time_ns())
            m.write_text(json.dumps({"cwd": cwd}))
            self.marker = m
            _fold_dead_markers(d)
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        if self.marker is not None:
            try:
                self.marker.unlink()
            except OSError:                 # held open elsewhere: mark it finished, never abandoned
                try:
                    os.replace(self.marker, self.marker.with_name(".done-" + self.marker.name))
                except OSError:
                    pass
        return False


def health_report(d=None, fold=True, kinds=("killed", "errors")):
    """(record, problems): the counters, and one line per verb and kind in `kinds` with a
    count since they were last acknowledged. Folds first, so an abandoned run shows even
    when no hook has run since; a dead marker the fold could not count yet, and a record
    that cannot be read, are problems too, never a clean reading."""
    d = health_dir() if d is None else Path(d)
    pending = _fold_dead_markers(d) if fold else 0
    out = [f"{pending} abandoned run(s) not yet counted (the record was busy)"] if pending else []
    f = d / "health.json"
    try:
        rec = json.loads(f.read_text())
        if not _valid_record(rec):
            raise ValueError("not a health record")
    except FileNotFoundError:
        return None, out
    except Exception:
        return None, out + [f"the hook-health record {f} cannot be read; the next write sets it aside"]
    for verb, v in sorted(rec["verbs"].items()):
        if "killed" in kinds and v.get("killed"):
            lk = v.get("last_killed") or {}
            out.append(f"{verb} was abandoned {v['killed']}x (last {lk.get('at', '?')}, "
                       f"{lk.get('seconds', '?')}s, in {lk.get('cwd') or '?'}); those calls ran unchecked")
        if "errors" in kinds and v.get("errors"):
            le = v.get("last_error") or {}
            out.append(f"{verb} swallowed {v['errors']} error(s) (last {le.get('at', '?')}: "
                       f"{le.get('what', '?')} in {le.get('where', '?')}); those calls ran unchecked")
    if rec.get("reset"):
        out.append(str(rec["reset"]))
    return rec, out


def health_ack(d=None):
    """Zero the counters and restart `since`: the reading is seen. True when it landed."""
    def change(rec):
        rec.clear(); rec.update({"since": _utc(), "verbs": {}})
    return _health_update(change, d)
