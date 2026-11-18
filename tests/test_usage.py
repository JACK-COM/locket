"""usage's behaviour suite: the transcript and state.db readers, the ledger, the fold and the SessionEnd hook."""
import datetime as dt
import json
import os
import sqlite3
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path

from tests.harness import fixture_env

import memscan  # noqa: E402
import paths  # noqa: E402
import usage as us  # noqa: E402

def _suite():
    """Offline, against fixture transcripts and a fixture state.db: each response counted
    once, a second record changes nothing, a deleted source keeps its days, the report."""
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        proj = t / "projects/-x"
        (proj / "s1/subagents").mkdir(parents=True)
        today = dt.date.today()
        ts = lambda d: dt.datetime.combine(today - dt.timedelta(days=d), dt.time(12)).astimezone().isoformat()
        cwd = str(memscan.HOME / "code/app")

        def resp(i, d, sid="s1", model="claude-x", **u):
            return json.dumps({"type": "assistant", "timestamp": ts(d), "cwd": cwd, "sessionId": sid,
                               "requestId": f"r{i}", "message": {"id": f"m{i}", "model": model, "usage": {
                                   "input_tokens": 10, "cache_creation_input_tokens": 100,
                                   "cache_read_input_tokens": 1000, "output_tokens": 5, **u}}})
        # one response written three times (three content blocks), a second on another day,
        # a synthetic error line, a user line
        (proj / "s1.jsonl").write_text("\n".join([resp(1, 0)] * 3 + [
            resp(2, 1), resp(9, 0, model="<synthetic>"),
            json.dumps({"type": "user", "message": {"usage": "no"}})]) + "\n")
        # a resumed session copying response 1, and a subagent's own response
        (proj / "s2.jsonl").write_text(resp(1, 0, sid="s2") + "\n")
        (proj / "s1/subagents/a.jsonl").write_text(resp(3, 0, output_tokens=7) + "\n")

        # an older Hermes: sessions only, missing columns; a current one: WAL, with the
        # per-model table carrying an auxiliary call the sessions row leaves out
        old_db, db = t / "old.db", t / "state.db"
        con = sqlite3.connect(old_db)
        con.execute("CREATE TABLE sessions (id TEXT, started_at REAL, model TEXT, input_tokens INT, output_tokens INT)")
        con.execute("INSERT INTO sessions VALUES ('h0', ?, 'glm', 7, 3)", (time.time(),))
        con.commit(); con.close()
        con = sqlite3.connect(db)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE sessions (id TEXT, started_at REAL, cwd TEXT, model TEXT, "
                    "input_tokens INT, output_tokens INT, cache_read_tokens INT)")
        con.execute("CREATE TABLE session_model_usage (session_id TEXT, model TEXT, task TEXT, "
                    "api_call_count INT, input_tokens INT, output_tokens INT, cache_read_tokens INT)")
        con.execute("INSERT INTO sessions VALUES ('h1', ?, NULL, 'glm', 40, 20, 0)", (time.time(),))
        con.executemany("INSERT INTO session_model_usage VALUES ('h1', ?, ?, 1, ?, ?, 0)",
                        [("glm", "", 40, 20), ("tiny", "title_generation", 10, 2)])
        con.commit()    # left open, so the WAL file exists while it is read
        old = us.read_hermes(old_db)
        assert [(k[2], v["input"], v["output"]) for k, v in old.items()] == [("glm", 7, 3)], dict(old)

        ledger = t / "usage.csv"
        rows, notes = us.record(proj.parent, db, ledger=ledger, quiet=True)
        con.close()
        assert not notes, notes
        c = [r for r in rows if r["source"] == "claude"]
        today_c = [r for r in c if r["date"] == today.isoformat()]
        assert len(today_c) == 1 and today_c[0]["responses"] == 2, f"a response counted more than once: {today_c}"
        assert today_c[0]["output"] == 12 and today_c[0]["sessions"] == 1 and today_c[0]["project"] == "~/code/app", today_c
        assert sum(r["responses"] for r in c) == 3, "the other day or the synthetic line is miscounted"
        h = [r for r in rows if r["source"] == "hermes"]
        assert sorted((r["model"], r["input"], r["responses"]) for r in h) == [("glm", 40, 1), ("tiny", 10, 1)], h
        assert all(r["cache_write"] == 0 and r["project"] == "" for r in h), h

        before = ledger.read_text()
        us.record(proj.parent, db, ledger=ledger, quiet=True, full=True)
        assert ledger.read_text() == before, "a second record changed the ledger"

        # the source deletes yesterday's transcript and half of today's: both days stay as recorded
        def days(rows):
            out = defaultdict(int)
            for r in rows:
                out[(r["source"], r["date"])] += r["responses"]
            return dict(out)
        # the incremental record opens only files touched since the day before the last
        # record: a new response lands, and an old file is not reread
        old = time.time() - 5 * 86400
        (proj / "s4.jsonl").write_text(resp(5, 3, sid="s4") + "\n")
        os.utime(proj / "s4.jsonl", (old, old))
        (proj / "s3.jsonl").write_text(resp(4, 0, sid="s3") + "\n")
        rows, _ = us.record(proj.parent, db, ledger=ledger, quiet=True)
        recorded = days(rows)
        assert recorded[("claude", today.isoformat())] == 3, "the incremental record missed a new response"
        assert ("claude", (today - dt.timedelta(days=3)).isoformat()) not in recorded, \
            "the incremental record reread an old file"

        # the source deletes one of today's files and all of yesterday's: neither day shrinks
        (proj / "s1/subagents/a.jsonl").unlink()
        rows, _ = us.record(proj.parent, db, ledger=ledger, quiet=True)
        (proj / "s1.jsonl").unlink()
        rows, _ = us.record(proj.parent, db, ledger=ledger, quiet=True)
        assert days(rows) == recorded, "a day shrank when its source deleted part of it"
        # a missing source is a note, never an error, and keeps what was recorded
        rows, notes = us.record(t / "none", t / "none.db", ledger=ledger, quiet=True, full=True)
        assert len(notes) == 2 and days(rows) == recorded, notes

        s = us.summarize(rows, 7)
        assert s["typical_day"] is None and "not enough history" in us.render(s)
        assert s["by_source"]["claude"]["responses"] == 4 and abs(s["cache_read_share"] - 4000 / 4490) < 1e-9, s
        text = us.render(s, "day")
        assert "claude" in text and "hermes" in text and "by day:" in text, text
        # a long history: the typical day is the median, counting days with nothing as zero
        hist = [{"date": (today - dt.timedelta(days=d)).isoformat(), "source": "claude", "project": "", "model": "m",
                 **{k: 0 for k in us.NUMS}, "responses": 10, "output": 100} for d in range(7, 36) if d % 2]
        s = us.summarize(hist + rows, 7)
        assert s["baseline_days"] == us.BASELINE and s["typical_day"]["output"] == 50, s["typical_day"]
        json.dumps(s)
        (t / "bad.csv").write_text("date,tokens\n")
        try:
            us.load(t / "bad.csv")
            raise AssertionError("a ledger with other columns was read")
        except SystemExit:
            pass
        # the SessionEnd hook stays silent and exits 0 even on that ledger
        import contextlib
        import io
        env = dict(os.environ)
        (t / "st").mkdir()
        (t / "st" / "usage.csv").write_bytes((t / "bad.csv").read_bytes())
        os.environ.update(LOCKET_USAGE_CLAUDE=str(t / "none"), LOCKET_USAGE_HERMES=str(t / "none.db"))
        stdin, sys.stdin = sys.stdin, io.StringIO("{}")
        err = io.StringIO()
        try:
            with paths.use(paths.PATHS.replace(registry=t / "st" / "corpora.json")), \
                    contextlib.redirect_stdout(err), contextlib.redirect_stderr(err):
                assert us.hook() == 0 and not err.getvalue(), f"the hook spoke or failed: {err.getvalue()}"
        finally:
            sys.stdin = stdin
            os.environ.clear(); os.environ.update(env)
    # The fold: every whole month that ended more than FOLD_AFTER days ago becomes one row per
    # month, source, project and model; totals are exact; recent days stay daily; a folded
    # month absorbs a stray day of its own month; and the report never reads a month row.
    f_today = dt.date(2026, 11, 11)
    f_day = lambda d: (f_today - dt.timedelta(days=d)).isoformat()
    f_row = lambda date, src="claude": {"date": date, "source": src, "project": "p", "model": "m",
                                        **{k: 1 for k in us.NUMS}}
    f_rows = [f_row(f_day(d)) for d in range(0, 200)] + [f_row(f_day(d), "hermes") for d in range(150, 200)]
    folded = us.fold(f_rows, f_today)
    cut = us.fold_cutoff(f_today)
    assert cut == "2026-07-01", cut
    months = [r for r in folded if not us._is_day(r)]
    assert months and all(r["date"] < cut[:7] for r in months)
    assert all(r["date"] >= cut for r in folded if us._is_day(r))
    assert len(months) == len({(r["date"], r["source"], r["project"], r["model"]) for r in months}), \
        "a month folded into more than one row"
    assert us._sum(folded) == us._sum(f_rows), "the fold changed a total"
    assert us.fold(folded, f_today) == folded, "a second fold changed the ledger"
    absorbed = us.fold(folded + [f_row("2026-05-14")], f_today)
    assert us._sum(absorbed)["output"] == us._sum(folded)["output"] + 1 \
        and sum(1 for r in absorbed if r["date"] == "2026-05" and r["source"] == "claude") == 1
    s7 = us.summarize(folded, 7, f_today)
    assert s7["total"]["output"] == 7 and s7["baseline_days"] == us.BASELINE, s7["baseline_days"]
    # A source row of a month the ledger already folded is never read back (the clock stepped
    # forward past a fold, then back), and a month never folded passes for the fold to take.
    back = us._unfolded([f_row("2026-05-20"), f_row("2026-05-20", "hermes"), f_row("2026-11-10")], folded)
    assert [(r["date"], r["source"]) for r in back] == [("2026-11-10", "claude")], back
    late = dict(f_row("2026-05-20"), model="a-model-first-seen-later")
    assert us._unfolded([late], folded) == [late], "a new model's day in a folded month was dropped"
    assert us._sum(us.fold(folded + us._unfolded([late], folded), f_today))["output"] == us._sum(folded)["output"] + 1
    fresh_ledger = us.fold(us._unfolded([f_row("2026-03-02", "hermes")], []), f_today)
    assert [r["date"] for r in fresh_ledger] == ["2026-03"], "a new ledger lost a source's old day"
    return 0


class UsageBehaviour(unittest.TestCase):
    def test_behaviour(self):
        with fixture_env():
            _suite()


if __name__ == "__main__":
    unittest.main()
