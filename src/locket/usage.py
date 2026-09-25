"""usage: how many tokens your agents spent, against what a typical day spends.

Claude Code writes every response's token counts into its transcripts and deletes each
transcript after 30 days; Hermes keeps a running total per session in its state.db and
prunes sessions after 90. `locket usage record` copies both into one ledger,
`~/.locket/usage.csv`, one row per day, source, project and model, so a day survives its
source. It is safe to run any number of times: it rewrites only the days its sources still
hold, and never replaces a day with a smaller one, which is what a day half-deleted by its
source looks like. `locket install --hooks` runs it when each Claude Code session ends, and
the report runs it first, so the ledger stays current without a scheduler.

`locket usage` reports the last 7 days (`today`, `month` for 1 or 30) per source, against
the median day of the 28 before, and the projects that spent most. Token kinds are never
added together: a cache read costs a fraction of a fresh input token and an output token
several times one, so a single total would rank a cheap day above a dear one. The
cache-read share is the part of all input served from the cache, the one number a user
moves by how they work: a long session re-reads its context cheaply, a fresh one pays for
it again.

It reports tokens only. A price table would go stale and a subscription is not billed per
token; the host's plan limits have no source Locket can verify, and other hosts are not
read. `--claude DIR` and `--hermes DB` (or LOCKET_USAGE_CLAUDE and LOCKET_USAGE_HERMES)
point at a source somewhere else; a missing source is skipped with a line saying so.
"""
import csv
import datetime as dt
import glob
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import memscan

LEDGER = memscan.REGISTRY.parent / "usage.csv"
FIELDS = ("date", "source", "project", "model", "sessions", "responses",
          "input", "cache_write", "cache_read", "output", "reasoning")
KINDS = ("input", "cache_write", "cache_read", "output")
NUMS = FIELDS[4:]
PERIODS = {"today": 1, "week": 7, "month": 30}
BASELINE = 28       # days before the period whose median day is "typical"
MIN_BASELINE = 7    # fewer ledger days than this and there is no typical day yet


def _claude_dir(flag=None):
    if flag or os.environ.get("LOCKET_USAGE_CLAUDE"):
        return Path(flag or os.environ["LOCKET_USAGE_CLAUDE"]).expanduser()
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or memscan.CLAUDE).expanduser() / "projects"


def _hermes_db(flag=None):
    if flag or os.environ.get("LOCKET_USAGE_HERMES"):
        return Path(flag or os.environ["LOCKET_USAGE_HERMES"]).expanduser()
    return Path(os.environ.get("HERMES_HOME") or memscan.HERMES).expanduser() / "state.db"


def _project(cwd):
    if not cwd:
        return ""
    home = str(memscan.HOME)
    return "~" + cwd[len(home):] if cwd == home or cwd.startswith(home + "/") else cwd


def _day(ts):
    """The local calendar day of an ISO timestamp or epoch seconds, as YYYY-MM-DD."""
    if isinstance(ts, (int, float)):
        return dt.datetime.fromtimestamp(ts).date().isoformat()
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return t.astimezone().date().isoformat()


def _blank():
    return {k: 0 for k in NUMS} | {"_sessions": set()}


def read_claude(root, since=None, notes=None):
    """{(date, project, model): counts} from the transcripts under `root`, each response
    once. Claude Code writes a response's usage on every line of its content blocks, and
    a resumed session copies the lines it resumes, so a line counts only on the first
    sight of its message id and request id: without that, a day reads about 2.5 times
    its size. Only files touched on or after `since` (epoch seconds) are opened."""
    rows = defaultdict(_blank)
    if not root.is_dir():
        if notes is not None:
            notes.append(f"claude: no transcripts at {root}; skipped")
        return rows
    seen = set()
    for f in sorted(glob.glob(str(root / "**" / "*.jsonl"), recursive=True)):
        try:
            if since and os.path.getmtime(f) < since:
                continue
            fh = open(f, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"usage"' not in line or '"assistant"' not in line:
                    continue            # cheap: most lines are tool output
                try:
                    d = json.loads(line)
                    msg = d["message"]
                    u = msg["usage"]
                except (ValueError, KeyError, TypeError):
                    continue
                model = msg.get("model") or ""
                if d.get("type") != "assistant" or not d.get("timestamp") or model == "<synthetic>":
                    continue
                key = (msg.get("id"), d.get("requestId"))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    r = rows[(_day(d["timestamp"]), _project(d.get("cwd")), model)]
                except ValueError:
                    continue
                r["_sessions"].add(d.get("sessionId"))
                r["responses"] += 1
                r["input"] += u.get("input_tokens") or 0
                r["cache_write"] += u.get("cache_creation_input_tokens") or 0
                r["cache_read"] += u.get("cache_read_input_tokens") or 0
                r["output"] += u.get("output_tokens") or 0
                r["reasoning"] += (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
    return rows


def read_hermes(db, notes=None):
    """{(date, project, model): counts} from Hermes's state.db, a session counted on the day
    it started. `session_model_usage` holds every call per session and model, auxiliary
    ones (titles, compression, review) included, which the `sessions` row leaves out;
    a Hermes too old to have it is read from `sessions`. A subagent's calls sit on its
    own session, never its parent's. Read-only, and tolerant of a missing column."""
    rows = defaultdict(_blank)
    if not db.is_file():
        if notes is not None:
            notes.append(f"hermes: no state.db at {db}; skipped")
        return rows
    cols = {"input_tokens": "input", "cache_write_tokens": "cache_write", "cache_read_tokens": "cache_read",
            "output_tokens": "output", "reasoning_tokens": "reasoning", "api_call_count": "responses"}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            have = {r[1] for r in con.execute("PRAGMA table_info(sessions)")}
            per_model = {r[1] for r in con.execute("PRAGMA table_info(session_model_usage)")}
            if "started_at" not in have:
                raise sqlite3.OperationalError("no sessions.started_at")
            if {"session_id", "model"} <= per_model:
                use = [c for c in cols if c in per_model]
                q = (f"SELECT s.started_at, s.{'cwd' if 'cwd' in have else 'NULL'}, u.model, u.session_id, "
                     f"{', '.join('u.' + c for c in use)} FROM session_model_usage u JOIN sessions s ON s.id = u.session_id")
            else:
                use = [c for c in cols if c in have]
                q = (f"SELECT started_at, {'cwd' if 'cwd' in have else 'NULL'}, "
                     f"{'model' if 'model' in have else 'NULL'}, id, {', '.join(use)} FROM sessions")
            for rec in con.execute(q):
                v = dict(zip(["started_at", "cwd", "model", "id", *use], rec))
                r = rows[(_day(v["started_at"]), _project(v.get("cwd")), v.get("model") or "")]
                r["_sessions"].add(v.get("id"))
                for c in use:
                    r[cols[c]] += v[c] or 0
        finally:
            con.close()
    except sqlite3.Error as e:
        if notes is not None:
            notes.append(f"hermes: {db} unreadable ({e}); skipped")
    return rows


def load(path=None):
    path = path or LEDGER
    try:
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            if tuple(rd.fieldnames or ()) != FIELDS:
                raise SystemExit(f"{path}: columns are not {', '.join(FIELDS)}; move it aside to start a new ledger")
            return [{**r, **{k: int(r[k] or 0) for k in NUMS}} for r in rd]
    except FileNotFoundError:
        return []


def _fresh(source, found):
    return [{"date": d, "source": source, "project": p, "model": m, **{k: c[k] for k in NUMS},
             "sessions": len(c["_sessions"])} for (d, p, m), c in found.items()]


def merge(old, fresh, since_day=None):
    """The ledger after a record: each (source, day) the sources returned replaces the
    recorded one, unless it has fewer responses, which is a day its source has begun to
    delete. A day before `since_day` was not read this time and stays as recorded."""
    def by_day(rows):
        out = defaultdict(list)
        for r in rows:
            out[(r["source"], r["date"])].append(r)
        return out
    kept, new = by_day(old), by_day(fresh)
    for key, rows in new.items():
        if since_day and key[1] < since_day:
            continue
        if sum(r["responses"] for r in rows) >= sum(r["responses"] for r in kept.get(key, [])):
            kept[key] = rows
    return sorted((r for rows in kept.values() for r in rows),
                  key=lambda r: (r["date"], r["source"], r["project"], r["model"]))


def record(claude=None, hermes=None, full=False, ledger=None, quiet=False):
    """Read both sources into the ledger. Claude transcripts are opened only when touched
    since the day before the last record, unless `full` or there is no ledger yet: a line
    dated on or after that day lives in a file modified on or after it."""
    ledger = ledger or LEDGER
    notes = []
    since = since_day = None
    if not full and ledger.is_file():
        start = dt.date.fromtimestamp(ledger.stat().st_mtime) - dt.timedelta(days=1)
        since_day = start.isoformat()
        since = time.mktime(start.timetuple())
    claude_rows = _fresh("claude", read_claude(_claude_dir(claude), since, notes))
    hermes_rows = _fresh("hermes", read_hermes(_hermes_db(hermes), notes))
    old = load(ledger)
    rows = merge(old, [r for r in claude_rows if not since_day or r["date"] >= since_day], since_day)
    rows = merge(rows, hermes_rows)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, ledger)
    if not quiet:
        for n in notes:
            print(n, file=sys.stderr)
        print(f"recorded {len(rows)} rows ({len(rows) - len(old):+d}) in {ledger}")
    return rows, notes


# --- report ---------------------------------------------------------------------

def _h(n):
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def _share(t):
    fed = t["input"] + t["cache_write"] + t["cache_read"]
    return t["cache_read"] / fed if fed else None


def _sum(rows):
    t = {k: 0 for k in NUMS}
    for r in rows:
        for k in NUMS:
            t[k] += r[k]
    return t


def summarize(rows, days, today=None):
    """The period's totals by source and by project and model, and the typical day: the
    median over the BASELINE days before the period, counted from the ledger's first day,
    so a machine with a week of history is not compared against three empty ones."""
    today = today or dt.date.today()
    start = today - dt.timedelta(days=days - 1)
    base_start = max(start - dt.timedelta(days=BASELINE),
                     min((dt.date.fromisoformat(r["date"]) for r in rows), default=start))
    in_period = [r for r in rows if start.isoformat() <= r["date"] <= today.isoformat()]
    per_day = defaultdict(lambda: {k: 0 for k in NUMS})
    for r in rows:
        if base_start.isoformat() <= r["date"] < start.isoformat():
            for k in NUMS:
                per_day[r["date"]][k] += r[k]
    n_base = (start - base_start).days
    base_days = [per_day.get((base_start + dt.timedelta(days=i)).isoformat(), {k: 0 for k in NUMS})
                 for i in range(n_base)]
    typical = ({k: statistics.median(d[k] for d in base_days) for k in NUMS}
               if n_base >= MIN_BASELINE else None)
    group = lambda key: {v: _sum([r for r in in_period if key(r) == v])
                         for v in sorted({key(r) for r in in_period})}
    total = _sum(in_period)
    return {
        "from": start.isoformat(), "to": today.isoformat(), "days": days,
        "total": total, "per_day": {k: total[k] / days for k in NUMS},
        "cache_read_share": _share(total),
        "typical_day": typical, "typical_cache_read_share": _share(typical) if typical else None,
        "baseline_days": n_base,
        "by_source": group(lambda r: r["source"]),
        "by_project": group(lambda r: f'{r["source"]}  {r["project"] or "(no directory)"}'),
        "by_model": group(lambda r: f'{r["source"]}  {r["model"] or "(unknown)"}'),
        "by_day": group(lambda r: r["date"]),
    }


def _fit(name, width=33):
    """A project path cut from the left, whole components kept, so the part that tells
    two projects apart survives."""
    if len(name) <= width:
        return name
    src, _, path = name.partition("  ")
    keep = path.rsplit("/", 1)[-1]
    for part in reversed(path.split("/")[:-1]):
        if len(src) + len(part) + len(keep) + 5 > width:
            break
        keep = f"{part}/{keep}"
    return f"{src}  …/{keep}"[:width]


def render(s, by="project", top=8):
    head = f"{'':<34}{'responses':>10}{'input':>9}{'cache write':>13}{'cache read':>12}{'output':>9}"
    line = lambda name, t: (f"{_fit(name):<34}{t['responses']:>10,}{_h(t['input']):>9}"
                            f"{_h(t['cache_write']):>13}{_h(t['cache_read']):>12}{_h(t['output']):>9}")
    span = s["to"] if s["days"] == 1 else f'{s["from"]} to {s["to"]}'
    out = [f"locket usage: {span} ({s['days']} day{'s' * (s['days'] > 1)})", "", head]
    out += [line(src, t) for src, t in s["by_source"].items()] or ["  nothing recorded in this period"]
    if len(s["by_source"]) > 1:
        out.append(line("all", s["total"]))
    out.append("")
    typ, pd = s["typical_day"], s["per_day"]
    if typ is None:
        out.append(f"typical day: not enough history yet ({s['baseline_days']} of {MIN_BASELINE} days before this period)")
    else:
        ratio = lambda k: (f"{pd[k] / typ[k]:.1f}x" if typ[k] else "new")
        out.append(f"per day vs typical (median of the {s['baseline_days']} days before): "
                   + ", ".join(f"{k.replace('_', ' ')} {ratio(k)}" for k in KINDS))
    share, tshare = s["cache_read_share"], s["typical_cache_read_share"]
    if share is not None:
        out.append(f"cache-read share: {share:.0%} of input tokens came from the cache"
                   + (f" (typical {tshare:.0%})" if tshare is not None else ""))
    groups = s[f"by_{by}"]
    if groups:
        ranked = sorted(groups.items(), key=lambda kv: (kv[0] if by == "day" else -kv[1]["responses"]))
        out += ["", f"by {by}" + (f", top {top} by responses" if by != "day" and len(ranked) > top else "") + ":", head]
        out += [line(k, t) for k, t in (ranked if by == "day" else ranked[:top])]
    return "\n".join(out)


def main(argv):
    import argparse
    p = argparse.ArgumentParser(prog="locket usage", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("period", nargs="?", default="week", choices=[*PERIODS, "record"])
    p.add_argument("--by", choices=("project", "model", "day"), default="project")
    p.add_argument("--json", action="store_true", help="the summary as JSON")
    p.add_argument("--claude", metavar="DIR", help="Claude Code's projects folder")
    p.add_argument("--hermes", metavar="DB", help="Hermes's state.db")
    p.add_argument("--full", action="store_true", help="record: reread every transcript")
    p.add_argument("--quiet", action="store_true", help="record: print nothing")
    a = p.parse_args(argv)
    if a.period == "record":
        record(a.claude, a.hermes, a.full, quiet=a.quiet)
        return 0
    rows, notes = record(a.claude, a.hermes, quiet=True)
    s = summarize(rows, PERIODS[a.period])
    if a.json:
        print(json.dumps({**s, "notes": notes}, indent=2))
    else:
        print(render(s, a.by))
        for n in notes:
            print(n)
    return 0


def hook():
    """SessionEnd: record, silently, and never fail the host's exit."""
    try:
        sys.stdin.read()
        record(quiet=True)
    except (Exception, SystemExit):   # load() exits on a foreign ledger; neither may reach the host's shutdown
        pass
    return 0


def selftest():
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
        old = read_hermes(old_db)
        assert [(k[2], v["input"], v["output"]) for k, v in old.items()] == [("glm", 7, 3)], dict(old)

        ledger = t / "usage.csv"
        rows, notes = record(proj.parent, db, ledger=ledger, quiet=True)
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
        record(proj.parent, db, ledger=ledger, quiet=True, full=True)
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
        rows, _ = record(proj.parent, db, ledger=ledger, quiet=True)
        recorded = days(rows)
        assert recorded[("claude", today.isoformat())] == 3, "the incremental record missed a new response"
        assert ("claude", (today - dt.timedelta(days=3)).isoformat()) not in recorded, \
            "the incremental record reread an old file"

        # the source deletes one of today's files and all of yesterday's: neither day shrinks
        (proj / "s1/subagents/a.jsonl").unlink()
        rows, _ = record(proj.parent, db, ledger=ledger, quiet=True)
        (proj / "s1.jsonl").unlink()
        rows, _ = record(proj.parent, db, ledger=ledger, quiet=True)
        assert days(rows) == recorded, "a day shrank when its source deleted part of it"
        # a missing source is a note, never an error, and keeps what was recorded
        rows, notes = record(t / "none", t / "none.db", ledger=ledger, quiet=True, full=True)
        assert len(notes) == 2 and days(rows) == recorded, notes

        s = summarize(rows, 7)
        assert s["typical_day"] is None and "not enough history" in render(s)
        assert s["by_source"]["claude"]["responses"] == 4 and abs(s["cache_read_share"] - 4000 / 4490) < 1e-9, s
        text = render(s, "day")
        assert "claude" in text and "hermes" in text and "by day:" in text, text
        # a long history: the typical day is the median, counting days with nothing as zero
        hist = [{"date": (today - dt.timedelta(days=d)).isoformat(), "source": "claude", "project": "", "model": "m",
                 **{k: 0 for k in NUMS}, "responses": 10, "output": 100} for d in range(7, 36) if d % 2]
        s = summarize(hist + rows, 7)
        assert s["baseline_days"] == BASELINE and s["typical_day"]["output"] == 50, s["typical_day"]
        json.dumps(s)
        (t / "bad.csv").write_text("date,tokens\n")
        try:
            load(t / "bad.csv")
            raise AssertionError("a ledger with other columns was read")
        except SystemExit:
            pass
        # the SessionEnd hook stays silent and exits 0 even on that ledger
        import contextlib
        import io
        saved, env = globals()["LEDGER"], dict(os.environ)
        globals()["LEDGER"] = t / "bad.csv"
        os.environ.update(LOCKET_USAGE_CLAUDE=str(t / "none"), LOCKET_USAGE_HERMES=str(t / "none.db"))
        stdin, sys.stdin = sys.stdin, io.StringIO("{}")
        err = io.StringIO()
        try:
            with contextlib.redirect_stdout(err), contextlib.redirect_stderr(err):
                assert hook() == 0 and not err.getvalue(), f"the hook spoke or failed: {err.getvalue()}"
        finally:
            globals()["LEDGER"], sys.stdin = saved, stdin
            os.environ.clear(); os.environ.update(env)
    print("usage selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["--selftest"] else main(sys.argv[1:]))
