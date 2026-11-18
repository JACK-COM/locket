"""Ledger schemas: the one decidable refusal, at the write tools, in the shell and in the sweep."""
import contextlib
import io
import json
import unittest

from tests.harness import StoreCase, shown as _shown

import bashguard  # noqa: E402
import corpora  # noqa: E402
import corpus  # noqa: E402
import help_text  # noqa: E402
import ledger  # noqa: E402
import manifest  # noqa: E402
from commands import hygiene  # noqa: E402


class Ledgers(StoreCase):
    def setUp(self):
        super().setUp()
        d = self.tmp
        led = (d / "proj").resolve(); led.mkdir()
        hist = led / "HISTORY-Proj-Sessions.csv"
        hdr = "date,title,summary,area,repo,commit_hash,issue,pr\n"
        good = '2026-09-20,Composite corpus,"Two stores joined into one, links resolved.",api,grp/sub/proj,5ab427a,ENG-42,#155\n'
        hist.write_text(hdr + good)
        long_row = f'2026-09-20,{"t" * 91},{"s" * 241},ios,a b,zz,two words,155\n'
        self.led, self.hist, self.hdr, self.good, self.long_row = led, hist, hdr, good, long_row

    def unpack(self, manifest_text=None):
        if manifest_text is not None:
            (self.led / manifest.MANIFEST).write_text(manifest_text)
            corpora.clear_settings()
        return self.led, self.hist, self.hdr, self.good, self.long_row

    def test_the_built_in_schema(self):
        led, hist, hdr, good, long_row = self.unpack()
        assert ledger.ledger_schema_for(hist) == (None, None), "no schema governs a CSV outside every store"
        (led / manifest.MANIFEST).write_text("{}\n")
        sch, glob = ledger.ledger_schema_for(hist)
        assert sch is not None and glob == "HISTORY-*-Sessions.csv", "the built-in governs the file inside a store"
        assert ledger.ledger_schema_for(led / "notes.csv") == (None, None)
        assert ledger.ledger_decision(str(hist), {"content": hdr + good}) == (None, False), "a clean whole-file write passes"
        assert ledger.ledger_decision(str(hist), {"content": hdr}) == (None, False), "a header alone passes"
        msg, blocking = ledger.ledger_decision(str(hist), {"content": hdr + good + long_row})
        assert blocking and "row 3: title is 91 characters, cap 90" in msg \
            and "summary is 241 characters, cap 240" in msg and "commit_hash is 'zz'" in msg and "issue is 'two words'" in msg and "pr is '155'" in msg \
            and "repo is 'a b'" in msg, msg
        mr = good.replace("#155", "!155")
        assert ledger.ledger_decision(str(hist), {"content": hdr + mr}) == (None, False), "a GitLab MR is a pr"
        msg, blocking = ledger.ledger_decision(str(hist), {"content": "date,title\n2026-01-01,x\n"})
        assert blocking and "header is date,title" in msg, "a header off the schema is one finding"
        msg, blocking = ledger.ledger_decision(str(hist), {"old_string": "x", "new_string": long_row})
        assert blocking and "row 1: title is 91" in msg, "an Edit fragment is checked against the header on disk"
        assert ledger.ledger_decision(str(hist), {"old_string": "x", "new_string": "Composite"}) == (None, False), \
            "a fragment that is not a row is not judged"
        assert ledger.ledger_decision(str(hist), {"old_string": "x", "new_string": ",,,,,,,\n"}) == (None, False), \
            "an empty row is a blank line"
        msg, blocking = ledger.ledger_decision(str(hist), {"old_string": "x", "new_string": "2026-09-20,,ok,,,,,\n"})
        assert blocking and "title is required" in msg, msg
        for dt in ("2026-09-23T03:12Z", "2026-09-23T03:12:45Z"):
            assert ledger.ledger_decision(str(hist), {"old_string": "x", "new_string": f"{dt},t,s,,,,,\n"}) == (None, False), \
                f"a UTC instant is a date: {dt}"
        for dt in ("2026-09-23T03:12", "2026-09-23T03:12:45-07:00", "2026-09-23 03:12Z"):
            msg, blocking = ledger.ledger_decision(str(hist), {"old_string": "x", "new_string": f"{dt},t,s,,,,,\n"})
            assert blocking and "date" in msg, f"only a Z-suffixed instant passes: {dt}"

    def test_bashguard_reads_the_rows(self):
        led, hist, hdr, good, long_row = self.unpack("{}\n")
        # bashguard: heredoc rows and echo rows are read; an opaque append warns
        msg, blocking = bashguard.ledger_bash_decision(f"cat >> {hist} <<'EOF'\n{long_row}EOF\n", None)
        assert blocking and "bashguard" in msg and "cap 90" in msg, msg
        msg, blocking = bashguard.ledger_bash_decision(f'echo "{good.strip()}" >> {hist}', None)
        assert (msg, blocking) == (None, False), msg
        msg, blocking = bashguard.ledger_bash_decision(f"python3 gen.py >> {hist}", None)
        assert msg and not blocking and "cannot read" in msg, msg
        assert bashguard.ledger_bash_decision(f"cat {hist}", None) == (None, False), "a read is nothing"
        assert bashguard.ledger_bash_decision("echo hi >> other.csv", str(led)) == (None, False)
        # a body counts only against the ledger its own pipeline writes
        msg, blocking = bashguard.ledger_bash_decision(f"printf '{good.strip()}\\n' >> {hist} && echo '# t' > a.md", None)
        assert (msg, blocking) == (None, False), f"a second redirect is not the ledger's content: {msg}"
        msg, blocking = bashguard.ledger_bash_decision(f"printf 'not-a-date,t,s,,,,,\\n' | tee -a {hist}; echo '# t' > a.md", None)
        assert blocking and "date" in msg and "header" not in msg, msg
        msg, blocking = bashguard.ledger_bash_decision(f"echo '# t' > a.md; python3 gen.py >> {hist}", None)
        assert msg and not blocking and "cannot read" in msg, "an unrelated echo does not vouch for an opaque append"
        msg, blocking = bashguard.ledger_bash_decision(f"cat <<'EOF' > {hist}\n{long_row}EOF\n", None)
        assert blocking and "cap 90" in msg, "a redirect after the heredoc opener is the same pipeline"
        hist2 = led / "HISTORY-Other-Sessions.csv"; hist2.write_text(hdr)
        msg, blocking = bashguard.ledger_bash_decision(
            f"python3 gen.py >> {hist} && printf 'not-a-date,t,s,,,,,\\n' >> {hist2}", None)
        assert blocking and "date" in msg, "an opaque write first never excuses a readable bad row after it"
        msg, blocking = bashguard.ledger_bash_decision(f"cat <<'EOF' \\\n  >> {hist}\n{long_row}EOF\n", None)
        assert blocking and "cap 90" in msg, "a redirect on a continued opener line is the same pipeline"
        for before in ("cat <<<x\n", "echo 'a <<b'\n", "cat <<'E-F'\nx\nE-F\n"):
            msg, blocking = bashguard.ledger_bash_decision(f"{before}cat >> {hist} <<'EOF'\n{long_row}EOF\n", None)
            assert blocking and "cap 90" in msg, f"the ledger heredoc after {before!r} is read: {msg}"
        hist2.unlink()

    def test_a_manifest_overrides_and_extends(self):
        led, hist, hdr, good, long_row = self.unpack("{}\n")
        # a manifest overrides the built-in for its store, and governs another CSV
        (led / manifest.MANIFEST).write_text(json.dumps({"ledgers": [
            {"path": "HISTORY-*-Sessions.csv", "caps": {"summary": 20}},
            {"path": "LEDGER-Closures.csv", "required": ["slug"], "caps": {"note": 10}}]}))
        corpora.clear_settings()
        msg, blocking = ledger.ledger_decision(str(hist), {"content": hdr + good})
        assert blocking and "characters, cap 20" in msg and manifest.MANIFEST in msg, msg
        clos = led / "LEDGER-Closures.csv"; clos.write_text("slug,note\n")
        msg, blocking = ledger.ledger_decision(str(clos), {"old_string": "x", "new_string": ",a very long note here\n"})
        assert blocking and "slug is required" in msg and "note is 21 characters" in msg, msg
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = _shown(hygiene.cmd_ledgers(led))
        assert n == 1 and "2 ledger(s) checked, 1 finding(s)" in buf.getvalue(), buf.getvalue()
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()
        assert "@@" not in help_text.usage() and "caps title ≤90" in help_text.usage(), "the help renders the built-ins from the constant"
        # an entry naming a built-in's glob MERGES onto it: only the keys it names change
        (led / manifest.MANIFEST).write_text(json.dumps({"ledgers": [
            {"path": "HISTORY-*-Sessions.csv", "enum": {"area": ["docs", ""]}}]}))
        corpora.clear_settings()
        ok = '2026-09-20,Docs pass,"A row under the store\'s own areas.",docs,,,,\n'
        assert ledger.ledger_decision(str(hist), {"content": hdr + ok}) == (None, False), "a store's own area passes"
        msg, blocking = ledger.ledger_decision(str(hist), {"content": hdr + ok.replace(",docs,", ",ui,")})
        assert blocking and "area is 'ui'" in msg, "the store's enum replaces the built-in's for that column"
        msg, blocking = ledger.ledger_decision(str(hist), {"content": hdr + ok.replace("2026-09-20", "20 Sep")})
        assert blocking and "date" in msg, "the built-in date pattern survives an enum override"
        msg, blocking = ledger.ledger_decision(str(hist), {"content": hdr + f'2026-09-20,{"t" * 91},s,docs,,,,\n'})
        assert blocking and "cap 90" in msg, "the built-in caps survive an enum override"
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()
        # `extends`: a file named outside the glob inherits a built-in by declaration
        odd = led / "HISTORY-session-summary.csv"
        assert ledger.ledger_schema_for(odd) == (None, None), "outside the glob, nothing governs it"
        (led / manifest.MANIFEST).write_text(json.dumps({"ledgers": [
            {"path": odd.name, "extends": "HISTORY-*-Sessions.csv", "enum": {"area": ["docs", ""]}}]}))
        corpora.clear_settings()
        assert manifest.validate_manifest(led) == [], manifest.validate_manifest(led)
        assert ledger.ledger_decision(str(odd), {"content": hdr + ok}) == (None, False), "extends inherits the built-in"
        msg, blocking = ledger.ledger_decision(str(odd), {"content": hdr + ok.replace("2026-09-20", "20 Sep")})
        assert blocking and "date" in msg, "the inherited date pattern holds"
        (led / manifest.MANIFEST).write_text(json.dumps({"ledgers": [{"path": odd.name, "extends": "nope"}]}))
        corpora.clear_settings()
        assert any("is no built-in" in f for f in manifest.validate_manifest(led)), manifest.validate_manifest(led)
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()
        # an entry naming one file inside the built-in's glob governs the sweep alone
        (led / manifest.MANIFEST).write_text(json.dumps({"ledgers": [
            {"path": hist.name, "enum": {"area": ["docs", ""]}}]}))
        corpora.clear_settings(); hist.write_text(hdr + ok)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = _shown(hygiene.cmd_ledgers(led))
        assert n == 0 and "1 ledger(s) checked, 0 finding(s)" in buf.getvalue(), buf.getvalue()
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings(); hist.write_text(hdr + good)

    def test_rulings_and_claims(self):
        led, hist, hdr, good, long_row = self.unpack("{}\n")
        # rulings and claims: one glob covers the single-store and the project name
        rh, ch = "Ruling,Category,Area,Date\n", "Claim,Category,Area,Source,Date\n"
        rul = '"Pro unlocks by a one-time purchase, never a subscription.",product,pricing,2026-09-26\n'
        for name in ("RULINGS.csv", "RULINGS-Proj.csv"):
            assert ledger.ledger_schema_for(led / name)[1] == "RULINGS*.csv", name
            assert ledger.ledger_decision(str(led / name), {"content": rh + rul}) == (None, False), name
        assert ledger.ledger_schema_for(led / "CLAIMS-Proj.csv")[1] == "CLAIMS*.csv"
        assert ledger.ledger_schema_for(led / "REFERENCE-Proj-Rulings.csv") == (None, None), \
            "a name outside the glob is governed only through `extends`"
        rp = led / "RULINGS-Proj.csv"; rp.write_text(rh)
        assert ledger.ledger_decision(str(rp), {"old_string": "x", "new_string": rul.replace(",pricing,", ",pricing|release,")}) \
            == (None, False), "a pipe-joined Area is several keys"
        msg, blocking = ledger.ledger_decision(str(rp), {"old_string": "x", "new_string": rul.replace(",pricing,", ",Pricing,")})
        assert blocking and "Area is 'Pricing'" in msg, "an Area is a lowercase key, because pointers query it"
        msg, blocking = ledger.ledger_decision(str(rp), {"old_string": "x", "new_string": rul.replace(",2026-09-26", ",")})
        assert blocking and "Date is required" in msg, msg
        cp = led / "CLAIMS-Proj.csv"; cp.write_text(ch)
        msg, blocking = ledger.ledger_decision(str(cp), {"old_string": "x", "new_string": '"An empty product list means the agreement is unsigned.",code,iap,,2026-09-26\n'})
        assert blocking and "Source is required" in msg, "a claim without a source is a finding"
        rp.unlink(); cp.unlink()

    def test_size_and_the_sweep(self):
        led, hist, hdr, good, long_row = self.unpack("{}\n")
        rh = "Ruling,Category,Area,Date\n"
        rul = '"Pro unlocks by a one-time purchase, never a subscription.",product,pricing,2026-09-26\n'
        # the size reading: a declared CSV past the limit is a finding, never split or capped
        assert ledger.large_csvs(led) == [], "a small ledger reported past 5 MB"
        assert [p.name for p, _n in ledger.large_csvs(led, limit=1)] == [hist.name], ledger.large_csvs(led, limit=1)
        saved, ledger.CSV_REPORT_BYTES = ledger.CSV_REPORT_BYTES, 1
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                n = _shown(hygiene.cmd_ledgers(led))
            assert "declared CSV(s) past the" in buf.getvalue() and "never split it" in buf.getvalue(), buf.getvalue()
            assert n >= 1, "a ledger past the size reading counted as clean"
        finally:
            ledger.CSV_REPORT_BYTES = saved
        # the sweep walks the store: a ledger under a subdirectory is checked, one
        # under an excluded directory is not
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()
        (led / "memory").mkdir(exist_ok=True)
        (led / "memory" / "RULINGS.csv").write_text(rh + rul.replace(",pricing,", ",Pricing,"))
        (led / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
        (led / "node_modules" / "pkg" / "README.md").write_text("A dependency's readme is never prose of this store.\n")
        assert not any("node_modules" in str(p) for p in corpus.md_files(led)), "a dependency tree is not walked"
        (led / manifest.MANIFEST).write_text(json.dumps({"excluded_dirs": ["drafts"]})); corpora.clear_settings()
        assert not any("node_modules" in str(p) for p in corpus.md_files(led)), "a store's own exclusions do not re-admit it"
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()
        (led / "archive").mkdir(exist_ok=True)
        (led / "archive" / "RULINGS.csv").write_text(rh + rul.replace(",pricing,", ",Pricing,"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = _shown(hygiene.cmd_ledgers(led))
        assert n == 1 and "memory/RULINGS.csv" in buf.getvalue() and "archive/" not in buf.getvalue(), buf.getvalue()
        (led / "memory" / "RULINGS.csv").unlink(); (led / "archive" / "RULINGS.csv").unlink()
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()


class OversizedField(StoreCase):
    def test_a_field_past_the_csv_limit_blocks(self):
        """The csv reader raises on a field past its limit; read as an error, the write
        landed with no cap checked, so it is a finding instead."""
        led = (self.tmp / "proj").resolve(); led.mkdir()
        (led / manifest.MANIFEST).write_text("{}\n"); corpora.clear_settings()
        hist = led / "HISTORY-Proj-Sessions.csv"
        hdr = "date,title,summary,area,repo,commit_hash,issue,pr\n"
        hist.write_text(hdr)
        row = f'2026-09-20,t,"{"s" * 140000}",,,,,\n'
        msg, blocking = ledger.ledger_decision(str(hist), {"content": hdr + row})
        self.assertTrue(blocking and "does not parse as CSV" in msg, msg)
        msg, blocking = ledger.ledger_decision(str(hist), {"old_string": "x", "new_string": row})
        self.assertTrue(blocking, msg)


if __name__ == "__main__":
    unittest.main()
