"""fsio: the read-only SQLite helpers."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import tests.harness  # noqa: F401  (puts the source under test on sys.path)

import fsio  # noqa: E402


class SqliteRows(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def db(self, name):
        p = Path(self.dir.name) / name
        con = sqlite3.connect(p)
        con.execute("create table t(a)")
        con.execute("insert into t values (1)")
        con.commit()
        con.close()
        return p

    def test_uri_characters_in_a_path_name_the_file(self):
        # Unquoted, `?` starts the URI's parameters and `#` its fragment, so the
        # connection opened some other path and saw no table.
        for name in ("odd?x=1#frag.db", "sp ace.db", "pct%41.db"):
            self.assertEqual(fsio.sqlite_rows(self.db(name), "select a from t"), [(1,)], name)

    def test_failure_is_none_never_a_raise(self):
        p = self.db("ok.db")
        self.assertIsNone(fsio.sqlite_rows(Path(self.dir.name) / "missing.db", "select 1"))
        self.assertIsNone(fsio.sqlite_rows(p, "select * from no_such_table"))

    def test_read_only(self):
        p = self.db("ro.db")
        with self.assertRaises(sqlite3.OperationalError), fsio.sqlite_ro(p) as con:
            con.execute("insert into t values (2)")


if __name__ == "__main__":
    unittest.main()
