"""configure's suite: Locket's embed settings written, read back and layered, offline."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from tests.harness import fixture_env

import _embed  # noqa: E402
import _settings  # noqa: E402
import configure as cf  # noqa: E402
from commands import settings  # noqa: E402

def _suite():
    """Offline, against a throwaway settings file: write, read back, precedence across the
    piece, global and legacy layers, clearing, a refused bad file, and the migration."""
    names = [v for _, env in _embed.CONFIG_KEYS.values() for v in env] + ["PANOPLY_CONFIG"]
    saved_legacy, saved_env = cf.LEGACY, {v: os.environ.pop(v, None) for v in names}
    try:
        with tempfile.TemporaryDirectory() as t, _embed.settings():
            os.environ["PANOPLY_CONFIG"] = str(Path(t) / "panoply" / "config.json")
            cf.LEGACY = Path(t) / "locket" / "config.json"
            cfg, load = _settings.path(), lambda: _settings.load()
            assert load() == ({}, []) and cf.apply()["model"] == (_embed.SHIPPED["model"], "default")
            run = lambda *a: settings._set_embed(settings.parser().parse_args(["embedder", *a]))
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                assert run("--model", "m1", "--no-check", "--no-autostart") == 0
                assert load()[0] == {"locket": {"embed": {"model": "m1", "autostart": False}}}, load()
                assert cf.apply()["model"] == ("m1", "locket") and _embed.MODEL == "m1"
                assert run("--model", "m-all", "--ollama-host", "http://g:1", "--no-check", "--global") == 0
                got = cf.apply()
                assert got["model"] == ("m1", "locket") and got["ollama_host"] == ("http://g:1", "global"), got
                os.environ["MEMFIND_MODEL"] = "m-env"
                assert cf.apply()["model"] == ("m-env", "env MEMFIND_MODEL"), "the environment must beat the file"
                del os.environ["MEMFIND_MODEL"]
                assert run("--model", "default") == 0 and cf.apply()["model"] == ("m-all", "global"), load()
                assert run("--reset") == 0 and load()[0] == {"embed": {"model": "m-all", "ollama_host": "http://g:1"}}
                assert run("--reset", "--global") == 0 and load()[0] == {}, load()
                assert run("--ollama-host", "studio:11434") == 2, "a host with no scheme was written"
                assert run("--model", "") == 2 and run("--venv", "~no-such-user-x/v") == 2, "a value that cannot take effect was written"
                assert run() == 2, "a run naming nothing to change wrote"
                before = cfg.read_text()
                assert run("--model", "m2", "--no-check", "--dry-run") == 0 and cfg.read_text() == before
                cfg.write_text('{"embed": {"model": 3, "colour": "red"}, "grille": {"embd": {}, "embed": {"model": "g"}}}')
                assert len(load()[1]) == 3 and cf.apply()["model"][1] == "default", f"a bad file took effect: {load()}"
                assert run("--model", "m3", "--no-check") == 0, "a structural finding blocked a write"
                assert load()[0]["locket"] == {"embed": {"model": "m3"}} and load()[0]["grille"]["embed"] == {"model": "g"}
                assert run("--no-autostart") == 0 and run("--autostart", "default") == 0
                assert load()[0]["locket"] == {"embed": {"model": "m3"}}, "--autostart default did not clear it"
                assert run("--autostart") == 0 and load()[0]["locket"]["embed"]["autostart"] is True
                cfg.write_text('{"locket": {"embed": {"model": "m4"}}}')
                assert run("--venv", "rel/v") == 0
                assert load()[0]["locket"]["embed"]["venv"] == os.path.abspath("rel/v"), "a relative --venv was stored as given"
                cfg.write_text("{not json")
                assert load()[0] == {} and _settings.unreadable(load()[1])
                assert run("--model", "m5", "--no-check") == 1, "a write over an unreadable file went ahead"
                assert run("--reset", "--model", "m5", "--no-check") == 0
                assert cfg.with_name("config.json.bak").read_text() == "{not json", "--reset kept no copy"
                # Locket 0.10's file: read below the top level, then moved into it
                cf.LEGACY.parent.mkdir()
                cf.LEGACY.write_text('{"embed": {"model": "m-old", "ollama_host": "http://old:1", "venv": "", "colour": "x"}}')
                cfg.write_text("[]")
                assert "left in place" in cf.migrate() and cf.LEGACY.exists(), "migrated over an unreadable file"
                cfg.write_text('{"embed": {"model": "m-top"}}')
                got = cf.apply()
                assert got["model"] == ("m-top", "global") and got["ollama_host"][1] == "~/.locket/config.json", got
                assert cf.migrate() and not cf.LEGACY.exists() and cf.LEGACY.with_name("config.json.migrated").exists()
                assert load() == ({"embed": {"model": "m-top", "ollama_host": "http://old:1"}}, []), load()
                assert cf.migrate() is None, "a second migration ran"
                cf.LEGACY.write_text('{"embed": {"model": "m-again"}}')
                assert "migrated-" in cf.migrate() and cf.LEGACY.with_name("config.json.migrated").exists(), \
                    "a second migration overwrote the first one's copy"
    finally:
        cf.LEGACY = saved_legacy
        for v, val in saved_env.items():
            if val is not None:
                os.environ[v] = val
        cf.apply()
    return 0


class ConfigureBehaviour(unittest.TestCase):
    def test_behaviour(self):
        with fixture_env():
            _suite()


class VenvMismatch(unittest.TestCase):
    def test_a_venv_built_by_another_python_is_named(self):
        running = f"python{sys.version_info.major}.{sys.version_info.minor}"
        saved = _embed.VENV
        with tempfile.TemporaryDirectory() as t:
            try:
                _embed.VENV = Path(t)
                self.assertEqual(_embed.venv_mismatch(), [], "an absent lib/ is not a mismatch")
                (Path(t) / "lib" / "python3.0" / "site-packages").mkdir(parents=True)
                self.assertEqual(_embed.venv_mismatch(), ["python3.0"])
                (Path(t) / "lib" / running / "site-packages").mkdir(parents=True)
                self.assertEqual(_embed.venv_mismatch(), [], "a venv holding the running version matches")
            finally:
                _embed.VENV = saved


if __name__ == "__main__":
    unittest.main()
