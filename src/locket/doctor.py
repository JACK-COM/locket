"""What this machine's install can be checked for, one line per check, with the command
that fixes a failure.
"""
import os
import shlex
import shutil
import sys
from functools import cached_property

import adapters
import corpora
import corpus
import health
import install
import ledger
import manifest
import paths
import probes
import _settings
import trust


def _system_line():
    """The OS and interpreter, which Step 7 of the install doc reports: the tools are
    tested on macOS and untested on Linux and Windows."""
    import platform
    mac = platform.mac_ver()[0]
    osname = f"macOS {mac}" if mac else f"{platform.system()} {platform.release()}"
    return f"{osname}, Python {platform.python_version()} ({sys.executable})"


def cmd_status():
    link = install.BIN / "locket"
    t = install.link_target(link)
    if install.PACKAGED:
        print(f"command: {install.command() or 'locket (packaged, not on PATH)'}")
    else:
        print(f"command: {link} -> {t}" if t else f"command: {link} not installed (`locket install`)")
    print(f"scripts: {install.HERE}")
    print(f"on PATH: {'yes' if str(install.BIN) in os.environ.get('PATH', '').split(os.pathsep) else 'no'}")
    print(f"system: {_system_line()}")
    print(f"registry: {paths.PATHS.registry} ({len(trust.registry())} added with `locket init`; "
          f"host memory folders are found without it)")
    print(install.embedder_line())
    n = len(corpora.all_corpora())
    print(f"corpora: {n}  (`locket corpora` lists them)")
    print(adapters.gate_status())
    return 0


class Machine:
    """What several checks read, found once: every store, those holding memory (named first,
    a project slug after), and the declared CSVs with their sizes. `version` is this
    release's, for a host handshake that names its client."""

    def __init__(self, version=""):
        self.version = version

    @cached_property
    def stores(self):
        return corpora.all_corpora()

    @cached_property
    def full(self):
        def holds(r):
            try:
                return corpus.has_content(r)
            except OSError:
                return False
        return sorted((n for n, r in self.stores.items() if holds(r)), key=lambda n: (n.startswith("-"), n))

    @cached_property
    def sizes(self):
        return probes.declared_sizes(self.stores, self.full)


def check_command(m, row):
    """The `locket` command on PATH."""
    w = shutil.which("locket")
    row("ok", "command", w) if w else row("fail", "command", "`locket` is not on PATH", "locket install")


def check_selftest(m, row):
    """This install end to end on scratch stores."""
    broken = probes.smoke()
    row("ok", "selftest", "the write hook fires and an audit finds a pair") if broken is None \
        else row("fail", "selftest", broken[:160], "locket selftest")


def check_stores(m, row):
    """The stores found, those holding memory named first."""
    if m.stores:
        # Named stores with something in them first; a project slug or an empty auto-memory
        # (one per folder a session ever ran in) is counted, not listed.
        empty = len(m.stores) - len(m.full)
        shown = ", ".join(m.full[:6]) + (f", and {len(m.full) - 6} more" if len(m.full) > 6 else "")
        row("ok", "stores", (f"{len(m.full)} with memory ({shown})" if m.full else "none holds memory yet")
            + (f"; {empty} empty, not listed" if empty else ""))
    else:
        row("fail", "stores", "none found", "locket init <folder of markdown>")


def check_trust(m, row):
    """Registry rows unpinned, store files never read, manifests acting as no store, and rows whose manifest is gone."""
    # A host store or a project memory is a store by convention and needs no manifest.
    conventional = set(corpora.raw_corpora().values())
    unpinned = [r for r in trust.registry() if str(r) not in trust.registry_pins()]
    if unpinned:
        row("warn", "trust", f"{len(unpinned)} stores registered before 0.12 act as no store until pinned: "
            + ", ".join(map(str, unpinned[:4])), "locket migrate  (at a terminal)")
    skipped = []
    for r in set(m.stores.values()):
        try:
            skipped += corpus.skipped_files(r)
        except OSError:
            continue
    if skipped:
        shown = "; ".join(f"{p} ({why})" for p, why in skipped[:4]) + (
            f"; and {len(skipped) - 4} more" if len(skipped) > 4 else "")
        row("warn", "skipped", f"{len(skipped)} store files are never read: {shown}",
            "replace a link or hard link with the file itself; a markdown file past the cap is not read")
    try:
        loose = trust.untrusted_stores()
    except Exception as e:                  # a failed scan is reported, never read as "none found"
        loose = None
        row("warn", "trust", f"the scan for untrusted stores failed ({type(e).__name__}: {e})")
    for d in loose or []:
        why = ("changed since you trusted it" if trust.changed_since_trust(d) else "not trusted")
        row("warn", "trust", f"{d} holds a {manifest.MANIFEST} that is {why}, so it acts as no store: "
            "its trigger rows, ledger schemas and sources are ignored",
            f"locket trust {shlex.quote(str(d))}  (at a terminal, if you made it), or delete its {manifest.MANIFEST}")
    if loose is not None:
        row("ok", "trust", "scope: directories Claude Code, Codex or Hermes have opened a session in, "
            "and their parents" + ("" if loose else "; no untrusted store found"))
    for gone in (p for p in trust.registry_stale() if p not in conventional):
        q = shlex.quote(str(gone))
        row("warn", "registry", f"{gone} is registered but its {manifest.MANIFEST} is gone, so `all` and the "
            "store list no longer include it", f"locket init {q}  (to keep it), or  locket forget {q}")


def check_config(m, row):
    """The settings file: readable, current, and not overridden by the environment unawares."""
    import configure
    cfg_data, cfg_problems = _settings.load()
    cfg_path = _settings.path()
    legacy = configure.legacy_path()
    if _settings.unreadable(cfg_problems):
        row("fail", "config", cfg_problems[0][:160], f"fix {cfg_path}, or `locket configure embedder --reset`")
    elif cfg_problems:
        row("warn", "config", "; ".join(cfg_problems)[:160], f"fix {cfg_path}; each bad key falls to its default")
    elif legacy.exists():
        row("warn", "config", f"{legacy} is " + ("still read" if configure.legacy_settings() is not None else
            "unreadable, left in place") + f"; settings now live in {cfg_path}", "locket migrate  (moves it)")
    else:
        resolved = configure.apply()
        in_file = {k for _, layer in configure.layers(cfg_data) for k in layer}
        overridden = [f"{k} by {src[4:]}" for k, (_, src) in resolved.items()
                      if src.startswith("env") and k in in_file]
        detail = f"{cfg_path}" if cfg_path.exists() else "no file; every setting is its default"
        row("warn", "config", f"{detail}; the environment overrides {', '.join(overridden)}",
            "unset the variable, or set the value with `locket configure embedder`") if overridden \
            else row("ok", "config", detail)


def check_embedder(m, row):
    """Which embedder answers, and whether its model is fetched."""
    import configure
    import _embed
    try:
        name, model = install.load_memfind().resolve_backend()
        src = configure.apply()["model"][1] if name == "ollama" else "fixed in-process model"
        missing = _embed.model_missing() if name == "onnx" else []
        if missing:
            # the rung resolves on its packages alone and fetches the model on first
            # use, which fails silently into word overlap where huggingface.co is blocked
            row("warn", "embedder", f"{name} ({model}), model not yet fetched: {', '.join(missing)}",
                _embed.manual_fetch_help())
        else:
            row("ok", "embedder", f"{name} ({model}; {src})")
    except RuntimeError as e:
        # a venv that exists but cannot serve (onnxruntime too old) names its own fix
        fix = str(e).split("Fix: ", 1)[1] if "Fix: " in str(e) else \
            f"ollama pull {install.load_memfind().MODEL}  (ollama 0.36+; or in-process: `locket help install`)"
        row("warn", "embedder", "none; `find` falls back to word overlap and misses paraphrases", fix)


def check_index(m, row):
    """A meaning index for every store that holds markdown."""
    # an empty store (an auto-memory folder no session has written to) has nothing to
    # index and `index all` skips it, so it never counts against this check
    # The meaning index builds from markdown only, so a store holding only CSV rows has none to build.
    unindexed = [n for n in m.full if not install.load_memfind().cache_paths(m.stores[n])[0].is_file()
                 and corpus.has_content(m.stores[n], csv=False)]
    if unindexed:
        row("warn", "index", f"not built for {', '.join(sorted(unindexed))}", "locket index all")
    elif m.stores:
        row("ok", "index", "built for every store")


def check_csv_size(m, row):
    """Declared CSVs past the size reading."""
    reading = f"{ledger.CSV_REPORT_BYTES / 1048576:g} MB"
    big = [s for s in m.sizes if s[3] > ledger.CSV_REPORT_BYTES]
    if big:
        more = f", and {len(big) - 4} more" if len(big) > 4 else ""
        row("warn", "csv size", "; ".join(f"{p.name} in {n} is {b / 1048576:.1f} MB" for n, p, _t, b in big[:4])
            + f"{more}, past the {reading} reading",
            "read the csv latency row; the file stays whole (RULINGS.csv on Area=substrate)")
    elif m.sizes:
        row("ok", "csv size", f"{len(m.sizes)} declared CSV(s), every one under {reading}")


def check_hook_health(m, row):
    """Live hook runs a host abandoned at its timeout, or whose catch-all swallowed an error."""
    # Live runs, every host alike: a hook the host abandoned at its timeout, or one whose
    # catch-all swallowed an error, let its call through unchecked.
    rec, problems = health.health_report()
    if problems:
        row("fail", "hook health", f"since {rec.get('since', '?')}: " + "; ".join(problems),
            "find the cause, then `locket doctor --ack` to mark these seen")
    else:
        row("ok", "hook health", "no hook run abandoned or failed open " +
            (f"since {rec.get('since', '?')}" if rec else "on record"))


def check_schema(m, row):
    """The manifest schema file editors read."""
    if not paths.PATHS.schema_file.is_file():
        row("warn", "schema", f"{paths.PATHS.schema_file} not written; editors get no manifest hints", "locket schema")


def check_old_links(m, row):
    """Links under the older command names."""
    old = [n for n in install.LINKS[1:] if (install.BIN / n).is_symlink() and install.ours(install.BIN / n)]
    if old:
        row("warn", "old links", f"{', '.join(old)} still linked in {install.BIN}", "locket install")


def check_hosts(m, row):
    """Each install target's own rows, in `adapters.TARGETS` order."""
    for target in adapters.TARGETS:
        target.checks(m, row)


# Run in order, each adding its rows through `row(level, what, detail, fix=None)`; a new
# check is one function placed here, a host's own checks live in its adapter module.
CHECKS = (check_command, check_selftest, check_stores, check_trust, check_config, check_embedder,
          check_index, check_csv_size, check_hosts, check_hook_health, check_schema, check_old_links)


def cmd_doctor(version=""):
    """Every install step a machine can check, one line each, with the command
    that fixes a failure. Exit 1 when anything fails; a warning is a rung
    missing that Locket works without."""
    rows = []

    def row(level, what, detail, fix=None):
        rows.append((level, what, detail, fix))

    m = Machine(version)
    for check in CHECKS:
        check(m, row)
    width = max(len(r[1]) for r in rows)
    for level, what, detail, fix in rows:
        print(f"{level:<4}  {what:<{width}}  {detail}")
        if fix and level != "ok":
            pad = f"{'':<4}  {'':<{width}}  "
            print(pad + "fix: " + fix.replace("\n", "\n" + pad + "     "))   # a multi-line fix keeps its column
    fails = sum(r[0] == "fail" for r in rows)
    warns = sum(r[0] == "warn" for r in rows)
    print(f"{len(rows)} checks: {len(rows) - fails - warns} ok, {warns} warning(s), {fails} failure(s)")
    return 1 if fails else 0
