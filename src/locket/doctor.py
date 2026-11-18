"""What this machine's install can be checked for, one line per check, with the command
that fixes a failure.
"""
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import _settings
import adapters
import hosts
import install
import memscan
import paths
import probes


def _system_line():
    """The OS and interpreter, which Step 7 of the install doc reports: the tools are
    tested on macOS and untested on Linux and Windows."""
    import platform
    mac = platform.mac_ver()[0]
    osname = f"macOS {mac}" if mac else f"{platform.system()} {platform.release()}"
    return f"{osname}, Python {platform.python_version()} ({sys.executable})"


def cmd_status():
    link = install.BIN / "locket"
    t = install._link_target(link)
    if install.PACKAGED:
        print(f"command: {install._command() or 'locket (packaged, not on PATH)'}")
    else:
        print(f"command: {link} -> {t}" if t else f"command: {link} not installed (`locket install`)")
    print(f"scripts: {install.HERE}")
    print(f"on PATH: {'yes' if str(install.BIN) in os.environ.get('PATH', '').split(os.pathsep) else 'no'}")
    print(f"system: {_system_line()}")
    print(f"registry: {paths.PATHS.registry} ({len(memscan.registry())} added with `locket init`; "
          f"host memory folders are found without it)")
    print(install._embedder_line())
    n = len(memscan.all_corpora())
    print(f"corpora: {n}  (`locket corpora` lists them)")
    print(adapters.gate_status())
    return 0


def cmd_doctor(version=""):
    """Every install step a machine can check, one line each, with the command
    that fixes a failure. Exit 1 when anything fails; a warning is a rung
    missing that Locket works without."""
    rows = []

    def row(level, what, detail, fix=None):
        rows.append((level, what, detail, fix))

    w = shutil.which("locket")
    row("ok", "command", w) if w else row("fail", "command", "`locket` is not on PATH", "locket install")

    broken = probes._smoke()
    row("ok", "selftest", "the write hook fires and an audit finds a pair") if broken is None \
        else row("fail", "selftest", broken[:160], "locket selftest")

    stores = memscan.all_corpora()
    full = []
    if stores:
        # Named stores with something in them first; a project slug or an empty auto-memory
        # (one per folder a session ever ran in) is counted, not listed.
        def _holds(r):
            try:
                return memscan.has_content(r)
            except OSError:
                return False
        full = sorted((n for n, r in stores.items() if _holds(r)), key=lambda n: (n.startswith("-"), n))
        empty = len(stores) - len(full)
        shown = ", ".join(full[:6]) + (f", and {len(full) - 6} more" if len(full) > 6 else "")
        row("ok", "stores", (f"{len(full)} with memory ({shown})" if full else "none holds memory yet")
            + (f"; {empty} empty, not listed" if empty else ""))
    else:
        row("fail", "stores", "none found", "locket init <folder of markdown>")
    # A host store or a project memory is a store by convention and needs no manifest.
    conventional = set(memscan._raw_corpora().values())
    # a store trusted by convention has no pin, so a rows file appearing in one is named
    for d in sorted((hosts.claude() / "projects").glob("*/memory")):
        if (d / "triggers.json").is_file():
            row("warn", "config", f"{d / 'triggers.json'} adds trigger rows to a project's auto-memory, "
                "which is trusted without a pin; read it, and delete it if you did not write it")
    unpinned = [r for r in memscan.registry() if str(r) not in memscan.registry_pins()]
    if unpinned:
        row("warn", "trust", f"{len(unpinned)} stores registered before 0.12 act as no store until pinned: "
            + ", ".join(map(str, unpinned[:4])), "locket migrate  (at a terminal)")
    skipped = []
    for r in set(stores.values()):
        try:
            skipped += memscan.skipped_files(r)
        except OSError:
            continue
    if skipped:
        shown = "; ".join(f"{p} ({why})" for p, why in skipped[:4]) + (
            f"; and {len(skipped) - 4} more" if len(skipped) > 4 else "")
        row("warn", "skipped", f"{len(skipped)} store files are never read: {shown}",
            "replace a link or hard link with the file itself; a markdown file past the cap is not read")
    try:
        loose = memscan.untrusted_stores()
    except Exception as e:                  # a failed scan is reported, never read as "none found"
        loose = None
        row("warn", "trust", f"the scan for untrusted stores failed ({type(e).__name__}: {e})")
    for d in loose or []:
        why = ("changed since you trusted it" if memscan.changed_since_trust(d) else "not trusted")
        row("warn", "trust", f"{d} holds a {memscan.MANIFEST} that is {why}, so it acts as no store: "
            "its trigger rows, ledger schemas and sources are ignored",
            f"locket trust {shlex.quote(str(d))}  (at a terminal, if you made it), or delete its {memscan.MANIFEST}")
    if loose is not None:
        row("ok", "trust", "scope: directories Claude Code, Codex or Hermes have opened a session in, "
            "and their parents" + ("" if loose else "; no untrusted store found"))
    for gone in (p for p in memscan.registry_stale() if p not in conventional):
        q = shlex.quote(str(gone))
        row("warn", "registry", f"{gone} is registered but its {memscan.MANIFEST} is gone, so `all` and the "
            "store list no longer include it", f"locket init {q}  (to keep it), or  locket forget {q}")

    import configure
    import _embed
    cfg_data, cfg_problems = _settings.load()
    cfg_path = _settings.path()
    legacy = configure._legacy_path()
    if _settings.unreadable(cfg_problems):
        row("fail", "config", cfg_problems[0][:160], f"fix {cfg_path}, or `locket configure embedder --reset`")
    elif cfg_problems:
        row("warn", "config", "; ".join(cfg_problems)[:160], f"fix {cfg_path}; each bad key falls to its default")
    elif legacy.exists():
        row("warn", "config", f"{legacy} is " + ("still read" if configure._legacy() is not None else
            "unreadable, left in place") + f"; settings now live in {cfg_path}", "locket migrate  (moves it)")
    else:
        resolved = configure.apply()
        in_file = {k for _, layer in configure._layers(cfg_data) for k in layer}
        overridden = [f"{k} by {src[4:]}" for k, (_, src) in resolved.items()
                      if src.startswith("env") and k in in_file]
        detail = f"{cfg_path}" if cfg_path.exists() else "no file; every setting is its default"
        row("warn", "config", f"{detail}; the environment overrides {', '.join(overridden)}",
            "unset the variable, or set the value with `locket configure embedder`") if overridden \
            else row("ok", "config", detail)
    try:
        name, model = install._memfind().resolve_backend()
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
            f"ollama pull {install._memfind().MODEL}  (ollama 0.36+; or in-process: `locket help install`)"
        row("warn", "embedder", "none; `find` falls back to word overlap and misses paraphrases", fix)

    # an empty store (an auto-memory folder no session has written to) has nothing to
    # index and `index all` skips it, so it never counts against this check
    # The meaning index builds from markdown only, so a store holding only CSV rows has none to build.
    unindexed = [n for n in full if not install._memfind().cache_paths(stores[n])[0].is_file()
                 and memscan.has_content(stores[n], csv=False)]
    if unindexed:
        row("warn", "index", f"not built for {', '.join(sorted(unindexed))}", "locket index all")
    elif stores:
        row("ok", "index", "built for every store")

    sizes = probes._declared_sizes(stores, full)
    reading = f"{memscan.CSV_REPORT_BYTES / 1048576:g} MB"
    big = [s for s in sizes if s[3] > memscan.CSV_REPORT_BYTES]
    if big:
        more = f", and {len(big) - 4} more" if len(big) > 4 else ""
        row("warn", "csv size", "; ".join(f"{p.name} in {n} is {b / 1048576:.1f} MB" for n, p, _t, b in big[:4])
            + f"{more}, past the {reading} reading",
            "read the csv latency row; the file stays whole (RULINGS.csv on Area=substrate)")
    elif sizes:
        row("ok", "csv size", f"{len(sizes)} declared CSV(s), every one under {reading}")

    settings = hosts.claude() / "settings.json"
    if hosts.claude().is_dir():
        hooks = probes._registered_hooks(settings) if settings.is_file() else {}
        if hooks is None:
            row("fail", "claude hooks", f"{settings} is not valid JSON", f"fix {settings}, then locket install --hooks")
        else:
            missing = [v for _, v, _t in install.CLAUDE_HOOKS if v not in hooks]
            stale = [v for v, c in hooks.items() if (p := probes._script_in(c)) and not p.is_file()]
            if missing:
                row("fail", "claude hooks", f"not registered: {', '.join(missing)}", "locket install --hooks")
            elif stale:
                row("fail", "claude hooks", f"point at a missing script: {', '.join(stale)}",
                    "locket uninstall --dry-run, then remove the stale entries and locket install --hooks")
            else:
                probe = probes._probe_hook(hooks["hook"])
                if probe is True:
                    row("ok", "claude hooks", "registered, and the write hook fires on a known fork")
                    for verb, what, lost in (("hook", "the write hook", "the write lands unchecked"),
                                             ("grepassist", "grepassist", "the search runs without its neighbours")):
                        if verb not in hooks:
                            continue
                        limit = probes._hook_timeout(settings, verb)
                        slow, opened = probes._hook_latency(hooks[verb], stores, limit, verb)
                        label = f"{verb} latency"
                        if opened is not None:
                            row("fail", label, f"{what} failed open on {opened[0]}: {opened[1]}",
                                "find the cause in that store (an unreadable file, a broken manifest)")
                            continue
                        if slow is None:
                            continue
                        name, secs = slow
                        if secs >= limit:
                            row("fail", label, f"{what} takes {secs:.0f}s or more on {name}, past its {limit:g}s "
                                f"timeout: Claude Code abandons it and {lost}",
                                "brew upgrade locket; if current, the store's walk is entering host state")
                        elif secs > limit / 2:
                            row("warn", label, f"{what} takes {secs:.1f}s on {name}, over half its {limit:g}s "
                                f"timeout; past it, {lost}", "find what the store's walk enters that it should not")
                        else:
                            row("ok", label, f"{secs:.1f}s on the slowest store ({name}), inside its {limit:g}s timeout")
                    # Only a registered source is scored row by row, so only sources are timed; the
                    # largest three, because the slowest file need not be the largest.
                    limit = probes._hook_timeout(settings, "hook")
                    timed, skipped, failed = [], [], None
                    for target in [s for s in sizes if s[2] is not None][:3]:
                        secs, problem, why = probes._csv_hook_latency(hooks["hook"], target, limit)
                        if problem:
                            failed = (target, problem)
                            break
                        if why:
                            skipped.append(f"{target[1].name} ({why})")
                        else:
                            timed.append((secs, target))
                            if secs >= limit:
                                break
                    fix = "choose a remedy against this reading (a faster scoring path, an index); never split the file"
                    if failed:
                        row("fail", "csv latency", f"the write hook failed on a row into {failed[0][1].name} in "
                            f"{failed[0][0]}: {failed[1]}", "find the cause in that store")
                    elif not timed:
                        row("warn" if skipped else "ok", "csv latency",
                            ("not timed: " + "; ".join(skipped)) if skipped else "no registered source CSV to time")
                    if timed and not failed:
                        secs, big = max(timed, key=lambda x: x[0])
                        where = f"{big[1].name} in {big[0]} ({big[3] / 1048576:.1f} MB, slowest of {len(timed)} timed)"
                        if secs >= limit:
                            row("fail", "csv latency", f"the write hook takes {secs:.0f}s or more on a row into {where}, "
                                f"past its {limit:g}s timeout: rows land unchecked", fix)
                        elif secs > limit / 2:
                            row("warn", "csv latency", f"the write hook takes {secs:.1f}s on a row into {where}, "
                                f"over half its {limit:g}s timeout", fix)
                        else:
                            row("ok", "csv latency", f"{secs:.1f}s on a row into {where}, inside its {limit:g}s timeout")
                else:
                    row("fail", "claude hooks", "registered, but the write hook " +
                        ("stayed silent on a known fork" if probe is False else f"failed: {probe}"),
                        f"run it by hand: {hooks['hook']}")

    # Live runs, every host alike: a hook the host abandoned at its timeout, or one whose
    # catch-all swallowed an error, let its call through unchecked.
    rec, problems = memscan.health_report()
    if problems:
        row("fail", "hook health", f"since {rec.get('since', '?')}: " + "; ".join(problems),
            "find the cause, then `locket doctor --ack` to mark these seen")
    else:
        row("ok", "hook health", "no hook run abandoned or failed open " +
            (f"since {rec.get('since', '?')}" if rec else "on record"))

    codex_cfg = hosts.codex() / "hooks.json"
    if hosts.codex().is_dir():
        hooks = probes._registered_hooks(codex_cfg) if codex_cfg.is_file() else {}
        missing = [v for _, v, _t in install.CLAUDE_HOOKS if v not in hooks] if hooks is not None else []
        if hooks is None:
            row("fail", "codex hooks", f"{codex_cfg} is not valid JSON", f"fix {codex_cfg}, then locket install --codex")
        elif missing:
            row("warn", "codex hooks", f"not registered: {', '.join(missing)}", "locket install --codex")
        else:
            probe = probes._probe_hook(hooks["hook"], codex=True)
            if probe is True:
                # the probe runs the command directly, so it proves the hook, not Codex's dispatch;
                # whether Codex will dispatch it is trust, which only Codex's own hooks/list may say
                trust = probes._codex_trust(version=version)
                lapsed = [k for k, s in (trust or {}).items() if s != "trusted"]
                gap = probes._profile_trust_gap(hosts.codex(), lapsed) if lapsed else {}
                if not trust:               # {} too: a reply naming no Locket entry proves nothing
                    row("warn", "codex hooks", "registered, and the write hook fires on a known fork in a patch; "
                        "Codex could not report trust, and skips a hook until it is trusted",
                        "in codex, run /hooks and trust each Locket entry")
                elif not lapsed:
                    row("ok", "codex hooks", f"registered, firing on a known fork in a patch, and trusted ({len(trust)} entries)")
                elif gap:
                    prof = ", ".join(f.name for f in gap)
                    row("warn", "codex hooks", f"{len(lapsed)} entries untrusted in config.toml, but trusted in {prof}, "
                        "which Codex writes under --profile and never reads back",
                        f"copy the [hooks.state.\"{codex_cfg}:…\"] tables from {prof} into config.toml; "
                        "recopy after any change to hooks.json")
                else:
                    row("warn", "codex hooks", "registered and firing, but Codex reports " +
                        ", ".join(f"{k.split(':', 1)[1]} {trust[k]}" for k in lapsed),
                        "in codex, run /hooks and trust each Locket entry")
            else:
                row("fail", "codex hooks", "registered, but the write hook " +
                    ("stayed silent on a known fork" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {hooks['hook']}")

    for label, cfg in (("claude", settings), ("codex", codex_cfg)):
        if cfg.is_file():
            dead = probes._dead_hooks(cfg)
            if dead:
                ev, cmd, why = dead[0]
                more = f" (and {len(dead) - 1} more)" if len(dead) > 1 else ""
                row("fail", f"{label} entries", f"{ev}: `{cmd}` cannot run: {why}{more}",
                    f"remove the dead entries from {cfg}, then locket install --{'hooks' if label == 'claude' else 'codex'}")
            else:
                row("ok", f"{label} entries", "every Locket entry, under every event, runs")

    import trigger
    rows_path = trigger.rows_file(hosts.claude())
    if rows_path.is_file():
        try:
            events = (json.loads(settings.read_text()).get("hooks") or {}) if settings.is_file() else {}
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            events = None
        unregistered = [e for e, _m in install.CLAUDE_TRIGGER
                        if events is not None and not install._trigger_command(events.get(e) if isinstance(events, dict) else None)]
        problems = trigger.findings(hosts.claude())
        if unregistered:
            row("fail", "triggers", f"{rows_path} has rows, but the hook is not registered for "
                f"{' or '.join(unregistered)}", "locket install --hooks")
        elif problems:
            row("fail", "triggers", f"{len(problems)} problem(s) in {rows_path}: {problems[0]}", "locket trigger check")
        elif events is not None:
            probe = probes._probe_trigger(install._trigger_command(events.get("PreToolUse")))
            if probe is True:
                row("ok", "triggers", f"{rows_path} lints clean; the hook is registered and fires on a known row")
            else:
                row("fail", "triggers", "registered, but the hook " +
                    ("stayed silent on a known row" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {install._trigger_command(events.get('PreToolUse'))}")

    import time
    if hosts.claude().is_dir():
        try:
            events = (json.loads(settings.read_text()).get("hooks") or {}) if settings.is_file() else {}
            registered = install._event_command(events.get(install.CLAUDE_USAGE[0]), install.CLAUDE_USAGE[1])
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            registered = None
        age = (time.time() - paths.PATHS.usage_ledger.stat().st_mtime) / 86400 if paths.PATHS.usage_ledger.is_file() else None
        seen = "never recorded" if age is None else f"last recorded {age:.1f} days ago"
        stale = registered and (p := probes._script_in(registered)) and not p.is_file()
        if stale:
            row("fail", "usage", f"the SessionEnd hook names a missing script: {p}",
                "locket uninstall --dry-run, then remove the stale entry and locket install --hooks")
        elif not registered:
            row("warn", "usage", f"{seen}; nothing records it as a session ends, so a day older than "
                "Claude Code keeps its transcripts is lost unless `locket usage` ran within it", "locket install --hooks")
        else:
            row("ok", "usage", f"recorded as each session ends; {seen}")

    hermes_cfg = hosts.hermes() / "config.yaml"
    if hermes_cfg.is_file():
        events = install._hermes_hooks(hermes_cfg.read_text(errors="replace"))
        missing = [f"{v} on {e}" for e, v in install.HERMES_HOOKS
                   if not any(install._is_locket_hook(c, v) for c in events.get(e, []))]
        retired = any(install._is_locket_hook(c, "deliver") for cmds in events.values() for c in cmds)
        if missing:
            row("warn", "hermes hooks", f"not in config.yaml: {', '.join(missing)}"
                + ("; `locket deliver` is retired, remove it" if retired else ""), "`locket help install`, Step 4")
        elif retired:
            row("warn", "hermes hooks", "`locket deliver` is retired and does nothing", "remove its pre_llm_call entry")
        else:
            row("ok", "hermes hooks", "all five registered")
        cmd = next((c for c in events.get("pre_tool_call", []) if install._is_locket_hook(c, "trigger")), None)
        hermes_rows = trigger.rows_file(hosts.hermes())
        problems = trigger.findings(hosts.hermes())
        if problems:
            row("fail", "hermes triggers", f"{len(problems)} problem(s) in {hermes_rows}: {problems[0]}",
                "locket trigger check hermes")
        elif hermes_rows.is_file() and cmd:
            probe = probes._probe_trigger(cmd, hermes=True)
            if probe is True:
                row("ok", "hermes triggers", f"{hermes_rows} lints clean; the hook refuses a call on a known row")
            else:
                row("fail", "hermes triggers", "registered, but the hook " +
                    ("stayed silent on a known row" if probe is False else f"failed: {probe}"),
                    f"run it by hand: {cmd}")

    if install.DESKTOP_CONFIG is not None and install.DESKTOP_CONFIG.is_file():
        try:
            srv = (json.loads(install.DESKTOP_CONFIG.read_text()).get("mcpServers") or {}).get("locket")
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            srv = None
        if srv:
            parts = [Path(srv.get("command", ""))] + [Path(a) for a in srv.get("args", []) if str(a).endswith(".py")]
            gone = [str(p) for p in parts if not p.is_file()]
            row("fail", "desktop server", f"names a missing file: {', '.join(gone)}", "locket install --desktop") \
                if gone else row("ok", "desktop server", "registered; restart Claude Desktop after a change")

    if not paths.PATHS.schema_file.is_file():
        row("warn", "schema", f"{paths.PATHS.schema_file} not written; editors get no manifest hints", "locket schema")

    old = [n for n in install.LINKS[1:] if (install.BIN / n).is_symlink() and install._ours(install.BIN / n)]
    if old:
        row("warn", "old links", f"{', '.join(old)} still linked in {install.BIN}", "locket install")

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
