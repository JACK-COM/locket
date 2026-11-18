"""Which stores exist, and making, trusting, upgrading and dropping one.
"""
import shlex
import sys
from pathlib import Path

import corpora
import corpus
import fsio
import manifest
import paths
import scaffold
import trust
from commands import Tool, Verb


def corpora_lines(width=70):
    """One line per store: markdown files plus registered CSV rows, the same
    sources `audit` counts. The CLI and the MCP server both print this, because
    a listing that counted markdown alone read as a half-indexed store beside an
    audit that counted six."""
    out = []
    for k, v in sorted(corpora.all_corpora().items()):
        rows = len(corpus.csv_rows(v))
        joined = len(corpora.roots_of(v)) - 1
        out.append(f"  {k:{width}s} {len(corpus.md_files(v)):4d} files"
                   + (f" + {rows} csv rows" if rows else "")
                   + (f"  [{joined} joined store{'s' if joined > 1 else ''}]" if joined else ""))
    for p in trust.registry_stale():
        out.append(f"  {str(p):{width}s} registered, but its {manifest.MANIFEST} is gone; "
                   f"`init` it again to keep it, `forget` it to drop the row")
    return out


def _corpora(rest):
    print("\n".join(corpora_lines()))
    return 0


def _init(rest):
    rest, flags = list(rest), {}
    led = [a for a in rest if a == "--ledgers" or a.startswith("--ledgers=")]
    for a in led:
        rest.remove(a)
    ledgers = led[-1].partition("=")[2] if led else None
    for f in ("--parent", "--name"):
        if f in rest:
            i = rest.index(f)
            if i + 1 >= len(rest):
                print(f"{f} needs a value", file=sys.stderr)
                return 1
            flags[f] = rest[i + 1]; del rest[i:i + 2]
    if len(rest) != 1:
        print("init needs a directory:  locket init <dir> [--name <label>] [--parent <corpus>] "
              "[--ledgers[=<Name>]]  (a ledger name takes `=`, never a space)", file=sys.stderr)
        return 1
    try:
        return scaffold.cmd_init(rest[0], parent=flags.get("--parent"), name=flags.get("--name"), ledgers=ledgers)
    except OSError as e:                # a refused write (a symlinked manifest or ledger) is named
        print(f"init: {e}", file=sys.stderr)
        return 1


def _migrate(rest):
    import migrations
    return migrations.run(rest[0] if rest and rest[0] != "all" else None)


def _withdraw(verb):
    """`forget` and `untrust`: withdrawing a store's guardrails is a person's act."""
    def run(rest):
        if not rest:
            print(f"{verb} needs a directory:  locket {verb} <dir>", file=sys.stderr)
            return 1
        d = Path(rest[0]).expanduser()
        if any(fsio.same(d, r) for r in trust.registry()) and not trust.person_confirms(
                f"Withdraw trust from {d}? Its trigger rows, ledger schemas and sources stop acting.",
                [], "withdraw it"):
            print(f"registry: unchanged. Withdrawing a store's guardrails is a person's act at a "
                  f"terminal; ask the user to run:  locket {verb} {shlex.quote(str(d))}")
            return 1
        print(f"registry: {paths.PATHS.registry} ({'row removed' if trust.forget(rest[0]) else 'not listed'})")
        return 0
    return run


def _trust(rest):
    if not rest or not Path(rest[0]).expanduser().is_dir():
        print("trust needs a directory:  locket trust <dir>", file=sys.stderr)
        return 1
    return 0 if trust.grant_trust(rest[0]) else 1


def _schema(rest):
    if rest:
        findings = manifest.validate_manifest(Path(rest[0]).expanduser())
        print("\n".join(findings) if findings else f"{manifest.manifest_file(Path(rest[0]).expanduser())}: valid")
        return 1 if findings else 0
    return scaffold.cmd_schema()


def _init_tool(args, session):
    """Write a folder's manifest. Never trusts it: trust is a person's act at a
    terminal, so the reply names that command instead."""
    import contextlib
    import io
    d = str(args.get("path") or "").strip()
    if not d:
        raise ValueError("path is required: the absolute path of the folder to make a store")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = scaffold.cmd_init(d, confirm=lambda _d: False)
    return (out.getvalue() + err.getvalue()).strip() or f"init returned {rc}"


def _forget_tool(args, session):
    d = str(args.get("path") or "").strip()
    if not d:
        raise ValueError("path is required")
    if any(fsio.same(Path(d).expanduser(), r) for r in trust.registry()):
        return (f"{d} is trusted, and withdrawing a store's guardrails is a person's act: ask the user "
                f"to run  locket untrust {d}  at a terminal")
    return f"{d} is not in the registry"


PATH_ARG = {"path": {"type": "string", "description": "Absolute path of the folder."}}

SEE = "scan"
VERBS = (
    Verb("corpora", "", "every store on this machine", None, run=_corpora,
         mcp=Tool("Every memory store on this machine, with markdown file and "
                  "registered CSV-row counts, the sources every check reads.",
                  # the same counts `audit` reports: a listing that counted markdown
                  # alone read as a half-indexed store
                  lambda a, s: "\n".join(corpora_lines(60)) or "no memory stores found on this machine")),
    Verb("init", "<dir> [--name <label>] [--parent <corpus>] [--ledgers[=<Name>]]",
         "make a folder of markdown a store; running it again changes nothing",
         "locket init ~/notes                   the folder becomes a store, named after it\n"
         "locket init ~/notes --name work       names the store `work`; works on a Claude Code project store too\n"
         "locket init ~/docs --parent notes     ~/docs joins the store `notes`: two folders, one store\n"
         "locket init ~/notes --ledgers         adds RULINGS-notes.csv (decisions), CLAIMS-notes.csv\n"
         "                                      (facts with sources) and HISTORY-notes-Sessions.csv (one\n"
         "                                      row per session), header only; rows off their columns\n"
         "                                      are refused as they are written\n"
         "locket init ~/notes --ledgers=Work    the same files, named RULINGS-Work.csv and so on",
         run=_init,
         mcp=Tool("Make a folder of markdown a memory store every other tool can find: "
                  "writes locket.json if absent. It does not trust the folder: a person must "
                  "run `locket trust <path>` at a terminal before the store acts, and "
                  "the reply names that command for you to pass on. Idempotent.", _init_tool, PATH_ARG)),
    Verb("migrate", "[<dir>|all]", "finish an upgrade: move older settings files into their current home and rename "
         "each store's older memfind.json manifest to locket.json; safe to run any time",
         "locket migrate                        everything this machine needs after an upgrade\n"
         "locket migrate ~/notes                one store's manifest only", run=_migrate),
    Verb("schema", "[<dir>]", "write the manifest schema, or check a manifest",
         "locket schema                         write ~/.locket/locket.schema.json for editor hints\n"
         "locket schema ~/notes                 check that store's locket.json against it", run=_schema),
    Verb("trust", "<dir>", "let a store's locket.json act (trigger rows, ledger schemas, sources); "
         "asks you to type yes at a terminal, so an agent cannot grant it",
         "locket trust ~/Software/myapp", run=_trust),
    Verb("untrust", "<dir>", "withdraw a store's trust; its locket.json stays, inert",
         "locket untrust ~/Software/myapp", run=_withdraw("untrust")),
    Verb("forget", "<dir>", "stop listing a store and withdraw its trust; its files and locket.json stay",
         "locket forget ~/old-notes", run=_withdraw("forget"),
         mcp=Tool("Drop a store from the registry. The folder's locket.json "
                  "stays in the folder, inert; its index under ~/.locket/cache stays until uninstall.",
                  _forget_tool, PATH_ARG)),
)
