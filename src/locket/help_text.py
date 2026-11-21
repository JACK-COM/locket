"""The help text both fronts print: memscan's full help (`locket help scan`), its MANIFEST
section alone (`locket help manifest`), and the top-level epilog and corpus note `locket -h`
and each verb's `-h` end with.
"""
import manifest

USAGE = """memscan: duplication, drift and link checks over a memory corpus.

USAGE
  locket <mode> [corpus]          (memscan.py <mode> [corpus] by path)

  The corpus argument is optional and takes a shorthand (council, hermes)
  or registered name, any substring of a Claude Code project slug (myapp), a
  literal path, or `all` for every corpus on the machine in turn. Omit it to
  use the corpus the current directory sits inside, else the one belonging to
  it.

  A corpus is a directory of markdown files. A host's core directory is its
  COUNCIL store, recognised on sight: `~/.claude` on Claude Code, where the
  members are memory, rules, skills, agents and agent-memory plus the markdown
  at the root, and `~/.hermes` on Hermes, where the member is memories plus
  SOUL.md. Everything else under the home is host state and never corpus, and
  each `~/.claude/projects/<slug>/memory` is a store of its own. Any other
  directory becomes one with `init <dir>`, which writes a `locket.json` manifest into it
  and registers the path in `~/.locket/corpora.json` so that `corpora`, `all`
  and a name can find it. The manifest alone is honoured by the write-time
  gate and by a literal path; only the registry makes a store LISTABLE, since
  nothing searches the disk for manifests. The manifest may also name the
  store, rename the ledger, holding-space and index files this tool treats
  specially, exclude individual files (`excluded_files`: relative paths or
  globs, skipped by every mode and by the gate), declare the code roots
  `cites` resolves against, and register reference CSVs. A manifest acts only
  once a person has trusted its directory (`locket trust`); until then it is
  read as no manifest. The manifest, the registry row and the index under
  ~/.locket/cache are all this tool owns for a corpus.

MANIFEST
  `locket.json` at the corpus root, a JSON object (the older name `memfind.json`
  is still read; `migrate` renames it); `schema` writes its JSON
  Schema for an editor and checks a manifest against it. Every key is optional and
  a missing key takes the default shown; a broken file degrades to all the
  defaults rather than taking a hook down. `$schema` and `_comment` are ignored.
  This is the whole list, grouped by the question each key answers.

  IS THIS FILE CORPUS AT ALL?
  An excluded file does not exist for the tool: nothing indexes, pairs or
  guards it. To keep a file in the corpus but change how it is scored, give it
  a surface role in the next group instead.

  members          Which top-level directories are corpus? A list of names;
                   the rest of the root is host state, and markdown at the
                   root is always in. Default: every directory, except on a
                   host's core directory, where it is the host's member list
                   above. REPLACES that list, so a Hermes that authors its own
                   skills writes ["memories", "skills"].
  excluded_dirs    Which directories, at any depth, are not corpus? A list of
                   names. REPLACES the default: archive, in-flight,
                   harness-logs, discovery-research. Whatever the list says,
                   .memfind, backups, backup and every dot-directory stay
                   excluded.
  excluded_files   Which single files are not corpus? A list of
                   corpus-relative paths or glob patterns, each tried against
                   the relative path and the basename: "notes/draft.md",
                   "*.csv", "scratch-*". Skipped by every mode, the index and
                   the write-time gate. Also removes a registered CSV it
                   matches. Use it for a file that holds no claims, such as a
                   draft or a template whose instances copy its shape.
                   Default: none.

  HOW IS THIS FILE SCORED FOR FORKS?
  Each list names files by basename and gives them a surface role. A corpus
  file in none of them is DURABLE: every claim in it is scored against every
  other file in the store. Each list REPLACES its default.

  index_files      Which files POINT at other files? A line restating its
                   target is the index doing its job, not a fork. Default:
                   MEMORY.md, ARCHIVE.md, CURATION-INDEX.md,
                   CURATION-SOURCES-INDEX.md, FRONTEND-INDEX.md,
                   PLATFORM-INDEX.md.
  ledger_surfaces  Which markdown files grow by appending entries? They are
                   never scored, because chronology is their value and an
                   append is the discipline, not a breach. Not the same as
                   `ledgers` below, which checks the row shape of a CSV.
                   Default: HARNESS.md, CHECKPOINTS.md, EXPLORATIONS.md.
  holding_spaces   Which files stage entries on their way to a durable home?
                   Scored only against THEMSELVES: a restated entry fires, a
                   fact already held elsewhere does not, since an entry is
                   meant to resemble what it may become. Unlike an excluded
                   file it is fully corpus: indexed, searchable, and read by
                   `graduated` for entries whose lesson has landed elsewhere.
                   Default: SHORT_TERM.md, OPEN_QUESTIONS.md,
                   MEMORY-proposed.md, PROMOTIONS.md.

  MAY THIS FILE OUTGROW A WHOLE READ?
  whole_read_exempt
                   Which files does the SessionStart size sweep skip? A list
                   of names: a mixed surface its own load contract says is
                   never read whole, or a file a project instrument already
                   caps. It exempts nothing else; every other mode still reads
                   the file, and a file may carry this and a surface role at
                   once. Default: none.

  WHICH CSV ROWS COUNT, AND IN WHAT SHAPE?
  A CSV is scored for forks only when `sources` lists it, and its rows are
  checked for shape only when a `ledgers` schema governs it; a rulings file
  usually takes both.

  sources          Which CSVs hold claims? A list of {"path", "text",
                   "label"}: `path` is the file, absolute or relative to the
                   store, `text` the column holding the sentence, `label` the
                   column that names a row in output (an Area). Default: the
                   older standalone reference-sources.json if present, else
                   none.
  ledgers          What shape must a CSV ledger's rows take? A list of
                   {"path", "extends", "columns", "required", "caps", "enum",
                   "pattern"}; a row outside its schema is refused at write
                   time. It says nothing about forks. `path` is a basename
                   glob or a store-relative path; `columns` the exact header;
                   `required` the columns that may not be empty; `caps`
                   {column: max characters}; `enum` {column: [values]}, ""
                   among them where empty is allowed; `pattern` {column: regex
                   the whole value must match}. Every key but `path` is
                   optional. A built-in schema governs its glob wherever the
                   file sits, corpus or not, and is the authority on it. An
                   entry naming that glob, or a path under it, merges onto it
                   for this store: `caps`, `enum` and `pattern` change only the
                   columns they name, `columns` and `required` replace.
                   `extends` names a built-in's glob to merge onto for a file
                   whose name falls outside it.
                   Default: the built-ins alone, which are:
@@BUILTIN_LEDGERS@@

  HOW DOES THIS STORE CONNECT TO THE REST?
  name             What do `corpora` and a store argument call this store? A
                   string. Default: the directory's basename. Host stores and
                   project memories are named by the host and ignore it.
  belongs_to       Which corpus does this directory JOIN? The name of another
                   corpus as `corpora` lists it. Every mode, the index and the
                   write-time gate then resolve a file here to the parent, the
                   listing shows the parent alone, and its files appear as
                   `<name>/<path>` (this store's `name`, else its basename).
                   The store keeps its own `members`, `excluded_*` and
                   `code_roots` for its own walk; the file-name lists union
                   upward. Written by `init <dir> --parent <corpus>`. A name
                   that resolves to nothing, to itself, to a store under it on
                   disk or whose own walk reaches it, or into a cycle is
                   ignored and the store stands alone. Default: none.
  code_roots       Where do `cites` path:line citations resolve? A list of
                   directories, taken outright. Default: the project's own
                   working directory (for a project memory) plus ~/.claude and
                   ~/.hermes where they exist.
  triggers         Which rows file does `locket trigger` read for this store?
                   A path, store-relative or absolute. Default: triggers.json
                   at the store root.

  On Hermes's ~/.hermes the three surface-role lists default to empty
  instead, because its MEMORY.md is the whole memory rather than an index; a
  manifest there still wins. Store the file under version control: it is
  authored configuration, where the index is derived and rebuilt.

MODES
  audit     cross-file claim pairs that currently AGREE, so fork RISK rather
            than proof of a fork. An index line matching its own target is the
            index doing its job, not a finding.
  across    claim pairs that NO single store holds both sides of, over every
            store on the machine at once. Every other mode and the write-time
            gate score a corpus against itself, so a fact written into two
            stores is invisible to both until this runs. A file registered by
            two manifests is one file here, named with every holder. Tags a
            pair with a holding-space side: promoted into another store and
            never retired, which `graduated` cannot see. Takes no corpus.
  dupes     repeated claims INSIDE one file, which audit skips by construction.
            Use on the holding spaces (SHORT_TERM, OPEN_QUESTIONS, proposed).
            Name a CSV registered under `sources` and its rows are ranked by
            MEANING, since a reference table breaks one-fact-per-row by
            paraphrase: closest pairs, `--incoming rows.csv` against the table,
            `--against file.md` for rows a rule already states, `--json` for a
            worklist. Ranked, never a verdict; word overlap, said so, with no
            embedder.
  graduated holding-space claims a durable file or CSV row now holds: an entry
            promoted and never retired. Prints its scope even when clean, so a
            zero says what it was scored against. Scores ONE store: a promotion
            into a second store is `across`'s to tag. Run with `links` and
            `across` at session close.
  ledgers   every schema-governed CSV in the corpus, row by row: a header off
            the schema, a required field empty, a field over its cap, outside
            its enum or off its pattern. The same check the write-time gate
            runs on a row as it is written, over the rows already there.
            Then every declared CSV (a ledger or a registered source) past
            5 MB, each a finding: the file stays whole, and `doctor` times
            the write hook on the largest.
  budget    bytes, dated history clauses listed by line (the reconsolidation
            worklist: `*Refinement, <date>:*`, `<date>: Resolved:`, `Update
            <date>`), longest prose paragraph. Nothing else measures a file
            that only disagrees with the world.
  pointers  every index line paired to the file it points at, a missing
            target first. NEEDS A READER: a hook asserting what its target now
            refutes scores LOW and is invisible to audit. `hooks` is its older
            name; whether the write-time gate is armed is `locket status`.
  links     wikilink targets resolving to no file.
  cites     `path:line` citations resolved against the code, with the enclosing
            def/class for each. Prints a missing file or a past-EOF line as a
            defect; everything else NEEDS A READER, since only a reader knows
            whether the scope is the one the sentence claimed. Takes optional
            path substrings to narrow the list.
  corpora   every corpus on this machine, with file and CSV-row counts, and
            any registered path whose manifest has since gone.
  init DIR  make DIR a corpus every mode can find: manifest, registry row,
            index. Idempotent. The one command for a folder that already
            holds markdown. With `--parent NAME`, DIR joins the corpus NAME
            instead (`belongs_to` in its manifest, index rebuilt at the
            parent): two folders in two places, one corpus. Works on a
            project memory too, which otherwise takes no manifest. The
            joined files are keyed `<label>/<path>`, the label being the
            manifest's `name` or else the folder's; `--name LABEL` sets it,
            and a label clashing with the parent's name or one of its
            directories is refused. A host auto-memory folder is always
            `memory`, so it takes `--name`. A standalone store's manifest is
            written with `name` (the folder's, or `--name LABEL`) and nothing
            else, so every other key keeps tracking the built-in default.
            With `--ledgers[=NAME]`, also writes HISTORY-NAME-Sessions.csv,
            RULINGS-NAME.csv and CLAIMS-NAME.csv at DIR (header row only,
            NAME defaulting to the store's) and registers the last two as
            sources: rows the write-time check refuses off-schema, and
            rulings and claims ranked beside the markdown. Existing files
            are kept.
  schema [DIR]  write the manifest's JSON Schema to ~/.locket/locket.schema.json
            (`init` does this too), so an editor mapped to that path hints every
            `locket.json` by filename; with DIR, check that directory's manifest
            against it instead: an unknown key with a did-you-mean, a wrong type,
            a malformed `sources` or `ledgers` entry. `locket help install`
            names the editor side.
  migrate [DIR|all]  rename a store's older `memfind.json` manifest to
            `locket.json` in place; every store on the machine when DIR is
            omitted. Until a store is migrated the old name is still read.
  trust DIR  let DIR's manifest act: its trigger rows, ledger schemas and
            sources. Asks a person at a terminal to type yes; an agent cannot
            grant it. Until then a manifest outside the host stores is inert.
            Trust is pinned to DIR's locket.json, reference-sources.json and
            rows file as they stand: after editing any of them, or a pull that
            changes them, the store is inert until you trust it again.
  forget DIR  drop DIR's registry row, which also withdraws its trust (`untrust`
            is the same verb), once a person confirms at a terminal. Its
            manifest stays, inert.

HOOK MODES (PreToolUse entry points, JSON on stdin; not for manual use)
  Claude Code (matcher: tool names below), Codex (the same, in hooks.json) and
  Hermes (`pre_tool_call`, matcher write_file|patch, terminal, search_files)
  hand the same payload; the tool names are mapped on the way in, and a Codex
  patch, native or through the shell, is read as the Write or Edit it amounts
  to. Hermes can hear a BLOCK and nothing else on a
  tool event, so a nudge there refuses the call once, leading with how to
  proceed: the same call repeated passes once state.db shows the nudge was
  delivered, and the third attempt passes regardless.
  deliver     retired; does nothing, so an older Hermes config that still
              registers it on `pre_llm_call` keeps running. Remove the entry.
  hook        Write|Edit. On a `.csv` path a schema governs, the ledger check:
              a row over a cap, outside its enum or off its pattern BLOCKS,
              the one refusal here on a question the tool can decide (§2d).
              On markdown, the 2c novelty nudge and the 2e memfind gate. Both
              subtract text already filed under archive/ first, since 4 archives
              superseded text BEFORE it is rewritten over, so a match there is a
              move rather than a fork. The novelty nudge WARNS in a supervised
              session and blocks only under CLAUDE_AUTONOMOUS; the 2e gate
              blocks, being dischargeable by running one command.
  bashguard   Bash. A heredoc or echo into a schema-governed ledger is read as
              rows and checked as `hook` checks them; a redirect whose rows it
              cannot read warns and names the sweep. Then the same corpora, written through a shell redirect, a
              heredoc or an in-place edit (sed, perl or ruby -i, gawk -i
              inplace), which `hook` never sees because it matches TOOLS
              rather than paths. Blocks an unambiguous write, warns on a
              scripted one and on a memory file handed to any command not
              known to be read-only. A Python heredoc or `-c` program is
              parsed and judged by where its writing calls write, so a
              memory file named only as data passes. Silent for ledgers, holding spaces,
              archive/, the scratchpad and autonomous runs. `MEMWRITE-OK:
              <reason>` in the command declares the exception and passes.
  grepassist  Grep|Bash. Runs memfind on a prose-shaped search pattern and
              injects the ranked neighbours, because 2e says ask the question
              BEFORE the grep and prose alone did not achieve it. Fires only on
              a pattern carrying no regex metacharacter and two words of four
              characters or more, so an identifier or an alternation never
              reaches it. Resolves the corpus from the search path, else from
              the cwd. Silent with the embedder down. Every grep it reads is
              logged, returned or declined and why, with the pattern's first
              80 characters, to ~/.locket/grepassist.csv (mode 0600;
              LOCKET_GREPASSIST_LOG), so a hook gone quiet shows as a rate.

EXAMPLES
  locket corpora                 # what corpora exist, and their sizes
  locket audit all               # forks in every corpus, one after another
  locket audit myapp             # forks across one project's corpus
  locket audit                   # ... same, if cwd is inside that project
  locket budget council          # oversized files across the council store
  locket dupes ~/.claude/memory  # a literal path scopes a run to one member
  locket links council           # dangling [[wikilinks]] and CSV pointers
  locket cites myapp             # do the code citations still point anywhere
  locket cites myapp models.py   # ... only the ones naming models.py

Every mode prints the directory it resolved and whether the write-time hook
reaches it, because a literal path can name a directory the hook never guards.
The module docstring carries the design rationale."""


def manifest_help():
    """The MANIFEST section of `usage()` alone, for `locket help manifest`: a
    slice of the one text, never a second copy of it."""
    text = usage()
    start = text.index("\nMANIFEST\n") + 1
    return text[start:text.index("\nMODES\n", start)].rstrip() + "\n"


def usage():
    """USAGE with the built-in ledger schemas rendered from manifest.LEDGER_SCHEMAS, so the
    help is a view of the constant and never a copy of it."""
    pad = " " * 19
    lines = []
    for glob, sch in manifest.LEDGER_SCHEMAS.items():
        lines.append(f"{pad}  {glob}: {','.join(sch.get('columns', []))}")
        if sch.get("required"):
            lines.append(f"{pad}    required {', '.join(sch['required'])}")
        if sch.get("caps"):
            lines.append(f"{pad}    caps " + ", ".join(f"{c} ≤{n}" for c, n in sch["caps"].items()))
        for c, vals in sch.get("enum", {}).items():
            lines.append(f"{pad}    {c} in " + "|".join(v or "<empty>" for v in vals))
        for c, rx in sch.get("pattern", {}).items():
            lines.append(f"{pad}    {c} matches {rx}")
    return USAGE.replace("@@BUILTIN_LEDGERS@@", "\n".join(lines))


EPILOG = """examples:
  locket find "the fact you are about to write" council
  locket find "the same fact" all           every store, one list per store
  locket index notes                        refresh one cache; `find` builds it on demand too
  locket graduated && locket links && locket across
                                            the session-close trio
  locket init ~/notes                       any folder of markdown becomes a store

The corpus argument takes a shorthand (council, hermes, codex) or registered name, a project slug
substring, a literal path, or `all`; omitted, it is the store the current
directory sits in. `--index`, `--siblings`, `--embedder` and `--selftest` are
accepted as verbs too. `locket help scan` explains what a store is and how a
manifest shapes it, `locket help manifest` prints the manifest keys alone, and
`locket help find` explains how to read a ranking.
Needs nothing but Python 3 for the checks; `find` needs an embedder (ollama with
embeddinggemma-2:270m, or onnxruntime in the venv the Panoply pieces share) and says so when none answers.
"""


CORPUS_NOTE = """
corpus: a store's name as `locket corpora` lists it, or any part of one that
matches a single store; a folder path; or `all`. Omitted, it is the store the
current directory sits in."""
