"""A store's manifest: its file names, the unchecked read, the schema it is validated
against, and the defaults a store without one gets. Trust is not decided here;
trust.py's _read_manifest is the read that acts.
"""
import json
import sys
from pathlib import Path, PurePosixPath

import fsio


# A per-corpus manifest, optional. Its presence makes ANY directory a corpus for
# the write-time gate and for a literal path, so a memory store on a host this
# file has never heard of is covered by writing one file into it; its absence
# means the host defaults below. Every key it may carry is listed once, under
# MANIFEST in USAGE; `settings`, `csv_sources` and `all_corpora` read them, and
# only from a trusted store (`trusted`). It is the only thing this tool owns
# inside a corpus; the index lives in ~/.locket/cache.
MANIFEST = "locket.json"


OLD_MANIFEST = "memfind.json"      # the name before 2026-09-21; read, migrated by `migrate`, never written


LEGACY_MANIFEST = "reference-sources.json"


def manifest_file(root):
    """The manifest path for `root`: `locket.json`, else the older `memfind.json`
    where only that exists, else the new name, for writing. Every read goes
    through here so a store not yet migrated keeps working."""
    root = Path(root)
    new = root / MANIFEST
    if new.is_file():
        return new
    old = root / OLD_MANIFEST
    return old if old.is_file() else new


def has_manifest(root):
    return manifest_file(root).is_file()


def _manifest_data(root):
    """The manifest file's own contents, trusted or not: for `init` editing a
    file it is about to rewrite, never for acting on one."""
    m = manifest_file(root)
    if not m.is_file():
        return {}
    try:
        with fsio.open_safe(m, binary=True) as f:
            data = json.loads(f.read().decode("utf-8"))     # strict: a garbled name is no name
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):           # JSONDecodeError, and a non-UTF-8 file
        return {}


# The manifest's shape, as JSON Schema, in the one place the reader lives. `schema`
# writes it to ~/.locket/locket.schema.json for an editor to read by filename and
# `validate_manifest` checks a manifest against it at `init`, so an agent authoring
# one by hand gets the same hint an editor would. Every key optional; nothing here
# is required for a directory to be a store.
_STR_LIST = {"type": "array", "items": {"type": "string"}}


MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "locket.schema.json",
    "title": "Locket manifest (locket.json)",
    "description": "Per-corpus configuration for Locket (memscan.py and memfind.py). "
                   "Every key is optional and a missing key takes the tool's default; "
                   "`locket --help` MANIFEST is the authority.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "$schema": {"type": "string", "description": "Editor hint only; ignored by the tool."},
        "_comment": {"type": "string", "description": "Free text for the human reader; ignored by the tool."},
        "name": {"type": "string",
                 "description": "What do `corpora` and a store argument call this store? Default: the "
                                "directory's basename. Host stores and project memories ignore it."},
        "members": {**_STR_LIST,
                    "description": "Which top-level directories are corpus? The rest of the root is "
                                   "host state. Root-level markdown is always in. Replaces the default."},
        "excluded_dirs": {**_STR_LIST,
                          "description": "Which directories, at any depth, are not corpus? Replaces the "
                                         "default (archive, in-flight, harness-logs, discovery-research)."},
        "excluded_files": {**_STR_LIST,
                           "description": "Which single files are not corpus? Corpus-relative paths or "
                                          "globs, invisible to every mode, the index and the write-time "
                                          "gate. For a file that holds no claims, such as a draft or a template."},
        "index_files": {**_STR_LIST,
                        "description": "Which files POINT at other files? A line restating its target "
                                       "is not a fork. Surface role, by basename; replaces the default."},
        "ledger_surfaces": {**_STR_LIST,
                            "description": "Which markdown files grow by appending entries? Never scored "
                                           "for forks. Not `ledgers`, which checks CSV row shape. Surface "
                                           "role, by basename; replaces the default."},
        "holding_spaces": {**_STR_LIST,
                           "description": "Which files stage entries on their way to a durable home? "
                                          "Scored only against themselves, yet fully corpus, unlike an "
                                          "excluded file. Surface role, by basename; replaces the default."},
        "whole_read_exempt": {**_STR_LIST,
                              "description": "Which files may outgrow a whole read? The size sweep skips "
                                             "them and nothing else does: a mixed surface never read whole "
                                             "by its own contract, or a file a project instrument already "
                                             "caps. Default: none."},
        "code_roots": {**_STR_LIST,
                       "description": "Where do `cites` path:line citations resolve? Directories, "
                                      "absolute or relative to the store."},
        "sources": {
            "type": "array",
            "description": "Which CSVs hold claims? Their rows are scored for forks like "
                           "any sentence in the corpus.",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["path", "text"],
                      "properties": {
                          "path": {"type": "string", "description": "The CSV, absolute or store-relative."},
                          "text": {"type": "string", "description": "Column holding the claim sentence."},
                          "label": {"type": "string", "description": "Column that names a row in output (an Area)."}}}},
        "ledgers": {
            "type": "array",
            "description": "What shape must a CSV ledger's rows take? A row outside its "
                           "schema is refused at write time; forks are `sources`' business.",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["path"],
                      "properties": {
                          "path": {"type": "string", "description": "Basename glob or store-relative path of the ledger."},
                          "extends": {"type": "string",
                                      "description": "A built-in's glob this entry merges onto though its path falls outside it."},
                          "columns": {**_STR_LIST, "description": "The exact header, in order."},
                          "required": {**_STR_LIST, "description": "Columns that may not be empty."},
                          "caps": {"type": "object", "additionalProperties": {"type": "integer"},
                                   "description": "{column: maximum characters}."},
                          "enum": {"type": "object", "additionalProperties": _STR_LIST,
                                   "description": "{column: allowed values}; include \"\" where empty is allowed."},
                          "pattern": {"type": "object", "additionalProperties": {"type": "string"},
                                      "description": "{column: regex the whole value must match}."}}}},
        "triggers": {"type": "string",
                     "description": "Which rows file does `locket trigger` read for this store? Store-relative "
                                    "or absolute. Default: triggers.json at the store root."},
        "belongs_to": {"type": "string",
                       "description": "Which corpus does this directory JOIN? Its name as `corpora` lists it; "
                                      "written by `init <dir> --parent <name>`."},
    },
}


def _check(value, schema, path, out):
    """Structural check of `value` against the manifest schema's subset of JSON Schema,
    through panoply-lib's checker, a generated copy beside this file."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import _manifest_schema
    _manifest_schema.check_value(value, schema, path, out)


def validate_manifest(root):
    """Findings for the manifest at `root` against MANIFEST_SCHEMA; [] when clean
    or absent. Structure only: a key the reader would silently ignore, a wrong
    type, a malformed source or ledger entry. Whether the values are right is
    the modes' business."""
    m = manifest_file(root)
    if not m.is_file():
        return []
    try:
        data = json.loads(fsio.read_safe(m))
    except (json.JSONDecodeError, OSError) as e:
        return [f"manifest does not parse: {e}"]
    out = []
    _check(data, MANIFEST_SCHEMA, "", out)
    for i, e in enumerate(data.get("ledgers") or [] if isinstance(data, dict) else []):
        x = e.get("extends") if isinstance(e, dict) else None
        if x is not None and x not in LEDGER_SCHEMAS:
            out.append(f"ledgers[{i}].extends: {x!r} is no built-in; one of {', '.join(LEDGER_SCHEMAS)}")
    return out


def _inside(rel):
    """Whether a manifest's relative path stays inside its store: not absolute, no
    `..`. A member `/` walked the whole disk inside the write hook."""
    p = PurePosixPath(str(rel).replace("\\", "/"))
    return bool(str(rel).strip()) and not p.is_absolute() and ".." not in p.parts and not str(rel).startswith("~")


# Files whose job is to POINT at other files. A hook line restating its target is
# doing its job; only a FALSE, decaying or uninformative restatement is a defect.
# SELF.md and SHORT_TERM.md are deliberately absent: they carry no index lines, and
# treating them as indexes mislabels the promotion-flow duplication between them.
INDEX_FILES = {"MEMORY.md", "ARCHIVE.md", "CURATION-INDEX.md", "CURATION-SOURCES-INDEX.md",
               "FRONTEND-INDEX.md", "PLATFORM-INDEX.md"}


# Files that accumulate ENTRIES rather than stating facts. Appending to one is the
# discipline rather than a breach of it, so both sets are exempt from the ordinary
# corpus-wide nudge. The stake is higher than noise: the hook exits 2, and the
# cycles write these on a schedule with no session in front of them, so a wrong
# nudge here is a blocked cycle that cannot report its own failure.
#
# LEDGER: chronology is the whole value, entries repeat by design, nothing is
# checked. HOLDING: a staging area whose entries are promoted or discarded after
# observation, so a DUPLICATE ENTRY is a defect (Peer ruling 2026-08-15) while a
# new one is the file doing its job. A holding space is therefore scored against
# ITSELF and nothing else: restating an item it already holds fires, and turning
# over an observation about a fact that lives in MEMORY.md does not.
LEDGER_SURFACES = {"HARNESS.md", "CHECKPOINTS.md", "EXPLORATIONS.md"}


HOLDING_SPACES = {"SHORT_TERM.md", "OPEN_QUESTIONS.md", "MEMORY-proposed.md",
                  "PROMOTIONS.md"}


# --- Ledger CSVs: schema-checked append surfaces ------------------------------
# A ledger is queried, never loaded whole, so it may grow freely; what it may not
# do is grow a row past the shape a query can scan. The schema is the one thing
# about a ledger that is decidable at write time (§2d: an instrument blocks only
# on a question it can decide), so a row over a cap, outside an enum or off its
# pattern is REFUSED, where every prose check only warns. The built-in schemas are
# the session-history ledger every project keeps and the rulings and claims
# ledgers a store keeps once it has earned them (`RULINGS.csv` in a single store,
# `RULINGS-<Project>.csv` in a project, so one glob covers both). Each is keyed by
# a basename glob and applies wherever the file is, corpus or not, because the
# project root a ledger sits in is usually no store at all. Rulings and claims
# carry no text cap: their rows range past 2,000 characters in live stores, and a
# store that wants one sets it in its manifest. `Category` and `Area` take one
# lowercase key or several joined by `|`, because `Area=X` is how every pointer
# queries them; only a claim carries a `Source`. A manifest's `ledgers` list
# adds a schema for any other CSV in that store, or adjusts a built-in for its
# own file by naming the same glob or a path under it, merging onto it
# (`_merge_schema`). This constant is each schema's one home: `usage()` renders
# it into the help, and the project-dev skill points here rather than restating it. The date takes an optional UTC time, because
# a local-evening session lands on the next UTC day and a bare date cannot say
# which side of midnight it came from. `repo` names the repository the hash
# resolves in (`group/sub/repo` allowed), `issue` is one tracker-agnostic ticket
# key (`#12`, `ENG-42`) and `pr` a GitHub PR or GitLab MR (`#155`, `!155`).
_LEDGER_KEYS = r"[a-z][a-z0-9-]*(?:\|[a-z][a-z0-9-]*)*"


LEDGER_SCHEMAS = {
    "HISTORY-*-Sessions.csv": {
        "columns": ["date", "title", "summary", "area", "repo", "commit_hash", "issue", "pr"],
        "required": ["date", "title", "summary"],
        "caps": {"title": 90, "summary": 240},
        "pattern": {"date": r"20\d\d-\d\d-\d\d(?:T\d\d:\d\d(?::\d\d)?Z)?",
                    "repo": r"(?:[\w.-]+(?:/[\w.-]+)*)?",
                    "commit_hash": r"(?:[0-9a-f]{7,40})?",
                    "issue": r"(?:\S{1,40})?",
                    "pr": r"(?:[#!]\d+)?"},
    },
    "RULINGS*.csv": {
        "columns": ["Ruling", "Category", "Area", "Date"],
        "required": ["Ruling", "Category", "Area", "Date"],
        "pattern": {"Category": _LEDGER_KEYS, "Area": _LEDGER_KEYS, "Date": r"20\d\d-\d\d-\d\d"},
    },
    "CLAIMS*.csv": {
        "columns": ["Claim", "Category", "Area", "Source", "Date"],
        "required": ["Claim", "Category", "Area", "Source", "Date"],
        "pattern": {"Category": _LEDGER_KEYS, "Area": _LEDGER_KEYS, "Date": r"20\d\d-\d\d-\d\d"},
    },
}


_SCHEMA_KEYS = ("columns", "required", "caps", "enum", "pattern")
