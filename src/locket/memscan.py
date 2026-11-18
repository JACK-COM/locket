#!/usr/bin/env python3
"""memscan: duplication, drift and link checks over a flat memory corpus.

The unit is the CLAIM, not the file. Document-level similarity cannot see one
clause restated inside four files that are otherwise about different subjects,
and it scores sibling records sharing a whole vocabulary as duplicates.

Modes:

  audit    cross-file verbatim claim pairs. These are copies that currently
           AGREE, so they are fork RISK rather than proof of a fork.
  dupes    repeated claims INSIDE one file, which `audit` skips by construction.
           The uniqueness invariant on a holding space (prime-memory-discipline §2).
  budget   §5 as counting: bytes, dated history clauses (each listed by line,
           the reconsolidation worklist), longest prose paragraph. `audit` is
           blind to a file that only disagrees with the world, so nothing else
           measures this.
  pointers every index line paired to the file it points at, for adjudication
           (`hooks` is its older name). Needs a reader: a hook asserting what its topic file now refutes
           scores LOW on similarity and is invisible to `audit`.
  links    wikilink targets that resolve to no file.
  cites    every `path:line` citation resolved against the code it points at.
           The one claim class nothing else here can see, because the claim is
           ABOUT the code rather than IN it and no other mode opens the file.
  corpora  every corpus on the machine, with its file count.
  init     make a directory a corpus every mode can find: manifest, registry
           row, index. `forget` drops the row again.
  against  candidate owners for an incoming statement. Used by the write-time
           hook.

The corpus argument takes a registered name (`notes`), any substring of a project
slug (`myapp`), or a literal path. With no argument it takes THIS project's corpus.

Standard library only at import, by design: this runs inside a PreToolUse hook on
every memory write, so it pays interpreter startup and nothing else. memfind, and
numpy with it, loads only when a check asks for a ranking by meaning.
"""
import sys

import help_text
from commands import hooks
from scaffold import (  # noqa: F401  (re-exported)
    LEDGER_FILES, cmd_init, cmd_schema, migrate, scaffold_ledgers,
)
from bashguard import (  # noqa: F401  (re-exported)
    SCRATCH_MARKERS, TRUST_REFUSAL, bash_write_decision,
    ledger_bash_decision, oversized_decision, shell_store_config,
)
from adapters import emit, gate_status, host_of, normalise, read_payload  # noqa: F401  (re-exported)
from adapters.codex import parse_patch  # noqa: F401  (re-exported)
from adapters.hermes import ONCE_NOTE, refuse_once  # noqa: F401  (re-exported)
import bashguard
import grepassist
from grepassist import grep_assist, grep_words, prose_like  # noqa: F401  (re-exported)
from shellwrites import (  # noqa: F401  (re-exported)
    BASH_WRITE, BASH_WRITE_STARTS, bash_write_targets, write_matches,
)
from cites import (  # noqa: F401  (re-exported)
    CITE, citations, code_index, code_roots, enclosing_scope, project_dir_for,
    resolve_citation,
)

# The layers below this module, re-exported under the names its remaining code, the
# front and ~/.claude/hooks already use. Code inside a layer calls `<module>.<name>`,
# so a test patches a name where it lives (`fixture_env` patches trust's).
from fsio import (  # noqa: F401  (re-exported)
    MAX_FILE_BYTES, MAX_SCAN, MAX_STORE_BYTES, MAX_STORE_FILES, PATH_MAX, SafeOut,
    _inode, _regular, _replace_atomic, _same, open_safe, read_or_empty, read_safe,
    relay, sqlite_rows,
)
from health import (  # noqa: F401  (re-exported)
    HEALTH_MARKER_CAP, HEALTH_STALE_OTHER, HEALTH_STALE_POSIX, HEALTH_TEMP_AGE, HookRun,
    _fold_dead_markers, _marker_name, _utc, health_ack, health_dir, health_error,
    health_report,
)
from manifest import (  # noqa: F401  (re-exported)
    HOLDING_SPACES, INDEX_FILES, LEDGER_SCHEMAS, LEDGER_SURFACES, LEGACY_MANIFEST,
    MANIFEST, MANIFEST_SCHEMA, OLD_MANIFEST, _SCHEMA_KEYS, _inside, _manifest_data,
    has_manifest, manifest_file, validate_manifest,
)
from trust import (  # noqa: F401  (re-exported)
    _TRUST_CACHE, _granted, _read_manifest, _trusted_ids, changed_since_trust,
    config_digest, forget, grant_trust, person_confirms, pin_unpinned, register,
    registry, registry_pins, registry_stale, trusted, untrusted_stores,
    write_store_file,
)
from corpora import (  # noqa: F401  (re-exported)
    _clear_settings, _label, _phys_root_of, _raw_corpora, _root_of, _walked_by,
    all_corpora, corpus_for, corpus_for_cwd, corpus_here, corpus_key, excluded_file,
    parent_of, roots_of, search_root, settings, top_of,
)
from claims import (  # noqa: F401  (re-exported)
    CLAIMS_VERSION, ENTRY_TOKENS, FIRE_FLOOR, IDX_LINE, LINK, MIN_TOKENS, PAIR_FLOOR,
    ClaimIndex, _ENDS, _LINE_MEANINGFUL, _NO_JOIN_FROM, claims, cross_file_pairs, pointer_only,
    score_against, similar, strip, tokens, unclaimed, unwrap,
)
from shellscan import (  # noqa: F401  (re-exported)
    command_text, shell_code as _shell_code, strip_heredocs as _strip_heredocs,
)
from corpus import (  # noqa: F401  (re-exported)
    _archive_md, _own_md, _walk_files, archived_claims, claim_index, corpus, csv_rows, csv_sources,
    has_content, md_files, resolve, skipped_files, subtract_archived,
)
from ledger import (  # noqa: F401  (re-exported)
    LEAD_BLOCKING, LEAD_WARNING, _csv_rows_of, _ledger_header, check_ledger_text,
    CSV_REPORT_BYTES, declared_csvs, large_csvs,
    ledger_decision, ledger_message, ledger_schema_for, ledger_schemas,
    source_row_decision,
)
from gate import (  # noqa: F401  (re-exported)
    CONFIG_REFUSAL, HISTORY_CLAUSE, ISO_DATE, REGISTRY_REFUSAL,
    _runs_memfind, added_text, dated_series_pair, evaluate_write, hook_decision,
    is_registry, memfind_asked, memfind_asked_db, memfind_gate, semantic_neighbours,
    session_asked, session_ledger, store_config,
)


def main(argv):
    """`memscan.py <verb>`: the write-time hooks a host config calls by path, then any
    other verb through the CLI's dispatch."""
    if len(argv) < 2 or argv[1] in ("--help", "-h", "help"):
        print(help_text.usage())
        return 0 if len(argv) > 1 else 1
    if argv[1] in hooks.ENTRIES:
        return hooks.ENTRIES[argv[1]].run(argv[2:])
    import cli
    return cli.dispatch(argv[1], argv[2:])


if __name__ == "__main__":
    # memfind imports memscan; run as a script, this module is __main__, and without the
    # alias that import would load and run a second copy with caches of its own
    sys.modules.setdefault("memscan", sys.modules[__name__])
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:      # piping any mode into `head` is normal use
        sys.stderr.close()
        sys.exit(0)
