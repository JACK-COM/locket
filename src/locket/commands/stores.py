"""Every store on this machine.
"""


import corpora
import corpus
import manifest
import trust


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
