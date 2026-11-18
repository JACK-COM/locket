"""Which file already holds an idea, by meaning: memfind's verbs. memfind, and numpy with
it, loads only when one of these runs.
"""
import sys

import corpora
import corpus
import fsio
from commands import STORE_ARG, Tool, Verb, store_root


def _memfind():
    import install
    return install.load_memfind()


def _script(flag):
    """`run` handing the arguments to memfind under its own spelling of the verb."""
    return lambda rest: _memfind().main(["memfind.py", flag, *rest])


def _find(rest):
    if not rest or rest[0].startswith("-"):
        print('find needs a statement:  locket find "<statement>" [corpus] [-n N]', file=sys.stderr)
        return 1
    return _memfind().main(["memfind.py", *rest])


def _find_tool(args, session):
    statement = str(args.get("statement") or "").strip()
    if not statement:
        raise ValueError("statement is required")
    memfind = _memfind()
    top = int(args.get("top") or 5)
    name = args.get("store")
    roots = ([v for k, v in sorted(corpora.all_corpora().items()) if corpus.md_files(v)]
             if name == "all" else [store_root(name)])
    out = []
    for root in roots:
        rows = memfind.rank(statement, root, top=top, quiet=True)
        out.append(f"store: {root}")
        out.extend(f"[{c:.3f}] {fsio.relay(f, 160)}\n        {fsio.relay(s, 190)}" for c, f, s in rows)
        if memfind.LAST_RANK_MODE == "lexical":
            out.append("WORD OVERLAP ONLY: no embedder could be reached, so this cannot see a "
                       "paraphrase and a silent result is not evidence the store is silent.")
    session.asked = True
    out.append("Ranked candidates, not a verdict. Read the top file before writing; "
               "this tool cannot tell you a fact is new.")
    return "\n".join(out)


def _siblings_tool(args, session):
    rows = _memfind().siblings(store_root(args.get("store")), top=int(args.get("top") or 20), quiet=True)
    out = [f"[subject {c:.3f} | max claim {m:.3f}] {fa}  ::  {fb}" for c, m, fa, fb in rows]
    out.append("High subject with a LOW max claim is one subject written twice, a merge "
               "candidate; low subject with a max near 1.00 is shared boilerplate. Neither "
               "is a verdict, and contradictions are not detected: this is the shortlist a "
               "reader adjudicates.")
    return "\n".join(out)


SEE = "find"
VERBS = (
    Verb("find", '"<statement>" [corpus] [-n N]', "files most likely to own the statement, by meaning",
         'locket find "a fact to write"          the 5 closest files in this directory\'s store\n'
         'locket find "a fact" council -n 10     the 10 closest in the store `council`\n'
         'locket find "the same fact" all        every store, one list per store',
         run=_find, guide=lambda own: _memfind().main(["memfind.py", "--help"]),
         mcp=Tool("Which memory files already hold this idea, ranked by meaning. Call it "
                  "BEFORE writing a durable fact and read the top file; a paraphrase shares "
                  "no words with the claim it restates, so a grep finding nothing proves nothing.",
                  _find_tool,
                  {"statement": {"type": "string", "description": "The fact, in one sentence."},
                   "store": STORE_ARG,
                   "top": {"type": "integer", "description": "Rows to return (default 5)."}})),
    Verb("siblings", "[corpus] [-n N]", "file pairs sharing one subject: candidates to merge",
         "locket siblings                       the 30 closest pairs in this directory's store\n"
         "locket siblings notes -n 10           the 10 closest in `notes`",
         run=_script("--siblings"),
         mcp=Tool("File pairs sharing one subject: the shortlist for merges "
                  "and for a reader looking for contradictions.", _siblings_tool,
                  {"store": STORE_ARG,
                   "top": {"type": "integer", "description": "Pairs to return (default 20)."}})),
    Verb("index", "[corpus]", "build or refresh the semantic cache", "locket index council",
         run=_script("--index")),
    Verb("embedder", "", "which embedder answers on this machine", None, run=_script("--embedder")),
)
