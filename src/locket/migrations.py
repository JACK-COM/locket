"""Finishing an upgrade: ordered steps, each a no-op once done, so `locket migrate` is safe
to run any time. A release that moves something adds one step at the end of `MACHINE`
(run once per machine) or `STORE` (run on each store, or the one store named).
"""
import shlex
import shutil

import corpora
import corpus
import manifest
import paths
import scaffold
import trust


def settings():
    """0.12: ~/.locket/config.json's settings move into the shared settings file."""
    import configure
    try:
        moved = configure.migrate()
    except (OSError, TimeoutError) as e:
        moved = f"warning: {configure._legacy_path()} not moved ({e}); Locket still reads it"
    if moved:
        print(moved)


def old_caches():
    """0.12: the index lives in ~/.locket/cache, so a cache inside a store is dead weight;
    every one is offered for deletion at once, and named in one command otherwise."""
    import install
    stale = [c for c in install.caches() if c.is_dir() and not c.is_symlink()]
    if not stale:
        return
    print(f"index: {len(stale)} caches from before 0.12 are no longer read; the index now lives in "
          f"{paths.PATHS.cache}")
    if trust.person_confirms(f"Delete these {len(stale)} old caches?", [str(c) for c in stale], "delete them"):
        for c in stale:
            shutil.rmtree(c, ignore_errors=True)
        print(f"index: deleted {len(stale)} old caches")
    else:
        print("index: kept them; to delete them all:  rm -r " + " ".join(shlex.quote(str(c)) for c in stale))


def unbuilt_index():
    """0.12: a store indexed only in its old home has no index in the new one yet."""
    import install
    unbuilt = [n for n, r in corpora.all_corpora().items()
               if corpus.has_content(r, csv=False) and not install.load_memfind().cache_paths(r)[0].is_file()]
    if unbuilt:
        print(f"index: {len(unbuilt)} stores have no index in the new home yet; build them once:  locket index all")


def unpinned_trust():
    """0.12: trust is pinned to a store's configuration; pin the rows that predate it."""
    pinned = trust.pin_unpinned()
    if pinned is None:
        print("trust: stores registered before 0.12 are unpinned and act as no store until a person "
              "runs  locket migrate  at a terminal")
    for d in pinned or []:
        print(f"trust: pinned {d} to its configuration as it stands; a later change to its "
              f"locket.json or rows file needs  locket trust {shlex.quote(str(d))}  again")


def untrusted_manifests():
    """0.12: a manifest acts only once a person trusts it; name the ones that stopped."""
    try:
        loose = trust.untrusted_stores()
    except Exception as e:
        loose = []
        print(f"trust: the scan for untrusted stores failed ({type(e).__name__}: {e}); run  locket doctor")
    for d in loose:
        print(f"trust: {d} holds a {manifest.MANIFEST} that is no longer acted on until a person "
              f"trusts it:  locket trust {shlex.quote(str(d))}")


MACHINE = (settings, old_caches, unbuilt_index, unpinned_trust, untrusted_manifests)

# Each takes a store's root and returns "none" (never needed), "already" (done before)
# or what it did, which is printed.
STORE = (scaffold.migrate,)                 # the manifest's older name, memfind.json, becomes locket.json


def run(target=None):
    """Every step for this machine and each store, or `STORE` alone on the one store named."""
    one = target is not None
    if not one:
        for step in MACHINE:
            step()
    roots = [corpus.resolve(target)] if one else list(corpora._raw_corpora().values())   # joined stores too
    seen, n = set(), 0
    for r in roots:
        if r is None or r in seen:
            continue
        seen.add(r)
        for step in STORE:
            what = step(r)
            if what not in ("none", "already") or one:   # done or never needed is not news on a sweep
                print(f"{r}: {what}"); n += 1
    if not n:
        print("migrate: no store needs a change")
    return 0
