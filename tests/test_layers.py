"""Structure the module split depends on, read from the source rather than run."""
import ast
import unittest
from pathlib import Path

from tests.harness import SRC


class Layers(unittest.TestCase):
    def test_no_local_shadows_a_module_it_reads(self):
        """A function that binds a local under an imported module's name and also reads
        `<name>.attr` reads the local: moving code between modules qualifies its calls
        (`citations` becomes `cites.citations`) and silently turns such a local into a
        crash on the first call. The underscore modules are panoply-lib copies."""
        found = []
        for f in sorted(Path(SRC).rglob("*.py")):
            if f.name.startswith("_") and f.name != "__init__.py":
                continue
            tree = ast.parse(f.read_text())
            mods = {(a.asname or a.name).split(".")[0] for n in tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                a = fn.args
                bound = {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
                bound |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                read = {n.value.id for n in ast.walk(fn)
                        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
                for name in sorted(bound & mods & read):
                    found.append(f"{f.relative_to(SRC)}:{fn.lineno} binds {name}")
        self.assertEqual(found, [], "a local shadows a module the same function reads")

    def test_the_hook_path_never_loads_numpy(self):
        """Every hook imports the CLI or memscan before it reads its payload, and numpy
        costs about 60 ms of a hook's budget; a verb family needing the embedder imports
        it on use."""
        import subprocess
        import sys
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import cli, memscan; "
                "print(sorted(m for m in ('numpy', '_embed', 'memfind') if m in sys.modules))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "[]"), r.stderr[-400:])


if __name__ == "__main__":
    unittest.main()
