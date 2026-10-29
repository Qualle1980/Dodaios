"""No two test modules may share a basename, or pytest cannot collect them.

The test tree has no ``__init__.py`` packages, so pytest imports each module
under its bare basename. Two files called ``test_http_pool.py`` therefore claim
the same module name, and collecting both in one run fails outright:

    ERROR tests/unit/plugins/tool/test_http_pool.py
    import file mismatch … HINT: use a unique basename for your test file modules

That is not a warning. The second file is never run, and a full-suite
invocation ends in a collection error rather than a result — reproduced on
``main`` on 2026-10-28 with ``tests/unit/conductor/test_http_pool.py`` and
``tests/unit/plugins/tool/test_http_pool.py``.

The fix is a distinct name; this guard is what stops the next duplicate from
being merged, because the failure only shows when both files land in the same
invocation and a lane that happens to split them stays green.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_TESTS = _REPO / "tests"


def test_no_two_test_modules_share_a_basename() -> None:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in _TESTS.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        # A directory that IS a package imports as a dotted path, so its
        # modules cannot collide with another package's.
        if (path.parent / "__init__.py").is_file():
            continue
        by_name[path.name].append(path)

    clashes = {
        name: sorted(str(p.relative_to(_REPO)) for p in paths)
        for name, paths in by_name.items()
        if len(paths) > 1
    }
    assert not clashes, (
        "these test modules share a basename and cannot be collected in one "
        f"pytest run — give each a distinct name: {clashes}"
    )
