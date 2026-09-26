"""Every process pool in the pipeline starts its workers with ``spawn``, on every platform.

Windows can only spawn; Linux forks by default (macOS spawns since Python 3.8). A forked worker
inherits the parent's imported modules, open files and ``sys.path`` edits (the generators load
each other by path and register themselves in ``sys.modules``), so the same generator could
behave differently on Linux than on Windows. The generators and
the pipeline therefore ask for ``multiprocessing.get_context("spawn")`` explicitly, and this scan
of ``src/`` and ``scripts/`` refuses any pool, process or executor created without it.

The runner's own fan-out (``pipeline.generate.run_generate``) starts generator *processes* with
``subprocess``, which neither forks the interpreter's state nor goes through multiprocessing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCANNED = ("src", "scripts")

#: Constructors that start worker processes: concurrent.futures' executor (takes ``mp_context``)
#: and multiprocessing's Pool / Process / Manager, which must be reached through a spawn context
#: (``ctx.Pool(...)``), never through the module's default context.
EXECUTORS = {"ProcessPoolExecutor"}
CONTEXT_METHODS = {"Pool", "Process", "Manager"}


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_spawn_call(node: ast.AST) -> bool:
    """``get_context("spawn")`` under any spelling (``multiprocessing.get_context``, ``mp.get_...``)."""
    return (isinstance(node, ast.Call) and _name(node.func) == "get_context"
            and bool(node.args) and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "spawn")


def _spawn_names(tree: ast.AST) -> set[str]:
    """Names bound anywhere in the module to ``get_context("spawn")``."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_spawn_call(node.value):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
              and _is_spawn_call(node.value) and isinstance(node.target, ast.Name)):
            names.add(node.target.id)
    return names


def _is_spawn_context(node: ast.AST, spawn_names: set[str]) -> bool:
    return _is_spawn_call(node) or (isinstance(node, ast.Name) and node.id in spawn_names)


def pools_in(source: str, filename: str = "<source>") -> list[tuple[int, str, bool]]:
    """(line, what, spawned) for every process pool, process or executor ``source`` creates."""
    tree = ast.parse(source, filename=filename)
    spawn = _spawn_names(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _name(node.func)
        if called in EXECUTORS:
            context = next((k.value for k in node.keywords if k.arg == "mp_context"), None)
            found.append((node.lineno, called,
                          context is not None and _is_spawn_context(context, spawn)))
        elif called in CONTEXT_METHODS:
            owner = node.func.value if isinstance(node.func, ast.Attribute) else None
            if owner is None and called != "Pool" and called != "Process":
                continue                  # a bare Manager() is someone else's class
            if owner is not None and _name(owner) not in ("multiprocessing", "mp", "pool") \
                    and not _is_spawn_context(owner, spawn):
                continue                  # e.g. SomeClass.Process: not multiprocessing's
            found.append((node.lineno, called,
                          owner is not None and _is_spawn_context(owner, spawn)))
    return found


def _python_files() -> list[Path]:
    return sorted(path for folder in SCANNED for path in (REPO / folder).rglob("*.py")
                  if "__pycache__" not in path.parts)


def test_every_pool_in_src_and_scripts_is_spawned():
    unspawned, total = [], 0
    for path in _python_files():
        source = path.read_text(encoding="utf-8-sig")
        if not any(word in source for word in (*EXECUTORS, *CONTEXT_METHODS)):
            continue
        for line, what, spawned in pools_in(source, str(path)):
            total += 1
            if not spawned:
                unspawned.append(f"{path.relative_to(REPO).as_posix()}:{line} {what}")
    assert not unspawned, (
        "process pools created without multiprocessing.get_context('spawn') (Linux would fork "
        "them): " + ", ".join(unspawned))
    # not vacuous: the two AddBio generators' --jobs pools are found and pass
    assert total >= 2


def test_the_addbio_generators_pools_are_the_ones_found():
    for relative in ("scripts/poc/generate_addbio_unified8.py",
                     "scripts/poc/generate_addbio_faithful.py"):
        found = pools_in((REPO / relative).read_text(encoding="utf-8"), relative)
        assert [(what, spawned) for _, what, spawned in found] == [("ProcessPoolExecutor", True)]


@pytest.mark.parametrize("source, spawned", [
    ("ProcessPoolExecutor(max_workers=2)", False),
    ("ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('fork'))", False),
    ("ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'))", True),
    ("ctx = mp.get_context('spawn')\nProcessPoolExecutor(2, mp_context=ctx)", True),
    ("ctx = mp.get_context('fork')\nProcessPoolExecutor(2, mp_context=ctx)", False),
    ("concurrent.futures.ProcessPoolExecutor()", False),
    ("multiprocessing.Pool(4)", False),
    ("from multiprocessing import Pool\nPool(4)", False),
    ("multiprocessing.Process(target=f)", False),
    ("multiprocessing.get_context('spawn').Pool(4)", True),
    ("ctx = multiprocessing.get_context('spawn')\nctx.Pool(4)\nctx.Process(target=f)", True),
])
def test_the_scan_tells_a_spawned_pool_from_a_forked_one(source, spawned):
    found = pools_in(source)
    assert found, source
    assert all(flag is spawned for _, _, flag in found), (source, found)


def test_the_scan_ignores_what_is_not_a_process_pool():
    assert pools_in("ThreadPoolExecutor(4)\nsubprocess.Popen(['x'])\nself.Manager()") == []
