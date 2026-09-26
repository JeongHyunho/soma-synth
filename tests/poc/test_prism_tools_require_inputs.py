"""The PRISM tools name their inputs explicitly and create nothing before checking them.

Each tool asks for its folder or file (``--run``, ``--npz``, ``--out``; there is no default), says
where the bundles are under ``SOMA_DATA_ROOT``, and checks its inputs before it makes a folder or
writes a file.
A tool that writes refuses a folder inside a bundle (viewer_paths.writable_folder), since the hint
leads there: it works on a scratch copy of the take.
The Blender ones run here with stand-in ``bpy``/``mathutils``/``addon_utils``/``render_amass``
modules: every refusal happens before any of them is used.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "scripts" / "poc"
TOOLS = ("prism_to_smplx.py", "export_anim_data.py", "blender_prism_anim.py", "probe_align.py")
#: The tools that write where their argument points (probe_align only reads its npz).
WRITERS = ("prism_to_smplx.py", "export_anim_data.py", "blender_prism_anim.py")
#: A take folder name of a bundle (``prism-subj001-take002``, ``...-faithful-v1``): no tool may
#: carry one as a default.
TAKE_FOLDER = re.compile(r"prism-subj\d+-take\d+")
#: What a PRISM take of a bundle holds, as far as these tools read it.
TAKE_FILES = ("small_reference.npz", "large_reference.npz", "development_reference.npz",
              "manifest.json", "anim_data.npz")

#: Runs a tool as __main__ with stand-ins for Blender's modules, its folder first on the path as
#: `python <tool>` puts it (the Blender tools insert it themselves).
_DRIVER = """
import os, runpy, sys, types
for name in ("bpy", "mathutils", "addon_utils", "render_amass"):
    module = types.ModuleType(name)
    for attribute in ("Vector", "Matrix", "Quaternion"):
        setattr(module, attribute, lambda *args, **kwargs: None)
    sys.modules[name] = module
script, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
"""


def _run(tool: str, *arguments: str, cwd: Path, data_root: Path | None = None):
    env = {key: value for key, value in os.environ.items()
           if key not in ("SOMA_DATA_ROOT", "SOMA_SOURCE_ROOT", "SOMA_ANIM_DIR")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    if data_root is not None:
        env["SOMA_DATA_ROOT"] = str(data_root)
    return subprocess.run([sys.executable, "-c", _DRIVER, str(POC / tool), *arguments],
                          cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                          check=False, timeout=120)


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


@pytest.mark.parametrize("tool", (*TOOLS, "smpl_recon.py"))
def test_no_tool_hardcodes_a_take_folder(tool):
    assert not TAKE_FOLDER.search((POC / tool).read_text(encoding="utf-8"))


@pytest.mark.parametrize("tool", TOOLS)
def test_nothing_runs_at_import(tool):
    """Module level holds imports, constants and definitions: no argument parsing, no folder made
    (the work is in main(), called under ``if __name__ == "__main__"``)."""
    tree = ast.parse((POC / tool).read_text(encoding="utf-8"))
    calls = [node for statement in tree.body if not isinstance(statement, ast.If)
             for node in ast.walk(statement)
             if isinstance(node, ast.Call) and not isinstance(statement, (ast.FunctionDef,
                                                                           ast.ClassDef))]
    names = {ast.unparse(call.func) for call in calls}
    assert not names & {"os.makedirs", "os.mkdir", "ap.parse_args", "_ap.parse_args",
                        "np.load", "np.savez", "open"}, names
    guard = tree.body[-1]
    assert isinstance(guard, ast.If) and "__main__" in ast.unparse(guard.test)
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)


@pytest.mark.parametrize("tool, arguments, flag", [
    ("prism_to_smplx.py", [], "--out"),
    ("export_anim_data.py", [], "--run"),
    ("blender_prism_anim.py", ["--"], "--run"),
    ("blender_prism_anim.py", ["--", "--smoke"], "--run"),
    ("probe_align.py", ["--"], "--npz"),
    ("smpl_recon.py", ["--verify-recon"], "--run"),
])
def test_a_tool_without_its_input_says_which_and_where_the_bundles_are(tool, arguments, flag,
                                                                       tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    data_root = tmp_path / "data"
    done = _run(tool, *arguments, cwd=work, data_root=data_root)
    assert done.returncode != 0
    assert flag in done.stderr and "there is no default" in done.stderr, done.stderr
    assert str(data_root / "runs" / "experimental_generation_poc_demo") in done.stderr
    if tool in WRITERS:                                   # the hint leads to the bundles: not into one
        assert "never into a bundle" in done.stderr, done.stderr
    assert "Traceback" not in done.stderr
    assert _files(tmp_path) == ["work"]                   # nothing made, not even the data root
    unset = _run(tool, *arguments, cwd=work)
    assert "SOMA_DATA_ROOT is not set" in unset.stderr and unset.returncode != 0


@pytest.mark.parametrize("tool, arguments, missing", [
    ("export_anim_data.py", ["--run", "{run}", "--pkl", "{pkl}"], "small_reference.npz"),
    ("blender_prism_anim.py", ["--", "--run", "{run}", "--smoke"], "anim_data.npz"),
    ("probe_align.py", ["--", "--npz", "{run}/prism_take002_smplx.npz"], "no SMPL-X animation npz"),
    ("prism_to_smplx.py", ["--out", "{run}/new/out.npz", "--pkl", "{pkl}"], "no PRISM take pickle"),
    ("export_anim_data.py", ["--run", "{full}", "--pkl", "{pkl}"], "no PRISM take pickle"),
])
def test_missing_inputs_are_refused_before_anything_is_written(tool, arguments, missing, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    full = tmp_path / "full"                              # holds the npz, not the pickle
    full.mkdir()
    for name in ("small_reference.npz", "development_reference.npz"):
        (full / name).write_bytes(b"")
    before = _files(tmp_path)
    arguments = [a.format(run=run, full=full, pkl=tmp_path / "absent.pkl") for a in arguments]
    done = _run(tool, *arguments, cwd=tmp_path)
    assert done.returncode != 0 and missing in done.stderr, done.stderr
    assert "Traceback" not in done.stderr
    assert _files(tmp_path) == before                     # no figures/, no new/, no anim_data.npz


@pytest.mark.parametrize("tool, arguments", [
    ("export_anim_data.py", ["--run", "{take}", "--pkl", "{pkl}"]),
    ("blender_prism_anim.py", ["--", "--run", "{take}", "--smoke"]),
    ("prism_to_smplx.py", ["--out", "{take}/prism_take002_smplx.npz", "--pkl", "{pkl}"]),
])
def test_a_writing_tool_refuses_a_take_of_a_bundle(tool, arguments, tmp_path):
    """A take of a bundle -- where the hint leads -- is refused before anything is written, though
    it holds every input: only the bundle's generator writes there. The bundle here is outside any
    lineage container, so its INDEX.json alone refuses it. Every input exists (the pickle too, empty),
    so a tool without the refusal would go on and fail reading it."""
    bundle = tmp_path / "elsewhere" / "prism_faithful_full"
    take = bundle / "prism-subj001-take002"
    take.mkdir(parents=True)
    (bundle / "INDEX.json").write_text("{}", encoding="utf-8")
    for name in TAKE_FILES:
        (take / name).write_bytes(b"")
    pkl = tmp_path / "take002.pkl"
    pkl.write_bytes(b"")
    before = _files(tmp_path)
    arguments = [a.format(take=take, pkl=pkl) for a in arguments]
    done = _run(tool, *arguments, cwd=tmp_path)
    assert done.returncode != 0
    assert "refusing to write into" in done.stderr and "INDEX.json" in done.stderr, done.stderr
    assert "scratch folder" in done.stderr, done.stderr
    assert "Traceback" not in done.stderr
    assert _files(tmp_path) == before                     # the bundle is as it was
