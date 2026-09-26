"""The scripts that run inside Blender's own Python import nothing it lacks.

Blender ships its own Python. The viewer bundle's has numpy, and scipy and matplotlib installed into
it (docs/guides/small_large_viewer.md), and nothing else: no PyYAML, no protobuf, no soma_synth, no
smpl18. The scene builder, the viewer app and the Blender renderers must therefore import only the
standard library, numpy, Blender's modules and their sibling scripts -- and so must every sibling
they reach. ``smpl_recon`` once reached ``generate_prism_faithful`` at import, which imports
``soma_synth.pipeline`` (PyYAML) and ``smpl18.model`` (protobuf): the repository's viewer then
failed to import in the bundle's Blender. Checked here with the AST, without Blender.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "scripts" / "poc"

#: What runs through blender.exe: viewer_launch.py builds with viewer_scene.py and opens with
#: viewer_open.py; deploy_viewer.ps1 imports viewer_app.py (the bundle's app); the PRISM renderers
#: and the alignment probe are run with ``blender --python``.
BLENDER_ENTRIES = ("viewer_scene.py", "viewer_open.py", "viewer_app.py", "render_prism_mesh_imu.py",
                   "blender_prism_anim.py", "probe_align.py")
BLENDER_MODULES = frozenset({"bpy", "mathutils", "addon_utils"})
#: The presentation code the viewer bundle carries beside the scripts (viewer_paths.anim_dir): a
#: sibling at run time, not in this repository.
EXTERNAL_SIBLINGS = frozenset({"render_amass", "gear_kit"})
#: Installed into the viewer bundle's Blender Python besides numpy (small_large_viewer.md), used by
#: siblings the entries reach (smpl_recon's rotations, imu_plot's plots); never by an entry itself.
VIEWER_PYTHON_EXTRAS = frozenset({"scipy", "matplotlib"})
#: Imports inside a function that runs only in the teammate's Python, never inside Blender.
OUTSIDE_BLENDER = {
    ("smpl_recon.py", "_generator"): "the reconstruction check and ENRICH, run by viewer_launch.py "
                                     "with the teammate's Python",
}
NEVER = ("soma_synth", "smpl18", "yaml", "google")
STDLIB = frozenset(sys.stdlib_module_names) | {"__future__"}


def _imports(path: Path) -> list[tuple[str, int, str | None]]:
    """(top-level module, line, enclosing function or None) of every import in ``path``."""
    found: list[tuple[str, int, str | None]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno,
                              self.functions[0] if self.functions else None))

        def visit_ImportFrom(self, node):
            name = "." if node.level else (node.module or "").split(".")[0]
            found.append((name, node.lineno, self.functions[0] if self.functions else None))

    Visitor().visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def violations(poc: Path, entries, *, outside=OUTSIDE_BLENDER) -> tuple[list[str], set[str]]:
    """What the Blender-side closure of ``entries`` imports that Blender's Python lacks, and the
    closure itself. Entries may import the standard library, numpy, Blender's modules and siblings;
    a sibling they reach may also use the viewer Python's extras; nothing may reach ``NEVER``."""
    siblings = {p.stem for p in poc.glob("*.py")}
    problems: list[str] = []
    seen: set[str] = set()
    queue = [(name, True) for name in entries]
    while queue:
        name, entry = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        allowed = STDLIB | {"numpy"} | BLENDER_MODULES | EXTERNAL_SIBLINGS | siblings
        if not entry:
            allowed |= VIEWER_PYTHON_EXTRAS
        for module, line, function in _imports(poc / name):
            if function is not None and (name, function) in outside:
                continue
            where = f"{name}:{line} imports {module}"
            if module in NEVER or module == ".":
                problems.append(f"{where}, which Blender's Python lacks")
            elif module not in allowed:
                problems.append(f"{where}, outside the standard library, numpy, Blender's modules"
                                f"{'' if entry else ', ' + ', '.join(sorted(VIEWER_PYTHON_EXTRAS))}"
                                " and the sibling scripts")
            elif module in siblings:
                queue.append((f"{module}.py", False))
    return problems, seen


def test_the_blender_side_scripts_import_only_what_blenders_python_has():
    problems, closure = violations(POC, BLENDER_ENTRIES)
    assert problems == []
    # the walk is not vacuous: it reaches the siblings the viewer app and the scene builder use
    assert {"viewer_paths.py", "smpl_recon.py", "smpl_rig.py", "smpl_model.py", "imu_plot.py",
            "viewer_scene.py", "render_prism_mesh_imu.py"} <= closure
    assert "generate_prism_faithful.py" not in closure and "anthro_smpl.py" not in closure


def test_what_runs_through_blender_is_in_the_list():
    """The scripts viewer_launch.py hands to blender.exe and the module deploy_viewer.ps1 imports
    with it are the entries checked above."""
    launch = (POC / "viewer_launch.py").read_text(encoding="utf-8")
    through_blender = set()
    for line in launch.splitlines():
        if "blender" in line.lower() and "os.path.join(POC" in line:
            through_blender |= set(re.findall(r'os\.path\.join\(POC, "(\w+\.py)"\)', line))
    deploy = (POC / "deploy_viewer.ps1").read_text(encoding="utf-8")
    expression = re.search(r'\$expr = "([^"]+)"', deploy)
    assert expression, "deploy_viewer.ps1 no longer checks the bundle with Blender"
    through_blender |= {f"{name}.py" for name in re.findall(r"import (\w+)", expression.group(1))
                        if (POC / f"{name}.py").is_file()}
    assert through_blender >= {"viewer_scene.py", "viewer_open.py", "viewer_app.py"}
    assert through_blender <= set(BLENDER_ENTRIES)


def test_the_check_catches_a_sibling_that_reaches_soma_synth(tmp_path):
    """A mutant tree as smpl_recon was: the entry is clean, a sibling it imports pulls the
    generator in at import, and the generator imports soma_synth and smpl18."""
    (tmp_path / "entry.py").write_text("import bpy\nimport numpy\nimport recon\n", encoding="utf-8")
    (tmp_path / "recon.py").write_text("import scipy\nimport generator\n", encoding="utf-8")
    (tmp_path / "generator.py").write_text(
        "import os\nfrom soma_synth.pipeline import paths\nfrom smpl18.model import select\n",
        encoding="utf-8")
    problems, _ = violations(tmp_path, ["entry.py"])
    assert any("generator.py" in p and "soma_synth" in p for p in problems)
    assert any("smpl18" in p for p in problems)
    # the same import inside a function kept out of Blender is not followed
    (tmp_path / "recon.py").write_text("import scipy\n\ndef _load():\n    import generator\n",
                                       encoding="utf-8")
    problems, closure = violations(tmp_path, ["entry.py"], outside={("recon.py", "_load"): "x"})
    assert problems == [] and "generator.py" not in closure
    # an entry may not use the viewer Python's extras itself, and a lazy import still counts
    (tmp_path / "entry.py").write_text("import bpy\n\ndef run():\n    import scipy\n",
                                       encoding="utf-8")
    problems, _ = violations(tmp_path, ["entry.py"])
    assert problems and "scipy" in problems[0]


@pytest.mark.parametrize("name", BLENDER_ENTRIES)
def test_each_entry_exists(name):
    assert (POC / name).is_file()
