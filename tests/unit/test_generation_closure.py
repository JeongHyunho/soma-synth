"""The generation pipeline stands on its own: nothing it runs imports the parent project.

soma-synth was split out of the parent project (``soma_synthetic_imu``) with the package renamed
to ``soma_synth``. The dependency runs one way -- parent -> soma-synth -> smpl18 -- so no entry
point or library module here may load a ``soma_synthetic_imu`` module, and the GAITEX generator
reads GAITEX through this package's own copy of the native reader.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "scripts" / "poc"
PARENT_PACKAGE = "soma_synthetic_imu"

# Every entry point and helper the generators run or load by path, and the library modules the
# runner, the validator, the README renderer and the catalog use.
SCRIPTS = sorted(p.relative_to(REPO).as_posix() for p in POC.glob("generate_*.py")) + [
    "scripts/poc/anthro_smpl.py", "scripts/poc/smpl_recon.py", "scripts/poc/unified8_emit.py",
    "scripts/poc/synthesize_insole_heading.py", "scripts/diagnostics/pose_discontinuity_scan.py",
    "scripts/diagnostics/hknu_retarget_probe.py", "scripts/write_known_limitations.py"]
MODULES = [
    "soma_synth.cli",
    "soma_synth.pipeline.runner",
    "soma_synth.pipeline.generate",
    "soma_synth.pipeline.paths",
    "soma_synth.validation.runner",
    "soma_synth.docs_gen.readme",
    "soma_synth.datasets.catalog",
    "soma_synth.contracts.qmd_unified8_smpl18_spec",
    "soma_synth.gaitex_retarget.native",
]


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("SOMA_DATA_ROOT", "SOMA_SOURCE_ROOT", "SOMA_BODY_MODEL_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "packages" / "smpl18" / "src"), str(POC), str(REPO)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


_PROBE = (
    "import importlib, importlib.util, json, sys\n"
    "kind, target = sys.argv[1], sys.argv[2]\n"
    "if kind == 'script':\n"
    "    spec = importlib.util.spec_from_file_location('closure_probe', target)\n"
    "    module = importlib.util.module_from_spec(spec)\n"
    "    sys.modules[spec.name] = module\n"
    "    spec.loader.exec_module(module)\n"
    "else:\n"
    "    importlib.import_module(target)\n"
    "import soma_synth\n"
    "print(json.dumps({'parent': sorted(m for m in sys.modules if m.split('.')[0] == sys.argv[3]),\n"
    "                  'soma_synth': soma_synth.__file__}))\n"
)


@pytest.mark.parametrize("target", [("script", name) for name in SCRIPTS] + [("module", m) for m in MODULES],
                         ids=lambda t: t[1])
def test_the_generation_closure_loads_nothing_from_the_parent_project(target):
    """A clean interpreter imports each generation entry point or module and lists what it pulled
    in: no ``soma_synthetic_imu`` module may be among them, and ``soma_synth`` is this checkout's."""
    kind, name = target
    path = str(REPO / name) if kind == "script" else name
    result = subprocess.run([sys.executable, "-c", _PROBE, kind, path, PARENT_PACKAGE], cwd=REPO,
                            env=_clean_env(), capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded["parent"] == []
    assert Path(loaded["soma_synth"]).resolve().is_relative_to((REPO / "src").resolve())


def test_no_source_file_imports_the_parent_package():
    """The static side of the same rule, over every Python file of the package and the scripts."""
    offenders = []
    for path in sorted([*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            if any(n == PARENT_PACKAGE or n.startswith(PARENT_PACKAGE + ".") for n in names):
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{node.lineno}")
    assert offenders == []


def test_the_gaitex_generator_reads_through_the_copy():
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('gaitex_unified8_closure', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        "print(module.gaitex_adapter.__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(POC / "generate_gaitex_unified8.py")], cwd=REPO,
                            env=_clean_env(), capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "soma_synth.gaitex_retarget.native"
