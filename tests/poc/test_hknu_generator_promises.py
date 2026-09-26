"""What the HKNU generator writes without a repair pass afterwards (2026-09-14).

Three things the retired hknu_unified8_v1 lacked and had to be backfilled or reported: the
attribution block, the sex provenance string, and the discontinuity exclusions in the INDEX.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    sys.path.insert(0, str(REPO / "src"))
    return _load(REPO / "scripts" / "poc" / "generate_hknu_unified8.py",
                 "generate_hknu_unified8_promises")


def _identity(generator, tmp_path: Path):
    corpus = tmp_path / "runs" / "experimental_generation_poc_demo" / "hknu_smpl24_paired"
    take = corpus / "S01" / "S01_Walking_4KPH_0.npz"
    take.parent.mkdir(parents=True)
    take.write_bytes(b"digest only")
    return generator.identity_for(
        take, "S01", "Walking_4KPH_0", 1000, 100.0,
        {"height_m": 1.75, "mass_kg": 70.0}, root=tmp_path, paired=corpus)


def test_the_default_corpus_carries_no_version_suffix(generator, monkeypatch, tmp_path):
    """The default is resolved under SOMA_DATA_ROOT when asked, not a drive-letter constant."""
    monkeypatch.setenv("SOMA_DATA_ROOT", str(tmp_path))
    assert generator.PAIRED_LINEAGE == "hknu_smpl24_paired"
    assert generator.default_paired() == (
        tmp_path / "runs" / "experimental_generation_poc_demo" / "hknu_smpl24_paired")
    assert generator.default_paired().name == "hknu_smpl24_paired"


def test_the_manifest_carries_the_attribution_the_registry_resolves(generator, tmp_path):
    from soma_synth.contracts import source_attribution

    identity = _identity(generator, tmp_path)
    block = identity.extra_manifest["source_attribution"]
    assert block == source_attribution.resolved_for("hknu")
    for key in source_attribution.load().required_keys:
        assert block[key]


def test_the_sex_provenance_names_the_workbook_column_not_prism(generator, tmp_path):
    identity = _identity(generator, tmp_path)
    assert identity.sex_provenance == "measured:DatasetInfo.xlsx Gender"
    assert "prism" not in identity.sex_provenance


def test_the_discontinuity_scan_is_the_one_the_diagnostics_tool_runs(generator):
    """Same module by path, so the rule the INDEX applies is the rule --apply would apply."""
    assert generator.SCAN.MARKER == "pose_discontinuity_scan"
    assert callable(generator.SCAN.scan_take)
    assert generator.SCAN.GYRO_ABS_MAX_DEG_S == 8000.0
