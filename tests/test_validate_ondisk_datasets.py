"""Run the validator against the bundles under SOMA_DATA_ROOT (skipped when it is unset or they
are absent): L0/L2/L3/L4 clean, and any L1 FAIL confined to the listed schema-migration keys
(spec_id/spec_version/quality_gate absence + the PRISM drift keys).
"""

import os
from pathlib import Path

import pytest

from soma_synth.validation import checks

_ROOT = os.environ.get("SOMA_DATA_ROOT", "").strip()
#: Under a relative name no folder exists at when the variable is unset, so every test skips.
DATA_ROOT = Path(_ROOT) if _ROOT else Path("<SOMA_DATA_ROOT unset>")
POC = DATA_ROOT / "runs" / "experimental_generation_poc_demo"
AMASS = POC / "amass_faithful_full"      # no version suffix (docs/guides/RETENTION_RULES.md 1.1)
PRISM = POC / "prism_faithful_full"

pytestmark = pytest.mark.skipif(
    not _ROOT or not POC.exists(),
    reason="NEEDS_DATA: SOMA_DATA_ROOT unset, or no poc-demo datasets under it")

# L1 FAILs tolerated for older generations; anything else is a regression.
_ALLOWED_L1 = (
    "spec_id",
    "spec_version",
    "quality_gate",
    "smpl_global_orientation_world",
    "smpl_global_orientation_prism_world",
    "subject_scope_note",
)


def _fails(findings, level):
    return [f for f in findings if f.severity == "FAIL" and f.level == level]


def _sample_take_dirs(ds, n=3):
    return sorted(p for p in ds.iterdir() if p.is_dir())[:n]


def _assert_take_invariants(td: Path) -> None:
    tid = td.name
    assert _fails(checks.check_take_l2(td, tid), "L2") == [], f"{tid}: L2 physical FAIL"
    assert _fails(checks.check_take_l3_axes(td, tid), "L3") == [], f"{tid}: L3 axis FAIL"
    assert _fails(checks.check_take_l4_governance(td, tid), "L4") == [], f"{tid}: L4 governance FAIL"
    for f in _fails(checks.check_take_l1(td, tid), "L1"):
        assert any(tok in f.message for tok in _ALLOWED_L1), f"{tid}: unexpected L1 FAIL: {f.message}"


def test_prism_ondisk_invariants():
    if not PRISM.exists():
        pytest.skip("PRISM dataset absent")
    # L0 clean on the full 150-take dataset (deliverables, orphans, counts)
    assert _fails(checks.check_l0_structure(PRISM), "L0") == []
    for td in _sample_take_dirs(PRISM):
        _assert_take_invariants(td)


def test_amass_ondisk_invariants():
    if not AMASS.exists():
        pytest.skip("AMASS dataset absent")
    for td in _sample_take_dirs(AMASS):
        _assert_take_invariants(td)
