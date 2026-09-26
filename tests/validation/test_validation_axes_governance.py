"""Validator L3 (spec_S_v2 axis convention, cross-source) + L4 (governance / layout §6 guards).

L3 checks that orientation is a rigid constant relabel of the SMPL global rotation at each site's joint
(the defining spec_S_v2 property): synthetic must be exact (FAIL if dispersion > 1 deg), measured tolerates
mount noise (WARN only above 75 deg). Thresholds calibrated 2026-08-18 (AMASS 0.000 deg; PRISM feet ~30).
"""

import json
from pathlib import Path

import numpy as np

from soma_synth.contracts import qmd_unified8_smpl18_spec as spec
from soma_synth.validation import checks
from soma_synth.validation.runner import validate

# full conformant builders (identity orientations -> constant relabel -> L3-clean, L4-conformant manifest)
from test_validation_checks import make_dataset, make_take


# ------------------------------------------------------------------ L3 fixtures
def _rotz_quat(ang: np.ndarray) -> np.ndarray:
    h = ang / 2.0
    z = np.zeros_like(ang)
    return np.stack([np.cos(h), z, z, np.sin(h)], axis=-1).astype(np.float32)


def _axis_arrays(T: int = 40, break_site0: bool = False) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(T, dtype=np.float64)
    gR = np.tile(np.array([1, 0, 0, 0], np.float32), (T, 24, 1))
    ori = np.zeros((T, 8, 4), np.float32)
    for i, site in enumerate(spec.SITE_ORDER):
        j = spec.SITE_TO_JOINT[site]
        q = _rotz_quat(np.deg2rad(0.5 * t + 5.0 * i))  # distinct, time-varying
        gR[:, j] = q
        ori[:, i] = q  # constant (identity) relabel
    if break_site0:
        ang0 = np.deg2rad(1.0 * t)  # spans 0..~40 deg
        gR[:, spec.SITE_TO_JOINT["back_T4"]] = _rotz_quat(ang0)
        ori[:, 0] = _rotz_quat(2.0 * ang0)  # relabel varies with t
    return gR, ori


def _axis_take(root: Path, tid: str, small_mode: str = "synthetic_from_smpl", break_site0: bool = False) -> Path:
    td = root / tid
    td.mkdir(parents=True)
    gR, ori = _axis_arrays(break_site0=break_site0)
    np.savez(td / "small_reference.npz", imu_orientation=ori, small_mode=np.array(small_mode))
    np.savez(td / "large_reference.npz", smpl_global_orientation_world=gR)
    (td / "manifest.json").write_text(json.dumps({"source": {"source_name": "amass"}}), encoding="utf-8")
    return td


# ------------------------------------------------------------------ L3 tests
def test_l3_constant_relabel_passes(tmp_path: Path) -> None:
    td = _axis_take(tmp_path / "ds", "t0")
    findings = checks.check_take_l3_axes(td, "t0")
    assert [f for f in findings if f.severity == "FAIL"] == []
    assert [f for f in findings if f.severity == "WARN"] == []


def test_l3_synthetic_broken_relabel_fails(tmp_path: Path) -> None:
    td = _axis_take(tmp_path / "ds", "t0", small_mode="synthetic_from_smpl", break_site0=True)
    findings = checks.check_take_l3_axes(td, "t0")
    assert any(f.severity == "FAIL" and "back_T4" in f.message for f in findings)


def test_l3_measured_broken_relabel_does_not_fail(tmp_path: Path) -> None:
    td = _axis_take(tmp_path / "ds", "t0", small_mode="measured_physical", break_site0=True)
    findings = checks.check_take_l3_axes(td, "t0")
    # measured tolerates mount noise -> no FAIL (WARN only above 75 deg)
    assert [f for f in findings if f.severity == "FAIL"] == []


def test_l3_marker_mode_broken_relabel_does_not_fail(tmp_path: Path) -> None:
    """ADR-0039: a marker-fitted body is not a constant relabel of the SMPL joint, so the marker mode is
    judged like a measured sensor -- WARN above the wide band, never FAIL at the synthetic band."""
    td = _axis_take(tmp_path / "ds", "t0", small_mode="synthetic_from_markers", break_site0=True)
    findings = checks.check_take_l3_axes(td, "t0")
    assert [f for f in findings if f.severity == "FAIL"] == []
    # the broken site spans 0..~40 deg of drift, well under the 75 deg WARN band
    assert [f for f in findings if f.severity == "WARN"] == []


def test_l3_marker_mode_names_the_mode_when_it_warns(tmp_path: Path) -> None:
    td = _axis_take(tmp_path / "ds", "t0", small_mode="synthetic_from_markers", break_site0=True)
    findings = checks.check_take_l3_axes(td, "t0", bands=checks.AxisBands(measured_relabel_warn_deg=5.0))
    warned = [f for f in findings if f.severity == "WARN" and "back_T4" in f.message]
    assert warned and "synthetic_from_markers" in warned[0].message


def test_l3_rows_the_mask_calls_invalid_are_left_out(tmp_path: Path) -> None:
    """NaN rows under a False mask must neither crash the SVD nor count as dispersion (ADR-0039)."""
    td = _axis_take(tmp_path / "ds", "t0", small_mode="synthetic_from_smpl")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    T = z["imu_orientation"].shape[0]
    mask = np.ones((T, 8), bool)
    mask[5:15, 0] = False
    z["imu_orientation"][5:15, 0, :] = np.nan
    z["imu_valid_mask"] = mask
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l3_axes(td, "t0")
    assert [f for f in findings if f.severity in ("FAIL", "WARN")] == []


def test_l3_nan_under_a_true_mask_is_a_finding_and_not_a_crash(tmp_path: Path) -> None:
    """A NaN or zero-norm quaternion the mask calls valid used to reach the SVD,
    whose LinAlgError aborted the whole run; it is L2's FAIL, and L3 must fail the take, not the run."""
    td = _axis_take(tmp_path / "ds", "t0", small_mode="synthetic_from_markers")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_orientation"][1:, 0, :] = np.nan          # mask absent = all True; one finite frame remains
    z["imu_orientation"][:, 1, :] = 0.0              # zero-norm quaternions on site 1
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l3_axes(td, "t0")
    assert any(f.severity == "FAIL" and "back_T4" in f.message and "fewer than two" in f.message for f in findings)
    assert any(f.severity == "FAIL" and "wrist_l" in f.message for f in findings)
    assert not any(f.severity in ("FAIL", "WARN") and "shank_l" in f.message for f in findings)


def test_l3_reads_drifted_prism_world_key(tmp_path: Path) -> None:
    td = _axis_take(tmp_path / "ds", "t0")
    z = dict(np.load(td / "large_reference.npz", allow_pickle=True))
    z["smpl_global_orientation_prism_world"] = z.pop("smpl_global_orientation_world")
    np.savez(td / "large_reference.npz", **z)
    findings = checks.check_take_l3_axes(td, "t0")  # must still run via the drifted key
    assert [f for f in findings if f.severity == "FAIL"] == []


# ------------------------------------------------------------------ L4 per-take
def test_l4_clean_manifest_passes(tmp_path: Path) -> None:
    td = make_take(tmp_path / "ds", "t0")
    findings = checks.check_take_l4_governance(td, "t0")
    assert [f for f in findings if f.severity == "FAIL"] == []


def _mutate_manifest(td: Path, mutate) -> None:
    m = json.loads((td / "manifest.json").read_text(encoding="utf-8"))
    mutate(m)
    (td / "manifest.json").write_text(json.dumps(m), encoding="utf-8")


def test_l4_wrong_distribution_scope_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "ds", "t0")
    _mutate_manifest(td, lambda m: m.__setitem__("distribution_scope", "public"))
    findings = checks.check_take_l4_governance(td, "t0")
    assert any(f.severity == "FAIL" and "distribution_scope" in f.message for f in findings)


def test_l4_missing_poc_contract_id_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "ds", "t0")
    _mutate_manifest(td, lambda m: m.__setitem__("identity", {"pair_id": "pid"}))
    findings = checks.check_take_l4_governance(td, "t0")
    assert any(f.severity == "FAIL" and "contract_id" in f.message for f in findings)


def test_l4_canonical_token_in_lineage_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "ds", "t0")
    _mutate_manifest(td, lambda m: m.__setitem__("spec_version", "canonical-v1"))
    findings = checks.check_take_l4_governance(td, "t0")
    assert any(f.severity == "FAIL" and "canonical" in f.message.lower() for f in findings)


def test_l4_canonical_in_prose_not_flagged(tmp_path: Path) -> None:
    # Descriptive free-text may say "canonical ..." (layout §6 bans it only in lineage identifiers).
    td = make_take(tmp_path / "ds", "t0")
    _mutate_manifest(
        td, lambda m: m.__setitem__("unavailable_or_not_applied", ["canonical SOMA world GRF not generated"])
    )
    findings = checks.check_take_l4_governance(td, "t0")
    assert not any("canonical" in f.message.lower() for f in findings)


def test_l4_absolute_path_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "ds", "t0")
    _mutate_manifest(td, lambda m: m.__setitem__("run_id", r"E:\data\take"))
    findings = checks.check_take_l4_governance(td, "t0")
    assert any(f.severity == "FAIL" and "path" in f.message.lower() for f in findings)


def test_l4_spec_version_mismatch_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "ds", "t0")
    _mutate_manifest(td, lambda m: m.__setitem__("spec_version", "3.0.0"))
    findings = checks.check_take_l4_governance(td, "t0")
    assert any(f.severity == "FAIL" and "spec_version" in f.message for f in findings)


# ------------------------------------------------------------------ L4 dataset
def test_l4_dataset_banner_present_passes(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    findings = checks.check_l4_dataset(ds)
    assert [f for f in findings if f.severity == "FAIL"] == []


def test_l4_dataset_missing_banner_fails(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    (ds / "DATA_DESCRIPTION_EN.md").write_text("no banner here\n", encoding="utf-8")
    findings = checks.check_l4_dataset(ds)
    assert any(f.severity == "FAIL" and "INTERNAL-ONLY" in f.message for f in findings)


# ------------------------------------------------------------------ runner L4
def test_runner_l4_clean_dataset_ok(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    rep = validate(ds, level="L4")
    assert rep.ok, rep.summary()


def test_runner_l4_flags_axis_break(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    td = ds / "amass-take000"
    # break the relabel: replace orientation + global so site0 relabel varies (synthetic -> L3 FAIL)
    gR, ori = _axis_arrays(break_site0=True)
    s = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    s["imu_orientation"] = ori
    np.savez(td / "small_reference.npz", **s)
    ll = dict(np.load(td / "large_reference.npz", allow_pickle=True))
    ll["smpl_global_orientation_world"] = gR
    np.savez(td / "large_reference.npz", **ll)
    rep = validate(ds, level="L4")
    assert not rep.ok
    assert any(f.level == "L3" for f in rep.failures)
