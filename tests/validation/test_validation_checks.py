"""Generator-independent validator: L0 structure, L1 schema, L2 physical.

Fixtures build tiny spec-conformant faithful-v2 take dirs in tmp_path; each test mutates one thing.
Physical bands are the 2026-08-18 calibration (40+40 takes).
"""

import json
from pathlib import Path

import numpy as np

from soma_synth.validation import checks, npz_io
from soma_synth.validation.report import Finding, ValidationReport
from soma_synth.validation.runner import validate


# ------------------------------------------------------------------ fixtures
def _u(x):
    return np.array(x)  # unicode/scalar


def _write_small(path: Path, T: int, source: str, small_mode: str) -> None:
    d = {
        "imu_orientation": np.tile(np.array([1, 0, 0, 0], np.float32), (T, 8, 1)),
        "imu_acceleration": np.full((T, 8, 3), 0.0, np.float32),
        "imu_angular_velocity": np.zeros((T, 8, 3), np.float32),
        "imu_valid_mask": np.ones((T, 8), bool),
        "imu_confidence": np.ones((T, 8), np.float32),
        "imu_orientation_absolute_heading": np.ones((8,), bool),
        "sensor_codes": np.array(["a"] * 8),
        "mount_id": np.array(["m"] * 8),
        "small_mode": _u(small_mode),
        "axis_convention": _u("spec_S_v2"),
        "pair_id": _u("pid"),
        "timestamps_s": (np.arange(T) * 0.01).astype(np.float64),
        "frame_count": np.array(T, np.int64),
    }
    # gravity-included specific force: |a| ~ 9.8 -> within band
    d["imu_acceleration"][..., 2] = 9.81
    if small_mode == "measured_physical":
        d["q_anatomical_from_sensor"] = np.tile(np.array([1, 0, 0, 0], np.float32), (8, 1))
        d["p_segment_to_sensor_m"] = np.zeros((8, 3), np.float32)
        d["imu_synth_mask"] = np.zeros((T, 8), bool)
        d["gyro_provenance"] = np.array(["g"] * 8)
    np.savez(path, **d)


def _write_large(path: Path, T: int) -> None:
    np.savez(
        path,
        joint_names=np.array(["j"] * 18),
        joint_rotation=np.tile(np.array([1, 0, 0, 0], np.float32), (T, 18, 1)),
        joint_velocity=np.zeros((T, 18, 3), np.float32),
        root_velocity=np.zeros((T, 3), np.float32),
        pelvis_position_world_aux=np.zeros((T, 3), np.float32),
        smpl_global_orientation_world=np.tile(np.array([1, 0, 0, 0], np.float32), (T, 24, 1)),
        pair_id=_u("pid"),
        timestamps_s=(np.arange(T) * 0.01).astype(np.float64),
        frame_count=np.array(T, np.int64),
    )


def _write_anthro(path: Path, nbetas: int = 16) -> None:
    prov = _u("source_derived")
    np.savez(
        path,
        namespace=_u("anthro/x"),
        subject_scope=_u("sequence_constant_frame_invariant"),
        subject_scope_note=_u("note"),
        reference_poses=np.array(["rest", "neutral"]),
        joint_names=np.array(["j"] * 22),
        joint_position=np.zeros((2, 22, 3), np.float32),
        joint_position_frame=_u("world"),
        segment_names=np.array(["s"] * 13),
        segment_length=np.ones((13,), np.float32),
        fixed_joint_names=np.array(["f"] * 4),
        fixed_joint_rotation=np.zeros((4, 4), np.float32),
        sex=_u("F"),
        height=np.array(1.7, np.float32),
        body_mass=np.array(60.0, np.float32),
        betas=np.zeros((nbetas,), np.float32),
        betas_num=np.array(nbetas, np.int64),
        gender=_u("fema"),
        height_provenance=prov,
        body_mass_provenance=prov,
        sex_provenance=prov,
        betas_provenance=prov,
        skeleton_provenance=prov,
        fixed_joint_rotation_provenance=prov,
        reconstruction_note=_u("note"),
        pair_id=_u("pid"),
        subject_id=_u("subj"),
        source_asset_id=_u("asset"),
    )


def _write_root_translation(path: Path, T: int) -> None:
    np.savez(
        path,
        smpl_trans=np.zeros((T, 3), np.float32),
        frame_count=np.array(T, np.int64),
        pair_id=_u("pid"),
        source_asset_id=_u("asset"),
        smpl_trans_provenance=_u("source_derived"),
    )


def make_take(
    root: Path, take_id: str, *, source: str = "amass", small_mode: str = "synthetic_from_smpl", T: int = 20
) -> Path:
    td = root / take_id
    td.mkdir(parents=True)
    _write_small(td / "small_reference.npz", T, source, small_mode)
    _write_large(td / "large_reference.npz", T)
    _write_anthro(td / "anthro_reference.npz")
    _write_root_translation(td / "smpl_root_translation.npz", T)
    deliverables = [
        "small_reference.npz",
        "large_reference.npz",
        "anthro_reference.npz",
        "smpl_root_translation.npz",
        "manifest.json",
    ]
    manifest = {
        "artifact_class": "experimental_non_candidate",
        "artifact_label": "x",
        "distribution_scope": "internal_only",
        "generated_utc": "2026-08-18T00:00:00+00:00",
        "run_id": take_id,
        "identity": {"contract_id": "soma_paired_small_large_v3_POC_FAITHFUL", "pair_id": "pid"},
        "source": {"source_name": source},
        "window": {},
        "global_frame": {},
        "canonicalization_config": {},
        "deliverables": deliverables,
        "large_joint_names": [],
        "small_sites_provenance": {},
        "anthro_reconstruction": {},
        "faithful_content": {},
        "excluded_modalities": ["emg"],
        "excluded_modality_reasons": {},
        "unavailable_or_not_applied": [],
        "spec_documents": {},
        "validation": {},
        "safety": {},
        "spec_id": "qmd_unified8_smpl18",
        "spec_version": "faithful-v2",
        "quality_gate": "NOT_EVALUATED",
        # Promoted to universal by ADR-0040 D1. A fixture stands for a conformant bundle, so it
        # carries them: the vertical axis a reader needs to interpret the data, the relationship
        # between the source rate and the 100 Hz grid, and the credit owed to the source dataset.
        "up_axis": "z",
        "resample": {"applied": False},
        # The five keys the registry pins. An empty object stood here while the field was only
        # required to be present, which is precisely the gap that let gaitex and addbio grow
        # different shapes; a fixture standing for a conformant bundle has to carry the shape.
        "source_attribution": {
            "dataset_citation": "Fixture Dataset, https://example.invalid/fixture",
            "publication": "https://example.invalid/fixture-paper",
            "license_id": "CC-BY-4.0",
            "stable_source_id": "fixture_source_2026",
            "changes_made": "true; fixture",
        },
    }
    (td / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return td


def make_dataset(root: Path, source: str = "amass", n: int = 2, small_mode: str = "synthetic_from_smpl") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    takes = []
    for i in range(n):
        tid = f"{source}-take{i:03d}"
        make_take(root, tid, source=source, small_mode=small_mode)
        # `rel` is what the registry requires of every INDEX entry; `relative_path` is kept beside
        # it the way the generators write it since 2026-09-15, when the waiver that excused a
        # `relative_path`-only entry closed for prism and amass.
        takes.append(
            {"take_id": tid, "pair_id": "pid", "status": "ok", "frames": 20,
             "rel": tid, "relative_path": tid}
        )
    index = {
        "spec_id": "qmd_unified8_smpl18",
        "spec_version": "faithful-v2",
        "dataset_dirname": root.name,
        "source": source,
        "artifact_class": "experimental_non_candidate",
        "distribution_scope": "internal_only",
        "generated_utc": "2026-08-18T00:00:00+00:00",
        "counts": {"ok": n, "failed": 0, "excluded": 0, "total": n},
        "complete": True,
        "takes": takes,
    }
    (root / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    (root / "DATA_DESCRIPTION_EN.md").write_text("INTERNAL-ONLY\n", encoding="utf-8")
    # Required of every bundle since 2026-09-14 (dataset_limitations_v1). A fixture bundle has no
    # block in the repository config, so its copy is schema-checked and reported UNVERIFIED (WARN),
    # never FAIL -- the file's presence and shape are the hard rule.
    (root / "KNOWN_LIMITATIONS.json").write_text(json.dumps({
        "schema": "dataset_limitations_v1",
        "bundle": root.name,
        "generated_utc": "2026-09-14T00:00:00Z",
        "source_of_truth": "configs/datasets/known_limitations_v1.yaml",
        "entries": [{
            "id": f"{source}.internal_only",
            "kind": "rights", "severity": "do_not_use", "scope": {"level": "bundle"},
            "statement": "Fixture bundle; internal only.",
            "consequence": "Do not redistribute.",
            "evidence": "manifest.json distribution_scope",
            "status": "open", "since": "2026-09-14",
        }],
    }), encoding="utf-8")
    # Required of every bundle since 2026-09-16: the reduced model's per-subject fit record.
    # Presence is the validator's rule; the content here is a minimal well-formed file.
    (root / "reduced_model_fit.json").write_text(json.dumps({
        "schema": "reduced_model_fit", "method": "smpl18.reduce.fit_constants",
        "source": source, "settings": {}, "group_count": 0, "take_count": 0,
        "groups": {}, "takes": {}, "unfitted": {},
    }), encoding="utf-8")
    return root


# ------------------------------------------------------------------ npz_io
def test_read_npz_schema_reads_object_keys_without_pickle(tmp_path: Path) -> None:
    p = tmp_path / "x.npz"
    np.savez(p, num=np.zeros((3, 4), np.float32), obj=np.array(["a", "b"], dtype=object))
    sch = npz_io.read_npz_schema(p)
    assert sch["num"][0] == (3, 4)
    assert checks_dtype(sch["num"][1]) == "f4"
    assert sch["obj"][0] == (2,)
    assert sch["obj"][1].kind == "O"


def checks_dtype(dt):
    from soma_synth.contracts.qmd_unified8_smpl18_spec import dtype_class

    return dtype_class(dt)


# ------------------------------------------------------------------ report
def test_report_ok_and_to_dict() -> None:
    rep = ValidationReport(dataset_dir="d", spec_id="s", spec_version="v")
    assert rep.ok
    rep.add(Finding("L1", "schema", "WARN", "w"))
    assert rep.ok  # warnings don't fail
    rep.add(Finding("L1", "schema", "FAIL", "boom", take_id="t"))
    assert not rep.ok
    d = rep.to_dict()
    assert d["counts"]["FAIL"] == 1 and d["counts"]["WARN"] == 1
    assert "boom" in rep.summary()


# ------------------------------------------------------------------ L0
def test_l0_clean_dataset_passes(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    findings = checks.check_l0_structure(ds)
    assert [f for f in findings if f.severity == "FAIL"] == []


def test_l0_missing_deliverable_fails(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    (ds / "amass-take000" / "large_reference.npz").unlink()
    findings = checks.check_l0_structure(ds)
    assert any(f.severity == "FAIL" and "large_reference.npz" in f.message for f in findings)


def test_l0_orphan_take_dir_fails(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    make_take(ds, "amass-orphan")  # not in INDEX
    findings = checks.check_l0_structure(ds)
    assert any(f.severity == "FAIL" and "orphan" in f.message.lower() for f in findings)


def test_l0_smpl_root_translation_is_expected_not_orphan(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    findings = checks.check_l0_structure(ds)
    assert not any("smpl_root_translation" in f.message for f in findings)


def test_l0_tolerates_legacy_index_shape(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    legacy = {
        "counts": {"ok": 1, "skipped": 0, "failed": 0, "excluded_non_motion": 0, "total": 1},
        "takes": [{"take": "amass-take000", "sequence_id": "seq", "status": "ok", "frames": 20}],
    }
    (ds / "INDEX.json").write_text(json.dumps(legacy), encoding="utf-8")
    findings = checks.check_l0_structure(ds)
    assert [f for f in findings if f.severity == "FAIL"] == []


# ------------------------------------------------------------------ L1
def test_l1_clean_take_passes(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    findings = checks.check_take_l1(td, "t0")
    assert [f for f in findings if f.severity == "FAIL"] == []


def test_l1_measured_take_passes(tmp_path: Path) -> None:
    td = make_take(tmp_path / "prism_x", "t0", source="prism", small_mode="measured_physical")
    # PRISM also needs development_reference.npz to be complete
    _write_devref(td / "development_reference.npz", 20)
    findings = checks.check_take_l1(td, "t0")
    assert [f for f in findings if f.severity == "FAIL"] == []


def _write_devref(path: Path, T: int) -> None:
    np.savez(
        path,
        namespace=np.array("development_reference/x"),
        grf_feet_source_native=np.zeros((T, 2, 3), np.float32),
        grf_combined_source_native=np.zeros((T, 3), np.float32),
        cop_feet_world=np.zeros((T, 2, 3), np.float32),
        cop_combined_world=np.zeros((T, 3), np.float32),
        foot_contact_mask=np.zeros((T, 2), bool),
        vertical_force_provenance=np.array("measured"),
        cop_provenance=np.array("source_derived"),
        contacts_provenance=np.array("source_derived"),
        usage=np.array("development_reference_only"),
        pair_id=np.array("pid"),
        timestamps_s=(np.arange(T) * 0.01).astype(np.float64),
    )


def test_l1_drifted_large_key_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    # rename the world key to the drifted prism_world -> missing + unexpected
    z = dict(np.load(td / "large_reference.npz", allow_pickle=True))
    z["smpl_global_orientation_prism_world"] = z.pop("smpl_global_orientation_world")
    np.savez(td / "large_reference.npz", **z)
    findings = checks.check_take_l1(td, "t0")
    msgs = " ".join(f.message for f in findings if f.severity == "FAIL")
    assert "smpl_global_orientation_world" in msgs
    assert "smpl_global_orientation_prism_world" in msgs


def test_l1_betas_num_mismatch_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    z = dict(np.load(td / "anthro_reference.npz", allow_pickle=True))
    z["betas_num"] = np.array(10, np.int64)  # betas is length 16
    np.savez(td / "anthro_reference.npz", **z)
    findings = checks.check_take_l1(td, "t0")
    assert any(f.severity == "FAIL" and "betas" in f.message for f in findings)


def test_l1_cross_artifact_T_mismatch_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0", T=20)
    _write_large(td / "large_reference.npz", 19)  # T=19 vs small/root T=20
    findings = checks.check_take_l1(td, "t0")
    assert any(f.severity == "FAIL" and "T" in f.message and "frame" in f.message.lower() for f in findings)


def test_l1_missing_manifest_core_field_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    m = json.loads((td / "manifest.json").read_text(encoding="utf-8"))
    del m["spec_version"]
    (td / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    findings = checks.check_take_l1(td, "t0")
    assert any(f.severity == "FAIL" and "spec_version" in f.message for f in findings)


# ------------------------------------------------------------------ L2
def test_l2_clean_take_passes(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    findings = checks.check_take_l2(td, "t0")
    assert [f for f in findings if f.severity == "FAIL"] == []


def test_l2_nonfinite_accel_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_acceleration"][0, 0, 0] = np.nan
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "finite" in f.message.lower() for f in findings)


def _mask_out(td: Path, frames, site: int, *, fill=np.nan) -> None:
    """Mark (frames, site) invalid and put `fill` in every component of the three IMU arrays there."""
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_valid_mask"][frames, site] = False
    for key in ("imu_orientation", "imu_acceleration", "imu_angular_velocity"):
        z[key][frames, site, :] = fill
    z["imu_confidence"][frames, site] = 0.0
    np.savez(td / "small_reference.npz", **z)


def test_l2_nan_under_false_mask_passes(tmp_path: Path) -> None:
    """ADR-0039: a cell the mask calls invalid is NaN in every IMU array, and that is not a defect."""
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    _mask_out(td, slice(3, 8), 1)
    findings = checks.check_take_l2(td, "t0")
    assert [f for f in findings if f.severity == "FAIL"] == []


def test_l2_finite_value_under_false_mask_fails(tmp_path: Path) -> None:
    """A finite value under a False mask is a fill -- the contract forbids zeros and identities as stand-ins."""
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    _mask_out(td, slice(3, 8), 1, fill=0.0)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "invalid" in f.message and "fill" in f.message for f in findings)


def test_l2_nan_under_true_mask_still_fails_in_marker_mode(tmp_path: Path) -> None:
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_angular_velocity"][2, 4, 1] = np.nan  # mask stays True there
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "calls valid" in f.message for f in findings)


def test_l2_site_with_no_valid_frame_fails(tmp_path: Path) -> None:
    """Coverage is thresholded overall; the per-site floor is that a site must have at least one frame."""
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    _mask_out(td, slice(None), 5)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "site 5" in f.message and "no valid frame" in f.message for f in findings)
    # the overall coverage rule (7/8 valid) is not what caught it
    assert not any("coverage" in f.message for f in findings if f.severity == "FAIL")


def test_l2_confidence_must_be_zero_under_a_false_mask(tmp_path: Path) -> None:
    """ADR-0039 rule 3 names imu_confidence too; before 2026-09-07 the validator never read it."""
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    _mask_out(td, slice(3, 8), 1)
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_confidence"][3:8, 1] = 0.6
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "imu_confidence != 0.0" in f.message for f in findings)


def test_l2_confidence_under_a_true_mask_must_lie_in_the_unit_interval(tmp_path: Path) -> None:
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_confidence"][2, 4] = 0.0          # valid cell, zero confidence
    z["imu_confidence"][3, 4] = 1.5          # valid cell, above one
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "outside (0, 1]" in f.message and "2 cells" in f.message for f in findings)


def test_l2_non_finite_confidence_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_confidence"][2, 4] = np.nan
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "imu_confidence is not finite" in f.message for f in findings)


def test_l2_clean_confidence_passes(tmp_path: Path) -> None:
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    _mask_out(td, slice(3, 8), 1)
    findings = checks.check_take_l2(td, "t0")
    assert not any("imu_confidence" in f.message for f in findings)


def test_l2_bands_are_computed_over_valid_cells_only(tmp_path: Path) -> None:
    """The gyro guard must not see the NaN cells and must still see a real blow-up on a valid cell."""
    td = make_take(tmp_path / "gaitex_x", "t0", source="gaitex", small_mode="synthetic_from_markers")
    _mask_out(td, slice(0, 10), 2)
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_angular_velocity"][12, 2, 0] = 9000.0  # valid cell, above the 8000 deg/s guard
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "gyro" in f.message for f in findings)
    assert not any("finite" in f.message.lower() for f in findings)


def test_l2_gravity_free_accel_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_acceleration"][:] = 0.0
    z["imu_acceleration"][..., 0] = 1.7  # gravity-free-like magnitude
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "accel" in f.message.lower() for f in findings)


def test_l2_high_dynamic_accel_within_recalibrated_band(tmp_path: Path) -> None:
    # per-site median |a| ~64 (KIT running feet) must PASS the recalibrated [5, 80] band
    td = make_take(tmp_path / "amass_x", "t0")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_acceleration"][:] = 0.0
    z["imu_acceleration"][..., 0] = 64.0
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert not any(f.severity == "FAIL" and "accel" in f.message.lower() for f in findings)


def test_l2_jump_low_median_high_peak_passes(tmp_path: Path) -> None:
    # Airborne-dominant jump/hop: a site spends >50% of frames near-weightless (|f|~1) with
    # landing/takeoff spikes (|f|~40). Per-site median |a| < 5 but p95 >> 1g. This is real
    # high-dynamic data (the 5 AMASS BMLmovi/CMU jump takes) and must PASS, not be flagged.
    td = make_take(tmp_path / "amass_x", "t0", T=20)
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_acceleration"][:] = 0.0
    z["imu_acceleration"][:, :, 2] = 1.0        # 13 near-weightless (airborne) frames -> median ~1
    z["imu_acceleration"][:7, :, 2] = 40.0      # 7 landing/takeoff spikes -> p95 ~40 (gravity reached)
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert not any(f.severity == "FAIL" and "accel" in f.message.lower() for f in findings)


def test_l2_gravity_missing_low_median_low_peak_fails(tmp_path: Path) -> None:
    # Real gravity-missing bug: a site is low across the WHOLE take (median ~2, p95 ~8 < 1g) with
    # no frame ever reaching gravity. Must still FAIL — the peak floor (1g) does not let mild-motion
    # gravity-free signals slip through.
    td = make_take(tmp_path / "amass_x", "t0", T=20)
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_acceleration"][:] = 0.0
    z["imu_acceleration"][:, :, 0] = 2.0        # median ~2
    z["imu_acceleration"][:7, :, 0] = 8.0       # p95 ~8, still below 1g (9.80665)
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "accel" in f.message.lower() for f in findings)


def test_l2_bad_quat_norm_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["imu_orientation"][0, 0] = np.array([2.0, 0, 0, 0], np.float32)  # norm 2
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and "quat" in f.message.lower() for f in findings)


def test_l2_bad_timestamp_rate_fails(tmp_path: Path) -> None:
    td = make_take(tmp_path / "amass_x", "t0")
    z = dict(np.load(td / "small_reference.npz", allow_pickle=True))
    z["timestamps_s"] = (np.arange(20) * 0.02).astype(np.float64)  # 50 Hz not 100
    np.savez(td / "small_reference.npz", **z)
    findings = checks.check_take_l2(td, "t0")
    assert any(f.severity == "FAIL" and ("rate" in f.message.lower() or "dt" in f.message.lower()) for f in findings)


# ------------------------------------------------------------------ runner
def test_runner_validate_clean_dataset_ok(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    rep = validate(ds, level="L2")
    assert rep.ok
    assert rep.spec_version == "faithful-v2"


def test_runner_flags_bad_take(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    z = dict(np.load(ds / "amass-take000" / "small_reference.npz", allow_pickle=True))
    z["imu_orientation"][0, 0] = np.array([3.0, 0, 0, 0], np.float32)
    np.savez(ds / "amass-take000" / "small_reference.npz", **z)
    rep = validate(ds, level="L2")
    assert not rep.ok
    assert any(f.take_id == "amass-take000" for f in rep.failures)
