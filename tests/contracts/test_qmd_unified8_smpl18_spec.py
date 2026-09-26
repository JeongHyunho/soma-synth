"""Declarative dataset spec for spec_id=qmd_unified8_smpl18, spec_version=faithful-v2: structure
and declared invariants, not values.
"""

import pytest

from soma_synth.contracts.qmd_unified8_smpl18_spec import (
    ARTIFACTS,
    AXIS_CONVENTIONS,
    CAPABILITIES,
    SMALL_MODES,
    SPEC_ID,
    SPEC_VERSION,
    SUBJECT_SCOPES,
    SpecError,
    artifact_names,
    check_arrays,
    derive_capabilities,
    dtype_class,
    expected_deliverables,
)


# ------------------------------------------------------------------ identity
def test_spec_identity() -> None:
    assert SPEC_ID == "qmd_unified8_smpl18"
    assert SPEC_VERSION == "faithful-v2"


def test_allowed_enums_are_frozen() -> None:
    assert SMALL_MODES == frozenset({"measured_physical", "synthetic_from_smpl", "synthetic_from_markers"})
    assert AXIS_CONVENTIONS == frozenset({"spec_S_v2"})
    assert SUBJECT_SCOPES == frozenset(
        {"subject_constant_trial_frame_invariant", "sequence_constant_frame_invariant"}
    )
    assert CAPABILITIES == frozenset({"measured_physical", "development_reference"})


# ------------------------------------------------------------------ dtype_class
def test_dtype_class_maps_numpy_kinds() -> None:
    np = pytest.importorskip("numpy")
    assert dtype_class(np.dtype("float32")) == "f4"
    assert dtype_class(np.dtype("float64")) == "f8"
    assert dtype_class(np.dtype("bool")) == "bool"
    assert dtype_class(np.dtype("int64")) == "i8"
    assert dtype_class(np.dtype("<U19")) == "U"
    assert dtype_class(np.dtype("object")) == "object"


def test_dtype_class_rejects_unsupported() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(SpecError, match="unsupported-dtype"):
        dtype_class(np.dtype("complex64"))


# ------------------------------------------------------------------ artifacts
def test_artifact_names_are_the_five_frozen() -> None:
    assert artifact_names() == (
        "small",
        "large",
        "anthro",
        "root_translation",
        "development_reference",
    )


def test_small_core_is_thirteen_keys() -> None:
    core = {k.name for k in ARTIFACTS["small"].core}
    assert core == {
        "imu_orientation",
        "imu_acceleration",
        "imu_angular_velocity",
        "imu_valid_mask",
        "imu_confidence",
        "imu_orientation_absolute_heading",
        "sensor_codes",
        "mount_id",
        "small_mode",
        "axis_convention",
        "pair_id",
        "timestamps_s",
        "frame_count",
    }
    assert len(ARTIFACTS["small"].core) == 13


def test_small_measured_conditional_keys() -> None:
    cond = ARTIFACTS["small"].conditional
    assert set(cond) == {"measured_physical"}
    assert {k.name for k in cond["measured_physical"]} == {
        "q_anatomical_from_sensor",
        "p_segment_to_sensor_m",
        "imu_synth_mask",
        "gyro_provenance",
    }


def test_large_uses_canonical_world_key_not_prism_world() -> None:
    core = {k.name for k in ARTIFACTS["large"].core}
    assert "smpl_global_orientation_world" in core
    assert "smpl_global_orientation_prism_world" not in core
    # joint_names is (18,) in large (per-artifact namespacing; anthro differs)
    jn = next(k for k in ARTIFACTS["large"].core if k.name == "joint_names")
    assert jn.axes == ("18",)


def test_anthro_has_subject_scope_note_and_free_betas() -> None:
    core = {k.name: k for k in ARTIFACTS["anthro"].core}
    assert "subject_scope_note" in core
    # per-artifact namespacing: joint_names is (22,) here, not (18,)
    assert core["joint_names"].axes == ("22",)
    # betas free length: rank 1, symbolic N axis (not a fixed int)
    assert core["betas"].axes == ("N",)


def test_root_translation_core_frozen() -> None:
    core = {k.name for k in ARTIFACTS["root_translation"].core}
    assert core == {
        "smpl_trans",
        "frame_count",
        "pair_id",
        "source_asset_id",
        "smpl_trans_provenance",
    }
    st = next(k for k in ARTIFACTS["root_translation"].core if k.name == "smpl_trans")
    assert st.axes == ("T", "3")


def test_development_reference_is_capability_gated() -> None:
    art = ARTIFACTS["development_reference"]
    assert art.capability == "development_reference"
    core = {k.name for k in art.core}
    assert core == {
        "namespace",
        "grf_feet_source_native",
        "grf_combined_source_native",
        "cop_feet_world",
        "cop_combined_world",
        "foot_contact_mask",
        "vertical_force_provenance",
        "cop_provenance",
        "contacts_provenance",
        "usage",
        "pair_id",
        "timestamps_s",
    }
    # small/large/anthro/root_translation are always present (no capability gate)
    for name in ("small", "large", "anthro", "root_translation"):
        assert ARTIFACTS[name].capability is None


# ------------------------------------------------------------------ capabilities
def test_derive_capabilities_prism_measured() -> None:
    assert derive_capabilities("measured_physical", "prism") == frozenset(
        {"measured_physical", "development_reference"}
    )


def test_derive_capabilities_amass_synthetic() -> None:
    assert derive_capabilities("synthetic_from_smpl", "amass") == frozenset()


def test_derive_capabilities_gaitex_marker_synthetic_has_no_measured_keys() -> None:
    """A marker-synthesised small carries no physical sensor, so none of the measured-only keys."""
    assert derive_capabilities("synthetic_from_markers", "gaitex") == frozenset()


def test_marker_mode_is_relabel_tolerant_and_smpl_mode_is_not() -> None:
    """L3 reads this set: a marker-fitted body is not a constant relabel of the SMPL joint (ADR-0039)."""
    from soma_synth.contracts.qmd_unified8_smpl18_spec import RELABEL_TOLERANT_SMALL_MODES

    assert RELABEL_TOLERANT_SMALL_MODES == frozenset({"measured_physical", "synthetic_from_markers"})
    assert RELABEL_TOLERANT_SMALL_MODES < SMALL_MODES


def test_derive_capabilities_rejects_unknown_small_mode() -> None:
    with pytest.raises(SpecError, match="unknown-small-mode"):
        derive_capabilities("ideal", "prism")


# ------------------------------------------------------------------ deliverables
def test_expected_deliverables_measured_includes_devref_no_readme() -> None:
    d = expected_deliverables(frozenset({"measured_physical", "development_reference"}))
    assert "development_reference.npz" in d
    assert "smpl_root_translation.npz" in d
    assert "README.md" not in d
    assert set(d) == {
        "small_reference.npz",
        "large_reference.npz",
        "anthro_reference.npz",
        "manifest.json",
        "smpl_root_translation.npz",
        "development_reference.npz",
    }


def test_expected_deliverables_synthetic_omits_devref() -> None:
    d = expected_deliverables(frozenset())
    assert "development_reference.npz" not in d
    assert "smpl_root_translation.npz" in d


# ------------------------------------------------------------------ check_arrays
def _good_small(caps):
    arr = {
        "imu_orientation": ((100, 8, 4), "f4"),
        "imu_acceleration": ((100, 8, 3), "f4"),
        "imu_angular_velocity": ((100, 8, 3), "f4"),
        "imu_valid_mask": ((100, 8), "bool"),
        "imu_confidence": ((100, 8), "f4"),
        "imu_orientation_absolute_heading": ((8,), "bool"),
        "sensor_codes": ((8,), "U"),
        "mount_id": ((8,), "U"),
        "small_mode": ((), "U"),
        "axis_convention": ((), "U"),
        "pair_id": ((), "U"),
        "timestamps_s": ((100,), "f8"),
        "frame_count": ((), "i8"),
    }
    if "measured_physical" in caps:
        arr.update(
            {
                "q_anatomical_from_sensor": ((8, 4), "f4"),
                "p_segment_to_sensor_m": ((8, 3), "f4"),
                "imu_synth_mask": ((100, 8), "bool"),
                "gyro_provenance": ((8,), "U"),
            }
        )
    return arr


def test_check_arrays_clean_synthetic_small_passes() -> None:
    issues = check_arrays("small", frozenset(), _good_small(frozenset()))
    assert issues == ()


def test_check_arrays_clean_measured_small_passes() -> None:
    caps = frozenset({"measured_physical"})
    issues = check_arrays("small", caps, _good_small(caps))
    assert issues == ()


def test_check_arrays_missing_core_key_fails() -> None:
    arr = _good_small(frozenset())
    del arr["imu_orientation"]
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "imu_orientation" in msg for sev, msg in issues)


def test_check_arrays_forbidden_conditional_key_when_capability_absent() -> None:
    arr = _good_small(frozenset())
    arr["gyro_provenance"] = ((8,), "U")  # measured-only key on a synthetic small
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "gyro_provenance" in msg for sev, msg in issues)


def test_check_arrays_missing_required_conditional_key() -> None:
    caps = frozenset({"measured_physical"})
    arr = _good_small(caps)
    del arr["imu_synth_mask"]
    issues = check_arrays("small", caps, arr)
    assert any(sev == "FAIL" and "imu_synth_mask" in msg for sev, msg in issues)


def test_check_arrays_unexpected_key_fails() -> None:
    arr = _good_small(frozenset())
    arr["surprise"] = ((3,), "f4")
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "surprise" in msg for sev, msg in issues)


def test_check_arrays_rank_mismatch_fails() -> None:
    arr = _good_small(frozenset())
    arr["imu_orientation"] = ((100, 8), "f4")  # rank 2, want 3
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "imu_orientation" in msg and "rank" in msg for sev, msg in issues)


def test_check_arrays_fixed_axis_size_mismatch_fails() -> None:
    arr = _good_small(frozenset())
    arr["imu_orientation"] = ((100, 8, 3), "f4")  # last axis 3, want 4 (quat)
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "imu_orientation" in msg for sev, msg in issues)


def test_check_arrays_free_axis_T_must_be_consistent_within_artifact() -> None:
    arr = _good_small(frozenset())
    arr["timestamps_s"] = ((99,), "f8")  # T=99 while others T=100
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "T" in msg for sev, msg in issues)


def test_check_arrays_object_where_U_expected_now_fails() -> None:
    """ADR-0040 D2: a string array stored as ``object`` fails like any other dtype mismatch.

    It used to be excused with a WARN. It cannot be read back without ``allow_pickle=True``, so
    the defect hands arbitrary-code-execution risk to whoever consumes the bundle, and the paired
    contract §13 already lists dtype among the required validations. Where a source has the defect
    and has not yet been repaired, the exemption belongs in its dataset profile as a waiver with
    an expiry -- not in this function as a permanent blanket.
    """
    arr = _good_small(frozenset())
    arr["sensor_codes"] = ((8,), "object")  # object instead of U
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "sensor_codes" in msg for sev, msg in issues)
    assert not any(sev == "WARN" and "sensor_codes" in msg for sev, msg in issues)


def test_check_arrays_dtype_class_mismatch_fails() -> None:
    arr = _good_small(frozenset())
    arr["imu_acceleration"] = ((100, 8, 3), "f8")  # want f4
    issues = check_arrays("small", frozenset(), arr)
    assert any(sev == "FAIL" and "imu_acceleration" in msg for sev, msg in issues)


def test_check_arrays_unknown_artifact_raises() -> None:
    with pytest.raises(SpecError, match="unknown-artifact"):
        check_arrays("nope", frozenset(), {})
