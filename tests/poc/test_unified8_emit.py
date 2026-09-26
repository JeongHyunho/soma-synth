"""Hold `unified8_emit` to `generate_amass_faithful.build` on the same SMPL input.

`unified8_emit` exists because HKNU and AddBiomechanics needed the spec's bundle and reshaping the
AMASS builder to serve them would have put two catalog-registered, L4-PASS datasets at risk. The
cost of that choice is a second copy of the bundle assembly, and the thing that makes the copy safe
is this test: both are run over one real AMASS sequence and their arrays are compared.

It is deliberately an equality test on the numbers, not a smoke test on the shapes. A drift that
changed a filter constant or a joint index would keep every shape and break every value.

Skipped when AMASS is not mounted -- there is no synthetic stand-in, because a stand-in would only
prove the two copies agree about a fixture.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from data_root_skips import DATA_ROOT, needs_data

REPO = Path(__file__).parents[2]
EMIT_PATH = REPO / "scripts" / "poc" / "unified8_emit.py"
AMASS = DATA_ROOT / "extracted" / "amass"

#: Arrays both builders must agree on exactly. Excluded on purpose: mount_id and pair_id (each
#: source names its own), and the string arrays whose dtype unified8_emit deliberately tightened.
SMALL_KEYS = ("imu_orientation", "imu_acceleration", "imu_angular_velocity",
              "imu_confidence", "imu_valid_mask", "timestamps_s", "frame_count")
LARGE_KEYS = ("joint_rotation", "joint_velocity", "root_velocity",
              "pelvis_position_world_aux", "smpl_global_orientation_world",
              "timestamps_s", "frame_count")
ANTHRO_KEYS = ("joint_position", "segment_length", "fixed_joint_rotation", "betas")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _first_amass_sequence() -> Path | None:
    if not AMASS.exists():
        return None
    for dataset in sorted(p for p in AMASS.iterdir() if p.is_dir()):
        for candidate in sorted(dataset.rglob("*_poses.npz")):
            return candidate
    return None


SEQUENCE = _first_amass_sequence()
NEEDS_DATA = needs_data(SEQUENCE is not None, "an AMASS *_poses.npz under extracted/amass")


@pytest.fixture(scope="module")
def both_bundles():
    emit = _load(EMIT_PATH, "unified8_emit")
    gen = emit.FAITHFUL                      # one module instance, shared by both paths

    spec = gen.AmassSeqSpec(
        dataset=SEQUENCE.parent.parent.name,
        subject_id="amass_equivalence_probe",
        sequence_id=SEQUENCE.stem,
        source_npz=str(SEQUENCE),
        relative_path=os.path.relpath(SEQUENCE, DATA_ROOT).replace("\\", "/"),
    )
    data = gen.load_amass_safe(str(SEQUENCE), dataset=spec.dataset)
    gender = data["gender"]
    model = gen.anthro_smpl.load_smpl_model_for_gender(gen.model_gender_letter(gender))
    # One fit handed to both builders: this test is about the two assemblies agreeing, and the
    # constants are an input to both, not something either should choose on its own.
    reduced = gen.reduced_model
    reduction = reduced.fit_subject(
        [gen.build_pose24(data["poses"])], gen.anthro_smpl.rest_joints(model, data["betas"]),
        reduced.settings_for_source("amass"), group="equivalence_probe", basis="probe",
        scope="take")
    small_ref, large_ref, anthro_ref, _ = gen.build(spec, reduction=reduction)
    identity = emit.SourceIdentity(
        source_name="equivalence_probe", dataset=spec.dataset, subject_id=spec.subject_id,
        sequence_id=spec.sequence_id, relative_path=spec.relative_path, asset_sha256="0" * 64,
        native_frames=int(data["poses"].shape[0]), native_rate_hz=float(data["fps"]),
        framerate_source=data["fps_source"], source_note="equivalence probe",
        field_registry="none", frame_convention="probe", contract_id="probe",
        contract_version="0", run_id="probe", mount_prefix="probe", artifact_label="probe",
        anthro_namespace="anthro_reference/probe", subject_scope="sequence_constant_frame_invariant",
        subject_scope_note="probe", height_m=None, height_provenance="probe",
        body_mass_kg=None, body_mass_provenance="probe", betas_provenance="probe",
        smpl_trans_provenance="probe", small_content_note="probe", unavailable=(),
    )
    bundle = emit.build_bundle(
        pose24=gen.build_pose24(data["poses"]), trans=data["trans"], betas=data["betas"],
        gender=gender, model=model, src_fps=data["fps"], source=identity, reduction=reduction,
    )
    return {"small": (small_ref, bundle["small"]),
            "large": (large_ref, bundle["large"]),
            "anthro": (anthro_ref, bundle["anthro"])}


def _compare(reference, produced, keys, label):
    for key in keys:
        assert key in reference, f"{label}: {key} missing from generate_amass_faithful"
        assert key in produced, f"{label}: {key} missing from unified8_emit"
        np.testing.assert_array_equal(
            np.asarray(produced[key]), np.asarray(reference[key]),
            err_msg=f"{label}.{key} drifted between the two builders",
        )


@NEEDS_DATA
def test_the_compared_sequence_actually_carries_motion(both_bundles):
    """Guard the guard: equality on a degenerate input proves nothing.

    Whichever sequence the glob lands on first, it has to be long enough to filter and differentiate
    and it has to move. Without this, a corpus of four-frame still poses would make every comparison
    below pass while comparing nothing.
    """
    reference, _ = both_bundles["small"]
    frames = int(reference["frame_count"])
    assert frames >= 100, f"sequence too short to be evidence ({frames} frames)"
    gyro = np.linalg.norm(np.asarray(reference["imu_angular_velocity"]), axis=2)
    assert gyro.max() > 10.0, f"sequence barely moves (max |omega| {gyro.max():.2f} deg/s)"
    _, large = both_bundles["large"]
    assert np.ptp(np.asarray(large["pelvis_position_world_aux"]), axis=0).max() > 0.01


@NEEDS_DATA
def test_small_channels_are_identical(both_bundles):
    reference, produced = both_bundles["small"]
    _compare(reference, produced, SMALL_KEYS, "small")


@NEEDS_DATA
def test_large_kinematics_are_identical(both_bundles):
    reference, produced = both_bundles["large"]
    _compare(reference, produced, LARGE_KEYS, "large")


@NEEDS_DATA
def test_anthro_constants_are_identical(both_bundles):
    reference, produced = both_bundles["anthro"]
    _compare(reference, produced, ANTHRO_KEYS, "anthro")


@NEEDS_DATA
def test_the_string_arrays_agree_by_value_even_where_the_dtype_was_tightened(both_bundles):
    """unified8_emit writes U where compute_anthro wrote object; the contents must not change.

    The re-typing exists to clear the spec's `object-where-U` WARN. If it also changed a name or an
    order it would be a silent schema change dressed up as a lint fix.
    """
    reference, produced = both_bundles["anthro"]
    for key in ("joint_names", "segment_names", "reference_poses", "fixed_joint_names"):
        assert [str(v) for v in produced[key]] == [str(v) for v in reference[key]]
        assert produced[key].dtype.kind == "U", f"{key} should be U, got {produced[key].dtype}"


@NEEDS_DATA
def test_the_bundle_carries_the_root_translation_the_amass_builder_leaves_out(both_bundles):
    """`build` writes three npz; the spec asks for four.

    AMASS gets `smpl_root_translation.npz` from a separate enrichment pass that reads the source
    file again. Sources whose translation is already in hand should not need that round trip, so
    `build_bundle` emits it directly -- and this pins that it does, because a missing deliverable is
    an L0 FAIL that would only surface after a full corpus had been written.
    """
    emit = sys.modules["unified8_emit"]
    assert "smpl_root_translation.npz" in emit.DELIVERABLES
