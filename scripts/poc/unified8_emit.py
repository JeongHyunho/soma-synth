"""Emit a `qmd_unified8` take bundle from SMPL motion, whatever the source.

PRISM and AMASS reach the validator because their generators write the spec's five-file bundle.
HKNU and AddBiomechanics do not: they were written as one npz per trial under their own spec_id, so
the generator-independent validator -- the only check in this project that was not written by the
same hand as the data -- has never seen them. This module is what lets them in.

It is deliberately NOT a refactor of `generate_amass_faithful.build`. That function produces two
catalog-registered, L4-PASS datasets, and reshaping it to serve two more would put those at risk for
a convenience. Instead the physics is imported from it exactly as the other generators do, and the
bundle assembly that sits on top is written here once and held to the original by
`tests/poc/test_unified8_emit.py`, which runs both over the same SMPL input and compares arrays. The
duplication is real; the test is what stops it drifting.

What is NOT shared, and must not be, is the manifest's account of where the data came from. Feeding
AddBiomechanics through the AMASS builder would have been the short path and would have written
`source_name: "amass"` onto it. `SourceIdentity` exists so each source states its own provenance.

Two constraints the spec enforces that are easy to miss:
  * `unexpected-key` in any npz is a FAIL, so per-source extras (AddBio's per-site source backing,
    HKNU's measured channel) cannot ride along in the arrays. They belong in the manifest, which is
    where `small_sites_provenance` already is.
  * no absolute path may appear in any manifest value.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

_HERE = pathlib.Path(__file__).resolve()


def _load(path: pathlib.Path, name: str):
    spec = _ilu.spec_from_file_location(name, str(path))
    module = _ilu.module_from_spec(spec)
    # Registered before exec because `@dataclass` resolves its own module through sys.modules while
    # the class body runs; without the entry it raises on a None module. `AmassSeqSpec` sidesteps
    # the same trap by not being a dataclass -- registering fixes it for anything loaded this way.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FAITHFUL = _load(_HERE.parent / "generate_amass_faithful.py", "generate_amass_faithful")
ANTHRO = FAITHFUL.anthro_smpl

SPEC_ID = "qmd_unified8"
SPEC_VERSION = "faithful-v2"
DELIVERABLES = (
    "small_reference.npz",
    "large_reference.npz",
    "anthro_reference.npz",
    "smpl_root_translation.npz",
    "manifest.json",
)


@dataclass(frozen=True)
class SourceIdentity:
    """Everything the manifest must say that the SMPL arrays cannot.

    One dataclass rather than a dozen keyword arguments because these travel together and because a
    missing one is a provenance hole, not a default.
    """

    source_name: str
    dataset: str
    subject_id: str
    sequence_id: str
    relative_path: str          # SOMA_DATA_ROOT-relative, forward slashes; never absolute
    asset_sha256: str
    native_frames: int
    native_rate_hz: float
    framerate_source: str
    source_note: str
    field_registry: str
    frame_convention: str
    contract_id: str
    contract_version: str
    run_id: str
    mount_prefix: str
    artifact_label: str
    anthro_namespace: str
    subject_scope: str
    subject_scope_note: str
    height_m: float | None
    height_provenance: str
    body_mass_kg: float | None
    body_mass_provenance: str
    betas_provenance: str
    smpl_trans_provenance: str
    small_content_note: str
    unavailable: tuple[str, ...]
    site_provenance: dict[str, str] = field(default_factory=dict)
    extra_manifest: dict = field(default_factory=dict)
    #: Where the subject's sex was read from. `compute_anthro` stamps a PRISM string on every
    #: source; a generator that knows its own record names it here and the emitter writes that
    #: instead. Empty keeps the old behaviour, so generators that have not said are unchanged.
    sex_provenance: str = ""


def _quat_per_joint(stack: np.ndarray) -> np.ndarray:
    """[T,J,3,3] -> [T,J,4] w-first, sign-continuous along time per joint.

    Per joint on purpose: `rotmat_to_quat_wxyz` only applies its sign-continuity fix on the 2-D
    path, so handing it the bulk [T,J,3,3] array would skip it.
    """
    return np.stack(
        [FAITHFUL.rotmat_to_quat_wxyz(stack[:, j]) for j in range(stack.shape[1])], axis=1
    ).astype(np.float32)


def build_bundle(
    *,
    pose24: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    gender: str,
    model: dict,
    src_fps: float,
    source: SourceIdentity,
    rest_joints: np.ndarray | None = None,
) -> dict:
    """SMPL motion -> the five-file bundle's payloads. Returns a dict of artifact -> mapping.

    `rest_joints` is the skeleton the pose was fitted against. Given, it drives the forward
    kinematics, the sensor sites and the published anthro alike, so the five files describe one
    body. Omitted, it is derived from `betas` as before -- which is the
    same thing only while no generator corrects the skeleton after fitting the betas, and the HKNU
    generator now does.
    """
    pose24 = np.asarray(pose24, np.float64)
    n_src = pose24.shape[0]
    local_full = FAITHFUL.Rotation.from_rotvec(pose24.reshape(-1, 3)).as_matrix().reshape(
        n_src, 24, 3, 3
    )
    local = FAITHFUL.resample_rotations(local_full, src_fps, FAITHFUL.TARGET_RATE_HZ)
    trans_out = FAITHFUL.resample_trans(np.asarray(trans, np.float64), src_fps,
                                        FAITHFUL.TARGET_RATE_HZ)
    m = local.shape[0]
    if m < 4:
        raise ValueError(f"sequence too short after resample (M={m})")
    dt = 1.0 / FAITHFUL.TARGET_RATE_HZ

    gR = FAITHFUL.fk_global_rotation(local)
    j_rest = (ANTHRO.rest_joints(model, np.asarray(betas, np.float64))
              if rest_joints is None else np.asarray(rest_joints, np.float64))
    world = trans_out[:, None, :] + ANTHRO.fk_positions_batch(j_rest, local)

    timestamps_s = np.arange(m, dtype=np.float64) * dt
    timestamps_sha256 = FAITHFUL.sha256_hex(
        np.ascontiguousarray(timestamps_s, "<f8").tobytes()
    )
    canon_cfg = {
        "kind": f"{source.source_name}_unified8_synth", "frame_count": m,
        "target_rate_hz": FAITHFUL.TARGET_RATE_HZ, "source_fps": float(src_fps),
        "fps_source": source.framerate_source, "resample": "slerp_local+linear_trans",
        "frame_convention": source.frame_convention,
        "gravity_world_m_s2": FAITHFUL.GRAVITY_WORLD.tolist(),
        "chest_alpha": FAITHFUL.CHEST_ALPHA, "filter": "butter4_7hz_filtfilt_odd15",
        "angular_velocity": "rotation_log", "smpl_fk": "smpl24_pose_chain",
        "imu_synthesis": "fk_orientation+doublediff_accel+rotationlog_gyro",
    }
    canon_cfg_sha256 = FAITHFUL.sha256_hex(FAITHFUL.canonical_json_bytes(canon_cfg))
    pair_id = FAITHFUL.sha256_hex(
        FAITHFUL.canonical_json_bytes(
            [source.source_name, source.dataset, source.subject_id, source.sequence_id,
             source.asset_sha256, canon_cfg_sha256, source.contract_id, source.contract_version,
             str(FAITHFUL.TARGET_RATE_HZ)]
        )
    )

    # ------------------------------------------------------------------ small
    chest = world[:, 0] + FAITHFUL.CHEST_ALPHA * (world[:, 15] - world[:, 0])
    canon = FAITHFUL.canonical_axes(j_rest)
    n_sensors = FAITHFUL.N_SENSORS
    orientation = np.zeros((m, n_sensors, 4), np.float32)
    acceleration = np.zeros((m, n_sensors, 3), np.float32)
    angular_velocity = np.zeros((m, n_sensors, 3), np.float32)
    confidence = np.zeros((m, n_sensors), np.float32)
    for index, code in enumerate(FAITHFUL.SENSOR_CODES):
        joint, kind, proximal, distal, side = FAITHFUL.SITE_GEOM[code]
        relabel = FAITHFUL.anatomical_frame(kind, j_rest, canon, proximal, distal) @ (
            FAITHFUL._ROT_Y_180 if side == "L" else np.eye(3)
        )
        sensor = np.einsum("nij,jk->nik", gR[:, joint], relabel)
        position = chest if code == "back_T4" else world[:, FAITHFUL.SITE_POS_JOINT[code]]
        filtered = FAITHFUL.butter_filtfilt(position)
        velocity = FAITHFUL.butter_filtfilt(FAITHFUL.centered_diff(filtered, dt))
        acceleration_world = FAITHFUL.centered_diff(velocity, dt)
        orientation[:, index] = FAITHFUL.rotmat_to_quat_wxyz(sensor).astype(np.float32)
        acceleration[:, index] = FAITHFUL.specific_force(acceleration_world, sensor).astype(np.float32)
        angular_velocity[:, index] = FAITHFUL.angular_velocity_deg_s(sensor, dt).astype(np.float32)
        confidence[:, index] = FAITHFUL.SITE_CONFIDENCE.get(code, 0.6)

    small = {
        "sensor_codes": np.array(FAITHFUL.SENSOR_CODES, dtype=np.str_),
        "imu_orientation": orientation,
        "imu_acceleration": acceleration,
        "imu_angular_velocity": angular_velocity,
        "imu_valid_mask": np.ones((m, n_sensors), bool),
        "imu_confidence": confidence,
        "imu_orientation_absolute_heading": np.array(
            [True] * FAITHFUL.N_BODY + [False] * len(FAITHFUL.FOOT_CODES)
        ),
        "mount_id": np.array([f"{source.mount_prefix}::{c}" for c in FAITHFUL.SENSOR_CODES],
                             dtype=np.str_),
        "small_mode": np.str_("synthetic_from_smpl"),
        "axis_convention": np.str_("spec_S_v2"),
        "pair_id": np.str_(pair_id),
        "timestamps_s": timestamps_s,
        "frame_count": np.int64(m),
    }

    # ------------------------------------------------------------------ large
    pelvis_position = world[:, 0].astype(np.float64)
    smpl_global_orientation = _quat_per_joint(gR)

    root_velocity = np.zeros((m, 3), np.float64)
    root_velocity[1:] = (pelvis_position[1:] - pelvis_position[:-1]) / dt
    joint_rotation, joint_velocity = FAITHFUL.large_joint_arrays(gR, local, dt)

    large = {
        "joint_names": np.array(ANTHRO.JOINT24_NAMES, dtype=np.str_),
        "joint_rotation": joint_rotation,
        "joint_velocity": joint_velocity,
        "root_velocity": root_velocity.astype(np.float32),
        "pelvis_position_world_aux": pelvis_position.astype(np.float32),
        "smpl_global_orientation_world": smpl_global_orientation,
        "pair_id": np.str_(pair_id),
        "timestamps_s": timestamps_s,
        "frame_count": np.int64(m),
    }

    # ----------------------------------------------------------------- anthro
    sex_letter = {"f": "F", "m": "M"}.get(str(gender).strip().lower()[:1], "U")
    anthro = ANTHRO.compute_anthro(model, np.asarray(betas, np.float64),
                                   {"gender": sex_letter}, pair_id,
                                   j_rest=None if rest_joints is None else j_rest)
    anthro["namespace"] = source.anthro_namespace
    anthro["subject_scope"] = source.subject_scope
    anthro["subject_scope_note"] = source.subject_scope_note
    anthro["subject_id"] = source.subject_id
    anthro["source_asset_id"] = source.relative_path
    anthro["gender"] = str(gender)
    anthro["height"] = np.float32(source.height_m if source.height_m is not None
                                  else FAITHFUL.smpl_stature_m(model, betas))
    anthro["height_provenance"] = source.height_provenance
    anthro["body_mass"] = np.float32(source.body_mass_kg if source.body_mass_kg is not None
                                     else np.nan)
    anthro["body_mass_provenance"] = source.body_mass_provenance
    anthro["betas_provenance"] = source.betas_provenance
    if source.sex_provenance:
        anthro["sex_provenance"] = source.sex_provenance
    anthro["pair_id"] = np.str_(pair_id)

    # -------------------------------------------------------- root translation
    root_translation = {
        "smpl_trans": trans_out.astype(np.float32),
        "frame_count": np.int64(m),
        "pair_id": np.str_(pair_id),
        "source_asset_id": np.str_(source.relative_path),
        "smpl_trans_provenance": np.str_(source.smpl_trans_provenance),
    }

    # ------------------------------------------------------------- validation
    validation = {
        "imu_quat_norm_max_dev": float(np.max(np.abs(np.linalg.norm(orientation, axis=2) - 1))),
        "joint_rotation_quat_norm_max_dev":
            float(np.max(np.abs(np.linalg.norm(joint_rotation, axis=2) - 1))),
        "timestamp_step_max_err": float(np.max(np.abs(np.diff(timestamps_s) - dt))),
    }
    occiput = FAITHFUL.SENSOR_CODES.index("occiput")
    ojoint, okind, oprox, odist, _ = FAITHFUL.SITE_GEOM["occiput"]
    r_occ = np.einsum("nij,jk->nik", gR[:, ojoint],
                      FAITHFUL.anatomical_frame(okind, j_rest, canon, oprox, odist))
    truth = FAITHFUL.centered_diff(
        FAITHFUL.butter_filtfilt(
            FAITHFUL.centered_diff(
                FAITHFUL.butter_filtfilt(world[:, FAITHFUL.SITE_POS_JOINT["occiput"]]), dt)), dt)
    recovered = np.einsum("nij,nj->ni", r_occ,
                          acceleration[:, occiput, :].astype(np.float64)) + FAITHFUL.GRAVITY_WORLD
    validation["specific_force_inverse_max_abs_err_m_s2"] = float(np.max(np.abs(recovered - truth)))
    reconstructed = pelvis_position[0] + dt * np.cumsum(root_velocity, axis=0)
    validation["root_velocity_cumsum_recon_max_abs_err_m"] = float(
        np.max(np.abs(reconstructed - pelvis_position))
    )

    manifest = {
        "spec_id": SPEC_ID, "spec_version": SPEC_VERSION,
        "artifact_class": "experimental_non_candidate",
        "distribution_scope": "internal_only",
        "quality_gate": "NOT_EVALUATED",
        "artifact_label": source.artifact_label,
        "up_axis": "z",
        "excluded_modalities": ["emg"],
        "excluded_modality_reasons": {"emg": "not_in_synthetic_artifact_scope"},
        "run_id": source.run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "source_name": source.source_name, "dataset": source.dataset,
            "subject_id": source.subject_id, "sequence_id": source.sequence_id,
            "source_asset_sha256": source.asset_sha256,
            "relative_path": source.relative_path,
            "native_frames_total": int(source.native_frames),
            "native_rate_hz": float(source.native_rate_hz),
            "framerate_source": source.framerate_source,
            "field_registry": source.field_registry,
            "note": source.source_note,
        },
        "resample": {
            "source_fps": float(src_fps), "target_fps": FAITHFUL.TARGET_RATE_HZ,
            "frames_in": n_src, "frames_out": m,
            "method": "per-joint Slerp (local rotations) + linear interp (trans)",
            "noop": bool(float(src_fps) == float(FAITHFUL.TARGET_RATE_HZ)),
            "direction": ("upsample" if float(src_fps) < FAITHFUL.TARGET_RATE_HZ
                          else ("none" if float(src_fps) == FAITHFUL.TARGET_RATE_HZ
                                else "downsample")),
        },
        "window": {"frame_start": 0, "frame_end_exclusive": m, "frame_count": m},
        "identity": {
            "pair_id": pair_id, "contract_id": source.contract_id,
            "contract_version": source.contract_version,
            "timestamps_sha256": timestamps_sha256,
            "target_rate_hz": FAITHFUL.TARGET_RATE_HZ,
            "canonicalization_config_sha256": canon_cfg_sha256,
        },
        "canonicalization_config": canon_cfg,
        "global_frame": {"convention": source.frame_convention, "conforms_to_spec_G": True},
        "small_sites_provenance": dict(source.site_provenance),
        "spec_documents": {
            "small_sensors": "small/00_sensors.qmd",
            "small_streams": "small/01_streams.qmd",
            "large_kinematics": "large/02_kinematics.qmd",
            "anthropometry": "anthro/00_anthropometry.qmd",
        },
        "large_joint_names": list(ANTHRO.JOINT24_NAMES),
        "deliverables": list(DELIVERABLES),
        "faithful_content": {
            "small": source.small_content_note,
            "large": "root_velocity[M,3] + joint_rotation[M,24,4] (the source SMPL pose: [0] pelvis "
                     "global, [1:] local) + joint_velocity[M,24,3] + pelvis_position_world_aux + "
                     "smpl_global_orientation[M,24,4].",
            "anthro": "SMPL rest skeleton constants (joint_position[2,24,3], segment_length[15]) + "
                      "betas + gender.",
            "development_reference": "NONE.",
        },
        "anthro_reconstruction": FAITHFUL.anthro_reconstruction_block(),
        "unavailable_or_not_applied": list(source.unavailable),
        "validation": validation,
        "safety": {"npz_loader": "np.load(allow_pickle=False); audited core fields only; "
                                 "no code execution"},
    }
    manifest.update(source.extra_manifest)

    return {"small": small, "large": large, "anthro": anthro,
            "root_translation": root_translation, "manifest": manifest}


def write_data_description(out_root, index: dict, *, title: str, body: str) -> pathlib.Path:
    """`DATA_DESCRIPTION_EN.md`, which L0 and L4 both require at the dataset root.

    L4 checks for the INTERNAL-ONLY banner by substring, so it is written first and literally
    rather than composed, where a rename could quietly drop it.
    """
    counts = index["counts"]
    path = pathlib.Path(out_root) / "DATA_DESCRIPTION_EN.md"
    text = (
        f"# {title}\n\n"
        "**INTERNAL-ONLY.** This dataset is `experimental_non_candidate` with "
        "`quality_gate: NOT_EVALUATED`. It is not canonical output, it is not contract-compliant, "
        "and it must not be published, uploaded, or shared outside the team.\n\n"
        f"- `spec_id` / `spec_version`: `{index['spec_id']}` / `{index['spec_version']}`\n"
        f"- source: `{index['source']}`\n"
        f"- takes: {counts['ok']} ok, {counts['failed']} failed, {counts['excluded']} excluded "
        f"of {counts['total']}\n"
        f"- generated: {index['generated_utc']}\n\n"
        "## Per-take deliverables\n\n"
        "`small_reference.npz` (8-channel synthetic IMU in the spec sensor frame S) - "
        "`large_reference.npz` (SMPL 24-joint kinematics) - "
        "`anthro_reference.npz` (SMPL rest-skeleton constants) - "
        "`smpl_root_translation.npz` - `manifest.json`.\n\n"
        "Large carries all 24 SMPL joints as the source pose: `joint_rotation[:, 0]` is the pelvis "
        "global orientation and `joint_rotation[:, 1:]` the 23 local rotations.\n\n"
        f"{body}\n"
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_bundle(out_dir, bundle: dict) -> None:
    """The five files, at the names the spec's `expected_deliverables` asks for."""
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "small_reference.npz"), **bundle["small"])
    np.savez(os.path.join(out_dir, "large_reference.npz"), **bundle["large"])
    np.savez(os.path.join(out_dir, "anthro_reference.npz"), **bundle["anthro"])
    np.savez(os.path.join(out_dir, "smpl_root_translation.npz"), **bundle["root_translation"])
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(bundle["manifest"], fh, ensure_ascii=False, indent=2)
