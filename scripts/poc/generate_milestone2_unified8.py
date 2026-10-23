"""Milestone-2 (KAIST EXO Lab) -> a `qmd_unified8` dataset with Large kinetics.

The source is SMPL fits of overground-reconstructed treadmill walking (MoSh++, 300 betas, neutral
model) and the split-belt treadmill's force plates, frame-aligned with the fits. Small, Large
kinematics, anthro and the root translation come from the fits through the shared emitter, like
every SMPL source. The Large artifact also carries the kinetics of large/03_kinetics.qmd, which
no other source here has: measured GRF, CoP and free moment per foot, and the whole-body and joint
kinetics by inverse dynamics on the subject's SMPL body (soma_synth.kinetics.milestone2).

Frames. The inverse dynamics runs in the treadmill laboratory frame, where the plates measured.
The bundle's world is the overground reconstruction turned so the trial heading is +X (spec frame
G: +X forward, +Y left, +Z up); kinetics are carried into it by the reconstruction's own rigid
transform, so the CoP lies under the foot the Large kinematics place (checked per take, recorded
in manifest.validation).

Usage:
    python scripts/poc/generate_milestone2_unified8.py --out <dir> [--subjects S001 S002] [--only <take>]
                                                       [--source-root <milestone2 folder>]

`--source-root` defaults to `$SOMA_SOURCE_ROOT/milestone2`; the 300-component neutral model is read
from `<data root>/body_models/smpl300` (configs/kinetics/milestone2_inverse_dynamics_v1.yaml).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util as _ilu
import json
import pathlib
import sys
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))


def _load(path: pathlib.Path, name: str):
    spec = _ilu.spec_from_file_location(name, str(path))
    module = _ilu.module_from_spec(spec)
    sys.modules[name] = module          # `@dataclass` in the loaded module needs the entry
    spec.loader.exec_module(module)
    return module


EMIT = _load(_HERE.parent / "unified8_emit.py", "unified8_emit")

from soma_synth.addbio_retarget.shape_fit import load_clean_model  # noqa: E402
from soma_synth.contracts import source_attribution as ATTRIBUTION  # noqa: E402
from soma_synth.kinetics import jcs, milestone2, smpl_body  # noqa: E402
from soma_synth.pipeline import paths  # noqa: E402

SOURCE = "milestone2"
CONTRACT_ID = "soma_paired_small_large_v3_POC_FAITHFUL_MILESTONE2_SYNTH"
CONTRACT_VERSION = "0.0.1-poc-faithful-milestone2-synth"
KINETICS_CONFIG = _REPO / "configs" / "kinetics" / "milestone2_inverse_dynamics_v1.yaml"
FITS_DIR = "smpl_npz_milestone2_refit_overground"
PLATES_DIR = "m2_kinetics_for_hyunhu/kinetics"
FIT_SUFFIX = "_overground_trim.npz"
PLATE_SUFFIX = "_kinetics.npz"
#: The data provider confirmed (2026-10-16) that every subject is male; the fits use the neutral model.
SEX = "male"
SEX_PROVENANCE = "measured:data provider, all Milestone-2 subjects male (H5 sub_info sex code 0)"


@functools.lru_cache(maxsize=1)
def attribution() -> dict[str, str]:
    return ATTRIBUTION.resolved_for(SOURCE)


@functools.lru_cache(maxsize=1)
def kinetics_config() -> dict:
    return yaml.safe_load(KINETICS_CONFIG.read_text(encoding="utf-8"))


def model_path(cfg: dict) -> pathlib.Path:
    bm = cfg["body_model"]
    return paths.data_root() / bm["folder_under_data_root"] / bm["file"]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def takes_in(root: pathlib.Path) -> list[str]:
    fits = sorted(p.name[: -len(FIT_SUFFIX)] for p in (root / FITS_DIR).glob("*" + FIT_SUFFIX))
    plates = {p.name[: -len(PLATE_SUFFIX)] for p in (root / PLATES_DIR).glob("*" + PLATE_SUFFIX)}
    missing = [t for t in fits if t not in plates]
    if missing:
        raise ValueError(f"takes without force plates: {missing}")
    return fits


def subject_of(take: str) -> str:
    return take.split("_", 1)[0]


def heading_rotation(go: np.ndarray, j_rest: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation about Z that turns the trial's mean pelvis heading to +X, and the heading (deg)."""
    fwd = jcs.canonical_axes(j_rest)[1]
    f = np.einsum("tij,j->ti", Rotation.from_rotvec(go).as_matrix(), fwd)[:, :2]
    yaw = float(np.arctan2(f[:, 1].mean(), f[:, 0].mean()))
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), float(np.degrees(yaw))


def turned_pose(fit, j_rest: np.ndarray):
    """(pose24 (T,24,3), pelvis (T,3), R_h) of the overground fit with its heading turned to +X.

    The whole body turns about the world origin: the root rotation is pre-multiplied and the
    pelvis joint, ``J0 + trans`` in SMPL, moves with it. The emitter places the pelvis at the
    translation it is given (its forward kinematics puts the pelvis at the origin), so it is
    handed the pelvis joint's position: the bundle's world is then the SMPL world, turned, and
    the kinetics carried into it land where the body is.
    """
    go = np.asarray(fit["global_orient"], np.float64)
    R_h, _ = heading_rotation(go, j_rest)
    go_turned = (Rotation.from_matrix(R_h) * Rotation.from_rotvec(go)).as_rotvec()
    pelvis = (np.asarray(fit["trans"], np.float64) + j_rest[0]) @ R_h.T
    pose = np.concatenate([go_turned, np.asarray(fit["body_pose"], np.float64)], 1).reshape(-1, 24, 3)
    return pose, pelvis, R_h


def unavailable() -> tuple[str, ...]:
    return (
        "measured IMU (the exoskeleton's back IMU, on 30 takes, is device data and is not carried)",
        "sensor mount extrinsics (segment->sensor lever arm)",
        "joint_angles_jcs_deg (needs the ISB/JCS angle export)",
        "raw marker trajectories (the provider sent the SMPL fits only)",
        "quality gates",
    )


def kinetics_block(k: "milestone2.TakeKinetics", cfg: dict, recorded_mass: float, R_fix_yaw_spread: float,
                   cop_check: dict) -> dict:
    return {
        "spec_document": "large/03_kinetics.qmd",
        "config": "configs/kinetics/milestone2_inverse_dynamics_v1.yaml",
        "config_sha256": sha256_file(KINETICS_CONFIG),
        "method": "Newton-Euler inverse dynamics on the subject's SMPL body (300-beta neutral mesh, "
                  "voxel segment inertia at uniform density scaled to the force-plate mass); measured "
                  "plate wrench applied to the foot, or to the toes when the CoP is ahead of the toe joint",
        "frame_of_computation": "treadmill laboratory frame (inertial; where the plates measured), "
                                "carried into the bundle world by the overground reconstruction's "
                                "rigid transform and the heading turn",
        "filter": cfg["filter"],
        "contact_threshold_n": cfg["plates"]["contact_threshold_n"],
        "body_mass_force_plate_kg": round(k.body_mass_force_plate_kg, 3),
        "body_mass_recorded_kg": recorded_mass,
        "body_mass_note": "the plates weigh subject and exoskeleton together; the segment masses are "
                          "scaled to the plate mass so the measured loads balance the body that bore them",
        "joint_moment_perspective": "internal intersegmental moment on the child segment, N m",
        "joint_torque_component": dict(zip(k.joint_names, (int(c) for c in k.primary_component))),
        "component_provenance": {
            "grf": "measured", "cop": "measured (NaN without contact)", "free_moment": "measured",
            "com_acceleration": "source_derived (segment model)",
            "angular_momentum": "source_derived (segment model)",
            "joint_torque": "source_derived (inverse dynamics)",
            "joint_power": "source_derived (inverse dynamics)",
        },
        "com_acceleration_frame_note": "laboratory-frame (inertial) acceleration, rotated into the bundle "
                                       "world; it excludes the belt-speed variation the overground "
                                       "reconstruction adds to the pelvis path",
        "frames": len(k.grf),
        "grf_invalid_frames": int((~k.grf_valid).sum()),
        "kinematics_invalid_frames": int((~k.kinematics_valid).sum()),
        "kinetics_valid_frames": int(k.valid.sum()),
        "lab_to_overground_yaw_spread_deg": round(R_fix_yaw_spread, 4),
        "cop_under_foot_check": cop_check,
    }


def convert(take: str, root: pathlib.Path, out_root: pathlib.Path, models: dict, cfg: dict) -> dict:
    fit_path = root / FITS_DIR / f"{take}{FIT_SUFFIX}"
    plate_path = root / PLATES_DIR / f"{take}{PLATE_SUFFIX}"
    fit = dict(np.load(fit_path, allow_pickle=False))
    plates = dict(np.load(plate_path, allow_pickle=False))
    betas = np.asarray(fit["betas"], np.float64)
    _, j_rest = smpl_body.shaped(models["smpl_body"], betas)

    pose, trans, R_h = turned_pose(fit, j_rest)
    subject = subject_of(take)
    fit_sha, plate_sha = sha256_file(fit_path), sha256_file(plate_path)
    data_root = paths.data_root()
    height = str(plates["subj_height"]).strip()
    age = str(plates["subj_age"]).strip()
    excluded = cfg["excluded_takes"].get(take)
    source = EMIT.SourceIdentity(
        source_name=SOURCE,
        dataset="milestone2",
        subject_id=f"milestone2_{subject}",
        sequence_id=take[len(subject) + 1:],
        relative_path=paths.data_relative(fit_path, data_root),
        asset_sha256=hashlib.sha256(f"{fit_sha}\n{plate_sha}".encode()).hexdigest(),
        native_frames=int(pose.shape[0]),
        native_rate_hz=float(fit["fps"]),
        framerate_source="source_field:fps",
        source_note="KAIST EXO Lab Milestone-2: split-belt instrumented treadmill walking (level, 0.75/1.00/"
                    "1.25 m/s) fitted with MoSh++ onto SMPL (300 betas, neutral model) and reconstructed "
                    "overground by the provider; force plates frame-aligned with the fits.",
        field_registry="none (the provider's README defines the npz keys)",
        frame_convention="overground reconstruction turned so the trial heading is +X: right-handed, "
                         "Z-up, gravity along -Z (spec frame G)",
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        run_id=f"milestone2-{take}-unified8",
        mount_prefix="milestone2_unified8",
        artifact_label="PoC FAITHFUL Milestone-2 synthetic reference (SMPL-derived synthetic IMU and "
                       "Large kinetics from measured plates and inverse dynamics). NOT contract-compliant, "
                       "NOT quality-gate PASS. INTERNAL-ONLY.",
        anthro_namespace="anthro_reference/milestone2_smpl300_v1",
        subject_scope="subject_constant_trial_frame_invariant",
        subject_scope_note="betas are the provider's per-subject fit, identical across the subject's takes.",
        height_m=float(height) / 1000.0,
        height_provenance="measured:H5 sub_info height (mm)",
        body_mass_kg=float(plates["subj_weight"]),
        body_mass_provenance="measured:H5 sub_info weight (kg), the subject without the exoskeleton",
        betas_provenance="source_derived:MoSh++ fit by the provider (300 betas, basicmodel_neutral v1.1.0)",
        smpl_trans_provenance="source_derived:pelvis joint position (SMPL trans + the rest pelvis offset) "
                              "of the provider's overground reconstruction (belt speed integrated), "
                              "turned so the trial heading is +X",
        small_content_note="8ch synthetic IMU (back_T4, wrist_l/r, shank_l/r, occiput, foot_l/r) fully "
                           "FK-derived in the spec sensor frame S; specific force (gravity-included) + "
                           "rotation-log gyro. small_mode=synthetic_from_smpl, axis_convention=spec_S_v2.",
        unavailable=unavailable(),
        site_provenance={},
        sex_provenance=SEX_PROVENANCE,
        extra_manifest={
            "source_attribution": dict(attribution()),
            "source_files": {
                "smpl_fit": {"relative_path": paths.data_relative(fit_path, data_root), "sha256": fit_sha},
                "force_plates": {"relative_path": paths.data_relative(plate_path, data_root),
                                 "sha256": plate_sha},
            },
            "subject_measures": {"height_mm": float(height), "weight_kg": float(plates["subj_weight"]),
                                 "age_years": float(age) if age else None, "sex": SEX},
            "trial_condition": {"condition": str(fit["condition"]), "level": str(fit["level"]),
                                "trial": str(fit["trial"]),
                                "belt_speed_m_s": int(str(fit["condition"]).split("_")[1][:3]) / 100.0},
            "source_fit_fix": str(fit["fix_applied"]) if "fix_applied" in fit else None,
            "heading_turn_deg": None,
        },
    )
    bundle = EMIT.build_bundle(pose24=pose.reshape(-1, 72), trans=trans, betas=betas, gender=SEX,
                               model=models["anthro"], src_fps=float(fit["fps"]), source=source,
                               rest_joints=j_rest)
    bundle["anthro"]["gender"] = np.str_("neutral")    # the SMPL model the fits used

    # -------- kinetics into Large, in the bundle world
    k = milestone2.compute(fit, plates, models["smpl_body"], cfg)
    R_fix, d = milestone2.lab_to_overground(fit, j_rest[0])
    yaw = np.degrees(np.arctan2(R_fix[:, 1, 0], R_fix[:, 0, 0]))
    spread = float(np.ptp(np.unwrap(np.radians(yaw)))) * 180.0 / np.pi
    w = milestone2.to_frame(k, R_h @ R_fix, d @ R_h.T)
    large = bundle["large"]
    T = int(large["frame_count"])
    if len(k.grf) != T:
        raise ValueError(f"{take}: kinetics have {len(k.grf)} frames, the bundle {T}")
    # The CoP must lie under the foot that the Large kinematics place: the frame transform's check.
    world = (EMIT.ANTHRO.fk_positions_batch(j_rest, Rotation.from_rotvec(pose.reshape(-1, 3))
                                             .as_matrix().reshape(T, 24, 3, 3)) + trans[:, None])
    cop_check = {}
    for i, (side, ank, toe) in enumerate((("left", 7, 10), ("right", 8, 11))):
        on = k.foot_contact[:, i] & k.valid
        mid = 0.5 * (world[on, ank, :2] + world[on, toe, :2])
        dist = np.linalg.norm(w["cop"][on, i, :2] - mid, axis=1)
        cop_check[side] = {"median_m": round(float(np.median(dist)), 4),
                           "p95_m": round(float(np.percentile(dist, 95)), 4)}
    large.update({
        "com_acceleration": w["com_acceleration"].astype(np.float32),
        "angular_momentum": w["angular_momentum"].astype(np.float32),
        "grf": w["grf"].astype(np.float32),
        "cop": w["cop"][..., :2].astype(np.float32),
        "free_moment": k.free_moment.astype(np.float32),
        "joint_torque": k.joint_torque.astype(np.float32),
        "joint_power": k.joint_power.astype(np.float32),
        "kinetics_joint_names": np.array(k.joint_names, dtype=np.str_),
        "joint_moments_jcs": k.joint_moments_jcs.astype(np.float32),
        "joint_power_total": k.joint_power_total.astype(np.float32),
        "foot_contact": k.foot_contact,
        "grf_valid_mask": k.grf_valid,
        "kinetics_valid_mask": k.valid,
        "inverse_dynamics_residual": np.c_[w["residual_force"], w["residual_moment"]].astype(np.float32),
    })
    if list(k.joint_names) != list(large["joint_names"][1:]):
        raise ValueError(f"{take}: kinetics joint order differs from Large joint_names[1:]")

    # SMPL fit glitches (a joint turning tens of degrees in one frame) reach the synthetic IMU as
    # gyro and acceleration spikes no sensor would report. Their frames, padded by the filter's
    # reach, are not valid Small cells: NaN under imu_valid_mask False, confidence 0 (spec: the
    # mask governs the three IMU arrays in any mode). Large stays finite on every frame.
    glitch = ~k.kinematics_valid
    small = bundle["small"]
    if glitch.any():
        small["imu_valid_mask"][glitch] = False
        small["imu_confidence"][glitch] = 0.0
        for name in ("imu_orientation", "imu_acceleration", "imu_angular_velocity"):
            small[name][glitch] = np.nan

    man = bundle["manifest"]
    man["heading_turn_deg"] = round(float(np.degrees(np.arctan2(R_h[1, 0], R_h[0, 0]))), 3)
    man["kinetics"] = kinetics_block(k, cfg, float(plates["subj_weight"]), spread, cop_check)
    man["spec_documents"]["large_kinetics"] = "large/03_kinetics.qmd"
    man["small_fit_glitch_gaps"] = {
        "invalid_frames": int(glitch.sum()),
        "rule": "frames within quality.glitch_pad_frames of a joint rotating more than "
                "quality.max_rotation_step_deg in one frame (configs/kinetics/"
                "milestone2_inverse_dynamics_v1.yaml), all eight sites",
    }
    man["faithful_content"]["large"] += (
        " Kinetics (large/03_kinetics.qmd): grf[M,2,3], cop[M,2,2], free_moment[M,2] measured; "
        "com_acceleration, angular_momentum, joint_torque[M,23], joint_power[M,23] by inverse "
        "dynamics, with joint_moments_jcs, joint_power_total, the GRF/kinetics validity masks and "
        "the pelvis residual.")
    man["faithful_content"]["development_reference"] = (
        "NONE (the measured kinetics are in large, per the external spec).")
    EMIT.write_bundle(out_root / take, bundle)
    entry = {"take_id": take, "rel": take, "status": "ok", "frames": T,
             "pair_id": man["identity"]["pair_id"], "subject_id": source.subject_id,
             "sequence_id": source.sequence_id}
    if excluded:
        entry.update(status="excluded", reason=excluded, excluded_by="configs/kinetics/"
                     "milestone2_inverse_dynamics_v1.yaml excluded_takes")
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root", type=pathlib.Path, default=None,
                        help="the Milestone-2 source folder (default $SOMA_SOURCE_ROOT/milestone2)")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--only", action="append", default=[], help="one take (repeatable)")
    args = parser.parse_args(argv)
    cfg = kinetics_config()
    try:
        root = paths.existing_source_dir(SOURCE, args.source_root)
        mpath = model_path(cfg)
        if not mpath.is_file():
            raise paths.PathConfigError(f"the 300-component neutral SMPL model {mpath} is missing "
                                        "(scripts/poc/prepare_smpl_clean_npz.py --num-betas 300)")
        out_root = paths.check_output_dir(pathlib.Path(args.out), inputs=[root])
        takes = takes_in(root)
        if args.subjects:
            unknown = sorted(set(args.subjects) - {subject_of(t) for t in takes})
            if unknown:
                raise ValueError(f"--subjects {' '.join(unknown)}: not in {root}")
            takes = [t for t in takes if subject_of(t) in set(args.subjects)]
        if args.only:
            unknown = sorted(set(args.only) - set(takes))
            if unknown:
                raise ValueError(f"--only {' '.join(unknown)}: not a take of the selection")
            takes = [t for t in takes if t in set(args.only)]
        if not takes:
            raise ValueError("no take selected; nothing written")
    except (paths.PathConfigError, ValueError) as error:
        print(error)
        return 2
    models = {"anthro": load_clean_model(mpath), "smpl_body": smpl_body.load_model(mpath)}
    out_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    started = time.time()
    by_subject: dict[str, list[str]] = {}
    for t in takes:
        by_subject.setdefault(subject_of(t), []).append(t)
    for subject, group in by_subject.items():
        for t in group:
            try:
                entries.append(convert(t, root, out_root, models, cfg))
            except Exception as error:                          # noqa: BLE001
                entries.append({"take_id": t, "rel": t, "status": "failed",
                                "reason": f"{type(error).__name__}: {error}"})
                print(f"  FAILED {t}: {error}", flush=True)
        print(f"  {subject}: {len(entries)} takes so far ({time.time() - started:.0f}s)", flush=True)

    ok = [e for e in entries if e["status"] == "ok"]
    failed = [e for e in entries if e["status"] == "failed"]
    excluded = [e for e in entries if e["status"] == "excluded"]
    index = {
        "spec_id": EMIT.SPEC_ID, "spec_version": EMIT.SPEC_VERSION,
        "dataset_dirname": out_root.name,
        "source": SOURCE,
        "artifact_class": "experimental_non_candidate",
        "distribution_scope": "internal_only",
        "generated_utc": EMIT.datetime.now(EMIT.timezone.utc).isoformat(),
        "counts": {"total": len(entries), "ok": len(ok), "failed": len(failed), "excluded": len(excluded)},
        "complete": not failed,
        "takes": entries,
    }
    (out_root / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                         encoding="utf-8", newline="\n")
    EMIT.write_data_description(
        out_root, index,
        title="Milestone-2 (KAIST EXO Lab) - synthetic IMU reference with Large kinetics (qmd_unified8)",
        body=(
            "## What this is\n\n"
            "Treadmill walking of 20 subjects (level, belt speeds 0.75 / 1.00 / 1.25 m/s) on a split-belt "
            "instrumented treadmill, fitted onto SMPL (300 betas, neutral model) and reconstructed "
            "overground by the data provider. The eight Small channels are synthesised from the fits "
            "(`small_mode: synthetic_from_smpl`).\n\n"
            "**`large_reference.npz` carries kinetics** (large/03_kinetics.qmd): measured `grf`, `cop` and "
            "`free_moment` per foot, and `com_acceleration`, `angular_momentum`, `joint_torque` and "
            "`joint_power` of the 23 moving joints by inverse dynamics on the subject's SMPL body, with "
            "the lossless `joint_moments_jcs` and `joint_power_total`, `foot_contact`, `grf_valid_mask`, "
            "`kinetics_valid_mask` and the pelvis residual `inverse_dynamics_residual`. Joint moments are "
            "internal (parent-on-child) moments in N m; `joint_torque` is the primary JCS component "
            "(flexion/dorsiflexion; shoulder elevation). Apply `kinetics_valid_mask` before any use.\n\n"
            "## Known limits\n\n"
            "- Forward progression is reconstructed from the belt speed, not measured; the stance foot "
            "slides about 6 cm/s in the overground path. The kinetics are computed in the treadmill "
            "laboratory frame, where they are exact, and carried into the bundle world.\n"
            "- The subjects wore an exoskeleton: the plates weigh it with the subject, so the segment "
            "masses are scaled to the plate mass (see `manifest.kinetics.body_mass_*`).\n"
            "- `kinetics_valid_mask` is False around SMPL fit glitches (a joint turning more than the "
            "configured step in one frame, mostly near the end of a take) and where a foot's CoP is far "
            "from that foot (a step onto the other belt).\n"
            f"- {len(excluded)} of {len(entries)} takes carry `status: excluded` in `INDEX.json` (the "
            "overground reconstruction failed); select takes by `status == \"ok\"`.\n"
            "- The machine-readable list of what this bundle cannot promise is `KNOWN_LIMITATIONS.json`.\n"
        ),
    )
    print(f"\n{len(ok)}/{len(entries)} takes ok, {len(excluded)} excluded, {len(failed)} failed "
          f"in {time.time() - started:.0f}s -> {out_root}", flush=True)
    return 0 if index["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
