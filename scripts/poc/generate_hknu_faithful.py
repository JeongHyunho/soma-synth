"""HKNU FullBody -> 8-channel synthetic IMU, carried alongside the physical IMU at the same sites.

Every other source in this project synthesizes into a vacuum: AMASS and AddBiomechanics have no
worn sensors, so a synthetic channel can only be checked against itself.  HKNU has a real IMU on
every one of the eight Small sites, recorded at the same instant on the same body, and that is
the reason to bring it in at all.  This generator emits both and measures the agreement.

  spec site   HKNU IMU   segment
  back_T4  <- TA         thorax
  wrist_l  <- LFA        left forearm      (this generator mounts "wrist" on the forearm)
  wrist_r  <- RFA
  shank_l  <- LSK
  shank_r  <- RSK
  occiput  <- HE         head
  foot_l   <- LFT
  foot_r   <- RFT

The synthesis itself is not reimplemented.  `generate_amass_faithful` already defines the spec
sensor frame, the anatomical relabel, the double-differentiation filter chain and the
specific-force convention; this module imports them, so the two generators cannot drift apart by
someone editing one copy.  What is new here is the source-specific part: the HKNU retarget in
front of it, and the measured comparison behind it.

Comparison is on |f| and |omega| only.  Those are invariant under rotation, so they can be
compared without first agreeing on how the sensor housing was oriented -- and how it was
oriented is exactly the thing this project keeps getting wrong.  A frame-resolved comparison
needs the mounting resolved first and is deliberately left out.

Usage:
    python scripts/poc/generate_hknu_faithful.py --out <dir> [--subjects S01] [--trials 2]
                                                 [--root <HKNU source, default $SOMA_SOURCE_ROOT/hknu_fullbody>]

A missing source folder, a `--subjects` name that is not both a workbook SubjectID and a
`Dataset_Processed` folder (case-sensitive), and a workbook of no subject stop the run before the
output directory is created.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import pathlib
import sys
import time

import numpy as np
import openpyxl
import scipy.io as sio
from scipy.spatial.transform import Rotation

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))


def _load(path: pathlib.Path, name: str):
    spec = _ilu.spec_from_file_location(name, str(path))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The synthesis half, imported rather than copied. Editing it changes both generators at once.
FAITHFUL = _load(_HERE.parent / "generate_amass_faithful.py", "generate_amass_faithful")
#: The HKNU correspondence and its measured justification live with the diagnostics.
PROBE = _load(_REPO / "scripts/diagnostics/hknu_retarget_probe.py", "hknu_retarget_probe")

from soma_synth.addbio_retarget.pose_fit import (  # noqa: E402
    fit_pose,
    smpl_world_positions,
)
from soma_synth.addbio_retarget.shape_fit import (  # noqa: E402
    clean_model_path_for_gender,
    fit_betas,
    load_clean_model,
    measured_segment_lengths,
    normalise_biological_sex,
    segment_pairs_from,
    smpl_rest_joints,
)
from soma_synth.pipeline import paths

SPEC_ID = "hknu_smpl24_paired"
SPEC_VERSION = "faithful-v1"
ARTIFACT_CLASS = "experimental_non_candidate"
QUALITY_GATE = "NOT_EVALUATED"
DISTRIBUTION_SCOPE = "internal_only"

#: spec sensor code -> the HKNU IMU carrying a physical sensor on the same segment.
MEASURED_SITE = {
    "back_T4": "TA", "wrist_l": "LFA", "wrist_r": "RFA",
    "shank_l": "LSK", "shank_r": "RSK", "occiput": "HE",
    "foot_l": "LFT", "foot_r": "RFT",
}

STATIC_TRIALS = ("Npose", "Tpose")


def subjects_from_workbook(root: pathlib.Path) -> dict[str, dict]:
    book = openpyxl.load_workbook(root / "MATLAB/DatasetInfo.xlsx", read_only=True, data_only=True)
    rows = list(book["Subject"].iter_rows(values_only=True))
    book.close()
    header = [str(c) for c in rows[0]]
    out = {}
    for row in rows[1:]:
        name = row[header.index("SubjectID")]
        if not name:
            continue
        out[str(name)] = {
            "gender": normalise_biological_sex(str(row[header.index("Gender")])),
            "mass_kg": float(row[header.index("BodyMass")]),
            "height_m": float(row[header.index("BodyHeight")]),
        }
    return out


def unusable_inputs(centres: dict, rotations: dict) -> dict[str, int]:
    """Non-finite values in the arrays the retarget actually reads.

    Checked here rather than trusted: a census that keeps only the worst non-finite arrays per
    trial can miss a three-element gap in one wrist joint centre, and `fit_pose` then fails a
    hundred lines downstream with `SVD did not converge`, which names nothing.
    """
    bad = {}
    for name, value in list(centres.items()) + list(rotations.items()):
        count = int((~np.isfinite(np.asarray(value, float))).sum())
        if count:
            bad[name] = count
    return bad


def retarget(mat, model, gender: float, mass_kg: float, height_m: float,
             npose_centres, reference, static_mat):
    """HKNU trial -> SMPL-24 local rotations, world translation, betas.

    The correspondence, the distal centres that let the ankles and wrists be fitted, and the
    N-pose reference that places the head are those of
    ``scripts/diagnostics/hknu_retarget_probe.py``, which measures them.

    Three things are decided from the subject's standing take rather than left to the generic
    path, and each is measured in the
    HKNU fit diagnosis (parent project record, 2026-09-07):

      rest lengths   ten betas cannot reach this cohort's hip and shoulder separations, so the
                     rest skeleton is scaled onto the measured ones afterwards
      root placement the pelvis constant comes from anatomical directions and the root from the
                     hip centres, because two hip offsets cannot separate a displacement between
                     the two skeletons' pelvis origins from a rotation about the hip axis
      lumbar zero    the trunk's turn is measured against the standing reference, since PV's and
                     TA's frames are not parallel even when the subject is neutral
      root track     the root is solved each frame against every measured joint centre rather
                     than hung off the hip centres alone. A source that places its segments
                     independently does not close into a chain -- adjacent joint centres drift
                     6.8 mm through a stride -- so hanging the body on one point accumulates
                     that drift downward, and an accumulated position error is an acceleration
                     error. Spreading it takes the ground-reaction balance R2 from 0.66 to 0.88,
                     past the laboratory's own 0.85
    """
    table = PROBE.build_table()
    centres = PROBE.centres_from(mat)
    rotations = PROBE.rotations_from(mat)
    shape = fit_betas(
        model, measured_segment_lengths(segment_pairs_from(table), npose_centres),
        gender=gender, stature_m=height_m,
        mass_kg=mass_kg if model.get("faces") is not None else None,
    )
    rest, rescale = PROBE.rescale_rest_to_measured(
        smpl_rest_joints(model, shape.betas), npose_centres
    )
    fit = fit_pose(
        rest_joints=rest, correspondence=table, body_of=PROBE.body_map(),
        source_rotations=rotations, source_centres=centres,
        world_rotation=np.eye(3),          # the source frame is already Z-up
        reference_rotations=reference,
        root_placement=PROBE.root_placement_from(static_mat, rest,
                                                 PROBE.tracked_centres()),
        lumbar_from_reference=True,
    )
    return shape, rest, fit, rescale


def synthesise(rest, fit) -> dict:
    """The eight spec channels, through generate_amass_faithful's own site construction."""
    frames = fit.pose.shape[0]
    dt = 1.0 / FAITHFUL.TARGET_RATE_HZ
    local = Rotation.from_rotvec(fit.pose.reshape(-1, 3)).as_matrix().reshape(frames, 24, 3, 3)
    global_rot = FAITHFUL.fk_global_rotation(local)
    world = smpl_world_positions(rest, fit.pose, fit.trans)

    chest = world[:, 0] + FAITHFUL.CHEST_ALPHA * (world[:, 15] - world[:, 0])
    canon = FAITHFUL.canonical_axes(rest)
    orientation = np.zeros((frames, 8, 4), np.float32)
    force = np.zeros((frames, 8, 3), np.float32)
    omega = np.zeros((frames, 8, 3), np.float32)
    for index, code in enumerate(FAITHFUL.SENSOR_CODES):
        joint, kind, proximal, distal, side = FAITHFUL.SITE_GEOM[code]
        relabel = FAITHFUL.anatomical_frame(kind, rest, canon, proximal, distal) @ (
            FAITHFUL._ROT_Y_180 if side == "L" else np.eye(3)
        )
        sensor = np.einsum("nij,jk->nik", global_rot[:, joint], relabel)
        position = chest if code == "back_T4" else world[:, FAITHFUL.SITE_POS_JOINT[code]]
        filtered = FAITHFUL.butter_filtfilt(position)
        velocity = FAITHFUL.butter_filtfilt(FAITHFUL.centered_diff(filtered, dt))
        acceleration = FAITHFUL.centered_diff(velocity, dt)
        orientation[:, index] = FAITHFUL.rotmat_to_quat_wxyz(sensor).astype(np.float32)
        force[:, index] = FAITHFUL.specific_force(acceleration, sensor).astype(np.float32)
        omega[:, index] = FAITHFUL.angular_velocity_deg_s(sensor, dt).astype(np.float32)
    return {"orientation": orientation, "specific_force": force, "angular_velocity": omega}


def measured(mat) -> dict | None:
    """The physical IMU at the same eight sites, in each sensor's own housing frame.

    Left in the housing frame on purpose: resolving it into the spec frame needs the mounting,
    and the mounting is what the comparison below is built to avoid depending on.
    """
    imu = mat["IMU"]
    if not hasattr(getattr(imu, "TA"), "Acc"):
        return None                                   # static trial: no raw IMU stream
    frames = int(np.asarray(mat["Info"].Time, float).size)
    force = np.zeros((frames, 8, 3), np.float32)
    omega = np.zeros((frames, 8, 3), np.float32)
    for index, code in enumerate(FAITHFUL.SENSOR_CODES):
        sensor = getattr(imu, MEASURED_SITE[code])
        force[:, index] = np.asarray(sensor.Acc, float).T.astype(np.float32)
        omega[:, index] = np.degrees(np.asarray(sensor.Gyr, float).T).astype(np.float32)
    return {"specific_force": force, "angular_velocity": omega}


def agreement(synthetic: dict, physical: dict) -> dict:
    """Synthetic against measured, on quantities no frame convention can change.

    |f| and |omega| are invariant under rotation, so a disagreement here is a disagreement about
    the motion rather than about how the housing was mounted. That distinction matters: a
    91 degree mounting error can pass every position-based check.
    """
    out: dict[str, dict] = {}
    for index, code in enumerate(FAITHFUL.SENSOR_CODES):
        entry = {}
        for field, unit in (("specific_force", "m/s^2"), ("angular_velocity", "deg/s")):
            a = np.linalg.norm(synthetic[field][:, index], axis=1).astype(np.float64)
            b = np.linalg.norm(physical[field][:, index], axis=1).astype(np.float64)
            usable = np.isfinite(a) & np.isfinite(b)
            if usable.sum() < 3 or a[usable].std() == 0 or b[usable].std() == 0:
                entry[field] = {"n": int(usable.sum()), "unit": unit}
                continue
            entry[field] = {
                "n": int(usable.sum()),
                "unit": unit,
                "pearson_r": float(np.corrcoef(a[usable], b[usable])[0, 1]),
                "rms_difference": float(np.sqrt(np.mean((a[usable] - b[usable]) ** 2))),
                "synthetic_median": float(np.median(a[usable])),
                "measured_median": float(np.median(b[usable])),
            }
        out[code] = entry
    return out


def convert_subject(subject: str, info: dict, root: pathlib.Path, out_dir: pathlib.Path,
                    limit: int | None) -> list[dict]:
    folder = root / "Dataset_Processed" / subject
    static = sio.loadmat(folder / f"{subject}_Npose.mat",
                         struct_as_record=False, squeeze_me=True)
    npose_centres = PROBE.centres_from(static)
    reference = {name: Rotation.from_matrix(frames).mean().as_matrix()
                 for name, frames in PROBE.rotations_from(static).items()}
    model = load_clean_model(clean_model_path_for_gender(info["gender"]))

    names = [p.stem.replace(f"{subject}_", "", 1)
             for p in sorted(folder.glob(f"{subject}_*.mat"))
             if not p.name.endswith("_SubjectInfo.mat")]
    if limit:
        names = names[:limit]

    records = []
    for trial in names:
        mat = sio.loadmat(folder / f"{subject}_{trial}.mat",
                          struct_as_record=False, squeeze_me=True)
        rate = float(mat["Info"].SamplingRate)
        if rate != FAITHFUL.TARGET_RATE_HZ:
            raise ValueError(f"{subject} {trial}: {rate} Hz, expected {FAITHFUL.TARGET_RATE_HZ}")
        unusable = unusable_inputs(PROBE.centres_from(mat), PROBE.rotations_from(mat))
        if unusable:
            # Skipped rather than filled. One frame of a wrist centre could be interpolated and
            # nobody would see the difference, which is the reason not to: the artifact would
            # carry an invented sample with nothing marking it as invented.
            records.append({"subject": subject, "trial": trial, "skipped": True,
                            "reason": "non-finite retarget input", "arrays": unusable})
            print(f"    SKIP {subject} {trial}: non-finite in {unusable}", flush=True)
            continue
        shape, rest, fit, rescale = retarget(mat, model, info["gender"], info["mass_kg"],
                                             info["height_m"], npose_centres, reference, static)
        synthetic = synthesise(rest, fit)
        physical = measured(mat)
        record = {
            "subject": subject,
            "trial": trial,
            "frames": int(fit.pose.shape[0]),
            "gender": info["gender"],
            "betas": [round(float(b), 6) for b in shape.betas],
            "shape_residual_m": float(shape.residual_m),
            "pose_position_error_m": float(fit.position_error_m),
            "alignment_fitted": sorted(fit.alignment_children),
            "root_offset_mm": [round(float(v) * 1000, 3) for v in np.ravel(fit.root_offset)],
            "rest_rescale": rescale,
            "placed_from_reference": list(fit.placed_from_reference),
            "measured_imu": physical is not None,
        }
        if physical is not None:
            record["agreement"] = agreement(synthetic, physical)

        payload = {
            "spec_id": SPEC_ID, "spec_version": SPEC_VERSION,
            "artifact_class": ARTIFACT_CLASS, "quality_gate": QUALITY_GATE,
            "distribution_scope": DISTRIBUTION_SCOPE,
            "subject": subject, "trial": trial, "gender": info["gender"],
            "sampling_rate_hz": FAITHFUL.TARGET_RATE_HZ,
            "sensor_codes": np.array(FAITHFUL.SENSOR_CODES, dtype=object),
            "measured_site": np.array([MEASURED_SITE[c] for c in FAITHFUL.SENSOR_CODES],
                                      dtype=object),
            "betas": shape.betas.astype(np.float32),
            "pose": fit.pose.astype(np.float32),
            "trans": fit.trans.astype(np.float32),
            "rest_joints": rest.astype(np.float32),
            "synthetic_orientation": synthetic["orientation"],
            "synthetic_specific_force": synthetic["specific_force"],
            "synthetic_angular_velocity": synthetic["angular_velocity"],
        }
        if physical is not None:
            payload["measured_specific_force"] = physical["specific_force"]
            payload["measured_angular_velocity"] = physical["angular_velocity"]
        target = out_dir / subject
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f"{subject}_{trial}.npz", **payload)
        records.append(record)
    return records


def select_subjects(root: pathlib.Path, demographics: dict, wanted=None) -> list[str]:
    """The subjects a run converts: ``wanted`` as given, else every subject of the workbook.

    A ``wanted`` name must be a workbook subject and a folder of ``Dataset_Processed``, both
    matched exactly, case included (on NTFS the joined path exists whatever its case). An unknown
    name, and a workbook that lists no subject, raise ValueError: the run would write SUMMARY.json
    over zero trials and exit 0, or fail on the first unknown subject with its output created."""
    if wanted:
        processed = root / "Dataset_Processed"
        folders = ({p.name for p in processed.iterdir() if p.is_dir()}
                   if processed.is_dir() else set())
        known = sorted(set(demographics) & folders)
        unknown = sorted(set(wanted) - set(known))
        if unknown:
            raise ValueError(
                f"--subjects {' '.join(unknown)}: not an HKNU subject under {root} (names are "
                "case-sensitive; a subject is a SubjectID of MATLAB/DatasetInfo.xlsx with a folder "
                f"in Dataset_Processed; subjects there: {', '.join(known) or 'none'})")
        return list(wanted)
    if not demographics:
        raise ValueError(f"no subject in {root / 'MATLAB' / 'DatasetInfo.xlsx'}; nothing written")
    return sorted(demographics)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=None,
                        help="the HKNU source folder; default $SOMA_SOURCE_ROOT/hknu_fullbody")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=None,
                        help="first N trials per subject, for a smoke run")
    args = parser.parse_args(argv)
    root_given = args.root is not None
    try:
        if args.root is None:
            args.root = paths.source_dir("hknu")
        paths.body_model_dir()     # the models each subject's gender selects live here
    except paths.PathConfigError as error:
        print(error)
        return 2

    out_dir = args.out.resolve()
    if str(out_dir).lower().startswith(str(args.root.resolve()).lower()):
        raise SystemExit("refusing to write inside the read-only source tree")
    try:
        paths.check_output_dir(out_dir, inputs=[args.root])
        paths.existing_source_dir("hknu", args.root if root_given else None)
        # read before the output exists: an unknown subject used to fail in the conversion loop
        # with the output created, and a workbook of no subject wrote an empty SUMMARY.json
        demographics = subjects_from_workbook(args.root)
        chosen = select_subjects(args.root, demographics, args.subjects)
    except (paths.PathConfigError, ValueError) as error:
        print(error)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    records: list[dict] = []
    for subject in chosen:
        rows = convert_subject(subject, demographics[subject], args.root, out_dir, args.trials)
        records.extend(rows)
        paired = [r for r in rows if r.get("measured_imu")]
        skipped = [r for r in rows if r.get("skipped")]
        print(f"  {subject}: {len(rows) - len(skipped)} trials, {len(paired)} with a measured "
              f"IMU, {len(skipped)} skipped, {time.time() - started:6.1f}s elapsed", flush=True)

    summary = {
        "spec_id": SPEC_ID, "spec_version": SPEC_VERSION,
        "artifact_class": ARTIFACT_CLASS, "quality_gate": QUALITY_GATE,
        "distribution_scope": DISTRIBUTION_SCOPE,
        "source_root": str(args.root),
        "sensor_codes": FAITHFUL.SENSOR_CODES,
        "measured_site": MEASURED_SITE,
        "static_trials_without_measured_imu": list(STATIC_TRIALS),
        "subjects": len(chosen),
        "trials": sum(not r.get("skipped") for r in records),
        "trials_skipped": sum(bool(r.get("skipped")) for r in records),
        "trials_with_measured_imu": sum(bool(r.get("measured_imu")) for r in records),
        "elapsed_seconds": round(time.time() - started, 1),
        "records": records,
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8", newline="\n")
    total = sum(f.stat().st_size for f in out_dir.rglob("*.npz"))
    print(f"\n{summary['trials']} trials, {summary['trials_with_measured_imu']} paired, "
          f"{summary['trials_skipped']} skipped, {total / 2**30:.2f} GiB, "
          f"{summary['elapsed_seconds'] / 60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
