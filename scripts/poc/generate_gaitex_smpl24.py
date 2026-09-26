"""Retarget GAITEX onto SMPL-24, reusing the AddBiomechanics retarget unchanged.

The six steps are the ones ``generate_addbio_smpl24.py`` runs -- read the source, fit the
subject's shape once, run forward kinematics, fit the pose, record the provenance, write the
artifact -- and every module that does the work is imported rather than reimplemented. What
differs is upstream of them, in three places GAITEX differs from AddBiomechanics.

**The frames come from files, not a container.** ``gaitex_frames`` reads an ``.osim``, a ``.mot``
and a ``.trc`` where ``b3d_frames`` reads one ``.b3d``, and it repairs the root translation the
published IK left frozen. Joint centres are always recomputed by forward kinematics, because
GAITEX stores none -- the same branch AddBiomechanics takes for a wrap-repaired trial.

**The shape is pooled.** GAITEX rescales its model per trial, so one subject arrives as four
skeletons that disagree by about ``2.7`` percent on a femur. They are pooled to the median and
the spread they spanned is carried onto the artifact.

**Provenance follows the sensors.** The declared skeleton is complete; the nine inertial units
are not. A joint whose body wears nothing is marked absent rather than measured, so the solver's
relaxation is neither shipped nor labelled as an observation.

Nothing under ``extracted/`` is written. The corpus lands beside the other experimental corpora.

    python scripts/poc/generate_gaitex_smpl24.py --out <dir> [--subjects a b] [--max-frames N]

``--source-root`` defaults to ``$SOMA_SOURCE_ROOT/gaitex`` and ``--out`` to
``$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/gaitex_smpl24``. A missing source folder, a
``--subjects`` name the source does not hold (case-sensitive) and a source with no subject stop the
run before anything is written.

INTERNAL-ONLY. Experimental, non-candidate: this script produces no approved artifact and lifts
no hold.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from soma_synth.addbio_retarget.artifact import (  # noqa: E402
    build_artifact,
    write_artifact,
)
from soma_synth.addbio_retarget.osim_kinematics import parse_kinematics  # noqa: E402
from soma_synth.addbio_retarget.pose_fit import fit_pose  # noqa: E402
from soma_synth.addbio_retarget.shape_fit import (  # noqa: E402
    clean_model_path_for_gender,
    fit_betas,
    load_clean_model,
    measured_segment_lengths,
    segment_pairs_from,
    smpl_rest_joints,
)
from soma_synth.addbio_retarget.smpl_correspondence import (  # noqa: E402
    ABSENT,
    correspondence_for,
)
from soma_synth.gaitex_retarget import (  # noqa: E402
    GaitexReaderError,
    discover_subjects,
    knee_bracketing_pairs,
    load_limits,
    measure,
    open_trial,
    pool_segment_lengths,
    read_frames,
    restrict_to_driven,
    undriven_summary,
)
from soma_synth.pipeline import paths

#: Thresholds live in config, not in code constants. Without this file there is no gate,
#: and the run says so rather than falling back to bounds invented in code.
JOINT_RANGE_CONFIG = REPOSITORY_ROOT / "configs" / "datasets" / "gaitex_joint_ranges_v1.yaml"

#: The corpus's lineage directory under $SOMA_DATA_ROOT, the `--out` default; no version suffix
#: (retention rule 1.1, docs/guides/RETENTION_RULES.md).
OUT_LINEAGE = "gaitex_smpl24"
SOURCE_TOKEN = "gaitex"

# GAITEX records no sex anywhere -- not in scaling_settings, not in the metadata, not in the
# archive's own README. Neutral is therefore a choice this script makes and declares, not a value
# it read; the artifact carries the distinction.
GENDER = "neutral"


def file_hash(path: pathlib.Path, limit: int = 1 << 20) -> str:
    """The sibling generators' digest, byte for byte, so provenance fields stay comparable."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]


def conditions_of(root: pathlib.Path, subject: str) -> list[str]:
    return sorted(path.name for path in (root / subject).iterdir() if path.is_dir())


def longest_solved_span(solved: np.ndarray) -> tuple[int, int]:
    """The longest run of frames whose pelvis was solved, as a half-open interval.

    A take must sit on a constant time step. Everything downstream differentiates -- the gyro is
    a rotation log between neighbours and the accelerometer a double difference -- and all of it
    assumes the neighbour is one sample away. Dropping unsolved frames from the middle of a trial
    keeps the array dense while making that assumption false, and the gap then reads as motion:
    one such trial produced 8048 deg/s, which the validator's 8000 bound caught.

    So a gap ends the take rather than being stepped over. The longest span is kept and the rest
    is reported as coverage, which records what was dropped; an interpolation across a
    two-second hole would hide it.
    """
    edges = np.diff(np.concatenate([[0], solved.view(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0, 0
    best = int(np.argmax(stops - starts))
    return int(starts[best]), int(stops[best])


def subject_shape(root: pathlib.Path, subject: str, arguments, log) -> tuple | None:
    """Pool one subject's trials into a single set of segment targets, then fit betas once."""
    per_trial: list[dict[tuple[int, int], float]] = []
    usable: list[tuple[str, object, object]] = []

    for condition in conditions_of(root, subject):
        try:
            source = open_trial(root, subject, condition)
            frames = read_frames(source)
        except GaitexReaderError as error:
            log["unavailable"].append(
                {"subject": subject, "condition": condition,
                 "reason_class": type(error).__name__, "reason": str(error)}
            )
            continue

        table = restrict_to_driven(correspondence_for(source.topology), source.driven_bodies)
        names = source.topology.independent_coordinate_names
        wanted = np.unique(
            np.linspace(0, frames.pos.shape[0] - 1, min(arguments.shape_frames, frames.pos.shape[0]))
            .astype(int)
        )
        solved = wanted[frames.pelvis_solved[wanted]]
        if solved.size < 10:
            log["unavailable"].append(
                {"subject": subject, "condition": condition,
                 "reason_class": "TooFewShapeFrames",
                 "reason": "fewer than ten shape frames have a solved pelvis"}
            )
            continue

        placed = parse_kinematics(source.osim_text).forward_batch(
            dict(zip(names, frames.pos[solved].T))
        )
        pairs = knee_bracketing_pairs(segment_pairs_from(table))
        per_trial.append(measured_segment_lengths(pairs, placed.joint_centres))
        # The frames travel with the source. Reading them again in the writing pass would parse
        # the .mot and refit the pelvis over every frame a second time, which is most of the cost
        # of a trial.
        usable.append((condition, source, frames))

    if not per_trial:
        return None
    return pool_segment_lengths(per_trial), usable


def convert_subject(root, out_root, subject, smpl, digests, arguments, log, limits) -> int:
    pooled = subject_shape(root, subject, arguments, log)
    if pooled is None:
        return 0
    targets, usable = pooled

    shape = fit_betas(
        smpl,
        targets.lengths_m,
        gender=GENDER,
        stature_m=None,   # GAITEX writes height=-1 for every subject
        mass_kg=None,     # and mass=90 for every subject, a placeholder
    )
    rest = smpl_rest_joints(smpl, shape.betas)
    log["shape"][subject] = {
        "betas": [round(float(b), 5) for b in shape.betas],
        "residual_m": round(float(shape.residual_m), 5),
        "trials_pooled": targets.trials_pooled,
        "segment_spread_mm": {
            f"{a}-{b}": round(v * 1000.0, 2) for (a, b), v in sorted(targets.spread_m.items())
        },
    }

    written = 0
    for condition, source, frames in usable:
        names = source.topology.independent_coordinate_names
        full = correspondence_for(source.topology)
        table = restrict_to_driven(full, source.driven_bodies)
        body_of = {e.smpl_index: e.source_body for e in table if e.source_body}
        bodies = sorted(set(body_of.values()))

        start, stop = longest_solved_span(frames.pelvis_solved)
        span = stop - start
        if span < arguments.minimum_frames:
            log["unavailable"].append(
                {"subject": subject, "condition": condition,
                 "reason_class": "SolvedSpanTooShort",
                 "reason": f"longest solved span is only {span} frames"}
            )
            continue
        step = max(1, int(round(span / arguments.max_frames))) if arguments.max_frames else 1
        wanted = np.arange(start, stop, step)
        log["coverage"][f"{subject}/{condition}"] = {
            "span_frames": span,
            "trial_frames": int(frames.pos.shape[0]),
            "fraction": round(span / float(frames.pos.shape[0]), 4),
        }

        # Measured over the frames this take will actually carry, so the number describes the
        # take rather than the trial it was cut from. Recorded, never acted on: see the config.
        if limits is not None:
            keep = np.zeros(frames.pos.shape[0], dtype=bool)
            keep[wanted] = True
            log["joint_ranges"][f"{subject}/{condition}"] = measure(
                frames.pos, names, limits, frame_mask=keep
            ).as_manifest_block()

        placed = parse_kinematics(source.osim_text).forward_batch(
            dict(zip(names, frames.pos[wanted].T))
        )
        pose = fit_pose(
            rest_joints=rest,
            correspondence=table,
            body_of=body_of,
            source_rotations={body: placed.body_rotation[body] for body in bodies},
            source_centres=dict(placed.joint_centres),
            world_rotation=source.world_frame.rotation,
        )
        entries = {e.smpl_index: e for e in table}
        built = build_artifact(
            source_folder=SOURCE_TOKEN,
            subject=subject,
            variant=condition,
            pass_used="imu_ik",
            trial_name=f"{subject}_{condition}",
            betas=shape.betas,
            gender=GENDER,
            pose=pose.pose,
            trans=pose.trans,
            joint_provenance=pose.joint_provenance,
            source_coordinates=[entries[i].source_coordinates for i in range(24)],
            rest_alignment=pose.rest_alignment,
            root_offset=pose.root_offset,
            frame_rotation_source_to_world=source.world_frame.quaternion_wxyz,
            timestamps_s=frames.timestamps_s[wanted],
            wrap_repaired_coordinates=[],
            source_rate_hz=source.trial.source_rate_hz,
            position_error_m=pose.position_error_m,
            adapter_hash=file_hash(source.trial.coordinates_path),
            config_hash="root_recovered_from_pelvis_stations",
            code_hash=digests["code"],
            model_hash=digests["model"],
        )
        destination = out_root / subject / f"{condition}.npz"
        write_artifact(destination, built, compress=arguments.compress)
        written += 1
        log["errors"].append(float(pose.position_error_m))
        log["frames"] += int(wanted.size)
        log["undriven"][f"{subject}/{condition}"] = sorted(undriven_summary(full, table))
        log["kept_joints"][f"{subject}/{condition}"] = [
            e.smpl_index for e in table if e.provenance != ABSENT
        ]
    return written


def select_subjects(root: pathlib.Path, wanted=None) -> list[str]:
    """The subjects a run converts: ``wanted`` as given, else every subject the source holds.

    A ``wanted`` name is matched exactly against the source's subject listing
    (``discover_subjects``), case included: on NTFS the joined path exists whatever its case, and
    the run would then write the corpus under the misspelt name. An unknown name, and a selection
    of no subject, raise ValueError: the run would write ``_run.json`` over zero trials and exit 0.
    """
    found = list(discover_subjects(root))
    if wanted:
        unknown = sorted(set(wanted) - set(found))
        if unknown:
            raise ValueError(
                f"--subjects {' '.join(unknown)}: not a subject of the GAITEX source {root} "
                f"(names are case-sensitive; subjects there: {', '.join(found) or 'none'})")
        return list(wanted)
    if not found:
        raise ValueError(f"no subject (<subject>/<condition>/) under the GAITEX source {root}; "
                         "nothing written")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=None,
                        help="the GAITEX source folder; default $SOMA_SOURCE_ROOT/gaitex")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help=f"default $SOMA_DATA_ROOT/{paths.POC_DEMO}/{OUT_LINEAGE}")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="thin each trial to about this many frames; 0 keeps every frame")
    parser.add_argument("--shape-frames", type=int, default=400,
                        help="frames sampled per trial for the shape fit")
    parser.add_argument("--minimum-frames", type=int, default=100)
    parser.add_argument("--compress", action="store_true")
    arguments = parser.parse_args(argv)
    source_given = arguments.source_root is not None
    try:
        if arguments.source_root is None:
            arguments.source_root = paths.source_dir(SOURCE_TOKEN)
        if arguments.out is None:
            arguments.out = paths.lineage_dir(OUT_LINEAGE)
        paths.check_output_dir(arguments.out, inputs=[arguments.source_root])
        model_path = clean_model_path_for_gender(GENDER)
        paths.existing_source_dir(SOURCE_TOKEN, arguments.source_root if source_given else None)
        subjects = select_subjects(arguments.source_root, arguments.subjects)
    except (paths.PathConfigError, ValueError) as error:
        print(error)
        return 2

    if not model_path.exists():
        raise FileNotFoundError(f"body model not present: {model_path}")
    smpl = load_clean_model(model_path)
    digests = {
        "model": file_hash(model_path),
        "code": file_hash(pathlib.Path(__file__)),
    }

    # No config, no gate. The alternative -- bounds written here as a fallback -- is exactly a
    # code-constant fallback, and it would let a silent default screen the corpus.
    limits = load_limits(JOINT_RANGE_CONFIG) if JOINT_RANGE_CONFIG.exists() else None
    if limits is None:
        print(f"  no joint-range config at {JOINT_RANGE_CONFIG.name}; ranges will not be measured")

    log = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": SOURCE_TOKEN,
        "artifact_class": "experimental_non_candidate",
        "distribution_scope": "internal_only",
        "quality_gate": "NOT_EVALUATED",
        "gender_is_a_choice": "GAITEX records no sex; neutral is declared, not read",
        "root_translation": "recovered from the four pelvis stations; the published IK froze it",
        "subjects": len(subjects),
        "frames": 0,
        "errors": [],
        "shape": {},
        "undriven": {},
        "kept_joints": {},
        "coverage": {},
        "joint_ranges": {},
        "unavailable": [],
    }

    written = 0
    started = time.time()
    for index, subject in enumerate(subjects, start=1):
        written += convert_subject(
            arguments.source_root, arguments.out, subject, smpl, digests, arguments, log,
            limits,
        )
        print(f"  {index}/{len(subjects)} {subject}: {written} trials so far", flush=True)

    log["trials_written"] = written
    log["elapsed_s"] = round(time.time() - started, 1)
    if log["errors"]:
        log["position_error_m"] = {
            "median": round(float(np.median(log["errors"])), 4),
            "p95": round(float(np.percentile(log["errors"], 95)), 4),
        }
    # Grouped by class, not by message: a message names the trial, so counting messages
    # would give every entry a bucket of its own and summarise nothing.
    counts = collections.Counter(entry["reason_class"] for entry in log["unavailable"])
    log["unavailable_by_reason"] = dict(counts)

    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "_run.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {written} trials to {arguments.out}")
    print(f"unavailable: {len(log['unavailable'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
