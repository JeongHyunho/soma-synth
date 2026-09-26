"""Convert AddBiomechanics trials into the SMPL 24-joint raw reference layer.

Six steps, end to end: read the `.b3d`, fit the subject's shape once, run the OpenSim model to
place its bodies, carry those rotations onto SMPL, turn the world Z-up, and write an artifact
that says where it came from and what it does not contain.

    python scripts/poc/generate_addbio_smpl24.py --out <dir> --per-study 1 --trials 2
    python scripts/poc/generate_addbio_smpl24.py --out <dir> --only Hamner2013_Formatted_No_Arm/subject01

The source is `--source-root`, else `$SOMA_SOURCE_ROOT/addbiomechanics` (by default
`$SOMA_DATA_ROOT/extracted/addbiomechanics`). `--only <study>/<subject>` (repeatable) converts just
the named subjects, whichever split and variant they sit in. A selection of no payload stops the
run before anything is written.

`--out` is required and nothing is written without it; the source volume is never touched. This
moves no gate: AddBiomechanics remains ADDBIOMECHANICS_SOURCE_HOLD, and every artifact carries
`artifact_class=experimental_non_candidate`, `quality_gate=NOT_EVALUATED` and
`distribution_scope=internal_only`.

Trials without a dynamics pass are skipped and counted (53 in the cohort, all Uhlrich2023
No_Arm). Subjects whose `biological_sex` says `unknown` -- twenty of them, all vanderZee2022 --
get the neutral body, which is what the field says rather than a guess at it; the choice is
recorded on every artifact they produce. Pass `--unknown-sex-model` to decide otherwise, or an
empty value to skip them instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from soma_synth.addbio_retarget.artifact import (  # noqa: E402
    SPEC_ID as ARTIFACT_SPEC_ID,
    SPEC_VERSION as ARTIFACT_SPEC_VERSION,
    build_artifact,
    write_artifact,
)
from soma_synth.addbio_retarget.attribution import (  # noqa: E402
    KNOWN_GAPS,
    AttributionSourceUnknown,
)
from soma_synth.addbio_retarget.b3d_frames import (  # noqa: E402
    B3DLayoutError,
    open_b3d,
    read_pass_frames,
)
from soma_synth.addbio_retarget.osim_kinematics import parse_kinematics  # noqa: E402
from soma_synth.addbio_retarget.pose_fit import fit_pose  # noqa: E402
from soma_synth.addbio_retarget.shape_fit import (  # noqa: E402
    clean_model_path_for_gender,
    fit_betas,
    load_clean_model,
    measured_segment_lengths,
    normalise_biological_sex,
    segment_pairs_from,
    smpl_rest_joints,
)
from soma_synth.addbio_retarget.wrap_repair import (  # noqa: E402
    filter_spec_for,
    repair_positions,
)
from soma_synth.addbio_retarget.smpl_correspondence import (  # noqa: E402
    correspondence_for,
)
from soma_synth.pipeline import paths

SOURCE = "addbiomechanics"
_MODULES = (
    "artifact.py", "attribution.py", "b3d_frames.py", "osim_kinematics.py",
    "pose_fit.py", "shape_fit.py", "smpl_correspondence.py", "world_frame.py",
    "wrap_repair.py",
)


def code_hash() -> str:
    """A digest of the modules that made the artifact, so a rerun that differs is visible."""
    digest = hashlib.sha256()
    for name in _MODULES:
        digest.update(
            (REPO / "src/soma_synth/addbio_retarget" / name).read_bytes()
        )
    return digest.hexdigest()[:16]


def file_hash(path: pathlib.Path, limit: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]


def study_subject(value: str) -> tuple[str, str]:
    """`--only` value `<study>/<subject>`: the study folder and the folder holding the `.b3d`."""
    parts = value.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"{value!r} is not <study>/<subject>")
    return parts[0], parts[1]


def payload_files(root: pathlib.Path, per_study: int | None, only=None):
    """(split, variant, study, .b3d) to convert, in conversion order.

    `only` names `(study, subject)` pairs; the rest are passed over before `per_study` counts, so
    the cap applies to the named subjects. A named subject is never dropped in silence: one with
    no payload is an error, and so is a `per_study` cap that would leave one out (the error lists
    them)."""
    wanted = set(only) if only else None
    if wanted is not None:
        present = {
            (study.name, candidate.parent.name)
            for split in root.iterdir() if split.is_dir()
            for variant_dir in split.iterdir() if variant_dir.is_dir()
            for study in variant_dir.iterdir()
            if study.is_dir() and any(study.name == name for name, _ in wanted)
            for candidate in study.rglob("*.b3d") if candidate.stat().st_size > 0
        }
        missing = sorted(wanted - present)
        if missing:
            names = ", ".join(f"{a}/{b}" for a, b in missing)
            raise ValueError(f"no payload under {root} for: {names}")
    picked = []
    for split in sorted(p for p in root.iterdir() if p.is_dir()):
        for variant_dir in sorted(p for p in split.iterdir() if p.is_dir()):
            for study in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
                if wanted is not None and not any(study.name == name for name, _ in wanted):
                    continue
                taken = 0
                for candidate in sorted(study.rglob("*.b3d")):
                    if candidate.stat().st_size == 0:
                        continue          # ADR-0030 non-payload
                    if wanted is not None and (study.name, candidate.parent.name) not in wanted:
                        continue
                    picked.append((split.name, variant_dir.name, study.name, candidate))
                    taken += 1
                    if per_study is not None and taken >= per_study:
                        break
    if wanted is not None:
        dropped = sorted(wanted - {(study, path.parent.name) for _s, _v, study, path in picked})
        if dropped:
            listed = ", ".join(f"{a}/{b}" for a, b in dropped)
            raise ValueError(f"--per-study {per_study} would leave out subjects --only names: "
                             f"{listed}; drop the cap or name fewer subjects")
    return picked


def marker_path(out_root, study, subject) -> pathlib.Path:
    return out_root / study / subject / "_done.json"


def convert_subject(path, out_root, args, models, digests, log):
    b3d = open_b3d(path)
    gender = normalise_biological_sex(
        b3d.subject.biological_sex, fallback=args.unknown_sex_model
    )
    if gender is None:
        log["skipped_sex"].append(str(path.parent.name))
        return 0
    if gender not in models:
        model_path = clean_model_path_for_gender(gender)
        if not model_path.exists():
            raise FileNotFoundError(f"body model not present: {model_path}")
        models[gender] = load_clean_model(model_path)
        digests[gender] = file_hash(model_path)
    smpl = models[gender]

    table = correspondence_for(b3d.topology)
    kinematics = parse_kinematics(b3d.header.passes[-1].model_osim_text)
    coordinates = b3d.topology.independent_coordinate_names
    body_of = {e.smpl_index: e.source_body for e in table if e.source_body}
    bodies = sorted(set(body_of.values()))
    names = b3d.topology.joint_centre_names
    world = b3d.world_frame

    # shape is fitted once for the subject, from every eligible trial's frames
    shape_centres = []
    eligible = []
    for index, trial in enumerate(b3d.trials):
        if trial.length == 0:
            continue
        if "dynamics" not in trial.pass_names:
            log["skipped_trials_no_dynamics"] += 1
            continue
        eligible.append(index)
        if len(shape_centres) < args.shape_trials:
            wanted = np.unique(
                np.linspace(0, trial.length - 1, min(args.shape_frames, trial.length))
                .astype(int)
            )
            shape_centres.append(read_pass_frames(b3d, index, wanted).joint_centres)
    if not eligible:
        return 0
    stacked = np.concatenate(shape_centres)
    centres_for_shape = {name: stacked[:, i] for i, name in enumerate(names)}
    shape = fit_betas(
        smpl,
        measured_segment_lengths(segment_pairs_from(table), centres_for_shape),
        gender=gender,
        stature_m=b3d.subject.height_m or None,
        mass_kg=b3d.subject.mass_kg if smpl.get("faces") is not None else None,
        landmark_offsets=args.landmark_offsets,
    )
    rest = smpl_rest_joints(smpl, shape.betas)

    written = 0
    for index in eligible[: args.trials] if args.trials else eligible:
        trial = b3d.trials[index]
        step = max(1, int(round(trial.length / args.max_frames))) if args.max_frames else 1
        wanted = np.arange(0, trial.length, step)
        # the wrap repair reads the trial whole -- np.unwrap compares neighbours and the filter
        # reads the series as a whole -- so both passes are read at full length and thinned
        # afterwards. Thinning first would ask about a signal that was never sampled.
        every = np.arange(trial.length)
        batch = read_pass_frames(b3d, index, every)
        kinematic_pass = read_pass_frames(
            b3d, index, every, pass_index=trial.pass_names.index("kinematics")
        )
        fixed = repair_positions(kinematic_pass.pos, batch.pos, filter_spec_for(b3d, index))
        repaired_names = [coordinates[c] for c in fixed.repaired_indices]

        # batched, not frame by frame: the same arithmetic 110x faster, which is the difference
        # between eleven minutes and twenty hours across the cohort. A test holds the two paths
        # to the same numbers.
        placed = kinematics.forward_batch(dict(zip(coordinates, fixed.pos[wanted].T)))
        rotations = {body: placed.body_rotation[body] for body in bodies}
        # a repaired trial's stored centres were derived from the coordinates we just replaced,
        # so they still hold the corrupted heading; they are recomputed from what is actually
        # used. An untouched trial keeps the source's own centres, which its own coordinates
        # already reproduce to under a millimetre -- so nothing moves where nothing was wrong.
        if fixed.repaired:
            source_centres = {name: placed.joint_centres[name] for name in names}
        else:
            source_centres = {
                name: batch.joint_centres[wanted][:, i] for i, name in enumerate(names)
            }
        pose = fit_pose(
            rest_joints=rest,
            correspondence=table,
            body_of=body_of,
            source_rotations=rotations,
            source_centres=source_centres,
            world_rotation=world.rotation,
        )
        entries = {e.smpl_index: e for e in table}
        built = build_artifact(
            source_folder=log["study"],
            subject=path.parent.name,
            variant=log["variant"],
            pass_used="dynamics",
            trial_name=trial.original_name or trial.name or f"trial{index}",
            betas=shape.betas,
            gender=gender,
            pose=pose.pose,
            trans=pose.trans,
            joint_provenance=pose.joint_provenance,
            source_coordinates=[entries[i].source_coordinates for i in range(24)],
            rest_alignment=pose.rest_alignment,
            root_offset=pose.root_offset,
            frame_rotation_source_to_world=world.quaternion_wxyz,
            timestamps_s=batch.timestamps_s[wanted],
            wrap_repaired_coordinates=repaired_names,
            source_rate_hz=trial.source_rate_hz,
            position_error_m=pose.position_error_m,
            adapter_hash=file_hash(path),
            config_hash=str(args.landmark_offsets or "none"),
            code_hash=digests["code"],
            model_hash=digests[gender],
        )
        destination = (
            out_root / log["study"] / path.parent.name / f"trial_{index:04d}.npz"
        )
        write_artifact(destination, built, compress=args.compress)
        written += 1
        log["frames"] += int(wanted.size)
        log["errors"].append(float(pose.position_error_m))
        if repaired_names:
            log["wrap_repaired_trials"] += 1
            for name in repaired_names:
                log["wrap_repaired_coordinates"][name] = (
                    log["wrap_repaired_coordinates"].get(name, 0) + 1
                )
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=None,
                        help="the AddBiomechanics source folder; default "
                             "$SOMA_SOURCE_ROOT/addbiomechanics")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--per-study", type=int, default=None,
                        help="subjects per study; omit for the whole cohort")
    parser.add_argument("--only", action="append", type=study_subject, default=None,
                        metavar="STUDY/SUBJECT",
                        help="convert only this subject (repeatable)")
    parser.add_argument("--trials", type=int, default=None,
                        help="trials per subject; omit for all of them")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="thin each trial to about this many frames")
    parser.add_argument("--shape-trials", type=int, default=3)
    parser.add_argument("--shape-frames", type=int, default=60)
    parser.add_argument("--landmark-offsets", default=None,
                        help="name of a shape_fit landmark-offset table (e.g. v1), or none")
    parser.add_argument(
        "--unknown-sex-model", default="neutral",
        help="body model for subjects whose biological_sex says 'unknown'. Neutral by default, "
             "which is what the field says rather than a guess at it; the choice is recorded on "
             "every artifact and in SUMMARY.json. Pass 'skip' to leave them out instead",
    )
    parser.add_argument("--no-compress", dest="compress", action="store_false",
                        help="write uncompressed; about 2.6x the bytes for the same content")
    parser.add_argument("--resume", action="store_true",
                        help="skip subjects already finished, so a long run can be restarted")
    parser.set_defaults(compress=True)
    args = parser.parse_args(argv)
    if args.unknown_sex_model in ("", "none", "None", "skip"):
        args.unknown_sex_model = None       # skip those subjects instead of choosing for them
    if args.landmark_offsets in ("", "none", "None"):
        args.landmark_offsets = None

    try:
        if args.source_root is None:
            args.source_root = paths.source_dir(SOURCE)
        paths.check_output_dir(args.out, inputs=[args.source_root])
        paths.body_model_dir()     # convert_subject loads the models; its loop would not catch this
    except paths.PathConfigError as error:
        print(error)
        return 2
    if not args.source_root.is_dir():
        print("source root not present:", args.source_root)
        return 2
    if args.source_root in args.out.parents or args.out == args.source_root:
        print("refusing to write inside the source volume:", args.out)
        return 2

    models: dict[str, dict] = {}
    digests = {"code": code_hash()}
    totals = {"subjects": 0, "artifacts": 0, "resumed": 0}
    log = {
        "skipped_sex": [],
        "skipped_trials_no_dynamics": 0,
        "wrap_repaired_trials": 0,
        "wrap_repaired_coordinates": {},
        "unattributed": [],
        "errors": [],
        "frames": 0,
    }

    try:
        files = payload_files(args.source_root, args.per_study, args.only)
    except ValueError as error:
        print(error)
        return 2
    if not files:
        # the run would still write SUMMARY.json over zero subjects and exit 0
        caps = [f"--per-study {args.per_study}"] if args.per_study is not None else []
        caps.extend(f"--only {study}/{subject}" for study, subject in args.only or ())
        print(f"no payload (<split>/<variant>/<study>/<subject>/*.b3d, non-empty) selected under "
              f"{args.source_root}" + (f" by {' '.join(caps)}" if caps else "")
              + "; nothing written")
        return 2
    started = time.perf_counter()
    print(f"{len(files)} payload subjects to convert -> {args.out}")

    for position, (_split, variant, study, path) in enumerate(files, start=1):
        log["study"], log["variant"] = study, variant
        marker = marker_path(args.out, study, path.parent.name)
        if args.resume and marker.exists():
            totals["resumed"] += 1
            continue
        try:
            written = convert_subject(path, args.out, args, models, digests, log)
        except AttributionSourceUnknown as exc:
            log["unattributed"].append(f"{study}: {exc}")
            continue
        except B3DLayoutError as exc:
            print(f"  LAYOUT  {study}/{path.parent.name}: {exc}", flush=True)
            continue
        if written:
            totals["subjects"] += 1
            totals["artifacts"] += written
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(
                (json.dumps({"trials": written}) + "\n").encode("utf-8")
            )
        elapsed = time.perf_counter() - started
        remaining = elapsed / position * (len(files) - position)
        print(
            f"  [{position}/{len(files)}] {variant:<9} {study:<32} "
            f"{path.parent.name:<16} {written:>3} trials  "
            f"{log['frames']:>10,} frames  {elapsed / 60:6.1f} min elapsed, "
            f"~{remaining / 60:.0f} min left",
            flush=True,
        )

    errors = np.asarray(log["errors"]) if log["errors"] else np.zeros(0)
    summary = {
        "spec_id": ARTIFACT_SPEC_ID,
        "spec_version": ARTIFACT_SPEC_VERSION,
        "subjects": totals["subjects"],
        "artifacts": totals["artifacts"],
        "frames": log["frames"],
        "resumed_subjects": totals["resumed"],
        "compressed": bool(args.compress),
        "skipped_for_unresolved_sex": len(log["skipped_sex"]),
        "skipped_trials_without_dynamics": log["skipped_trials_no_dynamics"],
        # a count of zero here is a finding, not an absence: it says the cohort was checked
        "wrap_repaired_trials": log["wrap_repaired_trials"],
        "wrap_repaired_coordinates": log["wrap_repaired_coordinates"],
        "unattributed": log["unattributed"],
        "landmark_offsets": args.landmark_offsets,
        "unknown_sex_model": args.unknown_sex_model,
        "joint_centre_error_m": {
            "median": float(np.median(errors)) if errors.size else None,
            "max": float(errors.max()) if errors.size else None,
        },
        "known_gaps": KNOWN_GAPS,
        "note": (
            "experimental_non_candidate; moves no gate; AddBiomechanics remains "
            "ADDBIOMECHANICS_SOURCE_HOLD"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_bytes(
        (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
