"""AddBiomechanics SMPL-24 retarget -> the eight synthetic IMU channels of the Small spec.

The corpus this reads, `addbio_smpl24_raw`, carries pose, translation and shape but no sensor
channel at all. AddBiomechanics has no worn IMU, so unlike HKNU nothing here can be held against a
physical sensor; what it has instead is 1,085 subjects of dynamics-pass motion, which is the
reason to synthesize it.

The synthesis is imported from `generate_amass_faithful`, not restated: the spec sensor frame, the
anatomical relabel, the resampling, the butter4/7 Hz double-difference chain and the
specific-force convention all live there, so the generators cannot drift by someone editing one copy.
What is new is the source-specific part -- reading the retarget corpus, resampling from rates that
run 100 to 250 Hz, and recording which of the eight sites the source actually drives.

That last part is the thing to read before using this dataset. AddBiomechanics' OpenSim models
carry no neck, head or hand degrees of freedom, and three fifths of the corpus is the No_Arm
variant:

    site        driving joint      what the source says
    shank_l/r   4 / 5              measured
    foot_l/r    10 / 11            measured
    back_T4     9  (spine3)        derived -- the lumbar rotation split in equal thirds
    wrist_l/r   18 / 19            measured in With_Arm, ABSENT in No_Arm
    occiput     15 (head)          ABSENT everywhere

A joint the source does not drive sits at its rest rotation, which makes that channel the parent
segment carried rigidly: a signal that looks like an IMU trace and contains none of the motion the
site is named for. Dropping the channel would break the eight-channel spec, so every artifact
instead carries `site_orientation_source`, `site_position_source` and `site_source_backed`, and a
consumer can refuse what it should not train on. The distinction is per trial, not per dataset,
because it changes between the arm variants.

Usage:
    python scripts/poc/generate_addbio_faithful.py --out <dir> [--studies N] [--subjects N]
                                                   [--trials N] [--jobs N] [--force]
                                                   [--raw <corpus>] [--only STUDY/SUBJECT ...]

The retarget corpus is `--raw`, else the `ADDBIO_RETARGET_CORPUS` environment variable (kept as an
alias for `--raw`), else `$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/addbio_smpl24_raw`.
A selection of no subject, or of no trial, stops the run before the output directory is created.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import multiprocessing
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))


def _load(path: pathlib.Path, name: str):
    spec = _ilu.spec_from_file_location(name, str(path))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The synthesis half, imported rather than copied. Editing it changes every generator at once.
FAITHFUL = _load(_HERE.parent / "generate_amass_faithful.py", "generate_amass_faithful")

from soma_synth.addbio_retarget.shape_fit import (  # noqa: E402
    clean_model_path_for_gender,
    load_clean_model,
    smpl_rest_joints,
)
from soma_synth.pipeline import paths

#: Which retarget corpus to read, when set: an alias for `--raw`, which sets it. `--raw` goes
#: through the environment because on Windows the pool spawns rather than forks, so a worker
#: re-imports this module and knows only what the environment tells it.
RETARGET_CORPUS_ENV = "ADDBIO_RETARGET_CORPUS"
#: One directory per lineage, no version suffix (retention rule 1.1,
#: docs/guides/RETENTION_RULES.md).
#: The corpus at this name must be a wrap-repaired retarget.
RETARGET_LINEAGE = "addbio_smpl24_raw"
SPEC_ID = "addbio_smpl24_synth"
SPEC_VERSION = "faithful-v1"
ARTIFACT_CLASS = "experimental_non_candidate"
QUALITY_GATE = "NOT_EVALUATED"
DISTRIBUTION_SCOPE = "internal_only"
HOLD_NOTE = (
    "experimental_non_candidate; moves no gate; AddBiomechanics remains ADDBIOMECHANICS_SOURCE_HOLD"
)

#: Per site, the SMPL joint whose provenance decides whether the channel carries source motion.
#: The orientation joint is `SITE_GEOM[code][0]`; the position joint is `SITE_POS_JOINT[code]`,
#: except back_T4 whose position is the chest proxy built from the pelvis and the head.
CHEST_PROXY = "chest_proxy:pelvis+2/3*(head-pelvis)"

#: Fields copied from the retarget artifact so a synthetic take can be traced without it.
CARRIED = (
    "variant", "pass_used", "source_name", "source_version", "source_token",
    "stable_source_id", "dataset_citation", "publication", "license_id", "changes_made",
    "content_hashes", "lumbar_distribution", "frame_rotation_source_to_world",
    # which coordinates the retarget had to rebuild because the source filtered them across a
    # 2*pi wrap. A take that was repaired must not arrive downstream looking untouched
    "wrap_repaired_coordinates",
)


def retarget_corpus() -> pathlib.Path:
    """The retarget corpus: `ADDBIO_RETARGET_CORPUS` when set (`--raw` sets it), else the lineage
    directory under SOMA_DATA_ROOT. Resolved when asked, so importing this module reads nothing."""
    override = os.environ.get(RETARGET_CORPUS_ENV)
    return pathlib.Path(override) if override else paths.lineage_dir(RETARGET_LINEAGE)


def study_subject(value: str) -> tuple[str, str]:
    """`--only` value `<study>/<subject>`, as the retarget corpus names its two directory levels."""
    parts = value.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"{value!r} is not <study>/<subject>")
    return parts[0], parts[1]


def enumerate_subjects(corpus: pathlib.Path, studies: int | None = None,
                       subjects: int | None = None, only=None) -> tuple[list[str], list[tuple]]:
    """(studies, [(study, subject), ...]) in the order a run visits them.

    `only` restricts the walk to named `(study, subject)` pairs before `studies` and `subjects`
    take the first N. A named subject is never dropped in silence: a name the corpus does not
    hold is an error rather than an empty run, and so is a cap that would leave out a subject
    `only` names (the error lists them), so the run converts every named subject or none.

    Names are matched exactly as the corpus spells its folders. The check reads the directory
    listing rather than asking `is_dir()` of the joined path: on NTFS that answer ignores case,
    while the filters below do not, so `subject01` against `Subject01` passed the check and then
    selected nothing.
    """
    wanted = set(only) if only else None
    if wanted is not None:
        present = {(study.name, member.name)
                   for study in corpus.iterdir() if study.is_dir()
                   for member in study.iterdir() if member.is_dir()}
        missing = sorted(wanted - present)
        if missing:
            names = ", ".join(f"{a}/{b}" for a, b in missing)
            raise ValueError(f"not in the corpus {corpus} (names are case-sensitive): {names}")
    names = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    if wanted is not None:
        names = [s for s in names if s in {study for study, _ in wanted}]
    if studies:
        names = names[:studies]
    pairs = []
    for study in names:
        members = sorted(p.name for p in (corpus / study).iterdir() if p.is_dir())
        if wanted is not None:
            members = [s for s in members if (study, s) in wanted]
        if subjects:
            members = members[:subjects]
        pairs.extend((study, subject) for subject in members)
    if wanted is not None:
        dropped = sorted(wanted - set(pairs))
        if dropped:
            listed = ", ".join(f"{a}/{b}" for a, b in dropped)
            raise ValueError(f"--studies {studies} / --subjects {subjects} would leave out subjects "
                             f"--only names: {listed}; drop the cap or name fewer subjects")
    return names, pairs


def require_trials(corpus: pathlib.Path, pairs, trials: int | None = None, caps=()) -> None:
    """Refuse a selection that holds no retarget trial, before anything is written.

    `pairs` is what `enumerate_subjects` selected and `trials` the per-subject cap the run applies
    (`sorted(*.npz)[:trials]` when set). An empty selection would still write the run's summary,
    index or fit record over zero trials and exit 0. `caps` are the selection flags as given, for
    the message."""
    def selected(study: str, subject: str) -> list[pathlib.Path]:
        files = sorted((corpus / study / subject).glob("*.npz"))
        return files[:trials] if trials else files

    if any(selected(study, subject) for study, subject in pairs):
        return
    chosen = " ".join(c for c in caps if c)
    what = "no subject" if not pairs else "no retarget trial (<study>/<subject>/*.npz)"
    raise ValueError(f"{what} selected in the corpus {corpus}"
                     + (f" by {chosen}" if chosen else "") + "; nothing written")


def selection_caps(args) -> list[str]:
    """The flags that narrow an AddBio synth run, as given, for a refusal to name."""
    caps = [f"--{name} {value}" for name, value in (("studies", args.studies),
                                                    ("subjects", args.subjects),
                                                    ("trials", args.trials)) if value is not None]
    caps.extend(f"--only {study}/{subject}" for study, subject in args.only or ())
    return caps


def site_sources(joint_provenance) -> dict[str, np.ndarray]:
    """Which of the eight channels the source actually drives, per site.

    Read off the retarget's own `joint_provenance` rather than assumed from the variant name: the
    two agree today, but the provenance is what the conversion recorded and the variant name is a
    directory.
    """
    provenance = [str(p) for p in np.asarray(joint_provenance).tolist()]
    orientation, position, backed = [], [], []
    for code in FAITHFUL.SENSOR_CODES:
        driver = FAITHFUL.SITE_GEOM[code][0]
        orientation.append(provenance[driver])
        if code == "back_T4":
            position.append(CHEST_PROXY)
        else:
            position.append(provenance[FAITHFUL.SITE_POS_JOINT[code]])
        # The orientation joint is the sharp test: it sets the gyro directly and decides how
        # gravity projects into the accelerometer. A site whose driver is absent is the parent
        # segment repeated, whatever its position term does.
        backed.append(provenance[driver] != "absent")
    return {
        "site_orientation_source": np.array(orientation, dtype=np.str_),
        "site_position_source": np.array(position, dtype=np.str_),
        "site_source_backed": np.array(backed, dtype=bool),
    }


def synthesise(rest, local, world, dt) -> dict:
    """The eight spec channels, through generate_amass_faithful's own site construction."""
    frames = local.shape[0]
    global_rot = FAITHFUL.fk_global_rotation(local)
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


def convert_trial(source: pathlib.Path, destination: pathlib.Path, models: dict,
                  corpus: pathlib.Path | None = None) -> dict:
    """One retarget artifact -> one synthetic-IMU artifact. Returns a record, never a verdict.

    `corpus` is the retarget corpus `source` sits in (default `retarget_corpus()`); the artifact
    names its source relative to it."""
    corpus = retarget_corpus() if corpus is None else corpus
    with np.load(source, allow_pickle=False) as raw:
        gender = str(raw["gender"])
        betas = np.asarray(raw["betas"], np.float64)
        pose = np.asarray(raw["pose"], np.float64)
        trans = np.asarray(raw["trans"], np.float64)
        src_rate = float(raw["source_rate_hz"])
        provenance = raw["joint_provenance"]
        carried = {key: raw[key] for key in CARRIED if key in raw.files}
        subject = str(raw["subject"])
        trial_name = str(raw["trial_name"])

    if gender not in models:
        models[gender] = load_clean_model(clean_model_path_for_gender(gender))
    rest = smpl_rest_joints(models[gender], betas)

    frames_in = pose.shape[0]
    local_full = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(frames_in, 24, 3, 3)
    local = FAITHFUL.resample_rotations(local_full, src_rate, FAITHFUL.TARGET_RATE_HZ)
    trans_out = FAITHFUL.resample_trans(trans, src_rate, FAITHFUL.TARGET_RATE_HZ)
    frames = local.shape[0]
    if frames < 4:
        # Too short to differentiate twice. Recorded by name rather than padded: a padded take
        # would carry invented samples with nothing marking them as invented.
        return {"trial": trial_name, "skipped": True, "reason": "under 4 frames after resample",
                "frames_in": frames_in, "source_rate_hz": src_rate}

    world = trans_out[:, None, :] + FAITHFUL.anthro_smpl.fk_positions_batch(rest, local)
    channels = synthesise(rest, local, world, 1.0 / FAITHFUL.TARGET_RATE_HZ)
    sources = site_sources(provenance)

    payload = {
        "spec_id": SPEC_ID, "spec_version": SPEC_VERSION,
        "artifact_class": ARTIFACT_CLASS, "quality_gate": QUALITY_GATE,
        "distribution_scope": DISTRIBUTION_SCOPE, "hold_note": HOLD_NOTE,
        "subject": subject, "trial_name": trial_name, "gender": gender,
        "sampling_rate_hz": np.asarray(FAITHFUL.TARGET_RATE_HZ, np.int64),
        "source_rate_hz": np.asarray(src_rate, np.float64),
        "resample": np.asarray("slerp_local+linear_trans", np.str_),
        "sensor_codes": np.array(FAITHFUL.SENSOR_CODES, dtype=np.str_),
        "betas": betas.astype(np.float32),
        "pose": Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec()
                        .reshape(frames, 24, 3).astype(np.float32),
        "trans": trans_out.astype(np.float32),
        "rest_joints": rest.astype(np.float32),
        "joint_provenance": np.asarray(provenance, dtype=np.str_),
        "synthetic_orientation": channels["orientation"],
        "synthetic_specific_force": channels["specific_force"],
        "synthetic_angular_velocity": channels["angular_velocity"],
        "retarget_source": np.asarray(str(source.relative_to(corpus)).replace("\\", "/"),
                                      np.str_),
    }
    payload.update(sources)
    payload.update(carried)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return {
        "trial": trial_name, "frames": frames, "frames_in": frames_in,
        "source_rate_hz": src_rate,
        "sites_backed": int(sources["site_source_backed"].sum()),
    }


def marker_path(out_root: pathlib.Path, study: str, subject: str) -> pathlib.Path:
    return out_root / study / subject / "_done.json"


def convert_subject(study: str, subject: str, out_root: pathlib.Path,
                    trials: int | None, force: bool) -> dict:
    """Every trial of one subject. The unit of resume, matching the retarget corpus's own."""
    marker = marker_path(out_root, study, subject)
    if marker.exists() and not force:
        return {"study": study, "subject": subject, "resumed": True}

    corpus = retarget_corpus()
    sources = sorted((corpus / study / subject).glob("*.npz"))
    if trials:
        sources = sources[:trials]
    models: dict[str, dict] = {}
    records, skipped = [], []
    for source in sources:
        record = convert_trial(source, out_root / study / subject / source.name, models, corpus)
        (skipped if record.get("skipped") else records).append(record)

    summary = {
        "study": study, "subject": subject,
        "trials": len(records), "skipped": skipped,
        "frames": int(sum(r["frames"] for r in records)),
        "sites_backed": sorted({r["sites_backed"] for r in records}),
        "source_rates_hz": sorted({round(r["source_rate_hz"], 3) for r in records}),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return summary


def _worker(args) -> dict:
    study, subject, out_root, trials, force = args
    try:
        return convert_subject(study, subject, pathlib.Path(out_root), trials, force)
    except Exception as error:                                   # noqa: BLE001
        # Carried back rather than raised: one unreadable subject should not end a run of a
        # thousand, and a failure that is recorded by name can be re-run on its own.
        return {"study": study, "subject": subject, "failed": f"{type(error).__name__}: {error}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--studies", type=int, default=None, help="first N study directories")
    parser.add_argument("--subjects", type=int, default=None, help="first N subjects per study")
    parser.add_argument("--trials", type=int, default=None, help="first N trials per subject")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="ignore _done.json markers")
    parser.add_argument("--raw", default=None,
                        help=f"retarget corpus to read; default ${RETARGET_CORPUS_ENV}, else "
                             f"$SOMA_DATA_ROOT/{paths.POC_DEMO}/{RETARGET_LINEAGE}")
    parser.add_argument("--only", action="append", type=study_subject, default=None,
                        metavar="STUDY/SUBJECT",
                        help="convert only this subject of the corpus (repeatable)")
    args = parser.parse_args(argv)

    if args.raw:
        os.environ[RETARGET_CORPUS_ENV] = args.raw   # before the pool spawns, so workers see it
    try:
        corpus = retarget_corpus()
        paths.body_model_dir()     # checked here: a worker would record it as a failed subject
        out_root = paths.check_output_dir(pathlib.Path(args.out), inputs=[corpus])
    except paths.PathConfigError as error:
        print(error)
        return 2
    if not corpus.is_dir():
        print("retarget corpus not present:", corpus)
        return 2
    print(f"reading {corpus}", flush=True)

    try:
        studies, pairs = enumerate_subjects(corpus, args.studies, args.subjects, args.only)
        require_trials(corpus, pairs, args.trials, selection_caps(args))
    except ValueError as error:
        print(error)
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    work = [(study, subject, str(out_root), args.trials, args.force) for study, subject in pairs]

    print(f"{len(work)} subjects across {len(studies)} studies -> {out_root}", flush=True)
    started = time.time()
    done: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_worker, item): item for item in work}
            for index, future in enumerate(as_completed(futures), 1):
                done.append(future.result())
                _progress(done[-1], index, len(work), started)
    else:
        for index, item in enumerate(work, 1):
            done.append(_worker(item))
            _progress(done[-1], index, len(work), started)

    summary = {
        "spec_id": SPEC_ID, "spec_version": SPEC_VERSION,
        "artifact_class": ARTIFACT_CLASS, "quality_gate": QUALITY_GATE,
        "distribution_scope": DISTRIBUTION_SCOPE, "note": HOLD_NOTE,
        "source_dataset": str(corpus),
        "target_rate_hz": FAITHFUL.TARGET_RATE_HZ,
        "sensor_codes": FAITHFUL.SENSOR_CODES,
        # Counted from this run AND the resumed markers, so the headline is the corpus and not
        # this invocation. The retarget corpus's own SUMMARY reports 990 of its 1,085 subjects for
        # exactly the opposite reason, and it reads like a smaller cohort than it is.
        "subjects_written": sum(1 for d in done if not d.get("resumed") and not d.get("failed")),
        "subjects_resumed": sum(1 for d in done if d.get("resumed")),
        "subjects_failed": [d for d in done if d.get("failed")],
        "trials": sum(d.get("trials", 0) for d in done),
        "frames": sum(d.get("frames", 0) for d in done),
        "skipped": [s for d in done for s in d.get("skipped", [])],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"\n{summary['trials']} trials, {summary['frames']} frames, "
          f"{len(summary['skipped'])} skipped, {len(summary['subjects_failed'])} failed, "
          f"{summary['elapsed_seconds']}s", flush=True)
    return 1 if summary["subjects_failed"] else 0


def _progress(record: dict, index: int, total: int, started: float) -> None:
    rate = (time.time() - started) / max(index, 1)
    tail = (record.get("failed") or ("resumed" if record.get("resumed")
            else f"{record.get('trials', 0)} trials"))
    print(f"  [{index}/{total}] {record['study']}/{record['subject']}: {tail} "
          f"({rate:.1f}s/subject, ~{rate * (total - index) / 60:.0f} min left)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
