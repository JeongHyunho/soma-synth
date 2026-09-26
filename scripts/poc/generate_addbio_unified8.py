"""AddBiomechanics -> a `qmd_unified8_smpl18` dataset the generator-independent validator can check.

Emitted from the retarget corpus `addbio_smpl24_raw`, not from the 8-channel corpus
`addbio_smpl24_synth`, even though the latter is one step closer. The synth corpus is already
resampled to 100 Hz, so building from it would write `resample: {noop: true}` into every manifest
and quietly lose the fact that most of this cohort was recorded at 200-250 Hz and one study at
61 Hz. One hop from the retarget keeps the manifest true.

The channels are built by the same code either way, so the two corpora agree by construction; a
sample of takes is compared against `addbio_smpl24_synth` after the run to check that they do.

What this bundle must say, and does, in `small_sites_provenance`: **four of the eight sites are not
driven by the source.** AddBiomechanics' OpenSim models carry no neck, head or hand coordinates,
and three fifths of the corpus is the No_Arm variant, so `occiput` is unsourced everywhere and
`wrist_l/r` are unsourced in No_Arm. Those channels are the parent segment carried rigidly. The
spec fixes the artifact at eight channels and rejects extra keys, so the labels live in the
manifest -- per take, because they differ between the arm variants.

Usage:
    python scripts/poc/generate_addbio_unified8.py --out <dir> [--studies N] [--subjects N]
                                                   [--trials N] [--jobs N] [--force]
                                                   [--raw <corpus>] [--only STUDY/SUBJECT ...]

The retarget corpus is `--raw`, else the `ADDBIO_RETARGET_CORPUS` environment variable, else
`$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/addbio_smpl24_raw`. A selection of no
subject, or of no trial, stops the run before the output directory is created.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util as _ilu
import json
import multiprocessing
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

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
REDUCED = EMIT.REDUCED
SYNTH = _load(_HERE.parent / "generate_addbio_faithful.py", "generate_addbio_faithful")
# The same scan `scripts/diagnostics/pose_discontinuity_scan.py --apply` used to be run by hand
# after this generator (24 takes of the retired bundle); since 2026-09-15 it is applied while the
# INDEX is written, as the HKNU generator does, so no take is excluded in a pass someone forgets.
SCAN = _load(_HERE.parents[1] / "diagnostics" / "pose_discontinuity_scan.py",
             "pose_discontinuity_scan")

from soma_synth.addbio_retarget.shape_fit import (  # noqa: E402
    clean_model_path_for_gender,
    load_clean_model,
)
from soma_synth.pipeline import paths

#: Which retarget corpus to read, when set. `--raw` sets it before the pool starts: on Windows
#: the pool spawns rather than forks, so a worker re-imports this module and knows only what the
#: environment tells it.
RETARGET_CORPUS_ENV = "ADDBIO_RETARGET_CORPUS"
#: One directory per lineage, no version suffix (retention rule 1.1,
#: docs/guides/RETENTION_RULES.md).
#: The corpus at this name must be a wrap-repaired retarget.
RETARGET_LINEAGE = "addbio_smpl24_raw"
CONTRACT_ID = "soma_paired_small_large_v3_POC_FAITHFUL_ADDBIO_SYNTH"
CONTRACT_VERSION = "0.0.1-poc-faithful-addbio-synth"

UNAVAILABLE = (
    "development_reference / measured GRF / CoP (AddBiomechanics carries ground reaction data, "
    "but it is not converted into this bundle)",
    "measured IMU (AddBiomechanics has none; every channel here is synthesized)",
    "measured subject height and body mass (the retarget corpus does not carry them forward; "
    "height is an SMPL stature estimate and mass is NaN)",
    "source rotation for neck, head, collars and hands -- see small_sites_provenance",
    "sensor mount extrinsics (segment->sensor lever arm)",
    "joint_angles_jcs_deg (needs ISB/JCS convention)",
    "quality gates",
)


def retarget_corpus() -> pathlib.Path:
    """The retarget corpus: `ADDBIO_RETARGET_CORPUS` when set (`--raw` sets it), else the lineage
    directory under SOMA_DATA_ROOT. Resolved when asked, so importing this module reads nothing."""
    override = os.environ.get(RETARGET_CORPUS_ENV)
    return pathlib.Path(override) if override else paths.lineage_dir(RETARGET_LINEAGE)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def site_provenance(joint_provenance) -> dict[str, str]:
    """Per site: how the channel was built, and whether the source drives it.

    Reuses the synth generator's reading of `joint_provenance` so the two corpora cannot label the
    same take differently.
    """
    sources = SYNTH.site_sources(joint_provenance)
    out = {}
    for index, code in enumerate(EMIT.FAITHFUL.SENSOR_CODES):
        joint = EMIT.FAITHFUL.SITE_GEOM[code][0]
        orientation = str(sources["site_orientation_source"][index])
        position = str(sources["site_position_source"][index])
        backed = bool(sources["site_source_backed"][index])
        detail = ("synthetic_fk_specS:back_T4=spine3_ori+chest_axial_proxy(pelvis+2/3(head-pelvis))"
                  if code == "back_T4" else
                  f"synthetic_fk_specS:smpl_segment_ori(joint {joint})+"
                  f"doublediff_world_accel(joint {EMIT.FAITHFUL.SITE_POS_JOINT[code]})")
        warning = "" if backed else (
            " -- NOT SOURCE-BACKED: this joint has no source rotation, so it sits at its rest "
            "rotation and the segment is welded to its nearest driven ancestor. The channel "
            "carries none of the motion the site is named for."
        )
        out[code] = (f"{detail}; source_orientation={orientation}; source_position={position}; "
                     f"source_backed={backed}{warning}")
    return out


def identity_for(take: pathlib.Path, meta: dict) -> "EMIT.SourceIdentity":
    relative = paths.data_relative(take)
    study, subject, stem = meta["study"], meta["subject"], take.stem
    return EMIT.SourceIdentity(
        source_name="addbiomechanics",
        dataset=study,
        subject_id=f"addbio_{study}_{subject}",
        sequence_id=meta["trial_name"] or stem,
        relative_path=relative,
        asset_sha256=sha256_file(take),
        native_frames=int(meta["frames"]),
        native_rate_hz=float(meta["source_rate_hz"]),
        framerate_source="source_derived:b3d trial timestamps",
        source_note=(
            f"AddBiomechanics Dataset 1.0 Core, study token {meta['source_token']}, "
            f"variant {meta['variant']}, {meta['pass_used']} pass; OpenSim kinematics retargeted "
            "onto SMPL-24 by the OpenSim-to-SMPL retarget. Licence "
            f"{meta['license_id']}. AddBiomechanics remains ADDBIOMECHANICS_SOURCE_HOLD: this "
            "bundle is experimental_non_candidate and moves no gate."
        ),
        field_registry="b3d_field_semantics_v3 (activation_state=non_authorizing)",
        frame_convention="AddBiomechanics world rotated to right-handed Z-up, gravity along -Z "
                         "(the source OpenSim frame is Y-up; the retarget applied the 90 deg turn "
                         "about X recorded in frame_rotation_source_to_world)",
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        run_id=f"addbio-{study}-{subject}-{stem}-unified8",
        mount_prefix="addbio_unified8",
        artifact_label="PoC FAITHFUL AddBiomechanics synthetic reference (SMPL-derived synthetic "
                       "IMU from the OpenSim->SMPL retarget). NOT contract-compliant, "
                       "NOT quality-gate PASS. INTERNAL-ONLY.",
        anthro_namespace="anthro_reference/addbio_smpl_v1",
        subject_scope="subject_constant_trial_frame_invariant",
        subject_scope_note="betas are fit once per subject over sampled frames of that subject's "
                           "trials; the frozen-joint constants are fitted once per subject over all "
                           "of its trials (smpl18.reduce) and shared the same way.",
        height_m=None,
        height_provenance="estimated:smpl_rest_stature_yup (the retarget corpus does not carry the "
                          "b3d subject height forward)",
        body_mass_kg=None,
        body_mass_provenance="unavailable:not carried forward by the retarget corpus",
        betas_provenance="source_derived:segment lengths from OpenSim joint centres, fit against "
                         "the clean SMPL model (10 shape dims)",
        # compute_anthro stamps PRISM's string on every source; this generator's sex is the b3d
        # biological_sex the retarget carried forward as 'gender' (unknown -> the neutral model).
        sex_provenance="source_derived:addbiomechanics b3d biological_sex, carried by the retarget "
                       "corpus as 'gender' (unknown -> neutral)",
        smpl_trans_provenance="source_derived:pelvis position from the retarget, no windowing",
        small_content_note="8ch synthetic IMU (back_T4, wrist_l/r, shank_l/r, occiput, foot_l/r) "
                           "fully FK-derived in the spec sensor frame S; specific force "
                           "(gravity-included) + rotation-log gyro. small_mode="
                           "synthetic_from_smpl, axis_convention=spec_S_v2. FOUR OF THE EIGHT "
                           "SITES ARE NOT DRIVEN BY THE SOURCE -- see small_sites_provenance.",
        unavailable=UNAVAILABLE,
        site_provenance=meta["site_provenance"],
        extra_manifest={
            "source_attribution": {
                "dataset_citation": meta["dataset_citation"],
                "publication": meta["publication"],
                "license_id": meta["license_id"],
                "stable_source_id": meta["stable_source_id"],
                "changes_made": meta["changes_made"],
                "retarget_content_hashes": meta["content_hashes"],
            },
            "wrap_repaired_coordinates": meta["wrap_repaired_coordinates"],
            "source_hold": "ADDBIOMECHANICS_SOURCE_HOLD (NATIVE_AUDIT_PASS is not generation "
                           "authorization); this bundle moves no gate and is not registered in the "
                           "experimental catalog.",
        },
    )


@functools.lru_cache(maxsize=1)
def reduction_settings():
    """The addbiomechanics smpl18 profile's reduce settings, loaded once per process."""
    return REDUCED.settings_for_source("addbiomechanics")


def take_dirname(study: str, subject: str, take: pathlib.Path) -> str:
    return f"addbio_{study}_{subject}_{take.stem}"


def fit_subject(study: str, subject: str, takes: list[pathlib.Path], models: dict):
    """The subject's frozen-joint fit over all of its trials (`CorpusFits`).

    The rest skeleton is the one `build_bundle` derives, the gendered clean model at the trial's
    betas; a trial that cannot be read is left out here and fails on its own in `convert`."""
    members = []
    for take in takes:
        try:
            with np.load(take, allow_pickle=False) as z:
                gender = str(z["gender"])
                betas = np.asarray(z["betas"], np.float64)
                pose = np.asarray(z["pose"], np.float64)
        except Exception:                                       # noqa: BLE001
            continue
        if gender not in models:
            models[gender] = load_clean_model(clean_model_path_for_gender(gender))
        members.append((take_dirname(study, subject, take), pose,
                        EMIT.ANTHRO.rest_joints(models[gender], betas)))
    fits = REDUCED.CorpusFits(reduction_settings())
    if members:
        fits.fit(f"addbio_{study}_{subject}", members,
                 basis="every readable trial of the subject, native rate")
    return fits


def convert(take: pathlib.Path, study: str, subject: str, out_root: pathlib.Path,
            models: dict, reduction=None) -> dict:
    with np.load(take, allow_pickle=False) as z:
        meta = {
            "study": study, "subject": subject,
            "trial_name": str(z["trial_name"]),
            "variant": str(z["variant"]), "pass_used": str(z["pass_used"]),
            # older retarget corpora predate the field; absent is not the same as empty, and
            # a bundle must not report "nothing was repaired" about a take nobody checked
            "wrap_repaired_coordinates": (
                str(z["wrap_repaired_coordinates"])
                if "wrap_repaired_coordinates" in z.files
                else "not_recorded"
            ),
            "source_token": str(z["source_token"]),
            "stable_source_id": str(z["stable_source_id"]),
            "dataset_citation": str(z["dataset_citation"]),
            "publication": str(z["publication"]), "license_id": str(z["license_id"]),
            "changes_made": str(z["changes_made"]),
            "content_hashes": str(z["content_hashes"]),
            "source_rate_hz": float(z["source_rate_hz"]),
            "frames": int(z["pose"].shape[0]),
            "site_provenance": site_provenance(z["joint_provenance"]),
        }
        gender = str(z["gender"])
        betas = np.asarray(z["betas"], np.float64)
        pose = np.asarray(z["pose"], np.float64)
        trans = np.asarray(z["trans"], np.float64)

    if gender not in models:
        models[gender] = load_clean_model(clean_model_path_for_gender(gender))
    source = identity_for(take, meta)
    bundle = EMIT.build_bundle(pose24=pose, trans=trans, betas=betas, gender=gender,
                               model=models[gender], src_fps=meta["source_rate_hz"], source=source,
                               reduction=reduction)
    take_id = take_dirname(study, subject, take)
    EMIT.write_bundle(out_root / take_id, bundle)
    return {"take_id": take_id, "rel": take_id, "status": "ok",
            "frames": int(bundle["small"]["frame_count"]),
            "pair_id": bundle["manifest"]["identity"]["pair_id"],
            "subject_id": source.subject_id, "sequence_id": source.sequence_id}


def convert_subject(study: str, subject: str, out_root: str, trials: int | None,
                    force: bool) -> dict:
    """{"entries": INDEX entries, "groups": fit records, "takes": take -> fit group}.

    The fit covers every trial of the subject, resumed ones included, so a resumed subject's
    record is the same one the takes were written with."""
    root = pathlib.Path(out_root)
    takes = sorted((retarget_corpus() / study / subject).glob("*.npz"))
    if trials:
        takes = takes[:trials]
    models: dict[str, dict] = {}
    fits = fit_subject(study, subject, takes, models)
    entries = []
    for take in takes:
        take_id = take_dirname(study, subject, take)
        if not force and (root / take_id / "manifest.json").exists():
            entries.append({"take_id": take_id, "rel": take_id, "status": "ok",
                            "resumed": True})
            continue
        try:
            entries.append(convert(take, study, subject, root, models,
                                   reduction=fits.by_take.get(take_id)))
        except Exception as error:                              # noqa: BLE001
            entries.append({"take_id": take_id, "rel": take_id, "status": "failed",
                            "reason": f"{type(error).__name__}: {error}"})
    return {"entries": entries, "groups": fits.groups, "takes": fits.takes}


def _worker(args) -> dict:
    study, subject, out_root, trials, force = args
    try:
        return convert_subject(study, subject, out_root, trials, force)
    except Exception as error:                                  # noqa: BLE001
        return {"entries": [{"take_id": f"addbio_{study}_{subject}",
                             "rel": f"addbio_{study}_{subject}", "status": "failed",
                             "reason": f"{type(error).__name__}: {error}"}],
                "groups": {}, "takes": {}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--studies", type=int, default=None)
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--raw", default=None,
                        help=f"retarget corpus to read; default ${RETARGET_CORPUS_ENV}, else "
                             f"$SOMA_DATA_ROOT/{paths.POC_DEMO}/{RETARGET_LINEAGE}")
    parser.add_argument("--only", action="append", type=SYNTH.study_subject, default=None,
                        metavar="STUDY/SUBJECT",
                        help="emit only this subject of the corpus (repeatable); the subject's "
                             "frozen-joint fit still covers all of its trials")
    args = parser.parse_args(argv)

    if args.raw:
        os.environ[RETARGET_CORPUS_ENV] = args.raw   # before the pool spawns, so workers see it
    try:
        raw = retarget_corpus()
        paths.body_model_dir()     # checked here: a worker would record it as a failed subject
        out_root = paths.check_output_dir(pathlib.Path(args.out), inputs=[raw])
        # every take's relative_path is written against the data root, so a corpus outside it
        # would fail each take on its own and still leave an INDEX over zero takes
        paths.data_relative(raw)
    except paths.PathConfigError as error:
        print(error)
        return 2
    if not raw.is_dir():
        print("retarget corpus not present:", raw)
        return 2
    print(f"reading {raw}", flush=True)

    try:
        studies, pairs = SYNTH.enumerate_subjects(raw, args.studies, args.subjects, args.only)
        # an empty selection would still write INDEX.json, the fit record and the description
        SYNTH.require_trials(raw, pairs, args.trials, SYNTH.selection_caps(args))
    except ValueError as error:
        print(error)
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    work = [(study, s, str(out_root), args.trials, args.force) for study, s in pairs]

    print(f"{len(work)} subjects across {len(studies)} studies -> {out_root}", flush=True)
    started = time.time()
    entries: list[dict] = []
    fits = REDUCED.CorpusFits(reduction_settings())

    def collect(result: dict) -> None:
        entries.extend(result["entries"])
        fits.add_records(result["groups"], result["takes"])

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_worker, item) for item in work]
            for done, future in enumerate(as_completed(futures), 1):
                collect(future.result())
                if done % 25 == 0 or done == len(work):
                    rate = (time.time() - started) / done
                    print(f"  [{done}/{len(work)}] {len(entries)} takes "
                          f"({rate:.1f}s/subject, ~{rate * (len(work) - done) / 60:.0f} min left)",
                          flush=True)
    else:
        for done, item in enumerate(work, 1):
            collect(_worker(item))
            if done % 25 == 0 or done == len(work):
                print(f"  [{done}/{len(work)}] {len(entries)} takes", flush=True)

    # The pool returns subjects in completion order; the INDEX is sorted so that two runs of the
    # same corpus write the same file and the first entry the validator samples is always the
    # same take.
    entries.sort(key=lambda e: e["take_id"])

    # Pose-discontinuity scan on the bundle's own large artifact, applied while the INDEX is
    # written: a take whose retarget steps a joint past the spec's gyro band, or whose trunk is
    # inverted for the whole take, is excluded here with the measurement as the reason. Excluded
    # takes stay on disk and in the INDEX. A resumed take (`resumed: True`) was scanned by the run
    # that wrote it; its verdict is re-derived here rather than trusted, since the scan is cheap
    # next to the emission.
    scanned = 0
    scan_started = time.time()
    for entry in entries:
        if entry["status"] != "ok":
            continue
        take_dir = out_root / entry["rel"]
        if not (take_dir / "large_reference.npz").exists():
            continue
        verdict = SCAN.scan_take(take_dir, SCAN.GYRO_ABS_MAX_DEG_S, SCAN.TRUNK_LIMIT_DEG)
        scanned += 1
        if verdict["problems"]:
            entry["status"] = "excluded"
            entry["reason"] = "; ".join(verdict["problems"])
            entry["excluded_by"] = SCAN.MARKER
        if scanned % 2000 == 0:
            rate = scanned / max(time.time() - scan_started, 1e-9)
            print(f"  scanned {scanned} takes ({rate:.0f}/s)", flush=True)

    ok = [e for e in entries if e["status"] == "ok"]
    failed = [e for e in entries if e["status"] == "failed"]
    excluded = [e for e in entries if e["status"] == "excluded"]
    print(f"  scan: {scanned} takes, {len(excluded)} excluded", flush=True)
    index = {
        "spec_id": EMIT.SPEC_ID, "spec_version": EMIT.SPEC_VERSION,
        "dataset_dirname": out_root.name,
        "source": "addbiomechanics",
        "artifact_class": "experimental_non_candidate",
        "distribution_scope": "internal_only",
        "generated_utc": EMIT.datetime.now(EMIT.timezone.utc).isoformat(),
        "counts": {"total": len(entries), "ok": len(ok),
                   "failed": len(failed), "excluded": len(excluded)},
        "complete": not failed,
        "takes": entries,
    }
    (out_root / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                         encoding="utf-8", newline="\n")
    fits.write(out_root)
    EMIT.write_data_description(
        out_root, index,
        title="AddBiomechanics - synthetic IMU reference (qmd_unified8_smpl18)",
        body=(
            "## Read this before using the eight channels\n\n"
            "**Four of the eight sites are not driven by the source.** AddBiomechanics' OpenSim "
            "models carry no neck, head or hand coordinates, and three fifths of the corpus is the "
            "`No_Arm` variant:\n\n"
            "| site | driving SMPL joint | source |\n"
            "| --- | --- | --- |\n"
            "| `shank_l/r`, `foot_l/r` | 4/5, 10/11 | measured |\n"
            "| `back_T4` | 9 (spine3) | derived - the lumbar rotation split in equal thirds |\n"
            "| `wrist_l/r` | 18/19 | measured in `With_Arm`, **absent** in `No_Arm` |\n"
            "| `occiput` | 15 (head) | **absent everywhere** |\n\n"
            "An unsourced joint sits at its rest rotation, so the segment is welded to its nearest "
            "driven ancestor: the channel looks like an IMU trace and contains none of the motion "
            "the site is named for. Each take's `manifest.json` states this per site under "
            "`small_sites_provenance`, because it differs between the arm variants. Measured on 60 "
            "takes of the sibling corpus, welded sites move 4-9e-06 deg relative to `back_T4` in "
            "its own frame while driven sites move 0.66-109 deg.\n\n"
            "## Other limits\n\n"
            "- Source rates run 61-250 Hz against a 100 Hz target. One study, `Tiziana2019`, is "
            "**upsampled** (382 of 40,484 trials): those takes say 100 Hz and carry 30 Hz of "
            "bandwidth. Each manifest's `resample.direction` and `source.native_rate_hz` say which.\n"
            "- Subject height is an SMPL stature estimate and body mass is NaN: the retarget corpus "
            "does not carry the b3d subject anthropometry forward.\n"
            f"- {len(excluded)} of {len(entries)} takes carry `status: excluded` in `INDEX.json` "
            "(the retarget stepped a joint past the spec's 8000 deg/s band, or the trunk is "
            "inverted for the whole take); select takes by `status == \"ok\"`, never by "
            "directory listing.\n"
            "- The machine-readable list of what this bundle cannot promise is "
            "`KNOWN_LIMITATIONS.json` at the bundle root.\n"
            "- AddBiomechanics is under `ADDBIOMECHANICS_SOURCE_HOLD`. This bundle is "
            "`experimental_non_candidate` and moves no gate.\n\n"
            "Full record: the SOMA project's AddBio synthetic-IMU conversion record (2026-08-30).\n"
        ),
    )
    print(f"\n{len(ok)}/{len(entries)} takes ok, {len(excluded)} excluded, {len(failed)} failed "
          f"in {time.time() - started:.0f}s -> {out_root}", flush=True)
    return 0 if index["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
