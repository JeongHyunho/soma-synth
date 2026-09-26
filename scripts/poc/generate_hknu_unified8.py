"""HKNU -> a `qmd_unified8_smpl18` dataset the generator-independent validator can check.

`hknu_smpl24_paired` already carries the retargeted SMPL motion, the eight synthetic channels and
the measured IMU beside them, but under its own spec_id -- so the only check it has ever had is the
one written by the same hand as the data. This re-emits the same motion in the spec's five-file
bundle so the L0-L4 validator can have a look.

The motion is read from the paired corpus rather than retargeted again. That keeps the two in step
by construction: if the bundle disagreed with the corpus, the disagreement would be in this script,
and the audit report describing that corpus would silently stop applying.

Three things the emitted bundle says that earlier generations had to be repaired to say:
`source_attribution` is read from `configs/datasets/source_attribution_v1.yaml` at emit time
rather than backfilled afterwards; `sex_provenance` names the workbook column the sex was read
from rather than a PRISM constant; and the takes the pose-discontinuity scan rejects are marked
`excluded` in the INDEX as it is written, so the description beside it never contradicts it.

`small_mode` is `synthetic_from_smpl`, not `measured_physical`, even though HKNU has worn sensors on
all eight sites. `measured_physical` requires `q_anatomical_from_sensor` and `p_segment_to_sensor_m`
-- the mounting -- and this audit never resolved the mounting; it went out of its way to compare on
rotation-invariant magnitudes precisely so it would not have to. Claiming the mode would be claiming
the extrinsics. The measured channel stays in the paired corpus, and the manifest says where.

Usage:
    python scripts/poc/generate_hknu_unified8.py --out <dir> [--subjects S01 S02] [--trials N]
                                                 [--paired <corpus>] [--hknu-root <source>]

`--paired` defaults to `$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/hknu_smpl24_paired`
and `--hknu-root` (the HKNU source folder whose `MATLAB/DatasetInfo.xlsx` gives the measured
height and mass) to `$SOMA_SOURCE_ROOT/hknu_fullbody`. A missing input folder, a `--subjects` name
the corpus does not hold (case-sensitive) and a selection of no take stop the run before the
output directory is created.
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
# The same scan `scripts/diagnostics/pose_discontinuity_scan.py --apply` runs by hand, applied
# while the INDEX is being written, so an excluded take is excluded from the first version of
# the bundle rather than in a second pass someone has to remember.
SCAN = _load(_HERE.parents[1] / "diagnostics" / "pose_discontinuity_scan.py",
             "pose_discontinuity_scan")

from soma_synth.addbio_retarget.shape_fit import (  # noqa: E402
    clean_model_path_for_gender,
    load_clean_model,
)
from soma_synth.contracts import source_attribution as ATTRIBUTION  # noqa: E402
from soma_synth.pipeline import paths

# No version suffix: one directory per lineage (retention rule 1.1, docs/guides/RETENTION_RULES.md).
# The corpus is the 2026-09-08 repaired-fit retarget.
PAIRED_LINEAGE = "hknu_smpl24_paired"
CONTRACT_ID = "soma_paired_small_large_v3_POC_FAITHFUL_HKNU_SYNTH"
CONTRACT_VERSION = "0.0.1-poc-faithful-hknu-synth"
#: The workbook column `subjects_from_workbook` reads the sex from. Named here so the anthro
#: array says where its value came from instead of the PRISM string the shared emitter would
#: otherwise stamp (known limitation hknu.sex_provenance_mislabelled, fixed by this).
SEX_PROVENANCE = "measured:DatasetInfo.xlsx Gender"


@functools.lru_cache(maxsize=1)
def attribution() -> dict[str, str]:
    """`manifest.source_attribution`, read from the registry rather than typed here.

    A rights string is a config value, never a code constant; the registry refuses to hand one out
    while the source's interpretation is still pending, which is exactly when a generator must
    not write it.
    """
    return ATTRIBUTION.resolved_for("hknu")

def site_provenance(corpus: str) -> dict[str, str]:
    """Where each Small channel came from. `corpus` names the paired corpus this was published from.

    The name used to be a literal, which was true while there was one corpus. There are two now
    and the difference between them is how the body was fitted, so a bundle that names the wrong
    one sends anyone tracing a channel to a different body.
    """
    return {
        "back_T4": "synthetic_fk_specS:spine3_ori+chest_axial_proxy(pelvis+2/3(head-pelvis)); "
                   f"worn counterpart TA in {corpus}",
        "wrist_l": "synthetic_fk_specS:smpl_segment_ori(joint 18)+doublediff_world_accel(joint 20); "
                   "worn counterpart LFA",
        "wrist_r": "synthetic_fk_specS:smpl_segment_ori(joint 19)+doublediff_world_accel(joint 21); "
                   "worn counterpart RFA",
        "shank_l": "synthetic_fk_specS:smpl_segment_ori(joint 4)+doublediff_world_accel(joint 4); "
                   "worn counterpart LSK, which sits 128-161 mm distal of the knee centre this "
                   "site differentiates",
        "shank_r": "synthetic_fk_specS:smpl_segment_ori(joint 5)+doublediff_world_accel(joint 5); "
                   "worn counterpart RSK, same distal offset",
        "occiput": "synthetic_fk_specS:smpl_segment_ori(joint 15)+doublediff_world_accel(joint 15); "
                   "head placed from the N-pose reference (no head DOF is fitted from motion); "
                   "worn counterpart HE",
        "foot_l": "synthetic_fk_specS:smpl_segment_ori(joint 10)+doublediff_world_accel(joint 10); "
                  "worn counterpart LFT",
        "foot_r": "synthetic_fk_specS:smpl_segment_ori(joint 11)+doublediff_world_accel(joint 11); "
                  "worn counterpart RFT",
    }


def unavailable(corpus: str) -> tuple[str, ...]:
    """What this bundle does not carry, and why. `corpus` is where the measured IMU actually lives."""
    return (
        "development_reference / measured GRF / CoP (HKNU has force plates and insoles, but they "
        "are not carried into this bundle)",
        f"measured IMU (present in the source and in {corpus}; not carried here because "
        "small_mode=synthetic_from_smpl and the mounting extrinsics were never resolved)",
        "sensor mount extrinsics (segment->sensor lever arm)",
        "joint_angles_jcs_deg (needs ISB/JCS convention)",
        "physics-informed GRF / inverse dynamics / moments / powers / COM",
        "quality gates",
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_paired() -> pathlib.Path:
    """The paired corpus `--paired` defaults to: the lineage directory under SOMA_DATA_ROOT."""
    return paths.lineage_dir(PAIRED_LINEAGE)


def subject_anthropometry(hknu_root: pathlib.Path | None = None) -> dict[str, dict]:
    """Measured height and mass from the dataset workbook under `hknu_root`
    (`MATLAB/DatasetInfo.xlsx`; default the HKNU source folder, `paths.source_dir("hknu")`).

    Worth the extra read: HKNU is one of the few sources here with real subject anthropometry, and
    writing an SMPL stature estimate when a measured height exists would throw it away.
    """
    root = paths.source_dir("hknu") if hknu_root is None else pathlib.Path(hknu_root)
    generator = _load(_HERE.parent / "generate_hknu_faithful.py", "generate_hknu_faithful")
    return generator.subjects_from_workbook(root)


def corpus_root(paired: pathlib.Path) -> pathlib.Path:
    """The data root the corpus sits under, so `relative_path` reads the same on any drive.

    `relative_path` is provenance, not identity -- `pair_id` hashes the asset digest and not this
    -- but it still has to name the take the same way wherever the corpus was built, and a
    corpus built on another drive should not produce a different provenance string.
    """
    resolved = paired.resolve()
    for parent in resolved.parents:
        if parent.name == "runs":
            return parent.parent
    return resolved.parent


def identity_for(take: pathlib.Path, subject: str, trial: str, frames: int,
                 rate: float, info: dict, root: pathlib.Path | None = None,
                 paired: pathlib.Path | None = None) -> "EMIT.SourceIdentity":
    """`root` defaults to the data root and `paired` to `default_paired()`; `main` passes both."""
    root = paths.data_root() if root is None else root
    paired = default_paired() if paired is None else paired
    relative = str(take.resolve().relative_to(root)).replace("\\", "/")
    corpus = paired.resolve().name
    corpus_relative = paired.resolve().relative_to(corpus_root(paired)).as_posix()
    return EMIT.SourceIdentity(
        source_name="hknu",
        dataset="hknu_fullbody",
        subject_id=f"hknu_{subject}",
        sequence_id=trial,
        relative_path=relative,
        asset_sha256=sha256_file(take),
        native_frames=frames,
        native_rate_hz=rate,
        framerate_source="source_header:Info.SamplingRate",
        source_note="HKNU FullBody: optical mocap retargeted onto SMPL-24, then "
                    "synthesized. The source also carries a worn IMU on all eight sites; it is not "
                    f"in this bundle (see unavailable_or_not_applied) and lives in {corpus}, "
                    "where the two were compared.",
        field_registry="none (HKNU has no field registry in this project; the retarget "
                       "correspondence is recorded in the audit report)",
        frame_convention="HKNU lab world, right-handed, Z-up, gravity along -Z",
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        run_id=f"hknu-{subject}-{trial}-unified8",
        mount_prefix="hknu_unified8",
        artifact_label="PoC FAITHFUL HKNU synthetic reference (SMPL-derived synthetic IMU from the "
                       "Visual3D-to-SMPL retarget). NOT contract-compliant, NOT quality-gate PASS. "
                       "INTERNAL-ONLY.",
        anthro_namespace="anthro_reference/hknu_smpl_v1",
        subject_scope="subject_constant_trial_frame_invariant",
        subject_scope_note="betas are fit once per subject from the N-pose joint centres and shared "
                           "across that subject's trials; the frozen-joint constants are fitted once "
                           "per subject over all of its trials (smpl18.reduce) and shared the same way.",
        height_m=float(info["height_m"]),
        height_provenance="measured:DatasetInfo.xlsx BodyHeight",
        body_mass_kg=float(info["mass_kg"]),
        body_mass_provenance="measured:DatasetInfo.xlsx BodyMass",
        betas_provenance="source_derived:segment lengths from the N-pose joint centres, fit against "
                         "the clean SMPL model with measured stature and mass as constraints",
        smpl_trans_provenance="source_derived:pelvis position from the retarget (Visual3D PV "
                              "ProxEndPos chain), no windowing",
        small_content_note="8ch synthetic IMU (back_T4, wrist_l/r, shank_l/r, occiput, foot_l/r) "
                           "fully FK-derived in the spec sensor frame S; specific force "
                           "(gravity-included) + rotation-log gyro. small_mode="
                           "synthetic_from_smpl, axis_convention=spec_S_v2.",
        unavailable=unavailable(corpus),
        site_provenance=site_provenance(corpus),
        sex_provenance=SEX_PROVENANCE,
        extra_manifest={
            "paired_measured_counterpart": {
                "dataset": corpus,
                "relative_path": corpus_relative,
                "note": "same motion with the worn IMU beside the synthetic channel; agreement was "
                        "measured on |f| and |omega| there, median |omega| Pearson r 0.908-0.970 "
                        "across the eight sites.",
            },
            "source_attribution": dict(attribution()),
        },
    )


def take_dirname(subject: str, trial: str) -> str:
    return f"hknu_{subject}_{trial}"


def fit_members(take: pathlib.Path) -> tuple[str, str, np.ndarray, np.ndarray]:
    """(subject, take directory, pose, rest joints) of one corpus take, for the subject's fit."""
    with np.load(take, allow_pickle=False) as z:
        subject = str(z["subject"])
        return (subject, take_dirname(subject, str(z["trial"])),
                np.asarray(z["pose"], np.float64), np.asarray(z["rest_joints"], np.float64))


def convert(take: pathlib.Path, out_root: pathlib.Path, models: dict,
            anthropometry: dict, root: pathlib.Path | None = None,
            paired: pathlib.Path | None = None, reduction=None) -> dict:
    with np.load(take, allow_pickle=False) as z:
        subject = str(z["subject"])
        trial = str(z["trial"])
        gender = str(z["gender"])
        betas = np.asarray(z["betas"], np.float64)
        pose = np.asarray(z["pose"], np.float64)
        trans = np.asarray(z["trans"], np.float64)
        rest = np.asarray(z["rest_joints"], np.float64)
        rate = float(z["sampling_rate_hz"])

    if gender not in models:
        models[gender] = load_clean_model(clean_model_path_for_gender(gender))
    source = identity_for(take, subject, trial, pose.shape[0], rate,
                          anthropometry[subject], root, paired)
    # the retarget's own skeleton, not one recomputed from the betas: it is scaled onto the
    # subject's measured bone lengths and the pose was fitted against that
    bundle = EMIT.build_bundle(pose24=pose, trans=trans, betas=betas, gender=gender,
                               model=models[gender], src_fps=rate, source=source,
                               rest_joints=rest, reduction=reduction)
    out_dir = out_root / take_dirname(subject, trial)
    EMIT.write_bundle(out_dir, bundle)
    return {"take_id": out_dir.name, "rel": out_dir.name, "status": "ok",
            "frames": int(bundle["small"]["frame_count"]),
            "pair_id": bundle["manifest"]["identity"]["pair_id"],
            "subject_id": source.subject_id, "sequence_id": trial}


def select_subjects(paired: pathlib.Path, wanted=None, trials: int | None = None) -> list[str]:
    """The paired corpus's subject folders a run converts: all of them, or those ``wanted`` names.

    Names are matched exactly against the folder listing, case included (on NTFS the joined path
    exists whatever its case, while this selection compares strings). A name the corpus lacks, and
    a selection that holds no take (``<subject>/*.npz``, after the ``--trials`` cap), raise
    ValueError: the run would write INDEX.json, the fit record and the description over zero takes
    and exit 0."""
    subjects = sorted(p.name for p in paired.iterdir() if p.is_dir())
    if wanted:
        unknown = sorted(set(wanted) - set(subjects))
        if unknown:
            raise ValueError(
                f"--subjects {' '.join(unknown)}: not a subject of the paired corpus {paired} "
                f"(names are case-sensitive; subjects there: {', '.join(subjects) or 'none'})")
        subjects = [s for s in subjects if s in set(wanted)]

    def selected(subject: str) -> list[pathlib.Path]:
        takes = sorted((paired / subject).glob("*.npz"))
        return takes[:trials] if trials else takes

    if not any(selected(subject) for subject in subjects):
        chosen = [f"--subjects {' '.join(wanted)}" if wanted else "",
                  f"--trials {trials}" if trials else ""]
        caps = " ".join(c for c in chosen if c)
        raise ValueError(f"no take (<subject>/*.npz) selected in the paired corpus {paired}"
                         + (f" by {caps}" if caps else "") + "; nothing written")
    return subjects


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    # The corpus to read. There is more than one candidate corpus and they differ in content, so
    # the choice is an explicit argument rather than a constant.
    parser.add_argument("--paired", type=pathlib.Path, default=None,
                        help="the hknu_smpl24_paired corpus to publish (default "
                             f"$SOMA_DATA_ROOT/{paths.POC_DEMO}/{PAIRED_LINEAGE})")
    parser.add_argument("--hknu-root", type=pathlib.Path, default=None,
                        help="the HKNU source folder holding MATLAB/DatasetInfo.xlsx (default "
                             "$SOMA_SOURCE_ROOT/hknu_fullbody)")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args(argv)
    paired_given = args.paired is not None
    hknu_root_given = args.hknu_root is not None
    try:
        if args.paired is None:
            args.paired = default_paired()
        if args.hknu_root is None:
            args.hknu_root = paths.source_dir("hknu")
        paths.body_model_dir()     # the models each subject's gender selects live here
        out_root = paths.check_output_dir(pathlib.Path(args.out),
                                          inputs=[args.paired, args.hknu_root])
        # the corpus is listed and globbed, and a missing folder lists as empty
        if not args.paired.is_dir():
            named = ("as given" if paired_given
                     else f"{paths.DATA_ROOT_ENV}/{paths.POC_DEMO}/{PAIRED_LINEAGE}")
            raise paths.PathConfigError(f"the paired corpus {args.paired} ({named}) does not "
                                        "exist or is not a directory")
        paths.existing_source_dir("hknu", args.hknu_root if hknu_root_given else None)
        subjects = select_subjects(args.paired, args.subjects, args.trials)
    except (paths.PathConfigError, ValueError) as error:
        print(error)
        return 2
    anthropometry = subject_anthropometry(args.hknu_root)
    models: dict[str, dict] = {}
    out_root.mkdir(parents=True, exist_ok=True)

    paired = args.paired
    root = corpus_root(paired)

    entries: list[dict] = []
    fits = REDUCED.CorpusFits(REDUCED.settings_for_source("hknu"))
    started = time.time()
    for subject in subjects:
        takes = sorted((paired / subject).glob("*.npz"))
        if args.trials:
            takes = takes[: args.trials]
        # One set of frozen-joint constants per subject, fitted over every trial before any is
        # emitted, on the corpus's own (bone-length-scaled) skeleton.
        members = [fit_members(take) for take in takes]
        fits.fit(f"hknu_{subject}", [(rel, pose, rest) for _s, rel, pose, rest in members],
                 basis="every trial of the subject, native rate")
        for take, (_s, rel, _p, _r) in zip(takes, members):
            try:
                entries.append(convert(take, out_root, models, anthropometry, root, paired,
                                       reduction=fits.by_take[rel]))
            except Exception as error:                          # noqa: BLE001
                entries.append({"take_id": take.stem, "rel": f"hknu_{take.stem}",
                                "status": "failed",
                                "reason": f"{type(error).__name__}: {error}"})
                print(f"  FAILED {take.name}: {error}", flush=True)
        print(f"  {subject}: {len(entries)} takes so far "
              f"({time.time() - started:.0f}s)", flush=True)

    # The pose-discontinuity scan, on the bundle's own large artifact: a take whose retarget
    # steps a joint past the spec's gyro band, or whose trunk is inverted for the whole take,
    # is excluded in this INDEX, not in a second pass. Excluded takes stay on disk and in the
    # INDEX with the measurement as the reason, so nobody has to guess what was dropped or why.
    for entry in entries:
        if entry["status"] != "ok":
            continue
        verdict = SCAN.scan_take(out_root / entry["rel"], SCAN.GYRO_ABS_MAX_DEG_S,
                                 SCAN.TRUNK_LIMIT_DEG)
        if verdict["problems"]:
            entry["status"] = "excluded"
            entry["reason"] = "; ".join(verdict["problems"])
            entry["excluded_by"] = SCAN.MARKER
            print(f"  EXCLUDED {entry['take_id']}: {entry['reason']}", flush=True)

    ok = [e for e in entries if e["status"] == "ok"]
    failed = [e for e in entries if e["status"] == "failed"]
    excluded = [e for e in entries if e["status"] == "excluded"]
    index = {
        "spec_id": EMIT.SPEC_ID, "spec_version": EMIT.SPEC_VERSION,
        "dataset_dirname": out_root.name,
        "source": "hknu",
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
        title="HKNU FullBody - synthetic IMU reference (qmd_unified8_smpl18)",
        body=(
            "## What this is, and what it is not\n\n"
            "Optical mocap retargeted onto SMPL-24, then synthesized into the eight Small channels. "
            "`small_mode` is `synthetic_from_smpl`: every channel here is FK-derived.\n\n"
            "HKNU also recorded a **worn IMU on all eight sites**, and it is deliberately not in "
            "this bundle. Carrying it would mean `small_mode: measured_physical`, which the spec "
            "gates behind `q_anatomical_from_sensor` and `p_segment_to_sensor_m` - the mounting - "
            "and the mounting was never resolved. The measured stream lives beside the synthetic "
            f"one in `{paired.resolve().name}`, where the two were compared on the "
            "rotation-invariant magnitudes |f| and |omega| (median |omega| Pearson r 0.908-0.970 "
            "across the eight sites, over 263 paired trials).\n\n"
            "## Known limits\n\n"
            "- The `shank_l/r` sites differentiate the **knee joint centre**, while the worn sensor "
            "sits 128-161 mm distal of it. The lever arm contributes a term this channel cannot "
            "carry; specific-force agreement falls to 0.475/0.548 there. A limit of the site "
            "definition, not of the retarget.\n"
            "- The head carries no fitted rotation: it is placed from the subject's N-pose "
            "reference, so `occiput` follows the thorax through a fixed offset.\n"
            "- Subject height and mass are **measured** (from the dataset workbook), unlike the "
            "AMASS bundle where height is an SMPL stature estimate and mass is absent.\n"
            f"- {len(excluded)} of {len(entries)} takes carry `status: excluded` in `INDEX.json` "
            "(pose discontinuity or inverted trunk in the source retarget); select takes by "
            "`status == \"ok\"`, never by directory listing.\n"
            "- The machine-readable list of what this bundle cannot promise is "
            "`KNOWN_LIMITATIONS.json` at the bundle root.\n\n"
            "Full audit: the SOMA project's HKNU full-body conversion audit report (2026-08-29); "
            "the body-fit repair this corpus carries: "
            "its HKNU fit diagnosis (2026-09-07).\n"
        ),
    )
    print(f"\n{len(ok)}/{len(entries)} takes ok, {len(excluded)} excluded, {len(failed)} failed "
          f"in {time.time() - started:.0f}s -> {out_root}", flush=True)
    return 0 if index["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
