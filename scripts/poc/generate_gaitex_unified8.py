"""Emit the GAITEX unified8 bundles: large/anthro from the SMPL retarget, small from the markers.

``unified8_emit`` builds large, anthro and the root translation from the retargeted SMPL motion, as
it does for the other three sources, and it also builds an SMPL-derived small. That small is then
REPLACED here. GAITEX carries optical markers on every one of the eight sites, and its worn XSens
units let five of them be checked against a real sensor, so its small is synthesised from the marker
clusters (``soma_synth.gaitex_synthesis.marker_small``) rather than from SMPL forward
kinematics. The emitter is left alone: it is held to byte equality with ``generate_amass_faithful``
by a test, and everything GAITEX decides for itself sits in this wrapper.

Three things a reader of the arrays needs to know, all decided on 2026-09-06 and recorded in
ADR-0039:

* ``small_mode`` is ``synthetic_from_markers`` (ADR-0039). Orientation is a rigid relabel of a body
  fitted to skin markers, not of the SMPL joint ``large`` carries; the two disagree by a few degrees
  on the plates and by tens of degrees on the wrists and head, where the SMPL joints are undriven.
* Small keeps every frame of the retarget span except the ends where no site is valid. Where a site
  has no signal -- a marker gap longer than the gap policy bridges, a filter edge, a withdrawn glitch
  -- its cell is NaN in the three IMU arrays, False in ``imu_valid_mask`` and 0.0 in
  ``imu_confidence``. One thing is bridged: a gap in a cluster's pose of at most the settings'
  ``max_interpolated_gap_frames`` is interpolated before synthesis and ships valid; each manifest
  lists those frames per site. Apply the mask before any filter, derivative or loss. ``large`` is
  finite on every frame.
* Four sites have no registered offset. ``back_T4`` borrows the ``chest`` plate it physically is --
  so that channel is the STERNUM sensor, not a T4 sensor -- and ``occiput``, ``wrist_l``, ``wrist_r``
  sit at their body's own origin as a declared, provisional zero.

    python scripts/poc/generate_gaitex_unified8.py --out <dir> [--subjects a b]

``--retarget`` defaults to ``$SOMA_DATA_ROOT/runs/experimental_generation_poc_demo/gaitex_smpl24`` and
``--extracted`` to ``$SOMA_SOURCE_ROOT/gaitex``. The manifest names the marker and imu files by their
logical id ``extracted/gaitex/...`` wherever ``--extracted`` physically is. A missing input folder, a
``--subjects`` name the corpus does not hold (case-sensitive) and a selection of no take stop the run
before the output directory is created.

INTERNAL-ONLY. Experimental, non-candidate. Lifts no hold and approves no generation: the settings
file it reads says ``authorises_generation: false`` and the manifest repeats that.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util as _ilu
import json
import pathlib
import sys
import time
from types import SimpleNamespace

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

from soma_synth.addbio_retarget.attribution import attribution_fields  # noqa: E402
from soma_synth.addbio_retarget.shape_fit import (  # noqa: E402
    clean_model_path_for_gender,
    load_clean_model,
)
from soma_synth.gaitex_synthesis import marker_small  # noqa: E402
from soma_synth.pipeline import paths
from soma_synth.sensors import site_registry  # noqa: E402
# The native GAITEX reader (kept apart from the governed source-parsing audit code).
from soma_synth.gaitex_retarget import native as gaitex_adapter  # noqa: E402

#: The retarget corpus's lineage directory under $SOMA_DATA_ROOT; no version suffix.
RETARGET_LINEAGE = "gaitex_smpl24"
SOURCE = "gaitex"
SETTINGS = _REPO / "configs" / "datasets" / "gaitex_synthesis_v1_1.yaml"
SITES = _REPO / "configs" / "datasets" / "gaitex_sensor_sites_v1.yaml"

CONTRACT_ID = "soma_paired_small_large_v3_POC_FAITHFUL_GAITEX_SYNTH"
CONTRACT_VERSION = "0.0.2-poc-marker-small"
SMALL_MODE = "synthetic_from_markers"
MOUNT_PREFIX = "gaitex_marker"

#: Resolved from the registry rather than restated, so the artifact and the crosswalk cannot
#: drift apart. The hashes are not part of attribution and are supplied only to satisfy the
#: signature; the retarget npz carries the real ones.
ATTRIBUTION = attribution_fields(
    "gaitex", adapter_hash="-", config_hash="-", code_hash="-", model_hash="-")

#: The DRAFT governance decision this generation runs under without resolving. Recorded on every
#: manifest so the artifact says what the code cannot settle. Of the two drafts open on
#: 2026-09-06, the second, the GAITEX internal-use authorization, was recorded as a decision that
#: day (RECORDED_AUTHORIZATIONS) before this generation, so only the hold scope remains.
RECORDED_CONFLICTS = (
    "research/decisions/DRAFT_deterministic_execution_hold_scope.md",
)
RECORDED_AUTHORIZATIONS = (
    "research/decisions/2026-09-06_gaitex_internal_use_authorization.md",
)

# Which sites the source's worn units stand behind. Five, not four: back_T4 rides the sternum plate
# whose unit is XSens_Sternum. The other three have a marker-fitted rotation but no worn unit and no
# registered offset, and their manifest entries say both.
WORN_UNIT_BACKED = ("back_T4", "shank_l", "shank_r", "foot_l", "foot_r")
DECLARED_PLACEMENT = ("wrist_l", "wrist_r", "occiput")

SITE_PROVENANCE = {
    "back_T4": "marker_cluster:THOR1-4 rigid fit; sensor at the chest twin's registered plate offset "
               "(plate_normal 0.014 m into the body) -- the STERNUM sensor beneath that plate, NOT a T4 "
               "sensor (ADR-0037 puts the specification's T4 site 110-322 mm away); sensor frame S from "
               "the trial-mean world up and the plate's outward (anterior) normal; WORN-UNIT-BACKED: "
               "compared with XSens_Sternum where the take's file carries that unit (worn_unit_comparison "
               "says whether it did; worn_unit_compared_sites lists the sites compared on this take).",
    "wrist_l": "anatomical forearm frame from L_HLE, L_RSP, L_USP (not a rigid fit: L_HLE is on the "
               "humerus, ADR-0037 section 5), conditioned by the styloid-pair distance and an angular-rate "
               "ceiling; sensor at the styloid midpoint with a DECLARED ZERO offset (provisional, "
               "ADR-0037 section 6); S: +Y proximal, +Z out of the back of the wrist, +X = Y x Z "
               "(posterior on the left); NO WORN COUNTERPART.",
    "wrist_r": "as wrist_l, with the right-hand dorsal sign (+X anterior on the right): anatomical forearm "
               "frame from R_HLE, R_RSP, R_USP; DECLARED ZERO offset at the styloid midpoint (provisional); "
               "NO WORN COUNTERPART.",
    "shank_l": "marker_cluster:L_SHIN1-4 rigid fit; registered plate offset (plate_normal 0.014 m); S: +Y "
               "ankle centre to knee centre, +Z medial-to-lateral epicondyle (or the lateral plate normal "
               "when an epicondyle is absent), +X = Y x Z (posterior on the left); WORN-UNIT-BACKED: "
               "compared with XSens_LowerLeg_Left where the take's file carries it (see worn_unit_comparison).",
    "shank_r": "as shank_l on the right (+X anterior); WORN-UNIT-BACKED: compared with XSens_LowerLeg_Right "
               "where the take's file carries it (see worn_unit_comparison).",
    "occiput": "marker_cluster:R_HEAD, L_HEAD, SGL rigid fit; DECLARED ZERO offset at the cluster origin "
               "(provisional): the specification's occiput carries no marker; S: +Y the head-plane normal "
               "signed upward, +Z left-to-right head marker (subject right), +X = Y x Z (anterior); "
               "NO WORN COUNTERPART.",
    "foot_l": "marker_cluster:L_FOOT1-4 rigid fit on the dorsum; registered plate offset (plate_normal "
              "0.014 m); S: +Y the trial-mean world up (the insole's sole normal, since the dorsal plate "
              "normal tilts tens of degrees on a curved dorsum), +X calcaneus-to-plate reversed (posterior "
              "on the left), +Z = X x Y (lateral); WORN-UNIT-BACKED: compared with XSens_Foot_Left where "
              "the take's file carries it (see worn_unit_comparison). "
              "Both feet are instrumented and both are synthesised here -- the left/right asymmetry of "
              "the earlier SMPL-derived bundle was this project's joint mapping, not the source.",
    "foot_r": "as foot_l on the right (+X anterior, towards the toes); WORN-UNIT-BACKED: compared with "
              "XSens_Foot_Right where the take's file carries it (see worn_unit_comparison).",
}

#: anthro provenance strings. The emitter fills ``anthro.height`` with the SMPL rest stature at the
#: fitted betas whenever the source gives no height, and copies this string beside it; ``sex`` it
#: labels with the PRISM table's provenance for every source, which is wrong here (noted
#: 2026-09-07). Neither is a measurement and both strings say so.
HEIGHT_PROVENANCE = ("estimated:smpl_rest_stature_from_fitted_betas -- the neutral SMPL model's rest "
                     "stature at the betas fitted from segment lengths, NOT a measured stature; the "
                     "archive writes height=-1 for every subject, a placeholder")
SEX_PROVENANCE = ("declared:gender=neutral is a modelling choice, not a reading; the archive records no "
                  "sex (sex='U')")

UNAVAILABLE = (
    "development_reference / measured GRF / CoP (GAITEX records no force plate and no insole "
    "force; the archive carries none)",
    "measured accelerometer and gyroscope (GAITEX publishes the XSens orientation quaternions "
    "only; the worn-unit comparison on each manifest is an ORIENTATION comparison, and the "
    "synthesised accelerometer and gyroscope have no measured counterpart)",
    "sensor mount extrinsics for wrist_l, wrist_r and occiput (declared zero offset, provisional; "
    "the plate sites carry the registered 14 mm plate depth, whose 6 mm uncertainty is recorded "
    "in the settings file)",
    "joint_angles_jcs_deg (needs ISB/JCS convention)",
    "physics-informed GRF / inverse dynamics / moments / powers / COM",
    "quality gates",
    "subject sex, stature and mass (the archive writes mass=90 and height=-1 for every subject, "
    "which are placeholders; betas are fitted from bone lengths alone and gender is a declared "
    "choice of neutral rather than a reading; anthro.height is the SMPL rest stature at those "
    "betas, an estimate, and anthro.height_provenance says so)",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coverage(retarget_root: pathlib.Path) -> dict[str, dict]:
    """How much of each source trial the RETARGET kept, keyed ``subject/condition``.

    The retarget cuts a take at the first frame whose pelvis the markers stopped observing. The
    marker-driven small then trims only the ends of that span on which no site is valid, so the
    take's own coverage (``source_trial_coverage`` on the manifest) is recomputed per take from the
    pair window; this block supplies the trial length it is measured against.
    """
    return _run_log_block(retarget_root, "coverage")


def load_joint_ranges(retarget_root: pathlib.Path) -> dict[str, dict]:
    """How much of each take leaves anatomical range, keyed ``subject/condition``.

    Measured by the retarget on the SMPL side; the marker-driven small does not pass through the
    source's inverse kinematics, so a take whose ``solve_came_apart`` is true has a broken large
    beside a sound small. The manifest says so.
    """
    return _run_log_block(retarget_root, "joint_ranges")


def _run_log_block(retarget_root: pathlib.Path, key: str) -> dict[str, dict]:
    """A block of the retarget's run log. Its absence is refused rather than defaulted: without it a
    take's trial length, and so its coverage fraction, would silently become the retarget span."""
    path = retarget_root / "_run.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; the retarget run log supplies trial_frames and joint_ranges")
    return json.loads(path.read_text(encoding="utf-8")).get(key, {})


def confidence_by_site(settings: marker_small.SynthesisSettings,
                       comparison: dict[str, dict] | None = None) -> dict[str, float]:
    """``imu_confidence`` per site for one take, from the settings file; the wrapper adds nothing.

    A worn-unit-backed site is backed on a take only if its comparison ran there. Where the
    take's file lacks the unit, or the comparison failed, the site takes the settings'
    ``backed_site_without_comparison_on_this_take`` value instead of its usual one.
    """
    semantics = settings.document["imu_confidence"]
    sites = semantics["sites"]
    out = {name: float(sites[name]) for name in marker_small.SITE_ORDER}
    if comparison is not None:
        fallback = float(semantics["backed_site_without_comparison_on_this_take"])
        for name in WORN_UNIT_BACKED:
            if comparison.get(name, {}).get("status") != "ok":
                out[name] = fallback
    return out


def settings_digest(settings_path: pathlib.Path, sites_path: pathlib.Path) -> str:
    """One digest over both configuration files, folded into the pair identity so two generations
    under different settings never share a pair_id."""
    return hashlib.sha256((sha256_file(settings_path) + sha256_file(sites_path)).encode("ascii")).hexdigest()


def identity_for(take: pathlib.Path, subject: str, condition: str, frames: int, rate: float,
                 settings: marker_small.SynthesisSettings, extra: dict, config_digest: str):
    relative = paths.data_relative(take)
    return EMIT.SourceIdentity(
        source_name="gaitex",
        dataset="gaitex",
        subject_id=f"gaitex_{subject}",
        sequence_id=condition,
        relative_path=relative,
        asset_sha256=sha256_file(take),
        native_frames=frames,
        native_rate_hz=rate,
        framerate_source="source_header:trc DataRate and mot time column, both 100 Hz; the Qualisys "
                         "marker csv shares the grid frame for frame (verified 2026-09-06)",
        source_note="GAITEX: optical markers and nine worn inertial units. large/anthro come from the "
                    "AddBiomechanics retarget with a GAITEX reader (root recovered from the four pelvis "
                    "stations, since the published IK froze the pelvis translation). small comes from "
                    "the marker clusters directly: rigid-body pose per site, the registered plate lever "
                    "arm, a zero-phase Butterworth low-pass, and the spec sensor frame built from "
                    "landmarks. See small_synthesis.",
        field_registry="configs/datasets/gaitex_field_registry_v1.yaml",
        frame_convention="Qualisys laboratory frame, right-handed, Z-up, gravity along -Z; identical to "
                         "the spec world G (the retarget's OpenSim Y-up frame maps onto it by the same "
                         "+90 degree X rotation the trc applies, so no rotation is applied to the markers)",
        contract_id=CONTRACT_ID,
        # build metadata carries the settings + site-registry digest into pair_id (which hashes
        # contract_version); the version proper is CONTRACT_VERSION
        contract_version=f"{CONTRACT_VERSION}+cfg.{config_digest[:16]}",
        run_id=f"gaitex-{subject}-{condition}-unified8-marker",
        mount_prefix=MOUNT_PREFIX,
        artifact_label="PoC FAITHFUL GAITEX synthetic reference (marker-derived synthetic IMU small, "
                       "SMPL-retargeted large). NOT contract-compliant, NOT quality-gate PASS. "
                       "INTERNAL-ONLY.",
        anthro_namespace="anthro_reference/gaitex_smpl_v1",
        subject_scope="subject_constant_trial_frame_invariant",
        subject_scope_note="betas are fitted once per subject from segment lengths POOLED across "
                           "that subject's trials, because GAITEX rescales its OpenSim model per "
                           "trial and the resulting skeletons disagree by about 2.7 percent on a "
                           "femur. The disagreement is the knee centre moving: femur and tibia "
                           "deviations are anticorrelated (r = -0.766 right, -0.642 left), so the "
                           "hip-to-ankle span is fitted alongside its two parts.",
        height_m=None,
        height_provenance=HEIGHT_PROVENANCE,
        body_mass_kg=None,
        body_mass_provenance="absent:the archive writes mass=90 for every subject, a placeholder",
        betas_provenance="source_derived:segment lengths from forward kinematics on the scaled "
                         "model, pooled by median across the subject's trials, fitted against the "
                         "clean neutral SMPL model without stature or mass constraints",
        smpl_trans_provenance="source_derived:pelvis pose fitted to a shape learned from the four "
                              "pelvis stations (R_IAS, L_IAS, R_IPS, L_IPS); NOT the published "
                              "pelvis_tx/ty/tz, which the IMU IK left frozen",
        small_content_note="8ch synthetic IMU (back_T4, wrist_l/r, shank_l/r, occiput, foot_l/r) "
                           "synthesised from optical marker clusters in the spec sensor frame S: "
                           "specific force (gravity-included) + rotation-log gyro, zero-phase "
                           f"Butterworth {settings.filter_order}th order at {settings.cutoff_hz:g} Hz. "
                           f"small_mode={SMALL_MODE}, axis_convention=spec_S_v2. Every site carries "
                           "the motion of its own marker cluster; imu_valid_mask says where a site "
                           "has no signal (NaN there). Five sites are compared with the worn "
                           "XSens unit on the same segment where the take carries it; wrist_l, "
                           "wrist_r and occiput carry a declared zero offset. See small_synthesis, "
                           "worn_unit_comparison and small_sites_provenance.",
        unavailable=UNAVAILABLE,
        site_provenance=SITE_PROVENANCE,
        extra_manifest=extra,
    )


def replace_small(bundle: dict, small: dict[str, np.ndarray]) -> None:
    """Put the marker-derived arrays in place of the emitter's SMPL-derived ones.

    Written after the bundle is built rather than inside the emitter, because the emitter is
    shared with three other sources and pinned to byte equality by a test. What stays from the
    emitter: sensor_codes, timestamps_s, frame_count, pair_id, axis_convention, and
    imu_orientation_absolute_heading (which says what the spec's sensor at that site provides,
    not how this bundle was made -- the AMASS synthetic path carries the same values).
    """
    target = bundle["small"]
    frames = int(target["frame_count"])
    for key, array in small.items():
        if array.shape[0] != frames:
            raise ValueError(f"{key}: {array.shape[0]} frames against the bundle's {frames}")
        target[key] = array
    target["small_mode"] = np.str_(SMALL_MODE)
    target["mount_id"] = np.array([f"{MOUNT_PREFIX}::{c}" for c in target["sensor_codes"]], dtype=np.str_)


def _aligned(label: str, time_s: np.ndarray, index: np.ndarray, expected_s: np.ndarray, dt_s: float) -> None:
    """The rows joined by index must carry the retarget's own timestamps; a file one row short would
    otherwise shift small against large by a frame without a word."""
    if index.max() >= time_s.shape[0]:
        raise marker_small.MarkerSmallError(f"{label}: the retarget covers frames the file does not have")
    if not np.allclose(time_s[index], expected_s, atol=dt_s / 2.0, rtol=0.0):
        worst = float(np.abs(time_s[index] - expected_s).max())
        raise marker_small.MarkerSmallError(
            f"{label}: time column disagrees with the retarget timestamps by up to {worst:.4f} s")


def _input_record(path: pathlib.Path, frames: int, extracted: pathlib.Path | None = None) -> dict:
    """A source file the small was synthesised from, by its logical id `extracted/gaitex/...`.

    `extracted` is the folder the run read as the GAITEX source (default `paths.source_dir`)."""
    return {"relative_path": paths.logical_source_id(SOURCE, path, root=extracted),
            "sha256": sha256_file(path), "frames": int(frames)}


def take_dirname(subject: str, condition: str) -> str:
    return f"gaitex_{subject}_{condition}"


def prepare(take: pathlib.Path, models: dict, coverage: dict[str, dict],
            joint_ranges: dict[str, dict], registry: site_registry.SiteRegistry,
            settings: marker_small.SynthesisSettings, extracted: pathlib.Path,
            config_digest: str) -> SimpleNamespace:
    """Everything one take needs before its bundle is built: the retarget arrays, the marker
    synthesis and the pair window, the worn-unit comparison, and the source identity.

    Split from `emit` so a subject's frozen-joint constants can be fitted over the rows every one
    of its bundles will carry -- which only the pair window says -- before any bundle is built."""
    with np.load(take, allow_pickle=False) as handle:
        subject = str(handle["subject"])
        # The retarget writes one file per condition and names it after that condition, so the
        # stem is the condition itself.
        condition = take.stem
        gender = str(handle["gender"])
        betas = np.asarray(handle["betas"], np.float64)
        pose = np.asarray(handle["pose"], np.float64)
        trans = np.asarray(handle["trans"], np.float64)
        rate = float(handle["source_rate_hz"])
        timestamps = np.asarray(handle["timestamps_s"], np.float64)
    key = f"{subject}/{condition}"

    # --- markers: the Qualisys csv, which is the trc frame for frame
    trial = gaitex_adapter.TrialRef(subject=subject, condition=condition, directory=extracted / subject / condition)
    marker_csv = gaitex_adapter.marker_path(trial)
    imu_csv = gaitex_adapter.imu_path(trial)
    stream = gaitex_adapter.read_markers(marker_csv)
    imu_stream = gaitex_adapter.read_imu_orientation(imu_csv)
    source_index = np.rint(timestamps * rate).astype(int)
    _aligned("marker csv", stream.time_s, source_index, timestamps, settings.dt_s)
    imu_rows = source_index[source_index < imu_stream.time_s.shape[0]]
    _aligned("imu csv", imu_stream.time_s, imu_rows, timestamps[: imu_rows.size], settings.dt_s)

    syntheses = marker_small.synthesise_sites(stream, registry, settings)
    valid = np.stack([syntheses[name].signal.valid for name in marker_small.SITE_ORDER], axis=1)
    interpolated = np.stack([syntheses[name].interpolated for name in marker_small.SITE_ORDER], axis=1)
    window = marker_small.pair_window(valid, source_index, interpolated_by_site=interpolated)
    dead = [name for k, name in enumerate(marker_small.SITE_ORDER) if not window.valid[:, k].any()]
    if dead:
        raise marker_small.MarkerSmallError(
            f"{', '.join(dead)}: no valid frame inside the pair window; the take would fail L2 and is "
            f"refused here instead")

    # --- large / anthro / root on exactly the window's frames
    if gender not in models:
        models[gender] = load_clean_model(clean_model_path_for_gender(gender))
    rows = window.rows
    trial_frames = int(coverage[key]["trial_frames"])     # KeyError = the run log does not know this take
    comparison = marker_small.compare_with_worn_units(
        syntheses, imu_stream, window, registry, settings,
        {k: v for k, v in settings.document["worn_unit_counterpart_twin"].items() if k != "note"},
        int(settings.document["evidence_run"]["minimum_comparable_frames"]))
    compared = [name for name in WORN_UNIT_BACKED if comparison.get(name, {}).get("status") == "ok"]
    confidence = confidence_by_site(settings, comparison)
    extra = {
        "source_attribution": {
            "source_name": ATTRIBUTION["source_name"],
            "source_version": ATTRIBUTION["source_version"],
            "stable_source_id": ATTRIBUTION["stable_source_id"],
            "dataset_citation": ATTRIBUTION["dataset_citation"],
            "publication": ATTRIBUTION["publication"],
            "license_id": ATTRIBUTION["license_id"],
            "changes_made": "true; retargeted onto SMPL-24 (large) and synthesised from markers (small), "
                            "see spec_documents",
            "attribution_note": "GAITEX is CC BY 4.0 and its licence note conditions "
                                "redistribution on crediting the Zenodo dataset record, "
                                "which dataset_citation carries. publication names the "
                                "Scientific Data article, which is a different object.",
        },
        "worn_unit_backed_sites": list(WORN_UNIT_BACKED),
        "worn_unit_compared_sites": compared,
        "worn_unit_compared_sites_note": "the backed sites whose comparison ran on THIS take; a backed "
                                         "site absent here carries the declared-position confidence "
                                         "on this take (imu_confidence_semantics)",
        "declared_placement_sites": list(DECLARED_PLACEMENT),
        "root_translation": {
            "published": "frozen at the static scaling pose for every frame",
            "used": "recovered per frame from the pelvis marker stations",
            "why_it_matters": "the root is common to every joint of large; small no longer depends "
                              "on it, since each site's acceleration is differentiated from its own "
                              "marker cluster",
            "shape_fitted": "learned from the observed marker cluster, not the model's "
                            "declared stations",
            "known_limitation": "the recovered origin still steps a median 3.7 mm whenever "
                                "the set of visible pelvis markers changes; those steps are in "
                                "large's root_velocity and pelvis_position_world_aux, and are NOT "
                                "in small.",
        },
        "joint_ranges": dict(joint_ranges.get(key) or {
            "note": "not measured; the retarget run log was unavailable when this take was emitted"}),
        "generated_under": {
            "settings": {"config_id": settings.config_id, "version": settings.version,
                         "hold_status": settings.document["hold_status"]},
            "site_registry": {"config_id": registry.config_id, "version": registry.version},
            "recorded_conflicts": list(RECORDED_CONFLICTS),
            "recorded_authorizations": list(RECORDED_AUTHORIZATIONS),
            "authorises_generation": False,
            "note": "Generated with the governance draft above unresolved, by decision of "
                    "2026-09-06 (research/decisions/"
                    "2026-09-06_gaitex_internal_use_authorization.md; of the two drafts open then, "
                    "the GAITEX internal-use authorization has since been recorded as a decision "
                    "and is listed under recorded_authorizations; it lifts no hold). The hold "
                    "interpretation was not settled before this generation.",
        },
    }
    if joint_ranges.get(key, {}).get("solve_came_apart"):
        extra["joint_ranges"]["small_note"] = (
            "large's source inverse kinematics came apart on this take (see per_coordinate); "
            "small does not pass through that IK and is unaffected -- the pair is broken on one "
            "side only.")

    source = identity_for(take, subject, condition, trial_frames, rate, settings, extra, config_digest)
    return SimpleNamespace(
        take=take, subject=subject, condition=condition, gender=gender, betas=betas, pose=pose,
        trans=trans, rate=rate, rows=rows, window=window, syntheses=syntheses,
        confidence=confidence, settings=settings, registry=registry, source=source,
        source_index=source_index, trial_frames=trial_frames, marker_csv=marker_csv,
        imu_csv=imu_csv, stream=stream, imu_stream=imu_stream, comparison=comparison,
        model=models[gender], rel=take_dirname(subject, condition), extracted=extracted)


def fit_member(prepared: SimpleNamespace) -> tuple[str, np.ndarray, np.ndarray]:
    """(take directory, the window's pose rows, rest joints) for the subject's pooled fit; the rest
    skeleton is the one `build_bundle` derives from the gendered model and the betas."""
    return (prepared.rel, prepared.pose[prepared.rows],
            EMIT.ANTHRO.rest_joints(prepared.model, prepared.betas))


def emit(prepared: SimpleNamespace, out_root: pathlib.Path, reduction=None) -> dict:
    """Build and write the take's bundle from what `prepare` read, with the subject's fit."""
    take, condition = prepared.take, prepared.condition
    pose, trans, rows, window = prepared.pose, prepared.trans, prepared.rows, prepared.window
    syntheses, confidence = prepared.syntheses, prepared.confidence
    settings, registry, source = prepared.settings, prepared.registry, prepared.source
    source_index, trial_frames = prepared.source_index, prepared.trial_frames
    marker_csv, imu_csv = prepared.marker_csv, prepared.imu_csv
    stream, imu_stream, comparison = prepared.stream, prepared.imu_stream, prepared.comparison
    bundle = EMIT.build_bundle(
        pose24=pose[rows], trans=trans[rows], betas=prepared.betas, gender=prepared.gender,
        model=prepared.model, src_fps=prepared.rate, source=source, reduction=reduction,
    )

    # the shared anthro emitter stamps every source with PRISM's sex provenance; this source's is declared
    bundle["anthro"]["sex_provenance"] = SEX_PROVENANCE

    # --- small from the markers, on the same frames
    smpl_small_orientation = np.asarray(bundle["small"]["imu_orientation"], np.float64)  # before it goes
    small_arrays = marker_small.stack_small(syntheses, window, confidence)
    replace_small(bundle, small_arrays)
    dispersion = marker_small.relabel_dispersion_vs_large(
        small_arrays["imu_orientation"], small_arrays["imu_valid_mask"],
        np.asarray(bundle["large"]["smpl_global_orientation_world"], np.float64))
    offset = marker_small.relabel_offset_vs_reference(
        small_arrays["imu_orientation"].astype(np.float64), small_arrays["imu_valid_mask"], smpl_small_orientation)
    guard = float(settings.document["validation"]["relabel_offset_guard_deg"])
    flipped = [name for name in WORN_UNIT_BACKED if offset[name] is not None and offset[name] > guard]
    if flipped:
        raise marker_small.MarkerSmallError(
            f"{', '.join(flipped)}: the marker sensor frame is more than {guard:g} deg from the SMPL "
            f"anatomical frame of the same site ({', '.join('%.0f' % offset[n] for n in flipped)} deg); "
            f"a sign is wrong and the take is refused")

    manifest = bundle["manifest"]
    # the emitter saw only the window's rows, so its window block says [0, m); say what those rows are
    manifest["window"] = {
        "frame_start": int(rows.start), "frame_end_exclusive": int(rows.stop),
        "frame_count": window.frames,
        "basis": "rows of the retarget npz arrays (pose, trans, timestamps_s); source_window gives the "
                 "same span on the marker csv index",
    }
    manifest["source_window"] = window.as_manifest(trial_frames)
    manifest["source_trial_coverage"] = {
        "span_frames": window.frames, "trial_frames": trial_frames,
        "fraction": round(window.frames / float(trial_frames), 4),
        "retarget_span_frames": int(source_index.size),
        "note": "the take is the retarget's pelvis-solved span minus leading/trailing frames on "
                "which no marker site is valid; interior gaps stay and are masked (see source_window)",
    }
    # the emitter's validation block measured the SMPL small this generator replaced
    validation = manifest.get("validation", {})
    removed = [key for key in validation if key.startswith("specific_force_inverse")]
    for key in removed:
        del validation[key]
    valid_cells = small_arrays["imu_valid_mask"]
    validation["imu_quat_norm_max_dev"] = float(np.max(np.abs(
        np.linalg.norm(small_arrays["imu_orientation"][valid_cells].astype(np.float64), axis=-1) - 1.0)))
    validation["imu_quat_norm_max_dev_note"] = "recomputed on the marker-derived small over mask-valid cells"
    if removed:
        validation["removed"] = {key: "measured the replaced SMPL-derived small; the marker path has no "
                                      "position truth to invert against" for key in removed}
    manifest["validation"] = validation
    manifest["small_synthesis"] = {
        "small_mode": SMALL_MODE,
        "settings": {"config_id": settings.config_id, "version": settings.version},
        "site_registry": {"config_id": registry.config_id, "version": registry.version,
                          "site_set": "external_spec_eight"},
        "inputs": {
            "marker_csv": _input_record(marker_csv, stream.positions_m.shape[0], prepared.extracted),
            "imu_csv": _input_record(imu_csv, imu_stream.quaternions.shape[0], prepared.extracted),
            "retarget_npz": {"relative_path": paths.data_relative(take),
                             "sha256": source.asset_sha256, "frames": int(pose.shape[0])},
            "note": "source.* and identity.pair_id name the retarget npz; the marker and imu files are "
                    "the small's own inputs and are hashed here. The configuration digest in "
                    "identity.contract_version covers the settings file and the site registry.",
        },
        "filter": {"type": "butterworth", "order": settings.filter_order, "zero_phase": True,
                   "cutoff_hz": settings.cutoff_hz, "edge_trim_frames": settings.edge_trim_frames},
        "gap_policy": {"max_interpolated_gap_frames": settings.max_gap_frames,
                       "note": "a gap in a cluster's pose no longer than this is bridged (Hermite "
                               "translation, squad rotation) before synthesis and ships VALID; "
                               "source_window.sites[*].interpolated_runs_bundle_rows lists those frames"},
        "forearm_conditioning": {"styloid_pair_distance_tolerance_m": settings.forearm_pair_tolerance_m,
                                 "max_rate_deg_per_s": settings.forearm_max_rate_deg_per_s,
                                 "max_gap_twist_deg": settings.forearm_max_gap_twist_deg,
                                 "label_swap_max_withdrawn_fraction": settings.forearm_label_swap_max_withdrawn_fraction},
        "sensor_frame_minimum_frames": settings.frame_minimum_frames,
        "relabel_offset_guard_deg": guard,
        "lever_arm_depth_uncertainty_m": settings.document["lever_arm"]["depth_uncertainty_m"],
        "sites": {name: syntheses[name].manifest_block() for name in marker_small.SITE_ORDER},
        "imu_confidence_semantics": settings.document["imu_confidence"],
        "imu_confidence_on_this_take": confidence,
        "mask_discipline": "imu_valid_mask False cells are NaN in imu_orientation, imu_acceleration "
                           "and imu_angular_velocity and 0.0 in imu_confidence; True cells are finite. "
                           "A True cell is observed or, inside a gap the gap policy bridged, "
                           "interpolated (listed per site in source_window). large is finite on every "
                           "frame. Apply the mask to small before any filter, derivative or loss, and "
                           "to large as well if frames are paired (ADR-0039).",
        "canonicalization_config_note": "canonicalization_config describes the shared emitter, which "
                                        "produced large, anthro, the root translation and the "
                                        "SMPL-derived small this generator REPLACED. Its filter and "
                                        "imu_synthesis entries described that replaced small and "
                                        "describe nothing shipped here: large's joint_velocity and "
                                        "root_velocity are unfiltered finite differences, and this "
                                        "small's filter is small_synthesis.filter.",
    }
    manifest["worn_unit_comparison"] = {
        "method": "compare_orientation: a constant mounting rotation (solved on frames turning faster "
                  "than the settings' minimum rate) and a constant world-frame offset between the "
                  "optical and inertial frames are both taken out, then the residual angle per frame "
                  "is reported; frames the mask calls invalid are NaN on the comparison's own time "
                  "axis, never packed together; the measured stream never leaves the comparison",
        "sites": comparison,
    }
    manifest["relabel_dispersion_vs_large_deg"] = {
        "what": "median geodesic dispersion of smpl_global_orientation_world[joint]^T @ imu_orientation "
                "about its mean over valid frames, per site -- what validator L3 measures; a "
                "marker-fitted body is not a constant relabel of the SMPL joint, which is why "
                "small_mode is judged like a measured sensor. null where fewer than two frames are valid",
        "sites": dispersion,
    }
    manifest["relabel_offset_vs_smpl_anatomical_deg"] = {
        "what": "angle of the MEAN rotation between this small's sensor frame and the emitter's "
                "SMPL-anatomical sensor frame of the same site (the replaced small), over valid "
                "frames. Dispersion cannot see a constant error; a plate normal signed the wrong "
                "way or a wrist with swapped styloids is a half turn here. Worn-unit-backed sites are "
                "refused above relabel_offset_guard_deg; the declared sites (wrists, occiput) sit on "
                "SMPL joints the source's inverse kinematics never drove and are recorded only",
        "sites": offset,
    }
    out_dir = out_root / prepared.rel
    json.dumps(manifest, allow_nan=False)    # a NaN anywhere in the manifest fails here, not in a reader
    EMIT.write_bundle(out_dir, bundle)
    return {
        "take_id": prepared.rel, "rel": out_dir.name, "status": "ok",
        "frames": int(bundle["small"]["frame_count"]),
        "pair_id": bundle["manifest"]["identity"]["pair_id"],
        "subject_id": source.subject_id, "sequence_id": condition,
    }


DATA_DESCRIPTION_TITLE = (
    "GAITEX - synthetic IMU reference (qmd_unified8_smpl18), small from the markers"
)


def data_description_body(settings) -> str:
    """The bundle-root description, as a function so it can be rewritten without a full run.

    It was inline in main(), which meant a wording fix could only ship with a regeneration of
    all 72 takes. Only `settings.cutoff_hz` varies.
    """
    return (
            # The bundle replaces one of the same name whose Small was SMPL-derived, so this has
            # to be the first thing a holder of that copy reads. It was duplicated into
            # DATASET_DESCRIPTION_EN.md as well, because that file was not what the validator
            # required at the root nor where a reader trained on the PRISM and AMASS bundles
            # looked first -- the duplication ADR-0040 D3 removed by retiring the second file.
            "## Read this first if you hold an earlier copy\n\n"
            "**The directory name is unchanged and the contents are not.** This replaces the "
            "bundle of 2026-09-05, whose `small_mode` was `synthetic_from_smpl`. Every take's "
            "`pair_id` is different, because `contract_version` is hashed into it, and the frame "
            "counts differ. Nothing here can be matched to the older bundle by take identity; "
            "match by subject and condition instead. **If you built anything against the earlier "
            "copy, rebuild it.**\n\n"
            "## What this is, and what it is not\n\n"
            "Optical markers and an OpenSim inverse kinematics driven by nine worn inertial units. "
            "`large`, `anthro` and `smpl_root_translation` come from the SMPL-24 retarget of that "
            "inverse kinematics. **`small` does not: it is synthesised from the marker clusters "
            "directly** (`small_mode` = `synthetic_from_markers`, ADR-0039), site by site -- a rigid "
            "body fitted to each plate or landmark set, the registered plate lever arm, a zero-phase "
            f"Butterworth low-pass at {settings.cutoff_hz:g} Hz, and the spec sensor frame S built "
            "from landmarks. Every one of the eight channels carries the motion of its own segment; "
            "the earlier SMPL-derived bundle had four channels resting on undriven joints.\n\n"
            "**Five sites are compared with the worn XSens unit on the same segment** -- "
            "`back_T4` (the sternum plate's unit), both shanks, both feet -- by taking out a constant "
            "mounting rotation and a constant world offset and reporting the residual angle; each "
            "manifest's `worn_unit_comparison` carries the numbers and its status per site, and "
            "`worn_unit_compared_sites` names the sites on which the comparison ran on that take. "
            "No acceptance band is applied: the residual is reported, not judged. `wrist_l`, "
            "`wrist_r` and `occiput` have a marker-fitted rotation but no worn unit and no registered "
            "offset: their sensor sits at the body's own origin as a **declared, provisional zero** "
            "(ADR-0037 section 6). `imu_confidence` is 0.6 on a compared site, 0.3 on a declared site "
            "and on a backed site whose comparison did not run on that take, 0.0 on an invalid cell "
            "(`small_synthesis.imu_confidence_on_this_take`).\n\n"
            "## The mask is not decoration\n\n"
            "`imu_valid_mask` is informative on this corpus, for the first time in this lineage. "
            "Where a site has no signal -- a marker gap the 20-frame policy does not bridge, the "
            "17-frame edge of every filtered run, a withdrawn glitch -- its cell is **NaN** in the "
            "three IMU arrays, False in the mask and 0.0 in `imu_confidence`. Nothing is filled with "
            "zeros. One thing is bridged: a gap in a cluster's pose of at most 20 frames is "
            "interpolated before synthesis and ships as a valid cell; "
            "`source_window.sites[*].interpolated_runs_bundle_rows` lists exactly which frames, and "
            "`usable_runs_bundle_rows` each site's contiguous valid runs, both in bundle rows. `large` "
            "is finite on every frame. Apply the mask before any filter, derivative or loss, and if "
            "you pair frames, apply it to `large` too.\n\n"
            "## Known limits\n\n"
            "- **`back_T4` is the sternum sensor.** The plate is anterior, on the sternum, and the "
            "specification's T4 site is 110-322 mm away (ADR-0037). The channel is compared with "
            "that sternum unit; it is not a posterior-thorax sensor.\n"
            "- **Placement of the wrists and the occiput is a declared zero**, not a measurement and "
            "not the ADR-0037 rule, which the hold keeps out of configuration. Their accelerometers "
            "therefore carry no lever-arm term.\n"
            "- **`imu_orientation` is a rigid relabel of the marker-fitted body, not of the SMPL "
            "joint.** The two disagree, most on the wrists and head, where GAITEX's inverse "
            "kinematics never drove the SMPL joint; each manifest's `relabel_dispersion_vs_large_deg` "
            "(the spread) and `relabel_offset_vs_smpl_anatomical_deg` (the mean) give the per-site "
            "values, and the generation report tabulates them. This is why the spec_S_v2 axis check "
            "(L3) judges this corpus like a measured sensor.\n"
            "- **The sensor frame is one constant per take**, built from landmarks: trunk and feet use "
            "the trial-mean world up as their +Y (a static-calibration analogue), so a take's mean "
            "lean or foot pitch is absorbed into that constant.\n"
            "- **At least four takes carry joint angles no body can produce** in `large`, because the "
            "source's own inverse kinematics came apart on them (`joint_ranges.solve_came_apart`). "
            "`small` does not pass through that IK and is unaffected on those takes: the pair is "
            "broken on one side only. Filter on `solve_came_apart`, not on `frames_outside_any`.\n"
            "- The take is the retarget's longest pelvis-solved span, minus leading/trailing frames on "
            "which no marker site is valid. `source_window` and `source_trial_coverage` give the span, "
            "the trial length and the fraction.\n"
            "- GAITEX rescales its OpenSim model **per trial**; betas are fitted once per subject from "
            "pooled medians (see `anthro`).\n"
            "- Subject sex, stature and mass are **absent** (the archive writes placeholders). Gender "
            "neutral is a declared choice. `anthro.height` is therefore the SMPL rest stature at the "
            "fitted betas -- an estimate, labelled `estimated:` in `height_provenance` -- and "
            "`anthro.body_mass` is NaN.\n"
            "- GAITEX publishes the XSens **orientation quaternions only**: the worn-unit comparison is "
            "an orientation comparison, and the synthesised accelerometer and gyroscope have no "
            "measured counterpart. It records no ground reaction force either.\n"
            "- Generated with one governance draft unresolved (the deterministic-execution hold scope), "
            "recorded on every manifest under `generated_under.recorded_conflicts`; the GAITEX "
            "internal-use authorization recorded on 2026-09-06 is under "
            "`generated_under.recorded_authorizations` and lifts no hold.\n\n"
            "Evidence: the SOMA project's GAITEX marker-small evidence report (2026-09-06) and the "
            "generation report it accompanies.\n"
    )


def select_subjects(retarget: pathlib.Path, wanted=None) -> list[str]:
    """The retarget corpus's subject folders a run converts: all of them, or those ``wanted`` names.

    Names are matched exactly against the folder listing, case included (on NTFS the joined path
    exists whatever its case, while this selection compares strings). A name the corpus lacks, and
    a selection that holds no take (``<subject>/*.npz``), raise ValueError: the run would write
    INDEX.json, the fit record and the description over zero takes and exit 0."""
    subjects = sorted(p.name for p in retarget.iterdir() if p.is_dir())
    if wanted:
        unknown = sorted(set(wanted) - set(subjects))
        if unknown:
            raise ValueError(
                f"--subjects {' '.join(unknown)}: not a subject of the retarget corpus {retarget} "
                f"(names are case-sensitive; subjects there: {', '.join(subjects) or 'none'})")
        subjects = [s for s in subjects if s in set(wanted)]
    if not any(any((retarget / subject).glob("*.npz")) for subject in subjects):
        chosen = f" by --subjects {' '.join(wanted)}" if wanted else ""
        raise ValueError(f"no take (<subject>/*.npz) selected in the retarget corpus {retarget}"
                         f"{chosen}; nothing written")
    return subjects


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retarget", type=pathlib.Path, default=None,
                        help=f"the retarget corpus; default $SOMA_DATA_ROOT/{paths.POC_DEMO}/"
                             f"{RETARGET_LINEAGE}")
    parser.add_argument("--extracted", type=pathlib.Path, default=None,
                        help="the GAITEX source folder; default $SOMA_SOURCE_ROOT/gaitex")
    parser.add_argument("--settings", type=pathlib.Path, default=SETTINGS)
    parser.add_argument("--sites", type=pathlib.Path, default=SITES)
    parser.add_argument("--subjects", nargs="*", default=None)
    arguments = parser.parse_args(argv)
    retarget_given = arguments.retarget is not None
    extracted_given = arguments.extracted is not None
    try:
        if arguments.retarget is None:
            arguments.retarget = paths.lineage_dir(RETARGET_LINEAGE)
        if arguments.extracted is None:
            arguments.extracted = paths.source_dir(SOURCE)
        paths.body_model_dir()     # checked before prepare(), whose per-take guard would swallow it
        out_root = paths.check_output_dir(pathlib.Path(arguments.out),
                                          inputs=[arguments.retarget, arguments.extracted])
        # every take's relative_path is written against the data root, so a corpus outside it
        # would fail each take on its own and still leave an INDEX over zero takes
        paths.data_relative(arguments.retarget)
        # both inputs are listed or globbed, and a missing folder lists as empty
        if not arguments.retarget.is_dir():
            named = ("as given" if retarget_given
                     else f"{paths.DATA_ROOT_ENV}/{paths.POC_DEMO}/{RETARGET_LINEAGE}")
            raise paths.PathConfigError(f"the retarget corpus {arguments.retarget} ({named}) does "
                                        "not exist or is not a directory")
        paths.existing_source_dir(SOURCE, arguments.extracted if extracted_given else None)
        subjects = select_subjects(arguments.retarget, arguments.subjects)
    except (paths.PathConfigError, ValueError) as error:
        print(error)
        return 2
    models: dict[str, dict] = {}
    coverage = load_coverage(arguments.retarget)
    joint_ranges = load_joint_ranges(arguments.retarget)
    settings = marker_small.load_settings(arguments.settings)
    registry = site_registry.load_site_registry(arguments.sites)
    config_digest = settings_digest(arguments.settings, arguments.sites)
    out_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    fits = REDUCED.CorpusFits(REDUCED.settings_for_source("gaitex"))
    started = time.time()

    def failed(subject: str, take: pathlib.Path, error: Exception) -> dict:
        print(f"  FAILED {take.name}: {error}", flush=True)
        return {"take_id": take_dirname(subject, take.stem), "rel": take_dirname(subject, take.stem),
                "status": "failed", "reason": f"{type(error).__name__}: {error}"}

    for subject in subjects:
        takes = sorted((arguments.retarget / subject).glob("*.npz"))
        results: dict[pathlib.Path, object] = {}
        for take in takes:
            try:
                results[take] = prepare(take, models, coverage, joint_ranges, registry, settings,
                                        arguments.extracted, config_digest)
            except Exception as error:                          # noqa: BLE001
                results[take] = failed(subject, take, error)
        # One set of frozen-joint constants per subject, over exactly the rows its bundles carry.
        prepared = [r for r in results.values() if isinstance(r, SimpleNamespace)]
        if prepared:
            fits.fit(f"gaitex_{subject}", [fit_member(p) for p in prepared],
                     basis="every take of the subject, the pair-window rows its bundle carries, "
                           "native rate")
        for take in takes:
            result = results[take]
            if isinstance(result, SimpleNamespace):
                try:
                    result = emit(result, out_root, reduction=fits.by_take[result.rel])
                except Exception as error:                      # noqa: BLE001
                    result = failed(subject, take, error)
            entries.append(result)
        print(f"  {subject}: {len(entries)} takes so far ({time.time() - started:.0f}s)", flush=True)

    ok = [entry for entry in entries if entry["status"] == "ok"]
    index = {
        "spec_id": EMIT.SPEC_ID, "spec_version": EMIT.SPEC_VERSION,
        "dataset_dirname": out_root.name,
        "source": "gaitex",
        "artifact_class": "experimental_non_candidate",
        "distribution_scope": "internal_only",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "counts": {
            "total": len(entries), "ok": len(ok),
            "failed": sum(1 for e in entries if e["status"] == "failed"), "excluded": 0,
        },
        "complete": len(ok) == len(entries),
        "takes": entries,
    }
    (out_root / "INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    fits.write(out_root)
    EMIT.write_data_description(
        out_root, index,
        title=DATA_DESCRIPTION_TITLE,
        body=data_description_body(settings),
    )
    print(f"\n{len(ok)}/{len(entries)} takes ok in {time.time() - started:.0f}s -> {out_root}",
          flush=True)
    return 0 if index["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
