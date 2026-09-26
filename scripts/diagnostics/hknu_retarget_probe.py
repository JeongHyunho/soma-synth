"""Can the HKNU skeleton be retargeted onto SMPL-24, and with what error?

A correspondence table on its own proves nothing -- it is a claim about what maps to what -- so
this actually runs the shape fit and the pose fit through the modules the conversion will
reuse, and reports the residuals.  A table that looks right and a fit that lands 300 mm from
the measured joint are indistinguishable until the fit is run.

The toes and hand tips carry centres (DISTAL_CORRESPONDENCE, on by default): without them the
ankles and wrists have no descendant to aim at, never get their own alignment constant, and
inherit one fitted for a different body, which can put 91 deg of rotation into a standing
ankle and point the foot at the ceiling.  `--no-distal-centres` runs the table without them, for
comparison.

Two things about reuse are checked rather than assumed:

  * pose_fit hardcodes the trunk body as "torso" (_TRUNK_BODY, used in three places).  HKNU's
    trunk is TA, so the rotations handed over alias it; without that the three spine joints
    fall to `absent` silently and the torso stops following the subject.
  * The Visual3D segment table makes PV and TA each other's parent.  The tree here is rooted
    at PV by construction, which is the resolution of that cycle, not a workaround for it.

Usage:
    python scripts/diagnostics/hknu_retarget_probe.py --out <output folder>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation

from soma_synth.addbio_retarget.pose_fit import RootPlacement, fit_pose
from soma_synth.addbio_retarget.shape_fit import (
    clean_model_path_for_gender,
    fit_betas,
    load_clean_model,
    measured_segment_lengths,
    normalise_biological_sex,
    rescale_rest_joints,
    segment_pairs_from,
    smpl_rest_joints,
)
from soma_synth.addbio_retarget.smpl_correspondence import (
    ABSENT,
    DERIVED_LUMBAR,
    MEASURED,
    SMPL24_NAMES,
    SmplJointSource,
)
from soma_synth.pipeline import paths


def __getattr__(name: str):
    """``SOURCE_ROOT``, the HKNU source folder, resolved from the environment when read.

    This module is loaded by the HKNU generator, so it reads nothing at import; the diagnostics
    that used the old constant (``probe.SOURCE_ROOT``) get ``paths.source_dir("hknu")``."""
    if name == "SOURCE_ROOT":
        return paths.source_dir("hknu")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: pose_fit reads the trunk under this name; HKNU calls the same body TA.
TRUNK_ALIAS = "torso"

#: SMPL joint -> (HKNU segment driving its rotation, centre name giving its position).
#: Written out in full, including the joints nothing feeds, so that the absent set is a
#: decision on the page rather than the silent remainder of a lookup.
#:
#: The centres are chosen by measuring, not by naming.  In this model "proximal" means the
#: cranial end for the trunk segments -- PV.ProxEndPos sits above PV.DistEndPos, and so does
#: TA's -- so the name alone points the wrong way.  Reading the heights of the source against
#: SMPL's own rest layout, relative to the hip centres in each:
#:
#:   SMPL pelvis  +85 mm above the hips   <-  PV.ProxEndPos   +81 mm
#:   SMPL neck   +615 mm                  <-  TA.ProxEndPos  +585 mm
#:   SMPL spine1 +215 mm                  <-  nothing: the model has no lumbar landmark
#:   SMPL head   +704 mm                  <-  nothing: every HE point sits near +840 mm
#:
#: A joint with no centre still contributes its rotation.  Inventing a position target for it
#: would put a number where a measurement does not exist: TA.ProxEndPos taken as "the waist"
#: leaves a 439 mm residual on pelvis->spine1 and drags the fitted stature 134 mm past the
#: subject's own recorded height.
CORRESPONDENCE: dict[int, tuple[str | None, str | None, str]] = {
    0:  ("PV",  "PELVIS", MEASURED),
    1:  ("LTH", "LHJC",   MEASURED),
    2:  ("RTH", "RHJC",   MEASURED),
    3:  (None,  None,     DERIVED_LUMBAR),
    4:  ("LSK", "LKJC",   MEASURED),
    5:  ("RSK", "RKJC",   MEASURED),
    6:  (None,  None,     DERIVED_LUMBAR),
    7:  ("LFT", "LAJC",   MEASURED),
    8:  ("RFT", "RAJC",   MEASURED),
    9:  (None,  "THORAX_BASE", DERIVED_LUMBAR),  # TA's caudal end, at SMPL spine3's height
    10: (None,  None,     ABSENT),   # toes: no segment distal to the foot
    11: (None,  None,     ABSENT),
    12: (None,  "NECK",   ABSENT),   # centre is the thorax's cranial end; rotation unmeasured
    13: (None,  None,     ABSENT),   # collars: the model goes thorax -> upper arm directly
    14: (None,  None,     ABSENT),
    15: ("HE",  None,     MEASURED),  # rotation measured here, unlike AddBiomechanics
    16: ("LUA", "LSJC",   MEASURED),
    17: ("RUA", "RSJC",   MEASURED),
    18: ("LFA", "LEJC",   MEASURED),
    19: ("RFA", "REJC",   MEASURED),
    20: ("LHA", "LWJC",   MEASURED),
    21: ("RHA", "RWJC",   MEASURED),
    22: (None,  None,     ABSENT),   # hand tips
    23: (None,  None,     ABSENT),
}


#: Added by --distal-centres.  pose_fit fits a joint's own alignment constant only when the
#: joint has a descendant carrying a centre, so the ankles and wrists inherit unless SMPL's toes
#: and hand tips have one.  HKNU supplies these points: RFT.DistEndPos is Marker.RTOE to the
#: metre, RHA.DistEndPos sits at the finger, and both are rigid in their own segment's frame
#: (sd 0 mm).  It is the same thing AddBiomechanics gets
#: from `mtp_l`/`mtp_r`, which is why its ankles do not inherit.
#:
#: The body listed is the foot/hand itself, because HKNU has no separate toe or hand-tip
#: segment; the emitted local rotation at these joints then comes out identity, which is right
#: for a source whose foot is one rigid body.  The `measured` label overstates it -- the
#: position is measured, the rotation is not -- but pose_fit's target filter keys on provenance
#: (pose_fit.py:222-223), so the table has no way to say that.
DISTAL_CORRESPONDENCE: dict[int, tuple[str, str, str]] = {
    10: ("LFT", "LTOE", MEASURED),
    11: ("RFT", "RTOE", MEASURED),
    22: ("LHA", "LHAND", MEASURED),
    23: ("RHA", "RHAND", MEASURED),
}


def build_table(pelvis_variant: str = "anatomical", distal: bool = True
                ) -> tuple[SmplJointSource, ...]:
    table = dict(CORRESPONDENCE)
    if distal:
        table.update(DISTAL_CORRESPONDENCE)
    if pelvis_variant == "three_target":
        table[3] = (None, "SPINE1_PROXY", DERIVED_LUMBAR)
    if pelvis_variant == "pelvis_axis":
        table[3] = (None, "SPINE1_AXIS", DERIVED_LUMBAR)
    return tuple(
        SmplJointSource(
            smpl_index=index,
            smpl_name=SMPL24_NAMES[index],
            provenance=provenance,
            centre_joint=centre,
            source_coordinates=(body,) if body else (),
            source_body=TRUNK_ALIAS if provenance == DERIVED_LUMBAR and body is None else body,
        )
        for index, (body, centre, provenance) in sorted(table.items())
    )


def body_map(distal: bool = True) -> dict[int, str]:
    table = dict(CORRESPONDENCE)
    if distal:
        table.update(DISTAL_CORRESPONDENCE)
    return {i: b for i, (b, _, p) in table.items() if b and p == MEASURED}


def stack(matrix) -> np.ndarray:
    return np.moveaxis(np.asarray(matrix, float), 2, 0)


def _mean_frame(frames: np.ndarray) -> np.ndarray:
    """One rotation standing for a static trial's 501 near-identical frames."""
    return Rotation.from_matrix(np.asarray(frames, float)).mean().as_matrix()


def centres_from(mat, pelvis_variant: str = "anatomical",
                 distal: bool = True) -> dict[str, np.ndarray]:
    """Every position target, as (T, 3).

    Landmark joint centres are used rather than segment end points: at the hip and shoulder
    the parent's distal end sits 160 mm and 250 mm from the joint, being a midpoint of the
    pelvis and thorax rather than the joint itself.
    """
    landmark, segment = mat["Landmark"], mat["Segment"]
    out = {name: np.asarray(getattr(landmark, name), float).T
           for name in getattr(landmark, "_fieldnames", [])}
    out["PELVIS"] = np.asarray(segment.PV.ProxEndPos, float).T
    out["NECK"] = np.asarray(segment.TA.ProxEndPos, float).T
    out["THORAX_BASE"] = np.asarray(segment.TA.DistEndPos, float).T
    if distal:
        out["LTOE"] = np.asarray(segment.LFT.DistEndPos, float).T
        out["RTOE"] = np.asarray(segment.RFT.DistEndPos, float).T
        out["LHAND"] = np.asarray(segment.LHA.DistEndPos, float).T
        out["RHAND"] = np.asarray(segment.RHA.DistEndPos, float).T
    if pelvis_variant == "pelvis_axis":
        # The pelvis's two alignment targets are its hip centres, and they are nearly collinear:
        # they pin the lateral axis and leave the pitch about it free.  Measured in the N-pose,
        # the fitted trunk then leans 21-23 deg while the hip axis itself lands within 2.7 deg,
        # and the neck's distance from the pelvis is right to within 22 mm -- a direction error,
        # not a size error.
        #
        # The third direction comes from the pelvis segment's own long axis, which the source
        # does measure.  Only the direction carries information here; how far along it spine1
        # sits is taken as the segment's own length, so no number is invented.
        axis = (np.asarray(segment.PV.ProxEndPos, float)
                - np.asarray(segment.PV.DistEndPos, float)).T
        out["SPINE1_AXIS"] = out["PELVIS"] + axis
    if pelvis_variant == "three_target":
        # Diagnostic only.  _targets_of gives the pelvis its children with a centre, and here
        # that is just the two hips -- two directions, which pin the lateral axis and leave the
        # pitch about it free, and which also drop below the three _root_alignment needs before
        # it will fit a translation at all.  Moving the pelvis onto the hip midpoint and letting
        # PV.ProxEndPos stand in for spine1 buys a third target; it is anatomically wrong (SMPL
        # puts spine1 215 mm above the hips, PV.ProxEndPos is 81 mm) and is measured here only
        # to test whether the target count is what drives the upper-body residual.
        out["PELVIS"] = 0.5 * (out["RHJC"] + out["LHJC"])
        out["SPINE1_PROXY"] = np.asarray(segment.PV.ProxEndPos, float).T
    return out


#: What the rest skeleton is scaled to, as SMPL edges -> the two N-pose landmarks measuring them.
#: Only bones the source actually measures at both ends appear; nothing here invents a length.
RESCALE_BONES: dict[tuple[int, int], tuple[str, str]] = {
    (1, 4): ("LHJC", "LKJC"), (2, 5): ("RHJC", "RKJC"),
    (4, 7): ("LKJC", "LAJC"), (5, 8): ("RKJC", "RAJC"),
    (7, 10): ("LAJC", "LTOE"), (8, 11): ("RAJC", "RTOE"),
    (16, 18): ("LSJC", "LEJC"), (17, 19): ("RSJC", "REJC"),
    (18, 20): ("LEJC", "LWJC"), (19, 21): ("REJC", "RWJC"),
    (20, 22): ("LWJC", "LHAND"), (21, 23): ("RWJC", "RHAND"),
}

#: The trunk is scaled as one path, pelvis -> neck. Its four offsets are not measured separately
#: -- the source has no lumbar landmark at all -- so scaling them individually would invent a
#: spine. `PELVIS`/`NECK` are PV's and TA's cranial ends, the pair `shape_fit` fits as one bone.
RESCALE_CHAINS: dict[tuple[int, int], tuple[str, str]] = {(0, 12): ("PELVIS", "NECK")}

#: The hips: the source measures how far apart they are, not where either one is relative to a
#: pelvis origin the two skeletons do not share. Ten betas reach 104-135 mm against a measured
#: 180-263, so this is the one number the shape fit cannot deliver however it is weighted.
#: and the shoulders, for the same reason and with the same evidence: the source measures the
#: separation of the two joint centres, and SMPL's rest pair is 132 mm narrower on the median
#: subject, which puts both arms that much too medial and every arm landmark with them.
RESCALE_SPANS: dict[tuple[int, int], tuple[str, str]] = {
    (1, 2): ("LHJC", "RHJC"), (16, 17): ("LSJC", "RSJC"),
}


def rescale_rest_to_measured(rest_joints, npose_centres) -> tuple[np.ndarray, dict]:
    """The subject's rest skeleton on their own measured lengths, directions untouched.

    Every target is a median over the N-pose frames of the distance between two stored joint
    centres, which is how `shape_fit.measured_segment_lengths` reads a bone; the difference is
    only that here the length is set rather than asked for.
    """
    def distance(near: str, far: str) -> float:
        return float(np.median(np.linalg.norm(
            np.asarray(npose_centres[far], float) - np.asarray(npose_centres[near], float),
            axis=-1)))

    return rescale_rest_joints(
        rest_joints,
        bone_targets={edge: distance(*pair) for edge, pair in RESCALE_BONES.items()
                      if pair[0] in npose_centres and pair[1] in npose_centres},
        chain_targets={edge: distance(*pair) for edge, pair in RESCALE_CHAINS.items()},
        span_targets={edge: distance(*pair) for edge, pair in RESCALE_SPANS.items()},
    )


#: The joints whose measured centre the root is solved against every frame: the hips, knees,
#: ankles and shoulders.
#:
#: Which of them, measured over 36 walking takes. "off" is the constant
#: CoM offset, the rest are RMS joint-centre residuals in mm:
#:
#:   root solved against            R2 AP  R2 vt   |r|    off  knee  ankle  shldr
#:   the hip centres alone          0.693  0.894  0.088  36.9   9.4   11.6   56.7
#:   + knees and ankles             0.786  0.867  0.082  35.6   7.6   10.0   58.5
#:   + shoulders            <- this 0.854  0.892  0.066  27.8  19.7   22.0   40.2
#:   + elbows and wrists            0.877  0.900  0.058  33.2  36.0   37.8   22.8
#:   the laboratory's own body      0.854  0.897  0.061     -     -      -      -
#:
#: There is a real trade here and it is not subtle: every joint added pulls the root toward the
#: joints that are hardest to reach, so the balance improves and the legs get worse. The elbows
#: and wrists hang off a shoulder girdle whose SMPL geometry is about 50 mm from the source's,
#: and no root offset can fix that -- including them spends the legs on it and puts the knee at
#: 36 mm, past the 35 mm this probe is held to. Stopping at the shoulders is the one set that
#: clears every acceptance criterion with margin, and its balance R2 lands on the laboratory's.
#:
#: The better next step is to stop compensating and fix the girdle: the source measures
#: `THORAX_BASE -> SJC`, so the collar-to-shoulder chain could be scaled the way the limbs are.
#: Not done here.
TRACKED_JOINTS = (1, 2, 4, 5, 7, 8, 16, 17)


def tracked_centres(distal: bool = True) -> tuple[tuple[int, str], ...]:
    table = dict(CORRESPONDENCE)
    if distal:
        table.update(DISTAL_CORRESPONDENCE)
    return tuple((index, table[index][1]) for index in TRACKED_JOINTS)


def root_placement_from(static_mat, rest_joints, tracked=()) -> RootPlacement:
    """The two anatomical directions that fix the pelvis, and the landmark it must hit.

    The pelvis is the one bone whose constant the child offsets cannot decide here.  SMPL's
    pelvis joint and Visual3D's `PV.ProxEndPos` are not the same point -- measured on this
    cohort's N-poses, SMPL's sits 18 mm in front of the hip-centre midpoint and the source's
    sits 44 mm in front -- and a Kabsch fit given only the two hips answers that 26 mm
    displacement with a rotation about the hip axis, because that is the one axis two hip
    offsets say nothing about.  It comes out as 23 deg of forward pitch, and since the lumbar
    rule turns the pelvis into the trunk, the whole upper body leans with it.

    So the pelvis is given directions instead, one per axis it needs:

      lateral    LHJC - RHJC against SMPL's own left hip - right hip.  Both skeletons agree
                 this is the lateral axis; they disagree about its *length* by about 100 mm on
                 this cohort, which is why `RootPlacement` normalises before fitting.
      superior   PV.ProxEndPos -> TA.ProxEndPos against SMPL's pelvis -> neck.  This is the
                 trunk axis, and it is the pair `shape_fit` already treats as one bone.

    The superior direction is the trunk's and not the pelvis segment's own long axis, which was
    the first thing tried.  SMPL's rest skeleton does not agree with itself about which way is
    up: pelvis -> spine1 leans 17 deg back of vertical while pelvis -> neck leans 4.6 deg, 12.9
    deg apart.  Matching the source's pelvis long axis to pelvis -> spine1 therefore left the
    trunk 13 deg forward and the cohort's lean unchanged at 24.6 deg (measured).  The trunk axis
    is also the one that decides where the upper body's mass sits, which is what the CoM and the
    balance residual are about.

    Read off the *static N-pose*, not the trial: the trunk axis is rigid in the pelvis only
    while the subject stands, and this is the same trial and the same assertion the head's
    constant already rests on -- that in a standing take these joints are at their rest angle.

    The translation then puts SMPL's hip-centre midpoint on the measured one: the best-located
    landmark pair the source has near the root, and the pair that carries the legs.
    """
    rest = np.asarray(rest_joints, float)
    segment = static_mat["Segment"]
    pelvis_rotation = stack(segment.PV.SegRot)                     # (T,3,3) segment -> world
    landmark = static_mat["Landmark"]

    def in_pelvis(world_vector: np.ndarray) -> np.ndarray:
        return np.einsum("tji,tj->ti", pelvis_rotation, world_vector).mean(axis=0)

    lateral = (np.asarray(landmark.LHJC, float) - np.asarray(landmark.RHJC, float)).T
    superior = (np.asarray(segment.TA.ProxEndPos, float)
                - np.asarray(segment.PV.ProxEndPos, float)).T
    return RootPlacement(
        rest_directions=np.stack([rest[1] - rest[2], rest[12] - rest[0]]),
        measured_directions=np.stack([in_pelvis(lateral), in_pelvis(superior)]),
        anchor_joints=(1, 2),
        anchor_centres=("LHJC", "RHJC"),
        tracked=tuple(tracked),
    )


def rotations_from(mat) -> dict[str, np.ndarray]:
    segment = mat["Segment"]
    out = {name: stack(getattr(segment, name).SegRot)
           for name in getattr(segment, "_fieldnames", [])}
    out[TRUNK_ALIAS] = out["TA"]
    return out


def static_pose_geometry(mat) -> dict:
    """Is the T-pose a T and the N-pose an N, for every subject?"""
    centres = centres_from(mat)
    out = {}
    for side in ("R", "L"):
        shoulder = centres[f"{side}SJC"].mean(axis=0)
        wrist = centres[f"{side}WJC"].mean(axis=0)
        vector = wrist - shoulder
        out[side] = {
            "shoulder_to_wrist_m": [round(float(v), 4) for v in vector],
            "length_m": round(float(np.linalg.norm(vector)), 4),
            "vertical_drop_m": round(float(vector[2]), 4),
            "horizontal_fraction": round(
                float(np.linalg.norm(vector[:2]) / max(np.linalg.norm(vector), 1e-9)), 4
            ),
        }
    return out


def probe_subject(subject: str, root: Path, gender: str, mass_kg: float, height_m: float,
                  trial: str, max_frames: int, pelvis_variant: str = "anatomical",
                  landmark_offsets: str | None = None, distal: bool = True,
                  reference_pose: str | None = "Npose", repaired: bool = False) -> dict:
    folder = root / "Dataset_Processed" / subject
    table = build_table(pelvis_variant, distal)
    result: dict = {"subject": subject, "gender": gender}

    # static poses
    statics = {}
    for name in ("Npose", "Tpose"):
        mat = sio.loadmat(folder / f"{subject}_{name}.mat",
                          struct_as_record=False, squeeze_me=True)
        statics[name] = static_pose_geometry(mat)
        if name == "Npose":
            npose_centres = centres_from(mat, pelvis_variant, distal)
            static_mat = mat        # kept whole: the pelvis placement reads rotations too
        if name == reference_pose:
            # SMPL's head sits at the end of the chain with no child to aim at, so no offset can
            # place it; the standing trial is where its angle is known to be neutral, and the
            # frame difference read there is the constant.  N-pose rather than T-pose: the arms
            # are down, so the shoulders are nowhere near their limits and the trial is the one
            # the source itself calibrated IMU2SegRot from (residual 0.49 deg).
            reference = {k: _mean_frame(v) for k, v in rotations_from(mat).items()}
    result["static_poses"] = statics
    if reference_pose is None:
        reference = None

    # shape fit.  Bone lengths come from the N-pose, where the subject stands still and the
    # marker set is least occluded; the model is rigid so any trial gives the same numbers
    # (length sd 3.7e-08 m).
    model = load_clean_model(clean_model_path_for_gender(gender))
    pairs = segment_pairs_from(table)
    targets = measured_segment_lengths(pairs, npose_centres)
    shape = fit_betas(
        model, targets, gender=gender, stature_m=height_m,
        mass_kg=mass_kg if model.get("faces") is not None else None,
        landmark_offsets=landmark_offsets,
    )
    rest = smpl_rest_joints(model, shape.betas)
    # `repaired` is what the generator does: the rest skeleton scaled onto the measured
    # lengths, the pelvis placed from directions, the lumbar zeroed on the standing take. Off by
    # default; a run without it measures the unrepaired fit, not the one in the corpus.
    if repaired:
        rest, result["rest_rescale"] = rescale_rest_to_measured(rest, npose_centres)
    result["shape"] = {
        "segments_measured": len(targets),
        "betas": [round(float(b), 4) for b in shape.betas],
        "residual_m": float(shape.residual_m),
        "worst_segments_m": dict(
            sorted(((f"{a}-{b}", round(float(v), 5))
                    for (a, b), v in shape.segment_residuals.items()),
                   key=lambda kv: -abs(kv[1]))[:5]
        ),
        "target_stature_m": height_m,
        "target_mass_kg": mass_kg,
        "fitted_stature_m": shape.stature_m,
        "fitted_mass_kg": shape.mass_kg,
    }

    # pose fit, through the module the conversion will reuse, unmodified.
    # "ALL" fits every trial of the subject so the residual can be quoted as a distribution
    # against the same statistic the AddBiomechanics retarget is quoted with;
    # one trial per subject cannot say whether a number is typical.
    names = ([p.stem.replace(f"{subject}_", "", 1)
              for p in sorted(folder.glob(f"{subject}_*.mat"))
              if not p.name.endswith("_SubjectInfo.mat")] if trial == "ALL" else [trial])
    poses = []
    for name in names:
        mat = sio.loadmat(folder / f"{subject}_{name}.mat",
                          struct_as_record=False, squeeze_me=True)
        frames = int(np.asarray(mat["Info"].Time, float).size)
        step = max(1, frames // max_frames)
        wanted = np.arange(0, frames, step)
        rotations = {k: v[wanted] for k, v in rotations_from(mat).items()}
        centres = {k: v[wanted] for k, v in centres_from(mat, pelvis_variant, distal).items()}
        fit = fit_pose(
            rest_joints=rest,
            correspondence=table,
            body_of=body_map(distal),
            source_rotations=rotations,
            source_centres=centres,
            world_rotation=np.eye(3),   # the source frame is already Z-up
            reference_rotations=reference,
            root_placement=root_placement_from(static_mat, rest) if repaired else None,
            lumbar_from_reference=repaired,
        )
        poses.append({
            "trial": name,
            "frames_fitted": int(len(wanted)),
            "position_error_m": float(fit.position_error_m),
            "per_joint_error_mm": {
                SMPL24_NAMES[i]: round(float(v) * 1000, 2)
                for i, v in sorted(fit.per_joint_error_m.items(), key=lambda kv: -kv[1])
            },
            "provenance_counts": {
                kind: int(sum(p == kind for p in fit.joint_provenance))
                for kind in (MEASURED, DERIVED_LUMBAR, ABSENT)
            },
            "root_offset_mm": [round(float(v) * 1000, 2) for v in np.ravel(fit.root_offset)]
            if fit.root_offset is not None else None,
        })
    result["poses"] = poses
    result["pose"] = poses[0]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="the HKNU source folder; default $SOMA_SOURCE_ROOT/hknu_fullbody")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial", default="Walking_4KPH_0",
                        help="a trial name, or ALL for every trial of every subject")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--pelvis-variant", choices=("anatomical", "three_target", "pelvis_axis"),
                        default="anatomical",
                        help="three_target is a diagnostic: see centres_from")
    parser.add_argument("--no-reference-pose", dest="reference_pose", action="store_const",
                        const=None, default="Npose",
                        help="do not hand pose_fit a static trial, so the joints it cannot fit "
                             "keep an ancestor's constant")
    parser.add_argument("--no-distal-centres", dest="distal_centres", action="store_false",
                        help="give the toes and hand tips no centres, so the ankles and "
                             "wrists inherit their constants (for comparison)")
    parser.set_defaults(distal_centres=True)
    parser.add_argument("--repaired", action="store_true",
                        help="run the fit the generator uses (rest scaled to the measured "
                             "lengths, pelvis from directions, lumbar zeroed on the standing "
                             "take). Off runs the unrepaired fit")
    parser.add_argument("--landmark-offsets", default=None,
                        help="apply shape_fit's marker-centre-to-SMPL-joint offsets, e.g. v1. "
                             "Calibrated on AddBiomechanics, so on this source it is a test of "
                             "transfer, not a setting known to be right")
    args = parser.parse_args()
    if args.root is None:
        args.root = paths.source_dir("hknu")

    import openpyxl
    workbook = openpyxl.load_workbook(
        args.root / "MATLAB/DatasetInfo.xlsx", read_only=True, data_only=True
    )
    rows = list(workbook["Subject"].iter_rows(values_only=True))
    workbook.close()
    header = [str(c) for c in rows[0]]
    demographics = {
        str(row[header.index("SubjectID")]): {
            "gender": normalise_biological_sex(str(row[header.index("Gender")])),
            "mass_kg": float(row[header.index("BodyMass")]),
            "height_m": float(row[header.index("BodyHeight")]),
        }
        for row in rows[1:] if row[header.index("SubjectID")]
    }

    table = build_table(args.pelvis_variant, args.distal_centres)
    print("=" * 78)
    print("HKNU Visual3D 15 segments -> SMPL-24 correspondence")
    print("=" * 78)
    counts = {MEASURED: 0, DERIVED_LUMBAR: 0, ABSENT: 0}
    for entry in table:
        counts[entry.provenance] += 1
        print(f"  {entry.smpl_index:2d} {entry.smpl_name:15s} {entry.provenance:15s} "
              f"body={str(entry.source_body):8s} centre={entry.centre_joint}")
    print(f"\n  totals: {counts}   (24 joints accounted for: "
          f"{sum(counts.values()) == 24})")
    print("  root = PV by construction; TA is its child, resolving the "
          "PV<->TA parent cycle in the Segment sheet")
    print(f"  reuse: trunk rotations handed to pose_fit under the alias {TRUNK_ALIAS!r}")

    subjects = args.subjects or sorted(demographics)
    results = []
    for subject in subjects:
        info = demographics[subject]
        record = probe_subject(subject, args.root, info["gender"], info["mass_kg"],
                               info["height_m"], args.trial, args.max_frames,
                               args.pelvis_variant, args.landmark_offsets,
                               args.distal_centres, args.reference_pose, args.repaired)
        results.append(record)
        shape = record["shape"]
        errors = [p["position_error_m"] * 1000 for p in record["poses"]]
        print(f"  {subject} {info['gender']:7s} shape_res={shape['residual_m'] * 1000:6.2f}mm "
              f"stature {shape['target_stature_m']:.3f}->{shape['fitted_stature_m']:.3f} "
              f"trials={len(errors):3d} pose_err median={np.median(errors):6.1f} "
              f"max={max(errors):6.1f} mm", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "correspondence": [
            {"index": e.smpl_index, "name": e.smpl_name, "provenance": e.provenance,
             "body": e.source_body, "centre": e.centre_joint} for e in table
        ],
        "provenance_counts": counts,
        "trunk_alias": TRUNK_ALIAS,
        "pelvis_variant": args.pelvis_variant,
        "landmark_offsets": args.landmark_offsets,
        "distal_centres": args.distal_centres,
        "reference_pose": args.reference_pose,
        "repaired": args.repaired,
        "subjects": results,
    }
    (args.out / "retarget_probe.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\nwrote {args.out / 'retarget_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
