"""Fit SMPL shape to a subject, from the bones the source actually measured.

Bone lengths come from the stored joint centres and are pose-invariant, so shape is a
per-subject constant fitted once rather than per frame. The `.osim` scale factors are not used:
they were measured uniform at 0.2 across every segment, which is a mesh display scale and not
anthropometry.

Two secondary terms exist because bone lengths alone do not span the shape space -- girth moves
`betas` without moving any joint centre. Stature comes from the shaped mesh's vertical extent,
mass from its enclosed volume times a stated density. Both are weak by default and both are
recorded in the fit's provenance, along with whether they were used at all.

Gender is a decision, not a default. AddBiomechanics writes `female`, `male`, `f` and `unknown`,
and an unreadable field resolves to `None` here so the caller has to choose and record what it
did. Nothing silently becomes a man -- which is what the AMASS generator does, loading the male model
for anything that is not female.

For this cohort the choice is `neutral`, and it is a faithful reading rather than a guess: the
twenty subjects concerned say `unknown` in the field itself, and SMPL ships a neutral body for
exactly that. Their sex is not recorded anywhere we hold -- not in any other header field, not
in the 1 MB of embedded OpenSim XML, and not beside the file, since the source tree contains
nothing but `.b3d`. Inventing it from the publication would mean pairing paper rows to `p1..p10`
with a mapping the archive does not provide, which the attribution registry forbids.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from smpl18.model import select as smpl18_select

from soma_synth.pipeline import paths as data_paths

from .smpl_correspondence import SMPL24_PARENTS, SmplJointSource

__all__ = [
    "BODY_DENSITY_KG_M3",
    "GENDER_FEMALE",
    "GENDER_MALE",
    "GENDER_NEUTRAL",
    "LANDMARK_OFFSETS",
    "LANDMARK_OFFSETS_V1",
    "ShapeFit",
    "UnresolvedGender",
    "clean_model_path_for_gender",
    "fit_betas",
    "load_clean_model",
    "measured_segment_lengths",
    "normalise_biological_sex",
    "rescale_rest_joints",
    "segment_pairs_from",
    "smpl_mass_kg",
    "smpl_rest_joints",
    "smpl_segment_lengths",
    "smpl_shaped_vertices",
    "smpl_stature",
    "smpl_volume_m3",
]

GENDER_MALE = "male"
GENDER_FEMALE = "female"
GENDER_NEUTRAL = "neutral"

#: Whole-body density. A stated assumption, not a measurement: published values sit between
#: about 985 and 1050 kg/m3 depending on composition, so the mass term is deliberately weak.
BODY_DENSITY_KG_M3 = 1000.0

#: SMPL's own rest frame is Y-up, independent of whatever the source's world frame does.
_SMPL_UP_AXIS = 1

_DEFAULT_STATURE_WEIGHT = 0.3
_DEFAULT_MASS_WEIGHT = 0.05
_DEFAULT_REGULARISATION = 1e-3

#: How far SMPL's idea of a bone sits from OpenSim's, in metres, as `smpl - measured`.
#:
#: OpenSim and SMPL do not put "the hip" or "the shoulder" in the same place, so fitting betas
#: to raw OpenSim segment lengths spends shape parameters papering over a landmark disagreement.
#: These are the medians measured over 94 subjects spanning every study and both variants, and
#: they are per-side on purpose: the released SMPL templates are themselves left-right
#: asymmetric by up to 7 mm, in different directions for the male and female models, so a
#: symmetric correction would be wrong on one side.
#:
#: Held out one study at a time -- estimated on the others, applied to the study withheld --
#: this cuts pooled bone-length rms from 22.9 mm to 15.3 mm and stature bias from +49 mm to
#: +31 mm, improving 19 of 23 held-out groups. The remaining +31 mm is not explained by this
#: and is not corrected away here.
LANDMARK_OFFSETS_V1: dict[tuple[int, int], float] = {
    (0, 1): -0.0150,     # pelvis -> left_hip
    (0, 2): -0.0107,     # pelvis -> right_hip
    (0, 3): -0.0081,     # pelvis -> spine1
    (1, 4): -0.0312,     # left_hip -> left_knee
    (2, 5): -0.0318,     # right_hip -> right_knee
    (4, 7): +0.0126,     # left_knee -> left_ankle
    (5, 8): +0.0079,     # right_knee -> right_ankle
    (7, 10): +0.0005,    # left_ankle -> left_foot
    (8, 11): +0.0045,    # right_ankle -> right_foot
    (16, 18): -0.0386,   # left_shoulder -> left_elbow
    (17, 19): -0.0378,   # right_shoulder -> right_elbow
    (18, 20): +0.0120,   # left_elbow -> left_wrist
    (19, 21): +0.0183,   # right_elbow -> right_wrist
}

LANDMARK_OFFSETS = {"v1": LANDMARK_OFFSETS_V1}


class UnresolvedGender(ValueError):
    """The subject's recorded sex does not select a body model, and no fallback was given."""


def normalise_biological_sex(
    value: str | None, *, strict: bool = False, fallback: str | None = None
) -> str | None:
    """Map the header's `biological_sex` onto a body model, or admit that it does not."""
    initial = str(value or "").strip().lower()[:1]
    if initial == "m":
        return GENDER_MALE
    if initial == "f":
        return GENDER_FEMALE
    if fallback is not None:
        return fallback
    if strict:
        raise UnresolvedGender(
            f"biological_sex {value!r} selects no body model; choose a fallback and record it"
        )
    return None


def clean_model_path_for_gender(
    gender: str, *, root: str | pathlib.Path | None = None
) -> pathlib.Path:
    """Where the extracted, redistribution-free body model for a gender lives.

    The licence-gated `.pkl` is never touched here; only the clean npz that
    scripts/poc/smpl_model.py produced from it under a whitelisting unpickler.

    Which file a gender selects is `smpl18`'s to say (`smpl18.model.select`): that package owns
    the SMPL-24 conversion and its model convention, and the pipeline records the set it resolves
    (`configs/datasets/source_pipelines_v1.yaml` body_model_sets, pipeline/body_models.py). Only
    the directory is decided here: `<root>/body_models/smpl` when a data root is passed, else
    `pipeline.paths.body_model_dir()` (`SOMA_DATA_ROOT/body_models/smpl`, the folder the runner
    hashes; `SOMA_BODY_MODEL_DIR` is deferred until the runner can record another).
    """
    if gender not in (GENDER_MALE, GENDER_FEMALE, GENDER_NEUTRAL):
        raise UnresolvedGender(f"no body model is defined for gender {gender!r}")
    base = (pathlib.Path(root) / "body_models" / "smpl") if root else data_paths.body_model_dir()
    try:
        return smpl18_select.model_path_for_gender(gender, base)
    except smpl18_select.UnresolvedGender as error:      # the same refusal, this module's class
        raise UnresolvedGender(str(error)) from None


def load_clean_model(path: str | pathlib.Path) -> dict:
    """Load an extracted SMPL npz.

    The triangle list is accepted under either of two keys: `faces`, or `f` as in the original
    SMPL files. The male and female models need not use the same one, and reading only one
    spelling would drop the mesh for the other gender and, with it, the mass term -- quietly
    fitting men and women by different criteria. `faces_key` records which key was found, so a
    difference between the models stays visible in provenance.
    """
    path = pathlib.Path(path)
    with np.load(path, allow_pickle=False) as data:
        model = {
            "v_template": np.asarray(data["v_template"], dtype=np.float64),
            "shapedirs": np.asarray(data["shapedirs"], dtype=np.float64),
            "J_regressor": np.asarray(data["J_regressor"], dtype=np.float64),
        }
        for key in ("faces", "f"):
            if key in data.files:
                model["faces"] = np.asarray(data[key], dtype=np.int64)
                model["faces_key"] = key
                break
        model["parents"] = np.asarray(
            data["kintree_parents"] if "kintree_parents" in data.files else SMPL24_PARENTS,
            dtype=np.int64,
        )
    return model


def smpl_shaped_vertices(model: dict, betas) -> np.ndarray:
    """v_template + shapedirs . betas, with betas padded or truncated to the model's width."""
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    width = shapedirs.shape[2]
    padded = np.zeros(width, dtype=np.float64)
    given = np.asarray(betas, dtype=np.float64).ravel()
    take = min(width, given.size)
    padded[:take] = given[:take]
    return np.asarray(model["v_template"], dtype=np.float64) + np.einsum(
        "vij,j->vi", shapedirs, padded
    )


def smpl_rest_joints(model: dict, betas) -> np.ndarray:
    return np.asarray(model["J_regressor"], dtype=np.float64) @ smpl_shaped_vertices(
        model, betas
    )


def smpl_segment_lengths(
    model: dict, betas, pairs: list[tuple[int, int]] | None = None
) -> dict[tuple[int, int], float]:
    """Distance between joint pairs in the rest pose, in metres.

    Defaults to every edge of the tree. Explicit pairs may also span several joints: the
    pelvis-to-shoulder distance is a shape quantity the source can measure at both ends, and
    it is the only handle on torso length, since nothing between the two is measured at all.
    """
    joints = smpl_rest_joints(model, betas)
    if pairs is None:
        pairs = [
            (int(parent), child)
            for child, parent in enumerate(SMPL24_PARENTS)
            if parent >= 0 and child < joints.shape[0] and parent < joints.shape[0]
        ]
    return {
        (int(a), int(b)): float(np.linalg.norm(joints[b] - joints[a]))
        for a, b in pairs
        if a < joints.shape[0] and b < joints.shape[0]
    }


def smpl_stature(model: dict, betas) -> float:
    """Vertical extent of the shaped mesh in SMPL's own Y-up rest frame."""
    vertices = smpl_shaped_vertices(model, betas)
    return float(
        vertices[:, _SMPL_UP_AXIS].max() - vertices[:, _SMPL_UP_AXIS].min()
    )


def smpl_volume_m3(model: dict, betas) -> float:
    """Volume enclosed by the shaped mesh, independent of face winding."""
    faces = model.get("faces")
    if faces is None:
        raise ValueError("the body model carries no faces; volume needs a closed mesh")
    vertices = smpl_shaped_vertices(model, betas)
    triangles = vertices[np.asarray(faces, dtype=np.int64)]
    signed = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    )
    return float(abs(signed.sum()) / 6.0)


def smpl_mass_kg(model: dict, betas, *, density_kg_m3: float = BODY_DENSITY_KG_M3) -> float:
    return smpl_volume_m3(model, betas) * density_kg_m3


#: Pairs that are not a single bone but whose distance is still a shape the source measures.
#: Nothing between the pelvis and the shoulder is measured -- no neck, no collar, and the three
#: spine joints are a rule rather than a measurement -- so without these the torso's length is
#: unconstrained and the arms end up wherever the mean body happens to put them.
_TORSO_PATHS = ((0, 16), (0, 17))


def segment_pairs_from(
    correspondence: tuple[SmplJointSource, ...],
    *,
    include_torso_paths: bool = True,
) -> dict[tuple[int, int], tuple[str, str]]:
    """Joint pairs with a stored centre at both ends: every tree edge, plus the torso paths."""
    centres = {entry.smpl_index: entry.centre_joint for entry in correspondence}
    pairs: dict[tuple[int, int], tuple[str, str]] = {}
    candidates = [
        (int(parent), child)
        for child, parent in enumerate(SMPL24_PARENTS)
        if parent >= 0
    ]
    if include_torso_paths:
        candidates += list(_TORSO_PATHS)
    for near_index, far_index in candidates:
        near, far = centres.get(near_index), centres.get(far_index)
        if near and far:
            pairs[(near_index, far_index)] = (near, far)
    return pairs


def measured_segment_lengths(
    pairs: dict[tuple[int, int], tuple[str, str | None]],
    centres: dict[str, np.ndarray],
) -> dict[tuple[int, int], float]:
    """Per-segment length as the median over frames.

    The median rather than a single frame because not every segment is rigid -- hip to knee
    slides with the walker knee, and ankle to toe opens with the subtalar -- and because one
    corrupt frame should not set a subject's thigh length for good.
    """
    lengths: dict[tuple[int, int], float] = {}
    for segment, (near, far) in pairs.items():
        if not near or not far or near not in centres or far not in centres:
            continue
        distance = np.linalg.norm(
            np.asarray(centres[far], dtype=np.float64)
            - np.asarray(centres[near], dtype=np.float64),
            axis=-1,
        )
        if distance.size:
            lengths[segment] = float(np.median(distance))
    return lengths


def rescale_rest_joints(
    rest_joints,
    *,
    parents=SMPL24_PARENTS,
    bone_targets: dict[tuple[int, int], float] | None = None,
    chain_targets: dict[tuple[int, int], float] | None = None,
    span_targets: dict[tuple[int, int], float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Put the rest skeleton on the measured lengths, keeping every direction it already has.

    `fit_betas` spends ten shape parameters on a body that has to satisfy bone lengths, stature
    and mass at once, and for some cohorts the space does not contain the answer. On HKNU the
    measured hip-centre separation is 180-263 mm while ten betas can only reach 104-135, so the
    fit lands 105 mm narrow on the median subject and each hip is 52 mm too medial before a
    single frame is posed. The limb bones come out 25 mm short by the same squeeze.

    Large is a skeleton and a set of rotations, so the lengths can simply be *set*. This scales
    each offset to what the source measured and leaves its direction alone, which keeps the rest
    posture -- and therefore every alignment constant fitted against it -- exactly where it was.

      bone_targets   one tree edge (parent, child) -> length. The offset is scaled.
      chain_targets  (ancestor, descendant) -> straight-line distance. Every offset along the
                     path is scaled by one factor, so the chain keeps its shape. Used for the
                     trunk, where the source measures the pelvis-to-neck distance but nothing
                     in between: scaling the four spine offsets separately would invent
                     lumbar geometry the source never saw.
      span_targets   (a, b) -> separation, for a pair whose distance is measured but whose
                     individual positions are not: the hips. Both move about their shared
                     midpoint along their own axis, so the pelvis neither rises nor tips.

    What this does not do is move the mesh. The vertices stay bound to the joints they were
    bound to, so a part follows its joint but keeps its own size; a skeleton-and-rotation
    artifact needs no more than that, and the residual it leaves is reported rather than
    hidden. Returns the new joints and a per-target record of the scale each one took.
    """
    rest = np.asarray(rest_joints, dtype=np.float64).copy()
    parents = np.asarray(parents, dtype=np.int64)
    offsets = {j: rest[j] - rest[int(parents[j])] for j in range(1, len(rest))}
    applied: dict[str, dict] = {"span": {}, "chain": {}, "bone": {}}

    def path(ancestor: int, descendant: int) -> list[int]:
        walk, node = [], descendant
        while node != ancestor:
            walk.append(node)
            node = int(parents[node])
            if node < 0:
                raise ValueError(f"joint {descendant} is not below joint {ancestor}")
        return list(reversed(walk))

    for (a, b), wanted in (span_targets or {}).items():
        # Both joints move by half the shortfall along the axis joining them, so the pair's
        # midpoint does not move and nothing above them tips. The displacement is exactly along
        # that axis, so the separation changes by exactly the shortfall whether or not the two
        # share a parent -- the shoulders hang off their own collars and do not.
        axis = rest[a] - rest[b]
        current = float(np.linalg.norm(axis))
        unit = axis / max(current, 1e-12)
        shift = 0.5 * (float(wanted) - current) * unit
        offsets[a] = offsets[a] + shift
        offsets[b] = offsets[b] - shift
        applied["span"][f"{a}-{b}"] = dict(before_m=current, after_m=float(wanted))

    chained: set[int] = set()
    for (a, b), wanted in (chain_targets or {}).items():
        walk = path(a, b)
        current = float(np.linalg.norm(sum(offsets[j] for j in walk)))
        scale = float(wanted) / max(current, 1e-12)
        for j in walk:
            offsets[j] = offsets[j] * scale
        chained.update(walk)
        applied["chain"][f"{a}-{b}"] = dict(before_m=current, after_m=float(wanted), scale=scale)

    for (near, far), wanted in (bone_targets or {}).items():
        if int(parents[far]) != near:
            raise ValueError(f"bone ({near}, {far}) is not a tree edge")
        if far in chained:
            raise ValueError(f"bone ({near}, {far}) is already inside a scaled chain")
        current = float(np.linalg.norm(offsets[far]))
        scale = float(wanted) / max(current, 1e-12)
        offsets[far] = offsets[far] * scale
        applied["bone"][f"{near}-{far}"] = dict(before_m=current, after_m=float(wanted),
                                                scale=scale)

    for joint in range(1, len(rest)):
        rest[joint] = rest[int(parents[joint])] + offsets[joint]
    return rest, applied


@dataclass(frozen=True)
class ShapeFit:
    betas: np.ndarray
    gender: str | None
    residual_m: float
    segment_residuals: dict[tuple[int, int], float]
    stature_m: float | None
    mass_kg: float | None
    provenance: dict = field(default_factory=dict)


def fit_betas(
    model: dict,
    targets: dict[tuple[int, int], float],
    *,
    n_betas: int = 10,
    gender: str | None = None,
    stature_m: float | None = None,
    mass_kg: float | None = None,
    regularisation: float = _DEFAULT_REGULARISATION,
    stature_weight: float = _DEFAULT_STATURE_WEIGHT,
    mass_weight: float = _DEFAULT_MASS_WEIGHT,
    landmark_offsets: str | None = None,
) -> ShapeFit:
    """Least squares from measured bone lengths (and optionally stature and mass) to betas.

    `landmark_offsets` names a correction table (see LANDMARK_OFFSETS). It is opt-in and
    recorded: applying it changes every beta in the corpus, so which run used it must be
    readable off the artifact rather than inferred from a code version.
    """
    if not targets:
        raise ValueError("no measured segments to fit to")
    if mass_kg is not None and model.get("faces") is None:
        raise ValueError("a mass term needs the body model's faces; none were supplied")

    if landmark_offsets is not None:
        if landmark_offsets not in LANDMARK_OFFSETS:
            raise ValueError(
                f"unknown landmark offset table {landmark_offsets!r}; "
                f"known: {sorted(LANDMARK_OFFSETS)}"
            )
        table = LANDMARK_OFFSETS[landmark_offsets]
        targets = {s: v + table.get(s, 0.0) for s, v in targets.items()}

    segments = sorted(targets)
    wanted = np.array([targets[s] for s in segments], dtype=np.float64)

    def residuals(betas: np.ndarray) -> np.ndarray:
        lengths = smpl_segment_lengths(model, betas, segments)
        parts = [np.array([lengths[s] for s in segments]) - wanted]
        if stature_m is not None:
            parts.append(
                np.array([stature_weight * (smpl_stature(model, betas) - stature_m)])
            )
        if mass_kg is not None:
            parts.append(
                np.array([mass_weight * (smpl_mass_kg(model, betas) / mass_kg - 1.0)])
            )
        if regularisation > 0.0:
            parts.append(np.sqrt(regularisation) * betas)
        return np.concatenate(parts)

    # trf, not lm: a No_Arm subject supplies nine bones for ten betas, and Levenberg-Marquardt
    # refuses outright when residuals are fewer than variables.
    solution = least_squares(residuals, np.zeros(n_betas, dtype=np.float64), method="trf")
    betas = np.asarray(solution.x, dtype=np.float64)

    final = smpl_segment_lengths(model, betas, segments)
    per_segment = {s: float(final[s] - targets[s]) for s in segments}
    worst = float(np.sqrt(np.mean(np.square(list(per_segment.values())))))

    return ShapeFit(
        betas=betas,
        gender=gender,
        residual_m=worst,
        segment_residuals=per_segment,
        stature_m=smpl_stature(model, betas),
        mass_kg=(smpl_mass_kg(model, betas) if model.get("faces") is not None else None),
        provenance={
            "segments_used": len(segments),
            "regularisation": regularisation,
            "stature_used": stature_m is not None,
            "stature_weight": stature_weight if stature_m is not None else None,
            "mass_used": mass_kg is not None,
            "mass_weight": mass_weight if mass_kg is not None else None,
            "body_density_kg_m3": BODY_DENSITY_KG_M3 if mass_kg is not None else None,
            "n_betas": int(n_betas),
            "landmark_offsets": landmark_offsets,
            "faces_key": model.get("faces_key"),
            "optimiser": "scipy.least_squares(trf)",
            "converged": bool(solution.success),
        },
    )
