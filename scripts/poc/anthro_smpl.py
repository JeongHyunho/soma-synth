"""Anthro (anthropometry) deliverable — SMPL rest-pose skeleton constants.

Per ``anthro/00_anthropometry.qmd`` the Anthro dataset holds subject-level,
trial/frame-invariant skeleton constants. Combined with Large's motion joints they
reconstruct the full SMPL 22-chain (24 joints minus the two hands) and the rigid
biomechanical skeleton:

* ``joint_position``      [2, 22, 3]  root-relative rest joint centres, two reference
                                      poses (0 = T-pose = SMPL rest; 1 = upright-neutral)
* ``segment_length``      [13]        major bone lengths (pose-invariant)
* ``fixed_joint_rotation``[4, 4]      spine1/spine2/left_collar/right_collar constants
* ``sex`` / ``height`` / ``body_mass``            measured subject anthropometry
* ``betas`` (+ ``gender``)            SMPL shape (source-derived reference layer)

**Frame.** ``joint_position`` is stored in the SMPL *canonical rest frame* (Y-up, arms
along X) — the frame the SMPL pose parameters are defined against. To place the skeleton
in the PRISM world (Z-up, spec frame ``G``) combine these rest offsets with Large's pelvis
global orientation via forward kinematics (see ``reconstruct_positions_frozen_fixed``).

**Fixed joints.** In the SMPL parameterisation spine1/spine2/left_collar/right_collar are
zero at *rest*, but the subject holds a habitual shoulder-girdle/trunk posture (e.g. the
collars sit ~24 deg off rest for this take). The SOMA Large model *freezes* these 4 joints
(folding trunk motion into ``spine3`` and treating the shoulder girdle as rigid); freezing at
identity leaves ~0.3 m of wrist reconstruction error. Which joints freeze, how the joint below
absorbs them and how the four constants are fitted all belong to ``smpl18.reduce``
(``packages/smpl18``); the pipeline fits them once per subject through
``soma_synth.pipeline.reduced_model`` and hands them to :func:`compute_anthro`. The
residual per-frame articulation the freeze discards is preserved losslessly in the raw SMPL
reference layer, not here.

Pure NumPy/scipy for the math; no body-model asset is required to import this module (the model
is passed in), so it is unit-testable on a tiny synthetic model. The two things it does not decide
for itself -- which file a gender selects (``smpl18.model.select``) and which joints the reduced
model freezes (``smpl18.skeleton.definition``) -- come from ``packages/smpl18``.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SMPL18_SRC = _REPO / "packages" / "smpl18" / "src"
if _SMPL18_SRC.is_dir() and str(_SMPL18_SRC) not in sys.path:
    sys.path.insert(0, str(_SMPL18_SRC))   # the same in-tree bootstrap the generators use for src/
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))          # soma_synth.pipeline.paths: where the models are

from smpl18.model import select as smpl18_select   # noqa: E402
from smpl18.skeleton import definition as smpl18_skeleton   # noqa: E402
from soma_synth.pipeline import paths as data_paths


# SMPL-24 kinematic tree (parent index per joint; root = -1).
SMPL24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

JOINT24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]

HAND_INDICES = (22, 23)
KEEP22 = [i for i in range(24) if i not in HAND_INDICES]        # SMPL chain minus hands
JOINT22_NAMES = [JOINT24_NAMES[i] for i in KEEP22]

# Joints Anthro freezes as subject constants (spine1, spine2, L/R collar), as smpl18 defines them.
FIXED_JOINTS = smpl18_skeleton.FROZEN_JOINTS
FIXED_JOINT_NAMES = smpl18_skeleton.FROZEN_JOINT_NAMES

# Major segments (name, parent_joint, child_joint) — hands excluded (spec §origin·remark).
SEGMENTS = [
    ("thigh_l", 1, 4), ("thigh_r", 2, 5),
    ("shank_l", 4, 7), ("shank_r", 5, 8),
    ("foot_l", 7, 10), ("foot_r", 8, 11),
    ("trunk", 0, 9), ("neck", 9, 12), ("head", 12, 15),
    ("upperarm_l", 16, 18), ("upperarm_r", 17, 19),
    ("forearm_l", 18, 20), ("forearm_r", 19, 21),
]
SEGMENT_NAMES = [s[0] for s in SEGMENTS]

def smpl_model_path_for_gender(gender: str) -> str:
    """Map a subject gender ('M'/'F'/'male'/'female') to its clean-npz model path.

    The file name comes from ``smpl18.model.select`` and the directory from
    ``soma_synth.pipeline.paths.body_model_dir()`` (``SOMA_BODY_MODEL_DIR``, else
    ``SOMA_DATA_ROOT/body_models/smpl``; the pipeline runner hashes the same folder), resolved on
    each call. The per-file environment overrides stay here because they belong to these generators,
    not to the convention.
    """
    g = str(gender).strip().lower()[:1]
    overrides = {"m": "SOMA_SMPL_MODEL_MALE", "f": "SOMA_SMPL_MODEL_FEMALE"}
    genders = {"m": "male", "f": "female"}
    if g not in genders:
        raise ValueError("unknown gender for model selection: %r" % (gender,))
    override = os.environ.get(overrides[g])
    if override:
        return override
    return str(smpl18_select.model_path_for_gender(genders[g], data_paths.body_model_dir()))


def load_smpl_model_for_gender(gender: str) -> dict[str, Any]:
    """Select the gendered clean-npz path and load it."""
    return load_smpl_model(smpl_model_path_for_gender(gender))

# Upright-neutral (I-pose) approximation: adduct the shoulders ~90 deg so the arms hang
# down (+X arm -> -Y) from the T-pose. Documented approximation for the biomech reference
# pose; the T-pose (index 0) is the exact SMPL rest skeleton and the one reconstruction uses.
_NEUTRAL_SHOULDER_L = np.array([0.0, 0.0, -np.pi / 2])          # left_shoulder (16)
_NEUTRAL_SHOULDER_R = np.array([0.0, 0.0, np.pi / 2])           # right_shoulder (17)


def load_smpl_model(path: str | None = None) -> dict[str, Any]:
    """Load an SMPL model .npz (v_template, shapedirs, J_regressor, kintree_parents).

    Without a path, the male model in ``body_model_dir()`` (the per-file overrides do not apply)."""
    path = path or str(smpl18_select.model_path_for_gender("male", data_paths.body_model_dir()))
    with np.load(path, allow_pickle=False) as data:
        parents = data["kintree_parents"] if "kintree_parents" in data.files else SMPL24_PARENTS
        return {
            "v_template": np.asarray(data["v_template"], np.float64),      # (V,3)
            "shapedirs": np.asarray(data["shapedirs"], np.float64),        # (V,3,B)
            "J_regressor": np.asarray(data["J_regressor"], np.float64),    # (24,V)
            "parents": np.asarray(parents, np.int64),
        }


def rest_joints(model: dict[str, Any], betas: np.ndarray) -> np.ndarray:
    """Rest-pose (T-pose) joint centres J = J_regressor @ (v_template + shapedirs·betas)."""
    betas = np.asarray(betas, np.float64)
    n = model["shapedirs"].shape[2]
    b = np.zeros(n, np.float64)
    b[: min(n, betas.shape[0])] = betas[: min(n, betas.shape[0])]
    v_shaped = model["v_template"] + np.einsum("vij,j->vi", model["shapedirs"], b)
    return model["J_regressor"] @ v_shaped                                 # (24,3)


def segment_lengths(j_rest: np.ndarray) -> np.ndarray:
    """Bone lengths (m) for the major segments; pose-invariant."""
    return np.array(
        [np.linalg.norm(j_rest[c] - j_rest[p]) for _name, p, c in SEGMENTS], dtype=np.float64
    )


def smpl_fk_positions(j_rest: np.ndarray, local_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics for joint positions and global rotations.

    ``J_posed[j] = J_posed[parent] + G[parent] @ (J_rest[j] - J_rest[parent])``,
    ``G[j] = G[parent] @ local_R[j]``. Returns (positions (24,3), global_R (24,3,3)).
    """
    parents = SMPL24_PARENTS
    G = np.empty((24, 3, 3), np.float64)
    Jp = np.empty((24, 3), np.float64)
    G[0] = local_R[0]
    Jp[0] = j_rest[0]
    for j in range(1, 24):
        p = int(parents[j])
        G[j] = G[p] @ local_R[j]
        Jp[j] = Jp[p] + G[p] @ (j_rest[j] - j_rest[p])
    return Jp, G


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """[...,3,3] -> [...,4] quaternion (w,x,y,z)."""
    q_xyzw = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(R.shape[:-2] + (4,))
    return np.concatenate((q_xyzw[..., 3:4], q_xyzw[..., :3]), axis=-1)


def mean_rotation(rotmats: np.ndarray) -> np.ndarray:
    """Chordal-L2 mean of a stack of rotation matrices [N,3,3] -> [3,3]."""
    return Rotation.from_matrix(np.asarray(rotmats, np.float64)).mean().as_matrix()


def upright_neutral_local_R() -> np.ndarray:
    """Local rotations for the documented upright-neutral (I-pose) reference."""
    local = np.repeat(np.eye(3)[None], 24, axis=0).astype(np.float64)
    local[16] = Rotation.from_rotvec(_NEUTRAL_SHOULDER_L).as_matrix()
    local[17] = Rotation.from_rotvec(_NEUTRAL_SHOULDER_R).as_matrix()
    return local


def reconstruct_positions_frozen_fixed(
    j_rest: np.ndarray,
    local_R_full: np.ndarray,
    fixed_local_R: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct joint positions the SOMA way: freeze the 4 fixed joints to a constant,
    keep every other joint's source rotation. Models what Large(18)+Anthro(4 frozen) yields.

    ``fixed_local_R`` is (4,3,3) in FIXED_JOINTS order; defaults to identity.
    """
    local = local_R_full.copy()
    for i, j in enumerate(FIXED_JOINTS):
        local[j] = np.eye(3) if fixed_local_R is None else fixed_local_R[i]
    Jp, _ = smpl_fk_positions(j_rest, local)
    return Jp


def fk_positions_batch(j_rest: np.ndarray, local_R: np.ndarray) -> np.ndarray:
    """[T,24,3,3] local rotation matrices -> [T,24,3] joint-center positions (pelvis at origin)."""
    T = local_R.shape[0]
    G = np.empty((T, 24, 3, 3), np.float64)
    Jp = np.zeros((T, 24, 3), np.float64)
    G[:, 0] = local_R[:, 0]
    for j in range(1, 24):
        p = int(SMPL24_PARENTS[j])
        G[:, j] = G[:, p] @ local_R[:, j]
        Jp[:, j] = Jp[:, p] + np.einsum("tij,j->ti", G[:, p], j_rest[j] - j_rest[p])
    return Jp


def _height_to_m(height: float) -> float:
    h = float(height)
    return h / 100.0 if h > 3.0 else h                          # accept cm or m


def compute_anthro(
    model: dict[str, Any],
    betas: np.ndarray,
    subj_info: dict[str, Any],
    pair_id: str,
    fixed_joint_local_R: np.ndarray | None = None,
    j_rest: np.ndarray | None = None,
    fixed_joint_provenance: str | None = None,
) -> dict[str, Any]:
    """Assemble the anthro_reference payload from an SMPL model + subject metadata.

    ``fixed_joint_local_R`` (4,3,3, FIXED_JOINTS order) holds the fitted constants of
    spine1/spine2/left_collar/right_collar, and ``fixed_joint_provenance`` says how they were
    fitted (the caller knows whether it pooled a subject or had one take). Without constants the
    fixed joints default to the SMPL-rest identity (higher reconstruction error -- see module
    docstring).
    """
    betas = np.asarray(betas, np.float64)
    # The skeleton to publish. A generator may have corrected the betas-derived one -- HKNU scales it
    # onto the subject's measured bone lengths, since ten betas cannot reach that cohort's hip
    # and shoulder separations -- and the artifact must carry the skeleton the pose was actually
    # fitted against, not one recomputed here and quietly disagreeing with it.
    scaled = j_rest is not None
    j_rest = rest_joints(model, betas) if j_rest is None else np.asarray(j_rest, np.float64)
    j_root = j_rest - j_rest[0]

    jp_tpose = j_root[KEEP22]
    Jp_neutral, _ = smpl_fk_positions(j_rest, upright_neutral_local_R())
    jp_neutral = (Jp_neutral - Jp_neutral[0])[KEEP22]
    joint_position = np.stack([jp_tpose, jp_neutral]).astype(np.float32)   # (2,22,3)

    if fixed_joint_local_R is None:
        fixed_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(FIXED_JOINTS), 1))
        fixed_provenance = (
            "smpl_rest_identity (fixed joints identity at SMPL rest; higher reconstruction error)"
        )
    else:
        fixed_quat = _rotmat_to_wxyz(np.asarray(fixed_joint_local_R, np.float64))
        fixed_provenance = fixed_joint_provenance or (
            "reduced_model_fit: constants fitted by smpl18.reduce.fit_constants to minimise the "
            "joint-centre displacement the freeze causes; the absorbing joints (spine3, shoulders) "
            "in Large take over the removed rotation so distal global orientation is preserved "
            "exactly."
        )
    # One constant per joint over the whole motion, not a per-pose snapshot.
    fixed_joint_rotation = fixed_quat.astype(np.float32)  # (4,4)

    sex = str(subj_info.get("gender", "")).strip().upper()[:1] or "U"
    gender = {"M": "male", "F": "female"}.get(sex, "neutral")

    return {
        "namespace": "anthro_reference/prism_smpl_v1",
        "subject_scope": "subject_constant_trial_frame_invariant",
        # Name arrays are numpy 'U', never object: an object array forces allow_pickle on every
        # reader (ADR-0040 D2), and the 57,827 AMASS warnings of 2026-09-07 were exactly these
        # four arrays written as object by this function and re-typed afterwards.
        "reference_poses": np.array(["t_pose", "upright_neutral"], dtype="U"),
        "joint_names": np.array(JOINT22_NAMES, dtype="U"),                   # (22,)
        "joint_position": joint_position,                                    # (2,22,3) m, root-rel
        "joint_position_frame": "smpl_canonical_rest_yup",
        "segment_names": np.array(SEGMENT_NAMES, dtype="U"),                 # (13,)
        "segment_length": segment_lengths(j_rest).astype(np.float32),        # (13,) m
        "fixed_joint_names": np.array(FIXED_JOINT_NAMES, dtype="U"),         # (4,)
        "fixed_joint_rotation": fixed_joint_rotation,                        # (4,4) wxyz
        "sex": sex,                                                          # 'M'/'F'
        "height": np.float32(_height_to_m(subj_info.get("height", 0.0))),   # m
        "body_mass": np.float32(subj_info.get("weight", 0.0)),              # kg
        "betas": betas.astype(np.float32),                                  # (B,) SMPL shape
        "betas_num": np.int64(betas.shape[0]),
        "gender": gender,
        "height_provenance": "measured:prism_subj_info.height (source unit undeclared upstream; "
                             "interpreted as cm by value range and emitted in SOMA spec unit m)",
        "body_mass_provenance": "measured:prism_subj_info.weight (source unit undeclared upstream; "
                                "interpreted as kg and emitted in SOMA spec unit kg)",
        "sex_provenance": "measured:prism_subj_info.gender",
        "betas_provenance": "source_derived:prism_pkl smpl_params.betas (SMPL 10-dim, constant across take)",
        "skeleton_provenance": (
            "derived:SMPL rest-pose J_regressor@(v_template+shapedirs·betas), then scaled onto the "
            "source's measured bone lengths, directions unchanged"
            if scaled else
            "derived:SMPL rest-pose J_regressor@(v_template+shapedirs·betas)"),
        "fixed_joint_rotation_provenance": fixed_provenance,
        "reconstruction_note": (
            "Large motion joints (18, reduced-model: spine3/shoulders re-fitted to absorb the weld) "
            "+ these fixed_joint_rotation (4, weld constants) = SMPL 22-chain. Distal global "
            "orientations match raw SMPL exactly; only a small trunk position offset remains. "
            "joint_position (T-pose) supplies rest offsets; map to PRISM world via Large's pelvis "
            "global orientation (pelvis_position_world_aux as the world anchor)."
        ),
        "pair_id": pair_id,
    }
