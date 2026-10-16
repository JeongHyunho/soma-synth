"""SMPL shape and forward kinematics on a clean (chumpy-free) model npz, any number of betas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["SmplModel", "fk", "load_model", "shaped"]


@dataclass(frozen=True)
class SmplModel:
    v_template: np.ndarray       # (V,3)
    shapedirs: np.ndarray        # (V,3,K)
    J_regressor: np.ndarray      # (24,V)
    weights: np.ndarray          # (V,24)
    faces: np.ndarray            # (F,3)
    parents: np.ndarray          # (24,) with -1 at the root

    @property
    def num_betas(self) -> int:
        return int(self.shapedirs.shape[2])


def load_model(path: str | Path) -> SmplModel:
    z = np.load(path)
    parents = np.asarray(z["kintree_parents"], np.int64).copy()
    parents[0] = -1
    return SmplModel(
        v_template=np.asarray(z["v_template"], np.float64),
        shapedirs=np.asarray(z["shapedirs"], np.float64),
        J_regressor=np.asarray(z["J_regressor"], np.float64),
        weights=np.asarray(z["weights"], np.float64),
        faces=np.asarray(z["f"], np.int64),
        parents=parents,
    )


def shaped(model: SmplModel, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest-pose vertices and joints for ``betas``. All betas must fit the model's shape space."""
    betas = np.asarray(betas, np.float64)
    if len(betas) > model.num_betas:
        raise ValueError(f"{len(betas)} betas but the model has {model.num_betas} shape components")
    v = model.v_template + model.shapedirs[:, :, : len(betas)] @ betas
    return v, model.J_regressor @ v


def fk(model: SmplModel, rest_joints: np.ndarray, global_orient, body_pose, trans):
    """Global joint rotations (T,24,3,3) and joint positions (T,24,3) from axis-angle pose."""
    trans = np.asarray(trans, np.float64)
    T = len(trans)
    aa = np.concatenate([np.asarray(global_orient, np.float64), np.asarray(body_pose, np.float64)], 1)
    local = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)
    R = np.empty_like(local)
    P = np.empty((T, 24, 3))
    J = np.asarray(rest_joints, np.float64)
    for j in range(24):
        p = model.parents[j]
        if p < 0:
            R[:, j] = local[:, j]
            P[:, j] = J[j] + trans
        else:
            R[:, j] = R[:, p] @ local[:, j]
            P[:, j] = P[:, p] + np.einsum("tij,j->ti", R[:, p], J[j] - J[p])
    return R, P
