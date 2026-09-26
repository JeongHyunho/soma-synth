"""Regression tests for the native-SMPL forward used by the viewer (exact-shape path).

A tiny synthetic SMPL-shaped model pins the forward math (identity, skinning partition, rigid FK,
global<->local round-trip) without Blender or licence-gated data. An optional block runs
against the real clean SMPL model npz when present in the body-model folder (``SOMA_BODY_MODEL_DIR``,
else ``<SOMA_DATA_ROOT>/body_models/smpl``), checking the shape actually deforms.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

POC = Path(__file__).parents[2] / "scripts" / "poc"
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

import smpl_model as sm   # noqa: E402  (numpy only; no bpy)


def _real_model() -> str:
    """The licensed clean male model in the configured body-model folder, or "" when none is."""
    from soma_synth.pipeline import paths
    try:
        return str(paths.body_model_dir() / "SMPL_MALE_clean.npz")
    except paths.PathConfigError:
        return ""


REAL_MODEL = _real_model()


def _toy_model(nV=8, nJ=3, seed=0):
    """A structurally valid mini SMPL: J_regressor one-hot, weights row-stochastic, chain parents."""
    rng = np.random.default_rng(seed)
    v_template = rng.standard_normal((nV, 3))
    shapedirs = rng.standard_normal((nV, 3, 10)) * 0.05
    posedirs = rng.standard_normal((nV, 3, (nJ - 1) * 9)) * 0.02
    J_regressor = np.zeros((nJ, nV))
    for j in range(nJ):
        J_regressor[j, j] = 1.0                       # joint j sits on vertex j
    w = rng.random((nV, nJ)) + 0.05
    weights = w / w.sum(axis=1, keepdims=True)        # rows sum to 1
    parents = np.array([-1] + list(range(nJ - 1)), dtype=np.int64)   # 0<-1<-2 chain
    return dict(v_template=v_template, shapedirs=shapedirs, posedirs=posedirs,
                J_regressor=J_regressor, weights=weights,
                faces=np.array([[0, 1, 2]], np.int32), kintree_parents=parents)


def _rand_global(T, nJ, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((T, nJ, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return sm.quat_wxyz_to_rotmat(q)


def test_forward_identity_returns_template():
    m = _toy_model()
    Rid = np.broadcast_to(np.eye(3), (1, 3, 3, 3)).copy()
    out = sm.smpl_forward(m, np.zeros(10), Rid, np.zeros((1, 3)))
    assert np.abs(out["verts"][0] - m["v_template"]).max() < 1e-10
    J = m["J_regressor"] @ m["v_template"]
    assert np.abs(out["joints"][0] - J).max() < 1e-10


def test_skinning_weights_partition_of_unity():
    m = _toy_model()
    assert np.abs(m["weights"].sum(axis=1) - 1.0).max() < 1e-12


def test_translation_shifts_rigidly():
    m = _toy_model()
    Rg = _rand_global(1, 3, seed=1)
    t = np.array([[0.3, -1.2, 4.0]])
    a = sm.smpl_forward(m, np.zeros(10), Rg, np.zeros((1, 3)))["verts"][0]
    b = sm.smpl_forward(m, np.zeros(10), Rg, t)["verts"][0]
    assert np.abs((b - a) - t[0]).max() < 1e-9


def test_bone_lengths_are_pose_invariant():
    """FK must not stretch bones: segment lengths are identical across arbitrary poses."""
    m = _toy_model()
    Rg = _rand_global(12, 3, seed=2)
    J = sm.smpl_forward(m, np.zeros(10), Rg, np.zeros((12, 3)))["joints"]
    par = m["kintree_parents"]
    for k in range(1, 3):
        L = np.linalg.norm(J[:, k] - J[:, par[k]], axis=1)
        assert L.std() < 1e-9


def test_global_local_roundtrip():
    m = _toy_model()
    Rg = _rand_global(1, 3, seed=5)[0]
    Rl = sm.global_to_local_R(Rg, m["kintree_parents"])
    G = np.empty_like(Rg)
    G[0] = Rl[0]
    for k in range(1, 3):
        G[k] = G[m["kintree_parents"][k]] @ Rl[k]
    assert np.abs(G - Rg).max() < 1e-12


def test_real_male_model_shape_deforms():
    """When the licensed clean model is present, non-zero betas must change the body stature."""
    if not REAL_MODEL or not os.path.isfile(REAL_MODEL):
        import pytest
        pytest.skip("clean SMPL model npz not present in the body-model folder "
                    "(SOMA_BODY_MODEL_DIR / SOMA_DATA_ROOT unset, or the file is absent)")
    m = sm.load_model_npz(REAL_MODEL)
    v0, _ = sm.smpl_shape(m, np.zeros(10))
    v1, _ = sm.smpl_shape(m, np.array([2.5, 1.0, 1.7, 0, 0, 0, 0, 0, 0, 0]))
    assert 1.3 < np.ptp(v0[:, 1]) < 2.1                      # neutral native stature is human-scale (Y up)
    assert np.abs(v1 - v0).max() > 0.01                      # betas visibly deform the mesh
