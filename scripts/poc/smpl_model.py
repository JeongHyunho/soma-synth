"""Safe loader + clean-extract for the license-gated MPI SMPL body model (basicmodel_*.pkl).

The MPI SMPL `.pkl` is a protocol-0 pickle mixing numpy arrays, a `scipy.sparse` J_regressor, and
`chumpy.Ch` leaf arrays. We NEVER unpickle it unrestricted: a whitelist Unpickler allows only
numpy / scipy.sparse / copyreg / builtins, and shims every `chumpy.*` global to a passive `_Ch`
holder whose numpy payload we extract afterward (chumpy is unimportable on modern numpy anyway).
No code from the pickle executes. This mirrors the PRISM restricted-unpickler security posture.

    python smpl_model.py --extract <basicmodel_m_*.pkl> <out_clean.npz>

The clean npz holds only what SMPL forward needs: v_template[6890,3], shapedirs[6890,3,10],
posedirs[6890,3,207], J_regressor[24,6890], weights[6890,24], faces[13776,3], kintree_parents[24].
Body payloads stay under the data root (SOMA_BODY_MODEL_DIR, else <SOMA_DATA_ROOT>/body_models/smpl);
the model is INTERNAL-ONLY, non-redistributable.
"""
from __future__ import annotations

import argparse
import builtins
import copyreg
import importlib
import pickle

import numpy as np


class _Ch:
    """Passive stand-in for a chumpy.Ch leaf: capture pickle state, expose its numpy payload."""

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            for part in state:
                if isinstance(part, dict):
                    self.__dict__.update(part)
        elif isinstance(state, dict):
            self.__dict__.update(state)


def _safe_find_class(module: str, name: str):
    if module.startswith("numpy"):
        return getattr(importlib.import_module(module), name)
    if module.startswith("scipy.sparse"):
        return getattr(importlib.import_module(module), name)
    if module in ("copy_reg", "copyreg"):
        return getattr(copyreg, name)
    if module in ("__builtin__", "builtins"):
        return getattr(builtins, name)
    if module.startswith("chumpy"):
        return _Ch                      # shim: no chumpy code runs
    raise pickle.UnpicklingError(f"blocked global {module}.{name}")


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _safe_find_class(module, name)


def _as_array(v):
    """Resolve a loaded value to a plain numpy array (unwrap _Ch / scipy.sparse)."""
    if isinstance(v, _Ch):
        for key in ("x", "r", "_result"):
            a = v.__dict__.get(key)
            if isinstance(a, np.ndarray):
                return np.asarray(a)
        arrs = [a for a in v.__dict__.values() if isinstance(a, np.ndarray)]
        if arrs:
            return np.asarray(max(arrs, key=lambda a: a.size))
        raise ValueError(f"chumpy leaf without ndarray payload; attrs={list(v.__dict__)}")
    if hasattr(v, "toarray"):            # scipy.sparse matrix (J_regressor)
        return np.asarray(v.toarray())
    return np.asarray(v)


def load_model_npz(path: str) -> dict:
    """Load the clean-extracted SMPL model npz as float64 arrays for numerically stable forward."""
    z = np.load(path)
    m = {k: np.asarray(z[k]) for k in z.files}
    if "faces" not in m and "f" in m:
        # extract_clean() renames the SMPL pickle's `f` to `faces`; a model npz taken straight off
        # the pickle keeps `f`. It is the same triangle list either way — both genders share the SMPL
        # topology, and the female file's `f` was checked equal to the male file's `faces` — so the
        # alias is accepted rather than making the caller care which vintage of file it holds.
        m["faces"] = m["f"]
    if "faces" in m:
        m["faces"] = m["faces"].astype(np.int32)
    for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights"):
        m[k] = m[k].astype(np.float64)
    m["kintree_parents"] = m["kintree_parents"].astype(np.int64)
    return m


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[...,4] unit quaternion (w,x,y,z) -> [...,3,3] rotation matrix."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def global_to_local_R(Rg: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Per-joint GLOBAL rotations [24,3,3] -> LOCAL (parent-relative); root local == root global."""
    Rl = np.empty_like(Rg)
    Rl[0] = Rg[0]
    for k in range(1, len(parents)):
        Rl[k] = Rg[parents[k]].T @ Rg[k]
    return Rl


def smpl_shape(model: dict, betas: np.ndarray):
    """betas[10] -> (v_shaped[6890,3], J_rest[24,3]). Pure shape blend, no pose."""
    betas = np.asarray(betas, np.float64).reshape(-1)[:10]
    v_shaped = model["v_template"] + model["shapedirs"] @ betas
    J = model["J_regressor"] @ v_shaped
    return v_shaped, J


def smpl_forward(model: dict, betas: np.ndarray, Rglob: np.ndarray, trans: np.ndarray) -> dict:
    """Full SMPL forward for verification.

    Rglob[T,24,3,3] = per-joint GLOBAL rotations (prism-world), trans[T,3] = SMPL root translation.
    Returns posed world verts[T,6890,3] + joints[T,24,3]. Includes pose blend shapes (posedirs).
    The Blender rig reproduces this via armature LBS (minus posedirs); this stays the numeric oracle.
    """
    parents = model["kintree_parents"]
    W = model["weights"]                                   # [6890,24]
    posedirs = model["posedirs"]                           # [6890,3,207]
    v_shaped, J = smpl_shape(model, betas)
    T, nJ = Rglob.shape[0], len(parents)
    eye = np.eye(3)
    verts = np.empty((T, v_shaped.shape[0], 3))
    joints = np.empty((T, nJ, 3))
    for t in range(T):
        Rg = Rglob[t]
        Rl = global_to_local_R(Rg, parents)
        pf = (Rl[1:] - eye).reshape(-1)                    # [207] pose feature
        vp = v_shaped + posedirs @ pf                      # [6890,3] pose-corrected rest
        Gt = np.empty((nJ, 3))
        Gt[0] = J[0]
        for k in range(1, nJ):
            Gt[k] = Gt[parents[k]] + Rg[parents[k]] @ (J[k] - J[parents[k]])
        acc = np.zeros_like(vp)
        for k in range(nJ):
            acc += W[:, k:k + 1] * ((vp - J[k]) @ Rg[k].T + Gt[k])
        verts[t] = acc + trans[t]
        joints[t] = Gt + trans[t]
    return dict(verts=verts, joints=joints, faces=model["faces"])


def load_smpl_pkl(path: str) -> dict:
    """Restricted-unpickle the SMPL model .pkl into a dict (values still raw _Ch/sparse/ndarray)."""
    with open(path, "rb") as f:
        obj = _SafeUnpickler(f, encoding="latin1").load()
    if not isinstance(obj, dict):
        raise TypeError(f"SMPL pkl top-level is {type(obj)}, expected dict")
    return obj


def extract_clean(pkl_path: str, out_npz: str) -> dict:
    """Load the model and write a compact, chumpy-free npz with exactly the SMPL forward inputs."""
    m = load_smpl_pkl(pkl_path)
    v_template = _as_array(m["v_template"]).astype(np.float32)          # [6890,3]
    shapedirs = _as_array(m["shapedirs"]).astype(np.float32)           # [6890,3,10] (or ...,>10)
    posedirs = _as_array(m["posedirs"]).astype(np.float32)            # [6890,3,207]
    J_regressor = _as_array(m["J_regressor"]).astype(np.float32)      # [24,6890]
    weights = _as_array(m["weights"]).astype(np.float32)              # [6890,24]
    faces = _as_array(m["f"]).astype(np.int64)                        # [13776,3]
    kintree = _as_array(m["kintree_table"]).astype(np.int64)         # [2,24]
    parents = kintree[0].copy()
    parents[0] = -1                                                    # root has no parent

    nV = v_template.shape[0]
    nJ = J_regressor.shape[0]
    assert v_template.shape == (nV, 3), v_template.shape
    assert shapedirs.shape[:2] == (nV, 3), shapedirs.shape
    assert posedirs.shape == (nV, 3, (nJ - 1) * 9), posedirs.shape
    assert J_regressor.shape == (nJ, nV), J_regressor.shape
    assert weights.shape == (nV, nJ), weights.shape
    assert faces.ndim == 2 and faces.shape[1] == 3, faces.shape
    assert parents.shape == (nJ,) and parents[0] == -1, parents

    shapedirs = shapedirs[:, :, :10]                                   # SMPL uses 10 betas
    out = dict(v_template=v_template, shapedirs=shapedirs, posedirs=posedirs,
               J_regressor=J_regressor, weights=weights, faces=faces.astype(np.int32),
               kintree_parents=parents.astype(np.int32))
    np.savez(out_npz, **out)
    return {k: v.shape for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Safe clean-extract of an MPI SMPL model pkl.")
    ap.add_argument("--extract", nargs=2, metavar=("PKL", "OUT_NPZ"), required=True)
    args = ap.parse_args()
    shapes = extract_clean(args.extract[0], args.extract[1])
    print("[smpl_model] wrote", args.extract[1])
    for k, s in shapes.items():
        print(f"    {k:16s} {s}")
