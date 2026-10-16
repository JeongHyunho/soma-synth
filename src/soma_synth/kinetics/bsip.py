"""Body-segment inertial parameters from the shaped SMPL mesh.

The rest-pose mesh is filled with a regular grid of voxels (inside = odd number of mesh crossings
along +z), every voxel goes to the SMPL part that dominates the skinning weights of its nearest
vertex, and each part gets the mass, centre of mass and inertia of its voxels at a uniform density.
Masses are then scaled to a given total, so only the mass *distribution* comes from the mesh.

Everything is expressed in the rest pose, where every SMPL joint rotation is the identity: a
part's rest axes are the world axes, so the posed inertia is ``R I R^T`` with the part's global
rotation ``R``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["SegmentInertia", "segment_inertia_from_mesh", "voxelise"]


@dataclass(frozen=True)
class SegmentInertia:
    """Per-part inertial parameters in the rest pose (world axes)."""

    mass: np.ndarray          # (B,) kg
    com: np.ndarray           # (B,3) m, rest-pose world position
    inertia: np.ndarray       # (B,3,3) kg m^2 about the part's centre of mass, rest axes
    voxel_size: float
    volume: float             # m^3 of the filled mesh

    @property
    def total_mass(self) -> float:
        return float(self.mass.sum())


def voxelise(vertices: np.ndarray, faces: np.ndarray, voxel_size: float) -> np.ndarray:
    """Centres (N,3) of the grid voxels inside a closed triangle mesh.

    Rays are cast along +z through each (x, y) column; a voxel centre is inside when an odd number
    of the column's surface crossings lie above it. The grid is offset by a fraction of a voxel so
    that rays do not pass exactly through vertices or edges.
    """
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    h = float(voxel_size)
    lo = v.min(0) - h
    hi = v.max(0) + h
    jitter = np.array([0.1234, 0.3141, 0.0]) * h
    xs = np.arange(lo[0], hi[0], h) + jitter[0]
    ys = np.arange(lo[1], hi[1], h) + jitter[1]
    zs = np.arange(lo[2], hi[2], h) + 0.5 * h
    ny = len(ys)

    tri = v[f]                                         # (F,3,3)
    cols, zhit = [], []
    for a, b, c in tri:
        tmin = np.minimum(np.minimum(a, b), c)
        tmax = np.maximum(np.maximum(a, b), c)
        i0, i1 = np.searchsorted(xs, [tmin[0], tmax[0]])
        j0, j1 = np.searchsorted(ys, [tmin[1], tmax[1]])
        if i1 <= i0 or j1 <= j0:
            continue
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
        px, py = gx.ravel(), gy.ravel()
        # Barycentric coordinates in the xy projection.
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-18:
            continue
        l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
        l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
        l3 = 1.0 - l1 - l2
        hit = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not hit.any():
            continue
        z = l1[hit] * a[2] + l2[hit] * b[2] + l3[hit] * c[2]
        ii = np.repeat(np.arange(i0, i1), j1 - j0)[hit]
        jj = np.tile(np.arange(j0, j1), i1 - i0)[hit]
        cols.append(ii * ny + jj)
        zhit.append(z)
    cols = np.concatenate(cols)
    zhit = np.concatenate(zhit)

    order = np.lexsort((zhit, cols))
    cols, zhit = cols[order], zhit[order]
    starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
    ends = np.r_[starts[1:], len(cols)]
    out = []
    for s, e in zip(starts, ends):
        if (e - s) % 2:
            continue                                   # open column: skip rather than guess
        zc = zhit[s:e]
        col = cols[s]
        # Inside between crossings 0-1, 2-3, ...
        for k in range(0, e - s, 2):
            k0, k1 = np.searchsorted(zs, [zc[k], zc[k + 1]])
            if k1 > k0:
                n = k1 - k0
                out.append(np.c_[np.full(n, xs[col // ny]), np.full(n, ys[col % ny]), zs[k0:k1]])
    return np.concatenate(out) if out else np.empty((0, 3))


def segment_inertia_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    skin_weights: np.ndarray,
    total_mass: float,
    voxel_size: float,
    part_of_joint: np.ndarray | None = None,
) -> SegmentInertia:
    """Mass, COM and inertia per SMPL part from the rest-pose mesh, scaled to ``total_mass``.

    ``part_of_joint`` maps each skinning joint to the part it is counted in (identity by default),
    so rigid groups can be formed before the inertia is computed.
    """
    v = np.asarray(vertices, np.float64)
    W = np.asarray(skin_weights, np.float64)
    nj = W.shape[1]
    part = np.arange(nj) if part_of_joint is None else np.asarray(part_of_joint, np.int64)
    nb = int(part.max()) + 1

    cells = voxelise(v, faces, voxel_size)
    if len(cells) == 0:
        raise ValueError("the mesh produced no inside voxels; is it closed?")
    _, nearest = cKDTree(v).query(cells)
    owner = part[np.argmax(W[nearest], axis=1)]

    dv = voxel_size ** 3
    volume = len(cells) * dv
    rho = float(total_mass) / volume
    m_vox = rho * dv
    mass = np.zeros(nb)
    com = np.zeros((nb, 3))
    inertia = np.zeros((nb, 3, 3))
    cube = m_vox * voxel_size ** 2 / 6.0               # a uniform cube about its own centre
    for b in range(nb):
        p = cells[owner == b]
        if len(p) == 0:
            continue
        mass[b] = m_vox * len(p)
        com[b] = p.mean(0)
        r = p - com[b]
        inertia[b] = m_vox * (np.eye(3) * (r ** 2).sum() - r.T @ r) + np.eye(3) * cube * len(p)
    return SegmentInertia(mass=mass, com=com, inertia=inertia, voxel_size=float(voxel_size),
                          volume=float(volume))
