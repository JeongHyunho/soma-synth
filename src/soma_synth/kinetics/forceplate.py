"""Force-plate wrenches as loads on the body: sign, reference point, CoP, free moment, foot body.

A plate reports the force and moment the body exerts on it, the moment about the plate origin.
The load on the body is the opposite wrench. Moving it to the world origin, ``N = M + O x F``,
lets the inverse dynamics use it without a CoP, so unloaded frames need no special case there.
The CoP and the vertical free moment are derived from the same wrench on a horizontal plate
surface at height zero, and only where the vertical load exceeds a threshold: below it the CoP
is undefined and is reported as NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PlateLoad", "assign_foot_body", "plate_load"]


@dataclass(frozen=True)
class PlateLoad:
    force: np.ndarray             # (T,3) N on the body
    moment_at_origin: np.ndarray  # (T,3) N m on the body, about the world origin
    contact: np.ndarray           # (T,) bool, vertical load above the threshold
    cop: np.ndarray               # (T,3) m on the plate surface (z = 0), NaN without contact
    free_moment: np.ndarray       # (T,) N m about the vertical through the CoP, NaN without contact


def plate_load(force_on_plate, moment_on_plate, plate_origin, contact_threshold_n) -> PlateLoad:
    """Body-side wrench and CoP from plate force (N) and moment about ``plate_origin`` (N m)."""
    F = -np.asarray(force_on_plate, np.float64)
    O = np.asarray(plate_origin, np.float64)
    N = -(np.asarray(moment_on_plate, np.float64) + np.cross(O, -F))
    contact = F[:, 2] > float(contact_threshold_n)
    cop = np.full(F.shape, np.nan)
    free = np.full(len(F), np.nan)
    c = contact
    cop[c, 2] = 0.0
    cop[c, 0] = -N[c, 1] / F[c, 2]
    cop[c, 1] = N[c, 0] / F[c, 2]
    free[c] = N[c, 2] - (cop[c, 0] * F[c, 1] - cop[c, 1] * F[c, 0])
    return PlateLoad(force=F, moment_at_origin=N, contact=contact, cop=cop, free_moment=free)


def assign_foot_body(load: PlateLoad, ankle_pos, toe_pos, ankle_body: int, toe_body: int) -> np.ndarray:
    """Body receiving the load per frame: the toes when the CoP is ahead of the toe joint along the
    foot's horizontal heading, otherwise the foot; -1 without contact."""
    ankle_pos = np.asarray(ankle_pos, np.float64)
    toe_pos = np.asarray(toe_pos, np.float64)
    heading = toe_pos - ankle_pos
    heading[:, 2] = 0.0
    heading /= np.linalg.norm(heading, axis=1, keepdims=True)
    ahead = np.nansum((load.cop - toe_pos) * heading, axis=1) > 0
    return np.where(load.contact, np.where(ahead, toe_body, ankle_body), -1)
