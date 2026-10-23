"""Joint-coordinate-system projection of SMPL joint moments and angular velocities.

Anatomical segment frames follow the generators' construction (``anatomical_frame`` in the
AMASS/PRISM generators): limb segments take Y = proximal long axis and X = body-forward made
orthogonal to it; foot segments take X = forward and Y = up; trunk, neck and head take
X = forward, Y = up. Z = X x Y throughout, which is subject-right for the legs and trunk.
Forearm and hand differ: the SMPL rest palm faces down, so their anterior (volar) X is down;
with body-forward the wrist's flexion axis would be the deviation axis.

Axes (ISB joint coordinate systems, Wu et al. 2002/2005): e1 is fixed in the parent frame, e3 in the child frame and
e2 = normalise(e3 x e1) floats. Sagittal joints use e1 = parent Z, e3 = child Y; the shoulder
uses the Y-X-Y convention, e1 = thorax Y, e3 = humerus Y, so component 1 is elevation.

Signs: the primary component's sign is set from the rest geometry so that the anatomically
positive motion of each joint (``POSITIVE_MOTION``) gives a positive value on both sides.
The other two components follow the right side's axes and are negated on the left: the
sagittal-conjugation canonicalisation (diag(1, 1, -1)) that reports both sides with right-side signs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "JOINTS",
    "JcsCalibration",
    "calibrate",
    "canonical_axes",
    "project",
]

# The 23 moving joints of the Large kinematics, SMPL index order (large/02_kinematics.qmd).
# name, SMPL joint (= child body), JCS parent body, kind, side, primary component, distal point.
# The shoulders take the thorax (spine3) as their JCS parent, as ISB does, rather than the collar.
# The collars' primary component is the floating axis e2, the girdle's elevation. The hands have no
# joint beyond them, so their distal point continues the wrist-hand line ("beyond:<joint>").
JOINTS: tuple[tuple[str, int, int, str, str, int, str], ...] = (
    ("left_hip", 1, 0, "sagittal", "L", 0, "joint:4"),
    ("right_hip", 2, 0, "sagittal", "R", 0, "joint:5"),
    ("spine1", 3, 0, "sagittal", "M", 0, "joint:6"),
    ("left_knee", 4, 1, "sagittal", "L", 0, "joint:7"),
    ("right_knee", 5, 2, "sagittal", "R", 0, "joint:8"),
    ("spine2", 6, 3, "sagittal", "M", 0, "joint:9"),
    ("left_ankle", 7, 4, "sagittal", "L", 0, "joint:10"),
    ("right_ankle", 8, 5, "sagittal", "R", 0, "joint:11"),
    ("spine3", 9, 6, "sagittal", "M", 0, "joint:12"),
    ("left_foot", 10, 7, "sagittal", "L", 0, "forward"),
    ("right_foot", 11, 8, "sagittal", "R", 0, "forward"),
    ("neck", 12, 9, "sagittal", "M", 0, "joint:15"),
    ("left_collar", 13, 9, "sagittal", "L", 1, "joint:16"),
    ("right_collar", 14, 9, "sagittal", "R", 1, "joint:17"),
    ("head", 15, 12, "sagittal", "M", 0, "up"),
    ("left_shoulder", 16, 9, "shoulder", "L", 1, "joint:18"),
    ("right_shoulder", 17, 9, "shoulder", "R", 1, "joint:19"),
    ("left_elbow", 18, 16, "sagittal", "L", 0, "joint:20"),
    ("right_elbow", 19, 17, "sagittal", "R", 0, "joint:21"),
    ("left_wrist", 20, 18, "sagittal", "L", 0, "joint:22"),
    ("right_wrist", 21, 19, "sagittal", "R", 0, "joint:23"),
    ("left_hand", 22, 20, "sagittal", "L", 0, "beyond:20"),
    ("right_hand", 23, 21, "sagittal", "R", 0, "beyond:21"),
)

# Direction the distal point moves in the positive primary motion, from the rest (T) pose.
# hip/elbow/trunk/neck/head flexion: forward; knee flexion: backward; ankle/MTP dorsiflexion: up;
# shoulder and collar elevation: up (the arm is already abducted 90 degrees at rest); wrist and hand
# flexion: down (the SMPL rest palm faces down, so flexion moves the hand toward the floor).
POSITIVE_MOTION = {
    "hip": "forward", "knee": "backward", "ankle": "up", "foot": "up", "spine1": "forward",
    "spine2": "forward", "spine3": "forward", "neck": "forward", "head": "forward",
    "collar": "up", "shoulder": "up", "elbow": "forward", "wrist": "down", "hand": "down",
}

# body -> (frame kind, proximal joint, distal joint) for the anatomical frame
_SEGMENT = {
    0: ("trunk", 0, 0), 3: ("trunk", 0, 0), 6: ("trunk", 0, 0), 9: ("trunk", 0, 0),
    12: ("trunk", 0, 0), 15: ("trunk", 0, 0), 13: ("trunk", 0, 0), 14: ("trunk", 0, 0),
    1: ("limb", 1, 4), 2: ("limb", 2, 5), 4: ("limb", 4, 7), 5: ("limb", 5, 8),
    7: ("foot", 0, 0), 8: ("foot", 0, 0), 10: ("foot", 0, 0), 11: ("foot", 0, 0),
    16: ("limb", 16, 18), 17: ("limb", 17, 19),
    18: ("volar", 18, 20), 19: ("volar", 19, 21), 20: ("volar", 20, 22), 21: ("volar", 21, 23),
    22: ("volar", 20, 22), 23: ("volar", 21, 23),
}


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def canonical_axes(j_rest: np.ndarray):
    """Subject up / forward / right unit vectors from the rest joints (same as the generators)."""
    j = np.asarray(j_rest, np.float64)
    up = _unit(j[15] - j[0])
    right = j[2] - j[1]
    right = _unit(right - right.dot(up) * up)
    fwd = np.cross(up, right)
    if fwd.dot(j[10] - j[7]) < 0:
        fwd = -fwd
    return up, _unit(fwd), right


def _anatomical(body, j_rest, canon):
    up, fwd, _ = canon
    kind, pj, dj = _SEGMENT[body]
    if kind in ("limb", "volar"):
        y = _unit(j_rest[pj] - j_rest[dj])
        a = fwd if kind == "limb" else -up  # the rest palm faces down: volar is down
        x = _unit(a - a.dot(y) * y)
    else:                                 # trunk and foot: X forward, Y up
        x = fwd
        y = _unit(up - up.dot(x) * x)
    return np.stack([x, y, np.cross(x, y)], axis=1)


@dataclass(frozen=True)
class JcsCalibration:
    names: tuple[str, ...]
    child: np.ndarray            # (J,) SMPL body of the child segment
    parent: np.ndarray           # (J,) body whose frame carries e1
    kind: tuple[str, ...]
    primary: np.ndarray          # (J,) component index reported in joint_torque
    sign: np.ndarray             # (J,3) per-component sign applied after the projection
    anatomical: np.ndarray       # (24,3,3) rest anatomical frame per body (columns X,Y,Z)


def _axes_rest(kind, A_parent, A_child):
    if kind == "sagittal":
        e1, e3 = A_parent[:, 2], A_child[:, 1]
    else:
        e1, e3 = A_parent[:, 1], A_child[:, 1]
    e2 = np.cross(e3, e1)
    n = np.linalg.norm(e2)
    if n < 1e-6:                            # aligned at rest (a hanging arm): any perpendicular
        e2 = np.cross(e3, A_parent[:, 0])
        n = np.linalg.norm(e2)
    return e1, e2 / n, e3


def calibrate(j_rest: np.ndarray) -> JcsCalibration:
    """Anatomical frames and per-joint signs from the subject's rest joints."""
    j_rest = np.asarray(j_rest, np.float64)
    canon = canonical_axes(j_rest)
    up, fwd, _ = canon
    direction = {"forward": fwd, "backward": -fwd, "up": up, "down": -up}
    A = np.stack([_anatomical(b, j_rest, canon) for b in range(24)])
    sign = np.ones((len(JOINTS), 3))
    for i, (name, child, parent, kind, side, prim, distal) in enumerate(JOINTS):
        e = _axes_rest(kind, A[parent], A[child])
        p = j_rest[child]
        if distal.startswith("joint:"):
            d = j_rest[int(distal.split(":")[1])]
        elif distal.startswith("beyond:"):
            d = 2.0 * p - j_rest[int(distal.split(":")[1])]
        else:
            d = p + 0.1 * direction[distal]
        group = name.split("_")[-1] if "_" in name else name
        motion = np.cross(e[prim], d - p)   # distal velocity for a unit rotation about the axis
        s = np.sign(motion.dot(direction[POSITIVE_MOTION[group]]))
        if s == 0:
            raise ValueError(f"{name}: the primary axis does not move the distal point")
        sign[i, prim] = s
        if side == "L":
            for k in range(3):
                if k != prim:
                    sign[i, k] = -1.0
    return JcsCalibration(
        names=tuple(j[0] for j in JOINTS),
        child=np.array([j[1] for j in JOINTS]),
        parent=np.array([j[2] for j in JOINTS]),
        kind=tuple(j[3] for j in JOINTS),
        primary=np.array([j[5] for j in JOINTS]),
        sign=sign,
        anatomical=A,
    )


def project(cal: JcsCalibration, rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Project per-joint world vectors (T,J,3) onto each joint's instantaneous JCS axes.

    ``rotation`` is the bodies' global rotation (T,24,3,3) from rest axes to world.
    """
    R = np.asarray(rotation, np.float64)
    out = np.empty(vectors.shape)
    for i in range(len(cal.names)):
        Ap = R[:, cal.parent[i]] @ cal.anatomical[cal.parent[i]]
        Ac = R[:, cal.child[i]] @ cal.anatomical[cal.child[i]]
        if cal.kind[i] == "sagittal":
            e1, e3 = Ap[:, :, 2], Ac[:, :, 1]
        else:
            e1, e3 = Ap[:, :, 1], Ac[:, :, 1]
        e2 = np.cross(e3, e1)
        n = np.linalg.norm(e2, axis=1, keepdims=True)
        e2 = np.where(n > 1e-6, e2 / np.maximum(n, 1e-12), np.nan)
        v = vectors[:, i]
        out[:, i] = np.stack([(v * e1).sum(1), (v * e2).sum(1), (v * e3).sum(1)], 1) * cal.sign[i]
    return out
