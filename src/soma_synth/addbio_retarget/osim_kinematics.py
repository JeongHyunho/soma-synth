"""Forward kinematics of the OpenSim model embedded in a .b3d, without OpenSim.

Only rotations need this. AddBiomechanics already stored the joint centres it computed, so
positions are read rather than recomputed; what is *not* stored is how each bone is turned about
its own axis, and that is precisely what a downstream IMU orientation is a rigid relabel of.

Everything the model needs is in the file: joint frames with their offsets, the axis of each
degree of freedom, and the function relating it to a coordinate -- including the walker knee's
splines, which turn one coordinate into three rotations and two translations. That is the
largest implementation risk, and it is measurable: AddBiomechanics stores
`world_frame_joint_centers`, the same quantity this module computes, written by the software
being reproduced, so the two can be compared joint by joint.

Convention, matching OpenSim's CustomJoint: the transform of the child frame in the parent frame
is the three rotations composed in the order the model lists them, and a translation read in the
parent frame -- not carried around by the rotations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

__all__ = [
    "ForwardResult",
    "KinematicJoint",
    "KinematicModel",
    "OffsetFrame",
    "TransformAxis",
    "UnsupportedJoint",
    "compile_function",
    "evaluate_function",
    "parse_kinematics",
]

_GROUND = "ground"
_METADATA_TAGS = frozenset({"coordinates", "axis"})


class UnsupportedJoint(ValueError):
    """A joint type this module does not know how to move."""


# --------------------------------------------------------------------------- functions


def _floats(node: ET.Element | None) -> np.ndarray:
    if node is None or not (node.text or "").strip():
        return np.zeros(0)
    return np.array([float(v) for v in node.text.split()], dtype=np.float64)


def _zero(value):
    """Nothing, shaped like whatever was asked for."""
    return np.zeros(np.shape(value)) if np.ndim(value) else 0.0


def compile_function(node: ET.Element) -> Callable[[float], float]:
    """Turn one OpenSim function node into a callable, once, at parse time.

    Compiled rather than cached: a cache keyed on the XML element's identity looks like an
    obvious optimisation and is unsound, because CPython recycles `id()` once an element is
    collected, so the next subject's spline silently answers with the previous subject's
    curve. That failure only shows up when several models are read in one process -- exactly
    what a cohort sweep does and a single-subject test does not.

    Every compiled function takes a scalar or a whole trial's worth of values and answers in
    kind. A trial converted frame by frame costs 1.3 ms a frame, which is twenty hours across
    the cohort; the same work batched is the same arithmetic with the loop moved into numpy.
    """
    tag = node.tag
    if tag == "function":
        inner = next(iter(node), None)
        return _zero if inner is None else compile_function(inner)
    if tag == "LinearFunction":
        coefficients = _floats(node.find("coefficients"))
        slope = float(coefficients[0]) if coefficients.size else 1.0
        intercept = float(coefficients[1]) if coefficients.size > 1 else 0.0
        return lambda value: slope * value + intercept
    if tag == "Constant":
        constant = node.find("value")
        amount = float((constant.text or "0").strip()) if constant is not None else 0.0
        return lambda value: (
            np.full(np.shape(value), amount) if np.ndim(value) else amount
        )
    if tag == "MultiplierFunction":
        inner = node.find("function")
        scale_node = node.find("scale")
        scale = float((scale_node.text or "1").strip()) if scale_node is not None else 1.0
        wrapped = compile_function(inner) if inner is not None else _zero
        return lambda value: scale * wrapped(value)
    if tag in ("SimmSpline", "NaturalCubicSpline"):
        x = _floats(node.find("x"))
        y = _floats(node.find("y"))
        if x.size < 2:
            raise ValueError("a spline needs at least two knots")
        spline = CubicSpline(x, y, bc_type="natural")
        first, last = float(x[0]), float(x[-1])
        low, high = float(y[0]), float(y[-1])
        slope_low, slope_high = float(spline(first, 1)), float(spline(last, 1))

        def evaluate(value):
            # Beyond the knots a cubic runs away. OpenSim clamps its coordinates, so straight
            # continuation is both closer to intent and safer than an unbounded curve.
            if not np.ndim(value):
                if value < first:
                    return low + slope_low * (value - first)
                if value > last:
                    return high + slope_high * (value - last)
                return float(spline(value))
            values = np.asarray(value, dtype=np.float64)
            out = np.asarray(spline(np.clip(values, first, last)), dtype=np.float64)
            below = values < first
            above = values > last
            if below.any():
                out = np.where(below, low + slope_low * (values - first), out)
            if above.any():
                out = np.where(above, high + slope_high * (values - last), out)
            return out

        return evaluate
    raise UnsupportedJoint(f"unknown function type {tag!r}")


def evaluate_function(node: ET.Element, value: float) -> float:
    """Evaluate one OpenSim function node at a coordinate value."""
    return compile_function(node)(value)


def _axis_function(axis: ET.Element) -> ET.Element | None:
    for child in axis:
        if child.tag in _METADATA_TAGS:
            continue
        return child
    return None


# --------------------------------------------------------------------------- structure


@dataclass(frozen=True)
class OffsetFrame:
    name: str
    body: str
    translation: np.ndarray
    orientation: np.ndarray

    def rotation(self) -> np.ndarray:
        return Rotation.from_euler("XYZ", self.orientation).as_matrix()


@dataclass(frozen=True)
class TransformAxis:
    name: str
    axis: np.ndarray
    coordinate: str | None
    function: Callable[[float], float] | None

    def value(self, coordinates: Mapping[str, float]) -> float:
        if self.coordinate is None or self.function is None:
            return 0.0
        return self.function(float(coordinates.get(self.coordinate, 0.0)))


@dataclass(frozen=True)
class KinematicJoint:
    name: str
    kind: str
    parent_frame: OffsetFrame
    child_frame: OffsetFrame
    coordinates: tuple[str, ...]
    axes: tuple[TransformAxis, ...]

    def local_transform(self, q: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Child frame expressed in the parent frame."""
        if self.kind == "CustomJoint":
            rotation = np.eye(3)
            translation = np.zeros(3)
            for axis in self.axes:
                magnitude = axis.value(q)
                if axis.name.startswith("rotation"):
                    rotation = rotation @ Rotation.from_rotvec(
                        axis.axis * magnitude
                    ).as_matrix()
                else:
                    translation = translation + axis.axis * magnitude
            return rotation, translation
        if self.kind == "PinJoint":
            angle = float(q.get(self.coordinates[0], 0.0)) if self.coordinates else 0.0
            return Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix(), np.zeros(3)
        if self.kind == "UniversalJoint":
            first = float(q.get(self.coordinates[0], 0.0)) if self.coordinates else 0.0
            second = float(q.get(self.coordinates[1], 0.0)) if len(self.coordinates) > 1 else 0.0
            rotation = (
                Rotation.from_rotvec([first, 0.0, 0.0]).as_matrix()
                @ Rotation.from_rotvec([0.0, second, 0.0]).as_matrix()
            )
            return rotation, np.zeros(3)
        if self.kind == "WeldJoint":
            return np.eye(3), np.zeros(3)
        raise UnsupportedJoint(f"joint type {self.kind!r} is not implemented")

    def local_transform_batch(
        self, q: Mapping[str, np.ndarray], frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The same transform for a whole trial: (frames, 3, 3) and (frames, 3).

        Identical arithmetic to `local_transform`, with the loop in numpy. A test holds the two
        to the same answer, because once the conversion runs on this path every number in the
        corpus depends on them not drifting apart.
        """
        def column(name: str) -> np.ndarray:
            value = q.get(name)
            if value is None:
                return np.zeros(frames)
            return np.broadcast_to(np.asarray(value, dtype=np.float64), (frames,))

        if self.kind == "CustomJoint":
            rotation = np.tile(np.eye(3), (frames, 1, 1))
            translation = np.zeros((frames, 3))
            for axis in self.axes:
                if axis.coordinate is None or axis.function is None:
                    continue
                magnitude = np.asarray(
                    axis.function(column(axis.coordinate)), dtype=np.float64
                )
                magnitude = np.broadcast_to(magnitude, (frames,))
                if axis.name.startswith("rotation"):
                    turn = Rotation.from_rotvec(
                        axis.axis[None, :] * magnitude[:, None]
                    ).as_matrix()
                    rotation = rotation @ turn
                else:
                    translation = translation + axis.axis[None, :] * magnitude[:, None]
            return rotation, translation
        if self.kind == "PinJoint":
            angle = column(self.coordinates[0]) if self.coordinates else np.zeros(frames)
            vectors = np.zeros((frames, 3))
            vectors[:, 2] = angle
            return Rotation.from_rotvec(vectors).as_matrix(), np.zeros((frames, 3))
        if self.kind == "UniversalJoint":
            first = column(self.coordinates[0]) if self.coordinates else np.zeros(frames)
            second = (
                column(self.coordinates[1]) if len(self.coordinates) > 1 else np.zeros(frames)
            )
            about_x = np.zeros((frames, 3))
            about_x[:, 0] = first
            about_y = np.zeros((frames, 3))
            about_y[:, 1] = second
            rotation = (
                Rotation.from_rotvec(about_x).as_matrix()
                @ Rotation.from_rotvec(about_y).as_matrix()
            )
            return rotation, np.zeros((frames, 3))
        if self.kind == "WeldJoint":
            return np.tile(np.eye(3), (frames, 1, 1)), np.zeros((frames, 3))
        raise UnsupportedJoint(f"joint type {self.kind!r} is not implemented")


@dataclass(frozen=True)
class ForwardResult:
    joint_centres: dict[str, np.ndarray]
    body_rotation: dict[str, np.ndarray]
    body_origin: dict[str, np.ndarray]


@dataclass(frozen=True)
class KinematicModel:
    joints: tuple[KinematicJoint, ...]

    def forward(self, coordinates: Mapping[str, float]) -> ForwardResult:
        """Place every body, given coordinate values. Missing coordinates are zero."""
        rotation = {_GROUND: np.eye(3)}
        origin = {_GROUND: np.zeros(3)}
        centres: dict[str, np.ndarray] = {}

        for joint in _in_tree_order(self.joints):
            parent_body = joint.parent_frame.body
            if parent_body not in rotation:
                raise UnsupportedJoint(
                    f"joint {joint.name!r} hangs off {parent_body!r}, which nothing places"
                )
            parent_rotation = rotation[parent_body] @ joint.parent_frame.rotation()
            parent_origin = (
                origin[parent_body] + rotation[parent_body] @ joint.parent_frame.translation
            )

            local_rotation, local_translation = joint.local_transform(coordinates)
            frame_rotation = parent_rotation @ local_rotation
            frame_origin = parent_origin + parent_rotation @ local_translation
            centres[joint.name] = frame_origin

            child_rotation = joint.child_frame.rotation()
            rotation[joint.child_frame.body] = frame_rotation @ child_rotation.T
            origin[joint.child_frame.body] = (
                frame_origin - rotation[joint.child_frame.body] @ joint.child_frame.translation
            )

        return ForwardResult(
            joint_centres=centres,
            body_rotation={k: v for k, v in rotation.items() if k != _GROUND},
            body_origin={k: v for k, v in origin.items() if k != _GROUND},
        )

    def forward_batch(
        self, coordinates: Mapping[str, np.ndarray], frames: int | None = None
    ) -> ForwardResult:
        """Place every body for a whole trial at once.

        Same model as `forward`, and a test holds them to the same numbers. This is the path
        the conversion runs on: per frame the work is a hundred-odd small scipy calls, which
        is twenty hours across the cohort and about ten minutes batched.
        """
        lengths = {int(np.shape(v)[0]) for v in coordinates.values() if np.ndim(v)}
        if len(lengths) > 1:
            raise ValueError(f"coordinates disagree on length: {sorted(lengths)}")
        if frames is None:
            if not lengths:
                raise ValueError("nothing says how many frames to place; pass `frames`")
            frames = lengths.pop()

        identity = np.tile(np.eye(3), (frames, 1, 1))
        rotation = {_GROUND: identity}
        origin = {_GROUND: np.zeros((frames, 3))}
        centres: dict[str, np.ndarray] = {}

        for joint in _in_tree_order(self.joints):
            parent_body = joint.parent_frame.body
            if parent_body not in rotation:
                raise UnsupportedJoint(
                    f"joint {joint.name!r} hangs off {parent_body!r}, which nothing places"
                )
            parent_rotation = rotation[parent_body] @ joint.parent_frame.rotation()
            parent_origin = origin[parent_body] + np.einsum(
                "tij,j->ti", rotation[parent_body], joint.parent_frame.translation
            )

            local_rotation, local_translation = joint.local_transform_batch(coordinates, frames)
            frame_rotation = parent_rotation @ local_rotation
            frame_origin = parent_origin + np.einsum(
                "tij,tj->ti", parent_rotation, local_translation
            )
            centres[joint.name] = frame_origin

            child_rotation = joint.child_frame.rotation()
            rotation[joint.child_frame.body] = frame_rotation @ child_rotation.T
            origin[joint.child_frame.body] = frame_origin - np.einsum(
                "tij,j->ti", rotation[joint.child_frame.body], joint.child_frame.translation
            )

        return ForwardResult(
            joint_centres=centres,
            body_rotation={k: v for k, v in rotation.items() if k != _GROUND},
            body_origin={k: v for k, v in origin.items() if k != _GROUND},
        )


def _in_tree_order(joints: tuple[KinematicJoint, ...]) -> list[KinematicJoint]:
    """Parents before children, without trusting the file's own ordering."""
    placed = {_GROUND}
    remaining = list(joints)
    ordered: list[KinematicJoint] = []
    while remaining:
        progressed = False
        for joint in list(remaining):
            if joint.parent_frame.body in placed:
                ordered.append(joint)
                placed.add(joint.child_frame.body)
                remaining.remove(joint)
                progressed = True
        if not progressed:
            unreachable = ", ".join(j.name for j in remaining)
            raise UnsupportedJoint(f"joints with no path to ground: {unreachable}")
    return ordered


# --------------------------------------------------------------------------- parsing


def _offset_frames(joint: ET.Element) -> dict[str, OffsetFrame]:
    frames: dict[str, OffsetFrame] = {}
    for element in joint.iter("PhysicalOffsetFrame"):
        name = element.get("name")
        socket = element.find("socket_parent")
        if not name or socket is None or not (socket.text or "").strip():
            continue
        parts = socket.text.strip().strip("/").split("/")
        body = _GROUND if parts[0] == _GROUND else (
            parts[1] if parts[0] == "bodyset" and len(parts) >= 2 else parts[-1]
        )
        translation = _floats(element.find("translation"))
        orientation = _floats(element.find("orientation"))
        frames[name] = OffsetFrame(
            name=name,
            body=body,
            translation=translation if translation.size == 3 else np.zeros(3),
            orientation=orientation if orientation.size == 3 else np.zeros(3),
        )
    return frames


def _resolve_frame(reference: str | None, frames: dict[str, OffsetFrame]) -> OffsetFrame:
    token = (reference or "").strip().strip("/")
    if token in frames:
        return frames[token]
    parts = token.split("/")
    if parts and parts[-1] in frames:
        return frames[parts[-1]]
    if parts[0] == _GROUND:
        return OffsetFrame(_GROUND, _GROUND, np.zeros(3), np.zeros(3))
    if parts[0] == "bodyset" and len(parts) >= 2:
        return OffsetFrame(parts[1], parts[1], np.zeros(3), np.zeros(3))
    return OffsetFrame(token, token, np.zeros(3), np.zeros(3))


def parse_kinematics(xml_text: str) -> KinematicModel:
    """Read the joints, frames and degree-of-freedom functions of an embedded model."""
    root = ET.fromstring(xml_text)
    objects = root.find(".//JointSet/objects")
    joints: list[KinematicJoint] = []
    for element in objects if objects is not None else []:
        kind = element.tag
        if kind not in ("CustomJoint", "PinJoint", "UniversalJoint", "WeldJoint"):
            raise UnsupportedJoint(
                f"joint {element.get('name')!r} is a {kind}, which is not implemented"
            )
        frames = _offset_frames(element)
        parent = element.find("socket_parent_frame")
        child = element.find("socket_child_frame")
        listing = element.find("./coordinates")
        coordinates = tuple(
            c.get("name") for c in (listing if listing is not None else []) if c.get("name")
        )

        axes: list[TransformAxis] = []
        spatial = element.find(".//SpatialTransform")
        for axis in spatial if spatial is not None else []:
            coordinate = axis.find("coordinates")
            named = (coordinate.text or "").strip() if coordinate is not None else ""
            direction = _floats(axis.find("axis"))
            function_node = _axis_function(axis)
            axes.append(
                TransformAxis(
                    name=axis.get("name") or "",
                    axis=direction if direction.size == 3 else np.zeros(3),
                    coordinate=named or None,
                    function=compile_function(function_node) if function_node is not None else None,
                )
            )

        joints.append(
            KinematicJoint(
                name=element.get("name") or "",
                kind=kind,
                parent_frame=_resolve_frame(
                    parent.text if parent is not None else None, frames
                ),
                child_frame=_resolve_frame(child.text if child is not None else None, frames),
                coordinates=coordinates,
                axes=tuple(axes),
            )
        )
    return KinematicModel(joints=tuple(joints))
