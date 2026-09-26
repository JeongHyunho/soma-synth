"""Read the skeleton definition that each .b3d header carries, and derive the two array orders.

A `.b3d` processing-pass frame stores `pos` and `world_frame_joint_centers` as bare `double`
arrays. Neither carries names, and both are shorter than the model's own lists -- `pos` by the
constraint-dependent coordinates, the centres by the joints those coordinates belong to. The
orders are therefore *derived from the model in the same file*, never hardcoded, so a cohort
whose skeleton differs cannot silently shift every downstream joint by one.

Serialisation note: in these models a `TransformAxis` carries its function as a direct child
whose tag *is* the concrete type (`LinearFunction`, `SimmSpline`, `MultiplierFunction`); there
is no `<function>` wrapper. Looking for the wrapper finds nothing and makes every joint look
rotation-only, which in turn claims the femur and tibia hold a constant distance across the
walker knee. They do not.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

__all__ = ["OsimJoint", "OsimTopology", "parse_osim"]

_AXIS_METADATA_TAGS = frozenset({"coordinates", "axis"})


@dataclass(frozen=True)
class OsimJoint:
    """One joint of the model, with the facts the retarget needs about it."""

    name: str
    parent_body: str | None
    child_body: str | None
    coordinates: tuple[str, ...]
    translates: bool

    def fixed_in(self) -> frozenset[str]:
        """Bodies in which this joint's centre is a fixed point.

        The centre is the joint frame origin: always fixed in the child body, and also fixed
        in the parent body when the joint contributes no translation. ``ground`` is not a
        body of the subject and never anchors anything.
        """
        bodies = {self.child_body}
        if not self.translates:
            bodies.add(self.parent_body)
        bodies.discard(None)
        bodies.discard("ground")
        return frozenset(bodies)


@dataclass(frozen=True)
class OsimTopology:
    """The model's joints plus the coordinate coupling that shrinks the on-disk arrays."""

    joints: tuple[OsimJoint, ...]
    dependent_coordinates: tuple[str, ...]
    #: The model's own gravity vector, which is what says which way is up. ``None`` when the
    #: model does not declare one -- callers must refuse rather than assume a convention.
    gravity: tuple[float, float, float] | None = None

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """Every coordinate the model declares, in model order."""
        return tuple(c for j in self.joints for c in j.coordinates)

    @property
    def independent_coordinate_names(self) -> tuple[str, ...]:
        """The order of `SubjectOnDiskProcessingPassFrame.pos`."""
        dependent = set(self.dependent_coordinates)
        return tuple(c for c in self.coordinate_names if c not in dependent)

    @property
    def centre_joints(self) -> tuple[OsimJoint, ...]:
        """The order of `SubjectOnDiskProcessingPassFrame.world_frame_joint_centers`.

        A joint driven entirely by dependent coordinates carries no independent degree of
        freedom and is absent from the stored centres.
        """
        dependent = set(self.dependent_coordinates)
        return tuple(
            j for j in self.joints
            if not (j.coordinates and all(c in dependent for c in j.coordinates))
        )

    @property
    def joint_centre_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.centre_joints)

    def rigid_centre_pairs(self) -> list[list[bool]]:
        """Which pairs of stored centres must hold a constant distance, by construction.

        Used as a self-check against the frame data: the prediction is one-directional --
        a predicted-rigid pair that measures non-rigid is a real defect, while the converse
        only means a degree of freedom did not move in the frames looked at.
        """
        homes = [j.fixed_in() for j in self.centre_joints]
        return [[bool(a & b) for b in homes] for a in homes]


def _body_from_socket(path: str | None, local_frames: dict[str, str]) -> str | None:
    """'/bodyset/femur_r/femur_r_offset' -> 'femur_r'; '/ground' -> 'ground'."""
    if not path:
        return None
    parts = path.strip().strip("/").split("/")
    if parts[0] == "ground":
        return "ground"
    if parts[0] == "bodyset":
        return parts[1] if len(parts) >= 2 else None
    return local_frames.get(parts[-1])


def _local_offset_frames(joint: ET.Element) -> dict[str, str]:
    """Offset frames declared inside the joint, mapped to the body they hang off."""
    frames: dict[str, str] = {}
    for offset in joint.iter("PhysicalOffsetFrame"):
        name = offset.get("name")
        socket = offset.find("socket_parent")
        if not name or socket is None or not socket.text:
            continue
        parts = socket.text.strip().strip("/").split("/")
        if parts[0] == "ground":
            frames[name] = "ground"
        elif parts[0] == "bodyset" and len(parts) >= 2:
            frames[name] = parts[1]
        else:
            frames[name] = parts[-1]
    return frames


def _axis_function(axis: ET.Element) -> ET.Element | None:
    """The function element of a TransformAxis, wrapped or not."""
    for child in axis:
        if child.tag in _AXIS_METADATA_TAGS:
            continue
        if child.tag == "function":
            return next(iter(child), None)
        return child
    return None


def _joint_translates(joint: ET.Element) -> bool:
    """True when a translation axis is driven by a coordinate through a non-constant function."""
    spatial = joint.find(".//SpatialTransform")
    if spatial is None:
        return False
    for axis in spatial:
        if not (axis.get("name") or "").startswith("translation"):
            continue
        coords = axis.find("coordinates")
        driven = coords is not None and (coords.text or "").strip() != ""
        if not driven:
            continue
        function = _axis_function(axis)
        if function is not None and function.tag != "Constant":
            return True
    return False


def _joint_coordinates(joint: ET.Element) -> tuple[str, ...]:
    listing = joint.find("./coordinates")
    if listing is None:
        return ()
    return tuple(c.get("name") for c in listing if c.get("name"))


def parse_osim(xml_text: str) -> OsimTopology:
    """Parse the scaled OpenSim model embedded in a .b3d processing pass."""
    root = ET.fromstring(xml_text)

    joints: list[OsimJoint] = []
    joint_objects = root.find(".//JointSet/objects")
    for element in joint_objects if joint_objects is not None else []:
        local_frames = _local_offset_frames(element)
        parent = element.find("socket_parent_frame")
        child = element.find("socket_child_frame")
        joints.append(
            OsimJoint(
                name=element.get("name") or "",
                parent_body=_body_from_socket(
                    parent.text if parent is not None else None, local_frames
                ),
                child_body=_body_from_socket(
                    child.text if child is not None else None, local_frames
                ),
                coordinates=_joint_coordinates(element),
                translates=_joint_translates(element),
            )
        )

    dependent: list[str] = []
    constraint_objects = root.find(".//ConstraintSet/objects")
    for element in constraint_objects if constraint_objects is not None else []:
        name = element.find("dependent_coordinate_name")
        if name is not None and (name.text or "").strip():
            dependent.append(name.text.strip())

    return OsimTopology(
        joints=tuple(joints),
        dependent_coordinates=tuple(dependent),
        gravity=_gravity(root),
    )


def _gravity(root: ET.Element) -> tuple[float, float, float] | None:
    """The model's gravity vector -- the only statement in the file about which way is up."""
    element = root.find(".//Model/gravity")
    if element is None:
        element = root.find(".//gravity")
    if element is None or not (element.text or "").strip():
        return None
    parts = element.text.split()
    if len(parts) != 3:
        return None
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None
