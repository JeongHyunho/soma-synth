"""Address and decode `.b3d` frames without nimblephysics.

A `.b3d` is an 8-byte little-endian header length, a `SubjectOnDiskHeader` protobuf, then a
flat run of fixed-width records: for each trial, for each frame, one `SubjectOnDiskSensorFrame`
followed by one `SubjectOnDiskProcessingPassFrame` per processing pass *of that trial*. There is
no index and no delimiter, so a frame is reached only by summing the trials before it -- and a
slip lands on a neighbouring record that decodes perfectly well into the wrong pose. The layout
is therefore checked against the file's own length before anything is read.

Pass counts vary per trial, so the `dynamics` pass does not sit at a fixed index; it is looked
up per trial from that trial's own pass headers.

Files are opened read-only and seeked; a single subject can exceed 2 GB and is never read whole.
"""

from __future__ import annotations

import pathlib
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .osim_topology import OsimTopology, parse_osim
from .world_frame import WorldFrameTransform, transform_for_gravity

from soma_synth.vendor.nimblephysics import SubjectOnDisk_pb2 as _protobuf

__all__ = [
    "B3DFile",
    "B3DLayoutError",
    "PassFrames",
    "SubjectMeta",
    "TrialSpan",
    "open_b3d",
    "read_pass_frames",
]

_HEADER_LENGTH_BYTES = 8
_DYNAMICS = "dynamics"


class B3DLayoutError(RuntimeError):
    """The file does not match the layout its own header declares."""


@dataclass(frozen=True)
class SubjectMeta:
    biological_sex: str
    height_m: float
    mass_kg: float
    age_years: int
    num_dofs: int
    href: str
    subject_tags: tuple[str, ...]


@dataclass(frozen=True)
class TrialSpan:
    index: int
    name: str
    original_name: str
    split_index: int
    start_byte: int
    frame_size: int
    pass_count: int
    pass_names: tuple[str, ...]
    length: int
    timestep_s: float
    num_force_plates: int
    force_plate_corners: tuple[float, ...]

    @property
    def dynamics_pass_index(self) -> int:
        """Where this trial keeps its dynamics pass, which is not a per-file constant."""
        try:
            return self.pass_names.index(_DYNAMICS)
        except ValueError as exc:
            raise B3DLayoutError(
                f"trial {self.index} has no {_DYNAMICS} pass: {self.pass_names}"
            ) from exc

    @property
    def source_rate_hz(self) -> float:
        return 1.0 / self.timestep_s if self.timestep_s > 0 else float("nan")


@dataclass(frozen=True)
class B3DFile:
    path: pathlib.Path
    header: object
    payload_offset: int
    trials: tuple[TrialSpan, ...]
    topology: OsimTopology
    subject: SubjectMeta
    trailing_bytes: int

    @property
    def world_frame(self) -> WorldFrameTransform:
        """The rotation into the corpus Z-up world, taken from this file's own gravity."""
        if self.topology.gravity is None:
            raise B3DLayoutError(
                f"{self.path.name}: the model declares no gravity, so which way is up is "
                "unknown; refusing to assume a frame convention"
            )
        return transform_for_gravity(self.topology.gravity)


@dataclass(frozen=True)
class PassFrames:
    trial_index: int
    pass_index: int
    pass_name: str
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    source_rate_hz: float
    pos: np.ndarray                     # (T, num_dofs)
    joint_centres: np.ndarray           # (T, num_centres, 3)
    ground_contact_force: np.ndarray    # (T, num_contact_bodies, 3)
    ground_contact_cop: np.ndarray      # (T, num_contact_bodies, 3)
    com_pos: np.ndarray                 # (T, 3)


def _pass_name(value: int) -> str:
    return _protobuf.ProcessingPassType.Name(value)


def _read_header(path: pathlib.Path):
    with path.open("rb") as handle:
        prefix = handle.read(_HEADER_LENGTH_BYTES)
        if len(prefix) < _HEADER_LENGTH_BYTES:
            raise B3DLayoutError(f"{path.name}: shorter than the header length prefix")
        (header_size,) = struct.unpack("<q", prefix)
        if header_size <= 0:
            raise B3DLayoutError(f"{path.name}: declares a header of {header_size} bytes")
        blob = handle.read(header_size)
        if len(blob) < header_size:
            raise B3DLayoutError(
                f"{path.name}: header declares {header_size} bytes, file holds {len(blob)}"
            )
    header = _protobuf.SubjectOnDiskHeader()
    header.ParseFromString(blob)
    return header, _HEADER_LENGTH_BYTES + header_size


def _model_text(header) -> str:
    """The scaled model, taken from the dynamics pass -- the one the retarget consumes."""
    for entry in header.passes:
        if _pass_name(entry.pass_type) == _DYNAMICS and entry.model_osim_text:
            return entry.model_osim_text
    for entry in header.passes:
        if entry.model_osim_text:
            return entry.model_osim_text
    raise B3DLayoutError("no processing pass carries a model")


def open_b3d(path: str | pathlib.Path, *, strict_size: bool = True) -> B3DFile:
    """Read the header, lay out every trial, and check the arithmetic against the file size."""
    path = pathlib.Path(path)
    header, payload_offset = _read_header(path)

    offset = payload_offset
    spans: list[TrialSpan] = []
    for index, trial in enumerate(header.trial_header):
        pass_names = tuple(_pass_name(p.type) for p in trial.processing_pass_header)
        frame_size = (
            header.raw_sensor_frame_size
            + len(pass_names) * header.processing_pass_frame_size
        )
        spans.append(
            TrialSpan(
                index=index,
                name=trial.name,
                original_name=trial.original_name,
                split_index=trial.split_index,
                start_byte=offset,
                frame_size=frame_size,
                pass_count=len(pass_names),
                pass_names=pass_names,
                length=trial.trial_length,
                timestep_s=trial.trial_timestep,
                num_force_plates=trial.num_force_plates,
                force_plate_corners=tuple(trial.force_plate_corners),
            )
        )
        offset += trial.trial_length * frame_size

    size = path.stat().st_size
    if offset > size:
        raise B3DLayoutError(
            f"{path.name}: layout needs {offset} bytes, file holds {size}"
        )
    trailing = size - offset
    if strict_size and trailing:
        raise B3DLayoutError(
            f"{path.name}: {trailing} bytes past the end of the declared layout"
        )

    topology = parse_osim(_model_text(header))
    subject = SubjectMeta(
        biological_sex=header.biological_sex,
        height_m=header.height_m,
        mass_kg=header.mass_kg,
        age_years=header.age_years,
        num_dofs=header.num_dofs,
        href=header.href,
        subject_tags=tuple(header.subject_tag),
    )
    return B3DFile(
        path=path,
        header=header,
        payload_offset=payload_offset,
        trials=tuple(spans),
        topology=topology,
        subject=subject,
        trailing_bytes=trailing,
    )


def _as_points(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array.reshape(-1, 3) if array.size else array.reshape(0, 3)


def read_pass_frames(
    b3d: B3DFile,
    trial_index: int,
    frame_indices: Iterable[int],
    *,
    pass_index: int | None = None,
) -> PassFrames:
    """Decode selected frames of one processing pass of one trial.

    ``pass_index`` defaults to that trial's dynamics pass, which is the pass the retarget
    design pairs with the model it also reads from the header.
    """
    if not 0 <= trial_index < len(b3d.trials):
        raise B3DLayoutError(
            f"trial {trial_index} is outside 0..{len(b3d.trials) - 1}"
        )
    trial = b3d.trials[trial_index]
    if pass_index is None:
        pass_index = trial.dynamics_pass_index
    if not 0 <= pass_index < trial.pass_count:
        raise B3DLayoutError(
            f"trial {trial_index} has {trial.pass_count} passes; asked for {pass_index}"
        )

    wanted = np.asarray(list(frame_indices), dtype=np.int64)
    if wanted.size and (wanted.min() < 0 or wanted.max() >= trial.length):
        raise B3DLayoutError(
            f"trial {trial_index} holds {trial.length} frames; asked for "
            f"{wanted.min()}..{wanted.max()}"
        )

    record_size = b3d.header.processing_pass_frame_size
    pos, centres, force, cop, com = [], [], [], [], []
    with b3d.path.open("rb") as handle:
        for frame in wanted:
            handle.seek(
                trial.start_byte
                + int(frame) * trial.frame_size
                + b3d.header.raw_sensor_frame_size
                + pass_index * record_size
            )
            blob = handle.read(record_size)
            if len(blob) < record_size:
                raise B3DLayoutError(
                    f"trial {trial_index} frame {frame}: record truncated"
                )
            record = _protobuf.SubjectOnDiskProcessingPassFrame()
            record.ParseFromString(blob)
            pos.append(np.asarray(record.pos, dtype=np.float64))
            centres.append(_as_points(record.world_frame_joint_centers))
            force.append(_as_points(record.ground_contact_force))
            cop.append(_as_points(record.ground_contact_center_of_pressure))
            com.append(np.asarray(record.com_pos, dtype=np.float64))

    stack = np.stack if pos else (lambda seq: np.empty((0,)))
    return PassFrames(
        trial_index=trial_index,
        pass_index=pass_index,
        pass_name=trial.pass_names[pass_index],
        frame_indices=wanted,
        timestamps_s=wanted.astype(np.float64) * trial.timestep_s,
        source_rate_hz=trial.source_rate_hz,
        pos=stack(pos),
        joint_centres=stack(centres),
        ground_contact_force=stack(force),
        ground_contact_cop=stack(cop),
        com_pos=stack(com),
    )
