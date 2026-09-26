"""Read-only reader for the GAITEX native layout.

GAITEX (Munz, Spilz & Oppel, Zenodo ``10.5281/zenodo.15729056``) ships one directory per
subject and condition. Each trial carries optical marker positions, Xsens IMU orientation,
labelled temporal segments, and a metadata mapping that binds OpenSim bodies to their
marker sets and IMU names.

Two header facts break a naive reader and are handled here rather than by callers:

* the marker file names its time column ``time_[s]`` while the IMU file names it
  ``time [s]`` -- underscore versus space;
* the raw IMU file orders quaternion components ``QW, QX, QY, QZ`` while the
  segment-registered file orders them ``QX, QY, QZ, QW``. Components are therefore matched
  by suffix, never by position, and the native order is preserved on the returned stream so
  that downstream provenance can record what the source actually said.

The reader converts marker positions from millimetres to metres. It performs no gap
handling, no filtering and no synthesis; those are separate stages.

This module is analysis-only. It does not generate artifacts and it lifts no hold.

This reader derives from the parent project's GAITEX source-parsing adapter. It is kept separate
because the parent's governed audit hashes its own copy (ADR-0041).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "GaitexReadError",
    "ImuOrientationStream",
    "MarkerStream",
    "TemporalSegment",
    "TrialMetadata",
    "TrialRef",
    "discover_trials",
    "read_imu_orientation",
    "read_markers",
    "read_metadata",
    "read_timestamps",
]


_MM_PER_M = 1000.0

_MARKER_TIME_COLUMN = "time_[s]"
_IMU_TIME_COLUMN = "time [s]"

# ``L_SHIN1_X_[mm]`` -> name ``L_SHIN1``, axis ``X``
_MARKER_COLUMN_RE = re.compile(r"^(?P<name>.+)_(?P<axis>[XYZ])_\[mm\]$")
# ``XSens_Pelvis_QW`` -> name ``XSens_Pelvis``, component ``W``
_IMU_COLUMN_RE = re.compile(r"^(?P<name>.+)_Q(?P<component>[WXYZ])$")

_MARKER_AXES = ("X", "Y", "Z")
_QUATERNION_COMPONENTS = ("W", "X", "Y", "Z")

_MARKER_FILE_PREFIX = "qualisys_marker_data_"
_IMU_FILE_PREFIX = "xsens_imu_data_"
_IMU_REGISTERED_INFIX = "segment_registered"
_TIMESTAMPS_FILE_PREFIX = "timestamps_"
_METADATA_FILE_PREFIX = "metadata_"


class GaitexReadError(RuntimeError):
    """Raised when a GAITEX file does not match the documented native layout."""


@dataclass(frozen=True)
class TrialRef:
    """Locator for one subject/condition trial."""

    subject: str
    condition: str
    directory: Path

    @property
    def stem(self) -> str:
        """The ``<subject>_<condition>`` suffix every file in the trial carries."""
        return f"{self.subject}_{self.condition}"

    def file(self, prefix: str, suffix: str) -> Path:
        return self.directory / f"{prefix}{self.stem}{suffix}"


@dataclass(frozen=True)
class MarkerStream:
    """Optical marker positions on the native time grid.

    ``positions_m`` is ``(frames, markers, 3)`` in metres. Occlusions are left exactly as
    the source encodes them -- GAITEX writes an exact ``(0, 0, 0)`` triplet, which this
    reader does *not* interpret. Sentinel handling belongs to the gap stage so that the
    conversion is explicit and testable on its own.
    """

    time_s: np.ndarray
    positions_m: np.ndarray
    marker_names: tuple[str, ...]

    def index_of(self, marker: str) -> int:
        try:
            return self.marker_names.index(marker)
        except ValueError as exc:  # pragma: no cover - message matters more than branch
            raise GaitexReadError(
                f"marker {marker!r} is not present; trial carries {len(self.marker_names)} markers"
            ) from exc

    def select(self, markers: Sequence[str]) -> np.ndarray:
        """Return ``(frames, len(markers), 3)`` for the named markers, in the order given."""
        return self.positions_m[:, [self.index_of(m) for m in markers], :]


@dataclass(frozen=True)
class ImuOrientationStream:
    """Xsens orientation quaternions on the native time grid.

    ``quaternions`` is ``(frames, sensors, 4)`` ordered ``(w, x, y, z)`` regardless of how
    the source file ordered its columns. ``native_component_order`` records the source
    order so provenance can state it.
    """

    time_s: np.ndarray
    quaternions: np.ndarray
    imu_names: tuple[str, ...]
    native_component_order: tuple[str, ...]

    def index_of(self, imu: str) -> int:
        try:
            return self.imu_names.index(imu)
        except ValueError as exc:  # pragma: no cover
            raise GaitexReadError(
                f"IMU {imu!r} is not present; trial carries {self.imu_names}"
            ) from exc


@dataclass(frozen=True)
class TemporalSegment:
    """One labelled interval from the trial's timestamps file."""

    label: str
    start_s: float
    end_s: float
    velocity_km_h: float | None


@dataclass(frozen=True)
class TrialMetadata:
    """The trial's marker/body/IMU binding, as declared by the source."""

    body_to_markers: Mapping[str, tuple[str, ...]]
    body_to_imu: Mapping[str, str]

    @property
    def imu_to_body(self) -> dict[str, str]:
        return {imu: body for body, imu in self.body_to_imu.items()}


def _read_header_and_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Return the comma-separated header fields and the numeric body of a GAITEX CSV."""
    if not path.is_file():
        raise GaitexReadError(f"missing GAITEX file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise GaitexReadError(f"empty GAITEX file: {path}")
        header = [field.strip() for field in header_line.rstrip("\r\n").split(",")]
        matrix = np.loadtxt(handle, delimiter=",", dtype=np.float64, ndmin=2)
    if matrix.size == 0:
        matrix = np.empty((0, len(header)), dtype=np.float64)
    if matrix.shape[1] != len(header):
        raise GaitexReadError(
            f"{path}: header declares {len(header)} columns but body has {matrix.shape[1]}"
        )
    return header, matrix


def read_markers(path: Path) -> MarkerStream:
    """Read ``qualisys_marker_data_<subject>_<condition>.csv``."""
    header, matrix = _read_header_and_matrix(path)
    if header[0] != _MARKER_TIME_COLUMN:
        raise GaitexReadError(
            f"{path}: expected first column {_MARKER_TIME_COLUMN!r}, found {header[0]!r}"
        )

    # Preserve first-appearance order so the returned axis matches the file.
    order: list[str] = []
    columns: dict[str, dict[str, int]] = {}
    for position, field in enumerate(header[1:], start=1):
        match = _MARKER_COLUMN_RE.match(field)
        if match is None:
            raise GaitexReadError(f"{path}: unparsable marker column {field!r}")
        name = match.group("name")
        if name not in columns:
            columns[name] = {}
            order.append(name)
        axis = match.group("axis")
        if axis in columns[name]:
            raise GaitexReadError(f"{path}: duplicate column for marker {name!r} axis {axis}")
        columns[name][axis] = position

    incomplete = [name for name in order if set(columns[name]) != set(_MARKER_AXES)]
    if incomplete:
        raise GaitexReadError(
            f"{path}: markers missing an axis column: {sorted(incomplete)[:5]}"
        )

    picks = np.array([[columns[name][axis] for axis in _MARKER_AXES] for name in order])
    positions_mm = matrix[:, picks.reshape(-1)].reshape(matrix.shape[0], len(order), 3)
    return MarkerStream(
        time_s=matrix[:, 0].copy(),
        positions_m=positions_mm / _MM_PER_M,
        marker_names=tuple(order),
    )


def read_imu_orientation(path: Path) -> ImuOrientationStream:
    """Read ``xsens_imu_data[_segment_registered]_<subject>_<condition>.csv``.

    Components are matched by their ``_Q<component>`` suffix, so the raw file's
    ``QW, QX, QY, QZ`` and the registered file's ``QX, QY, QZ, QW`` both load correctly and
    both come back ``(w, x, y, z)``.
    """
    header, matrix = _read_header_and_matrix(path)
    if header[0] != _IMU_TIME_COLUMN:
        raise GaitexReadError(
            f"{path}: expected first column {_IMU_TIME_COLUMN!r}, found {header[0]!r}"
        )

    order: list[str] = []
    columns: dict[str, dict[str, int]] = {}
    native_order: list[str] = []
    for position, field in enumerate(header[1:], start=1):
        match = _IMU_COLUMN_RE.match(field)
        if match is None:
            raise GaitexReadError(f"{path}: unparsable IMU column {field!r}")
        name = match.group("name")
        component = match.group("component")
        if name not in columns:
            columns[name] = {}
            order.append(name)
        if component in columns[name]:
            raise GaitexReadError(f"{path}: duplicate component Q{component} for {name!r}")
        columns[name][component] = position
        if len(order) == 1:
            native_order.append(component)

    incomplete = [name for name in order if set(columns[name]) != set(_QUATERNION_COMPONENTS)]
    if incomplete:
        raise GaitexReadError(
            f"{path}: IMUs missing a quaternion component: {sorted(incomplete)}"
        )

    picks = np.array(
        [[columns[name][component] for component in _QUATERNION_COMPONENTS] for name in order]
    )
    quaternions = matrix[:, picks.reshape(-1)].reshape(matrix.shape[0], len(order), 4)
    return ImuOrientationStream(
        time_s=matrix[:, 0].copy(),
        quaternions=quaternions,
        imu_names=tuple(order),
        native_component_order=tuple(native_order),
    )


def read_timestamps(path: Path) -> tuple[TemporalSegment, ...]:
    """Read ``timestamps_<subject>_<condition>.csv`` into labelled segments."""
    if not path.is_file():
        raise GaitexReadError(f"missing GAITEX file: {path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise GaitexReadError(f"empty GAITEX file: {path}")

    header = [field.strip() for field in lines[0].split(",")]
    if header[0] != "label":
        raise GaitexReadError(f"{path}: expected first column 'label', found {header[0]!r}")

    def column(candidates: tuple[str, ...]) -> int | None:
        for candidate in candidates:
            if candidate in header:
                return header.index(candidate)
        return None

    start_at = column(("temporal_segment_start_[s]",))
    end_at = column(("temporal_segment_end_[s]",))
    velocity_at = column(("velocities_[km_h]",))
    if start_at is None or end_at is None:
        raise GaitexReadError(f"{path}: missing segment start/end columns in {header}")

    segments: list[TemporalSegment] = []
    for row_number, line in enumerate(lines[1:], start=2):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < len(header):
            raise GaitexReadError(f"{path}:{row_number}: expected {len(header)} fields")
        velocity: float | None = None
        if velocity_at is not None and fields[velocity_at]:
            velocity = float(fields[velocity_at])
        segments.append(
            TemporalSegment(
                label=fields[0],
                start_s=float(fields[start_at]),
                end_s=float(fields[end_at]),
                velocity_km_h=velocity,
            )
        )
    return tuple(segments)


def read_metadata(path: Path) -> TrialMetadata:
    """Read ``metadata_<subject>_<condition>.json``.

    The source nests the binding under ``inverse_kinematics`` as
    ``{body: {"marker_names": [...], "imu_name": "..."}}``.
    """
    if not path.is_file():
        raise GaitexReadError(f"missing GAITEX file: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise GaitexReadError(f"{path}: expected a JSON object at the top level")
    bindings = document.get("inverse_kinematics")
    if not isinstance(bindings, dict):
        raise GaitexReadError(f"{path}: missing an 'inverse_kinematics' object")

    body_to_markers: dict[str, tuple[str, ...]] = {}
    body_to_imu: dict[str, str] = {}
    for body, binding in bindings.items():
        if not isinstance(binding, dict):
            raise GaitexReadError(f"{path}: binding for body {body!r} is not an object")
        markers = binding.get("marker_names")
        if not isinstance(markers, list) or not all(isinstance(m, str) for m in markers):
            raise GaitexReadError(f"{path}: body {body!r} has no marker_names list")
        body_to_markers[body] = tuple(markers)
        imu = binding.get("imu_name")
        if isinstance(imu, str) and imu:
            body_to_imu[body] = imu
    return TrialMetadata(body_to_markers=body_to_markers, body_to_imu=body_to_imu)


def _is_trial_directory(directory: Path, subject: str, condition: str) -> bool:
    stem = f"{subject}_{condition}"
    return (directory / f"{_MARKER_FILE_PREFIX}{stem}.csv").is_file()


def discover_trials(root: Path) -> tuple[TrialRef, ...]:
    """Enumerate ``<root>/<subject>/<condition>`` trials, sorted for determinism.

    A directory counts as a trial only when it holds the marker file whose name matches its
    own subject and condition. Provenance and bookkeeping directories such as
    ``_provenance`` therefore never enter the listing.
    """
    root = Path(root)
    if not root.is_dir():
        raise GaitexReadError(f"GAITEX root is not a directory: {root}")

    def subdirectories(parent: Path) -> Iterator[Path]:
        for child in sorted(parent.iterdir(), key=lambda item: item.name):
            if child.is_dir() and not child.name.startswith("_"):
                yield child

    trials: list[TrialRef] = []
    for subject_dir in subdirectories(root):
        for condition_dir in subdirectories(subject_dir):
            if _is_trial_directory(condition_dir, subject_dir.name, condition_dir.name):
                trials.append(
                    TrialRef(
                        subject=subject_dir.name,
                        condition=condition_dir.name,
                        directory=condition_dir,
                    )
                )
    return tuple(trials)


def marker_path(trial: TrialRef) -> Path:
    return trial.file(_MARKER_FILE_PREFIX, ".csv")


def imu_path(trial: TrialRef, *, segment_registered: bool = False) -> Path:
    prefix = _IMU_FILE_PREFIX
    if segment_registered:
        prefix = f"{_IMU_FILE_PREFIX}{_IMU_REGISTERED_INFIX}_"
    return trial.file(prefix, ".csv")


def timestamps_path(trial: TrialRef) -> Path:
    return trial.file(_TIMESTAMPS_FILE_PREFIX, ".csv")


def metadata_path(trial: TrialRef) -> Path:
    return trial.file(_METADATA_FILE_PREFIX, ".json")
