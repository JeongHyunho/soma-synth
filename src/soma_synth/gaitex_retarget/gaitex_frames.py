"""Read GAITEX trials in the shape the retarget already consumes, and repair the one coordinate
GAITEX publishes but did not solve.

``b3d_frames`` hands the retarget five things: per-frame OpenSim coordinate values, joint
centres, timestamps, the model text, and the subject's anthropometry. GAITEX supplies four of
them from ordinary files -- an ``.osim`` model, an ``.mot`` of coordinates, a ``.trc`` of
markers, a metadata JSON -- and supplies no joint centres at all, which is no obstacle because
the retarget already has a branch that recomputes them by forward kinematics.

**The root translation is the exception, and it is not a formatting difference.** GAITEX's IK
was driven by nine inertial units and solved orientation only: ``pelvis_tx``, ``pelvis_ty`` and
``pelvis_tz`` are frozen at their initial-pose values for every frame of every trial. A frozen
root is not a small error. The root is common to every site, so freezing it removes the same
term from all eight accelerometers at once.

**How large that term is depends on how it is measured, and the published figure overstates
it.** Taking the RMS of the recovered track over every frame gives ``1.88`` m/s^2 across the
cohort. But the recovered track steps at every change in which pelvis markers are visible (see
below), and those steps are differentiated twice along with the motion. Excluding a twenty-
sample window either side of each visibility change -- 0.6 per cent of frames -- drops the
median to ``0.911``; the 21 trials whose visible subset never changes give ``0.785``
independently. So the freeze costs of the order of ``0.8`` m/s^2, and roughly half of the
published ``1.88`` is this module's own artifact. For scale, that is above the per-channel
placement budgets in ADR-0037 (``0.064`` to ``0.980`` m/s^2) and well below the placement
error the emitted bundle actually carries, which ADR-0037 measures at about ``1.15`` g on the
shank and foot.

It is recoverable, because the model declares ``R_IAS``, ``L_IAS``, ``R_IPS`` and ``L_IPS`` as
stations on the pelvis body and the ``.trc`` observes all four. Fitting the observed cluster to
the declared shape gives the pelvis pose, and its translation stands in for the coordinate the
IK left behind.

**It is not exact, and the inexactness is not small.** The ``.osim``'s declared station
locations are left-right bit-symmetric -- a scaled generic template, not this subject's own
placement -- so they disagree with the observed cluster by up to ``94.9`` mm on one pair, and
the fit leaves a ``37`` mm median residual where a shape learned from the observations leaves
``2`` mm. Kabsch has no shape freedom to absorb that, so the recovered origin depends on which
markers are visible and steps a median ``14.6`` mm whenever the visible subset changes. The
residual is computed by ``solve_pose_series`` and this module does not yet consult it; the
recorded fix is to fit a learned shape. Until then, treat the root as good to a few centimetres
and expect a transient at each occlusion boundary.

What checks the recovery is forward kinematics, not the frozen value. Comparing the first frame
against the freeze is tempting and weak: the freeze holds the *static scaling* pose, which need
not be where the subject stood when the dynamic trial began, and for one published trial it is
``37`` mm out on the vertical. Driving the model with the recovered root and asking where it
predicts markers on other bodies is the question that has an answer -- the residual falls from
about ``148`` mm to about ``50`` mm across three trials.

Occlusions are blanked before the fit. GAITEX marks a lost marker as an exact zero triplet, and
fitting a cluster with the origin standing in for a hip would put the pelvis on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from ..addbio_retarget.osim_topology import OsimTopology, parse_osim
from ..addbio_retarget.world_frame import WorldFrameTransform, transform_for_gravity
from ..kinematics import markers as mk, rigid_body

__all__ = [
    "GaitexFile",
    "GaitexFrames",
    "GaitexReaderError",
    "GaitexSubject",
    "GaitexTrial",
    "MissingPelvisStations",
    "discover_subjects",
    "open_trial",
    "read_frames",
]

# The four pelvis landmarks the scaled model declares as stations on the pelvis body. Together
# they determine the body's pose, which is what the frozen coordinates failed to record.
PELVIS_STATIONS = ("R_IAS", "L_IAS", "R_IPS", "L_IPS")

# Coordinates already in metres. Everything else in a GAITEX `.mot` is in degrees, because the
# file's own header says `inDegrees=yes`.
TRANSLATIONAL_COORDINATES = frozenset({"pelvis_tx", "pelvis_ty", "pelvis_tz"})

# Directories under the extraction root that are not subjects.
NON_SUBJECT_PREFIXES = ("_", ".")


class GaitexReaderError(RuntimeError):
    """Raised when a GAITEX trial cannot be read as the retarget needs it."""


class MissingPelvisStations(GaitexReaderError):
    """Raised when a trial's markers cannot determine the pelvis pose.

    Distinct from the general error because it is a data condition rather than a defect: at
    least one trial in the published cohort simply does not observe all four stations, and the
    caller should record it as unavailable rather than fall back to the frozen root.
    """


@dataclass(frozen=True)
class GaitexSubject:
    """What GAITEX says about the person. It says very little.

    ``scaling_settings_*.xml`` carries ``<mass>90</mass>`` for every subject in the cohort and
    ``<height>-1</height>`` for every one, and no file anywhere records sex. These are therefore
    ``None`` rather than a number, so that a shape fit falls back to bone lengths alone instead
    of being anchored to a placeholder.
    """

    name: str
    biological_sex: str | None
    height_m: float | None
    mass_kg: float | None


@dataclass(frozen=True)
class GaitexTrial:
    """One subject-condition recording."""

    index: int
    subject: str
    condition: str
    name: str
    original_name: str
    length: int
    source_rate_hz: float
    root: pathlib.Path

    @property
    def model_path(self) -> pathlib.Path:
        return (
            self.root / "ik_imus" / "models"
            / f"scaled_model_{self.subject}_{self.condition}.osim"
        )

    @property
    def coordinates_path(self) -> pathlib.Path:
        return (
            self.root / "ik_imus" / "results_imu_ik"
            / f"ik_segment_registered_imu_data_{self.subject}_{self.condition}.mot"
        )

    @property
    def markers_path(self) -> pathlib.Path:
        return (
            self.root / "ik_imus"
            / f"marker_data_osim_format_{self.subject}_{self.condition}.trc"
        )

    @property
    def metadata_path(self) -> pathlib.Path:
        return self.root / f"metadata_{self.subject}_{self.condition}.json"


@dataclass(frozen=True)
class GaitexFile:
    """A trial's model and topology, in the shape ``generate_*_smpl24`` reads a ``B3DFile``."""

    trial: GaitexTrial
    topology: OsimTopology
    osim_text: str
    world_frame: WorldFrameTransform
    subject: GaitexSubject
    driven_bodies: tuple[str, ...]


@dataclass(frozen=True)
class GaitexFrames:
    """Per-frame coordinates on the retarget's terms.

    ``pos`` is ordered by ``topology.independent_coordinate_names``, in radians for rotational
    coordinates and metres for the three translational ones -- the convention
    ``PassFrames.pos`` uses. No joint centres: GAITEX publishes none, and the retarget's
    recompute branch supplies them.

    ``pelvis_residual_m`` and ``pelvis_visible_count`` say how well the root was recovered and
    from how many markers. Both are needed, and the reason is not the obvious one: a
    three-marker fit still leaves three constraints -- the triangle's side lengths -- so its
    residual is real rather than identically zero. It is however systematically SMALLER than a
    four-marker residual (median 0.8 mm against 2.1 mm on one published trial), because there is
    less for it to disagree with. Read alone it therefore ranks the weaker evidence higher, which
    is why the count travels beside it.
    """

    timestamps_s: np.ndarray
    source_rate_hz: float
    pos: np.ndarray
    root_frozen_value: np.ndarray
    pelvis_solved: np.ndarray
    pelvis_residual_m: np.ndarray
    pelvis_visible_count: np.ndarray


def discover_subjects(root: pathlib.Path) -> tuple[str, ...]:
    """Subject directories under an extraction root, in a stable order."""
    return tuple(sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(NON_SUBJECT_PREFIXES)
        and any(child.is_dir() for child in path.iterdir())
    ))


def _read_mot(path: pathlib.Path) -> tuple[list[str], np.ndarray, bool]:
    """Column names, the numeric block, and whether rotations are stored in degrees."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_degrees = None
    header_end = None
    for index, line in enumerate(text):
        stripped = line.strip()
        if stripped.lower().startswith("indegrees"):
            in_degrees = stripped.split("=", 1)[1].strip().lower() == "yes"
        if stripped == "endheader":
            header_end = index
            break
    if header_end is None:
        raise GaitexReaderError(f"no endheader in {path.name}")
    if in_degrees is None:
        raise GaitexReaderError(f"{path.name} does not declare inDegrees")
    columns = text[header_end + 1].split()
    rows = [line.split() for line in text[header_end + 2:] if line.strip()]
    values = np.asarray(rows, dtype=np.float64)
    if values.shape[1] != len(columns):
        raise GaitexReaderError(
            f"{path.name}: {values.shape[1]} numeric columns against {len(columns)} names"
        )
    return columns, values, in_degrees


def _pelvis_stations(osim_text: str) -> dict[str, np.ndarray]:
    """Where the scaled model puts the four pelvis landmarks, in the pelvis body frame."""
    out: dict[str, np.ndarray] = {}
    for marker in ET.fromstring(osim_text).iter("Marker"):
        name = marker.get("name")
        if name in PELVIS_STATIONS:
            location = marker.findtext("location")
            if location:
                out[name] = np.asarray([float(v) for v in location.split()], dtype=np.float64)
    return out


def _first_data_line(lines: list[str]) -> int:
    """Index of the first sample row in an OpenSim-format ``.trc``.

    The layout is five header lines -- file type, the metadata names, the metadata values, the
    marker names, the axis labels -- and some writers add a blank line after the axis row. The
    axis row is the landmark because it is the only header line whose first two fields are empty
    while later fields are filled; the marker-name row above it names ``Frame#`` and ``Time``.
    """
    for index in range(min(len(lines), 8)):
        fields = lines[index].split("\t")
        if len(fields) >= 3 and not fields[0].strip() and not fields[1].strip():
            start = index + 1
            while start < len(lines) and not lines[start].strip():
                start += 1
            return start
    raise GaitexReaderError("no axis row found in the first eight lines; not an OpenSim .trc")


def _declared_frame_count(lines: list[str]) -> int | None:
    """``NumFrames`` from the file's own metadata row, or ``None`` if it does not say."""
    if len(lines) < 3:
        return None
    keys = [value.strip() for value in lines[1].split("\t")]
    values = lines[2].split("\t")
    if "NumFrames" not in keys:
        return None
    slot = keys.index("NumFrames")
    if slot >= len(values):
        return None
    try:
        return int(float(values[slot]))
    except ValueError:
        return None


def _read_trc_cluster(
    path: pathlib.Path, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Observed positions of the named markers in metres, and their timestamps.

    Occlusions are blanked here rather than by the caller. GAITEX writes a lost marker as an
    exact zero triplet, and a rigid fit that accepts the origin as a hip puts the pelvis on the
    floor for that frame.

    The data begins after the axis row, which is found rather than assumed, and the frame count
    is checked against the header's own ``NumFrames``. A reader that started at a fixed line
    would drop frame 0 silently, because the remaining frames are all valid, and would make the
    marker series look as though it began one sample after the coordinates.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[3].split("\t")
    columns = {value.strip(): index for index, value in enumerate(header) if value.strip()}
    missing = [name for name in names if name not in columns]
    if missing:
        raise MissingPelvisStations(f"{path.name} does not observe {', '.join(missing)}")
    rows = [line.split("\t") for line in lines[_first_data_line(lines):] if line.strip()]
    declared = _declared_frame_count(lines)
    if declared is not None and declared != len(rows):
        raise GaitexReaderError(
            f"{path.name} declares NumFrames {declared} and carries {len(rows)} data rows"
        )
    observed = np.empty((len(rows), len(names), 3), dtype=np.float64)
    timestamps = np.empty(len(rows), dtype=np.float64)
    for frame, row in enumerate(rows):
        timestamps[frame] = float(row[1])
        for slot, name in enumerate(names):
            start = columns[name]
            observed[frame, slot] = (
                float(row[start]), float(row[start + 1]), float(row[start + 2])
            )
    observed /= 1000.0  # a trc is millimetres
    blanked, _ = mk.apply_occlusion_sentinel(observed)
    return blanked, timestamps


def _pelvis_origin_track(
    trial: GaitexTrial, osim_text: str, coordinate_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis body origin on the COORDINATE time grid, and which of those frames were solved.

    The two files share a start and a step but not a length: for one published trial both begin
    at ``0`` and step by ``0.010``, and the markers run two samples longer, to ``195.440``
    against ``195.420``. They are matched on time rather than on row number, which costs
    nothing here and is what keeps a source whose grids genuinely differ from being joined
    wrongly and silently.

    A coordinate time no marker frame is within half a sample of stays unsolved rather than
    borrowing a neighbour -- the same treatment an occluded frame gets. On this corpus that
    refusal never fires on a real frame: every matched pair agrees to under a microsecond,
    measured over all 72 trials. It is a guard against a source that does not hold, not a
    description of this one. The two grids are not offset by a sample; see ``_first_data_line``.
    """
    stations = _pelvis_stations(osim_text)
    absent = [name for name in PELVIS_STATIONS if name not in stations]
    if absent:
        raise MissingPelvisStations(
            f"{trial.model_path.name} declares no station for {', '.join(absent)}"
        )
    observed, marker_times = _read_trc_cluster(trial.markers_path, PELVIS_STATIONS)
    template = np.stack([stations[name] for name in PELVIS_STATIONS])
    centroid = template.mean(axis=0)

    # The shape is learned from the observations, not taken from the model. The .osim's declared
    # station locations are left-right bit-symmetric -- a scaled generic template rather than
    # this subject's own placement -- and disagree with the observed cluster by up to 94.9 mm on
    # one pair. Kabsch has no shape freedom, so fitting the template leaves a 37 mm median
    # residual AND makes the recovered origin depend on which markers happen to be visible: it
    # steps a median 16.8 mm at each change, and the emitter differentiates that twice into every
    # accelerometer. Against a shape learned from the cluster the residual is 1.8 mm and the step
    # is 3.7 mm, while the recovered track itself moves by a median of 0.1 mm -- the learned shape
    # does not change the answer, it removes the artifact.
    learned = rigid_body.mean_local_shape(observed)

    # The learned shape holds the true relative geometry in an arbitrary reference orientation.
    # One Procrustes alignment of the declared template onto it recovers the rotation between
    # that reference and the model's body frame, which is what makes the result a pelvis body
    # origin rather than a marker centroid. Were the template congruent this whole construction
    # would reduce to fitting it directly, so it generalises the old one rather than replacing it.
    to_body = rigid_body.proper_rotation_from_covariance(learned.T @ (template - centroid))

    pose = rigid_body.solve_pose_series(learned, observed)
    body_rotation = np.einsum("fij,kj->fik", pose.rotation, to_body)
    marker_origin = pose.translation - np.einsum("fij,j->fi", body_rotation, centroid)
    marker_origin[~pose.solved] = np.nan

    if marker_times.size < 2:
        raise GaitexReaderError(f"{trial.markers_path.name} carries no usable time column")
    tolerance = float(np.median(np.diff(marker_times))) / 2.0
    nearest = np.searchsorted(marker_times, coordinate_times)
    nearest = np.clip(nearest, 1, marker_times.size - 1)
    left, right = nearest - 1, nearest
    take_left = np.abs(coordinate_times - marker_times[left]) <= np.abs(
        coordinate_times - marker_times[right]
    )
    matched = np.where(take_left, left, right)
    within = np.abs(coordinate_times - marker_times[matched]) <= tolerance

    origin = np.full((coordinate_times.size, 3), np.nan)
    origin[within] = marker_origin[matched[within]]
    solved = within & np.isfinite(origin).all(axis=1)
    origin[~solved] = np.nan

    # Carried, not discarded. `solved` says a pose was computed; these say how much to trust it.
    # The two are not separable: a three-marker residual is smaller than a four-marker one on
    # this corpus (0.8 mm against 2.1 mm) because three points leave only three constraints, so
    # comparing residuals across frames with different counts ranks the thinner evidence first.
    # About five per cent of solved frames here are three-marker.
    residual = np.full(coordinate_times.size, np.nan)
    visible = np.zeros(coordinate_times.size, dtype=np.int8)
    residual[solved] = pose.residual_rms_m[matched[solved]]
    visible[solved] = pose.visible_count[matched[solved]]
    return origin, solved, residual, visible


def open_trial(root: pathlib.Path, subject: str, condition: str) -> GaitexFile:
    """Model, topology, world frame and driven-body set for one subject-condition trial."""
    trial = GaitexTrial(
        index=0,
        subject=subject,
        condition=condition,
        name=f"{subject}_{condition}",
        original_name=condition,
        length=0,          # filled in below, once the coordinate file has been counted
        source_rate_hz=0.0,
        root=root / subject / condition,
    )
    for required in (trial.model_path, trial.coordinates_path, trial.metadata_path):
        if not required.is_file():
            raise GaitexReaderError(f"missing {required.name} for {subject}/{condition}")

    osim_text = trial.model_path.read_text(encoding="utf-8")
    topology = parse_osim(osim_text)
    _, values, _ = _read_mot(trial.coordinates_path)
    length = int(values.shape[0])
    rate = 1.0 / float(np.median(np.diff(values[:, 0]))) if length > 1 else 0.0
    metadata = json.loads(trial.metadata_path.read_text(encoding="utf-8"))

    return GaitexFile(
        trial=replace(trial, length=length, source_rate_hz=round(rate, 6)),
        topology=topology,
        osim_text=osim_text,
        world_frame=transform_for_gravity(topology.gravity),
        subject=GaitexSubject(name=subject, biological_sex=None, height_m=None, mass_kg=None),
        driven_bodies=tuple(sorted(metadata.get("inverse_kinematics", {}))),
    )


def read_frames(source: GaitexFile, frame_indices=None) -> GaitexFrames:
    """Coordinates for the requested frames, with the root translation recovered.

    ``frame_indices`` selects rows after the recovery, never before it: the pelvis fit reads the
    marker series as it was recorded, and thinning first would ask about frames that were never
    sampled.
    """
    trial = source.trial
    columns, values, in_degrees = _read_mot(trial.coordinates_path)
    wanted = source.topology.independent_coordinate_names
    lookup = {name: index for index, name in enumerate(columns)}
    unknown = [name for name in wanted if name not in lookup]
    if unknown:
        raise GaitexReaderError(
            f"{trial.coordinates_path.name} has no column for {', '.join(unknown)}"
        )

    pos = np.empty((values.shape[0], len(wanted)), dtype=np.float64)
    for slot, name in enumerate(wanted):
        column = values[:, lookup[name]]
        if in_degrees and name not in TRANSLATIONAL_COORDINATES:
            column = np.deg2rad(column)
        pos[:, slot] = column

    timestamps = values[:, lookup["time"]] if "time" in lookup else values[:, 0]
    frozen = np.array(
        [pos[0, wanted.index(name)] for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    )
    origin, solved, residual, visible = _pelvis_origin_track(
        trial, source.osim_text, timestamps)
    for axis, name in enumerate(("pelvis_tx", "pelvis_ty", "pelvis_tz")):
        pos[:, wanted.index(name)] = origin[:, axis]
    if frame_indices is not None:
        selection = np.asarray(frame_indices, dtype=int)
        pos = pos[selection]
        timestamps = timestamps[selection]
        solved = solved[selection]
        residual = residual[selection]
        visible = visible[selection]

    return GaitexFrames(
        timestamps_s=timestamps,
        source_rate_hz=trial.source_rate_hz,
        pos=pos,
        root_frozen_value=frozen,
        pelvis_solved=solved,
        pelvis_residual_m=residual,
        pelvis_visible_count=visible,
    )
