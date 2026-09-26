"""Reading a GAITEX trial, and recovering the root its IK froze.

Two parts. The synthetic part builds a whole miniature trial -- ``.osim``, ``.mot``, ``.trc``
and metadata -- so that units, the dependent knee coordinate, occlusion blanking and a
missing station are checked against numbers this file chose and can therefore state exactly.
The corpus part runs the same reader over ``austra/gwo``, which is where the claims that
matter are not analytic: that the recovered root actually moves, and that driving forward
kinematics with it puts the model's stations nearer their observed markers than the frozen
constant does.

The synthetic marker series deliberately starts one sample *before* the coordinate series,
unlike the archive (``austra/gwo`` starts both at ``0.000`` and runs the markers two samples
longer). This exercises the join against a genuinely offset grid, which the corpus does not
provide, and keeps the fixtures independent of where the reader decides a ``.trc`` header ends.
The helpers here find that boundary for themselves rather than borrowing the reader's answer,
and ``TestTimeAlignment`` asserts the archive's real alignment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.addbio_retarget.osim_kinematics import parse_kinematics
from soma_synth.gaitex_retarget import gaitex_frames as gx
from soma_synth.kinematics import rigid_body as rb
from soma_synth.pipeline import paths


def _corpus_root() -> Path | None:
    """The GAITEX source folder (``SOMA_SOURCE_ROOT``, else ``<SOMA_DATA_ROOT>/extracted``), or
    None when neither variable resolves."""
    try:
        return paths.source_dir("gaitex")
    except paths.PathConfigError:
        return None


CORPUS_ROOT = _corpus_root()
NEEDS_CORPUS = pytest.mark.skipif(
    CORPUS_ROOT is None or not (CORPUS_ROOT / "austra" / "gwo").is_dir(),
    reason="SOMA_DATA_ROOT / SOMA_SOURCE_ROOT unset, or the GAITEX source is not under it",
)

DT_S = 0.01
SYNTHETIC_FRAMES = 40
ROOT_COORDINATES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")

# Where the miniature model puts its stations, in the body frame. The four pelvis landmarks
# are non-coplanar, so the cluster fit is determined rather than merely consistent.
PELVIS_LOCAL_M = {
    "R_IAS": (0.035, 0.020, 0.128),
    "L_IAS": (0.035, 0.022, -0.128),
    "R_IPS": (-0.092, 0.031, 0.048),
    "L_IPS": (-0.092, 0.034, -0.048),
}
TIBIA_LOCAL_M = {"R_FAL": (-0.005, -0.402, 0.053)}

# One value per coordinate, held constant across the trial: these tests are about units and
# column selection, not about motion. Rotations are degrees because the `.mot` says so; the
# three root translations are metres in the same file.
SYNTHETIC_COORDINATES = {
    "pelvis_tilt": 3.0,
    "pelvis_list": -2.0,
    "pelvis_rotation": 1.0,
    "pelvis_tx": 0.5,
    "pelvis_ty": 0.95,
    "pelvis_tz": -0.25,
    "hip_flexion_r": 30.0,
    "knee_angle_r": 12.5,
    "knee_angle_r_beta": 0.1466979,
    "knee_angle_l": -4.0,
    "knee_angle_l_beta": 0.0935194,
}

_MINI_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="mini_gaitex">
    <gravity>0 -9.80665 0</gravity>
    <JointSet>
      <objects>
        <CustomJoint name="ground_pelvis">
          <socket_parent_frame>/ground</socket_parent_frame>
          <socket_child_frame>/bodyset/pelvis</socket_child_frame>
          <coordinates>
            <Coordinate name="pelvis_tilt"/>
            <Coordinate name="pelvis_list"/>
            <Coordinate name="pelvis_rotation"/>
            <Coordinate name="pelvis_tx"/>
            <Coordinate name="pelvis_ty"/>
            <Coordinate name="pelvis_tz"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>pelvis_tilt</coordinates><axis>0 0 1</axis>
              <function>
                <LinearFunction><coefficients>1 0</coefficients></LinearFunction>
              </function>
            </TransformAxis>
            <TransformAxis name="translation1">
              <coordinates>pelvis_tx</coordinates><axis>1 0 0</axis>
              <function>
                <LinearFunction><coefficients>1 0</coefficients></LinearFunction>
              </function>
            </TransformAxis>
            <TransformAxis name="translation2">
              <coordinates>pelvis_ty</coordinates><axis>0 1 0</axis>
              <function>
                <LinearFunction><coefficients>1 0</coefficients></LinearFunction>
              </function>
            </TransformAxis>
            <TransformAxis name="translation3">
              <coordinates>pelvis_tz</coordinates><axis>0 0 1</axis>
              <function>
                <LinearFunction><coefficients>1 0</coefficients></LinearFunction>
              </function>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <PinJoint name="hip_r">
          <socket_parent_frame>/bodyset/pelvis</socket_parent_frame>
          <socket_child_frame>/bodyset/femur_r</socket_child_frame>
          <coordinates><Coordinate name="hip_flexion_r"/></coordinates>
        </PinJoint>
        <PinJoint name="knee_r">
          <socket_parent_frame>/bodyset/femur_r</socket_parent_frame>
          <socket_child_frame>/bodyset/tibia_r</socket_child_frame>
          <coordinates><Coordinate name="knee_angle_r"/></coordinates>
        </PinJoint>
        <PinJoint name="patellofemoral_r">
          <socket_parent_frame>/bodyset/femur_r</socket_parent_frame>
          <socket_child_frame>/bodyset/patella_r</socket_child_frame>
          <coordinates><Coordinate name="knee_angle_r_beta"/></coordinates>
        </PinJoint>
        <PinJoint name="knee_l">
          <socket_parent_frame>/bodyset/femur_l</socket_parent_frame>
          <socket_child_frame>/bodyset/tibia_l</socket_child_frame>
          <coordinates><Coordinate name="knee_angle_l"/></coordinates>
        </PinJoint>
        <PinJoint name="patellofemoral_l">
          <socket_parent_frame>/bodyset/femur_l</socket_parent_frame>
          <socket_child_frame>/bodyset/patella_l</socket_child_frame>
          <coordinates><Coordinate name="knee_angle_l_beta"/></coordinates>
        </PinJoint>
      </objects>
    </JointSet>
    <ConstraintSet>
      <objects>
        <CoordinateCouplerConstraint name="knee_angle_r_con">
          <dependent_coordinate_name>knee_angle_r_beta</dependent_coordinate_name>
        </CoordinateCouplerConstraint>
        <CoordinateCouplerConstraint name="knee_angle_l_con">
          <dependent_coordinate_name>knee_angle_l_beta</dependent_coordinate_name>
        </CoordinateCouplerConstraint>
      </objects>
    </ConstraintSet>
    <MarkerSet>
      <objects>
{markers}
      </objects>
    </MarkerSet>
  </Model>
</OpenSimDocument>
"""

_MARKER_ELEMENT = """        <Marker name="{name}">
          <socket_parent_frame>/bodyset/{body}</socket_parent_frame>
          <location>{location}</location>
        </Marker>"""


def mini_osim(stations: dict[str, tuple[str, tuple[float, float, float]]]) -> str:
    elements = "\n".join(
        _MARKER_ELEMENT.format(
            name=name, body=body, location=" ".join(repr(float(v)) for v in location)
        )
        for name, (body, location) in stations.items()
    )
    return _MINI_OSIM.format(markers=elements)


def pelvis_rotation_at(times_s: np.ndarray) -> Rotation:
    """A pose that turns about all three axes, so the cluster fit is not degenerate."""
    angles = np.column_stack(
        [0.05 + 0.4 * times_s, 0.3 * times_s, 0.1 * np.sin(7.0 * times_s)]
    )
    return Rotation.from_euler("xyz", angles)


def pelvis_origin_at(times_s: np.ndarray) -> np.ndarray:
    """The pelvis body origin in metres -- the quantity the frozen coordinates lost."""
    return np.column_stack(
        [
            0.40 + 0.60 * times_s,
            0.95 + 0.03 * np.sin(4.0 * times_s),
            -0.25 + 0.20 * times_s,
        ]
    )


def write_mot(path: Path, *, times_s: np.ndarray, values: dict[str, float]) -> Path:
    """A `.mot` with GAITEX's header, declaring degrees the way every published trial does."""
    names = ["time", *values]
    lines = [
        "inDegrees=yes",
        f"name={path.stem}",
        "DataType=double",
        "version=3",
        "OpenSimVersion=4.5.2-2025-05-03-6a4c6ec41",
        "endheader",
        "\t".join(names),
    ]
    for time in times_s:
        row = [repr(float(time)), *(repr(float(v)) for v in values.values())]
        lines.append("\t".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_trc(
    path: Path, *, times_s: np.ndarray, positions_m: np.ndarray, marker_names: list[str]
) -> Path:
    """The archive's OpenSim-format `.trc`: five header lines, millimetres, tab separated."""
    name_row = ["Frame#", "Time"]
    axis_row = ["", ""]
    for slot, name in enumerate(marker_names, start=1):
        name_row.extend([name, "", ""])
        axis_row.extend([f"X{slot}", f"Y{slot}", f"Z{slot}"])
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
        "OrigDataStartFrame\tOrigNumFrames",
        f"100\t100\t{times_s.size}\t{len(marker_names)}\tmm\t100\t1\t{times_s.size}",
        "\t".join(name_row),
        "\t".join(axis_row),
    ]
    for frame, time in enumerate(times_s):
        row = [str(frame), f"{time:.6f}"]
        row.extend(f"{value:.6f}" for value in positions_m[frame].reshape(-1) * 1000.0)
        lines.append("\t".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_trial(
    root: Path,
    subject: str = "kaya",
    condition: str = "gwo",
    *,
    occluded_times_s: tuple[float, ...] = (),
    omit_markers: tuple[str, ...] = (),
) -> gx.GaitexFile:
    """One synthetic trial, laid out where `GaitexTrial` looks for its four files."""
    directory = root / subject / condition
    stem = f"{subject}_{condition}"

    marker_times = np.arange(SYNTHETIC_FRAMES + 1) * DT_S
    coordinate_times = np.arange(1, SYNTHETIC_FRAMES + 1) * DT_S

    stations = {name: ("pelvis", value) for name, value in PELVIS_LOCAL_M.items()}
    stations.update({name: ("tibia_r", value) for name, value in TIBIA_LOCAL_M.items()})
    names = [name for name in stations if name not in omit_markers]

    local = np.asarray([stations[name][1] for name in names])
    rotation = pelvis_rotation_at(marker_times).as_matrix()
    observed = np.einsum("fij,mj->fmi", rotation, local) + pelvis_origin_at(marker_times)[
        :, None, :
    ]
    for time in occluded_times_s:
        frame = int(round(time / DT_S))
        for slot, name in enumerate(names):
            if name in ("R_IAS", "L_IAS"):
                observed[frame, slot] = 0.0  # the archive's occlusion sentinel

    (directory / "ik_imus" / "models").mkdir(parents=True, exist_ok=True)
    (directory / "ik_imus" / "models" / f"scaled_model_{stem}.osim").write_text(
        mini_osim(stations), encoding="utf-8"
    )
    write_mot(
        directory / "ik_imus" / "results_imu_ik"
        / f"ik_segment_registered_imu_data_{stem}.mot",
        times_s=coordinate_times,
        values=SYNTHETIC_COORDINATES,
    )
    write_trc(
        directory / "ik_imus" / f"marker_data_osim_format_{stem}.trc",
        times_s=marker_times,
        positions_m=observed,
        marker_names=names,
    )
    (directory / f"metadata_{stem}.json").write_text(
        json.dumps({"inverse_kinematics": {"pelvis": {}, "tibia_r": {}}}), encoding="utf-8"
    )
    return gx.open_trial(root, subject, condition)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> gx.GaitexFile:
    return build_trial(tmp_path_factory.mktemp("gaitex"))


@pytest.fixture(scope="module")
def corpus_trial() -> gx.GaitexFile:
    return gx.open_trial(CORPUS_ROOT, "austra", "gwo")


@pytest.fixture(scope="module")
def corpus_frames(corpus_trial: gx.GaitexFile) -> gx.GaitexFrames:
    return gx.read_frames(corpus_trial)


def model_stations(osim_text: str) -> dict[str, tuple[str, np.ndarray]]:
    """Every station the model declares, as (body, location in the body frame)."""
    out: dict[str, tuple[str, np.ndarray]] = {}
    for marker in ET.fromstring(osim_text).iter("Marker"):
        body = (marker.findtext("socket_parent_frame") or "").rsplit("/", 1)[-1]
        location = marker.findtext("location") or ""
        out[marker.get("name") or ""] = (
            body,
            np.asarray([float(v) for v in location.split()]),
        )
    return out


def read_trc(path: Path) -> tuple[dict[str, int], list[str], np.ndarray]:
    """Column index by marker name, the data rows, and their timestamps.

    Parsed independently of the module under test, and deliberately so: the header ends at
    the ``X1 Y1 Z1`` line plus any blank line after it, whatever the reader assumes.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    columns = {v.strip(): i for i, v in enumerate(lines[3].split("\t")) if v.strip()}
    start = 5
    while start < len(lines) and not lines[start].strip():
        start += 1
    rows = [line for line in lines[start:] if line.strip()]
    times = np.asarray([float(line.split("\t", 2)[1]) for line in rows])
    return columns, rows, times


def observations(
    columns: dict[str, int], rows: list[str], row_indices: np.ndarray, names: list[str]
) -> np.ndarray:
    """(frames, markers, 3) in metres, with the zero-triplet sentinel already blanked."""
    out = np.empty((row_indices.size, len(names), 3))
    for slot, row_index in enumerate(row_indices):
        fields = rows[int(row_index)].split("\t")
        for marker, name in enumerate(names):
            start = columns[name]
            out[slot, marker] = (
                float(fields[start]), float(fields[start + 1]), float(fields[start + 2])
            )
    out /= 1000.0
    out[np.all(out == 0.0, axis=2)] = np.nan
    return out


def fitted_pelvis_origin(
    shape: np.ndarray,
    centroid: np.ndarray,
    columns: dict[str, int],
    rows: list[str],
    row_indices: np.ndarray,
    learned: np.ndarray,
) -> np.ndarray:
    """The pelvis body origin implied by the four stations on the given marker rows.

    Uses the learned cluster shape the reader uses, because what this helper exists to check is
    WHICH marker row the join selected -- not which shape the fit is against. Its independence
    comes from re-reading the ``.trc`` itself; making it fit a different shape would only add a
    constant offset and would stop it separating a row from its neighbour.
    """
    observed = observations(columns, rows, row_indices, list(gx.PELVIS_STATIONS))
    to_body = rb.proper_rotation_from_covariance(learned.T @ (shape - centroid))
    pose = rb.solve_pose_series(learned, observed)
    body_rotation = np.einsum("fij,kj->fik", pose.rotation, to_body)
    return pose.translation - np.einsum("fij,j->fi", body_rotation, centroid)


def nearest_rows(marker_times: np.ndarray, wanted_s: np.ndarray) -> np.ndarray:
    """Row of the marker series closest in time to each wanted timestamp.

    Brute force on purpose, so it cannot share the reader's searchsorted, clip and tie rule and
    agree with the reader however the reader is wrong. O(n*m) and obviously correct.
    """
    return np.argmin(np.abs(marker_times[None, :] - wanted_s[:, None]), axis=1)


class TestUnits:
    """`pos` is radians for rotations and metres for the three root translations."""

    def test_a_degree_declaring_mot_comes_back_in_radians(
        self, synthetic: gx.GaitexFile
    ) -> None:
        frames = gx.read_frames(synthetic)
        names = synthetic.topology.independent_coordinate_names

        assert frames.pos[0, names.index("hip_flexion_r")] == pytest.approx(np.pi / 6)
        assert frames.pos[0, names.index("knee_angle_r")] == pytest.approx(
            np.deg2rad(12.5)
        )
        assert frames.pos[0, names.index("knee_angle_l")] == pytest.approx(
            np.deg2rad(-4.0)
        )

    def test_the_root_translations_are_metres_and_are_not_turned_into_radians(
        self, synthetic: gx.GaitexFile
    ) -> None:
        """`root_frozen_value` is the file's own three numbers, carried through untouched."""
        frames = gx.read_frames(synthetic)

        np.testing.assert_allclose(
            frames.root_frozen_value,
            [SYNTHETIC_COORDINATES[name] for name in ROOT_COORDINATES],
            atol=1e-12,
        )

    def test_the_recovered_root_is_the_pelvis_origin_in_metres(
        self, synthetic: gx.GaitexFile
    ) -> None:
        frames = gx.read_frames(synthetic)
        names = synthetic.topology.independent_coordinate_names
        columns = [names.index(name) for name in ROOT_COORDINATES]

        np.testing.assert_allclose(
            frames.pos[:, columns], pelvis_origin_at(frames.timestamps_s), atol=1e-9
        )

    @NEEDS_CORPUS
    def test_a_published_trial_reads_back_the_degrees_it_declares(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        """`ik_segment_registered_imu_data_austra_gwo.mot`, row one: 8.405170533432198."""
        names = corpus_trial.topology.independent_coordinate_names
        knee = np.rad2deg(corpus_frames.pos[0, names.index("knee_angle_r")])

        assert knee == pytest.approx(8.405, abs=1e-3)

    @NEEDS_CORPUS
    def test_a_published_trials_frozen_root_is_read_as_metres(
        self, corpus_frames: gx.GaitexFrames
    ) -> None:
        # The same three numbers the `.mot` repeats on every one of its 19543 rows.
        np.testing.assert_allclose(
            corpus_frames.root_frozen_value,
            [0.73388135, 1.02619013, -0.40812849],
            atol=1e-12,
        )


class TestDependentKneeCoordinate:
    """`knee_angle_*_beta` is coupled to the knee, so it is not a column of `pos`.

    It matters twice: a coordinate that is not read cannot be degree-converted, and `pos`
    has to keep the width and order `independent_coordinate_names` promises.
    """

    def test_the_beta_coordinates_are_absent_from_the_independent_names(
        self, synthetic: gx.GaitexFile
    ) -> None:
        names = synthetic.topology.independent_coordinate_names

        assert "knee_angle_r_beta" not in names
        assert "knee_angle_l_beta" not in names
        assert set(synthetic.topology.dependent_coordinates) == {
            "knee_angle_r_beta",
            "knee_angle_l_beta",
        }

    def test_the_mot_does_carry_the_beta_columns_that_pos_leaves_out(
        self, synthetic: gx.GaitexFile
    ) -> None:
        """Absence from `pos` is a decision about coupling, not a missing column."""
        frames = gx.read_frames(synthetic)
        names = synthetic.topology.independent_coordinate_names
        header = synthetic.trial.coordinates_path.read_text(encoding="utf-8").splitlines()
        columns = header[header.index("endheader") + 1].split()

        assert "knee_angle_r_beta" in columns
        assert "knee_angle_l_beta" in columns
        assert frames.pos.shape[1] == len(names)

    @NEEDS_CORPUS
    def test_a_published_model_couples_both_knees(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        names = corpus_trial.topology.independent_coordinate_names

        assert "knee_angle_r_beta" not in names
        assert "knee_angle_l_beta" not in names
        assert corpus_frames.pos.shape[1] == len(names)


class TestRootIsRecovered:
    @NEEDS_CORPUS
    def test_the_recovered_pelvis_height_moves_over_the_trial(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        """The frozen coordinate has a range of exactly zero; a real pelvis does not."""
        names = corpus_trial.topology.independent_coordinate_names
        height = corpus_frames.pos[corpus_frames.pelvis_solved, names.index("pelvis_ty")]

        assert np.ptp(height) > 0.020

    @NEEDS_CORPUS
    def test_the_frozen_value_is_still_reported_alongside_the_recovery(
        self, corpus_frames: gx.GaitexFrames
    ) -> None:
        """Recovering the root does not throw away what the source actually published."""
        assert corpus_frames.root_frozen_value.shape == (3,)
        assert np.isfinite(corpus_frames.root_frozen_value).all()


class TestRecoveryIsCorrect:
    """Forward kinematics, driven twice, judged against markers neither run touched.

    The pelvis stations are excluded because the recovery was fitted to them. What is left
    are stations on the other bodies the trial's IK drove: if the root were wrong, every one
    of them would be displaced by the same error, and that is what the comparison measures.
    """

    @staticmethod
    def median_station_error_m(
        source: gx.GaitexFile,
        frames: gx.GaitexFrames,
        witnesses: list[str],
        stations: dict[str, tuple[str, np.ndarray]],
        observed: np.ndarray,
        pos: np.ndarray,
    ) -> float:
        model = parse_kinematics(source.osim_text)
        names = source.topology.independent_coordinate_names
        placed = model.forward_batch(dict(zip(names, pos.T)))
        errors = []
        for slot, name in enumerate(witnesses):
            body, location = stations[name]
            predicted = placed.body_origin[body] + np.einsum(
                "fij,j->fi", placed.body_rotation[body], location
            )
            errors.append(np.linalg.norm(predicted - observed[:, slot], axis=1))
        return float(np.nanmedian(np.stack(errors, axis=1)))

    @NEEDS_CORPUS
    def test_a_station_the_model_declares_but_the_trc_never_observes_is_excluded(
        self, corpus_trial: gx.GaitexFile
    ) -> None:
        """`L_TAM` is why the witness set is an intersection and not the model's own list."""
        stations = model_stations(corpus_trial.osim_text)
        columns, _, _ = read_trc(corpus_trial.trial.markers_path)

        assert "L_TAM" in stations
        assert "L_TAM" not in columns

    @NEEDS_CORPUS
    def test_the_recovered_root_places_other_bodies_far_better_than_the_frozen_one(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        stations = model_stations(corpus_trial.osim_text)
        columns, rows, marker_times = read_trc(corpus_trial.trial.markers_path)
        driven = set(corpus_trial.driven_bodies) - {"pelvis"}
        witnesses = sorted(
            name for name, (body, _) in stations.items()
            if body in driven and name in columns
        )
        assert len(witnesses) >= 20

        # Every fiftieth solved frame: the comparison is a median over the trial, and the
        # answer does not move in the third digit between this and the whole 19523.
        selection = np.flatnonzero(corpus_frames.pelvis_solved)[::50]
        frames = gx.read_frames(corpus_trial, frame_indices=selection)
        matched = nearest_rows(marker_times, frames.timestamps_s)
        assert np.max(np.abs(marker_times[matched] - frames.timestamps_s)) < 1e-6
        observed = observations(columns, rows, matched, witnesses)

        names = corpus_trial.topology.independent_coordinate_names
        frozen_pos = frames.pos.copy()
        for axis, name in enumerate(ROOT_COORDINATES):
            frozen_pos[:, names.index(name)] = frames.root_frozen_value[axis]

        recovered_m = self.median_station_error_m(
            corpus_trial, frames, witnesses, stations, observed, frames.pos
        )
        frozen_m = self.median_station_error_m(
            corpus_trial, frames, witnesses, stations, observed, frozen_pos
        )

        # Measured 53.1 mm against 146.9 mm on this trial. The assertion is the factor of
        # two, not the millimetres: the residual is the model's own scaling and soft tissue,
        # which this test has no business pinning.
        assert recovered_m * 2.0 < frozen_m
        assert recovered_m < 0.100


class TestOcclusion:
    """A lost marker is an exact zero triplet, and must not be fitted as a position."""

    def test_an_occluded_pelvis_cluster_gives_an_unsolved_frame_with_a_nan_root(
        self, tmp_path: Path
    ) -> None:
        occluded = (0.10, 0.11)
        source = build_trial(tmp_path, occluded_times_s=occluded)
        frames = gx.read_frames(source)
        names = source.topology.independent_coordinate_names
        columns = [names.index(name) for name in ROOT_COORDINATES]

        hit = np.isin(np.round(frames.timestamps_s, 6), occluded)
        assert hit.sum() == len(occluded)
        assert not frames.pelvis_solved[hit].any()
        assert np.isnan(frames.pos[hit][:, columns]).all()

    def test_no_frame_survives_as_solved_with_the_sentinel_fitted_into_its_pose(
        self, tmp_path: Path
    ) -> None:
        """Fitting two lost hips as points at the origin drags the pelvis off by centimetres.

        The check is that every frame still marked solved sits on the pose this fixture
        built, so a frame rescued by averaging in the sentinel cannot pass as one of them.
        """
        source = build_trial(tmp_path, occluded_times_s=(0.10, 0.11))
        frames = gx.read_frames(source)
        names = source.topology.independent_coordinate_names
        columns = [names.index(name) for name in ROOT_COORDINATES]

        solved = frames.pelvis_solved
        np.testing.assert_allclose(
            frames.pos[solved][:, columns],
            pelvis_origin_at(frames.timestamps_s[solved]),
            atol=1e-9,
        )

    def test_every_frame_but_the_occluded_ones_still_solves(self, tmp_path: Path) -> None:
        source = build_trial(tmp_path, occluded_times_s=(0.10, 0.11))
        frames = gx.read_frames(source)

        assert frames.pelvis_solved.sum() == SYNTHETIC_FRAMES - 2

    @NEEDS_CORPUS
    def test_a_published_trial_withdraws_exactly_its_underdetermined_frames(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        """Three visible stations still determine the pelvis; fewer than three do not."""
        columns, rows, marker_times = read_trc(corpus_trial.trial.markers_path)
        every = np.arange(len(rows))
        cluster = observations(columns, rows, every, list(gx.PELVIS_STATIONS))
        visible = np.isfinite(cluster).all(axis=2).sum(axis=1)

        underdetermined = set(np.round(marker_times[visible < 3], 6))
        unsolved = set(
            np.round(corpus_frames.timestamps_s[~corpus_frames.pelvis_solved], 6)
        )
        partly_occluded = np.round(marker_times[(visible == 3)], 6)

        assert underdetermined
        assert underdetermined <= unsolved
        # A single lost station is not a lost frame: those still solve from the other three.
        assert not underdetermined & set(partly_occluded)
        assert corpus_frames.pelvis_solved[
            np.isin(np.round(corpus_frames.timestamps_s, 6), partly_occluded)
        ].all()

    @NEEDS_CORPUS
    def test_the_unsolved_frames_carry_nan_rather_than_a_stale_root(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        names = corpus_trial.topology.independent_coordinate_names
        columns = [names.index(name) for name in ROOT_COORDINATES]

        assert np.isnan(corpus_frames.pos[~corpus_frames.pelvis_solved][:, columns]).all()
        assert np.isfinite(
            corpus_frames.pos[corpus_frames.pelvis_solved][:, columns]
        ).all()


class TestMissingStations:
    """A trial whose markers cannot determine the pelvis is refused, by its own exception."""

    def test_a_trc_missing_one_station_raises_missing_pelvis_stations(
        self, tmp_path: Path
    ) -> None:
        source = build_trial(tmp_path, omit_markers=("R_IPS",))

        with pytest.raises(gx.MissingPelvisStations, match="R_IPS"):
            gx.read_frames(source)

    def test_the_refusal_is_a_data_condition_not_a_general_reader_error(
        self, tmp_path: Path
    ) -> None:
        """Callers separate the two, so the subclass relation is part of the contract."""
        assert issubclass(gx.MissingPelvisStations, gx.GaitexReaderError)
        assert gx.MissingPelvisStations is not gx.GaitexReaderError

    @NEEDS_CORPUS
    @pytest.mark.skipif(
        CORPUS_ROOT is None or not (CORPUS_ROOT / "jung-hee" / "gwo").is_dir(),
        reason="GAITEX jung-hee/gwo is not extracted",
    )
    def test_jung_hee_gwo_is_refused_because_its_trc_never_observes_r_ips(self) -> None:
        source = gx.open_trial(CORPUS_ROOT, "jung-hee", "gwo")
        columns, _, _ = read_trc(source.trial.markers_path)

        assert "R_IPS" not in columns
        with pytest.raises(gx.MissingPelvisStations, match="R_IPS"):
            gx.read_frames(source)


class TestTimeAlignment:
    """The two files are joined on their timestamps, never on their row numbers."""

    @NEEDS_CORPUS
    def test_the_two_files_start_together_and_end_two_samples_apart(
        self, corpus_trial: gx.GaitexFile
    ) -> None:
        """Both series begin at t=0; the markers simply run two samples longer.

        ``marker_times[0]`` is the assertion that separates a real offset from a reader that
        skipped the first row; the sizes and the end time at 195.44 alone would not.
        """
        _, _, marker_times = read_trc(corpus_trial.trial.markers_path)
        frames = gx.read_frames(corpus_trial)

        assert marker_times[0] == pytest.approx(0.0)
        assert frames.timestamps_s[0] == pytest.approx(0.0)
        assert frames.timestamps_s[-1] == pytest.approx(195.42)
        assert marker_times[-1] == pytest.approx(195.44)
        assert marker_times.size == frames.timestamps_s.size + 2

    @NEEDS_CORPUS
    def test_the_reader_keeps_every_row_the_file_says_it_has(
        self, corpus_trial: gx.GaitexFile
    ) -> None:
        """The header's ``NumFrames`` against what the reader returns.

        This is the check the source offers about itself, and the one that would have caught
        the reader starting a line too late.
        """
        lines = corpus_trial.trial.markers_path.read_text(
            encoding="utf-8", errors="replace").splitlines()
        declared = gx._declared_frame_count(lines)
        _, _, marker_times = read_trc(corpus_trial.trial.markers_path)

        assert declared is not None
        assert marker_times.size == declared

    @NEEDS_CORPUS
    def test_a_coordinate_frame_the_markers_never_observed_stays_unsolved(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        """The join refuses a coordinate time no marker frame is near.

        On this corpus the two grids are aligned, so no real frame exercises the refusal --
        which is exactly why it is asked of a coordinate time past the end of the marker
        series rather than of frame 0. Frame 0 is now solved, and should be: both files
        start at t=0.
        """
        assert corpus_frames.timestamps_s[0] == pytest.approx(0.0)
        assert corpus_frames.pelvis_solved[0]

        _, _, marker_times = read_trc(corpus_trial.trial.markers_path)
        beyond = np.array([marker_times[-1] + 1.0])
        origin, solved, _, _ = gx._pelvis_origin_track(
            corpus_trial.trial, corpus_trial.osim_text, beyond)
        assert not solved[0]
        assert np.isnan(origin[0]).all()

    @NEEDS_CORPUS
    def test_the_recovered_root_is_the_cluster_seen_at_that_same_timestamp(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        """Fitting the neighbouring marker row instead would shift the whole track.

        The pelvis moves about 3 mm in a sample, so the two candidates are far apart
        compared with the fit itself, which is why this separates them.
        """
        stations = model_stations(corpus_trial.osim_text)
        shape = np.stack([stations[name][1] for name in gx.PELVIS_STATIONS])
        centroid = shape.mean(axis=0)
        columns, rows, marker_times = read_trc(corpus_trial.trial.markers_path)

        selection = np.flatnonzero(corpus_frames.pelvis_solved)[::500]
        frames = gx.read_frames(corpus_trial, frame_indices=selection)
        names = corpus_trial.topology.independent_coordinate_names
        root = frames.pos[:, [names.index(name) for name in ROOT_COORDINATES]]

        matched = nearest_rows(marker_times, frames.timestamps_s)
        every_row = np.arange(len(rows))
        learned = rb.mean_local_shape(
            observations(columns, rows, every_row, list(gx.PELVIS_STATIONS)))
        at_the_same_time = fitted_pelvis_origin(
            shape, centroid, columns, rows, matched, learned)
        one_sample_later = fitted_pelvis_origin(
            shape, centroid, columns, rows, matched + 1, learned)

        np.testing.assert_allclose(root, at_the_same_time, atol=1e-9)
        # And the neighbour is a genuinely different answer, so the first line is not
        # satisfied by both -- a median of about 1.8 mm against a fit good to nanometres.
        assert np.median(np.linalg.norm(root - one_sample_later, axis=1)) > 1e-3


class TestTheConvertedAnglesAreAHumansAngles:
    """A units error does not raise; it produces a knee that flexes by one degree or by four
    thousand, and every downstream number stays finite. The `.mot` declares `inDegrees=yes` and
    the retarget wants radians, so the conversion is checked against physiology rather than
    against a formula -- which is the only check that would notice it going missing.
    """

    # Generous on purpose. This is a units guard, not a gait laboratory.
    PLAUSIBLE_RANGE_DEG = {
        "knee_angle_r": (2.0, 90.0),
        "knee_angle_l": (2.0, 90.0),
        "hip_flexion_r": (5.0, 80.0),
        "ankle_angle_r": (2.0, 60.0),
    }

    @NEEDS_CORPUS
    def test_each_joint_sweeps_an_angle_a_walking_person_sweeps(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        names = corpus_trial.topology.independent_coordinate_names
        solved = corpus_frames.pelvis_solved
        for coordinate, (low, high) in self.PLAUSIBLE_RANGE_DEG.items():
            if coordinate not in names:
                continue
            degrees = np.rad2deg(corpus_frames.pos[solved, names.index(coordinate)])
            assert low <= float(np.ptp(degrees)) <= high, coordinate

    @NEEDS_CORPUS
    def test_the_recovered_pelvis_stands_at_a_human_height(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        """The root is metres and must not have been swept up in the degree conversion."""
        names = corpus_trial.topology.independent_coordinate_names
        height = corpus_frames.pos[corpus_frames.pelvis_solved, names.index("pelvis_ty")]
        assert 0.6 < float(np.median(height)) < 1.3


class TestTheFitSaysHowWellItFitted:
    """`solved` is not a quality claim, and for a while nothing else was carried.

    The reader fits a shape learned from the observed cluster rather than the model's declared
    stations, which are a left-right symmetric template incongruent with any real pelvis. The
    residual is the evidence that the learned shape is the right call, and it is now on the
    frames rather than discarded inside the solver.
    """

    @NEEDS_CORPUS
    def test_the_residual_is_millimetres_not_centimetres(
        self, corpus_frames: gx.GaitexFrames
    ) -> None:
        """Fitting the declared template leaves about 37 mm on this corpus; the learned shape
        leaves about 2. The bound here is loose enough to survive a different trial and tight
        enough that a return to the template fails it."""
        solved = corpus_frames.pelvis_solved
        residual = corpus_frames.pelvis_residual_m[solved]
        assert np.isfinite(residual).all()
        assert float(np.median(residual)) < 0.010

    @NEEDS_CORPUS
    def test_an_unsolved_frame_carries_no_residual_to_mistake_for_a_good_fit(
        self, corpus_frames: gx.GaitexFrames
    ) -> None:
        """Zero would read as a perfect fit, which is exactly backwards."""
        unsolved = ~corpus_frames.pelvis_solved
        if unsolved.any():
            assert np.isnan(corpus_frames.pelvis_residual_m[unsolved]).all()
            assert (corpus_frames.pelvis_visible_count[unsolved] == 0).all()

    @NEEDS_CORPUS
    def test_a_three_marker_residual_is_smaller_without_being_better(
        self, corpus_frames: gx.GaitexFrames
    ) -> None:
        """Why the visible count has to travel beside the residual.

        A rigid fit of three points has nine coordinates against six pose degrees of freedom, so
        three constraints remain -- the triangle's side lengths -- and the residual does measure
        those. It is not identically zero.

        What is true is subtler and matters more: the three-marker residual is systematically
        SMALLER than the four-marker one, not because those frames fitted better but because
        there is less for them to disagree with. So a residual cannot be compared across frames
        with different counts, and reading one without the other ranks the weaker evidence
        higher.
        """
        solved = corpus_frames.pelvis_solved
        counts = corpus_frames.pelvis_visible_count[solved]
        residual = corpus_frames.pelvis_residual_m[solved]
        assert set(np.unique(counts)) <= {3, 4}
        if (counts == 3).any() and (counts == 4).any():
            three = float(np.median(residual[counts == 3]))
            four = float(np.median(residual[counts == 4]))
            assert three > 0.0, "a three-marker fit still measures the triangle it was given"
            assert three < four, "fewer constraints leave less to violate"

    @NEEDS_CORPUS
    def test_the_learned_shape_does_not_move_the_answer(
        self, corpus_trial: gx.GaitexFile, corpus_frames: gx.GaitexFrames
    ) -> None:
        """The point of the learned shape is to remove an artifact, not to change the root.

        Fitting the declared template gives a track that agrees with this one to well under a
        millimetre in the median. What differs is the step at each change of visible subset,
        which the template's shape error turns into spurious acceleration.
        """
        stations = model_stations(corpus_trial.osim_text)
        template = np.stack([stations[name][1] for name in gx.PELVIS_STATIONS])
        centroid = template.mean(axis=0)
        observed, _ = gx._read_trc_cluster(
            corpus_trial.trial.markers_path, gx.PELVIS_STATIONS)

        declared = rb.solve_pose_series(template - centroid, observed)
        by_template = declared.translation - np.einsum(
            "fij,j->fi", declared.rotation, centroid)
        by_template[~declared.solved] = np.nan

        names = corpus_trial.topology.independent_coordinate_names
        shipped = corpus_frames.pos[:, [names.index(n) for n in ROOT_COORDINATES]]
        # the two series are on different grids; compare where both are finite and aligned
        common = min(by_template.shape[0], shipped.shape[0])
        difference = np.linalg.norm(by_template[:common] - shipped[:common], axis=1)
        assert float(np.nanmedian(difference)) < 0.005
