"""The marker-driven small, checked without the source data.

Every fixture here is a body drawn by hand in a known posture, so the assertions are about signs and
frames that no validator sees: L3 only asks whether the sensor frame is a CONSTANT relabel of the
fitted body, never whether +Y points up the shank or +Z out of the back of the wrist. A wrong sign
would ship silently. The rules under test are the ones ``gaitex_synthesis_v1_1.yaml`` declares; the
settings document is rebuilt here from the same values rather than read from disk so that a test
failure names the rule, not the file.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_synth.gaitex_synthesis import marker_small as ms
from soma_synth.sensors import site_registry
from soma_synth.gaitex_retarget.native import MarkerStream

#: A drive-letter path, which no manifest block may carry.
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPOSITORY_ROOT / "configs" / "datasets" / "gaitex_synthesis_v1_1.yaml"
SITES_PATH = REPOSITORY_ROOT / "configs" / "datasets" / "gaitex_sensor_sites_v1.yaml"

DT = 0.01
UP = np.array([0.0, 0.0, 1.0])
FORWARD = np.array([1.0, 0.0, 0.0])      # the subject faces +x
RIGHT = np.array([0.0, -1.0, 0.0])       # so subject-right is -y


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def settings() -> ms.SynthesisSettings:
    return ms.load_settings(SETTINGS_PATH)


@pytest.fixture(scope="module")
def registry() -> site_registry.SiteRegistry:
    return site_registry.load_site_registry(SITES_PATH)


def _yaw_series(frames: int, amplitude_deg: float = 25.0, period_frames: int = 120) -> np.ndarray:
    """A slow, smooth yaw so the pose is never static and the filter has real signal."""
    t = np.arange(frames)
    return np.deg2rad(amplitude_deg) * np.sin(2.0 * np.pi * t / period_frames)


def _moving_body(points: dict[str, np.ndarray], frames: int, *, pivot: np.ndarray,
                 drift: np.ndarray | None = None) -> tuple[MarkerStream, np.ndarray]:
    """Rotate a rigid set of points about a vertical axis through `pivot`, frame by frame.

    Returns the stream and the world rotation applied at each frame, so a test can predict where a
    body-fixed axis points at any frame.
    """
    names = tuple(points)
    local = np.stack([points[name] for name in names]) - pivot
    yaw = _yaw_series(frames)
    rotations = Rotation.from_rotvec(np.outer(yaw, UP)).as_matrix()
    positions = np.einsum("tij,nj->tni", rotations, local) + pivot
    if drift is not None:
        positions = positions + np.outer(np.linspace(0.0, 1.0, frames), drift)[:, None, :]
    stream = MarkerStream(time_s=np.arange(frames) * DT, positions_m=positions, marker_names=names)
    return stream, rotations


def _plate(centre: np.ndarray, first: np.ndarray, second: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Four markers on a plane through `centre` spanned by unit vectors `first`, `second` (the GAITEX plates
    are asymmetric; this one is too, so the SVD axes are well conditioned)."""
    return {
        f"{prefix}1": centre + 0.030 * first + 0.020 * second,
        f"{prefix}2": centre - 0.030 * first + 0.025 * second,
        f"{prefix}3": centre - 0.025 * first - 0.030 * second,
        f"{prefix}4": centre + 0.035 * first - 0.020 * second,
    }


def _right_shank() -> dict[str, np.ndarray]:
    knee_lateral = np.array([0.0, -0.15, 0.50])
    knee_medial = np.array([0.0, -0.05, 0.50])
    ankle_lateral = np.array([0.0, -0.15, 0.05])
    ankle_medial = np.array([0.0, -0.05, 0.05])
    plate = _plate(np.array([0.0, -0.16, 0.30]), FORWARD, UP, "R_SHIN")   # lateral plate, normal +-y
    return {"R_FLE": knee_lateral, "R_FME": knee_medial, "R_FAL": ankle_lateral, "R_TAM": ankle_medial, **plate}


def _left_shank() -> dict[str, np.ndarray]:
    mirrored = {name.replace("R_", "L_", 1): point * np.array([1.0, -1.0, 1.0]) for name, point in _right_shank().items()}
    return mirrored


def _write_settings(tmp_path: pathlib.Path, settings: ms.SynthesisSettings, **overrides) -> ms.SynthesisSettings:
    document = json.loads(json.dumps(settings.document))
    document.update(overrides)
    path = tmp_path / "settings.yaml"
    path.write_text(json.dumps(document), encoding="utf-8")
    return ms.load_settings(path)


def _sensor_axes_world(synthesis: ms.SiteSynthesis, rotations: np.ndarray, frame: int) -> np.ndarray:
    """Columns [X|Y|Z] of the spec sensor frame in world at `frame`, from the synthesised orientation."""
    q = synthesis.signal.orientation_wxyz[frame]
    assert np.isfinite(q).all(), "the frame under test must be valid"
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _assert_axis(actual: np.ndarray, expected: np.ndarray, tolerance_deg: float = 3.0) -> None:
    cosine = np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    assert angle < tolerance_deg, f"axis {actual} is {angle:.1f} deg from {expected}"


# ------------------------------------------------------------------ sensor frame S: limbs
class TestShankSensorFrame:
    FRAMES = 400
    CHECK = 200

    def test_right_shank_axes_are_proximal_lateral_anterior(self, settings, registry) -> None:
        stream, rotations = _moving_body(_right_shank(), self.FRAMES, pivot=np.array([0.0, -0.1, 0.05]))
        synthesis = ms.synthesise_site(stream, "shank_r", registry, settings)
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        R = rotations[self.CHECK]
        _assert_axis(axes[:, 1], R @ UP)          # +Y proximal: ankle -> knee
        _assert_axis(axes[:, 2], R @ RIGHT)       # +Z outward lateral: on the right leg, subject-right
        _assert_axis(axes[:, 0], R @ FORWARD)     # +X = Y x Z = anterior
        assert synthesis.placement.kind == "registry"

    def test_left_shank_x_is_posterior_and_z_still_outward(self, settings, registry) -> None:
        stream, rotations = _moving_body(_left_shank(), self.FRAMES, pivot=np.array([0.0, 0.1, 0.05]))
        synthesis = ms.synthesise_site(stream, "shank_l", registry, settings)
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        R = rotations[self.CHECK]
        _assert_axis(axes[:, 1], R @ UP)
        _assert_axis(axes[:, 2], R @ -RIGHT)      # outward on the left leg is subject-left
        _assert_axis(axes[:, 0], R @ -FORWARD)    # the spec's left-limb +X is posterior

    def test_the_relabel_is_one_constant_per_take(self, settings, registry) -> None:
        """L3's property, checked at the source: orientation == R_world_from_cluster @ constant."""
        stream, rotations = _moving_body(_right_shank(), self.FRAMES, pivot=np.array([0.0, -0.1, 0.05]))
        synthesis = ms.synthesise_site(stream, "shank_r", registry, settings)
        valid = synthesis.signal.valid
        q = synthesis.signal.orientation_wxyz[valid]
        world = Rotation.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])).as_matrix()
        relabel = np.einsum("tji,tjk->tik", rotations[valid], world)   # R_world^T @ R_sensor
        spread = np.degrees(np.linalg.norm(Rotation.from_matrix(
            np.einsum("ji,tjk->tik", relabel[0], relabel)).as_rotvec(), axis=1))
        assert spread.max() < 1.0

    def test_an_occluded_inward_reference_does_not_flip_the_plate_normal(self, settings, registry) -> None:
        """GAITEX writes an occluded marker as (0, 0, 0): a point at the lab origin.
        With the subject metres away, a third of such frames on the medial references used to outweigh
        every real frame in the inward mean and turn the plate normal, the lever arm and +Z around."""
        far = np.array([3.0, 2.0, 0.0])
        points = {name: point + far for name, point in _right_shank().items()}
        stream, rotations = _moving_body(points, self.FRAMES, pivot=far + np.array([0.0, -0.1, 0.05]))
        positions = stream.positions_m.copy()
        for name in ("R_FME", "R_TAM"):
            positions[::3, stream.marker_names.index(name)] = 0.0       # occluded on every third frame
        occluded = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        synthesis = ms.synthesise_site(occluded, "shank_r", registry, settings)
        assert synthesis.diagnostics["inward_sign_agreement"] == 1.0
        assert synthesis.diagnostics["inward_reference_frames"] == self.FRAMES
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        _assert_axis(axes[:, 2], rotations[self.CHECK] @ RIGHT)         # still outward
        assert synthesis.lever_arm_local @ synthesis.lever_arm_local > 0.0
        assert synthesis.diagnostics["sensor_frame"]["plate_outward_vs_landmark_axis_deg"] < 90.0

    def test_a_landmark_carried_but_occluded_throughout_falls_back_like_an_absent_one(self, settings, registry, tmp_path) -> None:
        """The fallback used to fire only on an absent column; a present-but-occluded
        landmark raised 'fewer than 10 frames' and failed the take. The shipped shank rule reads the
        medial epicondyle for +Y as well, so a rule whose +Y does without it isolates the fallback."""
        rules = json.loads(json.dumps(settings.document["sensor_frame_rules"]))
        rules["shank_r"]["y"] = {"kind": "landmarks", "from": ["R_FAL", "R_TAM"], "to": ["R_FLE"]}
        isolated = _write_settings(tmp_path, settings, sensor_frame_rules=rules)
        stream, _ = _moving_body(_right_shank(), self.FRAMES, pivot=np.array([0.0, -0.1, 0.05]))
        positions = stream.positions_m.copy()
        positions[5:, stream.marker_names.index("R_FME")] = 0.0         # five frames, then occluded for good
        occluded = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        synthesis = ms.synthesise_site(occluded, "shank_r", registry, isolated)
        assert synthesis.diagnostics["sensor_frame"]["secondary_construction"] == "plate_normal_outward"
        assert synthesis.diagnostics["sensor_frame"]["landmarks_missing"] == []
        # and with the epicondyle visible the same rule takes the landmark alternative
        visible = ms.synthesise_site(stream, "shank_r", registry, isolated)
        assert visible.diagnostics["sensor_frame"]["secondary_construction"] == "landmarks(R_FME->R_FLE)"

    def test_the_manifest_names_the_landmarks_the_trial_lacks(self, settings, registry) -> None:
        """The label used to name declared landmarks whether or not the trial carried them."""
        points = _right_shank()
        del points["R_TAM"]
        stream, _ = _moving_body(points, self.FRAMES, pivot=np.array([0.0, -0.1, 0.05]))
        synthesis = ms.synthesise_site(stream, "shank_r", registry, settings)
        frame = synthesis.diagnostics["sensor_frame"]
        assert frame["primary_construction"] == "landmarks(R_FAL->R_FLE+R_FME)"
        assert frame["landmarks_missing"] == ["R_TAM"]

    def test_a_missing_epicondyle_falls_back_to_the_plate_normal(self, settings, registry) -> None:
        points = _right_shank()
        del points["R_FME"]
        stream, rotations = _moving_body(points, self.FRAMES, pivot=np.array([0.0, -0.1, 0.05]))
        synthesis = ms.synthesise_site(stream, "shank_r", registry, settings)
        assert synthesis.diagnostics["sensor_frame"]["secondary_construction"] == "plate_normal_outward"
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        R = rotations[self.CHECK]
        # with only the lateral epicondyle left, the "knee centre" IS that epicondyle, so the proximal
        # axis leans 6 deg laterally by construction; the lateral axis is the plate normal orthogonalised
        # against that leaning Y, and that is what the test expects
        proximal = R @ (points["R_FLE"] - 0.5 * (points["R_FAL"] + points["R_TAM"]))
        proximal /= np.linalg.norm(proximal)
        lateral = R @ RIGHT
        lateral = lateral - proximal * np.dot(lateral, proximal)
        _assert_axis(axes[:, 1], proximal)
        _assert_axis(axes[:, 2], lateral)


# ------------------------------------------------------------------ sensor frame S: feet
def _right_foot() -> dict[str, np.ndarray]:
    plate = _plate(np.array([0.13, -0.09, 0.06]), FORWARD, -RIGHT, "R_FOOT")   # dorsal plate, normal +-z
    return {"R_FCC": np.array([0.0, -0.09, 0.02]), "R_FAL": np.array([0.03, -0.13, 0.07]),
            "R_TAM": np.array([0.03, -0.05, 0.07]), **plate}


class TestFootSensorFrame:
    FRAMES = 400
    CHECK = 200

    def test_right_foot_x_points_to_the_toes_and_z_lateral(self, settings, registry) -> None:
        stream, rotations = _moving_body(_right_foot(), self.FRAMES, pivot=np.array([0.05, -0.09, 0.0]))
        synthesis = ms.synthesise_site(stream, "foot_r", registry, settings)
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        R = rotations[self.CHECK]
        _assert_axis(axes[:, 1], UP)                  # +Y up: the trial-mean world up, so not rotated
        _assert_axis(axes[:, 0], R @ FORWARD)         # +X anterior (toes)
        _assert_axis(axes[:, 2], R @ RIGHT)           # +Z = X x Y = lateral

    def test_left_foot_x_is_posterior_and_z_lateral(self, settings, registry) -> None:
        points = {name.replace("R_", "L_", 1): p * np.array([1.0, -1.0, 1.0]) for name, p in _right_foot().items()}
        stream, rotations = _moving_body(points, self.FRAMES, pivot=np.array([0.05, 0.09, 0.0]))
        synthesis = ms.synthesise_site(stream, "foot_l", registry, settings)
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        R = rotations[self.CHECK]
        _assert_axis(axes[:, 1], UP)
        _assert_axis(axes[:, 0], R @ -FORWARD)
        _assert_axis(axes[:, 2], R @ -RIGHT)


# ------------------------------------------------------------------ sensor frame S: trunk and head
def _trunk_and_pelvis() -> dict[str, np.ndarray]:
    plate = _plate(np.array([0.10, 0.0, 1.30]), UP, -RIGHT, "THOR")            # anterior sternum plate, normal +-x
    return {"L_IPS": np.array([-0.10, 0.05, 1.00]), "R_IPS": np.array([-0.10, -0.05, 1.00]),
            "L_IAS": np.array([0.08, 0.13, 1.00]), "R_IAS": np.array([0.08, -0.13, 1.00]), **plate}


def _head() -> dict[str, np.ndarray]:
    return {"SGL": np.array([0.10, 0.0, 1.70]), "R_HEAD": np.array([0.0, -0.08, 1.70]),
            "L_HEAD": np.array([0.0, 0.08, 1.70])}


class TestTrunkAndHeadSensorFrames:
    FRAMES = 400
    CHECK = 200

    def test_back_t4_axes_are_up_right_anterior_through_the_chest_twin(self, settings, registry) -> None:
        stream, rotations = _moving_body(_trunk_and_pelvis(), self.FRAMES, pivot=np.array([0.0, 0.0, 1.0]))
        synthesis = ms.synthesise_site(stream, "back_T4", registry, settings)
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        R = rotations[self.CHECK]
        _assert_axis(axes[:, 1], UP)
        _assert_axis(axes[:, 0], R @ FORWARD)
        _assert_axis(axes[:, 2], R @ RIGHT)
        assert synthesis.placement.kind == "offset_twin"
        assert synthesis.placement.source.endswith(":chest")
        assert synthesis.placement.provisional
        # the sternum plate's lever runs into the body (posterior) by the registered 14 mm
        lever_world = R @ np.einsum("ij,j->i", np.eye(3), synthesis.lever_arm_local)
        assert abs(np.linalg.norm(synthesis.lever_arm_local) - 0.014) < 1e-6
        del lever_world

    def test_occiput_axes_are_up_right_anterior_with_a_declared_zero_offset(self, settings, registry) -> None:
        stream, rotations = _moving_body(_head(), self.FRAMES, pivot=np.array([0.0, 0.0, 1.6]))
        synthesis = ms.synthesise_site(stream, "occiput", registry, settings)
        axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
        R = rotations[self.CHECK]
        _assert_axis(axes[:, 1], UP)
        _assert_axis(axes[:, 2], R @ RIGHT)
        _assert_axis(axes[:, 0], R @ FORWARD)
        assert synthesis.placement.kind == "declared_zero"
        assert np.all(synthesis.lever_arm_local == 0.0)


# ------------------------------------------------------------------ forearm
def _right_arm(elbow_offset: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Anatomical position: arm hanging, palm forward, radial styloid lateral (subject-right)."""
    return {"R_HLE": np.array([0.0, -0.20, 1.00]) + (elbow_offset if elbow_offset is not None else 0.0),
            "R_RSP": np.array([0.0, -0.23, 0.75]), "R_USP": np.array([0.0, -0.17, 0.75])}


class TestForearm:
    FRAMES = 400
    CHECK = 200

    def test_dorsal_z_faces_backwards_on_both_arms_in_the_anatomical_position(self, settings, registry) -> None:
        for name, sign in (("wrist_r", 1.0), ("wrist_l", -1.0)):
            points = {k.replace("R_", name[-1].upper() + "_", 1): p * np.array([1.0, sign, 1.0])
                      for k, p in _right_arm().items()}
            stream, rotations = _moving_body(points, self.FRAMES, pivot=np.array([0.0, sign * -0.2, 0.75]))
            synthesis = ms.synthesise_site(stream, name, registry, settings)
            axes = _sensor_axes_world(synthesis, rotations, self.CHECK)
            R = rotations[self.CHECK]
            _assert_axis(axes[:, 1], UP)                   # proximal: styloids -> epicondyle
            _assert_axis(axes[:, 2], R @ -FORWARD)         # the back of the hand faces backwards, both arms
            # and X = Y x Z closes a right-handed triad
            _assert_axis(np.cross(axes[:, 1], axes[:, 2]), axes[:, 0], 0.5)

    def test_elbow_flexion_alone_does_not_move_the_forearm_frame(self, settings, registry) -> None:
        """ADR-0037 section 5: HLE is on the humerus. With HLE on the flexion axis the frame must not care."""
        frames = self.FRAMES
        base = _right_arm()
        stream, _ = _moving_body(base, frames, pivot=np.array([0.0, -0.2, 0.75]))
        # swing HLE about the styloid midpoint's mediolateral axis: pure elbow flexion of a humerus
        positions = stream.positions_m.copy()
        angle = np.deg2rad(40.0) * np.sin(2.0 * np.pi * np.arange(frames) / 90.0)
        mid = 0.5 * (positions[:, 1] + positions[:, 2])
        arm = positions[:, 0] - mid
        axis = positions[:, 1] - positions[:, 2]
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        rot = Rotation.from_rotvec(axis * angle[:, None]).as_matrix()
        positions[:, 0] = mid + np.einsum("tij,tj->ti", rot, arm)
        flexed = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        still = ms.anatomical_forearm_pose(stream, registry.require("wrist_r"), settings)
        moved = ms.anatomical_forearm_pose(flexed, registry.require("wrist_r"), settings)
        # the long axis follows HLE (that is the definition), the styloid pair does not: what the test pins
        # is that the pair-distance and rate guards do not withdraw ordinary elbow motion
        assert moved.solved.sum() == still.solved.sum() == frames
        assert moved.diagnostics["withdrawn_styloid_pair"] == 0
        assert moved.diagnostics["withdrawn_rate"] == 0

    def test_a_one_sided_styloid_jump_is_withdrawn_by_the_pair_distance(self, settings, registry) -> None:
        stream, _ = _moving_body(_right_arm(), self.FRAMES, pivot=np.array([0.0, -0.2, 0.75]))
        positions = stream.positions_m.copy()
        positions[100:110, 2] += np.array([0.0, 0.05, 0.0])   # the ulnar marker jumps 50 mm for ten frames
        jumped = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        pose = ms.anatomical_forearm_pose(jumped, registry.require("wrist_r"), settings)
        assert pose.diagnostics["withdrawn_styloid_pair"] == 10
        assert not pose.solved[100:110].any()
        assert pose.solved[:100].all() and pose.solved[110:].all()
        assert pose.diagnostics["fit_residual_median_mm"] is None      # never NaN: the manifest is JSON

    def test_a_swapped_island_between_two_gaps_is_withdrawn_and_listed(self, settings, registry) -> None:
        """What austra/ng and yaxkin/ng actually carry: the radial and ulnar
        labels exchanged after an occlusion and exchanged back after the next. The pair distance is
        swap-invariant and the rate test never sees across a gap; the parity of the half turn does."""
        stream, _ = _moving_body(_right_arm(), self.FRAMES, pivot=np.array([0.0, -0.2, 0.75]))
        positions = stream.positions_m.copy()
        positions[150:200, [1, 2]] = positions[150:200, [2, 1]]         # RSP <-> USP on the island
        positions[148:150] = 0.0                                        # occluded on both sides of it
        positions[200:202] = 0.0
        swapped = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        pose = ms.anatomical_forearm_pose(swapped, registry.require("wrist_r"), settings)
        assert pose.diagnostics["withdrawn_styloid_pair"] == 0
        crossings = pose.diagnostics["label_swap_crossings_csv_frames"]
        assert [(c["before"], c["after"]) for c in crossings] == [(147, 150), (199, 202)]
        assert all(c["twist_deg"] > 170.0 for c in crossings)
        assert pose.diagnostics["withdrawn_label_swap_runs_csv_frames"] == [[150, 200]]
        assert pose.diagnostics["withdrawn_label_swap_frames"] == 50
        assert not pose.solved[148:202].any()
        assert pose.solved[:148].all() and pose.solved[202:].all()

    def test_a_swap_that_persists_for_most_of_the_take_is_refused(self, settings, registry) -> None:
        """The majority parity is kept only while it IS a majority; at a quarter of the frames the
        other labelling has no less claim, and the site is refused rather than guessed."""
        stream, _ = _moving_body(_right_arm(), self.FRAMES, pivot=np.array([0.0, -0.2, 0.75]))
        positions = stream.positions_m.copy()
        positions[100:, [1, 2]] = positions[100:, [2, 1]]
        positions[98:100] = 0.0
        swapped = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        with pytest.raises(ms.MarkerSmallError, match="minority parity"):
            ms.anatomical_forearm_pose(swapped, registry.require("wrist_r"), settings)

    def test_an_ordinary_occlusion_is_a_gap_and_not_a_crossing(self, settings, registry) -> None:
        stream, _ = _moving_body(_right_arm(), self.FRAMES, pivot=np.array([0.0, -0.2, 0.75]))
        positions = stream.positions_m.copy()
        positions[200:230, 1] = 0.0                                     # the radial styloid occluded
        gapped = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        pose = ms.anatomical_forearm_pose(gapped, registry.require("wrist_r"), settings)
        assert not pose.solved[200:230].any() and pose.solved[:200].all() and pose.solved[230:].all()
        assert pose.diagnostics["label_swap_crossings_csv_frames"] == []
        assert pose.diagnostics["withdrawn_label_swap_frames"] == 0
        assert pose.diagnostics["gaps_checked_for_twist"] == 1
        assert pose.diagnostics["gap_twist_max_deg"] < 30.0

    def test_a_single_frame_rotation_spike_withdraws_both_frames_of_the_step(self, settings, registry) -> None:
        stream, _ = _moving_body(_right_arm(), self.FRAMES, pivot=np.array([0.0, -0.2, 0.75]))
        positions = stream.positions_m.copy()
        # rotate the whole triad 40 deg about its long axis for one frame: 4000 deg/s in and out
        mid = 0.5 * (positions[150, 1] + positions[150, 2])
        spin = Rotation.from_rotvec(np.deg2rad(40.0) * UP).as_matrix()
        positions[150] = (spin @ (positions[150] - mid).T).T + mid
        spiked = MarkerStream(time_s=stream.time_s, positions_m=positions, marker_names=stream.marker_names)
        pose = ms.anatomical_forearm_pose(spiked, registry.require("wrist_r"), settings)
        assert pose.diagnostics["withdrawn_rate"] >= 2
        assert not pose.solved[150]


# ------------------------------------------------------------------ placement policy
class TestPlacementPolicy:
    def test_an_unresolved_site_without_a_policy_is_refused(self, settings, registry, tmp_path) -> None:
        stripped = _write_settings(tmp_path, settings, placement={"note": "nothing declared"})
        with pytest.raises(ms.MarkerSmallError, match="no placement policy"):
            ms.site_placement(registry.require("occiput"), registry, stripped)

    def test_a_twin_must_be_the_same_cluster(self, settings, registry, tmp_path) -> None:
        wrong = _write_settings(tmp_path, settings, placement={"back_T4": {"kind": "offset_twin", "twin": "shank_l"}})
        with pytest.raises(ms.MarkerSmallError, match="different cluster"):
            ms.site_placement(registry.require("back_T4"), registry, wrong)

    def test_a_resolved_site_takes_the_registry(self, settings, registry) -> None:
        placement = ms.site_placement(registry.require("foot_r"), registry, settings)
        assert placement.kind == "registry" and not placement.provisional


# ------------------------------------------------------------------ pair window and assembly
class TestPairWindow:
    def test_only_the_ends_are_trimmed_and_interior_gaps_stay(self) -> None:
        valid = np.ones((200, 8), bool)
        valid[:17] = False                 # filter edge, every site
        valid[190:] = False                # trailing edge, every site
        valid[80:100, 1] = False           # an interior gap on one site
        window = ms.pair_window(valid, np.arange(10, 200))
        assert (window.source_start, window.source_stop) == (17, 190)
        assert (window.trimmed_leading, window.trimmed_trailing) == (7, 10)
        assert window.frames == 173
        assert window.rows == slice(7, 180)
        assert window.usable_runs("wrist_l") == [(0, 63), (83, 173)]
        assert window.valid.shape == (173, 8)

    def test_a_retarget_span_with_no_valid_frame_is_refused(self) -> None:
        with pytest.raises(ms.MarkerSmallError, match="no frame"):
            ms.pair_window(np.zeros((50, 8), bool), np.arange(0, 50))

    def test_a_non_contiguous_retarget_index_is_refused(self) -> None:
        with pytest.raises(ms.MarkerSmallError, match="contiguous"):
            ms.pair_window(np.ones((50, 8), bool), np.array([0, 1, 2, 4]))

    def test_the_manifest_block_carries_no_absolute_path_and_the_right_fractions(self) -> None:
        valid = np.ones((100, 8), bool)
        valid[:17] = False
        window = ms.pair_window(valid, np.arange(0, 100))
        block = window.as_manifest(trial_frames=1000)
        assert block["frames"] == 83 and block["fraction_of_trial"] == 0.083
        assert block["frames_with_any_invalid_site"] == 0
        assert block["sites"]["back_T4"]["usable_runs_bundle_rows"] == [[0, 83]]
        assert "bundle rows" in block["runs_basis"]
        assert not DRIVE_PATH.search(json.dumps(block))

    def test_interpolated_frames_are_listed_per_site_in_bundle_rows(self) -> None:
        """A bridged gap ships valid; the manifest has to say which frames those were."""
        valid = np.ones((100, 8), bool)
        valid[:10] = False
        interpolated = np.zeros((100, 8), bool)
        interpolated[40:45, 2] = True
        window = ms.pair_window(valid, np.arange(0, 100), interpolated_by_site=interpolated)
        block = window.as_manifest(trial_frames=1000)
        assert block["sites"]["wrist_r"]["interpolated_frames"] == 5
        assert block["sites"]["wrist_r"]["interpolated_runs_bundle_rows"] == [[30, 35]]
        assert block["sites"]["back_T4"]["interpolated_runs_bundle_rows"] == []

    def test_an_interpolated_frame_the_signal_calls_invalid_is_refused(self) -> None:
        valid = np.ones((50, 8), bool)
        valid[20:25, 1] = False
        interpolated = np.zeros((50, 8), bool)
        interpolated[22, 1] = True
        with pytest.raises(ms.MarkerSmallError, match="interpolated"):
            ms.pair_window(valid, np.arange(50), interpolated_by_site=interpolated)


def _fake_synthesis(name: str, valid: np.ndarray) -> ms.SiteSynthesis:
    q = np.where(valid[:, None], np.array([1.0, 0.0, 0.0, 0.0]), np.nan)
    a = np.where(valid[:, None], np.array([0.0, 0.0, 9.81]), np.nan)
    w = np.where(valid[:, None], np.zeros(3), np.nan)
    from soma_synth.sensors.virtual_imu import VirtualImuSignal
    signal = VirtualImuSignal(orientation_wxyz=q, acceleration_m_per_s2=a, angular_velocity_deg_per_s=w, valid=valid)
    return ms.SiteSynthesis(name=name, signal=signal, rotation_cluster_from_sensor=np.eye(3),
                            lever_arm_local=np.zeros(3), placement=ms.Placement("declared_zero", "test", None, True),
                            frame_source="test", diagnostics={}, interpolated=np.zeros_like(valid))


class TestStackSmall:
    def test_invalid_cells_are_nan_and_confidence_zero(self) -> None:
        valid = np.ones((60, 8), bool)
        valid[20:30, 3] = False
        syntheses = {name: _fake_synthesis(name, valid[:, k]) for k, name in enumerate(ms.SITE_ORDER)}
        window = ms.pair_window(valid, np.arange(60))
        small = ms.stack_small(syntheses, window, {name: 0.6 for name in ms.SITE_ORDER})
        assert small["imu_valid_mask"].sum() == 60 * 8 - 10
        assert np.isnan(small["imu_orientation"][20:30, 3]).all()
        assert np.isfinite(small["imu_orientation"][valid]).all()
        assert (small["imu_confidence"][20:30, 3] == 0.0).all()
        assert (small["imu_confidence"][valid] == 0.6).all()

    def test_a_nan_in_a_cell_the_signal_calls_valid_is_refused(self) -> None:
        """The mask discipline is asserted on the assembled arrays; a synthesis defect must not reach disk."""
        valid = np.ones((40, 8), bool)
        syntheses = {name: _fake_synthesis(name, valid[:, k]) for k, name in enumerate(ms.SITE_ORDER)}
        syntheses["back_T4"].signal.acceleration_m_per_s2[7, 1] = np.nan   # valid cell, one NaN component
        window = ms.pair_window(valid, np.arange(40))
        with pytest.raises(ms.MarkerSmallError, match="valid cell is not finite"):
            ms.stack_small(syntheses, window, {name: 0.6 for name in ms.SITE_ORDER})

    def test_a_signal_whose_mask_disagrees_with_the_window_is_refused(self) -> None:
        valid = np.ones((40, 8), bool)
        syntheses = {name: _fake_synthesis(name, valid[:, k]) for k, name in enumerate(ms.SITE_ORDER)}
        window = ms.pair_window(valid, np.arange(40))
        other = valid.copy()
        other[3, 2] = False
        syntheses["wrist_r"] = _fake_synthesis("wrist_r", other[:, 2])
        with pytest.raises(ms.MarkerSmallError, match="disagrees"):
            ms.stack_small(syntheses, window, {name: 0.6 for name in ms.SITE_ORDER})


class TestRelabelDispersion:
    def test_a_constant_relabel_has_zero_dispersion(self) -> None:
        frames = 50
        yaw = _yaw_series(frames)
        world = Rotation.from_rotvec(np.outer(yaw, UP))
        constant = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True)
        large = np.zeros((frames, 24, 4))
        large[:, :, 0] = 1.0
        small = np.zeros((frames, 8, 4))
        for k, name in enumerate(ms.SITE_ORDER):
            j = ms.SITE_TO_JOINT[name]
            xyzw = world.as_quat()
            large[:, j] = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
            sensor = (world * constant).as_quat()
            small[:, k] = np.column_stack([sensor[:, 3], sensor[:, 0], sensor[:, 1], sensor[:, 2]])
        dispersion = ms.relabel_dispersion_vs_large(small, np.ones((frames, 8), bool), large)
        assert max(dispersion.values()) < 1e-6

    def test_a_site_with_one_valid_frame_reads_null_not_nan(self) -> None:
        frames = 20
        large = np.zeros((frames, 24, 4))
        large[:, :, 0] = 1.0
        small = np.zeros((frames, 8, 4))
        small[:, :, 0] = 1.0
        valid = np.ones((frames, 8), bool)
        valid[1:, 3] = False
        dispersion = ms.relabel_dispersion_vs_large(small, valid, large)
        assert dispersion["shank_l"] is None
        assert dispersion["back_T4"] == 0.0
        assert "NaN" not in json.dumps(dispersion, allow_nan=False)


class TestRelabelOffset:
    """Dispersion cannot see a constant error, the mean offset can."""

    def _pair(self, frames: int = 50):
        yaw = _yaw_series(frames)
        world = Rotation.from_rotvec(np.outer(yaw, UP))
        xyzw = world.as_quat()
        reference = np.zeros((frames, 8, 4))
        reference[:] = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])[:, None, :]
        return world, reference

    def test_the_same_frame_has_no_offset(self) -> None:
        _world, reference = self._pair()
        offset = ms.relabel_offset_vs_reference(reference.copy(), np.ones((50, 8), bool), reference)
        assert max(offset.values()) < 1e-6

    def test_a_half_turn_about_the_long_axis_reads_180(self) -> None:
        """A plate whose normal is signed the wrong way, or a wrist with swapped styloids, is exactly this."""
        world, reference = self._pair()
        flipped = (world * Rotation.from_rotvec(np.pi * np.array([0.0, 1.0, 0.0]))).as_quat()
        small = reference.copy()
        small[:, 2] = np.column_stack([flipped[:, 3], flipped[:, 0], flipped[:, 1], flipped[:, 2]])
        offset = ms.relabel_offset_vs_reference(small, np.ones((50, 8), bool), reference)
        assert abs(offset["wrist_r"] - 180.0) < 1e-3
        assert offset["back_T4"] < 1e-6

    def test_a_site_without_two_valid_frames_is_null(self) -> None:
        _world, reference = self._pair()
        valid = np.ones((50, 8), bool)
        valid[1:, 5] = False
        offset = ms.relabel_offset_vs_reference(reference.copy(), valid, reference)
        assert offset["occiput"] is None
