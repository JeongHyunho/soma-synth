"""The 18-joint reduced model goes through smpl18, once per subject, and says so.

What these hold: the settings are the source profile's and nobody else's, the frames of a subject's
takes are pooled as if concatenated, the record carries what smpl18's primer asks a corpus to carry,
the reduction is the package's own, takes posed on different skeletons are never fitted together,
and the bundle-root record round-trips to the same constants a sharded run needs.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from smpl18 import reduce as smpl18_reduce
from smpl18.profile import Profile
from smpl18.skeleton import definition as smpl18_skeleton
from smpl18.skeleton.rotations import axis_angle_to_matrix, matrix_to_quaternion

from soma_synth.pipeline import reduced_model, stages

REPO = Path(__file__).resolve().parents[2]
SOURCES = ("hknu", "prism", "gaitex", "amass", "addbiomechanics")

#: The 18 joints every bundle has stored since the reduced model was introduced. Pinned here, not
#: taken from the package, so a change on the package side shows up as a failure in this repo.
BUNDLE_JOINT18 = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "spine3", "left_foot", "right_foot", "neck", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)


def _rest_skeleton() -> np.ndarray:
    """A crude (24, 3) skeleton whose frozen joints all sit at a distance from their children."""
    offsets = {
        1: (0.09, -0.06, 0.0), 2: (-0.09, -0.06, 0.0), 3: (0.0, 0.10, 0.0),
        4: (0.0, -0.40, 0.0), 5: (0.0, -0.40, 0.0), 6: (0.0, 0.11, 0.0),
        7: (0.0, -0.40, 0.0), 8: (0.0, -0.40, 0.0), 9: (0.0, 0.12, 0.0),
        10: (0.0, -0.06, 0.15), 11: (0.0, -0.06, 0.15), 12: (0.0, 0.21, 0.0),
        13: (0.07, 0.16, 0.0), 14: (-0.07, 0.16, 0.0), 15: (0.0, 0.10, 0.0),
        16: (0.11, 0.05, 0.0), 17: (-0.11, 0.05, 0.0), 18: (0.26, 0.0, 0.0),
        19: (-0.26, 0.0, 0.0), 20: (0.25, 0.0, 0.0), 21: (-0.25, 0.0, 0.0),
        22: (0.08, 0.0, 0.0), 23: (-0.08, 0.0, 0.0),
    }
    rest = np.zeros((24, 3))
    for joint in range(1, 24):
        rest[joint] = rest[smpl18_skeleton.PARENTS[joint]] + np.array(offsets[joint])
    return rest


def _poses(seed: int, *lengths: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 0.15, (n, 24, 3)) for n in lengths]


@pytest.fixture(scope="module")
def amass_settings() -> reduced_model.ReductionSettings:
    return reduced_model.settings_for_source("amass")


# ------------------------------------------------------------------ settings


@pytest.mark.parametrize("source", SOURCES)
def test_the_numbers_are_the_source_profiles_own(source):
    given = stages.default_registry().for_source(source).smpl18_profile
    argument = str(REPO / given) if given.endswith(".yaml") else given
    profile = Profile.load(argument)
    settings = reduced_model.settings_for_source(source)
    assert settings.settings["reduce"] == profile.settings["reduce"]
    assert settings.provenance["reduce"] == {
        "sample_frames": int(profile.settings["reduce"]["sample_frames"]),
        "optimiser": str(profile.settings["reduce"]["optimiser"]),
        "max_evaluations": int(profile.settings["reduce"]["max_evaluations"]),
    }
    assert settings.provenance["profile_sha256"] == profile.sha256


@pytest.mark.parametrize("source", SOURCES)
def test_the_settings_record_names_no_absolute_path(source):
    """Manifests may carry no absolute path; the record goes into every one of them."""
    text = json.dumps(reduced_model.settings_for_source(source).provenance)
    assert ":\\\\" not in text and ":/" not in text and not text.startswith("/")


def test_a_source_without_a_profile_is_refused():
    class NoProfile:
        smpl18_profile = ""

    class Registry:
        path = Path("source_pipelines_v1.yaml")

        def for_source(self, name):
            return NoProfile()

    with pytest.raises(reduced_model.ReductionError, match="no smpl18_profile"):
        reduced_model.settings_for_source("amass", Registry())


# ------------------------------------------------------------------ pooling and fitting


def test_pooling_samples_the_concatenation_without_building_it():
    poses = _poses(3, 300, 900, 50)
    concatenated = np.concatenate(poses)
    expected = axis_angle_to_matrix(concatenated[smpl18_reduce.sample_indices(1250, 400)])
    pooled, total = reduced_model.pooled_rotations(poses, 400)
    assert total == 1250
    np.testing.assert_array_equal(pooled, expected)


def test_a_long_take_weighs_what_its_length_says():
    """A per-take cap would give the 50-frame take as many frames as the 5000-frame one."""
    short, long = _poses(4, 50, 5000)
    long[:] = 0.0
    short[:] = 0.3
    pooled, _ = reduced_model.pooled_rotations([short, long], 101)
    from_short = np.sum(~np.all(np.isclose(pooled, np.eye(3)), axis=(1, 2, 3)))
    assert from_short <= 2


def test_a_pose_that_is_not_24_joints_is_refused():
    with pytest.raises(reduced_model.ReductionError, match="24"):
        reduced_model.pooled_rotations([np.zeros((10, 22, 3))], 5)


def test_the_record_carries_what_the_primer_asks_for(amass_settings):
    fit = reduced_model.fit_subject(_poses(5, 200, 300), _rest_skeleton(), amass_settings,
                                    group="subject_a", basis="unit test")
    record = fit.record
    assert fit.constants.shape == (4, 3, 3)
    assert record["method"] == "smpl18.reduce.fit_constants"
    assert record["scope"] == "subject" and record["group"] == "subject_a"
    assert record["takes"] == 2 and record["frames_pooled"] == 500
    assert record["frames_measured"] == 500
    assert record["frozen_joints"] == list(smpl18_skeleton.FROZEN_JOINT_NAMES)
    assert record["absorbed_by"] == {"spine1": "spine3", "spine2": "spine3",
                                     "left_collar": "left_shoulder",
                                     "right_collar": "right_shoulder"}
    assert np.asarray(record["constants_wxyz"]).shape == (4, 4)
    assert record["residual_fitted_m"]["rms"] <= record["residual_initial_m"]["rms"] + 1e-12
    assert set(record["per_joint_rms_m"]) == {
        smpl18_skeleton.JOINT_NAMES[j] for j in smpl18_skeleton.AFFECTED_BY_FREEZE}
    assert isinstance(record["converged"], bool)
    assert record["settings"] == dict(amass_settings.provenance)
    json.dumps(record, allow_nan=False)


def test_the_constants_are_the_packages_fit(amass_settings):
    poses = _poses(6, 400)
    rest = _rest_skeleton()
    fit = reduced_model.fit_subject(poses, rest, amass_settings, group="g", basis="b")
    direct = smpl18_reduce.fit_constants(axis_angle_to_matrix(poses[0]), rest,
                                         settings=amass_settings.settings)
    np.testing.assert_array_equal(fit.constants, direct.constants)


def test_the_reduction_is_the_packages_and_keeps_distal_orientation():
    local = axis_angle_to_matrix(_poses(7, 30)[0])
    constants = axis_angle_to_matrix(np.random.default_rng(8).normal(0, 0.4, (4, 3)))
    reduced = reduced_model.reduce_local(local, constants)
    np.testing.assert_array_equal(reduced, smpl18_reduce.apply(local, constants))

    def world(rotations):
        out = np.zeros_like(rotations)
        out[:, 0] = rotations[:, 0]
        for joint in range(1, 24):
            out[:, joint] = out[:, smpl18_skeleton.PARENTS[joint]] @ rotations[:, joint]
        return out

    for joint in smpl18_skeleton.AFFECTED_BY_FREEZE:
        np.testing.assert_allclose(world(reduced)[:, joint], world(local)[:, joint], atol=1e-12)


def test_the_eighteen_joints_are_the_ones_bundles_have_always_stored():
    assert reduced_model.JOINT18_NAMES == BUNDLE_JOINT18
    assert tuple(smpl18_skeleton.JOINT_NAMES[j] for j in reduced_model.KEEP18) == BUNDLE_JOINT18


# ------------------------------------------------------------------ groups and the record file


def test_takes_on_one_skeleton_are_one_group():
    rest = _rest_skeleton()
    groups = reduced_model.group_by_skeleton("s1", [("a", _poses(1, 5)[0], rest),
                                                    ("b", _poses(2, 6)[0], rest.copy())])
    assert list(groups) == ["s1"]
    assert groups["s1"][0] == ["a", "b"]


def test_takes_on_different_skeletons_are_never_pooled():
    rest = _rest_skeleton()
    other = rest * 1.05
    groups = reduced_model.group_by_skeleton("s1", [("a", _poses(1, 5)[0], rest),
                                                    ("b", _poses(2, 6)[0], other)])
    assert len(groups) == 2
    assert all(name.startswith("s1|rest:") for name in groups)
    assert sorted(keys for keys, _, _ in groups.values()) == [["a"], ["b"]]


def test_the_record_file_round_trips_to_the_same_constants(tmp_path, amass_settings):
    fits = reduced_model.CorpusFits(amass_settings)
    rest = _rest_skeleton()
    fits.fit("s1", [("take_1", _poses(9, 120)[0], rest), ("take_2", _poses(10, 80)[0], rest)],
             basis="unit test")
    path = fits.write(tmp_path)
    assert path.name == reduced_model.FIT_FILE
    document = reduced_model.read_fit_file(tmp_path)
    assert document["takes"] == {"take_1": "s1", "take_2": "s1"}
    assert document["source"] == "amass" and document["group_count"] == 1
    rebuilt = reduced_model.fit_from_record(document["groups"]["s1"])
    np.testing.assert_allclose(rebuilt.constants, fits.by_take["take_1"].constants, atol=1e-12)
    np.testing.assert_allclose(matrix_to_quaternion(rebuilt.constants),
                               document["groups"]["s1"]["constants_wxyz"], atol=1e-12)

    elsewhere = reduced_model.CorpusFits(amass_settings)
    elsewhere.add_records(document["groups"], document["takes"])
    np.testing.assert_allclose(elsewhere.by_take["take_2"].constants, rebuilt.constants)


def test_a_take_naming_an_unfitted_group_is_refused(tmp_path, amass_settings):
    with pytest.raises(reduced_model.ReductionError, match="not fitted"):
        reduced_model.write_fit_file(tmp_path, amass_settings, {}, {"take_1": "ghost"})


def test_a_missing_record_file_reads_as_none(tmp_path):
    assert reduced_model.read_fit_file(tmp_path) is None


# ------------------------------------------------------------------ the AMASS fit pass


AMASS_ALL = REPO / "scripts" / "poc" / "generate_amass_faithful_all.py"


@pytest.fixture
def amass_all(monkeypatch):
    spec = importlib.util.spec_from_file_location("amass_all_reduced", AMASS_ALL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rest = _rest_skeleton()
    anthro = module.gen.anthro_smpl
    monkeypatch.setattr(anthro, "load_smpl_model_for_gender", lambda letter: {"letter": letter})
    # the skeleton scales with the first beta, so a different betas vector is a different body
    monkeypatch.setattr(anthro, "rest_joints", lambda model, betas: rest * (1.0 + float(betas[0])))
    return module


def _amass_tree(root: Path) -> str:
    amass = root / "extracted" / "amass"
    rng = np.random.default_rng(11)
    for dataset, subject, name, betas0 in (
        ("CMU", "01", "01_02_poses.npz", 0.0),
        ("CMU", "01", "01_03_poses.npz", 0.0),
        ("CMU", "01", "01_04_poses.npz", 0.2),     # same subject, other betas: its own group
        ("KIT", "10", "Walk_poses.npz", 0.0),
    ):
        folder = amass / dataset / subject
        folder.mkdir(parents=True, exist_ok=True)
        betas = np.zeros(16)
        betas[0] = betas0
        np.savez(folder / name, poses=rng.normal(0, 0.1, (40, 156)), trans=np.zeros((40, 3)),
                 betas=betas, gender=np.array("male"), mocap_framerate=np.array(120.0))
    (amass / "KIT" / "10" / "shape.npz").write_bytes(b"")   # non-motion: never fitted
    return str(amass)


def test_the_amass_pass_fits_each_group_once_and_writes_the_record(tmp_path, amass_all):
    specs = amass_all.enumerate_specs(_amass_tree(tmp_path), ["CMU", "KIT"])
    out_root = tmp_path / "bundle"
    fits = amass_all.ensure_fits(specs, str(out_root))
    document = reduced_model.read_fit_file(out_root)
    assert len(document["groups"]) == 3
    assert not (out_root / amass_all.FIT_LOCK).exists()
    shared = [amass_all.bundle_dirname(s) for s in specs if s.sequence_id in ("01_02_poses", "01_03_poses")]
    assert document["takes"][shared[0]] == document["takes"][shared[1]]
    np.testing.assert_array_equal(fits[shared[0]].constants, fits[shared[1]].constants)
    assert fits[shared[0]].record["takes"] == 2
    split = [amass_all.bundle_dirname(s) for s in specs if s.sequence_id == "01_04_poses"][0]
    assert document["takes"][split] != document["takes"][shared[0]]
    assert all("shape" not in rel for rel in document["takes"])


def test_a_second_process_reuses_the_record_instead_of_refitting(tmp_path, amass_all, monkeypatch):
    specs = amass_all.enumerate_specs(_amass_tree(tmp_path), ["CMU", "KIT"])
    out_root = tmp_path / "bundle"
    first = amass_all.ensure_fits(specs, str(out_root))

    def refit(*_args, **_kwargs):
        raise AssertionError("the fit file already covers this run")

    monkeypatch.setattr(amass_all, "fit_all_groups", refit)
    again = amass_all.ensure_fits(specs, str(out_root))
    for rel, fit in first.items():
        np.testing.assert_allclose(again[rel].constants, fit.constants, atol=1e-12)


def test_a_waiting_shard_takes_the_file_the_lock_holder_wrote(tmp_path, amass_all, monkeypatch):
    specs = amass_all.enumerate_specs(_amass_tree(tmp_path), ["CMU", "KIT"])
    out_root = tmp_path / "bundle"
    staging = tmp_path / "staging"
    amass_all.ensure_fits(specs, str(staging))            # the file the "other shard" will publish
    out_root.mkdir()
    lock = out_root / amass_all.FIT_LOCK
    lock.write_text("")
    monkeypatch.setattr(amass_all, "FIT_WAIT_POLL_S", 0.01)

    def other_shard():
        time.sleep(0.2)
        (out_root / reduced_model.FIT_FILE).write_bytes(
            (staging / reduced_model.FIT_FILE).read_bytes())
        lock.unlink()

    worker = threading.Thread(target=other_shard)
    worker.start()
    fits = amass_all.ensure_fits(specs, str(out_root))
    worker.join()
    assert len(fits) == 4


def test_a_lock_released_without_a_record_is_an_error(tmp_path, amass_all, monkeypatch):
    specs = amass_all.enumerate_specs(_amass_tree(tmp_path), ["CMU", "KIT"])
    out_root = tmp_path / "bundle"
    out_root.mkdir()
    lock = out_root / amass_all.FIT_LOCK
    lock.write_text("")
    monkeypatch.setattr(amass_all, "FIT_WAIT_POLL_S", 0.01)
    threading.Timer(0.1, lock.unlink).start()
    with pytest.raises(RuntimeError, match="fit lock was released"):
        amass_all.ensure_fits(specs, str(out_root))


def test_every_bundle_owes_the_record_file():
    from soma_synth.contracts import dataset_profiles

    registry = dataset_profiles.load()
    for source in SOURCES:
        assert reduced_model.FIT_FILE in registry.required_bundle_files(source)
