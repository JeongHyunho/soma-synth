"""Fitting SMPL shape to a subject's measured bone lengths.

The body model is licence-gated and lives outside the repository, so these tests inject a
miniature stand-in with the same interface (v_template, shapedirs, J_regressor, faces). That
also makes the arithmetic checkable: with a synthetic model the true betas are known, so the
fit can be asked to recover them exactly rather than merely to look plausible.

The real model is exercised separately and skipped when absent.
"""

import numpy as np
import pytest

from soma_synth.addbio_retarget.shape_fit import (
    BODY_DENSITY_KG_M3,
    GENDER_FEMALE,
    GENDER_MALE,
    GENDER_NEUTRAL,
    UnresolvedGender,
    fit_betas,
    measured_segment_lengths,
    normalise_biological_sex,
    smpl_mass_kg,
    smpl_segment_lengths,
    smpl_stature,
    smpl_volume_m3,
)
from soma_synth.addbio_retarget.smpl_correspondence import SMPL24_PARENTS

N_BETAS = 4
N_VERTS = 24


def tiny_model(seed=0):
    """A stand-in body model: one vertex per joint, so J_regressor is the identity.

    Shape directions are fixed random offsets, which is enough for the fit to have a unique
    answer while keeping the true betas known.
    """
    rng = np.random.default_rng(seed)
    template = np.zeros((N_VERTS, 3))
    for joint, parent in enumerate(SMPL24_PARENTS):
        if parent < 0:
            continue
        template[joint] = template[parent] + rng.normal(0.0, 0.25, 3)
    shapedirs = rng.normal(0.0, 0.05, (N_VERTS, 3, N_BETAS))
    return {
        "v_template": template,
        "shapedirs": shapedirs,
        "J_regressor": np.eye(N_VERTS),
        "parents": np.asarray(SMPL24_PARENTS),
    }


def segments_from(model, betas):
    return smpl_segment_lengths(model, betas)


# ---------------------------------------------------------------- gender normalisation


@pytest.mark.parametrize(
    ("written", "expected"),
    [("male", GENDER_MALE), ("Male", GENDER_MALE), ("M", GENDER_MALE),
     ("female", GENDER_FEMALE), ("f", GENDER_FEMALE), ("  FEMALE ", GENDER_FEMALE)],
)
def test_the_spellings_the_cohort_actually_uses_resolve(written, expected):
    assert normalise_biological_sex(written) == expected


@pytest.mark.parametrize("written", ["unknown", "", "   ", "other", "n/a"])
def test_anything_unresolvable_stays_unresolved(written):
    """Never quietly a man: an unreadable field is its own answer, and the caller decides."""
    assert normalise_biological_sex(written) is None


def test_asking_for_a_model_with_an_unresolved_sex_raises_rather_than_guessing():
    with pytest.raises(UnresolvedGender):
        normalise_biological_sex("unknown", strict=True)


def test_an_explicit_neutral_choice_is_still_recorded_as_a_choice():
    assert normalise_biological_sex("unknown", fallback=GENDER_NEUTRAL) == GENDER_NEUTRAL


def test_a_stated_sex_is_never_overridden_by_the_fallback():
    """The fallback answers `unknown`; it must not reach a subject who said male or female."""
    for written, expected in (("male", GENDER_MALE), ("f", GENDER_FEMALE),
                              ("Female", GENDER_FEMALE), ("M", GENDER_MALE)):
        assert normalise_biological_sex(written, fallback=GENDER_NEUTRAL) == expected


def test_the_neutral_model_has_its_own_file_rather_than_borrowing_one():
    """Neutral is a third body, not male under another name."""
    from soma_synth.addbio_retarget.shape_fit import clean_model_path_for_gender

    names = {
        gender: clean_model_path_for_gender(gender, root="/x").name
        for gender in (GENDER_MALE, GENDER_FEMALE, GENDER_NEUTRAL)
    }
    assert len(set(names.values())) == 3
    assert names[GENDER_NEUTRAL] == "SMPL_NEUTRAL_clean.npz"


# ---------------------------------------------------------------- measured lengths


def test_segments_are_taken_only_where_both_ends_have_a_centre():
    centres = {"a": np.zeros((3, 3)), "b": np.ones((3, 3))}
    pairs = {(0, 1): ("a", "b"), (1, 4): ("b", None)}
    lengths = measured_segment_lengths(pairs, centres)
    assert set(lengths) == {(0, 1)}
    assert lengths[(0, 1)] == pytest.approx(np.sqrt(3.0))


def test_a_length_that_wobbles_is_summarised_by_its_median_not_its_first_frame():
    """Hip-to-knee is not rigid: the walker knee slides. One frame is not the bone."""
    a = np.zeros((5, 3))
    b = np.zeros((5, 3))
    b[:, 0] = [1.0, 1.2, 1.1, 1.3, 100.0]        # last frame a gross outlier
    lengths = measured_segment_lengths({(0, 1): ("a", "b")}, {"a": a, "b": b})
    assert lengths[(0, 1)] == pytest.approx(1.2)


# ---------------------------------------------------------------- the fit


def test_the_fit_recovers_betas_it_was_given():
    model = tiny_model()
    truth = np.array([0.9, -1.4, 0.5, 0.2])
    targets = segments_from(model, truth)
    fit = fit_betas(model, targets, n_betas=N_BETAS, regularisation=0.0)
    np.testing.assert_allclose(fit.betas, truth, atol=1e-4)
    assert fit.residual_m == pytest.approx(0.0, abs=1e-6)


def test_the_fit_starts_from_zero_and_returns_zero_for_an_average_body():
    model = tiny_model()
    targets = segments_from(model, np.zeros(N_BETAS))
    fit = fit_betas(model, targets, n_betas=N_BETAS, regularisation=0.0)
    np.testing.assert_allclose(fit.betas, np.zeros(N_BETAS), atol=1e-5)


def test_regularisation_pulls_an_underdetermined_fit_toward_the_mean_body():
    """One bone cannot determine ten numbers; without a pull the answer wanders."""
    model = tiny_model()
    everything = segments_from(model, np.array([1.5, -1.0, 0.7, 0.4]))
    only_one = dict([next(iter(everything.items()))])
    loose = fit_betas(model, only_one, n_betas=N_BETAS, regularisation=0.0)
    tight = fit_betas(model, only_one, n_betas=N_BETAS, regularisation=10.0)
    assert np.linalg.norm(loose.betas) > 1e-3, "the loose fit should have moved somewhere"
    assert np.linalg.norm(tight.betas) < np.linalg.norm(loose.betas)


def test_every_segment_reports_its_own_residual():
    model = tiny_model()
    targets = segments_from(model, np.array([0.3, 0.0, 0.0, 0.0]))
    targets[(1, 4)] = targets[(1, 4)] + 0.05          # one bone that cannot be matched
    fit = fit_betas(model, targets, n_betas=N_BETAS, regularisation=0.0)
    assert set(fit.segment_residuals) == set(targets)
    worst = max(fit.segment_residuals, key=lambda k: abs(fit.segment_residuals[k]))
    assert worst == (1, 4)


def test_the_fit_records_what_it_used():
    model = tiny_model()
    targets = segments_from(model, np.zeros(N_BETAS))
    fit = fit_betas(model, targets, n_betas=N_BETAS, gender=GENDER_FEMALE, regularisation=0.1)
    assert fit.gender == GENDER_FEMALE
    assert fit.provenance["segments_used"] == len(targets)
    assert fit.provenance["regularisation"] == 0.1
    assert fit.provenance["stature_used"] is False
    assert fit.provenance["mass_used"] is False


def test_a_stature_term_moves_the_answer_and_is_recorded():
    model = tiny_model()
    targets = segments_from(model, np.zeros(N_BETAS))
    without = fit_betas(model, targets, n_betas=N_BETAS, regularisation=0.0)
    with_height = fit_betas(
        model, targets, n_betas=N_BETAS, regularisation=0.0,
        stature_m=float(smpl_stature(model, np.zeros(N_BETAS))) + 0.10,
    )
    assert with_height.provenance["stature_used"] is True
    assert not np.allclose(with_height.betas, without.betas)


def test_refusing_a_fit_with_nothing_to_fit_to():
    model = tiny_model()
    with pytest.raises(ValueError):
        fit_betas(model, {}, n_betas=N_BETAS)


# ---------------------------------------------------------------- landmark offsets


def test_the_offset_table_is_opt_in_and_named_in_the_result():
    from soma_synth.addbio_retarget.shape_fit import LANDMARK_OFFSETS_V1

    model = tiny_model()
    targets = segments_from(model, np.zeros(N_BETAS))
    plain = fit_betas(model, targets, n_betas=N_BETAS, regularisation=0.0)
    assert plain.provenance["landmark_offsets"] is None

    shifted = fit_betas(
        model, targets, n_betas=N_BETAS, regularisation=0.0, landmark_offsets="v1"
    )
    assert shifted.provenance["landmark_offsets"] == "v1"
    # the table shifts targets, so the answer must actually differ
    assert not np.allclose(shifted.betas, plain.betas)
    assert set(LANDMARK_OFFSETS_V1).issubset(set(targets))


def test_an_unknown_offset_table_is_refused_rather_than_ignored():
    model = tiny_model()
    targets = segments_from(model, np.zeros(N_BETAS))
    with pytest.raises(ValueError, match="unknown landmark offset table"):
        fit_betas(model, targets, n_betas=N_BETAS, landmark_offsets="v99")


def test_the_offsets_are_per_side_because_the_templates_are_not_symmetric():
    """A symmetric correction would be wrong on one side; the table must not assume symmetry."""
    from soma_synth.addbio_retarget.shape_fit import LANDMARK_OFFSETS_V1

    mirrored = [((0, 1), (0, 2)), ((1, 4), (2, 5)), ((4, 7), (5, 8)),
                ((7, 10), (8, 11)), ((16, 18), (17, 19)), ((18, 20), (19, 21))]
    for left, right in mirrored:
        assert left in LANDMARK_OFFSETS_V1 and right in LANDMARK_OFFSETS_V1
    assert any(
        LANDMARK_OFFSETS_V1[left] != LANDMARK_OFFSETS_V1[right] for left, right in mirrored
    )


def test_no_offset_is_large_enough_to_be_a_units_mistake():
    """Values are metres. A stray millimetre-scaled entry would be a 1000x error."""
    from soma_synth.addbio_retarget.shape_fit import LANDMARK_OFFSETS_V1

    assert all(abs(v) < 0.10 for v in LANDMARK_OFFSETS_V1.values())


# ---------------------------------------------------------------- loading a model


def test_a_clean_npz_round_trips_with_or_without_faces(tmp_path):
    from soma_synth.addbio_retarget.shape_fit import load_clean_model

    model = tiny_model()
    path = tmp_path / "m.npz"
    np.savez(path, v_template=model["v_template"], shapedirs=model["shapedirs"],
             J_regressor=model["J_regressor"])
    loaded = load_clean_model(path)
    assert "faces" not in loaded
    np.testing.assert_allclose(loaded["v_template"], model["v_template"])
    assert list(loaded["parents"]) == list(SMPL24_PARENTS)

    withfaces = tmp_path / "n.npz"
    np.savez(withfaces, v_template=model["v_template"], shapedirs=model["shapedirs"],
             J_regressor=model["J_regressor"], faces=np.zeros((2, 3), np.int64))
    assert "faces" in load_clean_model(withfaces)


def test_both_spellings_of_the_triangle_list_are_read(tmp_path):
    """The prepared male npz says `faces`; the female one says `f`.

    Honouring only one silently drops the mesh for one gender, which drops the mass term with
    it and fits men and women by different criteria.
    """
    from soma_synth.addbio_retarget.shape_fit import load_clean_model

    model = tiny_model()
    triangles = np.zeros((2, 3), np.int64)
    for key in ("faces", "f"):
        path = tmp_path / f"{key}.npz"
        np.savez(path, v_template=model["v_template"], shapedirs=model["shapedirs"],
                 J_regressor=model["J_regressor"], **{key: triangles})
        loaded = load_clean_model(path)
        assert loaded["faces"].shape == (2, 3)
        assert loaded["faces_key"] == key


def test_a_gender_with_no_model_is_refused_not_substituted():
    from soma_synth.addbio_retarget.shape_fit import clean_model_path_for_gender

    assert clean_model_path_for_gender(GENDER_MALE, root="/x").name == "SMPL_MALE_clean.npz"
    assert clean_model_path_for_gender(GENDER_FEMALE, root="/x").name == "SMPL_FEMALE_clean.npz"
    with pytest.raises(UnresolvedGender):
        clean_model_path_for_gender("unknown", root="/x")


@pytest.mark.parametrize("unset", ["absent", "blank"])
def test_without_a_root_or_soma_data_root_the_refusal_names_the_variable(monkeypatch, unset):
    """Without a root or SOMA_DATA_ROOT the call raises a PathConfigError that names the variable.

    The WearGait HKNU transport calls this with `root=None` unless `--clean-smpl-root` or
    CLEAN_SMPL_ROOT is given, and the message is what its failure row carries, so it has to say
    which variable to set."""
    from soma_synth.addbio_retarget.shape_fit import clean_model_path_for_gender
    from soma_synth.pipeline.paths import PathConfigError

    for name in ("SOMA_DATA_ROOT", "SOMA_BODY_MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)
    if unset == "blank":
        monkeypatch.setenv("SOMA_DATA_ROOT", "  ")
    for gender in (GENDER_MALE, GENDER_FEMALE, GENDER_NEUTRAL):
        with pytest.raises(PathConfigError, match="SOMA_DATA_ROOT is not set"):
            clean_model_path_for_gender(gender)
        with pytest.raises(PathConfigError, match="SOMA_DATA_ROOT is not set"):
            clean_model_path_for_gender(gender, root=None)


# ---------------------------------------------------------------- mass from the mesh


def cube_model(side=2.0):
    """A closed unit-ish cube, so the volume has an answer known in advance."""
    h = side / 2.0
    verts = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    faces = np.array([
        [0, 2, 3], [0, 3, 1],      # x = -h
        [4, 5, 7], [4, 7, 6],      # x = +h
        [0, 1, 5], [0, 5, 4],      # y = -h
        [2, 6, 7], [2, 7, 3],      # y = +h
        [0, 4, 6], [0, 6, 2],      # z = -h
        [1, 3, 7], [1, 7, 5],      # z = +h
    ])
    return {
        "v_template": verts,
        "shapedirs": np.zeros((8, 3, 1)),
        "J_regressor": np.eye(8),
        "faces": faces,
    }


def test_volume_is_the_enclosed_volume_whatever_the_winding():
    model = cube_model(side=2.0)
    assert smpl_volume_m3(model, np.zeros(1)) == pytest.approx(8.0)
    flipped = dict(model, faces=model["faces"][:, ::-1])
    assert smpl_volume_m3(flipped, np.zeros(1)) == pytest.approx(8.0)


def test_mass_is_volume_times_a_stated_density():
    model = cube_model(side=1.0)
    assert smpl_mass_kg(model, np.zeros(1)) == pytest.approx(BODY_DENSITY_KG_M3)
    assert smpl_mass_kg(model, np.zeros(1), density_kg_m3=500.0) == pytest.approx(500.0)


def test_mass_needs_a_mesh_and_says_so_when_there_is_none():
    with pytest.raises(ValueError, match="faces"):
        smpl_mass_kg(tiny_model(), np.zeros(N_BETAS))


def test_a_mass_term_is_only_taken_when_the_model_can_supply_one():
    model = tiny_model()
    targets = segments_from(model, np.zeros(N_BETAS))
    with pytest.raises(ValueError, match="faces"):
        fit_betas(model, targets, n_betas=N_BETAS, mass_kg=70.0)


# ---------------------------------------------------------------- setting the lengths

# Ten betas do not span every cohort. When the measured hip separation is 213 mm and the shape
# space tops out at 135, no weighting recovers it -- the answer is not in the space. Large is a
# skeleton and a set of rotations, so the lengths can be set afterwards; these tests hold the
# rule that setting them must not also move anything else.

from soma_synth.addbio_retarget.shape_fit import rescale_rest_joints  # noqa: E402
from soma_synth.addbio_retarget.smpl_correspondence import (  # noqa: E402
    SMPL24_PARENTS as PARENTS,
)


def a_skeleton(seed=11):
    rng = np.random.default_rng(seed)
    joints = np.zeros((24, 3))
    for child, parent in enumerate(PARENTS):
        if parent >= 0:
            joints[child] = joints[parent] + rng.normal(0.0, 0.15, 3)
    return joints


def length(joints, a, b):
    return float(np.linalg.norm(joints[b] - joints[a]))


def direction(joints, a, b):
    v = joints[b] - joints[a]
    return v / np.linalg.norm(v)


def test_asking_for_nothing_changes_nothing():
    joints = a_skeleton()
    scaled, applied = rescale_rest_joints(joints)
    assert np.abs(scaled - joints).max() < 1e-15
    assert applied == {"span": {}, "chain": {}, "bone": {}}


def test_a_bone_ends_at_the_length_it_was_given_and_points_where_it_did():
    joints = a_skeleton()
    scaled, applied = rescale_rest_joints(joints, bone_targets={(1, 4): 0.42})
    assert abs(length(scaled, 1, 4) - 0.42) < 1e-12
    assert np.abs(direction(scaled, 1, 4) - direction(joints, 1, 4)).max() < 1e-12
    assert applied["bone"]["1-4"]["after_m"] == 0.42


def test_scaling_a_bone_carries_its_subtree_and_leaves_the_rest_alone():
    joints = a_skeleton()
    scaled, _ = rescale_rest_joints(joints, bone_targets={(1, 4): 0.42})
    moved = scaled[4] - joints[4]
    assert np.abs((scaled[7] - joints[7]) - moved).max() < 1e-12      # the ankle follows
    assert abs(length(scaled, 4, 7) - length(joints, 4, 7)) < 1e-12   # unstretched
    for untouched in (0, 2, 5, 8, 12, 16):
        assert np.abs(scaled[untouched] - joints[untouched]).max() < 1e-12


def test_a_span_reaches_the_separation_without_moving_the_pair_s_midpoint():
    joints = a_skeleton()
    before = 0.5 * (joints[1] + joints[2])
    scaled, applied = rescale_rest_joints(joints, span_targets={(1, 2): 0.213})
    assert abs(length(scaled, 1, 2) - 0.213) < 1e-12
    assert np.abs(0.5 * (scaled[1] + scaled[2]) - before).max() < 1e-12
    assert np.abs(scaled[0] - joints[0]).max() < 1e-12
    assert applied["span"]["1-2"]["after_m"] == 0.213


def test_a_span_works_on_a_pair_that_does_not_share_a_parent():
    """The shoulders hang off their own collars; the hips are siblings. Both are measured the
    same way by the source, so both must be settable the same way here."""
    joints = a_skeleton()
    before = 0.5 * (joints[16] + joints[17])
    scaled, _ = rescale_rest_joints(joints, span_targets={(16, 17): 0.40})
    assert abs(length(scaled, 16, 17) - 0.40) < 1e-12
    assert np.abs(0.5 * (scaled[16] + scaled[17]) - before).max() < 1e-12
    assert np.abs(scaled[9] - joints[9]).max() < 1e-12                 # spine3 did not move
    assert abs(length(scaled, 16, 18) - length(joints, 16, 18)) < 1e-12


def test_a_chain_hits_the_end_to_end_distance_and_keeps_its_shape():
    joints = a_skeleton()
    before = [direction(joints, PARENTS[j], j) for j in (3, 6, 9, 12)]
    scaled, applied = rescale_rest_joints(joints, chain_targets={(0, 12): 0.55})
    assert abs(length(scaled, 0, 12) - 0.55) < 1e-12
    for j, was in zip((3, 6, 9, 12), before):
        assert np.abs(direction(scaled, PARENTS[j], j) - was).max() < 1e-12
    scale = applied["chain"]["0-12"]["scale"]
    assert abs(length(scaled, 9, 12) - scale * length(joints, 9, 12)) < 1e-12


def test_the_spans_and_the_bones_and_the_chains_all_land_together():
    joints = a_skeleton()
    scaled, _ = rescale_rest_joints(
        joints,
        span_targets={(1, 2): 0.213, (16, 17): 0.40},
        chain_targets={(0, 12): 0.55},
        bone_targets={(1, 4): 0.42, (4, 7): 0.41, (16, 18): 0.30},
    )
    assert abs(length(scaled, 1, 2) - 0.213) < 1e-12
    assert abs(length(scaled, 16, 17) - 0.40) < 1e-12
    assert abs(length(scaled, 0, 12) - 0.55) < 1e-12
    assert abs(length(scaled, 1, 4) - 0.42) < 1e-12
    assert abs(length(scaled, 4, 7) - 0.41) < 1e-12
    assert abs(length(scaled, 16, 18) - 0.30) < 1e-12


def test_a_bone_that_is_not_a_tree_edge_is_refused():
    with pytest.raises(ValueError, match="tree edge"):
        rescale_rest_joints(a_skeleton(), bone_targets={(0, 12): 0.55})


def test_a_bone_inside_a_scaled_chain_is_refused_rather_than_silently_overwritten():
    with pytest.raises(ValueError, match="inside a scaled chain"):
        rescale_rest_joints(a_skeleton(), chain_targets={(0, 12): 0.55},
                            bone_targets={(9, 12): 0.10})


def test_a_chain_whose_ends_are_not_related_is_refused():
    with pytest.raises(ValueError, match="not below"):
        rescale_rest_joints(a_skeleton(), chain_targets={(16, 12): 0.4})
