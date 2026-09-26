"""The emitted GAITEX bundle must hold the line it claims to hold.

Generation happened under a declared label -- experimental, non-candidate, not evaluated,
internal only -- and under an explicit promise that no source file was touched. Both of those
are claims about the artifact rather than about the code, so they are checked against the
artifact. The bundle is found under ``SOMA_DATA_ROOT``, so these skip where the variable is unset
or the bundle is not there.

**Every test here skips where the bundle is not present**, so the generator's side of the same
promises is checked in `test_gaitex_generator_promises.py`, which needs no data. Keep the two
in step: a promise added here that only the artifact can show belongs here, and one the code can
be asked about belongs there.

Since 2026-09-06 the small is synthesised from the markers (ADR-0039): the partition on the
manifest is worn-unit-backed / declared-placement, `imu_valid_mask` is informative and governs
NaN in the three IMU arrays, and the worn-unit comparison travels on the manifest.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest

from soma_synth.pipeline import paths


def _bundle() -> pathlib.Path | None:
    """The GAITEX bundle under ``SOMA_DATA_ROOT``, or None when the variable does not resolve."""
    try:
        return paths.data_root() / "runs/experimental_generation_poc_demo/gaitex_unified8"
    except paths.PathConfigError:
        return None


BUNDLE = _bundle()   # no version suffix

needs_bundle = pytest.mark.skipif(
    BUNDLE is None or not (BUNDLE / "INDEX.json").is_file(),
    reason="SOMA_DATA_ROOT unset, or the GAITEX bundle has not been emitted under it"
)
#: A drive-letter path (a letter, a colon, then a slash or backslash): no manifest value carries one.
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")

# Duplicated from the generator on purpose: a test that imports the value it is checking checks
# nothing. Five sites stand opposite a worn XSens unit (back_T4 through the sternum plate it is);
# three carry a marker-fitted rotation with a declared zero offset and no worn unit.
WORN_UNIT_BACKED = {"back_T4", "shank_l", "shank_r", "foot_l", "foot_r"}
DECLARED_PLACEMENT = {"wrist_l", "wrist_r", "occiput"}
IMU_ARRAYS = ("imu_orientation", "imu_acceleration", "imu_angular_velocity")


@pytest.fixture(scope="module")
def index() -> dict:
    return json.loads((BUNDLE / "INDEX.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def a_take() -> pathlib.Path:
    return sorted(path for path in BUNDLE.iterdir() if path.is_dir())[0]


@pytest.fixture(scope="module")
def manifest(a_take) -> dict:
    return json.loads((a_take / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def small(a_take) -> dict:
    with np.load(a_take / "small_reference.npz", allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


@needs_bundle
class TestTheDeclaredLabel:
    def test_the_index_calls_the_dataset_experimental_and_non_candidate(self, index) -> None:
        assert index["artifact_class"] == "experimental_non_candidate"

    def test_the_index_keeps_the_dataset_internal(self, index) -> None:
        assert index["distribution_scope"] == "internal_only"

    def test_no_quality_gate_is_claimed(self, index, manifest) -> None:
        """A gate this dataset never went through must not read as one it passed."""
        assert manifest["quality_gate"] == "NOT_EVALUATED"

    def test_the_spec_is_the_one_the_other_corpora_use(self, index) -> None:
        assert index["spec_id"] == "qmd_unified8_smpl18"

    def test_the_data_description_carries_the_internal_only_banner(self) -> None:
        """L4 looks for this substring, and a rename must not be able to drop it."""
        descriptions = list(BUNDLE.glob("*DATA_DESCRIPTION*.md"))
        assert descriptions
        assert "INTERNAL-ONLY" in descriptions[0].read_text(encoding="utf-8")

    def test_the_small_is_the_marker_derived_one(self, small) -> None:
        assert str(small["small_mode"]) == "synthetic_from_markers"
        assert str(small["mount_id"][0]).startswith("gaitex_marker::")


@needs_bundle
class TestConfidenceAgreesWithProvenance:
    """The defect this bundle exists partly to avoid.

    On the AddBiomechanics corpus `occiput` takes the default confidence, the highest in the
    bundle, while its own manifest says the channel carries none of the motion it is named for.
    A consumer reading the number rather than the prose gets the ranking backwards.
    """

    def test_no_declared_site_outranks_a_backed_one(self, small) -> None:
        codes = [str(code) for code in small["sensor_codes"]]
        mask = np.asarray(small["imu_valid_mask"], bool)
        confidence = np.asarray(small["imu_confidence"])
        # the first VALID frame of each site, since frame 0 may sit inside a filter edge
        first = {code: float(confidence[np.argmax(mask[:, k]), k]) for k, code in enumerate(codes)}
        assert max(first[c] for c in DECLARED_PLACEMENT) < min(first[c] for c in WORN_UNIT_BACKED)

    def test_the_manifest_names_the_same_five_sites_as_backed(self, manifest) -> None:
        assert set(manifest["worn_unit_backed_sites"]) == WORN_UNIT_BACKED
        assert set(manifest["declared_placement_sites"]) == DECLARED_PLACEMENT

    def test_every_declared_site_says_so_in_its_own_words(self, manifest) -> None:
        prose = manifest["small_sites_provenance"]
        for code in DECLARED_PLACEMENT:
            assert "DECLARED ZERO" in prose[code] and "NO WORN COUNTERPART" in prose[code]
        assert "STERNUM" in prose["back_T4"]

    def test_the_manifest_says_what_the_confidence_numbers_mean(self, manifest) -> None:
        semantics = manifest["small_synthesis"]["imu_confidence_semantics"]
        assert semantics["invalid_cell"] == 0.0
        assert set(semantics["sites"]) == WORN_UNIT_BACKED | DECLARED_PLACEMENT


@needs_bundle
class TestTheMaskIsDisciplined:
    """ADR-0039 on the artifact: valid cells finite, invalid cells NaN, and large finite throughout."""

    def test_valid_cells_are_finite_and_invalid_cells_are_nan(self, small) -> None:
        mask = np.asarray(small["imu_valid_mask"], bool)
        for key in IMU_ARRAYS:
            finite = np.isfinite(small[key]).all(axis=-1)
            assert finite[mask].all(), key
            assert not np.isfinite(small[key][~mask]).any(), key

    def test_confidence_is_zero_exactly_where_the_mask_is_false(self, small) -> None:
        mask = np.asarray(small["imu_valid_mask"], bool)
        confidence = np.asarray(small["imu_confidence"])
        assert (confidence[~mask] == 0.0).all()
        assert (confidence[mask] > 0.0).all()

    def test_every_site_has_a_valid_frame_and_the_time_axis_is_the_pairs(self, small, a_take) -> None:
        mask = np.asarray(small["imu_valid_mask"], bool)
        assert mask.any(axis=0).all()
        timestamps = np.asarray(small["timestamps_s"])
        assert np.isfinite(timestamps).all()
        assert np.allclose(np.diff(timestamps), 0.01)
        with np.load(a_take / "large_reference.npz", allow_pickle=False) as handle:
            assert int(handle["frame_count"]) == int(small["frame_count"])
            assert np.isfinite(handle["joint_rotation"]).all()

    def test_the_window_is_on_the_manifest_with_its_runs(self, manifest, small) -> None:
        window = manifest["source_window"]
        assert window["frames"] == int(small["frame_count"])
        assert window["csv_frame_stop"] - window["csv_frame_start"] == window["frames"]
        assert manifest["window"]["frame_count"] == window["frames"]
        mask = np.asarray(small["imu_valid_mask"], bool)
        for k, code in enumerate(str(c) for c in small["sensor_codes"]):
            runs = window["sites"][code]["usable_runs_bundle_rows"]
            assert runs and all(0 <= a < b <= window["frames"] for a, b in runs)
            assert sum(b - a for a, b in runs) == int(mask[:, k].sum())
            for a, b in window["sites"][code]["interpolated_runs_bundle_rows"]:
                assert mask[a:b, k].all()          # a bridged frame ships valid, and is listed

    def test_the_manifest_is_json_without_nan(self, a_take) -> None:
        """The contract forbids NaN in a manifest; json.dump writes it unless told not to."""
        text = (a_take / "manifest.json").read_text(encoding="utf-8")

        def refuse(token):
            raise AssertionError(f"manifest carries {token}")
        json.loads(text, parse_constant=refuse)


@needs_bundle
class TestTheWornUnitComparisonTravels:
    def test_the_five_backed_sites_are_compared(self, manifest) -> None:
        sites = manifest["worn_unit_comparison"]["sites"]
        assert set(sites) == WORN_UNIT_BACKED
        allowed = {"ok", "measured_counterpart_absent", "too_few_comparable_frames", "comparison_failed"}
        assert {entry["status"] for entry in sites.values()} <= allowed
        ok = [entry for entry in sites.values() if entry["status"] == "ok"]
        assert ok
        for entry in ok:
            assert 0.0 <= entry["median_deg"] <= entry["p95_deg"] <= entry["max_deg"]
            assert entry["frames"] > 0

    def test_the_relabel_dispersion_is_recorded_per_site(self, manifest) -> None:
        dispersion = manifest["relabel_dispersion_vs_large_deg"]["sites"]
        assert set(dispersion) == WORN_UNIT_BACKED | DECLARED_PLACEMENT

    def test_confidence_is_backed_only_where_the_comparison_ran(self, manifest, small) -> None:
        """A backed site is backed on a take only if its unit was there."""
        compared = set(manifest["worn_unit_compared_sites"])
        assert compared <= WORN_UNIT_BACKED
        statuses = manifest["worn_unit_comparison"]["sites"]
        assert compared == {s for s, entry in statuses.items() if entry["status"] == "ok"}
        codes = [str(code) for code in small["sensor_codes"]]
        mask = np.asarray(small["imu_valid_mask"], bool)
        confidence = np.asarray(small["imu_confidence"])
        for k, code in enumerate(codes):
            first = float(confidence[np.argmax(mask[:, k]), k])      # float32 on disk
            assert first == pytest.approx(0.6 if code in compared else 0.3, abs=1e-6), code

    def test_the_mean_offset_against_the_smpl_frame_is_recorded_and_bounded_on_backed_sites(self, manifest) -> None:
        """A plate signed the wrong way is a constant half turn, which the
        dispersion cannot see and the mean offset can."""
        offset = manifest["relabel_offset_vs_smpl_anatomical_deg"]["sites"]
        guard = manifest["small_synthesis"]["relabel_offset_guard_deg"]
        for code in WORN_UNIT_BACKED:
            assert offset[code] is not None and 0.0 <= offset[code] <= guard, code

    def test_the_emitters_smpl_side_validation_of_the_replaced_small_is_gone(self, manifest) -> None:
        """The inverse specific-force check measured a small that was thrown away."""
        assert not any(key.startswith("specific_force_inverse") for key in manifest["validation"])
        assert "recomputed" in manifest["validation"]["imu_quat_norm_max_dev_note"]


@needs_bundle
class TestWhatWasWithheldIsStated:
    def test_the_truncation_is_visible_rather_than_inferred(self, manifest) -> None:
        """A take can be a small fraction of its source trial, and the manifest has to say so."""
        coverage = manifest["source_trial_coverage"]
        assert set(coverage) >= {"span_frames", "trial_frames", "fraction"}
        assert 0.0 < coverage["fraction"] <= 1.0

    def test_the_missing_force_and_measured_inertial_streams_are_named(self, manifest) -> None:
        unavailable = " ".join(manifest["unavailable_or_not_applied"]).lower()
        assert "grf" in unavailable
        assert "accelerometer" in unavailable

    def test_the_root_is_recorded_as_recovered_not_as_published(self, manifest) -> None:
        assert "frozen" in manifest["root_translation"]["published"]
        assert "recovered" in manifest["root_translation"]["used"]

    def test_the_governance_draft_is_recorded_as_unresolved(self, manifest) -> None:
        under = manifest["generated_under"]
        assert under["authorises_generation"] is False
        assert len(under["recorded_conflicts"]) == 1
        assert under["recorded_authorizations"] == ["research/decisions/2026-09-06_gaitex_internal_use_authorization.md"]


@needs_bundle
def test_no_absolute_path_leaks_into_a_manifest(a_take) -> None:
    """The spec forbids it and L4 checks it; asserting it here names the reason."""
    manifest = json.loads((a_take / "manifest.json").read_text(encoding="utf-8"))

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    assert not [s for s in strings(manifest) if DRIVE_PATH.search(s)]
