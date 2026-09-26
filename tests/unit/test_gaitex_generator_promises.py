"""What the GAITEX generator promises, checked without the source data.

`test_gaitex_bundle_governance.py` checks the emitted artifact and therefore skips wherever the
corpus is absent -- all of its tests, on CI, every time. So a change that made the generator stamp
`quality_gate: PASS`, or leak an absolute path, or invert the confidence column, would land green
there because the assertions that would catch it never ran.

Two different things are being checked and only one of them needs the corpus. That a SHIPPED
bundle conforms is a fact about the artifact. That the GENERATOR promises the right thing is a
fact about the code, and the GAITEX-specific risk lives there -- the shared emitter is held to
byte equality with `generate_amass_faithful` by its own test, while everything this generator decided
for itself sits in the wrapper. So the wrapper is checked here, with no source data, and the
artifact stays checked there.

Since 2026-09-06 the wrapper's small comes from the markers (ADR-0039). The partition it promises
is no longer driven/undriven but worn-unit-backed/declared-placement, and the confidence column
comes from the settings file rather than from two constants.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATOR = REPOSITORY_ROOT / "scripts" / "poc" / "generate_gaitex_unified8.py"
SETTINGS = REPOSITORY_ROOT / "configs" / "datasets" / "gaitex_synthesis_v1_1.yaml"
#: A drive-letter path (a letter, a colon, then a slash or backslash), which no prose may carry.
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


@pytest.fixture(scope="module")
def generator():
    """The wrapper module, imported by path as the sibling test helpers do."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("gaitex_unified8_under_test", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def settings(generator):
    return generator.marker_small.load_settings(SETTINGS)


# Duplicated from the generator on purpose: a test that imports the value it checks checks
# nothing. Five sites stand opposite a worn XSens unit -- back_T4 through the sternum plate it
# physically is -- and three carry a marker-fitted rotation with a declared zero offset.
WORN_UNIT_BACKED = {"back_T4", "shank_l", "shank_r", "foot_l", "foot_r"}
DECLARED_PLACEMENT = {"wrist_l", "wrist_r", "occiput"}


class TestTheSitePartition:
    def test_the_generator_names_the_five_sites_a_worn_unit_backs(self, generator):
        assert set(generator.WORN_UNIT_BACKED) == WORN_UNIT_BACKED
        assert set(generator.DECLARED_PLACEMENT) == DECLARED_PLACEMENT
        assert set(generator.WORN_UNIT_BACKED) | set(generator.DECLARED_PLACEMENT) == set(generator.SITE_PROVENANCE)

    def test_every_declared_placement_site_says_so_in_its_own_prose(self, generator):
        """The array and the words have to agree, and this is the words."""
        for site in DECLARED_PLACEMENT:
            prose = generator.SITE_PROVENANCE[site]
            assert "DECLARED ZERO" in prose, site
            assert "NO WORN COUNTERPART" in prose, site

    def test_every_backed_site_names_its_worn_unit_as_compared_not_validated(self, generator):
        """The comparison has no acceptance band and does not run on every take,
        so 'validated against' would claim more than the manifest can show."""
        for site in WORN_UNIT_BACKED:
            prose = generator.SITE_PROVENANCE[site]
            assert "WORN-UNIT-BACKED" in prose and "XSens_" in prose, site
            assert "compared with" in prose and "validated against" not in prose, site
            assert "NO WORN COUNTERPART" not in prose, site
        source = GENERATOR.read_text(encoding="utf-8")
        assert "validated against" not in source

    def test_back_t4_says_it_is_the_sternum_sensor(self, generator):
        """The channel is validated against the sternum unit; calling it a T4 sensor would be false."""
        prose = generator.SITE_PROVENANCE["back_T4"]
        assert "STERNUM" in prose and "NOT a T4" in prose

    def test_the_left_foot_prose_does_not_blame_the_source(self, generator):
        """GAITEX instruments both calcanei, and the left-foot prose must say so."""
        prose = generator.SITE_PROVENANCE["foot_l"]
        assert "Both feet are instrumented" in prose


class TestConfidenceComesFromTheSettings:
    """The column is a pure function of the settings file, and the file says what it means."""

    def test_a_backed_site_is_more_confident_than_a_declared_one(self, generator, settings):
        confidence = generator.confidence_by_site(settings)
        assert max(confidence[s] for s in DECLARED_PLACEMENT) < min(confidence[s] for s in WORN_UNIT_BACKED)

    def test_the_settings_define_the_invalid_cell_as_zero(self, settings):
        semantics = settings.document["imu_confidence"]
        assert semantics["invalid_cell"] == 0.0
        assert set(semantics["sites"]) == WORN_UNIT_BACKED | DECLARED_PLACEMENT

    def test_a_backed_site_whose_comparison_did_not_run_is_not_backed_on_that_take(self, generator, settings):
        """hans/rd has no XSens_Prop... unit for shank_r and ziri/rd's foot_l
        comparison failed; 0.6 there would say a unit stood behind the channel when none did."""
        comparison = {name: {"status": "ok"} for name in WORN_UNIT_BACKED}
        comparison["shank_r"] = {"status": "measured_counterpart_absent"}
        comparison["foot_l"] = {"status": "comparison_failed"}
        confidence = generator.confidence_by_site(settings, comparison)
        declared = settings.document["imu_confidence"]["backed_site_without_comparison_on_this_take"]
        assert confidence["shank_r"] == confidence["foot_l"] == declared
        assert confidence["shank_l"] == confidence["back_T4"] == confidence["foot_r"] == 0.6
        assert declared <= max(confidence[s] for s in DECLARED_PLACEMENT)

    def test_the_settings_carry_the_execution_constants_rather_than_the_code(self, settings):
        """Execution constants live in the settings file, not in code."""
        assert settings.forearm_max_gap_twist_deg == 90.0
        assert 0.0 < settings.forearm_label_swap_max_withdrawn_fraction < 0.5
        assert settings.frame_minimum_frames == 10
        assert settings.document["validation"]["relabel_offset_guard_deg"] == 120.0
        assert settings.version == "1.1.1"

    def test_the_small_mode_the_wrapper_stamps_is_the_one_the_spec_admits(self, generator):
        from soma_synth.contracts.qmd_unified8_smpl18_spec import RELABEL_TOLERANT_SMALL_MODES, SMALL_MODES

        assert generator.SMALL_MODE in SMALL_MODES
        assert generator.SMALL_MODE in RELABEL_TOLERANT_SMALL_MODES

    def test_replace_small_keeps_the_pair_identity_and_swaps_the_arrays(self, generator):
        frames = 5
        bundle = {"small": {
            "sensor_codes": np.array(list(generator.marker_small.SITE_ORDER)),
            "imu_orientation": np.zeros((frames, 8, 4), np.float32),
            "imu_acceleration": np.zeros((frames, 8, 3), np.float32),
            "imu_angular_velocity": np.zeros((frames, 8, 3), np.float32),
            "imu_valid_mask": np.ones((frames, 8), bool),
            "imu_confidence": np.full((frames, 8), 0.6, np.float32),
            "small_mode": np.str_("synthetic_from_smpl"),
            "mount_id": np.array([f"gaitex_unified8::{c}" for c in generator.marker_small.SITE_ORDER]),
            "pair_id": np.str_("pid"),
            "frame_count": np.int64(frames),
        }}
        replacement = {
            "imu_orientation": np.full((frames, 8, 4), np.nan, np.float32),
            "imu_acceleration": np.full((frames, 8, 3), np.nan, np.float32),
            "imu_angular_velocity": np.full((frames, 8, 3), np.nan, np.float32),
            "imu_valid_mask": np.zeros((frames, 8), bool),
            "imu_confidence": np.zeros((frames, 8), np.float32),
        }
        generator.replace_small(bundle, replacement)
        assert str(bundle["small"]["small_mode"]) == "synthetic_from_markers"
        assert str(bundle["small"]["mount_id"][0]) == "gaitex_marker::back_T4"
        assert str(bundle["small"]["pair_id"]) == "pid"
        assert np.isnan(bundle["small"]["imu_orientation"]).all()

    def test_replace_small_refuses_a_frame_count_that_disagrees(self, generator):
        bundle = {"small": {"sensor_codes": np.array(list(generator.marker_small.SITE_ORDER)),
                            "frame_count": np.int64(5)}}
        with pytest.raises(ValueError, match="frames"):
            generator.replace_small(bundle, {"imu_valid_mask": np.ones((4, 8), bool)})


class TestAttributionTravelsWithTheBundle:
    """GAITEX is CC BY 4.0 and its licence note conditions redistribution on the credit.

    72 artifacts shipped naming the wrong archive before this was checked anywhere.
    """

    def test_the_bundle_credits_gaitex_and_not_the_sibling_archive(self, generator):
        attribution = generator.ATTRIBUTION
        assert attribution["source_name"] == "gaitex"
        assert attribution["stable_source_id"] == "gaitex_munz_2025"
        assert "addbiomechanics" not in attribution["dataset_citation"].lower()

    def test_the_citation_names_the_zenodo_record_the_licence_asks_for(self, generator):
        citation = generator.ATTRIBUTION["dataset_citation"]
        assert "10.5281/zenodo.15729056" in citation
        for surname in ("Munz", "Spilz", "Oppel"):
            assert surname in citation


class TestWhatTheBundleRefusesToClaim:
    def test_the_missing_force_and_measured_inertial_streams_are_named(self, generator):
        unavailable = " ".join(generator.UNAVAILABLE).lower()
        assert "grf" in unavailable
        assert "accelerometer" in unavailable
        assert "gyroscope" in unavailable

    def test_the_absent_subject_measurements_are_named_as_placeholders(self, generator):
        """The archive writes mass=90 and height=-1 for everyone; those must not read as data."""
        unavailable = " ".join(generator.UNAVAILABLE).lower()
        assert "stature" in unavailable and "mass" in unavailable
        assert "placeholder" in unavailable

    def test_the_anthro_provenance_strings_do_not_claim_a_measurement(self, generator):
        """The emitter fills anthro.height with the SMPL rest stature and labels sex with PRISM's
        table for every source; the wrapper's strings must say estimated and declared, never
        measured."""
        assert generator.HEIGHT_PROVENANCE.startswith("estimated:")
        assert generator.SEX_PROVENANCE.startswith("declared:")
        for text in (generator.HEIGHT_PROVENANCE, generator.SEX_PROVENANCE):
            assert "measured" not in text.lower() or "not a measured" in text.lower()
            assert "prism" not in text.lower()

    def test_the_declared_zero_placements_are_named_as_unavailable_extrinsics(self, generator):
        unavailable = " ".join(generator.UNAVAILABLE)
        for site in DECLARED_PLACEMENT:
            assert site in unavailable

    def test_the_governance_draft_is_recorded_not_resolved_and_the_authorization_named(self, generator):
        """Two drafts were open; the internal-use authorization became a recorded decision on
        2026-09-06, so one draft remains and the decision is named beside it. The records are
        the parent project's (its research/decisions); that they exist there is checked there."""
        assert len(generator.RECORDED_CONFLICTS) == 1
        for path in generator.RECORDED_CONFLICTS:
            assert path.startswith("research/decisions/DRAFT_")
        assert len(generator.RECORDED_AUTHORIZATIONS) == 1
        for path in generator.RECORDED_AUTHORIZATIONS:
            assert path.startswith("research/decisions/")
            assert not path.startswith("research/decisions/DRAFT_")
        assert "research/decisions/DRAFT_gaitex_internal_use_authorization.md" not in (
            *generator.RECORDED_CONFLICTS, *generator.RECORDED_AUTHORIZATIONS)


class TestTheShippedDocumentation:
    """The description is written from a template in this file, so the template is the check."""

    def test_the_description_keeps_the_internal_only_banner(self, generator):
        """L4 finds this by substring, so a rewording must not be able to drop it."""
        source = GENERATOR.read_text(encoding="utf-8")
        assert "INTERNAL-ONLY" in source

    def test_the_description_tells_a_consumer_which_field_to_filter_on(self, generator):
        """A take whose source kinematics came apart must be findable without reading prose."""
        source = GENERATOR.read_text(encoding="utf-8")
        assert "solve_came_apart" in source
        assert "frames_outside_any" in source

    def test_the_description_tells_a_consumer_to_apply_the_mask(self, generator):
        """The first informative mask in the lineage: NaN under False, large finite, apply before filtering."""
        source = GENERATOR.read_text(encoding="utf-8")
        assert "NaN" in source and "imu_valid_mask" in source
        assert "Apply the mask" in source or "Apply " in source and "mask" in source

    def test_the_description_admits_the_gap_policy_bridges(self, generator):
        """'Nothing is interpolated' was false while fill_pose_gaps bridged 20 frames."""
        source = GENERATOR.read_text(encoding="utf-8")
        assert "Nothing is interpolated" not in source
        assert "interpolated_runs_bundle_rows" in source and "bridged" in source


class TestNoAbsolutePathReachesTheManifest:
    def test_the_prose_blocks_carry_no_drive_letter(self, generator):
        """The spec forbids it and L4 checks it; here it is checked before anything is written.

        Only the assembled prose blocks are inspected -- DATA_ROOT itself is a path and is
        meant to be one.
        """
        for site, prose in generator.SITE_PROVENANCE.items():
            assert not DRIVE_PATH.search(prose), site
        for line in generator.UNAVAILABLE:
            assert not DRIVE_PATH.search(line)
        for value in generator.ATTRIBUTION.values():
            assert not DRIVE_PATH.search(str(value))

    def test_the_settings_file_carries_no_drive_letter(self):
        text = SETTINGS.read_text(encoding="utf-8")
        assert not DRIVE_PATH.search(json.dumps(json.loads(text), ensure_ascii=False))
        json.loads(text)  # and it is the JSON the loader expects
