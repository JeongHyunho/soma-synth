"""Config-declared sensor sites."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soma_synth.sensors import site_registry as sr

REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "datasets"
    / "gaitex_sensor_sites_v1.yaml"
)


@pytest.fixture(scope="module")
def registry() -> sr.SiteRegistry:
    return sr.load_site_registry(REGISTRY_PATH)


def write_registry(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "sites.yaml"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


MINIMAL_SITE = {
    "markers": ["A", "B", "C"],
    "offset": {"kind": "plate_normal", "depth_m": 0.014, "in_plane_offset_m": [0.0, 0.0]},
    "inward_reference": {"markers": ["D"]},
    "position_provenance": "source_derived",
    "rotation_provenance": "source_derived",
}


def minimal_document(**site_overrides) -> dict:
    site = {**MINIMAL_SITE, **site_overrides}
    return {
        "config_id": "test_sites",
        "version": "1.0.0",
        "sites": {"s": site},
        "site_sets": {"only": ["s"]},
    }


class TestShippedRegistry:
    def test_both_channel_sets_are_declared_without_code_changes(
        self, registry: sr.SiteRegistry
    ) -> None:
        """The whole point: neither the canonical six nor the spec's eight is in code."""
        canonical = [site.name for site in registry.require_set("repo_canonical_six")]
        spec = [site.name for site in registry.require_set("external_spec_eight")]

        assert canonical == ["chest", "wrist_l", "wrist_r", "foot_l", "foot_r", "head"]
        assert spec == [
            "back_T4",
            "wrist_l",
            "wrist_r",
            "shank_l",
            "shank_r",
            "occiput",
            "foot_l",
            "foot_r",
        ]

    def test_the_registry_is_marked_non_authorizing(self, registry: sr.SiteRegistry) -> None:
        assert registry.activation_state == "non_authorizing"

    def test_plate_sites_carry_an_inward_reference(self, registry: sr.SiteRegistry) -> None:
        for name in ("chest", "shank_l", "shank_r", "foot_l", "foot_r"):
            site = registry.require(name)
            assert site.offset.kind == "plate_normal"
            assert site.inward_reference is not None
            assert site.inward_reference.markers

    def test_sites_without_a_gaitex_sensor_are_declared_unresolved_with_a_reason(
        self, registry: sr.SiteRegistry
    ) -> None:
        """The occiput and T4 offsets are genuinely open; the registry must say so."""
        for name in ("head", "occiput", "back_T4", "wrist_l", "wrist_r"):
            site = registry.require(name)
            assert not site.offset.is_resolved
            assert len(site.offset.unresolved_reason) > 40
            assert site.position_provenance == "unavailable"
            # Rotation is observed even where position is not.
            assert site.rotation_provenance == "source_derived"

    def test_the_head_and_occiput_share_a_rigid_body(self, registry: sr.SiteRegistry) -> None:
        assert registry.require("head").markers == registry.require("occiput").markers

    def test_the_chest_and_t4_share_a_rigid_body(self, registry: sr.SiteRegistry) -> None:
        assert registry.require("chest").markers == registry.require("back_T4").markers

    def test_only_instrumented_sites_declare_a_measured_counterpart(
        self, registry: sr.SiteRegistry
    ) -> None:
        """The validation mapping lives in config, not in scripts."""
        assert registry.measured_counterparts() == {
            "chest": "XSens_Sternum",
            "shank_l": "XSens_LowerLeg_Left",
            "shank_r": "XSens_LowerLeg_Right",
            "foot_l": "XSens_Foot_Left",
            "foot_r": "XSens_Foot_Right",
        }
        for name in ("head", "occiput", "back_T4", "wrist_l", "wrist_r"):
            assert registry.require(name).measured_counterpart is None

    def test_every_referenced_marker_exists_in_the_gaitex_set(
        self, registry: sr.SiteRegistry
    ) -> None:
        """Guards against a config that names markers the source does not have."""
        gaitex_markers = {
            "L_FAL", "L_FAX", "L_FCC", "L_FLE", "L_FME", "L_FOOT1", "L_FOOT2", "L_FOOT3",
            "L_FOOT4", "L_FTC", "L_HEAD", "L_HLE", "L_HUM", "L_IAS", "L_IPS", "L_RSP",
            "L_SAE", "L_SHIN1", "L_SHIN2", "L_SHIN3", "L_SHIN4", "L_TAM", "L_THI1",
            "L_THI2", "L_THI3", "L_THI4", "L_TTC", "L_USP", "MAI", "PELV1", "PELV2",
            "PELV3", "PELV4", "R_FAL", "R_FAX", "R_FCC", "R_FLE", "R_FME", "R_FOOT1",
            "R_FOOT2", "R_FOOT3", "R_FOOT4", "R_FTC", "R_HEAD", "R_HLE", "R_HUM", "R_IAS",
            "R_IPS", "R_RSP", "R_SAE", "R_SHIN1", "R_SHIN2", "R_SHIN3", "R_SHIN4", "R_TAM",
            "R_THI1", "R_THI2", "R_THI3", "R_THI4", "R_TOE1", "R_TOE2", "R_TOE3", "R_TTC",
            "R_USP", "SGL", "SJN", "THOR1", "THOR2", "THOR3", "THOR4",
        }
        referenced = set(sr.marker_union(list(registry.sites.values())))
        assert referenced <= gaitex_markers
        assert len(gaitex_markers) == 70


class TestLookupRefusesToGuess:
    def test_an_undeclared_site_raises(self, registry: sr.SiteRegistry) -> None:
        with pytest.raises(sr.SiteRegistryError, match="not declared"):
            registry.require("sternum")

    def test_an_undeclared_set_raises(self, registry: sr.SiteRegistry) -> None:
        with pytest.raises(sr.SiteRegistryError, match="not declared"):
            registry.require_set("all_the_sites")


class TestValidation:
    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(sr.SiteRegistryError, match="missing site registry"):
            sr.load_site_registry(tmp_path / "absent.yaml")

    def test_malformed_json_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "sites.yaml"
        path.write_text("config_id: not json\n", encoding="utf-8")
        with pytest.raises(sr.SiteRegistryError, match="not valid JSON"):
            sr.load_site_registry(path)

    def test_a_site_needs_three_markers(self, tmp_path: Path) -> None:
        path = write_registry(tmp_path, minimal_document(markers=["A", "B"]))
        with pytest.raises(sr.SiteRegistryError, match="at least 3 markers"):
            sr.load_site_registry(path)

    def test_a_repeated_marker_is_refused(self, tmp_path: Path) -> None:
        path = write_registry(tmp_path, minimal_document(markers=["A", "A", "B"]))
        with pytest.raises(sr.SiteRegistryError, match="repeats a marker"):
            sr.load_site_registry(path)

    def test_an_unknown_provenance_is_refused(self, tmp_path: Path) -> None:
        path = write_registry(tmp_path, minimal_document(position_provenance="probably_fine"))
        with pytest.raises(sr.SiteRegistryError, match="position_provenance"):
            sr.load_site_registry(path)

    def test_a_plate_offset_without_a_depth_is_refused(self, tmp_path: Path) -> None:
        """The hold forbids the module supplying a depth of its own."""
        path = write_registry(
            tmp_path,
            minimal_document(offset={"kind": "plate_normal", "in_plane_offset_m": [0.0, 0.0]}),
        )
        with pytest.raises(sr.SiteRegistryError, match="forbids a code-side default"):
            sr.load_site_registry(path)

    def test_a_plate_offset_without_an_inward_reference_is_refused(
        self, tmp_path: Path
    ) -> None:
        document = minimal_document()
        del document["sites"]["s"]["inward_reference"]
        path = write_registry(tmp_path, document)
        with pytest.raises(sr.SiteRegistryError, match="inward_reference"):
            sr.load_site_registry(path)

    def test_an_unresolved_offset_without_a_reason_is_refused(self, tmp_path: Path) -> None:
        path = write_registry(tmp_path, minimal_document(offset={"kind": "unresolved"}))
        with pytest.raises(sr.SiteRegistryError, match="must say why"):
            sr.load_site_registry(path)

    def test_an_unknown_offset_kind_is_refused(self, tmp_path: Path) -> None:
        path = write_registry(tmp_path, minimal_document(offset={"kind": "guess"}))
        with pytest.raises(sr.SiteRegistryError, match="offset kind"):
            sr.load_site_registry(path)

    def test_a_set_naming_an_undeclared_site_is_refused(self, tmp_path: Path) -> None:
        document = minimal_document()
        document["site_sets"]["broken"] = ["s", "ghost"]
        path = write_registry(tmp_path, document)
        with pytest.raises(sr.SiteRegistryError, match="undeclared sites"):
            sr.load_site_registry(path)


class TestInwardReferenceGeometry:
    def test_the_direction_points_from_the_plate_towards_the_landmarks(self) -> None:
        frames = 5
        centroid = np.zeros((frames, 3))
        landmarks = np.zeros((frames, 2, 3))
        landmarks[:, 0] = [0.0, 0.0, -0.30]
        landmarks[:, 1] = [0.0, 0.0, -0.50]

        direction = sr.inward_reference_world(centroid, landmarks)

        assert direction.shape == (frames, 3)
        assert (direction[:, 2] < 0).all()
        np.testing.assert_allclose(direction[0], [0.0, 0.0, -0.40])

    def test_occluded_landmarks_do_not_poison_the_average(self) -> None:
        centroid = np.zeros((3, 3))
        landmarks = np.full((3, 2, 3), np.nan)
        landmarks[:, 0] = [0.0, 0.0, -0.30]

        direction = sr.inward_reference_world(centroid, landmarks)

        np.testing.assert_allclose(direction[:, 2], -0.30)

    def test_a_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(sr.SiteRegistryError, match="does not match"):
            sr.inward_reference_world(np.zeros((4, 3)), np.zeros((5, 2, 3)))


class TestMarkerUnion:
    def test_the_union_deduplicates_and_keeps_order(self, registry: sr.SiteRegistry) -> None:
        sites = registry.require_set("external_spec_eight")
        union = sr.marker_union(sites)

        assert len(union) == len(set(union))
        assert "THOR1" in union
        assert "L_SHIN1" in union
        # Inward-reference markers count as dependencies too.
        assert "L_FLE" in union
