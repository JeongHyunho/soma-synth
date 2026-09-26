"""This repository's own smpl18 profiles validate, and their referenced files exist.

`configs/smpl18/` holds the profiles for sources whose layout and field names cannot be published
with the package: they are authored here and published to `$SHARED_DATASET_PATH/smpl18/` for the
runner to find. The package checks the profiles it ships; these are ours to check.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from smpl18.profile import Profile

REPO = pathlib.Path(__file__).resolve().parents[2]
PROFILES = sorted((REPO / "configs" / "smpl18" / "profiles").glob("*.yaml"))
PACKAGE_SETTINGS = REPO / "packages" / "smpl18" / "configs" / "settings" / "default.yaml"


def test_there_are_profiles_to_check() -> None:
    assert PROFILES, "configs/smpl18/profiles holds no profile"


@pytest.mark.parametrize("path", PROFILES, ids=[path.stem for path in PROFILES])
def test_profile_validates_and_its_references_resolve(path: pathlib.Path) -> None:
    profile = Profile.load(path)
    assert profile.id == path.stem
    for entry in profile.referenced_files():
        assert entry.path.is_file(), f"{entry.role} -> {entry.path}"
        assert len(entry.sha256) == 64


@pytest.mark.parametrize("path", PROFILES, ids=[path.stem for path in PROFILES])
def test_profile_uses_the_engine_settings_the_package_ships(path: pathlib.Path) -> None:
    roles = {entry.role: entry for entry in Profile.load(path).referenced_files()}
    assert roles["settings[0]"].path == PACKAGE_SETTINGS.resolve()
    loaded = yaml.safe_load(PACKAGE_SETTINGS.read_text(encoding="utf-8"))
    assert loaded["schema"] == "smpl18_settings_v1"
