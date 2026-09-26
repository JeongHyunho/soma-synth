"""The spec reads the dataset profile registry on first use, not at import.

``MANIFEST_CORE_FIELDS`` is computed on first use, so importing the site table does not parse
``configs/datasets/dataset_profiles_v1.yaml``. These tests pin that the value, the way it is
reached, and the module's public names did not change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from soma_synth.contracts import dataset_profiles
from soma_synth.contracts import qmd_unified8_smpl18_spec as spec

REPO = Path(__file__).resolve().parents[2]


def test_importing_the_spec_does_not_parse_the_registry():
    """A clean interpreter, so a registry some other test already parsed cannot hide a regression."""
    code = (
        "import json\n"
        "from soma_synth.contracts import dataset_profiles\n"
        "from soma_synth.contracts import qmd_unified8_smpl18_spec as spec\n"
        "before = dataset_profiles.default_registry.cache_info().currsize\n"
        "fields = spec.MANIFEST_CORE_FIELDS\n"
        "after = dataset_profiles.default_registry.cache_info().currsize\n"
        "print(json.dumps({'before': before, 'after': after, 'n': len(fields)}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["before"] == 0, "importing the spec parsed the registry"
    assert report["after"] == 1 and report["n"] > 0


def test_the_value_is_the_registry_universal_manifest_keys():
    fields = spec.MANIFEST_CORE_FIELDS
    assert isinstance(fields, frozenset)
    assert fields is dataset_profiles.default_registry().universal.manifest_keys
    assert fields == dataset_profiles.load().universal.manifest_keys   # a fresh parse agrees
    assert spec.MANIFEST_CORE_FIELDS is fields                          # computed once


def test_it_is_still_importable_by_name():
    from soma_synth.contracts.qmd_unified8_smpl18_spec import (
        MANIFEST_CORE_FIELDS,
    )

    assert MANIFEST_CORE_FIELDS is spec.MANIFEST_CORE_FIELDS


def test_the_public_names_are_unchanged():
    """``dir()`` lists the lazy name, and the change added no other public name."""
    public = {name for name in dir(spec) if not name.startswith("_")}
    assert "MANIFEST_CORE_FIELDS" in public
    assert public == {
        "ARTIFACTS", "AXIS_CONVENTIONS", "ArtifactSpec", "CAPABILITIES", "DTYPE_CLASSES",
        "INDEX_FIELDS", "INDEX_TAKE_FIELDS", "KeySpec", "MANIFEST_CORE_FIELDS", "Mapping",
        "RELABEL_TOLERANT_SMALL_MODES", "SITE_ORDER", "SITE_TO_JOINT", "SMALL_MODES", "SPEC_ID",
        "SPEC_VERSION", "SUBJECT_SCOPES", "SpecError", "annotations", "artifact_names",
        "check_arrays", "dataclass", "dataset_profiles", "derive_capabilities", "dtype_class",
        "expected_deliverables", "field",
    }


def test_an_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="NO_SUCH_FIELD"):
        spec.NO_SUCH_FIELD  # noqa: B018 - the access is the test
