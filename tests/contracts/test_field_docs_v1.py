"""R0 — field_docs registry completeness: every spec field/enum has a one-line doc."""

import pytest

from soma_synth.contracts import field_docs_v1 as fd
from soma_synth.contracts import qmd_unified8_smpl18_spec as spec


def _spec_fields(artifact_name):
    art = spec.ARTIFACTS[artifact_name]
    keys = {k.name for k in art.core}
    for cap_keys in art.conditional.values():
        keys |= {k.name for k in cap_keys}
    return keys


def test_every_spec_field_documented_exactly():
    for name in spec.ARTIFACTS:
        keys = _spec_fields(name)
        docs = set(fd.FIELD_DOCS.get(name, {}))
        assert keys - docs == set(), f"{name}: undocumented fields {sorted(keys - docs)}"
        assert docs - keys == set(), f"{name}: docs for unknown fields {sorted(docs - keys)}"


def test_every_artifact_has_blurb():
    for name in spec.ARTIFACTS:
        assert fd.ARTIFACT_DOCS.get(name), f"{name}: missing artifact blurb"


def test_enum_values_documented():
    assert set(fd.ENUM_DOCS["small_mode"]) == spec.SMALL_MODES
    assert set(fd.ENUM_DOCS["axis_convention"]) == spec.AXIS_CONVENTIONS
    assert set(fd.ENUM_DOCS["subject_scope"]) == spec.SUBJECT_SCOPES


def test_summaries_are_nonempty_strings():
    for name in spec.ARTIFACTS:
        for field, text in fd.FIELD_DOCS[name].items():
            assert isinstance(text, str) and text.strip(), f"{name}.{field}: empty summary"


def test_summary_helper():
    assert fd.summary("small", "imu_orientation")
    assert fd.summary("large", "smpl_global_orientation_world")
    with pytest.raises(KeyError):
        fd.summary("small", "nonexistent")
    with pytest.raises(KeyError):
        fd.summary("no_such_artifact", "x")
