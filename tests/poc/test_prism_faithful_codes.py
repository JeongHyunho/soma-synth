from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "poc" / "generate_prism_faithful.py"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_prism_faithful", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docx_large_codes_alias_the_existing_kinematics_without_copying() -> None:
    generator = _load_generator_module()
    pelvis_position = np.zeros((3, 3), dtype=np.float32)
    pelvis_orientation = np.zeros((3, 4), dtype=np.float32)
    joint_rotations = np.zeros((3, 15, 4), dtype=np.float32)
    large = {
        "pelvis_position": pelvis_position,
        "pelvis_orientation": pelvis_orientation,
        "regional_orientation_parent_from_child": joint_rotations,
    }

    compatible = generator.add_docx_code_aliases(large)

    assert compatible["joint_positions"] is pelvis_position
    assert compatible["joint_rotations"] is joint_rotations
    assert compatible["pelvis_orientation"] is pelvis_orientation
    assert "joint_positions(root)" not in compatible
    assert "joint_rotations(root)" not in compatible


def test_docx_compatibility_metadata_does_not_claim_unverified_semantics() -> None:
    """The generator claims code-name and shape compatibility with the qmd spec and nothing more.

    The status was `partial_code_name_compatibility_only` while the codes followed the DOCX
    specification (reports/validation/2026-08-04_prism_small_large_code_audit.md in the parent
    project); the generator moved to the qmd code names before it was first committed, records
    the DOCX aliases as superseded, and every PRISM manifest since carries
    `code_name_and_shape_compatibility_only`. What must not change is the intent: each semantic
    limit still says what is not certified, and the codes it cannot fill are named as such.
    """
    generator = _load_generator_module()

    metadata = generator.docx_code_compatibility_metadata()

    assert metadata["status"] == "code_name_and_shape_compatibility_only"
    assert "large_qualified_mapping" not in metadata
    assert "superseded" in metadata["historical_docx"]["note"]
    limits = metadata["semantic_limits"]
    assert set(limits) == {"small", "root_velocity", "joint_rotation", "joint_velocity"}
    assert "미인증" in limits["small"]                   # units / mount extrinsics not certified
    assert "미확정" in limits["root_velocity"]           # heading normalisation undecided
    assert "ISB JCS" in limits["joint_rotation"] and "미적용" in limits["joint_rotation"]
    assert set(metadata["unavailable_codes"]) == {"grf", "cop", "joint_torques"}
    assert all("미생성" in reason for reason in metadata["unavailable_codes"].values())


def test_docx_code_name_aliases_serialize_as_flat_npz_keys() -> None:
    generator = _load_generator_module()
    large = {
        "pelvis_position": np.zeros((3, 3), dtype=np.float32),
        "pelvis_orientation": np.zeros((3, 4), dtype=np.float32),
        "regional_orientation_parent_from_child": np.zeros((3, 15, 4), dtype=np.float32),
    }
    compatible = generator.add_docx_code_aliases(large)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "large_reference.npz"
        np.savez(path, **compatible)
        with np.load(path, allow_pickle=False) as saved:
            assert saved["joint_positions"].shape == (3, 3)
            assert saved["joint_rotations"].shape == (3, 15, 4)
            assert "joint_positions(root)" not in saved.files
            assert "joint_rotations(root)" not in saved.files
