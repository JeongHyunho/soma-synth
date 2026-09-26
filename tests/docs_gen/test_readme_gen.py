"""R1/R2 — field-by-field README generator (per-dataset + top-level)."""

from pathlib import Path

from soma_synth import cli
from soma_synth.docs_gen import readme

from test_validation_checks import make_dataset


def test_dataset_readme_sections_and_determinism(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", source="amass", n=1)
    md = readme.render_dataset_readme(ds)
    assert "INTERNAL-ONLY" in md
    assert "qmd_unified8_smpl18" in md and "faithful-v2" in md
    assert "small_reference.npz" in md and "large_reference.npz" in md
    assert "imu_orientation" in md and "smpl_global_orientation_world" in md
    assert "CORE" in md
    assert "allow_pickle" in md
    assert "back_T4" in md and "SMPL joint" in md
    assert "```json" in md
    assert md == readme.render_dataset_readme(ds)  # deterministic


def test_dataset_readme_write(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    out = readme.write_dataset_readme(ds)
    assert out.exists() and out.name == "README.md"
    assert "INTERNAL-ONLY" in out.read_text(encoding="utf-8")


def test_top_level_readme_diffs(tmp_path: Path) -> None:
    a = make_dataset(tmp_path / "amass_x", source="amass", n=1)
    p = make_dataset(tmp_path / "prism_x", source="prism", small_mode="measured_physical", n=1)
    md = readme.render_top_level_readme([a, p])
    assert "qmd_unified8_smpl18" in md
    assert "amass" in md and "prism" in md
    assert "measured_physical" in md and "synthetic_from_smpl" in md
    assert "Common CORE fields" in md
    assert "development_reference" in md  # conditional presence matrix row


def test_cli_readme(tmp_path: Path) -> None:
    a = make_dataset(tmp_path / "amass_x", n=1)
    p = make_dataset(tmp_path / "prism_x", source="prism", small_mode="measured_physical", n=1)
    rc = cli.main(["readme", str(a), str(p), "--top-level-root", str(tmp_path)])
    assert rc == 0
    assert (a / "README.md").exists() and (p / "README.md").exists() and (tmp_path / "README.md").exists()
