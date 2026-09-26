"""Experimental-catalog registration + incremental verification ledger.

The ledger makes dataset expansion not re-verify from scratch: a take is trusted iff its content digest,
spec_version, generator digest, and validator digest all match a prior PASS.
"""

import json
from pathlib import Path

import numpy as np

from soma_synth.datasets import catalog
from soma_synth.validation.runner import validate

from test_validation_checks import make_dataset, make_take

_DELIVERABLES = [
    "small_reference.npz",
    "large_reference.npz",
    "anthro_reference.npz",
    "smpl_root_translation.npz",
    "manifest.json",
]


# --------------------------------------------------------------- digests
def test_take_digest_deterministic_and_change_sensitive(tmp_path: Path) -> None:
    td = make_take(tmp_path / "ds", "t0")
    d1 = catalog.take_digest(td, _DELIVERABLES)
    assert len(d1) == 64 and d1 == catalog.take_digest(td, _DELIVERABLES)
    (td / "manifest.json").write_text('{"changed": 1}', encoding="utf-8")
    assert catalog.take_digest(td, _DELIVERABLES) != d1


def test_module_digest_change_sensitive(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("x = 1\n", encoding="utf-8")
    d1 = catalog.module_digest([p])
    p.write_text("x = 2\n", encoding="utf-8")
    assert catalog.module_digest([p]) != d1


# --------------------------------------------------------------- ledger
def test_ledger_fresh_only_for_matching_key_and_pass(tmp_path: Path) -> None:
    ledger = catalog.ValidationLedger(tmp_path / "ledger.json")
    key = catalog.FreshnessKey("dig", "faithful-v2", "gen", "val")
    assert not ledger.is_fresh("t0", key)
    ledger.record("t0", key, "PASS")
    assert ledger.is_fresh("t0", key)
    # any component change -> stale
    assert not ledger.is_fresh("t0", catalog.FreshnessKey("dig2", "faithful-v2", "gen", "val"))
    assert not ledger.is_fresh("t0", catalog.FreshnessKey("dig", "faithful-v2", "gen", "val2"))
    # FAIL is never fresh
    ledger.record("t1", key, "FAIL")
    assert not ledger.is_fresh("t1", key)


def test_ledger_save_load_roundtrip(tmp_path: Path) -> None:
    lp = tmp_path / "ledger.json"
    key = catalog.FreshnessKey("d", "faithful-v2", "g", "v")
    ledger = catalog.ValidationLedger(lp)
    ledger.record("t0", key, "PASS")
    ledger.save()
    assert json.loads(lp.read_text(encoding="utf-8"))["schema"] == catalog.SCHEMA_LEDGER
    assert catalog.ValidationLedger(lp).is_fresh("t0", key)


# --------------------------------------------------------------- runner incremental
def test_runner_incremental_trusts_unchanged(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    lp = tmp_path / "ledger.json"
    ledger = catalog.ValidationLedger(lp)
    r1 = validate(ds, level="L2", ledger=ledger)
    assert r1.ok and r1.takes_checked == 2 and r1.takes_trusted == 0
    ledger.save()

    r2 = validate(ds, level="L2", ledger=catalog.ValidationLedger(lp))
    assert r2.ok and r2.takes_trusted == 2 and r2.takes_checked == 0


def test_runner_incremental_rechecks_changed_take(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    lp = tmp_path / "ledger.json"
    ledger = catalog.ValidationLedger(lp)
    validate(ds, level="L2", ledger=ledger)
    ledger.save()

    z = dict(np.load(ds / "amass-take000" / "small_reference.npz", allow_pickle=True))
    z["imu_confidence"] = (z["imu_confidence"] * 0.9).astype(np.float32)
    np.savez(ds / "amass-take000" / "small_reference.npz", **z)

    r = validate(ds, level="L2", ledger=catalog.ValidationLedger(lp))
    assert r.takes_checked == 1 and r.takes_trusted == 1


def test_runner_incremental_full_forces_recheck(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=2)
    lp = tmp_path / "ledger.json"
    ledger = catalog.ValidationLedger(lp)
    validate(ds, level="L2", ledger=ledger)
    ledger.save()
    r = validate(ds, level="L2", ledger=catalog.ValidationLedger(lp), full=True)
    assert r.takes_checked == 2 and r.takes_trusted == 0


# --------------------------------------------------------------- asset catalog
def test_register_dataset_appends_v1_assets(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    catdir = tmp_path / "catalog"
    catdir.mkdir()
    assets = catalog.register_dataset(
        catdir, ds, source="amass", spec_version="faithful-v2", data_root=tmp_path
    )
    ac = catalog.load_asset_catalog(catdir)
    assert ac["schema"] == catalog.SCHEMA_ASSET
    ids = [a["asset_id"] for a in ac["assets"]]
    assert any(i.startswith("qmd_unified8_smpl18:faithful-v2:amass:") and i.endswith(":small") for i in ids)
    a0 = ac["assets"][0]
    assert {"asset_id", "role", "relative_path", "sha256", "bytes", "source", "spec_id", "spec_version"} <= set(a0)
    # relative path is SOMA_DATA_ROOT-relative (no drive), forward slashes
    assert ":" not in a0["relative_path"][:3] and "\\" not in a0["relative_path"]
    # amass has no development_reference asset
    assert not any(i.endswith(":development") for i in ids)
    # source subdir created
    assert (catdir / "amass" / "source_manifest.json").exists()
    assert len(assets) == len(ac["assets"])


def test_supersede_spec_version(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "amass_x", n=1)
    catdir = tmp_path / "catalog"
    catdir.mkdir()
    catalog.register_dataset(catdir, ds, source="amass", spec_version="faithful-v1", data_root=tmp_path)
    catalog.register_dataset(catdir, ds, source="amass", spec_version="faithful-v2", data_root=tmp_path)
    n = catalog.supersede_spec_version(catdir, "faithful-v1", "faithful-v2")
    ac = catalog.load_asset_catalog(catdir)
    v1 = [a for a in ac["assets"] if a["spec_version"] == "faithful-v1"]
    assert n == len(v1) and n > 0 and all(a.get("superseded") for a in v1)
    assert all(not a.get("superseded") for a in ac["assets"] if a["spec_version"] == "faithful-v2")


def test_register_dataset_prism_includes_development(tmp_path: Path) -> None:
    ds = tmp_path / "prism_x"
    ds.mkdir()
    make_take(ds, "prism-t0", source="prism", small_mode="measured_physical")
    # add the PRISM-only development_reference so it is a complete measured take
    T = 20
    np.savez(
        ds / "prism-t0" / "development_reference.npz",
        namespace=np.array("development_reference/x"),
        grf_feet_source_native=np.zeros((T, 2, 3), np.float32),
        grf_combined_source_native=np.zeros((T, 3), np.float32),
        cop_feet_world=np.zeros((T, 2, 3), np.float32),
        cop_combined_world=np.zeros((T, 3), np.float32),
        foot_contact_mask=np.zeros((T, 2), bool),
        vertical_force_provenance=np.array("measured"),
        cop_provenance=np.array("source_derived"),
        contacts_provenance=np.array("source_derived"),
        usage=np.array("development_reference_only"),
        pair_id=np.array("pid"),
        timestamps_s=(np.arange(T) * 0.01).astype(np.float64),
    )
    index = {
        "spec_id": "qmd_unified8_smpl18", "spec_version": "faithful-v2", "dataset_dirname": "prism_x",
        "source": "prism", "artifact_class": "experimental_non_candidate", "distribution_scope": "internal_only",
        "generated_utc": "2026-08-18T00:00:00+00:00", "counts": {"ok": 1, "failed": 0, "excluded": 0, "total": 1},
        "complete": True,
        "takes": [{"take_id": "prism-t0", "pair_id": "pid", "status": "ok", "frames": 20, "relative_path": "prism-t0"}],
    }
    (ds / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    (ds / "DATA_DESCRIPTION_EN.md").write_text("INTERNAL-ONLY\n", encoding="utf-8")

    catdir = tmp_path / "catalog"
    catdir.mkdir()
    catalog.register_dataset(catdir, ds, source="prism", spec_version="faithful-v2", data_root=tmp_path)
    ids = [a["asset_id"] for a in catalog.load_asset_catalog(catdir)["assets"]]
    assert any(i.endswith(":development") for i in ids)
