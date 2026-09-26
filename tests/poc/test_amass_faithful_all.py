from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[2] / "scripts" / "poc" / "generate_amass_faithful_all.py"


def _mod():
    spec = importlib.util.spec_from_file_location("amass_all", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_tree(root):
    """Create a tiny fake AMASS tree; returns the amass root path."""
    amass = root / "extracted" / "amass"
    files = [
        ("CMU", "01", "01_02_poses.npz"),
        ("CMU", "01", "01_03_poses.npz"),
        ("KIT", "10", "Walk_poses.npz"),
        ("KIT", "11", "Run_poses.npz"),
        ("CMU", "02", "bad_seq_poses.npz"),   # build will fail for 'bad'
    ]
    for ds, sub, name in files:
        d = amass / ds / sub
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / name, poses=np.zeros((4, 156)), trans=np.zeros((4, 3)),
                 betas=np.zeros(16), gender=np.array("male"), mocap_framerate=np.array(120.0))
    return str(amass)


def _fake_build(spec):
    if "bad" in spec.sequence_id:
        raise ValueError("synthetic failure for %s" % spec.sequence_id)
    small = {"frame_count": np.int64(4), "imu": np.zeros((4, 8, 3), np.float32)}
    large = {"frame_count": np.int64(4), "jr": np.zeros((4, 18, 4), np.float32),
             "pelvis_position_world_aux": np.zeros((4, 3), np.float32)}
    anthro = {"joint_position": np.zeros((2, 22, 3), np.float32),
              "source_asset_id": spec.relative_path}
    manifest = {"identity": {"pair_id": "fake_%s" % spec.sequence_id},
                "deliverables": ["small_reference.npz", "large_reference.npz",
                                 "anthro_reference.npz", "smpl_root_translation.npz",
                                 "manifest.json"]}
    return small, large, anthro, manifest


def test_enumerate_finds_poses_and_identity(tmp_path):
    m = _mod()
    amass = _fake_tree(tmp_path)
    specs = m.enumerate_specs(amass)
    assert len(specs) == 5
    ids = {(s.dataset, s.subject_id, s.sequence_id) for s in specs}
    assert ("CMU", "amass_CMU_01", "01_02_poses") in ids
    assert ("KIT", "amass_KIT_10", "Walk_poses") in ids
    one = next(s for s in specs if s.sequence_id == "01_02_poses")
    assert one.relative_path == "extracted/amass/CMU/01/01_02_poses.npz"


def test_run_ok_skip_failed_accounting_and_resume(tmp_path):
    m = _mod()
    amass = _fake_tree(tmp_path)
    out = tmp_path / "out"
    specs = m.enumerate_specs(amass)
    idx = m.run(specs, str(out), build_fn=_fake_build, write_fn=m.write_outputs_generic)
    assert idx["counts"]["ok"] == 4
    assert idx["counts"]["failed"] == 1
    assert idx["counts"]["skipped"] == 0
    failed = [e for e in idx["takes"] if e["status"] == "failed"]
    assert len(failed) == 1 and "bad" in failed[0]["sequence_id"] and "reason" in failed[0]
    # resume: rerun -> the 4 good bundles validate and are not rebuilt, the bad one retried (fails
    # again). A resumed take is still an ok take in the INDEX (it validates and registers like any
    # other) and says so with `resumed`; recording it as skipped dropped it out of both.
    idx2 = m.run(specs, str(out), build_fn=_fake_build, write_fn=m.write_outputs_generic)
    assert idx2["counts"]["skipped"] == 0
    assert idx2["counts"]["resumed"] == 4
    assert idx2["counts"]["failed"] == 1
    assert idx2["counts"]["ok"] == 4
    resumed = [e for e in idx2["takes"] if e.get("resumed")]
    assert len(resumed) == 4 and all(e["pair_id"] and e["frames"] == 4 for e in resumed)


def test_index_entries_name_the_take_directory_and_the_bundle_has_five_files(tmp_path):
    """The retired bundle's INDEX had take_id/rel added after the fact and its root-translation
    sidecar came from a separate enrich pass; since 2026-09-15 the generator writes both."""
    m = _mod()
    amass = _fake_tree(tmp_path)
    out = tmp_path / "out"
    specs = m.enumerate_specs(amass)
    idx = m.run(specs, str(out), build_fn=_fake_build, write_fn=m.write_outputs_generic)
    assert idx["spec_id"] == "qmd_unified8_smpl18" and idx["source"] == "amass"
    ok = next(e for e in idx["takes"] if e["status"] == "ok")
    assert ok["take_id"] == ok["rel"] == ok["relative_path"]
    assert (out / ok["rel"] / "manifest.json").is_file()            # rel IS the take directory
    assert ok["source_relative_path"].startswith("extracted/amass/")
    assert {"frames", "pair_id"} <= set(ok)
    sidecar = out / ok["rel"] / "smpl_root_translation.npz"
    with np.load(sidecar, allow_pickle=False) as z:
        assert z["smpl_trans"].shape == (4, 3)
        assert str(z["source_asset_id"]) == ok["source_relative_path"]
        assert z["smpl_trans_provenance"].dtype.kind == "U"


def test_sharding_partitions_disjoint_and_covers_all(tmp_path):
    m = _mod()
    amass = _fake_tree(tmp_path)
    out = tmp_path / "out"
    specs = m.enumerate_specs(amass)
    i0 = m.run(specs, str(out), build_fn=_fake_build, write_fn=m.write_outputs_generic, shard=(0, 2))
    i1 = m.run(specs, str(out), build_fn=_fake_build, write_fn=m.write_outputs_generic, shard=(1, 2))
    seqs0 = {e["sequence_id"] for e in i0["takes"]}
    seqs1 = {e["sequence_id"] for e in i1["takes"]}
    assert seqs0.isdisjoint(seqs1)
    assert seqs0 | seqs1 == {s.sequence_id for s in specs}
    # each shard wrote its own shard index file
    assert (out / "INDEX_shard_0_of_2.json").exists()
    assert (out / "INDEX_shard_1_of_2.json").exists()


def test_merge_shards_into_index(tmp_path):
    m = _mod()
    amass = _fake_tree(tmp_path)
    out = tmp_path / "out"
    specs = m.enumerate_specs(amass)
    m.run(specs, str(out), build_fn=_fake_build, write_fn=m.write_outputs_generic, shard=(0, 2))
    m.run(specs, str(out), build_fn=_fake_build, write_fn=m.write_outputs_generic, shard=(1, 2))
    merged = m.merge_shards(str(out), total_specs=len(specs))
    assert merged["counts"]["ok"] + merged["counts"]["failed"] + merged["counts"]["skipped"] == 5
    assert len(merged["takes"]) == 5
    assert (out / "INDEX.json").exists()
    on_disk = json.loads((out / "INDEX.json").read_text(encoding="utf-8"))
    assert on_disk["counts"] == merged["counts"]


def test_non_motion_shape_files_are_excluded_not_failed(tmp_path):
    m = _mod()
    amass = _fake_tree(tmp_path)
    # add a shape.npz (gender+betas, no poses) beside a motion sequence
    shp = Path(amass) / "CMU" / "01" / "shape.npz"
    np.savez(shp, gender=np.array("male"), betas=np.zeros(16))
    specs = m.enumerate_specs(amass)
    assert any(s.sequence_id == "shape" for s in specs)          # enumerated, not dropped
    idx = m.run(specs, str(tmp_path / "out"), build_fn=_fake_build, write_fn=m.write_outputs_generic)
    excluded = [e for e in idx["takes"] if e["status"] == "excluded_non_motion"]
    assert len(excluded) == 1 and excluded[0]["sequence_id"] == "shape"
    assert idx["counts"]["excluded_non_motion"] == 1
    assert idx["counts"]["failed"] == 1                          # the 'bad' motion seq still fails
    assert idx["counts"]["ok"] == 4                              # shape not counted as ok/failed


def test_dataset_description_has_internal_only_banner(tmp_path):
    m = _mod()
    out = tmp_path / "out"
    out.mkdir()
    idx = {"counts": {"ok": 4, "failed": 1, "skipped": 0, "total": 5}, "takes": []}
    m.write_dataset_description(str(out), idx)
    md = (out / "DATA_DESCRIPTION_EN.md").read_text(encoding="utf-8")
    assert m.BANNER_TOKEN in md
    assert "synthetic" in md.lower()
    assert os.path.exists(out / "DATA_DESCRIPTION_EN.md")
    # ADR-0040 D3 retired the second English description; emitting it again would put a bundle
    # back into the state where two files claimed the same job.
    assert not os.path.exists(out / "DATASET_DESCRIPTION_EN.md")
