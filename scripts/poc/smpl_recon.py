"""Reconstruction for the Small/Large viewer — enrichment (one-time) vs. viewing (npz-only).

Two clearly separated phases (kept apart on purpose):

* ENRICH (one-time, needs the PRISM pkl): add the subject's shape + gender + the true SMPL
  translation to `development_reference.npz`, all `source_derived`. Run once per bundle.
    - betas[10]         : real SMPL shape (pkl smpl_params.betas[0]; constant across the take)
    - gender            : 'male'
    - smpl_trans[T,3]   : the FAITHFUL SMPL root translation (pkl smpl_params.trans, windowed).
                          NOTE: this is NOT large["pelvis_position_world_aux"] — that is the pelvis
                          JOINT world position and differs from the SMPL `trans` by up to ~0.1 m
                          (time-varying), so it must not be used as the AMASS `trans`.

* VIEW (npz-only): reconstruct SMPL-X animation from the THREE distributed npz only.
    - pose  : inverse SMPL FK of large["smpl_global_orientation_prism_world"] (LOSSLESS).
    - trans : development_reference["smpl_trans"] (faithful; set during ENRICH).
    - shape : development_reference["betas"] + ["gender"].
  The pkl is NEVER read in this phase; all three pair_id values are validated.

Shape caveat: SMPL betas are placed into the SMPL-X shape space (different PCA basis;
the official SMPL-X project says the spaces are incompatible and must not simply be copied). The
result is an **approximate, unvalidated cross-space subject-shape visualization**, not a validated
real SMPL-X shape. INTERNAL-ONLY.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

import viewer_paths  # same scripts/poc dir on sys.path; the standard library only


def _generator():
    """generate_prism_faithful, loaded on first use: its SMPL-24 FK and its PRISM pickle loader serve
    the reconstruction check and ENRICH, which run in the teammate's Python (viewer_launch.py runs
    this file with it). The viewer app imports this module inside Blender, whose Python carries
    neither PyYAML nor protobuf, and the generator pulls both in through soma_synth and smpl18;
    the VIEW helpers the app uses never call this."""
    import generate_prism_faithful

    return generate_prism_faithful


#: The one take ENRICH reads and the window the single-take bundle was cut from. The digest is
#: the take's sha256 (53,412,579 bytes).
ENRICH_TAKE = ("subj001", "take002.pkl")
ENRICH_TAKE_SHA256 = "5addb1c377e967115d0f65fbeae76616eb0f50495678fd99292f070d3a423eb2"
ENRICH_WINDOW = (1000, 2000)                      # [start, end) source frames at 100 Hz
ENRICH_FRAMES = ENRICH_WINDOW[1] - ENRICH_WINDOW[0]
SHAPE_CAVEAT = ("approximate unvalidated cross-space shape: SMPL betas placed in SMPL-X shape "
                "space (different PCA basis); not a validated real SMPL-X fit")


# --------------------------------------------------------------------------
# quaternion helper (inverse of generate_prism_faithful.rotmat_to_quat_wxyz)
# --------------------------------------------------------------------------
def quat_wxyz_to_rotmat(q_wxyz):
    """[...,4] (w,x,y,z) -> [...,3,3]. wxyz -> xyzw for scipy."""
    q = np.asarray(q_wxyz, np.float64)
    q_xyzw = np.concatenate((q[..., 1:4], q[..., 0:1]), axis=-1)
    return Rotation.from_quat(q_xyzw.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


# --------------------------------------------------------------------------
# VIEW phase: inverse FK  global-orientation quats -> SMPL local axis-angle pose
# --------------------------------------------------------------------------
def reconstruct_pose_from_global(global_quat_wxyz):
    """[T,24,4] world global quats -> poses [T,24,3] SMPL local axis-angle.

    Exact inverse of generate_prism_faithful.smpl24_global_rotation:
      local[0]   = G[0];   local[j>0] = G[parent(j)]^T @ G[j].
    """
    G = quat_wxyz_to_rotmat(global_quat_wxyz)             # [T,24,3,3]
    T = G.shape[0]
    parents = _generator().SMPL24_PARENTS
    local = np.empty_like(G)
    local[:, 0] = G[:, 0]
    for j in range(1, 24):
        local[:, j] = np.swapaxes(G[:, parents[j]], -1, -2) @ G[:, j]
    return Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(T, 24, 3)


def geodesic_angle_deg(Ra, Rb):
    rel = np.swapaxes(Ra, -1, -2) @ Rb
    ang = np.degrees(np.linalg.norm(
        Rotation.from_matrix(_generator().project_so3(rel.reshape(-1, 3, 3))).as_rotvec(), axis=1))
    return float(ang.max()), float(ang.mean())


def verify_reconstruction(large):
    """Confirm recovered pose re-FKs to the original global orientation (lossless SO(3))."""
    key = global_rotation_key(large)
    if not key:
        raise SystemExit(f"large_reference.npz has no [T,24,4] '{_ROTATION_PREFIX}*' array")
    gq = np.asarray(large[key], np.float64)                                   # [T,24,4]
    qn = np.linalg.norm(gq, axis=2)
    poses = reconstruct_pose_from_global(gq)
    G_orig = quat_wxyz_to_rotmat(gq)
    G_recon = _generator().smpl24_global_rotation(poses)
    max_deg, mean_deg = geodesic_angle_deg(G_orig, G_recon)
    return poses, {
        "frames": int(gq.shape[0]),
        "input_quat_norm_max_dev": float(np.max(np.abs(qn - 1.0))),
        "global_orientation_recon_max_deg": max_deg,
        "global_orientation_recon_mean_deg": mean_deg,
    }


# --------------------------------------------------------------------------
# ENRICH phase: real betas + gender + FAITHFUL SMPL trans from the PRISM pkl
# --------------------------------------------------------------------------
def enrich_source_pkl() -> str:
    """The take ENRICH reads, under the PRISM source folder ($SOMA_SOURCE_ROOT/prism)."""
    # viewer_paths: the same rule as the pipeline's paths.source_dir("prism"), without soma_synth
    root = (None if os.environ.get("SOMA_SOURCE_ROOT", "").strip()
            else viewer_paths.data_root(hint="or set SOMA_SOURCE_ROOT to the folder holding prism/"))
    return os.path.join(viewer_paths.source_folder(root, "prism"), *ENRICH_TAKE)


def load_shape_and_trans_from_pkl():
    """(betas[10], gender, smpl_trans_win[T,3]). Uses the real (non-zero) pkl shape + true trans."""
    raw = _generator().load_prism_safe(enrich_source_pkl())
    sp = raw["smpl_params"]
    betas_all = np.asarray(sp["betas"], np.float64)                 # [N,10]
    var = float(np.max(np.abs(betas_all - betas_all[0:1])))
    betas = betas_all.mean(axis=0)                                  # [10] (== betas[0], var~0)
    gender = str(np.asarray(sp["gender"]).item()) if "gender" in sp else "neutral"
    start, end = ENRICH_WINDOW
    smpl_trans_win = np.asarray(sp["trans"], np.float64)[start:end]  # [T,3] true SMPL root translation
    return betas.astype(np.float64), gender, smpl_trans_win.astype(np.float64), var


def assert_bundle_matches_pkl(run_dir):
    """Refuse to enrich a bundle not generated from the ENRICH take — guards --run against silently
    injecting the fixed subj001/take002 shape+trans into a different subject's bundle."""
    take = "/".join(ENRICH_TAKE)
    mpath = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(mpath):
        raise SystemExit(f"[enrich] no manifest.json in {run_dir}; cannot verify the bundle matches {take}")
    with open(mpath, encoding="utf-8") as f:
        m = json.load(f)
    src = (m.get("source") or {}).get("source_asset_sha256_take")
    if src != ENRICH_TAKE_SHA256:
        raise SystemExit(f"[enrich] bundle source hash {src} != {take} {ENRICH_TAKE_SHA256}; "
                         f"refusing to enrich a bundle not generated from {take}")
    fc = int((m.get("window") or {}).get("frame_count", -1))
    if fc != ENRICH_FRAMES:
        raise SystemExit(f"[enrich] bundle frame_count {fc} != {ENRICH_FRAMES}")


def enrich_dev(run_dir, dry_run=False):
    """Add betas([10]) + gender + smpl_trans([T,3]) + provenance to development_reference.npz.

    Idempotent and non-destructive: backs the file up once, never touches small/large. Refuses to
    enrich a bundle whose manifest source hash does not match the ENRICH take."""
    assert_bundle_matches_pkl(run_dir)
    dev_path = os.path.join(run_dir, "development_reference.npz")
    betas, gender, smpl_trans, var = load_shape_and_trans_from_pkl()
    if var > 1e-4:
        raise RuntimeError(f"betas not constant across take (max var {var}); representative [10] unsafe")

    z = np.load(dev_path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    T = int(np.asarray(data["timestamps_s"]).shape[0])
    if smpl_trans.shape != (T, 3):
        raise RuntimeError(f"smpl_trans shape {smpl_trans.shape} != ({T},3)")
    already = "betas" in data and "smpl_trans" in data

    data["betas"] = betas.astype(np.float32)                       # [10]
    data["gender"] = np.array(gender, dtype=object)
    data["betas_num"] = np.int64(betas.shape[0])
    data["smpl_trans"] = smpl_trans.astype(np.float32)             # [T,3] FAITHFUL SMPL root translation
    data["betas_provenance"] = np.array(
        "source_derived:prism_pkl smpl_params.betas[0] (SMPL 10-dim, constant across take)", dtype=object)
    data["smpl_trans_provenance"] = np.array(
        "source_derived:prism_pkl smpl_params.trans (windowed); the true SMPL root translation, "
        "NOT large.pelvis_position_world_aux (which is the pelvis joint world position)", dtype=object)
    data["shape_model"] = np.array(SHAPE_CAVEAT, dtype=object)

    if dry_run:
        return dev_path, betas, gender, already, "(dry-run)"

    backup = os.path.join(run_dir, "development_reference.pre_betas.bak.npz")
    if not os.path.exists(backup) and not already:
        shutil.copy2(dev_path, backup)
    np.savez(dev_path, **data)

    zz = np.load(dev_path, allow_pickle=True)
    assert zz["betas"].shape == (10,) and np.all(np.isfinite(zz["betas"])), "betas re-validate failed"
    assert zz["smpl_trans"].shape == (T, 3) and np.all(np.isfinite(zz["smpl_trans"])), "smpl_trans re-validate"
    assert str(zz["gender"]) == gender and str(zz["pair_id"]) != "", "gender/pair_id re-validate"
    return dev_path, betas, gender, already, ("backup=" + backup if os.path.exists(backup) else "no-backup")


# --------------------------------------------------------------------------
# VIEW phase helpers: read shape/trans from the npz only; validate pair_id
# --------------------------------------------------------------------------
def _require_shape(dev):
    """The shape reference must name the subject's body; the root translation is resolved separately
    because published bundles keep it out of the subject-scoped anthro npz."""
    for f in ("betas", "gender"):
        if f not in getattr(dev, "files", []):
            raise SystemExit(f"shape reference npz is missing '{f}'. For a locally generated bundle run "
                             f"enrichment once:  python smpl_recon.py --run <dir> --enrich")


def validate_pair_ids(small, large, dev):
    ids = {"small": str(small["pair_id"]), "large": str(large["pair_id"]), "dev": str(dev["pair_id"])}
    if len(set(ids.values())) != 1:
        raise SystemExit(f"pair_id mismatch across npz: {ids}")
    return ids["small"]


# The per-run reference carrying betas / smpl_trans is called `anthro_reference.npz` in newer dataset
# bundles and `development_reference.npz` in older ones. Readers probe for it instead of hard-coding a
# name, so a bundle of either vintage loads without a viewer change. Newer name wins if both exist.
SHAPE_REFERENCE_NAMES = ("anthro_reference.npz", "development_reference.npz")


def shape_reference_path(run_dir):
    """Path to the run's shape/translation reference npz, or "" if the run carries neither name."""
    for name in SHAPE_REFERENCE_NAMES:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    return ""


def load_shape_reference(run_dir):
    p = shape_reference_path(run_dir)
    if not p:
        raise SystemExit(f"no shape reference npz in {run_dir} (looked for {list(SHAPE_REFERENCE_NAMES)})")
    return np.load(p, allow_pickle=True)


ROOT_TRANSLATION_SIDECAR = "smpl_root_translation.npz"

# Datasets prefix the per-joint GLOBAL rotation array with their own source name
# (`..._prism_world`, `..._world`, ...). It is the same quantity, so it is resolved by shape rather
# than by one hard-coded name — any contract-following dataset loads without a code change.
IN_PLACE_TRANSLATION = "in-place (no root translation published with this run)"
GLOBAL_ROTATION_KEYS = ("smpl_global_orientation_prism_world", "smpl_global_orientation_world")
_ROTATION_PREFIX = "smpl_global_orientation"


def global_rotation_key(large):
    """Name of the [T,24,4] global-rotation array in a large reference, or "" when absent."""
    files = list(getattr(large, "files", []))
    for k in GLOBAL_ROTATION_KEYS:
        if k in files:
            return k
    for k in files:
        if k.startswith(_ROTATION_PREFIX):
            a = np.asarray(large[k])
            if a.ndim == 3 and a.shape[1:] == (24, 4):
                return k
    return ""


def _trans_from(path_or_zip, key, frames):
    """`key` from an npz, but only if it is a finite [frames,3] array."""
    try:
        z = np.load(path_or_zip, allow_pickle=True) if isinstance(path_or_zip, str) else path_or_zip
        if key not in getattr(z, "files", []):
            return None
        trans = np.asarray(z[key], np.float64)
    except Exception:
        return None
    if trans.shape != (frames, 3) or not np.all(np.isfinite(trans)):
        return None
    return trans


def root_translation(run_dir, shape_ref, frames):
    """(trans[T,3] float64, provenance) — the FAITHFUL SMPL root translation for a run.

    Published runs carry no root translation at all, and it cannot be derived from what they do carry:
    `large["pelvis_position_world_aux"]` is PRISM's pelvis marker (`imu_gt["Pelvis"]["pos_world"]`), a
    different quantity from the SMPL root that differs by a *time-varying* ~0.1 m, so no constant
    offset reconciles them. Sources are therefore tried in order of authority:

      1. `smpl_trans` on the shape reference   — locally enriched bundles
      2. `smpl_trans` on the large reference   — the canonical home once the dataset publishes it
      3. the `smpl_root_translation.npz` sidecar written by enrich_root_translation.py from the run's
         own source pickle (verified to reproduce the enriched values exactly)
      4. `trans` on a sibling SMPL-X animation npz, which emit_smplx_anim() wrote from `smpl_trans`
    """
    trans = _trans_from(shape_ref, "smpl_trans", frames)
    if trans is not None:
        return trans, "shape_reference.smpl_trans"

    large = os.path.join(run_dir, "large_reference.npz")
    trans = _trans_from(large, "smpl_trans", frames) if os.path.isfile(large) else None
    if trans is not None:
        return trans, "large_reference.smpl_trans"

    sidecar = os.path.join(run_dir, ROOT_TRANSLATION_SIDECAR)
    trans = _trans_from(sidecar, "smpl_trans", frames) if os.path.isfile(sidecar) else None
    if trans is not None:
        return trans, f"{ROOT_TRANSLATION_SIDECAR}:smpl_trans (recovered from the source recording)"

    for name in sorted(os.listdir(run_dir)):
        if name.endswith(("_smplx.npz", "_smpl.npz")):
            trans = _trans_from(os.path.join(run_dir, name), "trans", frames)
            if trans is not None:
                return trans, f"{name}:trans (== smpl_trans by construction)"

    # Last resort: show the motion IN PLACE. The pose and the IMU frames are still exact — only the
    # trajectory through the world is missing, so it may be displayed as long as it is labelled.
    # `pelvis_position_world_aux` is deliberately NOT guessed at here: in AMASS bundles it happens to
    # equal the root translation, but in PRISM it is the pelvis MARKER and differs by a time-varying
    # ~0.1 m. Same field name, different quantity — enrich_root_translation.py resolves it per dataset
    # and writes an explicit sidecar instead.
    return np.zeros((frames, 3), np.float64), IN_PLACE_TRANSLATION


def emit_smplx_anim(run_dir, out_path, fps=100):
    """NPZ-ONLY: reconstructed SMPL(24) -> SMPL-X AMASS npz for the Meshcapade addon.

    pose from large (lossless); trans + betas + gender from development_reference.npz (enriched).
    poses[T,165]: [:,0:66]=SMPL global(3)+body21(63) copied (mirrors prism_to_smplx.py); 66:165=0.
    betas[16]: real SMPL betas[0:10] placed in [0:10] -- APPROXIMATE cross-space (see SHAPE_CAVEAT).
    """
    small = np.load(os.path.join(run_dir, "small_reference.npz"), allow_pickle=True)
    large = np.load(os.path.join(run_dir, "large_reference.npz"), allow_pickle=True)
    dev = load_shape_reference(run_dir)
    _require_shape(dev)
    pair_id = validate_pair_ids(small, large, dev)

    poses, vinfo = verify_reconstruction(large)
    poses72 = poses.reshape(poses.shape[0], 72)
    T = poses72.shape[0]
    betas = np.asarray(dev["betas"], np.float64)
    gender = str(dev["gender"])
    trans, _ = root_translation(run_dir, dev, T)                 # FAITHFUL SMPL trans
    if trans.shape != (T, 3):
        raise SystemExit(f"smpl_trans shape {trans.shape} != ({T},3)")

    poses_x = np.zeros((T, 165), np.float64)
    poses_x[:, 0:66] = poses72[:, 0:66]
    betas16 = np.zeros(16, np.float64)
    nb = min(10, betas.shape[0])
    betas16[0:nb] = betas[:nb]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, trans=trans.astype(np.float64), gender=np.array(gender),
             mocap_framerate=np.float64(fps), betas=betas16, poses=poses_x)
    return out_path, {"frames": T, "gender": gender, "pair_id": pair_id,
                      "recon": vinfo, "shape_note": SHAPE_CAVEAT}


def smpl_inputs_from_run(run_dir, fps=100):
    """NPZ-ONLY, validated: dict(betas[10], quats[T,24,4] global w,x,y,z, trans[T,3], gender, fps, pair_id).

    The exact native-SMPL viewer inputs — betas + trans from the enriched dev npz, per-joint GLOBAL
    rotations from large. No pkl. Used both by --emit-smpl and by the interactive app's run-switch rebuild.
    """
    small = np.load(os.path.join(run_dir, "small_reference.npz"), allow_pickle=True)
    large = np.load(os.path.join(run_dir, "large_reference.npz"), allow_pickle=True)
    dev = load_shape_reference(run_dir)
    _require_shape(dev)
    pair_id = validate_pair_ids(small, large, dev)
    rot_key = global_rotation_key(large)
    if not rot_key:
        raise SystemExit(f"{run_dir}: large_reference.npz has no [T,24,4] '{_ROTATION_PREFIX}*' array "
                         f"(has {list(large.files)})")
    quats = np.asarray(large[rot_key], np.float64)                                # [T,24,4] w,x,y,z
    T = quats.shape[0]
    betas = np.asarray(dev["betas"], np.float64)[:10]      # AMASS ships 16; SMPL uses the first 10
    trans, trans_src = root_translation(run_dir, dev, T)
    if quats.shape != (T, 24, 4):
        raise SystemExit(f"quats shape {quats.shape} != ({T},24,4)")
    if trans.shape != (T, 3):
        raise SystemExit(f"smpl_trans shape {trans.shape} != ({T},3)")
    return {"betas": betas, "quats": quats, "trans": trans, "gender": str(dev["gender"]),
            "fps": int(fps), "pair_id": pair_id, "trans_source": trans_src, "rotation_key": rot_key}


def emit_smpl_anim(run_dir, out_path, fps=100):
    """NPZ-ONLY: package the EXACT native-SMPL viewer inputs (no cross-space shape hack) to a npz.

    The native-SMPL Blender rig (smpl_rig) turns the subject's own betas into the EXACT rest mesh and
    poses it by these rotations — the real subject shape, not an approximation.
    """
    large = np.load(os.path.join(run_dir, "large_reference.npz"), allow_pickle=True)
    _, vinfo = verify_reconstruction(large)             # validates quats: unit-norm + lossless inverse FK
    d = smpl_inputs_from_run(run_dir, fps)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, betas=d["betas"], quats=d["quats"], trans=d["trans"].astype(np.float64),
             gender=np.array(d["gender"]), fps=np.int64(fps), pair_id=np.array(d["pair_id"]))
    return out_path, {"frames": d["quats"].shape[0], "gender": d["gender"], "pair_id": d["pair_id"],
                      "recon": vinfo, "shape_note": "exact native SMPL shape from subject betas (no cross-space)"}


def main():
    ap = argparse.ArgumentParser(description="Enrich dev npz (one-time) or reconstruct SMPL-X anim (npz-only).")
    ap.add_argument("--run", default=None,
                    help="take folder holding the small/large/shape-reference npz (required)")
    ap.add_argument("--enrich", action="store_true", help="one-time: add betas+gender+smpl_trans to dev npz (reads pkl)")
    ap.add_argument("--emit-smplx", default=None, help="npz-only: write SMPL-X AMASS anim npz here")
    ap.add_argument("--emit-smpl", default=None, help="npz-only: write EXACT native-SMPL anim npz here")
    ap.add_argument("--verify-recon", action="store_true", help="report inverse-FK lossless metrics")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # no default: name the take folder
    args.run = viewer_paths.require(args.run, "--run", "the take folder")

    report = {"run": args.run}
    if args.enrich:
        dev_path, betas, gender, already, note = enrich_dev(args.run, dry_run=args.dry_run)
        report["enrich"] = {"dev_path": dev_path, "gender": gender,
                            "betas10": [round(float(x), 4) for x in betas],
                            "was_already_enriched": bool(already), "note": note}
    if args.verify_recon:
        large = np.load(os.path.join(args.run, "large_reference.npz"), allow_pickle=True)
        _, vinfo = verify_reconstruction(large)
        report["verify_recon"] = vinfo
    if args.emit_smplx:
        _, info = emit_smplx_anim(args.run, args.emit_smplx)
        report["emit_smplx"] = {"out": args.emit_smplx, **info}
    if args.emit_smpl:
        _, info = emit_smpl_anim(args.run, args.emit_smpl)
        report["emit_smpl"] = {"out": args.emit_smpl, **info}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
