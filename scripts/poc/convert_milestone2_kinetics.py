"""Milestone-2 kinetics by inverse dynamics, one npz per take: a check of the conversion alone.

The bundle generator (generate_milestone2_unified8.py) carries the same kinetics into the Large
artifact; this tool writes them on their own, with the per-take quality summary, so the inverse
dynamics can be examined without building bundles. Vectors are in the treadmill laboratory frame
turned so the trial heading is +X (+Y left, +Z up); the CoP stays on the treadmill surface.

    python scripts/poc/convert_milestone2_kinetics.py --smpl <m2_smpl_fits.zip|dir> \
        --kinetics <m2_kinetics.zip|dir> --out <folder> [--only S002_level_100mps_lv0_trial_01] [--jobs 4]

Numbers come from configs/kinetics/milestone2_inverse_dynamics_v1.yaml.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from soma_synth.kinetics import jcs, milestone2, smpl_body  # noqa: E402
from soma_synth.pipeline import paths  # noqa: E402

CONFIG = ROOT / "configs" / "kinetics" / "milestone2_inverse_dynamics_v1.yaml"
SMPL_SUFFIX = "_overground_trim.npz"
KIN_SUFFIX = "_kinetics.npz"


class _Source:
    """Members of a zip or files of a folder, by take name."""

    def __init__(self, path: str, suffix: str):
        if zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
            names = [n for n in self.zip.namelist() if n.endswith(suffix)]
        else:
            self.zip = None
            names = [str(p) for p in Path(path).rglob("*" + suffix)]
        self.members = {re.sub(re.escape(suffix) + "$", "", n.replace("\\", "/").rsplit("/", 1)[-1]): n
                        for n in names}

    def load(self, take: str):
        n = self.members[take]
        data = self.zip.read(n) if self.zip else Path(n).read_bytes()
        return np.load(io.BytesIO(data), allow_pickle=False)


def heading_to_x(R_pelvis: np.ndarray, fwd_rest: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation about Z taking the trial's mean pelvis heading to +X, and that heading (deg)."""
    f = np.einsum("tij,j->ti", R_pelvis, fwd_rest)[:, :2]
    yaw = np.arctan2(f[:, 1].mean(), f[:, 0].mean())
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), float(np.degrees(yaw))


def convert_take(take: str, smpl_src: str, kin_src: str, out_dir: str, model_path: str, cfg: dict) -> dict:
    t0 = time.time()
    S = _Source(smpl_src, SMPL_SUFFIX).load(take)
    K = _Source(kin_src, KIN_SUFFIX).load(take)
    model = smpl_body.load_model(model_path)
    k = milestone2.compute(S, K, model, cfg)
    _, J = smpl_body.shaped(model, np.asarray(S["betas"], np.float64))
    Rp = Rotation.from_rotvec(np.asarray(S["global_orient_treadmill"], np.float64)).as_matrix()
    Q, yaw = heading_to_x(Rp, jcs.canonical_axes(J)[1])
    w = milestone2.to_frame(k, Q, np.zeros(3))
    T = len(k.grf)
    out = {
        "joint_names": np.array(k.joint_names, dtype="U"),
        "grf": w["grf"].astype(np.float32),
        "cop": w["cop"][..., :2].astype(np.float32),
        "free_moment": k.free_moment.astype(np.float32),
        "foot_contact": k.foot_contact,
        "grf_valid_mask": k.grf_valid,
        "kinetics_valid_mask": k.valid,
        "max_rotation_step_deg": k.max_rotation_step_deg.astype(np.float32),
        "cop_to_foot_distance_m": k.cop_to_foot_distance.astype(np.float32),
        "com_acceleration": w["com_acceleration"].astype(np.float32),
        "angular_momentum": w["angular_momentum"].astype(np.float32),
        "joint_torque": k.joint_torque.astype(np.float32),
        "joint_power": k.joint_power.astype(np.float32),
        "joint_moments_jcs": k.joint_moments_jcs.astype(np.float32),
        "joint_power_total": k.joint_power_total.astype(np.float32),
        "inverse_dynamics_residual": np.c_[w["residual_force"], w["residual_moment"]].astype(np.float32),
        "segment_mass_kg": k.segment_mass_kg.astype(np.float32),
        "body_mass_force_plate_kg": np.float32(k.body_mass_force_plate_kg),
        "body_mass_recorded_kg": np.float32(float(K["subj_weight"])),
        "timestamps_s": np.arange(T) / float(cfg["sampling_rate_hz"]),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(out_dir) / f"{take}_id_kinetics.npz", **out)
    return summarise(take, k, cfg, float(K["subj_weight"]), yaw, time.time() - t0)


def summarise(take: str, k: "milestone2.TakeKinetics", cfg: dict, recorded_mass: float,
              heading_deg: float, seconds: float) -> dict:
    """Per-take quality numbers over the kinetics-valid frames away from the filter edges."""
    T = len(k.grf)
    e = int(cfg["quality"]["edge_frames"])
    keep = np.zeros(T, bool)
    keep[e:T - e] = True
    sl = keep & k.valid
    m = k.body_mass_force_plate_kg
    bw = m * -float(cfg["gravity_m_s2"][2])

    def rms(x):
        return float(np.sqrt(np.mean(np.sum(x[sl] ** 2, axis=-1))))

    ank = k.joint_names.index("right_ankle")
    glitch = k.max_rotation_step_deg > float(cfg["quality"]["max_rotation_step_deg"])
    return {
        "take": take,
        "frames": T,
        "excluded": cfg["excluded_takes"].get(take),
        "body_mass_force_plate_kg": round(m, 2),
        "body_mass_recorded_kg": recorded_mass,
        "mesh_volume_l": round(k.mesh_volume_m3 * 1000, 2),
        "heading_yaw_lab_deg": round(heading_deg, 2),
        "contact_fraction": [round(float(c), 3) for c in k.foot_contact.mean(0)],
        "grf_invalid_frames": int((~k.grf_valid).sum()),
        "fit_glitch_frames": int(glitch.sum()),
        "kinetics_invalid_frames": int((~k.valid).sum()),
        "first_glitch_frame": int(np.flatnonzero(glitch)[0]) if glitch.any() else None,
        "residual_force_rms_pct_bw": round(rms(k.residual_force) / bw * 100, 2),
        "residual_moment_rms_n_m_per_kg": round(rms(k.residual_moment) / m, 3),
        "right_ankle_min_primary_n_m_per_kg": round(float(np.percentile(k.joint_torque[sl, ank], 1) / m), 3),
        "right_ankle_max_power_w_per_kg": round(float(np.percentile(k.joint_power_total[sl, ank], 99) / m), 3),
        "seconds": round(seconds, 1),
    }


def _convert_or_report(*args) -> dict:
    """One take; a failure is recorded against the take instead of stopping the others."""
    try:
        return convert_take(*args)
    except Exception as exc:  # noqa: BLE001 - reported per take and in the exit code
        return {"take": args[0], "error": f"{type(exc).__name__}: {exc}"}


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def model_path(cfg: dict) -> Path:
    bm = cfg["body_model"]
    return paths.data_root() / bm["folder_under_data_root"] / bm["file"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smpl", required=True, help="m2_smpl_fits.zip or the folder of *_overground_trim.npz")
    ap.add_argument("--kinetics", required=True, help="m2_kinetics.zip or the folder of *_kinetics.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append", default=[], help="take name (repeatable)")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    model = str(model_path(cfg))
    takes = sorted(_Source(args.smpl, SMPL_SUFFIX).members)
    missing = sorted(set(takes) - set(_Source(args.kinetics, KIN_SUFFIX).members))
    if missing:
        raise SystemExit(f"takes without plates: {missing}")
    if args.only:
        unknown = sorted(set(args.only) - set(takes))
        if unknown:
            raise SystemExit(f"unknown takes: {unknown}")
        takes = args.only
    if Path(args.out).exists() and any(Path(args.out).iterdir()):
        raise SystemExit(f"{args.out} is not empty; choose an empty folder")

    jobs = [(t, args.smpl, args.kinetics, args.out, model, cfg) for t in takes]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            rows = list(ex.map(_convert_or_report, *zip(*jobs)))
    else:
        rows = [_convert_or_report(*j) for j in jobs]
    for r in rows:
        print(json.dumps(r))
    (Path(args.out) / "SUMMARY.json").write_text(json.dumps(
        {"config": cfg, "body_model": model, "takes": rows}, indent=1), encoding="utf-8")
    failed = [r["take"] for r in rows if "error" in r]
    if failed:
        print(f"{len(failed)} take(s) failed: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
