"""Milestone-2 kinetics by inverse dynamics: SMPL fits + split-belt treadmill force plates.

For every take this computes, in the treadmill laboratory frame (the inertial frame the plates
measured in), the Large kinetics of large/03_kinetics.qmd: per-foot GRF, CoP and free moment,
whole-body COM acceleration and angular momentum, and the joint moment and power of the 17
moving joints by Newton-Euler inverse dynamics on the SMPL body (soma_synth.kinetics). Vectors
are then rotated to the spec's global frame G (+X forward, +Y left, +Z up); JCS moments and
powers do not depend on the frame.

    python scripts/poc/convert_milestone2_kinetics.py --smpl <m2_smpl_fits.zip|dir> \
        --kinetics <m2_kinetics.zip|dir> --out <folder> [--only S002_level_100mps_lv0_trial_01] [--jobs 4]

Numbers come from configs/kinetics/milestone2_inverse_dynamics_v1.yaml.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from soma_synth.kinetics import bsip, forceplate, inverse_dynamics as idyn, jcs, smpl_body  # noqa: E402
from soma_synth.pipeline import paths  # noqa: E402

CONFIG = ROOT / "configs" / "kinetics" / "milestone2_inverse_dynamics_v1.yaml"
SMPL_SUFFIX = "_overground_trim.npz"
KIN_SUFFIX = "_kinetics.npz"


class _Source:
    """Members of a zip or files of a folder, by take name."""

    def __init__(self, path: str, suffix: str):
        self.path, self.suffix = path, suffix
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


def _heading_rotation(R_pelvis, fwd_rest):
    """Rotation about Z taking the trial's mean pelvis heading to +X."""
    f = np.einsum("tij,j->ti", R_pelvis, fwd_rest)[:, :2]
    yaw = np.arctan2(f[:, 1].mean(), f[:, 0].mean())
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), float(np.degrees(yaw))


def convert_take(take: str, smpl_src: str, kin_src: str, out_dir: str, model_path: str, cfg: dict) -> dict:
    t0 = time.time()
    S = _Source(smpl_src, SMPL_SUFFIX).load(take)
    K = _Source(kin_src, KIN_SUFFIX).load(take)
    fs = float(cfg["sampling_rate_hz"])
    dt = 1.0 / fs
    if abs(float(S["fps"]) - fs) > 1e-6 or abs(float(K["fps"]) - fs) > 1e-6:
        raise ValueError(f"{take}: sampling rate is not {fs} Hz")
    T = S["trans"].shape[0]
    if K["grf_left"].shape[0] != T or int(K["trim_start_idx"]) != int(S["trim_start_idx"]):
        raise ValueError(f"{take}: plates and SMPL fit are not frame-aligned")
    f = cfg["filter"]
    cut, order = float(f["cutoff_hz"]), int(f["order"])
    g = np.asarray(cfg["gravity_m_s2"], np.float64)

    model = smpl_body.load_model(model_path)
    v, J = smpl_body.shaped(model, S["betas"].astype(np.float64))

    pl = cfg["plates"]
    F_raw = {s: K[pl[s]["force"]].astype(np.float64) for s in ("left", "right")}
    mass_fp = float(-(F_raw["left"][:, 2] + F_raw["right"][:, 2]).mean() / -g[2])
    mass_rec = float(K["subj_weight"])
    seg = bsip.segment_inertia_from_mesh(v, model.faces, model.weights, mass_fp,
                                         float(cfg["segments"]["voxel_size_m"]))

    R, P = smpl_body.fk(model, J, S["global_orient_treadmill"], S["body_pose"], S["trans_treadmill"])
    q = cfg["quality"]
    step = idyn.rotation_step_deg(R).max(axis=1)
    glitch = step > float(q["max_rotation_step_deg"])
    pad = int(q["glitch_pad_frames"])
    kin_valid = ~np.convolve(glitch.astype(float), np.ones(2 * pad + 1), mode="same").astype(bool)
    Rf = idyn.smooth_rotations(R, cut, fs, order)
    Pf = idyn.lowpass(P, cut, fs, order)
    kin = idyn.body_kinematics(Rf, Pf, J, seg.com, dt)

    loads, plates, cop_dist = [], {}, {}
    feet = {"left": (7, 10), "right": (8, 11)}
    for side in ("left", "right"):
        M_raw = K[pl[side]["moment"]].astype(np.float64) * float(pl["moment_scale_to_n_m"])
        load = forceplate.plate_load(idyn.lowpass(F_raw[side], cut, fs, order),
                                     idyn.lowpass(M_raw, cut, fs, order),
                                     pl[side]["origin_m"], float(pl["contact_threshold_n"]))
        ank, toe = feet[side]
        body = forceplate.assign_foot_body(load, Pf[:, ank], Pf[:, toe], ank, toe)
        loads.append(idyn.ExternalLoad(load.force, load.moment_at_origin, body))
        plates[side] = load
        mid = 0.5 * (Pf[:, ank] + Pf[:, toe])
        cop_dist[side] = np.where(load.contact, np.linalg.norm((load.cop - mid)[:, :2], axis=1), np.nan)

    wr = idyn.joint_wrenches(model.parents, seg.mass, seg.inertia, kin, loads, g)
    wb = idyn.whole_body(seg.mass, seg.inertia, kin)

    cal = jcs.calibrate(J)
    Mj = wr.moment[:, cal.child]
    mom_jcs = jcs.project(cal, Rf, Mj)
    om_jcs = jcs.project(cal, Rf, kin.omega[:, cal.child] - kin.omega[:, cal.parent])
    idx = np.arange(len(cal.names))
    torque = mom_jcs[:, idx, cal.primary]
    power = torque * om_jcs[:, idx, cal.primary]
    power_total = (Mj * (kin.omega[:, cal.child] - kin.omega[:, model.parents[cal.child]])).sum(-1)

    # Laboratory frame -> spec global frame G (+X forward along the trial heading, +Z up).
    up, fwd_rest, _ = jcs.canonical_axes(J)
    Q, yaw_deg = _heading_rotation(Rf[:, 0], fwd_rest)
    rot = lambda x: np.einsum("ij,...j->...i", Q, x)  # noqa: E731
    grf = np.stack([rot(plates[s].force) for s in ("left", "right")], 1)
    cop3 = np.stack([rot(plates[s].cop) for s in ("left", "right")], 1)
    free = np.stack([plates[s].free_moment for s in ("left", "right")], 1)
    contact = np.stack([plates[s].contact for s in ("left", "right")], 1)
    dist = np.stack([cop_dist[s] for s in ("left", "right")], 1)

    grf_valid = ~(np.nan_to_num(dist, nan=0.0) > float(q["cop_to_foot_max_m"])).any(1)
    res_F, res_M = wr.force[:, 0], wr.moment[:, 0]
    e = int(q["edge_frames"])
    keep = np.zeros(T, bool)
    keep[e:T - e] = True
    sl = keep & kin_valid & grf_valid
    bw = mass_fp * -g[2]
    excluded = cfg["excluded_takes"].get(take)

    out = {
        "joint_names": np.array(cal.names, dtype="U"),
        "grf": grf.astype(np.float32),
        "cop": cop3[..., :2].astype(np.float32),
        "free_moment": free.astype(np.float32),
        "foot_contact": contact,
        "grf_valid_mask": grf_valid,
        "kinetics_valid_mask": kin_valid & grf_valid,
        "max_rotation_step_deg": step.astype(np.float32),
        "cop_to_foot_distance_m": dist.astype(np.float32),
        "com_acceleration": rot(wb.com_acceleration).astype(np.float32),
        "angular_momentum": rot(wb.angular_momentum).astype(np.float32),
        "joint_torque": torque.astype(np.float32),
        "joint_power": power.astype(np.float32),
        "joint_moments_jcs": mom_jcs.astype(np.float32),
        "joint_angular_velocity_jcs_rad_s": om_jcs.astype(np.float32),
        "joint_power_total": power_total.astype(np.float32),
        "joint_torque_primary_component": cal.primary.astype(np.int8),
        "id_residual_force": rot(res_F).astype(np.float32),
        "id_residual_moment": rot(res_M).astype(np.float32),
        "segment_mass_kg": seg.mass.astype(np.float32),
        "body_mass_force_plate_kg": np.float32(mass_fp),
        "body_mass_recorded_kg": np.float32(mass_rec),
        "lab_to_global_rotation": Q.astype(np.float32),
        "timestamps_s": np.arange(T) * dt,
        "frame_note": np.array(
            "G: +X trial heading, +Y left, +Z up, origin at the treadmill laboratory origin; CoP on the "
            "treadmill surface (walking in place); internal (parent-on-child) joint moments in N m, "
            "projected on ISB JCS axes; joint_torque/joint_power = primary component"),
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(Path(out_dir) / f"{take}_id_kinetics.npz", **out)

    def rms(x):
        return float(np.sqrt(np.nanmean(np.sum(x[sl] ** 2, axis=-1))))

    ank = cal.names.index("right_ankle")
    summary = {
        "take": take,
        "frames": T,
        "excluded": excluded,
        "body_mass_force_plate_kg": round(mass_fp, 2),
        "body_mass_recorded_kg": mass_rec,
        "mesh_volume_l": round(seg.volume * 1000, 2),
        "heading_yaw_lab_deg": round(yaw_deg, 2),
        "contact_fraction": [round(float(c), 3) for c in contact.mean(0)],
        "grf_invalid_frames": int((~grf_valid).sum()),
        "fit_glitch_frames": int(glitch.sum()),
        "kinetics_invalid_frames": int((~(kin_valid & grf_valid)).sum()),
        "first_glitch_frame": int(np.flatnonzero(glitch)[0]) if glitch.any() else None,
        "residual_force_rms_pct_bw": round(rms(res_F) / bw * 100, 2),
        "residual_moment_rms_n_m": round(rms(res_M), 2),
        "residual_moment_rms_n_m_per_kg": round(rms(res_M) / mass_fp, 3),
        "right_ankle_min_primary_n_m_per_kg": round(float(np.percentile(torque[sl, ank], 1) / mass_fp), 3),
        "right_ankle_max_power_w_per_kg": round(float(np.percentile(power_total[sl, ank], 99) / mass_fp), 3),
        "seconds": round(time.time() - t0, 1),
    }
    return summary


def _convert_or_report(*args) -> dict:
    """One take; a failure is recorded against the take instead of stopping the others."""
    try:
        return convert_take(*args)
    except Exception as exc:  # noqa: BLE001 - reported per take and in the exit code
        return {"take": args[0], "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smpl", required=True, help="m2_smpl_fits.zip or the folder of *_overground_trim.npz")
    ap.add_argument("--kinetics", required=True, help="m2_kinetics.zip or the folder of *_kinetics.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append", default=[], help="take name (repeatable)")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    bm = cfg["body_model"]
    model_path = str(paths.data_root() / bm["folder_under_data_root"] / bm["file"])
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

    jobs = [(t, args.smpl, args.kinetics, args.out, model_path, cfg) for t in takes]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            rows = list(ex.map(_convert_or_report, *zip(*jobs)))
    else:
        rows = [_convert_or_report(*j) for j in jobs]
    for r in rows:
        print(json.dumps(r))
    (Path(args.out) / "SUMMARY.json").write_text(json.dumps(
        {"config": cfg, "body_model": model_path, "takes": rows}, indent=1), encoding="utf-8")
    failed = [r["take"] for r in rows if "error" in r]
    if failed:
        print(f"{len(failed)} take(s) failed: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
