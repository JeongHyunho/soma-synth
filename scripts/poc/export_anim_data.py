"""Extract data for the Blender animation: 8-joint 3D positions + 6-IMU orientations + foot
GRF/contact.

Takes the imu_gt 3D positions in window [1000,2000) from PRISM take002.pkl (loaded safely), combines
them with the IMU orientations and GRF from the faithful output npz, and saves a small anim_data.npz
(INTERNAL-ONLY).

    python export_anim_data.py --run <take folder> [--pkl <take002.pkl>] [--data-root <dir>]

--run is required: the PRISM faithful take folder holding small_reference.npz and
development_reference.npz; anim_data.npz is written there. Without --pkl, prism/subj001/take002.pkl
is read from the source folder ($SOMA_SOURCE_ROOT, else <--data-root or $SOMA_DATA_ROOT>/extracted).
Every input is checked first, and only then is anything written.
A bundle's takes are written only by its generator, so --run inside a bundle or corpus (a folder
above holds INDEX.json), the lineage container, an evidence folder or a source folder is refused
(viewer_paths.writable_folder): copy the take folder to a scratch folder and pass the copy.
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer_paths  # noqa: E402

W0, W1 = 1000, 2000
#: What --run must hold; checked before anything is written.
INPUTS = ("small_reference.npz", "development_reference.npz")

# skeleton joints order
JOINTS = ["pelvis", "head", "knee_l", "knee_r", "foot_l", "foot_r", "wrist_l", "wrist_r"]
GT_KEY = {"pelvis": "Pelvis", "head": "Head", "knee_l": "L_Knee", "knee_r": "R_Knee",
          "foot_l": "L_Foot", "foot_r": "R_Foot", "wrist_l": "L_Wrist", "wrist_r": "R_Wrist"}
# bones (parent,child) index into JOINTS  (approx skeleton from 8 markers)
BONES = [("pelvis", "head"), ("pelvis", "knee_l"), ("knee_l", "foot_l"),
         ("pelvis", "knee_r"), ("knee_r", "foot_r"), ("head", "wrist_l"), ("head", "wrist_r")]


class _NumpyOnly(pickle.Unpickler):
    _A = {"_reconstruct", "scalar", "ndarray", "dtype"}
    def find_class(self, m, n):
        if m.startswith("numpy") and n in self._A:
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f"BLOCKED {m}.{n}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="anim_data.npz for blender_prism_anim.py (PRISM take002)")
    ap.add_argument("--run", default=None,
                    help="take folder holding small_reference.npz and development_reference.npz "
                         "(required; anim_data.npz is written there)")
    ap.add_argument("--pkl", default=None,
                    help="PRISM take pickle (default <source root>/prism/subj001/take002.pkl)")
    ap.add_argument("--data-root", default=None,
                    help="finds the source root when --pkl and $SOMA_SOURCE_ROOT are not given; "
                         "default $SOMA_DATA_ROOT")
    args = ap.parse_args(argv)
    run = viewer_paths.require(args.run, "--run", "the take folder",
                               writes=viewer_paths.COPY_THE_TAKE)
    viewer_paths.writable_folder(run, "--run")          # anim_data.npz goes there: never a bundle
    viewer_paths.require_files(run, INPUTS)
    pkl = args.pkl
    if not pkl:
        root = (None if os.environ.get("SOMA_SOURCE_ROOT", "").strip()
                else viewer_paths.data_root(args.data_root,
                                            hint="or pass --data-root, or --pkl"))
        pkl = os.path.join(viewer_paths.source_folder(root, "prism"), "subj001", "take002.pkl")
    if not os.path.isfile(pkl):
        raise SystemExit(f"no PRISM take pickle at {pkl}; pass --pkl")

    with open(pkl, "rb") as fh:
        raw = _NumpyOnly(fh).load()
    gt = raw["imu_gt"]

    def pos(site):
        return np.asarray(gt[site]["pos_world"], np.float64)[W0:W1]

    J = np.stack([pos(GT_KEY[j]) for j in JOINTS], axis=1)  # [F,8,3]
    bone_idx = np.array([[JOINTS.index(a), JOINTS.index(b)] for a, b in BONES], np.int64)

    # IMU: order = small_reference sensor_codes [chest,wrist_l,wrist_r,foot_l,foot_r,head]
    small = np.load(os.path.join(run, "small_reference.npz"), allow_pickle=True)
    dev = np.load(os.path.join(run, "development_reference.npz"), allow_pickle=True)
    imu_quat = np.asarray(small["imu_orientation"], np.float64)   # [F,6,4] wxyz
    pelvis_p, head_p = pos("Pelvis"), pos("Head")
    chest_p = pelvis_p + (2.0 / 3.0) * (head_p - pelvis_p)
    imu_pos = np.stack([chest_p, pos("L_Wrist"), pos("R_Wrist"),
                        pos("L_Foot"), pos("R_Foot"), head_p], axis=1)  # [F,6,3]

    grf = np.asarray(dev["grf_feet_source_native"], np.float64)   # [F,2,3]
    grf_vert = grf[:, :, 2]                                        # Z-up vertical
    contact = np.asarray(dev["foot_contact_mask"]).astype(bool)   # [F,2]
    foot_pos = np.stack([pos("L_Foot"), pos("R_Foot")], axis=1)   # [F,2,3]

    out = os.path.join(run, "anim_data.npz")
    np.savez(out,
             joints=np.array(JOINTS, dtype=object), J=J.astype(np.float32), bones=bone_idx,
             sensor_codes=small["sensor_codes"], imu_quat=imu_quat.astype(np.float32),
             imu_pos=imu_pos.astype(np.float32),
             foot_pos=foot_pos.astype(np.float32), grf_vert=grf_vert.astype(np.float32),
             contact=contact, grf_ref=float(np.nanpercentile(grf_vert[contact], 95) or 1.0))
    print("saved:", out)
    print("frames:", J.shape[0])
    print("world extent  X:[%.2f,%.2f] Y:[%.2f,%.2f] Z:[%.2f,%.2f]" % (
        J[..., 0].min(), J[..., 0].max(), J[..., 1].min(), J[..., 1].max(),
        J[..., 2].min(), J[..., 2].max()))
    print("foot z min (floor approx): %.3f" % foot_pos[..., 2].min())
    print("grf_vert 95pct (contact):", float(np.nanpercentile(grf_vert[contact], 95)))


if __name__ == "__main__":
    main()
