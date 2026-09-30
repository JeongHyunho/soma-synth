"""PRISM take002 (SMPL-24) -> SMPL-X format npz (input to the render_amass.py add-on).

The standard SMPL-H -> SMPL-X convention:
  SMPL/SMPL-H/SMPL-X share the first 66 dimensions (global 3 + body 21 joints x 3 = 63).
  From PRISM poses[T,72] (SMPL 24 joints), 0:66 is used as the SMPL-X body as it is, and
  jaw/eyes (66:75) and hands (75:165) are filled with 0 (neutral), giving 165 dimensions.
  betas are 0 (neutral body) because the shape spaces differ — the same as smplh_to_smplx.

INTERNAL-ONLY. The PRISM pkl is loaded only with an unpickler restricted to a numpy whitelist.

    python prism_to_smplx.py --out <npz> [--pkl <take002.pkl>] [--data-root <dir>]

--out is required. Without --pkl, prism/subj001/take002.pkl is read from the source folder
($SOMA_SOURCE_ROOT, else <--data-root or $SOMA_DATA_ROOT>/extracted). The pkl is checked first, and
only then is the --out folder created.
--out inside a bundle or corpus (a folder above holds INDEX.json), the lineage container, an evidence
folder or a source folder is refused (viewer_paths.writable_folder): name a file in a scratch folder.
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer_paths  # noqa: E402

W0, W1 = 1000, 2000   # the same window as the other outputs (10 s @ 100 Hz)
#: Where --out goes instead of a bundle's folder (viewer_paths.writable_folder refuses one).
SCRATCH_OUT = "name a file in a scratch folder"


class _NumpyOnly(pickle.Unpickler):
    _A = {"_reconstruct", "scalar", "ndarray", "dtype"}
    def find_class(self, m, n):
        if m.startswith("numpy") and n in self._A:
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f"BLOCKED {m}.{n}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PRISM take002 SMPL-24 -> SMPL-X npz")
    ap.add_argument("--out", default=None, help="output npz (required)")
    ap.add_argument("--pkl", default=None,
                    help="PRISM take pickle (default <source root>/prism/subj001/take002.pkl)")
    ap.add_argument("--data-root", default=None,
                    help="finds the source root when --pkl and $SOMA_SOURCE_ROOT are not given; "
                         "default $SOMA_DATA_ROOT")
    args = ap.parse_args(argv)
    out = viewer_paths.require(args.out, "--out", "the output npz", writes=SCRATCH_OUT)
    viewer_paths.writable_folder(os.path.dirname(os.path.abspath(out)), "--out",
                                 instead=SCRATCH_OUT)
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

    sp = raw["smpl_params"]
    poses72 = np.asarray(sp["poses"], np.float64)[W0:W1]      # [T,72] SMPL-24 axis-angle
    trans = np.asarray(sp["trans"], np.float64)[W0:W1]        # [T,3]
    fps = int(raw["info"]["data_info"]["fps"])                # 100
    gender_raw = str(raw["info"]["subj_info"]["gender"])      # 'M'
    gender = {"M": "male", "F": "female"}.get(gender_raw.upper()[:1], "neutral")

    T = poses72.shape[0]
    assert poses72.shape[1] == 72, poses72.shape
    poses_x = np.zeros((T, 165), dtype=np.float64)
    poses_x[:, 0:66] = poses72[:, 0:66]     # global(3) + body 21 joints (63); SMPL hands (66:72) dropped
    # 66:75 jaw/left_eye/right_eye = 0 ; 75:165 hands = 0 (neutral/flat)

    betas = np.zeros(16, dtype=np.float64)  # different shape space -> neutral (smplh_to_smplx convention)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, trans=trans, gender=np.array(gender), mocap_framerate=np.float64(fps),
             betas=betas, poses=poses_x)
    print("saved:", out)
    print(f"  poses {poses_x.shape}, frames {T}, fps {fps}, gender {gender} (from '{gender_raw}')")
    print(f"  window [{W0},{W1}); body from PRISM SMPL-24[:66], hands/face neutral, betas=0")


if __name__ == "__main__":
    main()
