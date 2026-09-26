"""PRISM take002 (SMPL-24) -> SMPL-X 포맷 npz (render_amass.py 애드온 입력).

SMPL-H -> SMPL-X 의 표준 규약:
  SMPL/SMPL-H/SMPL-X 는 앞 66차원(global3 + body 21관절×3=63)이 직접 호환.
  PRISM poses[T,72](SMPL 24관절)에서 0:66 을 그대로 SMPL-X body 로 쓰고,
  jaw/eyes(66:75)·hands(75:165) 는 0(neutral)로 채워 165차원을 만든다.
  betas 는 shape space 가 달라 0(neutral body) — smplh_to_smplx 와 동일.

INTERNAL-ONLY. PRISM pkl 은 numpy 화이트리스트 제한 unpickler 로만 로딩.

    python prism_to_smplx.py --out <npz> [--pkl <take002.pkl>] [--data-root <dir>]

--out 은 반드시 준다. --pkl 을 주지
않으면 원천 폴더($SOMA_SOURCE_ROOT, 없으면 <--data-root 또는 $SOMA_DATA_ROOT>/extracted)의
prism/subj001/take002.pkl 을 읽는다. pkl 이 있는지 먼저 확인하고, 그 뒤에야 --out 의 폴더를 만든다.
--out 이 번들·코퍼스(위 폴더에 INDEX.json 이 있다), lineage 컨테이너, 증거 폴더, 원천 폴더 안이면 거부한다
(viewer_paths.writable_folder): scratch 폴더의 파일을 준다.
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer_paths  # noqa: E402

W0, W1 = 1000, 2000   # 다른 산출물과 동일 window (10 s @ 100 Hz)
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
    poses_x[:, 0:66] = poses72[:, 0:66]     # global(3) + body 21관절(63); SMPL hands(66:72) drop
    # 66:75 jaw/left_eye/right_eye = 0 ; 75:165 hands = 0 (neutral/flat)

    betas = np.zeros(16, dtype=np.float64)  # shape space 상이 -> neutral (smplh_to_smplx 규약)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, trans=trans, gender=np.array(gender), mocap_framerate=np.float64(fps),
             betas=betas, poses=poses_x)
    print("saved:", out)
    print(f"  poses {poses_x.shape}, frames {T}, fps {fps}, gender {gender} (from '{gender_raw}')")
    print(f"  window [{W0},{W1}); body from PRISM SMPL-24[:66], hands/face neutral, betas=0")


if __name__ == "__main__":
    main()
