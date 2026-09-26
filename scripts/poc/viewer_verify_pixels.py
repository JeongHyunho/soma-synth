"""Automatic pixel check of viewer verify frames.

Requires exactly `verify_start.png`, `verify_mid.png`, `verify_end.png` and, per frame, asserts
SEPARATELY:
  (a) the SMPL-X mesh is visible = enough SKIN-TONE pixels (not white, not a pure R/G/B axis);
  (b) the 8 IMU RGB sensor axes are present = saturated red, green AND blue pixels.
Proving the mesh via skin tone (not merely "non-white") defeats a frame containing only RGB blocks.

Usage:  python viewer_verify_pixels.py --dir <verify_frames_dir>
Exit 0 = all three frames pass, 2 = a frame failed, 1 = a required frame is missing. INTERNAL-ONLY.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import cv2

REQUIRED = ["verify_start.png", "verify_mid.png", "verify_end.png"]


def imread_unicode(path):
    """cv2.imread fails on non-ASCII Windows paths; decode via a byte buffer."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def check_frame(path):
    im = imread_unicode(path)                  # BGR uint8
    if im is None:
        return {"ok": False, "error": "unreadable"}
    b = im[:, :, 0].astype(int)
    g = im[:, :, 1].astype(int)
    r = im[:, :, 2].astype(int)

    # (a) mesh = warm skin-tone pixels (SMPL-X skin ~ sRGB (231,180,155)); r>=g>=b, not near-white,
    #     not a pure primary. This is distinct from the emissive axis pixels below.
    skin = ((r > 140) & (r < 250) & (g > 95) & (g < 215) & (b > 70) & (b < 190) &
            (r >= g) & (g >= b) & ((r - b) > 18) & ((r - b) < 130))
    skin_px = int(skin.sum())

    # (b) axes = saturated single-channel-dominant emissive pixels
    red = (r > 110) & (r - g > 55) & (r - b > 55)
    green = (g > 85) & (g - r > 35) & (g - b > 35)
    blue = (b > 110) & (b - r > 45) & (b - g > 25)
    red_px, green_px, blue_px = int(red.sum()), int(green.sum()), int(blue.sum())

    mesh_visible = skin_px >= 500              # a real body fills thousands of skin px
    axes_rgb_present = red_px >= 15 and green_px >= 15 and blue_px >= 15
    return {
        "ok": True,
        "skin_px": skin_px, "mesh_visible": bool(mesh_visible),
        "red_px": red_px, "green_px": green_px, "blue_px": blue_px,
        "axes_rgb_present": bool(axes_rgb_present),
        "pass": bool(mesh_visible and axes_rgb_present),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    # exact set: precisely the three frames, no more, no fewer (a stale/extra render must not pass)
    present = {os.path.basename(p) for p in glob.glob(os.path.join(args.dir, "verify_*.png"))}
    if present != set(REQUIRED):
        print(json.dumps({"all_pass": False, "error": "verify_*.png set must be exactly the three frames",
                          "expected": sorted(REQUIRED), "present": sorted(present)}))
        sys.exit(1)
    frames = {name: check_frame(os.path.join(args.dir, name)) for name in REQUIRED}
    all_pass = all(v.get("pass") for v in frames.values())
    print(json.dumps({"all_pass": all_pass, "frames": frames}, ensure_ascii=False, indent=2))
    sys.exit(0 if all_pass else 2)


if __name__ == "__main__":
    main()
