"""frame_*.png sequence -> H.264 mp4, using the ffmpeg (libx264) bundled with imageio_ffmpeg.

cv2's avc1 (libopenh264) encoder is missing or broken depending on the distribution build, so the
imageio_ffmpeg ffmpeg binary is called directly (H.264 that PowerPoint and browsers play, yuv420p).
Crops the white background automatically.
"""
from __future__ import annotations
import argparse, glob, os, subprocess
import numpy as np
import cv2
import imageio_ffmpeg


def chroma_bbox(files, pad=0.07):
    y0 = x0 = 10**9; y1 = x1 = -1
    step = max(1, len(files)//60)
    for f in files[::step]:
        im = cv2.imread(f)
        if im is None: continue
        sat = im.max(axis=2).astype(np.int16) - im.min(axis=2).astype(np.int16)
        mask = (sat > 30) | (im.max(axis=2) < 242)       # saturation (skin, RGB axes) | ground shadow
        if not mask.any(): continue
        ys, xs = np.where(mask)
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    if y1 < 0: return None
    H, W = cv2.imread(files[0]).shape[:2]
    ph, pw = int((y1-y0)*pad), int((x1-x0)*pad)
    y0, y1 = max(0, y0-ph), min(H, y1+ph+1)
    x0, x1 = max(0, x0-pw), min(W, x1+pw+1)
    y1 -= (y1-y0) % 2; x1 -= (x1-x0) % 2
    return int(x0), int(y0), int(x1-x0), int(y1-y0)     # x,y,w,h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))
    if not files:
        raise SystemExit(f"no frames in {args.frames}")

    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if not args.no_crop:
        box = chroma_bbox(files)
        if box:
            x, y, w, h = box
            print(f"autocrop: {w}x{h} at ({x},{y})")
            vf = f"crop={w}:{h}:{x}:{y}," + vf

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    pattern = os.path.join(args.frames, "frame_%04d.png")
    cmd = [ff, "-y", "-framerate", str(args.fps), "-start_number", "1", "-i", pattern,
           "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    print("ffmpeg:", ff)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"MP4: {args.out}  ({len(files)} frames @ {args.fps}fps, {os.path.getsize(args.out):,} B)")


if __name__ == "__main__":
    main()
