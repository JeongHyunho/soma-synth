"""IMU signal plotting for the interactive viewer.

Click an IMU site → its **acceleration / angular-velocity / orientation** time-series with a vertical
frame cursor (3 stacked panels). matplotlib **Agg** (no GUI backend) so the exact same code runs both
headless (PNG for verification) and inside Blender's Python (rendered to an in-Blender image, updated
live on frame change). numpy-only — no scipy — so it works in Blender's bundled Python. INTERNAL-ONLY.

Data source: `small_reference.npz` (`imu_acceleration/_angular_velocity/_orientation`, `sensor_codes`).
The viewer's IMU site names differ in order from the sensor codes, so mapping is by NAME, not index.

What a curve is worth is written next to it, because a synthesized channel looks like any other:
- `imu_valid_mask` False means the arrays are NaN there (GAITEX marker gaps, 0.34 % of cells,
  but one wrist-trial is 18 % valid). Those spans are shaded grey in every panel and the title
  carries the site's valid fraction when it is not 100 %.
- `imu_confidence` is averaged over the VALID cells and shown in the title, marked "(corpus scale)"
  unless the manifest declares what the number means (`small_synthesis.imu_confidence_semantics`):
  PRISM's 0.4/0.7, AMASS's 0.4-0.6 and GAITEX's 0.6/0.3/0.0 are not one scale.
- A manifest that lists the site under `unbacked_sites` (a bundle whose channel is forward
  kinematics of a joint the source never observed) turns the title red and says so.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# IMU site names ARE the small_reference.npz sensor codes (00_sensors.qmd channel map): the trunk
# sensor is back_T4, the head sensor occiput, and the insole IMUs foot_l/foot_r.
SITES = ["back_T4", "occiput", "wrist_l", "wrist_r", "shank_l", "shank_r", "foot_l", "foot_r"]

# The unit of `imu_angular_velocity`, fixed by whoever WRITES the channel: both faithful builders
# fill it from `angular_velocity_deg_s()`. The plot said rad/s, wrong by 57.3x and wrong in the
# direction that still looks plausible — a shank peaking at 470 was labelled 470 rad/s, i.e. 27,000
# deg/s, which no limb does. Named here so a test can hold the plot and the generators to one answer.
GYRO_UNIT = "deg/s"
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")   # X=red, Y=green, Z=blue — matches the 3D RGB axes
INVALID_COLOR = "0.82"       # grey shading for frames the site could not be seen on ("grey" in the title)
MAX_SHADED_RUNS = 300        # a marker-gap corpus can have many short runs; shade a bounded number


MANIFEST_SITE_LISTS = {
    # manifest field -> data key. Each is a list of site names, absent on bundles that do not say.
    "unbacked_sites": "unbacked",             # FK of a joint the source never observed (GAITEX 09-05)
    "worn_unit_compared_sites": "compared",   # compared against the worn XSens unit on THIS take
    "declared_placement_sites": "declared",   # mount offset is a declared zero, not a measurement
}


def manifest_sites(run_dir):
    """{data key: set of sites} for every site list the run's manifest carries (empty sets otherwise).

    The declarations are the manifest's, not thresholds of ours: `imu_confidence` also moves with
    them, but a number is a hint and the manifest is the statement.
    """
    out = {key: set() for key in MANIFEST_SITE_LISTS.values()}
    out["confidence_semantics"] = False       # does the manifest say what its imu_confidence means?
    path = os.path.join(run_dir, "manifest.json")
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return out
    if not isinstance(manifest, dict):
        return out
    for field, key in MANIFEST_SITE_LISTS.items():
        listed = manifest.get(field)
        if isinstance(listed, list):
            out[key] = {str(s) for s in listed}
    synth = manifest.get("small_synthesis")
    out["confidence_semantics"] = bool(isinstance(synth, dict) and synth.get("imu_confidence_semantics"))
    return out


def unbacked_sites(run_dir):
    """Sites the run's manifest declares as NOT source-backed (an empty set when it declares none)."""
    return manifest_sites(run_dir)["unbacked"]


def load_imu(run_dir):
    """Read the IMU channels + sensor codes + timestamps from a run's small_reference.npz."""
    z = np.load(os.path.join(run_dir, "small_reference.npz"), allow_pickle=True)
    accel = np.asarray(z["imu_acceleration"], np.float64)            # [T,8,3]
    # Bundles emitted from 2026-09-06 carry an informative mask: where it is False the three IMU
    # arrays are NaN (ADR-0039), which matplotlib draws as a gap indistinguishable from a flat
    # signal at this scale. Older bundles are all-True, so an absent mask means "everything valid".
    valid = (np.asarray(z["imu_valid_mask"], bool) if "imu_valid_mask" in z.files
             else np.ones(accel.shape[:2], bool))
    return {
        "codes": [str(x) for x in z["sensor_codes"]],
        "accel": accel,
        "gyro": np.asarray(z["imu_angular_velocity"], np.float64),   # [T,8,3]
        "orient": np.asarray(z["imu_orientation"], np.float64),      # [T,8,4] (w,x,y,z)
        "valid": valid,                                              # [T,8]; False = NaN there
        "t": (np.asarray(z["timestamps_s"], np.float64) if "timestamps_s" in z.files else None),
        "confidence": (np.asarray(z["imu_confidence"], np.float64)   # [T,8], 0..1, or None
                       if "imu_confidence" in z.files else None),
        **manifest_sites(run_dir),                                   # "unbacked", "compared", "declared"
    }


def invalid_runs(valid_site):
    """Contiguous [start, stop) frame runs where a site has no signal. numpy-only, no scipy."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], (~valid_site).view(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def _valid_column(data, site):
    mask = data.get("valid")
    if mask is None:                                    # a caller-built dict without a mask
        return np.ones(data["accel"].shape[0], bool)
    return mask[:, sensor_index(data["codes"], site)]


def valid_fraction(data, site):
    """Share of frames on which this site carries a signal (1.0 when the run has no mask)."""
    col = _valid_column(data, site)
    return float(col.mean()) if col.size else 1.0


def invalid_spans(data, site):
    """[(start, stop)] frame index runs (stop exclusive) on which the site is masked out, as ints."""
    return [(int(a), int(b)) for a, b in invalid_runs(_valid_column(data, site))]


def sensor_index(codes, site):
    if site not in codes:
        raise KeyError(f"IMU {site!r} not in sensor codes {codes}")
    return codes.index(site)


def site_note(data, site):
    """What the reader must know about this site's curves, or "" when there is nothing to say.

    Mean confidence over the valid cells when the channel carries one, what the manifest says the
    site is (compared with a worn unit / a declared mount / not source-backed), and the valid
    fraction when it is below 100 %.
    """
    parts = []
    sidx = sensor_index(data["codes"], site)
    conf, valid = data.get("confidence"), _valid_column(data, site)
    if conf is not None:
        col = conf[:, sidx]
        if valid.any():
            col = col[valid]                  # the 0.0 of a missing cell is not a confidence
        # The number means something different on every corpus of the lineage (PRISM 0.4/0.7,
        # AMASS 0.4-0.6, GAITEX 0.6/0.3/0.0); only a manifest that declares its scale earns a bare
        # number, the others are marked so 0.40 on PRISM is not read against 0.60 on GAITEX.
        scale = "" if data.get("confidence_semantics") else " (corpus scale)"
        parts.append(f"confidence {float(np.mean(col)):.2f}{scale}")
    if site in (data.get("compared") or ()):
        parts.append("vs worn unit")
    elif site in (data.get("declared") or ()):
        parts.append("declared mount")
    n_masked = int(np.count_nonzero(~valid))
    if n_masked:
        parts.append(f"valid {100 * valid.mean():.0f}% ({n_masked:,} frames masked, grey)")
    if site in (data.get("unbacked") or ()):
        parts.append("NOT source-backed (FK of an unobserved joint)")
    return "   ·   ".join(parts)              # ~80 chars fit: the title is 5 in wide at 8 pt


def quat_to_euler_deg(q):
    """[...,4] (w,x,y,z) → (roll,pitch,yaw) degrees, ZYX intrinsic. numpy-only."""
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(np.stack([roll, pitch, yaw], axis=-1))


def make_figure(width_px=880, height_px=1120, dpi=140):
    """Create the reusable 3-panel figure (kept alive across frames for live cursor updates).
    Higher px + dpi = a crisper in-Blender plot (zoomable in the Image Editor without blur)."""
    fig, axes = plt.subplots(3, 1, figsize=(width_px / dpi, height_px / dpi), dpi=dpi, sharex=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.06, hspace=0.26)
    return fig, list(axes)


def draw(fig, axes, data, site, frame):
    """Redraw all three panels for `site` with the vertical cursor at `frame` (1-based)."""
    sidx = sensor_index(data["codes"], site)
    t = data["t"] if data["t"] is not None else np.arange(data["accel"].shape[0], dtype=float)
    xlab = "time (s)" if data["t"] is not None else "frame"
    panels = [
        ("acceleration [m/s²]", data["accel"][:, sidx, :], ("aX", "aY", "aZ")),
        (f"angular vel. [{GYRO_UNIT}]", data["gyro"][:, sidx, :], ("gX", "gY", "gZ")),
        ("orientation [deg]", quat_to_euler_deg(data["orient"][:, sidx, :]), ("roll", "pitch", "yaw")),
    ]
    fr = int(np.clip(frame - 1, 0, len(t) - 1))
    # A NaN run draws as a gap, and a gap looks like the plot simply having nothing to say there.
    # Shade it so the reader can tell "this site could not be seen" from "this site was still".
    runs = invalid_spans(data, site)
    step = (t[-1] - t[-2]) if len(t) > 1 else 1.0

    def right_edge(stop):                     # a gap that runs to the end still gets its last frame
        return t[stop] if stop < len(t) else t[-1] + step

    for ax, (ylab, arr, labels) in zip(axes, panels):
        ax.clear()
        for start, stop in runs[:MAX_SHADED_RUNS]:
            ax.axvspan(t[start], right_edge(stop), color=INVALID_COLOR, alpha=0.7, lw=0, zorder=0)
        for j in range(3):
            ax.plot(t, arr[:, j], color=AXIS_COLORS[j], lw=0.9, label=labels[j])
        ax.axvline(t[fr], color="0.15", lw=1.3, alpha=0.8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.legend(loc="upper right", fontsize=6, ncol=3, frameon=False)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
    # Second line rather than a longer first one: the figure is 880 px wide and a single-line
    # title ran off the canvas, cutting off the very warning it was there to give.
    title = f"IMU: {site}   ·   frame {frame}"
    if not _valid_column(data, site)[fr]:
        title += "   ·   CURSOR ON A MASKED FRAME"
    note = site_note(data, site)
    if len(runs) > MAX_SHADED_RUNS:
        note += f"   ·   {MAX_SHADED_RUNS}/{len(runs)} gaps shaded"
    flagged = site in (data.get("unbacked") or ())
    axes[0].set_title(title + (f"\n{note}" if note else ""),
                      fontsize=8, color=("#b22222" if flagged else "black"))
    axes[-1].set_xlabel(xlab, fontsize=8)


def figure_to_rgba_float(fig):
    """Render the figure and return a bottom-to-top float RGBA array ready for a Blender Image.pixels."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return np.flipud(buf).astype(np.float32) / 255.0, w, h    # Blender image rows go bottom-to-top


def render_png(run_dir, site, frame, out_png, **kw):
    """Headless convenience: render one plot PNG (used by the verification path)."""
    data = load_imu(run_dir)
    fig, axes = make_figure(**kw)
    draw(fig, axes, data, site, frame)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render one IMU plot PNG (headless verification).")
    ap.add_argument("--run", required=True)
    ap.add_argument("--site", default="wrist_l", choices=SITES)
    ap.add_argument("--frame", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("wrote", render_png(a.run, a.site, a.frame, a.out))
