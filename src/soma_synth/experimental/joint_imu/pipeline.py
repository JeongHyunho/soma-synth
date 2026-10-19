"""Inference adapter preserving the retained experiment's window contract."""
import hashlib
import json
from pathlib import Path

import numpy as np

SITES = ('TA', 'LFA', 'RFA', 'LSK', 'RSK', 'HE', 'LFT', 'RFT')
PHYSICS = 'SG7_degree6_quaternion_spline_0.001'


def runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def validate(time, markers, usable, sites):
    if tuple(sites) != SITES:
        raise ValueError(f'Expected sensor order {SITES}; no site substitution')
    if (time.ndim != 1 or len(time) < 512 or not np.isfinite(time).all()
            or not np.allclose(np.diff(time), .01, rtol=0, atol=1e-8)):
        raise ValueError('Expected >=512 samples on a uniform 100-Hz clock')
    if markers.shape != (len(time), 8, 3, 3):
        raise ValueError('Expected markers_m (T,8,3,3), M1/M2/M3 in metres')
    if usable.shape != (len(time),) or usable.dtype != bool:
        raise ValueError('Expected a boolean input-quality mask (T,)')


def physics(time, markers, usable, sites):
    """SG7 degree SIX, matching marker_quality/prepare.py, not legacy SG11.

    Mask uses input quality only. The conservative 73-frame edge exclusion
    is explicit; historical evaluation masks must be supplied for exact replay.
    """
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation
    from ..sg11.marker_geometry import frame
    from ..sg11 import quaternion_spline as ic
    time = np.asarray(time, dtype=float)
    markers = np.asarray(markers, dtype=float)
    usable = np.asarray(usable)
    validate(time, markers, usable, sites)
    good = usable & np.isfinite(markers).all((1, 2, 3))
    for s in range(8):
        good &= frame(markers[:, s])[2]
    out = np.full((len(time), 48), np.nan, dtype=np.float32)
    valid = np.zeros(len(time), dtype=bool)
    for a, b in runs(good):
        if b-a < 160:
            continue
        tt = time[a:b]-time[a]
        qt = tt[2:-2]
        keep = (tt >= max(tt[3], qt[2])) & (tt <= min(tt[-4], qt[-2]))
        ix = np.arange(a, b)[keep]
        for s in range(8):
            p, r, _ = frame(markers[a:b, s])
            q = ic.smooth_quaternions(r, .001)
            qv, w = ic.quaternion_bspline(qt, q, tt[keep])
            rot = Rotation.from_quat(np.roll(qv, -1, axis=1))
            acc = savgol_filter(p-p[:1], 7, 6, deriv=2, delta=.01,
                                axis=0, mode='interp')
            out[ix, 3*s:3*s+3] = rot.inv().apply(acc[keep]+[0, 0, 9.81])
            out[ix, 24+3*s:27+3*s] = w
        valid[a+73:b-73] = True
    return out, valid & np.isfinite(out).all(1)


def predict(time, markers, signals, valid, sites, checkpoint, metadata,
            device='cpu', batch_size=16):
    """Return predictive mean and marginal sigma in SI units; never sample.

    Real IMUs, target masks and anatomical reference poses are not inputs.
    Prepared signals must be the declared SG7 physics, not another model output.
    """
    import torch
    from .network import Synth
    time = np.asarray(time, dtype=float)
    markers = np.asarray(markers, dtype=np.float32)
    signals = np.asarray(signals, dtype=np.float32)
    valid = np.asarray(valid)
    validate(time, markers, valid, sites)
    if signals.shape != (len(time), 48) or batch_size < 1:
        raise ValueError('Expected signals (T,48) and positive batch size')
    info = json.loads(Path(metadata).read_text(encoding='utf-8'))
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if (info.get('checkpoint_sha256') != digest or info.get('format_version') != 1
            or info.get('physics') != PHYSICS or info.get('sites') != list(SITES)
            or info.get('sample_rate_hz') != 100
            or info.get('architecture') != 'joint_gated_multiscale_uncertainty'):
        raise ValueError('Checkpoint hash, architecture, physics or site contract mismatch')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ck['format_version'] != 1 or ck['split'] != info['split']:
        raise ValueError('Checkpoint provenance mismatch')
    cfg, bc = ck['config'], ck['base_config']
    if not (cfg['joint'] and cfg['probabilistic'] and cfg['family'] == 'multiscale'
            and bc['kind'] == 'gated' and bc['all_acceleration']):
        raise ValueError('Expected corrected joint gated multiscale probabilistic model')
    for name in ['mean', 'scale', 'sd', 'base_mean', 'base_scale']:
        value = ck[name]
        if value.numel() != 48 or not torch.isfinite(value).all():
            raise ValueError(f'Invalid {name}')
        if name in ['scale', 'sd', 'base_scale'] and not (value > 0).all():
            raise ValueError(f'Nonpositive {name}')
    model = Synth(cfg, ck['mean'], ck['scale'], bc, ck['base_mean'],
                  ck['base_scale'], ck['sd'])
    model.load_state_dict(ck['model'], strict=True)
    model.to(device).eval()
    good = valid & np.isfinite(signals).all(1) & np.isfinite(markers).all((1, 2, 3))
    starts = [s for a, b in runs(good) for s in range(a, b-511, 256)]
    if not starts:
        raise ValueError('No contiguous 512-frame input window')
    mean = np.full_like(signals, np.nan)
    sigma = np.full_like(signals, np.nan)
    output_valid = np.zeros(len(time), dtype=bool)
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            chunk = starts[offset:offset+batch_size]
            x = np.stack([signals[s:s+512].T for s in chunk])
            m = []
            for s in chunk:
                raw = markers[s:s+512].copy()
                raw -= raw[0].mean(axis=1)[None, :, None, :]
                m.append(raw.reshape(512, 72).T/.1)
            result = model(torch.as_tensor(x, device=device),
                           torch.as_tensor(np.stack(m), device=device))
            uncertainty = torch.exp(.5*model.log_variance)*model.sd
            if not torch.isfinite(result).all() or not torch.isfinite(uncertainty).all():
                raise ValueError('Nonfinite checkpoint output')
            result = result.cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
            for i, s in enumerate(chunk):
                mean[s+128:s+384] = result[i, :, 128:384].T
                sigma[s+128:s+384] = uncertainty[i, :, 128:384].T
                output_valid[s+128:s+384] = True
    return mean, sigma, output_valid, info
