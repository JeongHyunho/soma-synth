"""Audited 100-Hz marker-cluster pathway. Signals use SI units throughout."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .physics_synthesis import sg11
from .marker_geometry import frame

SITES = ('TA', 'LFA', 'RFA', 'LSK', 'RSK', 'HE', 'LFT', 'RFT')


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.where(edges == 1)[0], np.where(edges == -1)[0])



def synthesize(time, markers, usable, sites):
    """markers: (T,8,3,3), metres, M1/M2/M3; usable: input-quality mask.

    No IMUPos, IMURot, real IMU target or anatomical reference is read.
    Conservative 73-frame support guard reproduces historical comparison support.
    """
    time = np.asarray(time, dtype=float)
    markers = np.asarray(markers, dtype=float)
    usable = np.asarray(usable)
    if tuple(sites) != SITES:
        raise ValueError(f'Expected exact checkpoint site order {SITES}; no hand/forearm substitution')
    if markers.shape != (len(time), 8, 3, 3) or usable.shape != (len(time),) or usable.dtype != bool:
        raise ValueError('Expected markers (T,8,3,3) and boolean usable (T,)')
    if len(time) < 160 or not np.isfinite(time).all() or not np.allclose(np.diff(time), .01, rtol=0, atol=1e-8):
        raise ValueError('Expected >=160 frames, finite uniform 100-Hz timestamps; no resampling')
    geometry = [frame(markers[:, s]) for s in range(8)]
    good = usable.copy()
    for _, _, ok in geometry:
        good &= ok
    signals = np.full((len(time), 48), np.nan)
    orientation = np.full((len(time), 8, 4), np.nan)
    valid = np.zeros(len(time), bool)
    for start, stop in runs(good):
        if stop-start < 160:
            continue
        for s, (p, r, _) in enumerate(geometry):
            z, q = sg11(time[start:stop], p[start:stop], r[start:stop])
            signals[start:stop, 3*s:3*s+3] = z[:, :3]
            signals[start:stop, 24+3*s:27+3*s] = z[:, 3:]
            orientation[start:stop, s] = q
        valid[start+73:stop-73] = True
    valid &= np.isfinite(signals).all(axis=1)
    return signals, orientation, valid


def correct(signals, valid, checkpoint, metadata):
    """Load an explicit trained artifact; never train or access real targets here."""
    import torch
    from .neural_model import PoseCorrection
    checkpoint = Path(checkpoint)
    info = json.loads(Path(metadata).read_text(encoding='utf-8-sig'))
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (info.get('checkpoint_sha256') != sha or info.get('sites') != list(SITES)
            or info.get('sample_rate_hz') != 100 or info.get('physics') != 'SG11'
            or info.get('architecture') != 'pose_correction'
            or not info.get('training_participants')):
        raise ValueError('Checkpoint provenance, sites, physics, architecture or rate mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
    mean = state['mean'].reshape(48).float()
    scale = state['scale'].reshape(48).float()
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid checkpoint normalization')
    model = PoseCorrection(mean, scale).eval()
    model.load_state_dict(state, strict=True)
    output = np.full_like(signals, np.nan)
    corrected_valid = np.zeros(len(valid), bool)
    # Process each contiguous span independently. Context edges are never scored.
    with torch.inference_mode():
        for start, stop in runs(valid):
            if stop-start <= 30:
                continue
            x = torch.as_tensor(signals[start:stop].T.copy(), dtype=torch.float32)[None]
            x = (x-model.mean)/model.scale
            y = ((x+model(x))*model.scale+model.mean)[0].T.numpy()
            output[start+15:stop-15] = y[15:-15]
            corrected_valid[start+15:stop-15] = True
    return output, corrected_valid, info
