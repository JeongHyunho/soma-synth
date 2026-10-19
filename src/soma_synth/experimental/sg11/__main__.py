"""Write an experimental signal artifact, not a registered Small/Large bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .synthesis_pipeline import SITES, synthesize, correct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-metadata', type=Path)
    args = p.parse_args()
    if bool(args.checkpoint) != bool(args.checkpoint_metadata):
        p.error('Supply both checkpoint and checkpoint-metadata, or neither')
    if args.output.exists():
        p.error('Output must be a new directory')
    with np.load(args.input, allow_pickle=False) as data:
        time, markers, usable, sites = (data[k] for k in ('time_s', 'markers_m', 'usable', 'sites'))
    physics, q, valid = synthesize(time, markers, usable, sites)
    if not valid.any():
        p.error('No usable physics frames; no output written')
    arrays = dict(time_s=time, physics_si=physics, baseline_orientation_wxyz=q, physics_valid=valid, sites=np.array(SITES))
    manifest = dict(status='experimental_non_candidate', distribution='internal_only',
        source='dynamic_marker_clusters', method='SG11', sample_rate_hz=100,
        channel_order='24 specific-force channels (m/s^2), then 24 gyro channels (rad/s)',
        gravity_world_m_per_s2=[0, 0, -9.81], offline_noncausal=True,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        orientation='baseline smoothed cluster orientation; not corrected orientation',
        limitations=['Not a validated qmd_unified8_smpl18 bundle', 'No anatomical mounting calibration',
                     'No external-dataset accuracy claim', 'Input quality mask must be supplied'],
        physics_valid_frames=int(valid.sum()))
    if args.checkpoint:
        result, mask, info = correct(physics, valid, args.checkpoint, args.checkpoint_metadata)
        if not mask.any():
            p.error('No usable corrected frames; no output written')
        arrays.update(pose_corrected_si=result, pose_corrected_valid=mask)
        manifest.update(checkpoint=info, corrected_valid_frames=int(mask.sum()))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'signals.npz', **arrays)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
