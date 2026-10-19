"""python -m soma_synth.experimental.joint_imu --help"""
import argparse
import json
from pathlib import Path
import numpy as np
from .pipeline import physics, predict, PHYSICS


def main():
    p = argparse.ArgumentParser(description='Retained joint IMU model, inference only')
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--metadata', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--prepared-sg7', action='store_true',
                   help='Read physics_imu and physics_id from input for exact prepared replay')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new path')
    with np.load(args.input, allow_pickle=False) as z:
        time, markers, usable, sites = (z[k].copy() for k in
                                      ['time_s', 'markers_m', 'usable', 'sites'])
        if args.prepared_sg7:
            if str(z['physics_id'].item()) != PHYSICS:
                p.error('Prepared input must declare the exact SG7 physics_id')
            x, valid = z['physics_imu'].copy(), usable
        else:
            x, valid = physics(time, markers, usable, sites)
    mean, sigma, valid, info = predict(time, markers, x, valid, sites,
                                      args.checkpoint, args.metadata, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        np.savez_compressed(f, time_s=time, mean=mean, sigma=sigma, valid=valid,
                            sites=sites, metadata_json=np.asarray(json.dumps(info)))
    print(f'Saved {int(valid.sum())} valid frames to {args.output}')


if __name__ == '__main__':
    main()
