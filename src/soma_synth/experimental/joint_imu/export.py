"""Convert trusted research checkpoints to a self-contained weights-only bundle.

Run only on checkpoints whose origin you trust: research files contain NumPy
pickle objects. The inference command itself uses weights_only=True.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .pipeline import SITES, PHYSICS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-checkpoint', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trust-research-checkpoints', action='store_true', required=True)
    a = p.parse_args()
    meta = a.output.with_suffix('.json')
    if a.output.exists() or meta.exists():
        p.error('Choose unused output and metadata paths')
    ck = torch.load(a.research_checkpoint, map_location='cpu', weights_only=False)
    base = torch.load(a.base_checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(ck['config']); cfg.pop('base_checkpoint', None)
    if not (cfg['family'] == 'multiscale' and cfg['joint'] and cfg['probabilistic']
            and base['config']['kind'] == 'gated' and base['config']['all_acceleration']):
        p.error('Wrong research model family')
    # The teacher stores the initial gated checkpoint. Prove the supplied base
    # is the actual source instead of trusting an obsolete absolute path.
    for key, value in base['model'].items():
        if not torch.equal(ck['model']['teacher.'+key], value):
            p.error('Base checkpoint does not match retained teacher weights')
    bundle = dict(format_version=1, config=cfg, base_config=base['config'],
                  model=ck['model'], split=ck['split'], seed=int(ck['seed']),
                  epoch=int(ck['epoch']))
    for key in ['mean', 'scale', 'sd']:
        bundle[key] = torch.as_tensor(ck[key]).clone()
    for key in ['mean', 'scale']:
        bundle['base_'+key] = torch.as_tensor(base[key]).clone()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        torch.save(bundle, f)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    info = dict(format_version=1, architecture='joint_gated_multiscale_uncertainty',
                physics=PHYSICS, sample_rate_hz=100, sites=list(SITES),
                channel_order='acc:8xXYZ then gyro:8xXYZ',
                units=dict(mean_acc='m/s^2 specific force', mean_gyro='rad/s',
                           sigma='same units as corresponding mean'),
                split=ck['split'], seed=int(ck['seed']), epoch=int(ck['epoch']),
                checkpoint_sha256=sha(a.output), source_checkpoint_sha256=sha(a.research_checkpoint),
                base_checkpoint_sha256=sha(a.base_checkpoint),
                config=cfg, base_config=base['config'],
                window=dict(length=512, stride=256, keep_start=128, keep_end=384),
                status='experimental_non_candidate', distribution='internal_only')
    with meta.open('x', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(a.output)
    print(meta)


if __name__ == '__main__':
    main()
