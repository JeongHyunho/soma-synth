# Experimental marker-cluster SG11 + pose correction

This optional pathway preserves the HKNU experiments' calculations. It does not
replace the existing SMPL-derived HKNU generator or claim its SMPL reference frames.
Outputs are internal experimental signal artifacts, not registered Small/Large bundles.

## Pipeline

Dynamic M1/M2/M3 markers → geometric cluster frame (origin midpoint M2/M3)
→ SG11 position second derivative (degree 5, 11 samples)
+ smoothed quaternion cumulative B-spline gyro → specific force and angular rate
→ optional trained TCN3 pose correction → corrected specific force and angular rate.

The neural model has 64 channels and dilations 1/2/4. It predicts bounded local
translation (10 mm per component) and rotation (5 degrees per component), then uses
relative-motion physics. It does not use real IMUs at inference. The original model's
weights and normalization load strictly; no model is trained by this command.

## Input contract

NPZ with `time_s` (T), `markers_m` (T,8,3,3), `usable` boolean (T), and Unicode `sites`.
Order is **TA,LFA,RFA,LSK,RSK,HE,LFT,RFT**. These are forearms, not hands.
World Z is up. Marker units are metres. Sampling is exactly 100 Hz. No resampling,
gap filling, target-derived masking, or hand/forearm substitution is performed.
Supply a marker-quality mask; this module only adds geometry/finite-data checks.
Each valid span is synthesized separately with a conservative 73-frame edge guard.
Neural inference adds a 15-frame guard. Compare methods on their mask intersection.

```powershell
$env:PYTHONPATH = 'src'
python -m soma_synth.experimental.sg11 --input clusters.npz --output sg11_trial
python -m soma_synth.experimental.sg11 --input clusters.npz --output corrected_trial `
  --checkpoint best.pt --checkpoint-metadata checkpoint.json
```

Optional PyTorch is required only for correction. Checkpoints are not shipped.
Checkpoint JSON must include `checkpoint_sha256`, `sites` in the order above,
`sample_rate_hz: 100`, `physics: "SG11"`, `architecture: "pose_correction"`,
and `training_participants`. Also record validation/test participants, source run,
mask protocol and performance evidence when sharing an artifact. Never use a fold
checkpoint trained on an evaluation participant as unseen-person evidence.

Install with `pip install -c constraints.txt -e ".[dev,pose-correction]"`.
The experimental command does not need the SMPL submodule.

Outputs: `signals.npz` and `manifest.json`. `physics_si` / `pose_corrected_si` have
24 specific-force channels in m/s² followed by 24 gyro channels in rad/s.
`baseline_orientation_wxyz` belongs to the baseline, **not** the corrected trajectory.
Do not pair it with corrected signals and describe that as a verified corrected-pose
Small/Large bundle. The command does not emit reference joint angles or SMPL motion.

## Scope and provenance

This is offline/noncausal synthesis. Learned correction transfer to other datasets,
sensor placements and frame conventions is not established by these unit tests.
The historical gyro implementation is retained in `quaternion_spline.py`, under its
GPL-3.0-or-later terms; `LICENSE_IMUSIM.txt` and `IMUSIM_SOURCE.md` accompany it.
This extension is isolated and not imported by the default generators.
Source implementations: SOMA `physics_neural_full_20261005/prepare_data.py`,
`pose_correction_20261005/cluster_frame.py`, and `sg11_repeat_20261007/pose_model.py`.

## Verification (2026-10-12)

- Five extension tests passed, including checkpoint validation and gap isolation.
- With repository-pinned NumPy 2.2.6 / SciPy 1.16.2, the physics regression selection
  passed 27 tests; two PyTorch tests were skipped in that isolated environment and
  passed separately in the PyTorch environment.
- Historical SG11 kernel parity and trained fold-1 neural output parity were bit-exact
  on numerical fixtures. Checkpoint SHA256:
  `e468e47308ccf20d14a65887131ff1f29d04fc0adbb33b710796b099eae2c77d`.
- Command-line synthesis with that checkpoint produced finite corrected signals.
- These are implementation checks, not a new HKNU accuracy evaluation or a full
  repository test-suite pass. No dataset or trained checkpoint is committed.

## Module organization

| File | Responsibility |
|---|---|
| `physics_synthesis.py` | SG11 acceleration and spline-based gyro synthesis |
| `neural_model.py` | Learned translation and rotation correction with physics |
| `temporal_network.py` | Temporal convolutional network architecture |
| `marker_geometry.py` | Position and orientation from three-marker clusters |
| `quaternion_spline.py` | Attributed IMUSim quaternion/spline mathematics |
| `synthesis_pipeline.py` | Input validation, masks, checkpoint loading and execution |
| `__main__.py` | Python command-line entry point |
