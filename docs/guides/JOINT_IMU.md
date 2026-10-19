# Joint gated motion + multiscale + uncertainty model

This optional research pathway ports the retained `probabilistic_matched_20261017`
model. It is separate from the earlier SG11/plain-pose adapter and the default
SMPL-derived Small/Large generators. It produces experimental signal artifacts,
not registered Small/Large bundles. No new model is trained by these commands.

## Flow

```text
Three markers per sensor, 100 Hz, input-quality mask
       |
       +--> SG7 physics (degree 6 acceleration, smoothed quaternion spline gyro)
       |                         |
       +--> marker features ---> gated sensor-motion correction
                                 translation + rotation through relative-motion physics
                                                   |
Initial physics + markers ------------------------> multiscale signal network
                                                   | fast + pooled-by-4 slow branch
                                                   +--> bounded residual --> IMU mean
                                                   +--> log-variance --> marginal sigma
```

The two neural stages were trained jointly. The gated model uses all marker
acceleration channels, including head and feet, plus marker positions, velocities
and triangle geometry. Gate values modulate bounded local translation and rotation.
For the retained motion configuration (`cap=2`) the component bounds are 10 mm
and approximately 4 degrees. The signal residual is bounded to 0.25 training
target standard deviations per channel. That last residual is not strictly
constrained by the motion physics.

The variance head changes supervision during training through Gaussian NLL.
Inference returns the predictive mean and sigma, not random samples. Sigma is
marginal uncertainty, not a guaranteed calibrated confidence interval or a way
to recover every missing impact. The saved model still misses some running peaks
and produces some false fluctuations.

## Installation and checkpoint export

Use the repository's Python environment and install `pip install -e ".[pose-correction]"`.
Research checkpoints are distributed separately; do not commit participant data
or binary model artifacts to Git. A fold checkpoint is not an all-participant
deployment fit. Retain its train/validation/test participant lists when sharing it.

Convert a trusted retained checkpoint and its original gated initialization:

```powershell
python -m soma_synth.experimental.joint_imu.export `
  --research-checkpoint joint_best.pt --base-checkpoint gated_best.pt `
  --output artifacts/joint_fold3_seed20261016.pt --trust-research-checkpoints
```

The exporter verifies the original gated weights against the checkpoint's frozen
teacher, includes the jointly updated gated weights, normalization and configuration,
and writes a matching JSON with hashes and participant provenance. The frozen
teacher remains in the bundle for strict state compatibility; it is a training
reference, not a third correction stage. Inference loads a tensor-only bundle with
`weights_only=True` and does not read the original checkpoint path.

## Input and execution

Input NPZ (no pickled objects):

| Key | Contract |
| --- | --- |
| `time_s` | `(T,)`, finite uniform 100 Hz |
| `markers_m` | `(T,8,3,3)`, world positions in metres, M1/M2/M3 |
| `usable` | boolean `(T,)`, input quality only; never constructed from real IMU errors |
| `sites` | Unicode array: TA,LFA,RFA,LSK,RSK,HE,LFT,RFT |

World Z is up. The geometric origin is the midpoint of M2/M3. Sensor order means
thorax, left/right forearm, left/right shank, head, left/right foot. No hand/forearm
substitution, automatic time resampling, gap filling or target-derived alignment.

```powershell
python -m soma_synth.experimental.joint_imu markers.npz synthetic.npz `
  --checkpoint artifacts/joint_fold3_seed20261016.pt `
  --metadata artifacts/joint_fold3_seed20261016.json --device cpu
```

For replay of the research feature pack, also supply `physics_imu` `(T,48)` and
scalar Unicode `physics_id="SG7_degree6_quaternion_spline_0.001"`, use the original
physics validity mask as `usable`, and add `--prepared-sg7`. A label does not verify
numerical provenance: the caller must supply genuine SG7 features, never SG11 or
an already corrected signal. Raw-marker mode uses a conservative 73-frame physics
edge guard; its coverage need not equal a historical evaluation mask.

Output: `time_s`, `mean`, `sigma`, `valid`, `sites`, `metadata_json`.
Both arrays are `(T,48)`: 24 acceleration channels followed by 24 gyro channels.
Acceleration is specific force in m/s², gyro rad/s, sigma in corresponding units.
Unusable outputs are NaN. Only entries marked `valid` are synthesized outputs.

## Exact window behavior

Process each continuous usable span separately in 512-frame windows, stride 256.
Keep indices 128:384 of each window. Translate markers relative to each sensor's
first-frame three-marker centroid and divide positions by 0.1, exactly as training.
Do not use arbitrary full-sequence inference: GroupNorm and the pooled slow branch
depend on the window. Short spans and unmatched trailing edges remain invalid.

## Retained training objective (reference, not a training command)

The source objective includes Smooth L1 waveform loss, 0.05 first-difference loss,
motion and residual magnitude/temporal regularizers, 0.25 teacher-preservation loss,
0.05 excess local-fluctuation penalties at windows 3/7/15, 0.1 event surrogate,
and 0.1 Gaussian NLL. Robust top-group weight is zero and **spectral weight is zero**.
These describe the retained configuration; inclusion is not proof each term is
individually beneficial in this final uncertainty-aware model.

Code provenance: `probabilistic_matched_20261017/{network,frozen_network,pose_model,models,fit}.py`;
physics provenance: `marker_quality_20261014/prepare.py:synth`. Existing quaternion
spline implementation and its license notices are reused from `experimental.sg11`.
Network constructors were adapted to explicit portable configuration; tensor names
and forward equations are retained. No training or test suite was run for this port.
Numerical equivalence to saved predictions remains to be verified before treating
this integration as a validated replacement in the shared generation pipeline.

> **Note (2026-10-19).** The provenance files named above (`probabilistic_matched_20261017/…`,
> `marker_quality_20261014/prepare.py`) are not in this repository; they are in the research workspace
> the model was trained in. Neither are the checkpoints (`joint_best.pt`, `gated_best.pt`) or any saved
> predictions, so the numerical equivalence above cannot be checked from this repository alone.
