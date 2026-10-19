# IMUSim source mapping and remaining differences

Reference repository: https://github.com/martinling/imusim

Pinned commit: `294e171ae5f2da7384ffc7d1b9af347cfcb30ce2`.

The upstream source was cloned, inspected, and retained under `reference_imusim/`. The port in `quaternion_spline.py` is adapted from that source, Copyright (C) 2009-2011 University of Edinburgh, under GPL-3.0-or-later. The license is included in `LICENSE_IMUSIM.txt`; source attribution is retained in code. This is a Python 3/NumPy port of the ideal trajectory pathway, not a claim to run the original package or reproduce every part of the 2011 experiment.

## Algorithm mapping

| Local implementation | Upstream reference | Matching behavior |
|---|---|---|
| `spline_values` / `ideal_imu` | `imusim/maths/splines.py`, `trajectories/splined.py` | FITPACK splrep/splev; degree-five position splines; weight 1/stddev; default smoothing budget with weights; analytical second derivative |
| `smooth_quaternions` | `maths/quaternions.pyx`, `QuaternionArray.unflipped` and `.smoothed` | Sign continuity; cubic smoothing of each component on sample-index coordinates; normalization; exclusion of two boundary samples per end |
| `quaternion_bspline` | `maths/quat_splines.pyx`, `QuaternionBSpline` | Order-four cumulative Kim quaternion B-spline, original control extension/indexing, quaternion exponential products and analytical first derivative |
| gyro output | `trajectories/splined.py`, `platforms/gyroscopes.py` | Body angular rate `2 vector(conj(q) qdot)`; equivalent to upstream rotating rate to world then back into the sensor frame |
| accelerometer output | `platforms/accelerometers.py` | Specific force `R^T(a_world - g_world)` |
| valid samples | `maths/splines.py`, `maths/quat_splines.pyx`, partial-input wrappers | Intersect valid intervals; do not report extrapolated endpoints as valid measurements |

The port uses vectorized closed-form cumulative cubic basis polynomials in place of upstream's recursive B/tildeB evaluation. `test_imusim_core.py` independently evaluates the original recursion at fractional times and compares rotations; it checks angular rates against a central difference of that independent reference. These are algorithm-level checks, not a binary comparison with the compiled legacy package.

Defaults are now positionStdDev=0.001 m, position spline degree=5, rotationStdDev=0.001 per dimensionless quaternion component, and gravity magnitude 9.81 m/s². The previous implementation used degree-three position splines, a 10 Hz orientation filter, SciPy RotationSpline, and 9.8 m/s² gravity. The former code and documentation are retained in `previous_baseline/`.

## Deliberate dataset adaptations

- HKNU has world Z upward, whereas upstream's default gravity coordinates differ. We explicitly use downward gravity `[0,0,-9.81]` in the HKNU world, keeping the same sensor physics. `--gravity 9.8` is available to match the HKNU author example convention.
- HKNU optical IMUPos/IMURot trajectories are used directly. IMUSim permits direct sampled trajectories; a separate rigid-body skeleton is not needed for these measured sensor locations. This is not identical to the paper's body-model-driven validation experiment.
- Inputs must be uniformly sampled and finite. Upstream can split partially missing trajectories into valid regions; this adapter currently rejects missing optical data instead of silently bridging gaps. It does not synthesize across gaps.
- `--rotation-noise 0` disables component smoothing, corresponding to upstream smoothRotations=False. The old `--rotation-cutoff` parameter has been removed; a frequency cutoff and quaternion-component uncertainty are different quantities.
- Only orientation and first angular derivative are ported. Angular acceleration, magnetometer/field simulation, analog sensor transfer functions, ADC, clocks, radio, and the event simulator are not included.
- The default corresponds to ideal accelerometer/gyroscope behavior. No arbitrary noise is fitted or added to imitate HKNU hardware. The paper's waveform validation also presents idealized sensor traces without added noise. Realistic hardware simulation would need independently specified/calibrated sensor parameters.
- New NaNs at a few endpoint samples mean undefined spline values, not missing original measurements. Evaluation masks exclude them. Any future SOMA adapter must respect this mask.

## Evaluation limits

Primary comparisons remain against the supplied processed real IMU at existing timestamps. No timing shift, gain, or bias is fitted to targets. An additional matched-10-Hz comparison is labeled separately and is not part of the synthesis algorithm. Global source-based settings have not been selected by optimizing HKNU errors. Code fidelity and empirical error are separate questions; closer agreement with the original algorithm does not guarantee lower errors.
