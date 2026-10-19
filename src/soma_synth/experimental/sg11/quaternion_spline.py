# SPDX-License-Identifier: GPL-3.0-or-later
# Adapted from IMUSim, Copyright (C) 2009-2011 University of Edinburgh.
# NumPy/Python 3 port of the trajectory algorithms at upstream commit 294e171ae5f2da7384ffc7d1b9af347cfcb30ce2.
# See IMUSIM_SOURCE.md and LICENSE_IMUSIM.txt for source references, changes, and license terms.
# This software is distributed without warranty; see the GNU General Public License for details.
# ==========================================================================
# START HERE: THE PHYSICS ENGINE (called by run_IMUsim_direct.py)
# ==========================================================================
# Start reading at ideal_imu() in BLOCK 5, then follow its helper calls upward.
# This file defines calculations; it does not read MAT files or draw graphs.
# Running the main script calls these functions. Running this file alone does
# not launch an experiment because it has no main() call.
#
# INPUT -> CALCULATION -> OUTPUT
#   time + position -> spline_values(... derivative=2) -> world_acc
#   matrices -> smooth_quaternions() -> quaternion controls q
#   q + times -> quaternion_bspline() -> orientation + local angular_rate
#   world_acc + gravity + orientation -> local accelerometer reading
#   angular_rate                     -> local gyroscope reading
# ideal_imu() returns both arrays to synthesize(), then to process_trial().
#
# FRAMES: 'world' axes stay fixed in the room; 'sensor' axes move with the IMU.
# A sensor-to-world matrix R converts sensor vectors to world vectors.
# R inverse (R transpose for a rotation) converts world vectors to sensor axes.
# Acceleration and gyro outputs must use the sensor axes to match real channels.
#
# PHYSICS IN PLAIN LANGUAGE
# Position tells us WHERE the sensor is. Its first time derivative is velocity;
# its second derivative is acceleration. A fitted spline is a smooth curve
# through/near samples whose derivatives we can calculate analytically.
# Orientation tells us HOW the sensor is turned. Its time derivative gives the
# rotation rate. Quaternions store orientation in four numbers without the
# angle-wrap problems of differentiating Euler angles.
# An accelerometer measures specific force: f_sensor = R_inverse * (a_world-g).
# Here world Z points up, so g_world = [0,0,-9.81] m/s^2 by default.
# Example: stationary, aligned sensor -> acceleration [0,0,+9.81], gyro [0,0,0].
# The nonzero stationary reading is expected; it is not a synthesis error.
#
# ARRAY/NOTATION GUIDE
# N = original time samples; M = samples inside the valid spline interval.
# positions/acc/gyro: (N,3). Rotations: (N,3,3). Quaternions: (count,4).
# All local quaternion operations use [w,x,y,z]. SciPy uses [x,y,z,w], so
# np.roll at conversion boundaries reorders COMPONENTS, not sample timestamps.
# q[:,1:] selects vector components; [:,None] adds a dimension for multiplying
# one scalar per sample into several components. axis=1 means across columns.
# The @/Hamilton product describes rotation composition; multiplication order
# matters. A quaternion derivative is not itself a unit orientation.
#
# MODEL SCOPE
# This ports IMUSim's ideal trajectory algorithms, not its complete framework.
# No sensor bias, random electronic noise, saturation or hardware model is added.
# Smoothing parameters express input uncertainty; no real IMU target is fitted.
# Optical sensor positions are used directly; no additional body-to-sensor
# lever-arm offset is estimated here. Undefined edges remain NaN and the main
# script excludes them when evaluating. See IMUSIM_SOURCE.md for provenance.
# ==========================================================================
# BLOCK 1: Import numerical tools; all quaternions in this module use scalar-first w,x,y,z order.
import numpy as np  # Store samples and evaluate many quaternions together.
from scipy.interpolate import splrep, splev  # Use the same FITPACK entry points as original IMUSim.
from scipy.spatial.transform import Rotation  # Convert matrices without replacing the original quaternion spline algorithm.

# BLOCK 2: Implement the quaternion operations needed by the cumulative spline.
# CONTRACT - quaternion_bspline() calls this to combine rotations and derivatives.
# INPUT: a and b with four w,x,y,z components each (possibly many rows).
# OUTPUT: their Hamilton product with four components per result.
# This is not element-by-element multiplication: scalar/vector terms mix.
# The same algebra applies to unit orientations and non-unit derivatives.
def multiply(a, b):  # Calculate the Hamilton product of two arrays of w,x,y,z quaternions.
    scalar = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)  # Form the scalar part.
    vector = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])  # Form the vector part.
    return np.concatenate((scalar, vector), axis=-1)  # Return a quaternion with the same batch dimensions.

# CONTRACT - used when finding relative turns and sensor-frame angular rate.
# INPUT: q (...,4). OUTPUT: same shape, with x/y/z signs reversed.
# For a UNIT orientation this equals its inverse. For a general quaternion,
# conjugation alone is not the inverse (the squared length also matters).
def conjugate(q):  # Invert a unit quaternion by negating its vector part.
    return q * np.array([1.0, -1.0, -1.0, -1.0])  # Preserve the scalar component.

# CONTRACT - shared curve-fitting helper with TWO different callers.
# 1. ideal_imu(): time in seconds, positions (N,3), degree=5, derivative=2;
#    returns world acceleration (N,3) in m/s^2.
# 2. smooth_quaternions(): sample indices, components (N,4), degree=3,
#    derivative=0; returns smoothed quaternion components (N,4).
# INPUT: values (N,C), one independent curve per column; noise scale stddev.
# OUTPUT: (N,C) fitted values or derivatives evaluated at the supplied times.
# 'degree' chooses the polynomial degree; 'derivative' chooses what to read:
# 0 = curve itself, 1 = slope, 2 = change of slope. These are separate settings.
# splrep fits/stores a curve; splev evaluates that stored curve or derivative.
# Positive stddev weights residuals by 1/stddev; a larger uncertainty permits
# more departure from input samples. Zero requests exact interpolation.
# This smooths a trajectory before differentiation; it does not add noise.
def spline_values(time, values, stddev, degree, derivative=0):  # Reproduce IMUSim's scalar spline fitting settings.
    options = dict(k=degree)  # Original trajectory defaults use degree five for position and three for quaternion components.
    if stddev > 0:  # Weighted fitting activates FITPACK's default smoothing budget N-sqrt(2N).
        options['w'] = np.full(len(time), 1.0 / stddev)  # Match the original constant inverse-standard-deviation weights.
    else:  # Zero uncertainty requests interpolation rather than smoothing.
        options['s'] = 0  # Pass exactly the option used by original IMUSim for unsmoothed scalar data.
    return np.column_stack([splev(time, splrep(time, values[:, axis], **options), der=derivative) for axis in range(values.shape[1])])  # Fit each coordinate independently.

# BLOCK 3: Smooth quaternion components the same way as QuaternionArray.smoothed().
# CONTRACT - ideal_imu() supplies optical matrices (N,3,3) and noise scale.
# OUTPUT: normalized w,x,y,z controls, (N-4,4) with smoothing, else (N,4).
# These controls go to quaternion_bspline(); ideal_imu() aligns their times.
# Stages: matrix -> quaternion -> remove sign jumps -> optional component
# smoothing -> trim two samples at each end -> normalize to unit length.
# q and -q mean the SAME physical orientation. Sign continuity prevents an
# artificial jump in the numbers from misleading the component smoother.
# This function prepares controls; the later cumulative spline constructs the
# orientation trajectory and calculates its derivative.
def smooth_quaternions(matrices, stddev):  # Convert optical rotations to continuous-sign, optionally smoothed quaternions.
    q = np.roll(Rotation.from_matrix(matrices).as_quat(), 1, axis=1)  # Change SciPy's x,y,z,w order to IMUSim's w,x,y,z.
    for index in range(1, len(q)):  # Walk in time to remove equivalent but discontinuous quaternion signs.
        if np.dot(q[index - 1], q[index]) < 0:  # A negative dot product indicates a sign flip.
            q[index] *= -1  # Keep the same orientation with the continuous sign.
    if stddev > 0:  # Match smoothRotations=True in the original source.
        q = spline_values(np.arange(len(q), dtype=float), q, stddev, 3)  # Fit cubic component splines using sample indices, as upstream does.
        q = q[2:-2]  # The original partial-input cubic spline marks two boundary samples at each end invalid.
    norms = np.linalg.norm(q, axis=1, keepdims=True)  # Component smoothing changes quaternion lengths.
    if np.any(norms < 1e-10):  # Prevent normalization of a degenerate orientation.
        raise ValueError('Quaternion smoothing produced a near-zero quaternion.')  # Stop instead of generating misleading signals.
    return q / norms  # Restore unit rotations, as the original smoothed() implementation does.

# BLOCK 4: Evaluate the order-four cumulative quaternion B-spline of Kim, Kim, and Shin.
# CONTRACT - called by ideal_imu() with prepared quaternion controls.
# INPUT: equally spaced control times, q (K,4), evaluation_times (M,).
# OUTPUT: orientation (M,4) and sensor-frame angular velocity (M,3), rad/s.
# K is the remaining control count after optional smoothing-boundary trimming.
# The caller restricts evaluation times to the upstream valid interval.
#
# A cubic (order-four) curve combines four neighboring orientation controls.
# It need not pass exactly through every control. Instead of averaging angles,
# we combine a starting orientation with three smoothly weighted small turns:
#   q_curve = q_start * exp(beta1*log(delta1))
#                     * exp(beta2*log(delta2)) * exp(beta3*log(delta3)).
# Here * means Hamilton product. log encodes a relative rotation as a vector;
# exp converts a scaled vector back to a quaternion turn. beta controls how
# much of each neighboring turn contributes as time moves through an interval.
#
# The derivative is calculated with the product rule: (A*B)'=A'*B+A*B'.
# Finally omega_sensor = 2 * vector(conjugate(q_curve) * q_curve_derivative).
# The factor 2 is needed because a quaternion encodes HALF the rotation angle.
# This avoids estimating velocity by subtracting neighboring Euler angles.
# All M query times are evaluated together using arrays.
def quaternion_bspline(times, q, evaluation_times):  # Return smoothed sensor-to-world orientations and body-frame angular rates.
    # 4A - PREPARE relative turns between controls, and their quaternion logarithms.
    dt = times[1] - times[0]  # Original IMUSim assumes equally spaced quaternion control points.
    control = np.concatenate((q, q[-1:].repeat(2, axis=0), q[:1].repeat(2, axis=0)))  # Reproduce upstream's exact endpoint extension order.
    relative = multiply(conjugate(np.roll(control, 1, axis=0)), control)  # Calculate each neighboring quaternion increment.
    magnitude = np.linalg.norm(relative[:, 1:], axis=1)  # Obtain sine of the half-angle for each increment.
    half_angle = np.arctan2(magnitude, relative[:, 0])  # Calculate stable half-angles for the unit-quaternion logarithm.
    scale = np.divide(half_angle, magnitude, out=np.ones_like(magnitude), where=magnitude > 1e-14)  # Use the correct zero-angle limit.
    logarithms = np.column_stack((np.zeros(len(relative)), relative[:, 1:] * scale[:, None]))  # Store pure-vector quaternion logarithms.
    # 4B - LOCATE each query in its control interval: u runs from 0 toward 1.
    # The +2 is upstream spline indexing; it is not a time shift applied to data.
    # beta contains cumulative weights; beta_dot contains derivatives per SECOND.
    # Dividing by dt converts change per interval into change per second.
    shifted = (np.asarray(evaluation_times) - times[0]) / dt + 2  # Apply the same two-knot shift as upstream evaluate().
    interval = np.floor(shifted).astype(int)  # Select the active group of four control points.
    u = shifted - interval  # Express evaluation time as a fraction of the current interval.
    beta = np.column_stack(((5 + 3*u - 3*u*u + u**3) / 6, (1 + 3*u + 3*u*u - 2*u**3) / 6, u**3 / 6))  # Closed forms of upstream tildeB(i,4,t).
    beta_dot = np.column_stack(((1-u)**2 / 2, 0.5 + u - u*u, u*u / 2)) / dt  # Closed forms of upstream B(i,3,t)/dt.
    # 4C - COMPOSE the three weighted turns while carrying the derivative along.
    value = control[interval - 3].copy()  # Start with the first quaternion in the active four-point group.
    derivative = np.zeros_like(value)  # Its time derivative is zero within this interval before multiplying exponentials.
    for term in range(3):  # The cubic cumulative spline contains three exponential increments.
        log_increment = logarithms[interval - 2 + term]  # Select the same local increment used by upstream.
        exponent = log_increment[:, 1:] * beta[:, term, None]  # Scale the logarithm by its cumulative basis coefficient.
        angle = np.linalg.norm(exponent, axis=1)  # Obtain the exponential's half-angle.
        factor = np.sinc(angle / np.pi)  # Calculate sin(angle)/angle with a stable value at zero.
        exponential = np.column_stack((np.cos(angle), exponent * factor[:, None]))  # Evaluate the unit-quaternion exponential.
        exponential_dot = multiply(log_increment, exponential) * beta_dot[:, term, None]  # Differentiate this exponential analytically.
        derivative = multiply(derivative, exponential) + multiply(value, exponential_dot)  # Apply the product rule through all three increments.
        value = multiply(value, exponential)  # Advance the quaternion product.
    # 4D - CONVERT orientation derivative into a local, three-axis gyro reading.
    gyro = 2 * multiply(conjugate(value), derivative)[:, 1:]  # Reproduce upstream's sensor-frame angular rate 2*vector(conj(q)*qdot).
    return value, gyro  # No finite-difference approximation or SciPy RotationSpline is used.

# BLOCK 5: Combine IMUSim position, rotation, valid-domain, and ideal-sensor equations.
# CONTRACT - called by synthesize() after that wrapper validates the inputs.
# INPUT: time (N,), optical position (N,3), sensor-to-world matrices (N,3,3),
# position uncertainty (meters), quaternion uncertainty (dimensionless), gravity.
# OUTPUT: (acc, gyro), both (N,3), aligned with the ORIGINAL input timestamps.
# Only valid interior rows are filled; undefined boundary rows stay NaN.
# No files are read or written. The caller saves and compares these arrays.
# Follow steps 5A-5E below: the position and rotation branches meet in step 5E.
def ideal_imu(time, position, matrices, position_noise=0.001, rotation_noise=0.001, gravity=9.81):  # Accept only optical inputs and explicit physical settings.
    # 5A - TIME: use seconds since the first sample for numerical stability.
    local_time = time - time[0]  # Reduce timestamp magnitude for numerical stability.
    # 5B - POSITION BRANCH: fit a fifth-degree position spline and differentiate
    # twice. Result: acceleration in fixed WORLD axes, before accounting for gravity.
    world_acc = spline_values(local_time, position, position_noise, 5, derivative=2)  # Use upstream's fifth-degree smoothed position trajectory.
    # 5C - ROTATION BRANCH: prepare smooth quaternion controls and their times.
    # The controls represent orientation, not angular velocity yet.
    q = smooth_quaternions(matrices, rotation_noise)  # Use cubic component smoothing instead of the previous Butterworth filter.
    rotation_times = local_time[2:-2] if rotation_noise > 0 else local_time  # Align times after upstream's smoothing-boundary exclusion.
    # 5D - COMMON TIME DOMAIN: keep only times supported by BOTH splines.
    # quaternion_bspline() gives orientation and angular rate at those same times.
    # This mathematical validity mask is separate from the extra edge-seconds
    # margin applied later by process_trial() for comparison.
    valid = (local_time >= max(local_time[3], rotation_times[2])) & (local_time <= min(local_time[-4], rotation_times[-2]))  # Intersect position and quaternion spline valid domains.
    orientation, angular_rate = quaternion_bspline(rotation_times, q, local_time[valid])  # Evaluate only the upstream-defined spline interval.
    rotations = Rotation.from_quat(np.roll(orientation, -1, axis=1))  # Convert scalar-first spline output for vector rotation.
    # 5E - IDEAL SENSOR READINGS: allocate full-size arrays to preserve alignment.
    # Subtracting downward gravity means adding [0,0,+gravity] to world_acc.
    # Then inverse-rotate into sensor axes. Orientation is needed for acceleration;
    # the quaternion spline already supplies angular rate in sensor axes.
    # Both outputs go back through synthesize() to the main script's sensor loop.
    acc, gyro = np.full_like(position, np.nan), np.full_like(position, np.nan)  # Explicitly mark undefined edge samples instead of extrapolating.
    acc[valid] = rotations.inv().apply(world_acc[valid] + [0, 0, gravity])  # Use ideal accelerometer Rᵀ(a-g), with HKNU's upward world Z.
    gyro[valid] = angular_rate  # An ideal gyroscope reads the local angular rate directly.
    return acc, gyro  # Keep the original sample alignment; callers must mask the undefined boundary samples.
