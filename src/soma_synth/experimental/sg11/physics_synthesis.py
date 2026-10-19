"""SG11 specific-force synthesis and quaternion-spline angular velocity."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from . import quaternion_spline as ic


def sg11(time, position, rotation):
    """Match the historical SG11 kernel; output specific force and rad/s."""
    time = np.asarray(time, dtype=float)
    if len(time) < 11 or not np.allclose(np.diff(time), .01, rtol=0, atol=1e-8):
        raise ValueError('SG11 requires at least 11 uniformly spaced 100-Hz frames')
    time = time - time[0]
    q = ic.smooth_quaternions(rotation, .001)
    qt = time[2:-2]
    valid = (time >= max(time[3], qt[2])) & (time <= min(time[-4], qt[-2]))
    qv, w = ic.quaternion_bspline(qt, q, time[valid])
    r = Rotation.from_quat(np.roll(qv, -1, axis=1))
    a = savgol_filter(position-position[:1], 11, 5, deriv=2, delta=.01, axis=0, mode='interp')
    signal = np.full((len(time), 6), np.nan)
    orientation = np.full((len(time), 4), np.nan)
    signal[valid, :3] = r.inv().apply(a[valid] + [0, 0, 9.81])
    signal[valid, 3:] = w
    orientation[valid] = qv
    return signal, orientation
