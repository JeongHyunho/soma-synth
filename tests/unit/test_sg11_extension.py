import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from soma_synth.experimental.sg11.pipeline import sg11, synthesize, SITES


def test_constant_acceleration_and_rotation():
    t = np.arange(301)*.01
    p = .5*t[:, None]**2*np.array([.2, 0, 0])
    r = np.tile(np.eye(3), (len(t), 1, 1))
    z, q = sg11(t, p, r)
    np.testing.assert_allclose(z[73:-73, :3], np.tile([.2, 0, 9.81], (155, 1)), atol=1e-6)
    np.testing.assert_allclose(z[73:-73, 3:], 0, atol=1e-8)
    r = Rotation.from_rotvec(t[:, None]*[0, 0, .8]).as_matrix()
    z, q = sg11(t, p*0, r)
    np.testing.assert_allclose(z[73:-73, 3:], np.tile([0, 0, .8], (155, 1)), atol=.002)
    np.testing.assert_allclose(np.linalg.norm(q[73:-73], axis=1), 1, atol=1e-8)


def fixture():
    t = np.arange(700)*.01
    m = np.broadcast_to(np.array([[1., 1, 1], [1.1, 1, 1], [1, 1.1, 1]]), (700, 8, 3, 3)).copy()
    return t, m, np.ones(700, bool)


def test_gap_isolation_and_site_guard():
    t, m, mask = fixture()
    mask[330:370] = False
    x, _, valid = synthesize(t, m, mask, SITES)
    m[330:370] = 10000
    y, _, other = synthesize(t, m, mask, SITES)
    np.testing.assert_array_equal(valid, other)
    np.testing.assert_allclose(x[valid], y[valid])
    assert not valid[257:443].any()
    with pytest.raises(ValueError, match='site order'):
        synthesize(t, m, mask, ('TA', 'LHA', *SITES[2:]))


def test_rate_guard():
    t, m, mask = fixture()
    with pytest.raises(ValueError, match='100-Hz'):
        synthesize(t*2, m, mask, SITES)


def test_pose_zero_head_identity():
    torch = pytest.importorskip('torch')
    from soma_synth.experimental.sg11.pose_model import PoseCorrection
    model = PoseCorrection(np.zeros(48, np.float32), np.ones(48, np.float32)).eval()
    x = torch.randn(1, 48, 100)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), torch.zeros_like(x))


def test_checkpoint_contract_and_gap(tmp_path):
    torch = pytest.importorskip('torch')
    import hashlib
    import json
    from soma_synth.experimental.sg11.pose_model import PoseCorrection
    from soma_synth.experimental.sg11.pipeline import correct
    model = PoseCorrection(np.zeros(48, np.float32), np.ones(48, np.float32))
    path = tmp_path/'best.pt'
    torch.save({'state_dict': model.state_dict()}, path)
    info = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sites=list(SITES), sample_rate_hz=100, physics='SG11',
                architecture='pose_correction', training_participants=['fixture'])
    meta = tmp_path/'checkpoint.json'
    meta.write_text(json.dumps(info))
    x = np.random.default_rng(4).normal(size=(160, 48))
    valid = np.ones(160, bool)
    valid[70:80] = False
    y, mask, _ = correct(x, valid, path, meta)
    np.testing.assert_allclose(y[mask], x[mask], atol=1e-6)
    assert not mask[55:95].any()
    info['sites'][1] = 'LHA'
    meta.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='mismatch'):
        correct(x, valid, path, meta)
