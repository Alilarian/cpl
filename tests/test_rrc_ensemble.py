"""Vectorized ensemble: member independence, per-member clipping, standardization, save/load, numpy twin."""
import numpy as np
import torch

from research.reward_learning.ensemble import EnsembleMLP, RewardEnsemble, clip_grad_norm_per_member
from research.reward_learning.envs import NumpyRewardModel


def _net(**kw):
    return EnsembleMLP(35, 4, ensemble_size=3, hidden_layers=(32, 32), seeds=[10, 11, 12], **kw)


def test_member_init_is_seeded_and_distinct():
    a, b = _net(), _net()
    for la, lb in zip(a.layers, b.layers):
        assert torch.equal(la.weight, lb.weight)
    assert not torch.allclose(a.layers[0].weight[0], a.layers[0].weight[1])


def test_member_major_forward_equals_per_member_forward():
    net = _net()
    obs, act = torch.randn(3, 5, 7, 35), torch.randn(3, 5, 7, 4)
    out = net(obs, act, member_major=True)
    assert out.shape == (3, 5, 7)
    for m in range(3):
        shared = net(obs[m], act[m])  # all members on member m's batch
        assert torch.allclose(out[m], shared[m], atol=1e-6)


def test_members_get_independent_gradients():
    net = _net()
    obs, act = torch.randn(3, 8, 35), torch.randn(3, 8, 4)
    out = net(obs, act, member_major=True)
    out[1].sum().backward()  # loss touches member 1 only
    for layer in net.layers:
        assert torch.all(layer.weight.grad[0] == 0) and torch.all(layer.weight.grad[2] == 0)
        assert layer.weight.grad[1].abs().sum() > 0


def test_per_member_grad_clipping():
    net = _net()
    obs, act = torch.randn(3, 8, 35), torch.randn(3, 8, 4)
    out = net(obs, act, member_major=True)
    (out[0].sum() * 1000.0 + out[1].sum() * 1e-4).backward()
    norms = clip_grad_norm_per_member(net, 1.0)
    assert norms[0] > 1.0 and norms[1] < 1.0
    sq = torch.zeros(3)
    for layer in net.layers:
        for p in (layer.weight, layer.bias):
            sq += p.grad.reshape(3, -1).pow(2).sum(1)
    after = sq.sqrt()
    assert torch.isclose(after[0], torch.tensor(1.0), atol=1e-4)  # clipped
    assert torch.isclose(after[1], norms[1], rtol=1e-5)  # untouched


def test_standardization_save_load_and_numpy_twin(tmp_path):
    net = _net()
    mu, sigma = torch.tensor([0.1, -0.2, 0.3]), torch.tensor([0.5, 2.0, 1.5])
    model = RewardEnsemble(net, mu, sigma)
    obs, act = np.random.randn(11, 35).astype(np.float32), np.random.uniform(-1, 1, (11, 4)).astype(np.float32)
    raw = model.member_rewards(obs, act)
    expected = ((raw - mu[:, None]) / sigma[:, None]).mean(0)
    assert torch.allclose(model.reward(obs, act), expected, atol=1e-6)
    path = str(tmp_path / "r.pt")
    model.save(path, extra=dict(mu_true=0.6, sigma_true=0.3))
    loaded = RewardEnsemble.load(path)
    assert torch.allclose(loaded.reward(obs, act), expected, atol=1e-6)
    assert loaded.extra["mu_true"] == 0.6
    npm = NumpyRewardModel(path)
    r_np, std_np, _ = npm.reward_and_std(obs, act)
    r_t, std_t = loaded.reward_and_std(obs, act)
    assert np.allclose(r_np, r_t.numpy(), atol=1e-5) and np.allclose(std_np, std_t.numpy(), atol=1e-5)


def test_tanh_bounds_and_linear_ablation():
    x, a = torch.randn(100, 35) * 50, torch.randn(100, 4)
    assert _net()(x, a).abs().max() <= 1.0
    assert _net(output="linear")(x, a).abs().max() > 1.0
