"""Training loop on a tiny synthetic choice set: learns, exports, and pilot `best` == normal `best`."""
import json
import os

import numpy as np
import torch

from research.reward_learning.data import ChoiceSet, make_episode_split
from research.reward_learning.ensemble import RewardEnsemble
from research.reward_learning.train import train_reward_ensemble

CFG = dict(model=dict(ensemble_size=3, hidden_layers=[16, 16], output="tanh"), rrc=dict(beta=1.0),
           train=dict(lr=3e-3, batch_acts=32, grad_clip=1.0, max_epochs=12, patience=3, save_every=5, pilot_max_epochs=40))


def _choice_set(n=300, K=3, L=6, seed=0):
    rng = np.random.RandomState(seed)
    obs = rng.randn(n, K, L, 35).astype(np.float32)
    act = rng.uniform(-1, 1, (n, K, L, 4)).astype(np.float32)
    true_r = obs[..., 0] * 0.5  # a learnable ground-truth reward
    chosen = (true_r.sum(-1)).argmax(1)
    target = np.eye(K, dtype=np.float32)[chosen]
    eps = rng.randint(0, 30, (n, 1))
    return ChoiceSet(type="synthetic", kind="stacked", obs=obs, act=act, true_reward=true_r, target=target, gamma=1.0,
                     episodes=eps, row_index=np.arange(n), act_ids=np.arange(n))


def test_training_learns_and_pilot_best_matches_normal_best(tmp_path):
    cs = _choice_set()
    split = make_episode_split(cs.episodes, 30, 0.2, 0)
    s_norm = train_reward_ensemble(cs, split, CFG, str(tmp_path / "normal"), seed=0, device="cpu")
    s_pilot = train_reward_ensemble(cs, split, CFG, str(tmp_path / "pilot"), seed=0, device="cpu", pilot=True)
    assert s_norm["best_epochs"] == s_pilot["best_epochs"]
    assert s_norm["early_stop_epoch"] == s_pilot["early_stop_epoch"]
    a = RewardEnsemble.load(str(tmp_path / "normal" / "best.pt"))
    b = RewardEnsemble.load(str(tmp_path / "pilot" / "best.pt"))
    for (k, va), (_, vb) in zip(a.net.state_dict().items(), b.net.state_dict().items()):
        assert torch.allclose(va, vb), k
    for name in ("epoch3x", "best_ens", "last"):
        assert os.path.exists(tmp_path / "pilot" / f"{name}.pt")
    e3 = RewardEnsemble.load(str(tmp_path / "pilot" / "epoch3x.pt")).extra["member_epochs"]
    assert e3 == [min(3 * s_pilot["ens_best_epoch"], s_pilot["epochs_run"])] * 3
    assert not os.path.exists(tmp_path / "pilot" / "_epochs")
    # it actually learned: held-out NLL well below the uniform-choice NLL log(3)
    assert min(s_norm["best_val_nll"]) < 0.8 * np.log(3)
    # standardization: mean ~0, std ~1 over the training steps
    obs, act, _ = cs.subset(split["train_idx"]).valid_steps()
    z = a.standardized_members(obs, act)
    assert torch.allclose(z.mean(1), torch.zeros(3), atol=1e-3) and torch.allclose(z.std(1, unbiased=False), torch.ones(3), atol=1e-3)
    with open(tmp_path / "normal" / "train_summary.json") as f:
        assert json.load(f)["exports"]["best"]["member_epochs"] == s_norm["best_epochs"]
