"""
Consistency checks for MixedPIQL (research/algs/piql.py), the PIQL sibling of
MixedCPL trained on MixedFeedbackBuffer's named sub-batches.

1. A one-component mix trains exactly like that component's single-type PIQL
   (CreditAssignmentPIQL for credit_assignment, DemoPIQL for demo) -- the only
   difference is that MixedPIQL flattens (N, T-1) transitions to (N*(T-1),)
   before the IQL update, which changes float reduction order but not the math.
2. A two-component mix's reward loss is the component_weights-weighted sum of
   each single-type sibling's reward loss.
3. A two-component mix with different K and T pools every component's
   transitions into one IQL update.

Self-contained (synthetic obs/action, no real *_labels.npz) so it runs
anywhere torch is installed.

Run with:  pytest tests/test_mixed_piql.py -v
"""

import gym
import numpy as np
import torch

from research.algs.piql import CreditAssignmentPIQL, DemoPIQL, MixedPIQL
from research.networks.base import ActorCriticValueRewardPolicy
from research.networks.mlp import ContinuousMLPActor, ContinuousMLPCritic, MLPValue

OBS_DIM, ACT_DIM = 8, 3
REWARD_STEPS = 2


def make_model(cls, seed=0, **kwargs):
    torch.manual_seed(seed)
    model = cls(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32),
        network_class=ActorCriticValueRewardPolicy,
        dataset_class=None,
        network_kwargs=dict(
            actor_class=ContinuousMLPActor,
            actor_kwargs=dict(dropout=0.0, output_act=torch.nn.Tanh),
            critic_class=ContinuousMLPCritic,
            critic_kwargs=dict(ensemble_size=2),
            value_class=MLPValue,
            value_kwargs=dict(ensemble_size=1),
            reward_class=ContinuousMLPCritic,
            reward_kwargs=dict(ensemble_size=1),
            hidden_layers=[32, 32],
        ),
        optim_class=torch.optim.Adam,
        optim_kwargs=dict(lr=3e-4),
        device="cpu",
        beta=0.3333,
        expectile=0.7,
        reward_steps=REWARD_STEPS,
        **kwargs,
    )
    model.setup_optimizers()
    return model


def make_batch(K, T, B=5, seed=0, variable_label=False):
    g = torch.Generator().manual_seed(seed)
    batch = {
        "obs": torch.randn(B, K, T, OBS_DIM, generator=g),
        "action": torch.rand(B, K, T, ACT_DIM, generator=g) * 2 - 1,
    }
    batch["label"] = torch.randint(0, K, (B,), generator=g) if variable_label else torch.zeros(B)
    return batch


def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.network.parameters()])


def _check_single_component(sibling_cls, name, K, T, variable_label):
    sibling = make_model(sibling_cls)
    mixed = make_model(MixedPIQL, component_weights={name: 1.0})
    assert torch.equal(flat_params(sibling), flat_params(mixed))

    for step in range(2 * REWARD_STEPS):  # covers both the reward-training and frozen-reward phases
        batch = make_batch(K, T, seed=step, variable_label=variable_label)
        m_sibling = sibling.train_step(batch, step, 2 * REWARD_STEPS)
        m_mixed = mixed.train_step({name: batch}, step, 2 * REWARD_STEPS)
        for key in ("q_loss", "v_loss", "actor_loss", "reward"):
            assert np.isclose(m_sibling[key], m_mixed[key], rtol=1e-5, atol=1e-6), (step, key)
        if step < REWARD_STEPS:
            assert np.isclose(m_sibling["reward_loss"], m_mixed["reward_loss"], rtol=1e-6)
            assert np.isclose(m_sibling["reward_accuracy"], m_mixed[f"{name}_reward_accuracy"])

    assert torch.allclose(flat_params(sibling), flat_params(mixed), rtol=1e-4, atol=1e-6)


def test_single_credit_component_matches_credit_assignment_piql():
    _check_single_component(CreditAssignmentPIQL, "credit_assignment", K=6, T=5, variable_label=True)


def test_single_demo_component_matches_demo_piql():
    _check_single_component(DemoPIQL, "demo", K=9, T=12, variable_label=False)


def test_mixed_reward_loss_is_weighted_sum_of_siblings():
    weights = {"credit_assignment": 1.0, "demo": 0.1}
    mixed = make_model(MixedPIQL, component_weights=weights)
    credit = make_model(CreditAssignmentPIQL)
    demo = make_model(DemoPIQL)
    credit.network.load_state_dict(mixed.network.state_dict())
    demo.network.load_state_dict(mixed.network.state_dict())

    credit_batch = make_batch(K=6, T=5, B=7, seed=1, variable_label=True)
    demo_batch = make_batch(K=9, T=12, B=3, seed=2)

    expected = 0.0
    with torch.no_grad():
        for weight, model, batch in ((1.0, credit, credit_batch), (0.1, demo, demo_batch)):
            obs, action, _ = model._get_reward_batch(batch)
            loss, _ = model._get_reward_loss_and_accuracy(model.network.reward(obs, action), batch)
            expected += weight * loss.item()

    logs = mixed.validation_step({"credit_assignment": credit_batch, "demo": demo_batch})
    assert np.isclose(logs["reward_loss"], expected, rtol=1e-6)


def test_mixed_train_step_pools_transitions_across_components():
    mixed = make_model(MixedPIQL, component_weights={"credit_assignment": 1.0, "demo": 0.1})
    credit_batch = make_batch(K=6, T=5, B=7, seed=1, variable_label=True)
    demo_batch = make_batch(K=9, T=12, B=3, seed=2)
    batch = {"credit_assignment": credit_batch, "demo": demo_batch}

    seen = []
    iql_update = mixed._iql_update

    def capture(obs, next_obs, action, discount, reward, step):
        seen.append((obs.shape, next_obs.shape, action.shape, discount.shape, reward.shape))
        return iql_update(obs, next_obs, action, discount, reward, step)

    mixed._iql_update = capture

    n_transitions = 7 * 6 * (5 - 1) + 3 * 9 * (12 - 1)
    for step in range(2 * REWARD_STEPS):
        metrics = mixed.train_step(batch, step, 2 * REWARD_STEPS)
        has_reward_logs = {"reward_loss", "credit_assignment_reward_loss", "demo_reward_loss"} <= metrics.keys()
        assert has_reward_logs == (step < REWARD_STEPS)
        assert all(np.isfinite(v) for v in metrics.values())

    for obs_shape, next_obs_shape, action_shape, discount_shape, reward_shape in seen:
        assert obs_shape == next_obs_shape == (n_transitions, OBS_DIM)
        assert action_shape == (n_transitions, ACT_DIM)
        assert discount_shape == reward_shape == (n_transitions,)
