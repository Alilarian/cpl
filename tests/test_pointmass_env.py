"""
Tests for research/envs/pointmass.py (PointMassGymEnv).

Run with:  pytest tests/test_pointmass_env.py -v
"""

import numpy as np
import pytest
import gym

import research.envs  # triggers gym registration of all envs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(**kwargs) -> "PointMassGymEnv":
    from research.envs.pointmass import PointMassGymEnv
    return PointMassGymEnv(**kwargs)


def rollout(env, policy, max_steps=200):
    """Run one episode; return list of (obs, action, reward, done, info)."""
    obs = env.reset()
    transitions = []
    for _ in range(max_steps):
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        transitions.append((obs, action, reward, done, info))
        obs = next_obs
        if done:
            break
    return transitions


# ---------------------------------------------------------------------------
# 1. Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_shape(self):
        env = make_env()
        assert env.observation_space.shape == (2,)

    def test_observation_space_bounds(self):
        env = make_env()
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    def test_action_space_shape(self):
        env = make_env()
        assert env.action_space.shape == (2,)

    def test_action_space_bounds(self):
        # Normalised [-1, 1]² regardless of max_speed — matches SAC/MetaWorld convention.
        env = make_env(max_speed=0.01)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)

    def test_action_space_symmetric(self):
        env = make_env()
        assert np.allclose(env.action_space.low, -env.action_space.high)

    def test_action_space_independent_of_max_speed(self):
        # max_speed controls displacement magnitude, not the action bounds.
        for speed in [0.01, 0.05, 0.1]:
            env = make_env(max_speed=speed)
            assert np.allclose(env.action_space.low, -1.0)
            assert np.allclose(env.action_space.high, 1.0)


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_float32(self):
        env = make_env()
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_reset_obs_shape(self):
        env = make_env()
        obs = env.reset()
        assert obs.shape == (2,)

    def test_reset_obs_in_bounds(self):
        env = make_env()
        for _ in range(20):
            obs = env.reset()
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_fixed_init_train(self):
        env = make_env(init_pos=[0.0, 0.0])
        for _ in range(5):
            obs = env.reset()
            assert np.allclose(obs, [0.0, 0.0])

    def test_fixed_init_test(self):
        env = make_env(init_pos=[0.99, 0.99])
        for _ in range(5):
            obs = env.reset()
            assert np.allclose(obs, [0.99, 0.99])

    def test_random_init_varies(self):
        env = make_env()
        obs_list = [env.reset() for _ in range(10)]
        # Not all resets should give the same position
        assert not all(np.allclose(obs_list[0], o) for o in obs_list[1:])

    def test_random_init_never_starts_in_goal(self):
        # ~7% of uniform samples land in the goal — rejection sampling must
        # filter them out so every episode starts in a non-terminal state.
        env = make_env()
        for _ in range(200):
            obs = env.reset()
            dist_goal = np.linalg.norm(obs - env.goal)
            assert dist_goal > env.goal_dist_thresh, f"reset landed in goal: {obs}"

    def test_random_init_never_starts_in_trap(self):
        env = make_env()
        for _ in range(200):
            obs = env.reset()
            dist_trap = np.linalg.norm(obs - env.trap)
            assert dist_trap > env.trap_dist_thresh, f"reset landed in trap: {obs}"

    def test_episode_step_counter_resets(self):
        env = make_env()
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        assert env._episode_steps == 5
        env.reset()
        assert env._episode_steps == 0


# ---------------------------------------------------------------------------
# 3. Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_float32_obs(self):
        env = make_env()
        env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())
        assert obs.dtype == np.float32

    def test_step_obs_shape(self):
        env = make_env()
        env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (2,)

    def test_step_obs_stays_in_bounds(self):
        """Agent must never leave the unit square."""
        env = make_env()
        env.reset()
        for _ in range(500):
            obs, _, done, _ = env.step(env.action_space.sample())
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
            if done:
                env.reset()

    def test_step_cost_applied(self):
        """Neutral step (not goal/trap) should return exactly -step_cost."""
        # Place agent far from both goal and trap so terminal reward is 0.
        env = make_env(step_cost=0.01, init_pos=[0.5, 0.99])
        env.reset()
        # Take a zero action so agent doesn't move
        _, reward, done, _ = env.step(np.array([0.0, 0.0]))
        if not done:
            assert np.isclose(reward, -0.01)

    def test_zero_step_cost(self):
        """With step_cost=0 and no terminal, reward should be 0."""
        env = make_env(step_cost=0.0, init_pos=[0.5, 0.99])
        env.reset()
        _, reward, done, _ = env.step(np.array([0.0, 0.0]))
        if not done:
            assert np.isclose(reward, 0.0)

    def test_action_clipping(self):
        """Actions outside [-1,1] bounds should be clipped, not raise errors."""
        env = make_env()
        env.reset()
        oversized = np.array([10.0, 10.0])  # way outside [-1, 1]
        obs, _, _, _ = env.step(oversized)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_timeout_sets_discount(self):
        """Hitting max_ep_len without terminal should set info['discount']=1."""
        env = make_env(max_ep_len=5, init_pos=[0.5, 0.99], step_cost=0.0)
        env.reset()
        done = False
        info = {}
        for _ in range(5):
            _, _, done, info = env.step(np.array([0.0, 0.0]))
        assert done
        assert info.get("discount") == 1.0

    def test_info_keys(self):
        env = make_env()
        env.reset()
        _, _, _, info = env.step(env.action_space.sample())
        assert "succ" in info
        assert "crash" in info
        assert "goal" in info

    def test_episode_step_increments(self):
        env = make_env()
        env.reset()
        for i in range(1, 6):
            env.step(np.array([0.0, 0.0]))
            assert env._episode_steps == i


# ---------------------------------------------------------------------------
# 4. Terminal detection
# ---------------------------------------------------------------------------

class TestTerminalDetection:
    def test_goal_terminates_episode(self):
        """Agent at the goal should get done=True and succ=True."""
        env = make_env(goal=[0.3, 0.3], step_cost=0.0)
        # Place agent just inside goal radius
        env.reset()
        env._pos = np.array([0.3, 0.3])
        _, reward, done, info = env.step(np.array([0.0, 0.0]))
        assert done
        assert info["succ"] is True
        assert info["crash"] is False
        assert reward > 0

    def test_trap_terminates_episode(self):
        """Agent at the trap should get done=True and crash=True."""
        env = make_env(trap=[0.7, 0.7], step_cost=0.0)
        env.reset()
        env._pos = np.array([0.7, 0.7])
        _, reward, done, info = env.step(np.array([0.0, 0.0]))
        assert done
        assert info["crash"] is True
        assert info["succ"] is False
        assert reward < 0

    def test_succ_crash_mutually_exclusive(self):
        """goal and trap must not overlap by construction."""
        with pytest.raises(ValueError):
            # goal and trap at same spot — should raise
            make_env(goal=[0.5, 0.5], trap=[0.5, 0.5])


# ---------------------------------------------------------------------------
# 5. reward_func (vectorised oracle)
# ---------------------------------------------------------------------------

class TestRewardFunc:
    def test_goal_gives_positive_reward(self):
        env = make_env(goal=[0.3, 0.3], succ_rew_bonus=1.0)
        obs = np.array([[0.3, 0.3]])
        r = env.reward_func(obs, np.zeros((1, 2)), obs)
        assert r[0] == pytest.approx(1.0)

    def test_trap_gives_negative_reward(self):
        env = make_env(trap=[0.7, 0.7], crash_rew_penalty=-10.0)
        obs = np.array([[0.7, 0.7]])
        r = env.reward_func(obs, np.zeros((1, 2)), obs)
        assert r[0] == pytest.approx(-10.0)

    def test_neutral_gives_zero_reward(self):
        env = make_env()
        obs = np.array([[0.5, 0.99]])
        r = env.reward_func(obs, np.zeros((1, 2)), obs)
        assert r[0] == pytest.approx(0.0)

    def test_vectorised_batch(self):
        env = make_env(goal=[0.3, 0.3], trap=[0.7, 0.7])
        obses = np.array([[0.3, 0.3], [0.5, 0.99], [0.7, 0.7]])
        acts = np.zeros((3, 2))
        r = env.reward_func(obses, acts, obses)
        assert r.shape == (3,)
        assert r[0] == pytest.approx(1.0)
        assert r[1] == pytest.approx(0.0)
        assert r[2] == pytest.approx(-10.0)

    def test_reward_func_excludes_step_cost(self):
        """reward_func should return pure task reward, never step_cost."""
        env = make_env(step_cost=0.99)
        obs = np.array([[0.5, 0.99]])
        r = env.reward_func(obs, np.zeros((1, 2)), obs)
        # Neutral: reward_func = 0, not -0.99
        assert r[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. get_state / set_state
# ---------------------------------------------------------------------------

class TestStateSerialisation:
    def test_get_state_shape(self):
        env = make_env()
        env.reset()
        state = env.get_state()
        assert state.shape == (3,)  # [x, y, step]

    def test_get_state_dtype(self):
        env = make_env()
        env.reset()
        state = env.get_state()
        assert state.dtype == np.float32

    def test_set_state_restores_position(self):
        env = make_env()
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
        state = env.get_state()

        env.reset()  # move to a different position
        obs = env.set_state(state)
        assert np.allclose(obs, state[:2])
        assert np.allclose(env._pos, state[:2])

    def test_set_state_restores_step_counter(self):
        env = make_env()
        env.reset()
        for _ in range(7):
            env.step(env.action_space.sample())
        state = env.get_state()

        env.reset()
        env.set_state(state)
        assert env._episode_steps == 7

    def test_roundtrip_produces_same_trajectory(self):
        """Two runs from the same state with the same actions must match."""
        env = make_env(init_pos=[0.5, 0.5])
        env.reset()
        state = env.get_state()
        actions = [env.action_space.sample() for _ in range(10)]

        # First run
        env.set_state(state)
        traj_a = [env.step(a)[0].copy() for a in actions]

        # Second run from same state
        env.set_state(state)
        traj_b = [env.step(a)[0].copy() for a in actions]

        for a, b in zip(traj_a, traj_b):
            assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# 7. Expert & noisy-expert policies
# ---------------------------------------------------------------------------

class TestPolicies:
    def test_expert_moves_toward_goal(self):
        """Expert action from a point far from both goal and trap should
        reduce distance to goal after scaling by max_speed."""
        env = make_env(goal=[0.3, 0.3], trap=[0.7, 0.7])
        expert = env.make_expert_policy()
        pos = np.array([0.99, 0.99])
        action = expert(pos)   # in [-1,1]²
        # Apply the same scaling step() uses: velocity = action * max_speed
        new_pos = pos + action * env.max_speed
        assert np.linalg.norm(new_pos - env.goal) < np.linalg.norm(pos - env.goal)

    def test_expert_action_in_bounds(self):
        env = make_env()
        expert = env.make_expert_policy()
        for _ in range(50):
            obs = env.observation_space.sample()
            action = expert(obs)
            assert action in env.action_space

    def test_noisy_expert_action_in_bounds(self):
        env = make_env()
        noisy = env.make_noisy_expert_policy(eps=0.5)
        for _ in range(50):
            obs = env.observation_space.sample()
            action = noisy(obs)
            assert action in env.action_space

    def test_expert_reaches_goal(self):
        """Expert from a fixed start position should reach the goal."""
        env = make_env(goal=[0.3, 0.3], trap=[0.7, 0.7],
                       init_pos=[0.0, 0.0], step_cost=0.0, max_ep_len=500)
        expert = env.make_expert_policy()
        transitions = rollout(env, expert, max_steps=500)
        final_info = transitions[-1][-1]
        assert final_info["succ"] is True, "expert should reach goal from (0,0)"


# ---------------------------------------------------------------------------
# 8. Gym registration
# ---------------------------------------------------------------------------

class TestGymRegistration:
    def test_default_env_registers(self):
        env = gym.make("pm_navigation-v0")
        assert env.observation_space.shape == (2,)
        assert env.action_space.shape == (2,)

    def test_train_variant_fixed_init(self):
        env = gym.make("pm_navigation_train-v0")
        obs = env.reset()
        assert np.allclose(obs, [0.0, 0.0])

    def test_test_variant_fixed_init(self):
        env = gym.make("pm_navigation_test-v0")
        obs = env.reset()
        assert np.allclose(obs, [0.99, 0.99])

    def test_sparse_variant_no_step_cost(self):
        env = gym.make("pm_navigation_sparse-v0")
        # Sparse variant: place agent away from both regions, take zero step
        env.reset()
        env._pos = np.array([0.5, 0.99])
        _, reward, done, _ = env.step(np.array([0.0, 0.0]))
        if not done:
            assert np.isclose(reward, 0.0), "sparse variant should have zero step cost"
