"""
2-D navigation environment for CPL.

Standalone re-implementation of ReQueST's PointMassEnv (ReQueST/rqst/envs.py)
that drops the TensorFlow / sklearn / scipy dependencies that the original
file carries (needed for its other environments — MNIST, CarRacing — but not
for PointMass).

State  : agent position (x, y) ∈ [0, 1]²
Action : Cartesian velocity [vx, vy] ∈ [-max_speed, max_speed]²

Why Cartesian instead of the original polar (angle, speed)?
  SAC uses a Gaussian policy N(μ(s), Σ(s)).  A Gaussian is well-defined over
  R² but not over angle (circular topology).  Cartesian keeps the action space
  a proper Box that a tanh-squashed Gaussian handles correctly.
"""

from copy import deepcopy

import numpy as np
import gym
from gym import spaces


class PointMassGymEnv(gym.Env):
    """2-D navigation: reach the goal region, avoid the trap.

    Reward modes
    ------------
    shaped  (step_cost > 0, default 0.01)
        +succ_rew_bonus   when reaching the goal  (default +1)
        +crash_rew_penalty when entering the trap (default -10)
        -step_cost        every step
        Dense signal so SAC learns direction before any terminal event.

    sparse  (step_cost = 0.0)
        Terminal rewards only.
        Used when computing preference labels so that segment returns
        reflect only task success, not intermediate shaping.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        max_speed: float = 0.01,
        goal=None,
        trap=None,
        init_pos=None,
        step_cost: float = 0.01,
        max_ep_len: int = 1000,
        goal_dist_thresh: float = 0.15,
        trap_dist_thresh: float = 0.15,
        succ_rew_bonus: float = 100.0,
        crash_rew_penalty: float = -50.0,
    ):
        """
        Parameters
        ----------
        goal, trap : array-like (2,) or None
            Positions in [0,1]².  Defaults match ReQueST's make_pointmass_env:
            goal=[0.3,0.3], trap=[0.7,0.7].
        init_pos : array-like (2,) or None
            Fixed starting position.  None → uniform random each reset.
            Pass [0.0, 0.0] for train variant, [0.99, 0.99] for test variant.
        step_cost : float
            Per-step penalty.  0.0 gives purely sparse/terminal rewards.
        """
        if goal is None:
            goal = [0.3, 0.3]
        if trap is None:
            trap = [0.7, 0.7]

        self.goal = np.array(goal, dtype=np.float64)
        self.trap = np.array(trap, dtype=np.float64)
        self.init_pos = None if init_pos is None else np.array(init_pos, dtype=np.float64)

        if np.linalg.norm(self.goal - self.trap) < 2 * goal_dist_thresh:
            raise ValueError("goal and trap regions overlap — increase distance or reduce thresholds")

        self.max_speed = max_speed
        self.step_cost = step_cost
        self.max_ep_len = max_ep_len
        self._max_episode_steps = max_ep_len
        self.goal_dist_thresh = goal_dist_thresh
        self.trap_dist_thresh = trap_dist_thresh
        self.succ_rew_bonus = succ_rew_bonus
        self.crash_rew_penalty = crash_rew_penalty

        self._episode_steps = 0
        self._pos = np.zeros(2, dtype=np.float64)

        # --- Observation space: (x, y) ∈ [0, 1]² ---
        self.observation_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # --- Action space: normalised velocity [vx, vy] ∈ [-1, 1]² ---
        # Kept at [-1, 1]² (the SAC / MetaWorld convention) so that the
        # DiagonalGaussianMLPActor with squash_normal=True maps directly onto
        # it and log-probabilities are correctly scaled.
        # Actual displacement = action * max_speed, applied inside step().
        self.action_space = spaces.Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _in_goal(self, pos: np.ndarray) -> bool:
        return float(np.linalg.norm(pos - self.goal)) <= self.goal_dist_thresh

    def _in_trap(self, pos: np.ndarray) -> bool:
        return float(np.linalg.norm(pos - self.trap)) <= self.trap_dist_thresh

    # ------------------------------------------------------------------
    # Reward function (vectorised — used by oracle labeling scripts)
    # ------------------------------------------------------------------

    def reward_func(self, obses: np.ndarray, acts: np.ndarray, next_obses: np.ndarray) -> np.ndarray:
        """Analytic terminal reward.  Inputs: (N, 2) → output: (N,).

        +succ_rew_bonus   if next_obs is inside the goal radius
        +crash_rew_penalty if next_obs is inside the trap radius
        0                 otherwise

        Note: step_cost is NOT included here.  This is the pure task
        reward used by oracle advantage estimators.
        """
        dist_goal = np.linalg.norm(next_obses - self.goal, axis=1)
        dist_trap = np.linalg.norm(next_obses - self.trap, axis=1)
        at_goal = (dist_goal <= self.goal_dist_thresh).astype(float)
        at_trap = (dist_trap <= self.trap_dist_thresh).astype(float)
        return self.succ_rew_bonus * at_goal + self.crash_rew_penalty * at_trap

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        # Clip to declared bounds (protects against floating-point drift
        # from tanh-squashed Gaussian policies).
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )

        # Scale normalised action to actual displacement.
        # action ∈ [-1,1]² → velocity ∈ [-max_speed, max_speed]²
        velocity = action * self.max_speed

        # Apply velocity; stay inside unit square.
        new_pos = self._pos + velocity
        if (new_pos >= 0.0).all() and (new_pos < 1.0).all():
            self._pos = new_pos

        succ = self._in_goal(self._pos)
        crash = self._in_trap(self._pos)

        # Terminal reward + optional per-step penalty.
        obs_batch = self._pos[np.newaxis, :]
        act_batch = action[np.newaxis, :]
        reward = float(self.reward_func(obs_batch, act_batch, obs_batch)[0])
        reward -= self.step_cost

        done = succ or crash
        info = {"succ": succ, "crash": crash, "goal": self.goal.copy()}

        self._episode_steps += 1
        if self._episode_steps >= self.max_ep_len and not done:
            done = True
            # Timeout — not a true terminal state; tell SAC to bootstrap.
            info["discount"] = 1.0

        return self._pos.astype(np.float32).copy(), reward, done, info

    def reset(self, **kwargs):
        self._episode_steps = 0
        if self.init_pos is not None:
            self._pos = self.init_pos.copy()
        else:
            # Rejection-sample until the start position is outside both
            # the goal and trap regions.  Without this, ~14% of random
            # resets land inside a terminal region and the episode ends
            # on the very first step, producing degenerate transitions.
            while True:
                self._pos = np.random.random(2)
                if not self._in_goal(self._pos) and not self._in_trap(self._pos):
                    break
        return self._pos.astype(np.float32).copy()

    # ------------------------------------------------------------------
    # State serialisation (needed for segment replay during labeling)
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Serialise full env state → flat float32 array [x, y, step].

        The episode step counter is stored so that set_state restores the
        timeout counter exactly — important when replaying a segment that
        starts mid-episode.
        """
        return np.array(
            [self._pos[0], self._pos[1], float(self._episode_steps)],
            dtype=np.float32,
        )

    def set_state(self, state: np.ndarray) -> np.ndarray:
        """Restore env to a previously saved state; return current obs."""
        self._pos = state[:2].astype(np.float64).copy()
        self._episode_steps = int(state[2])
        return self._pos.astype(np.float32).copy()

    # ------------------------------------------------------------------
    # Heuristic expert policy (useful for generating suboptimal dataset)
    # ------------------------------------------------------------------

    def make_expert_policy(self, safety_margin: float = 0.05):
        """Heuristic that goes straight to goal and swings around the trap.

        Returns a callable: obs (2,) → Cartesian action (2,).
        Mirrors the logic from ReQueST's PointMassEnv.make_expert_policy().
        """
        goal = self.goal
        trap = self.trap
        trap_thresh = self.trap_dist_thresh
        max_speed = self.max_speed

        def policy(obs: np.ndarray) -> np.ndarray:
            obs = np.asarray(obs, dtype=np.float64)
            direction = goal - obs
            dist = np.linalg.norm(direction)
            if dist < 1e-9:
                return np.zeros(2, dtype=np.float32)

            unit = direction / dist
            # Check whether the straight-line path clips the trap.
            v = trap - obs
            proj = v.dot(unit)
            closest = obs + unit * proj
            if (
                proj > 0
                and np.linalg.norm(v) < trap_thresh + safety_margin
                and np.linalg.norm(closest - trap) < trap_thresh + safety_margin
            ):
                # Swing 90° around the trap.
                angle = np.arctan2(v[1], v[0]) + 0.5 * np.pi
                unit = np.array([np.cos(angle), np.sin(angle)])

            # Return a unit vector in [-1,1]²; step() scales by max_speed.
            return unit.astype(np.float32)

        return policy

    def make_noisy_expert_policy(self, eps: float = 0.5):
        """Expert with eps-fraction random actions (for suboptimal dataset)."""
        expert = self.make_expert_policy()

        def policy(obs: np.ndarray) -> np.ndarray:
            if np.random.random() < eps:
                return self.action_space.sample()
            return expert(obs)

        return policy

    # ------------------------------------------------------------------
    # Rendering (optional — for visual debugging)
    # ------------------------------------------------------------------

    def render(self, mode: str = "rgb_array", width: int = 256, height: int = 256):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4), dpi=width // 4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("PointMass Navigation")

        ax.add_patch(plt.Circle(self.goal, self.goal_dist_thresh, color="green", alpha=0.4, label="goal"))
        ax.add_patch(plt.Circle(self.trap, self.trap_dist_thresh, color="red", alpha=0.4, label="trap"))
        ax.plot(*self._pos, "bo", markersize=10, label="agent")
        ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        return img
