"""
LunarLander continuous environment wrapper for CPL.

Uses gymnasium LunarLander-v3 internally with:
  - Wind and turbulence disabled (fully deterministic physics)
  - Continuous action space (2-dim: main engine + lateral)
  - gym 0.23 API exposed outward (4-tuple step, single obs from reset)

gym 0.23's box2d is incompatible with numpy 2.x (b2Vec2 conversion
error). gymnasium's box2d package handles numpy 2.x correctly, so
gymnasium is used internally while the outward-facing API stays gym 0.23.

Requires: pip install "gymnasium[box2d]"

Implements get_state() / set_state() following the same pattern as
MetaWorldSawyerEnv so the env can be used with CPL's dataset collection
and state-restore pipeline.
"""

import gym
import numpy as np


class LunarLanderEnv(gym.Env):
    def __init__(self, sparse: bool = False, horizon: int = 1000):
        try:
            import gymnasium as gymn
            from gymnasium.envs.box2d.lunar_lander import FPS, LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "gymnasium is required for LunarLanderEnv.\n"
                "Install it with: pip install \"gymnasium[box2d]\""
            )

        self.FPS = FPS
        self.LEG_DOWN = LEG_DOWN
        self.SCALE = SCALE
        self.VIEWPORT_H = VIEWPORT_H
        self.VIEWPORT_W = VIEWPORT_W

        # Wind and turbulence explicitly disabled
        self._env = gymn.make(
            "LunarLander-v3",
            continuous=True,
            enable_wind=False,
            turbulence_power=0.0,
        )

        self.sparse = sparse
        self._max_episode_steps = min(horizon, self._env.spec.max_episode_steps or horizon)
        self._episode_steps = 0

        # Expose gym-compatible spaces (identical shape/dtype to gymnasium version)
        self.observation_space = gym.spaces.Box(
            low=self._env.observation_space.low,
            high=self._env.observation_space.high,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=self._env.action_space.low,
            high=self._env.action_space.high,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Core gym 0.23 API (4-tuple step, single obs from reset)
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._episode_steps = 0
        obs, _ = self._env.reset(**kwargs)
        return obs.astype(np.float32)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)

        done = terminated or truncated
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Infinite-horizon bootstrap, same as MetaWorld

        if self.sparse:
            reward = float(info.get("success", 0.0))

        return obs.astype(np.float32), float(reward), done, info

    def render(self, mode="rgb_array", **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._env.reset(seed=seed)

    # ------------------------------------------------------------------
    # State save / restore  (required for CPL dataset collection)
    # Ported from multi-type-feedback LunarLanderSaveLoadWrapper.
    # ------------------------------------------------------------------

    def get_state(self):
        uw = self._env.unwrapped
        if uw.lander is None:
            raise ValueError("Lander does not exist — call reset() first.")
        return {
            "lander": {
                "position": (float(uw.lander.position.x), float(uw.lander.position.y)),
                "angle": float(uw.lander.angle),
                "linearVelocity": (
                    float(uw.lander.linearVelocity.x),
                    float(uw.lander.linearVelocity.y),
                ),
                "angularVelocity": float(uw.lander.angularVelocity),
            },
            "legs": [
                {
                    "position": (float(leg.position.x), float(leg.position.y)),
                    "angle": float(leg.angle),
                    "ground_contact": bool(leg.ground_contact),
                }
                for leg in uw.legs
            ],
            "prev_shaping": float(uw.prev_shaping) if uw.prev_shaping is not None else None,
            "episode_steps": self._episode_steps,
        }

    def set_state(self, state):
        uw = self._env.unwrapped
        ls = state["lander"]
        uw.lander.position = ls["position"]
        uw.lander.angle = ls["angle"]
        uw.lander.linearVelocity = ls["linearVelocity"]
        uw.lander.angularVelocity = ls["angularVelocity"]

        for leg, saved_leg in zip(uw.legs, state["legs"]):
            leg.position = saved_leg["position"]
            leg.angle = saved_leg["angle"]
            leg.ground_contact = saved_leg["ground_contact"]

        uw.prev_shaping = state.get("prev_shaping", None)
        self._episode_steps = state.get("episode_steps", 0)

        return self._get_obs()

    # ------------------------------------------------------------------
    # Observation helper  (replicates gymnasium LunarLander._get_obs)
    # ------------------------------------------------------------------

    def _get_obs(self):
        uw = self._env.unwrapped
        pos = uw.lander.position
        vel = uw.lander.linearVelocity
        return np.array(
            [
                (pos.x - self.VIEWPORT_W / self.SCALE / 2) / (self.VIEWPORT_W / self.SCALE / 2),
                (pos.y - (uw.helipad_y + self.LEG_DOWN / self.SCALE)) / (self.VIEWPORT_H / self.SCALE / 2),
                vel.x * (self.VIEWPORT_W / self.SCALE / 2) / self.FPS,
                vel.y * (self.VIEWPORT_H / self.SCALE / 2) / self.FPS,
                uw.lander.angle,
                20.0 * uw.lander.angularVelocity / self.FPS,
                1.0 if uw.legs[0].ground_contact else 0.0,
                1.0 if uw.legs[1].ground_contact else 0.0,
            ],
            dtype=np.float32,
        )

    def get_obs(self):
        return self._get_obs()

    def __getattr__(self, name):
        return getattr(self._env, name)
