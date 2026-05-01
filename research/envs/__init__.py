# Register environment classes here
# Register the environments
from gym.envs import register

from .base import EmptyEnv

try:
    from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

    for env_name in ALL_V2_ENVIRONMENTS.keys():
        ID = f"mw_{env_name}"
        register(id=ID, entry_point="research.envs.metaworld:MetaWorldSawyerEnv", kwargs={"env_name": env_name})
        id_parts = ID.split("-")
        id_parts[-1] = "image-" + id_parts[-1]
        ID = "-".join(id_parts)
        register(id=ID, entry_point="research.envs.metaworld:get_mw_image_env", kwargs={"env_name": env_name})
except ImportError:
    print("[research] Warning: Could not import MetaWorld Environments.")

try:
    # Random init, shaped reward (step_cost=0.01) — used for SAC oracle training.
    register(id="pm_navigation-v0", entry_point="research.envs.pointmass:PointMassGymEnv", kwargs={})
    # Fixed init at (0,0) — training distribution.
    register(id="pm_navigation_train-v0", entry_point="research.envs.pointmass:PointMassGymEnv",
             kwargs={"init_pos": [0.0, 0.0]})
    # Fixed init at (0.99,0.99) — test distribution (distribution-shift evaluation).
    register(id="pm_navigation_test-v0", entry_point="research.envs.pointmass:PointMassGymEnv",
             kwargs={"init_pos": [0.99, 0.99]})
    # Sparse reward variants (no step cost) — for preference labeling oracle.
    register(id="pm_navigation_sparse-v0", entry_point="research.envs.pointmass:PointMassGymEnv",
             kwargs={"step_cost": 0.0})
except Exception:
    print("[research] Warning: Could not register PointMass environments.")

try:
    register(
        id="LunarLanderContinuous-cpl-v0",
        entry_point="research.envs.lunar_lander:LunarLanderEnv",
        kwargs={"sparse": False},
    )
    register(
        id="LunarLanderContinuous-cpl-sparse-v0",
        entry_point="research.envs.lunar_lander:LunarLanderEnv",
        kwargs={"sparse": True},
    )
except Exception:
    print("[research] Warning: Could not register LunarLander environments.")
