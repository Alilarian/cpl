"""
Visualise PointMass rollouts.

Rolls out several trajectories under different policies and renders them
on a single plot, then saves to a PNG.

Usage:
    python scripts/visualize_pointmass.py [--out results/pointmass_rollouts.png]
"""

import argparse
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research.envs  # noqa: F401 — register envs
from research.envs.pointmass import PointMassGymEnv


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

def random_policy(env):
    def _policy(obs):
        return env.action_space.sample()
    return _policy


def zero_policy(env):
    """Always stays put — stress-tests the timeout path."""
    def _policy(obs):
        return np.zeros(2, dtype=np.float32)
    return _policy


def rightward_policy(env):
    """Always moves right — illustrates boundary clamping."""
    def _policy(obs):
        return np.array([env.max_speed, 0.0], dtype=np.float32)
    return _policy


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def collect_rollout(env, policy_fn, max_steps=300):
    """Return list of positions (including reset pos) and terminal outcome."""
    obs = env.reset()
    positions = [obs.copy()]
    outcome = "timeout"

    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, _, done, info = env.step(action)
        positions.append(obs.copy())
        if done:
            if info.get("succ"):
                outcome = "goal"
            elif info.get("crash"):
                outcome = "trap"
            break

    return np.array(positions), outcome


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

OUTCOME_COLOR = {
    "goal": "green",
    "trap": "red",
    "timeout": "steelblue",
}

OUTCOME_LABEL = {
    "goal": "reached goal",
    "trap": "hit trap",
    "timeout": "timeout",
}


def plot_environment(ax, env):
    """Draw goal, trap, and unit-square boundary."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Unit square boundary
    square = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="square,pad=0",
        linewidth=1.5, edgecolor="black", facecolor="#f9f9f9",
    )
    ax.add_patch(square)

    # Goal region
    goal_circle = plt.Circle(
        env.goal, env.goal_dist_thresh,
        color="green", alpha=0.25, zorder=2, label="goal",
    )
    ax.add_patch(goal_circle)
    ax.plot(*env.goal, "g^", markersize=10, zorder=3)

    # Trap region
    trap_circle = plt.Circle(
        env.trap, env.trap_dist_thresh,
        color="red", alpha=0.25, zorder=2, label="trap",
    )
    ax.add_patch(trap_circle)
    ax.plot(*env.trap, "rx", markersize=10, markeredgewidth=2, zorder=3)


def plot_trajectory(ax, positions, outcome, alpha=0.8, label=None):
    color = OUTCOME_COLOR[outcome]
    ax.plot(
        positions[:, 0], positions[:, 1],
        color=color, alpha=alpha, linewidth=1.2, zorder=4,
    )
    # Start marker
    ax.plot(*positions[0], "o", color=color, markersize=6, zorder=5,
            markeredgecolor="black", markeredgewidth=0.5)
    # End marker
    ax.plot(*positions[-1], "*", color=color, markersize=10, zorder=5,
            markeredgecolor="black", markeredgewidth=0.5, label=label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/pointmass_rollouts.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Setup: one shared env per policy type so we can share the arena plot.
    # Using random init so trajectories start from diverse positions.
    # -----------------------------------------------------------------------
    env = PointMassGymEnv(
        goal=[0.3, 0.3],
        trap=[0.7, 0.7],
        step_cost=0.0,
        max_ep_len=300,
    )

    policies = [
        ("random",      random_policy(env),            6),
        ("expert",      env.make_expert_policy(),       4),
        ("noisy expert (ε=0.5)", env.make_noisy_expert_policy(eps=0.5), 4),
        ("rightward",   rightward_policy(env),          2),
    ]

    fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 5))
    fig.suptitle("PointMass Navigation — rollouts by policy", fontsize=14, y=1.01)

    for ax, (policy_name, policy_fn, n_trajs) in zip(axes, policies):
        plot_environment(ax, env)
        ax.set_title(f"{policy_name}\n({n_trajs} trajectories)", fontsize=11)

        seen_outcomes = set()
        for i in range(n_trajs):
            positions, outcome = collect_rollout(env, policy_fn)
            label = OUTCOME_LABEL[outcome] if outcome not in seen_outcomes else None
            seen_outcomes.add(outcome)
            plot_trajectory(ax, positions, outcome, alpha=max(0.4, 0.9 - i * 0.1), label=label)

        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved → {args.out}")

    # -----------------------------------------------------------------------
    # Also print a summary table.
    # -----------------------------------------------------------------------
    print("\nRollout summary (100 episodes each)\n" + "-" * 45)
    np.random.seed(args.seed)
    for policy_name, policy_fn, _ in policies:
        outcomes = [collect_rollout(env, policy_fn)[1] for _ in range(100)]
        goal_rate  = outcomes.count("goal")  / 100
        trap_rate  = outcomes.count("trap")  / 100
        timeout_rate = outcomes.count("timeout") / 100
        print(f"{policy_name:30s}  goal={goal_rate:.0%}  trap={trap_rate:.0%}  timeout={timeout_rate:.0%}")


if __name__ == "__main__":
    main()
