"""
Minimal, oracle-free reproduction of the seq-estop advantage-inconsistency bug.

Runs the REAL build_pairs() from generate_seq_estop_labels.py (no torch
forward passes, no GPU, no CHPC needed) on a hand-specified toy trajectory
where each timestep's "oracle quality" is known exactly. Confirms that
build_pairs' preferred/non-preferred assignment (purely positional, based on
t vs tau) very often contradicts the actual per-timestep quality ordering.

Usage:
    python scripts/toy_seq_estop_bug_demo.py
"""

import numpy as np

from scripts.generate_seq_estop_labels import build_pairs

# Toy trajectory: t = 0..9, a human presses e-stop at tau=6.
# "quality" stands in for whatever oracle rl_sum would say about each step;
# embedding it directly as the 1-D "obs" lets us read segment scores straight
# off the pairs build_pairs() returns, with zero oracle/model machinery.
quality = np.array([0.9, 0.8, 0.7, 0.5, 0.2, -0.2, -0.8, -1.0, -1.1, -1.1], dtype=np.float32)
T = len(quality)
h = 3
tau = 6

traj_obs    = quality.reshape(T, 1)            # (T, obs_dim=1) -- obs IS the quality score
traj_action = np.zeros((T, 1), dtype=np.float32)  # unused by build_pairs' slicing logic
traj_reward = np.zeros(T, dtype=np.float32)       # unused here

pairs = build_pairs(traj_obs, traj_action, traj_reward, tau, h)

print(f"Toy trajectory quality[t]: {quality.tolist()}")
print(f"h={h}  tau={tau}\n")
print(f"{'t':>3}  {'kind':>5}  {'preferred_score':>16}  {'nonpref_score':>14}  {'consistent?':>12}")

n_correct = 0
for pair in pairs:
    t = pair["timestep"]
    p_score = float(pair["obs"][0].sum())   # preferred segment's total quality
    n_score = float(pair["obs"][1].sum())   # non-preferred segment's total quality
    consistent = p_score > n_score
    n_correct += int(consistent)
    kind = "STOP" if pair["stop_event"] == 1.0 else "cont"
    mark = "OK" if consistent else "WRONG"
    print(f"{t:>3}  {kind:>5}  {p_score:>16.2f}  {n_score:>14.2f}  {mark:>12}")

rate = 100 * n_correct / len(pairs)
print(f"\n{n_correct}/{len(pairs)} = {rate:.0f}% of build_pairs()'s own labels agree with the "
      f"toy quality signal.")
assert rate < 50, (
    "Expected this toy trajectory to reproduce the reported bug "
    "(advantage-consistency well under 50%)."
)
print("Reproduced: build_pairs() assigns 'preferred' inconsistently with true segment quality.")
