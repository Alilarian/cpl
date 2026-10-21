# RRC + SAC baseline: plan

Status: **implemented and tested locally (stage B done); the CHPC stages (A, C1, C2) are ready to submit** (§5). This file replaces `docs/rrc_ppo_plan.md`.
Revised 2026-10-20 after your answers: the E-stop type is **cum_estop** and estop_hold is dropped. The true-reward oracles are **SAC**, so the RL step is **SAC with the exact oracle hyperparameters**, not PPO. You removed `multi-type-feedback/`, so nothing here uses SB3. Every stage uses **3 seeds (0, 1, 2)**, and **reward hacking is measured explicitly** (§3.6).
Target cluster: CHPC, `--partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai`.

**Run scope (user, 2026-10-21): pref (comparisons), corr and demo; 4 tasks; 3 seeds; no extra arms.** scalar, credit and cum_estop are dropped. The true-reward SAC runs, the 3×-epoch reward models and the untrained control are dropped too. That gives 12 reward jobs (3 types × 4 tasks, 3 seeds each = 36 ensembles of 3 networks) and 36 SAC runs at **2M env steps each** (user, 2026-10-21; the only deviation from the oracle SAC config, passed as `--total_steps` and recorded in each run's `reward_map.json`; the replay buffer stays at the oracle's 1.2M capacity), packed 4 per GPU into 9 jobs. They log to WandB project `rrc-sac-2m` (entity `a7a7`). The hacking trend test uses the second half of each run (1M–2M). Manifests: `slurm/manifests/rrc_sac/{offdist,reward_pref_corr_demo,sac_pref_corr_demo}.tsv`. Reward-hacking metrics (§3.6) still apply; their 0/1 references come from the off-distribution cache, not from true-reward SAC runs.

Pipeline: for each of the 6 feedback types (pref, corr, demo, scalar, credit, cum_estop), fit an RRC reward ensemble on ARIC's own choice structures. Then train SAC on the learned reward with the oracle's config, and evaluate with the true reward and success rate.

---

## 0. Decisions

| # | Question | Status |
|---|---|---|
| D1 | Which E-stop? | **Resolved: cum_estop only.** estop_hold is not used anywhere in this experiment, and its ARIC code is left untouched. |
| D2 | Which RL algorithm and config? | **Resolved: research SAC (`research/algs/sac.py`) with the oracle config** (§1.3). That config is byte-identical across the 4 tasks apart from `env`. |
| D3 | Which simulator? | **Resolved, nothing to install.** SAC runs in the research env (`MetaWorldSawyerEnv`, metaworld@04be337 + mujoco_py), the same env the oracles and feedback data used. |
| D4 | Which data is "~10k acts per type"? | **Confirmed.** the exact rows ARIC trained on, from `/scratch/general/vast/u1472210/mw_de_labels/<env>/`: `{pref,corr,scalar,credit}_labels.npz` with `subsample_n=10000, subsample_seed=0`, plus `demo_labels_K9.npz` (10k rows) and `cum_estop_labels/<env>/cum_estop_mix10000.npz`. Tasks: button-press, door-open, drawer-open, plate-slide. |
| D5 | Pilot task | **Confirmed: mw_door-open-v2.** Its oracle hits ≥0.9 at 80k steps and holds 1.00 over the last 10 evals (std 0). Full table in §1.3. |

Two places where the brief doesn't match the repo. I'll handle them as written in §2 unless you object:
- **Demonstrations already have an explicit choice set** (K=9: expert + 7 checkpoint-tier rollouts + the original segment, all from the same `s_0`). So I won't build random-policy alternatives, except as an ablation flag.
- **Labels come from oracle regret, not partial return.** Every generator scores options with `rl_sum = Σ_{t<T-1} r_t + V*(s_{T-1}) − V*(s_0)`, using the oracle SAC critic averaged over 64 actor samples. cum_estop uses the discounted U_E version of that score. Partial-return RRC is therefore misspecified relative to the labeler, and the true-reward partial-return ceiling (§3.4) will be below 1.0 for some types. That is expected. The check that must come out at about 1.0 recomputes the generator's own rule.

### Findings from implementation (2026-10-20)

- **F1. pref, corr, scalar and credit were labeled by a critic that isn't the local oracle mirror.** Recomputing the generators' `rl_sum` with `datasets/mw_de_labels/<env>/oracle/best_model.pt` reproduces the demo labels exactly (accuracy 1.000, correlation with stored `adv_scores` 1.000), which validates the grounding and formula. The other types don't match it:

  | Task | pref | corr | scalar | credit |
  |---|---|---|---|---|
  | button-press | 0.88 / 0.90 | 0.92 / 0.92 | 0.66 / 0.65 | 0.14 / 0.71 |
  | door-open | 0.86 / 0.89 | 0.66 / 0.66 | 0.73 / 0.74 | 0.16 / 0.67 |

  Each cell is accuracy / correlation with the stored scores. Monte Carlo noise is negligible: two value draws correlate at 1.000. These four types were generated on 2026-09-11/12, before the switch to `oracle_sac_1m` on 2026-09-21. `slurm/mw_de_generate_labels.sbatch` defaults to `runs/runs/chpc/oracle_sac_seeds/<env>/seed-1`, which isn't on this machine. The label-consistency check therefore scores every candidate in `paths.label_oracle_candidates` on a probe sample, keeps the one that best reproduces the stored scores, and reports all of them. On CHPC the expected labeler is `oracle_sac_seeds/<env>/seed-1`. If no candidate reaches ≈1.0, the labeler critic is unknown and that has to be resolved before trusting the check. It does not affect reward training, which uses only the stored labels.
- **F2. cum_estop no-stop rows look stored in reverse.** `scripts/generate_cum_estop_labels.py` stacks no-stop rows as `[weak counterfactual, accepted original]`, while its docstring, `CorrBuffer` and `DiscountedCorrCPL` all treat index 0 as preferred. If the CHPC mix was built with this code, ARIC's cum_estop runs trained on flipped labels for half their rows. The RRC loader doesn't rely on the stored order. It identifies the accepted original as the arm that is a full pool trajectory and makes it the preferred option (`nostop_preferred: original`; `as_stored` reproduces ARIC's reading). Preflight reports `nostop_original_at_index0_frac`: 0.0 confirms the flip, 1.0 means the CHPC files were built differently. The label-consistency report also gives the labeler-rule accuracy under both readings. ARIC code is unchanged.
- **F3. G can be large when a reward model barely separates random from oracle behavior.** For example, the local button-press pref model gives random-policy episodes 140 learned return vs 155 for oracle episodes. The normalized gap is still well defined, but its denominator is small, so `hacking.json` also stores the raw reference returns and the reward report gives the off-distribution success AUC. Read G together with them.

---

## 1. What exists (exploration summary)

### 1.1 Feedback datasets
Local mirrors are in `datasets/mw_de_labels/<env>/`. On CHPC the same names are in `/scratch/general/vast/u1472210/mw_de_labels/<env>/`. Obs is 35-d (trimmed), actions are 4-d, rewards are true dense reward divided by 10, in [0.01, 0.99].

| File | Arrays | Notes |
|---|---|---|
| `pool.npz` | obs (100k,64,35), action, reward, **state (100k,64,30)**, checkpoint_step | 64-step windows sampled with replacement from **200 deterministic episodes** of one checkpoint (49,973 distinct states). No episode id stored. |
| `pref_labels.npz` | obs/action/reward (N,2,64,·), adv_scores, gap | idx0 = higher rl_sum. The two arms are pool rows from different episodes. |
| `corr_labels.npz` | obs/action/reward (N,2,64,·), adv_scores, improvement | {expert rollout from s_0, original pool segment}. idx0 = winner. |
| `demo_labels_K9.npz` | obs/action/reward (10k,9,64,·), adv_scores (N,9) | idx0 = expert, always. Rows are kept only if the expert beats the best counterfactual. |
| `scalar_labels.npz` | obs/action/reward (N,2,16,·), adv_scores, **traj_id**, start_idx | Pairs of 16-step subsegments of one pool row (stride 12, lag 2–4). idx0 = preferred. |
| `credit_labels.npz` | obs/action (N,45,20,·), adv_scores (N,45), chosen_idx | **No rewards stored.** Row n is pool row n, and the windows are the 45 sliding 20-step windows of that segment (verified locally). True rewards come from the pool. |
| `cum_estop_labels/<env>/cum_estop_mix10000.npz` (CHPC only) | obs/action/reward (10k,2,64,·), **source** (0 = stop+correction, 1 = no-stop) | A shuffled 5k/5k mix. Both arms are full 64-step trajectories. Stop rows: the original vs a Π_M correction from `s_tau`, sharing the prefix, with idx0 = higher U_E. No-stop rows: the accepted original vs a weak Π_B rollout from `s_0`. `stop_index` isn't kept, so I'll recover it as the first step where the arms diverge. |

None of the label files stores env states, success flags, or rollout ids, except scalar's `traj_id`. I will recover episodes by exact matching against `pool.npz` (§3.1).

### 1.2 ARIC choice structures (`research/algs/cpl.py`, `cum_estop_cpl.py`, `scoring.py`)
All types share `score_segments` (γ^t from each segment's own t=0).

| Type | ARIC class / buffer | C | c* | k | γ | ARIC temperature / bias | Ties |
|---|---|---|---|---|---|---|---|
| pref | DemoCPL / CorrBuffer | 2 | 0 | 64 | 1.0 | α=0.1, λ=0.75 | none in practice |
| corr | DemoCPL / CorrBuffer | 2 | 0 | 64 | 1.0 | α=0.1, λ=0.75 | dropped below min gap |
| demo | DemoCPL / DemoBuffer | 9 | 0 | 64 | 1.0 | α=0.1, λ=0.75 | n/a |
| scalar | DemoCPL / CorrBuffer | 2 | 0 | 16 | 1.0 | α=0.1, λ=0.75 | exact ties dropped (δ=0) |
| credit | CreditAssignmentCPL / PMCreditAssignmentBuffer | 45 | chosen_idx | 20 | 1.0 | α=0.1, λ=0.75 | argmax |
| cum_estop | DiscountedCorrCPL / CorrBuffer | 2 | 0 | 64 | 0.99, from t=0 of the full arm, so the shared prefix cancels | α=0.1, λ=0.75 | n/a (single honest draw) |

- Every act is one row with one choice structure, and the loss is the batch mean over rows.
- ARIC CPL runs use `split: all`, so there is no held-out set. The buffers support a row-level `split/val_frac` (`RandomState(0)`), but the 200-episode pool makes that split leak, so I'll split by episode instead (§3.1).
- `research/algs/reward_cpl.py` (RewardCPL / RewardCreditCPL) already computes these losses with r_θ in place of α·log π. At bias=1 that is exactly the RRC loss for one network, so it serves as the **test oracle** for loss equivalence.

### 1.3 The oracle SAC, the env, and eval
- **Oracle config** is `config.yaml` in each oracle run dir. The local mirror is `datasets/mw_de_labels/<env>/oracle/`, and the live copy on CHPC is `/scratch/general/vast/u1472210/oracle_sac_1m/<env>/`. The settings are:
  - SAC with `tau 0.005`, `init_temperature 0.1`, `critic_freq 1`, `actor_freq 1`, `target_freq 2`, `random_steps 25000`.
  - `ActorCriticPolicy` with `DiagonalGaussianMLPActor` (squashed, log_std ∈ [−5, 2]) and `ContinuousMLPCritic` with ensemble 2, [256,256,256], ortho init.
  - Adam lr 3e-4; `ReplayBuffer` with capacity 1.2M, batch 512, discount 0.99, nstep 1.
  - **1M steps**, `env_runner: AsyncEnv`, eval with `eval_policy` **every 10k steps on 20 episodes**, checkpoints every 20k, `loss_metric: reward`, `seed: 0`.
  - `env_kwargs: {sparse: false}`.
- **Oracle curves** (eval success, seed 0, from `log.csv`):

  | Task | @100k | @250k | @500k | Final 3 | Last 10 | First ≥0.9 |
  |---|---|---|---|---|---|---|
  | button-press | 0.90 | 1.00 | 0.90 | 0.97 | 0.94±0.05 | 90k |
  | door-open | 0.60 | 1.00 | 1.00 | 1.00 | 1.00±0.00 | 80k |
  | drawer-open | 0.75 | 1.00 | 1.00 | 1.00 | 1.00±0.00 | 160k |
  | plate-slide | 0.50 | 0.95 | 1.00 | 0.98 | 0.98±0.02 | 130k |

  A second true-reward reference also exists: `runs/runs/chpc/oracle_sac_all/<env>/` has the same hyperparameters except checkpoint_freq 10k, is unseeded, and stopped at 770k–820k steps. Its final-3 success is 0.95–1.00.
- **Walltime**: the oracle runs took **43–60h on CPU** (6.5 and 4.6 steps/s). On the local RTX 4090 the same config runs at **~140 steps/s** once updates start, which is about 2h of training per 1M steps plus about 0.1h of evals. Expect 3–9h on rai-gpu-grn H200s, which ran 1.3–4.2× slower under load in earlier jobs.
- **Env** (`research/envs/metaworld.py`): horizon 250, dense reward ÷10, `randomize_hand=True`, obs trimmed 39→35, goal observable (`_partially_observable=False`). At the horizon it returns `done=True` with `info["discount"]=1.0`, so the replay buffer bootstraps through timeouts. There is no termination on success, and `get_state`/`set_state` exist.
- **Run machinery**:
  - `Config.get_trainer()` builds `Trainer(model, train_env_fn, eval_env_fn)`.
  - The config `wrapper` key is applied to **both** the train and eval envs, so it can't be used for a train-only reward swap.
  - The `AsyncEnv` worker (`research/utils/runners.py`) sends back only `(reward, done, discount)` and **drops `info`**.
  - `eval_policy` sums every scalar `info` key per episode into `eval/<key>`.

### 1.4 Repo conventions
- YAML configs with `import:` and a shallow merge (`research/utils/config.py`). `scripts/train.py --config --path [--set k=v]` drives the research Trainer.
- Logger: `research/utils/logger.py` (tensorboard + CSV; the CSV gets rows at eval steps; WandB entity `a7a7` when `WANDB_API_KEY` is set).
- Sweeps: `tools/create_sweep_manifest.py` writes a TSV manifest, run by `slurm/train_array.sbatch`, which runs `scripts/train.py`. Env setup is `setup_shell.sh`, and the GPU env is `CONDA_ENV_NAME=cpl_gpu`.

---

## 2. Deviations from the brief

1. **SAC replaces PPO.** The oracles are SAC, and the brief says to reuse the true-reward RL setup unchanged, so that setup is the oracle SAC config. The step budget is 1M, as for the oracles, and the eval protocol is the oracle's: every 10k steps on 20 episodes.
2. **E-stop = cum_estop.** One K=2 pair per act, scored like `DiscountedCorrCPL`: `R = Σ_{t=0}^{63} 0.99^t r_t` per full arm, so the identical prefix cancels exactly. ARIC's cumulative model has no per-step continue/halt choices and no halt action, so neither does this one. Metrics are also reported per branch (`source`).
3. **Demos** use ARIC's K=9 explicit set. Random-policy alternatives exist only as an ablation flag (`demo_alternatives: random`, K configurable, cached).
4. **β = 1.0 for all types.** ARIC's α=0.1 is a temperature on log π, and λ=0.75 is CPL's conservative bias. Neither is part of RRC, and `reward_cpl.py`'s docstring explains why λ≠1 doesn't carry over to an unnormalized reward.
5. **Train on 90% of ARIC's rows.** The held-out 10% is split by source episode. ARIC trained on all rows.
6. **Standalone reward trainer.** It is epoch-based with per-member early stopping, which doesn't fit the research Trainer. The SAC stage **does** use the research Trainer, through a thin driver that only swaps the env fns.

---

## 3. Design

### 3.1 Data loading and split (`research/reward_learning/data.py`)
- **Load through ARIC's own buffer classes** with `split="all"` and the same `subsample_n`/`subsample_seed`. Recompute the kept indices with the buffers' formula and assert the arrays match. ARIC code is not touched.
- Each type becomes a `ChoiceSet` with: `obs (N,K,L,35)`, `act (N,K,L,4)`, `step_mask`, `option_mask (N,K)` for variable K, soft `target (N,K)` (one-hot, or uniform over tied argmax with a reported tie count expected to be 0), `gamma`, `act_id`, `episode_ids`, `true_reward`, and `branch` for cum_estop.
- **Credit**: keep only the 64-step base segment plus window starts 0..44 and k=20. Assert all 45 windows agree with it. Per-step r is computed once per segment, and window returns come from a discounted 1-D convolution or cumsum. True rewards come from pool row n.
- **Episode ids**: union-find over `pool.npz` rows that share an identical obs vector, asserting exactly 200 components. Map each act's pool-derived arm to its pool row with an exact `obs[0]` hash match, falling back to `allclose`, and assert 100% are matched. Rollout arms (expert, tier, Π_M, Π_B) start at a pool state and inherit that episode.
- **Split**: per task, a val episode set E_val drawn with `split_seed=0`, shared by all members and seeds. An act goes to val if any of its episodes is in E_val. E_val is sized so val ≈ 10% of acts: about 20 of 200 episodes, or about 10 for pref, whose two arms come from two episodes. For pref I also report val metrics on acts where both arms are unseen. Indices are saved to `split.npz`.

### 3.2 RRC loss (`research/reward_learning/rrc_loss.py`)
- Option return `R_m = Σ_t γ^t r_m(s_t,a_t) · step_mask`, with each segment indexed from its own start using ARIC's γ.
- Per structure: `ℓ = −Σ_c target_c log softmax_c(β R_m)`, with invalid options set to −inf.
- Per act: mean over its structures, via a scatter-mean on `act_id`. `act_group: traj_id` for scalar is a config switch. Per batch: mean over acts.
- Batches are 128 acts, and each member gets its own per-epoch permutation.

### 3.3 Model and training (`research/reward_learning/ensemble.py`, `train.py`)
- Vectorized ensemble, E=3: batched linear layers (E, in, out). Input is concat(obs 35, act 4), unnormalized, with dims asserted against the env spaces. 3×256 hidden, LeakyReLU, tanh output (`output: linear` ablation flag). Each member is initialized from its own seed (`seed*100+m`).
- Adam, lr 3e-4, wd 0, **gradient-norm clip 1.0 per member** (computed manually, because a global clip would couple members).
- Every epoch: train and val NLL (mean per act) per member and for the ensemble (mean of raw member rewards), plus val choice accuracy.
- Per-member best checkpoint. Stop when no member has improved for 10 epochs, or at 200. Also save `last` and `epoch_{025,050,…}`. Pilot types (`--pilot`) freeze `best` at the same early-stopping point, so a pilot run's `best` is identical to a normal run's and the sweep reuses it. They then keep training to 3× the ensemble-best epoch e* (cap 600), saving every epoch, and export `epoch3x` (every member at epoch 3·e*).
- **Standardization** per exported checkpoint: μ_m and σ_m over all valid training-split (s,a). The reward is `mean_m (r_m − μ_m)/σ_m`, with μ and σ stored in the checkpoint.
- `RewardEnsemble` (nn.Module): `reward(obs, act)` batched, `no_grad`, device-aware, numpy or torch input. `RewardEnsemble.load(path, device)`.

### 3.4 Diagnostics (`research/reward_learning/diagnostics.py`, written to `report.json` per type)
- NLL and choice accuracy, train and val, per member and for the ensemble. cum_estop is also broken out per branch.
- **Ground-truth correlation**: Pearson and Spearman between predicted and true discounted segment returns, (a) on held-out options and (b) on off-distribution rollouts.
- **Off-distribution rollouts**: cached once per task with a fixed seed in `runs/rrc_sac/_cache/<task>/offdist.npz`. 50 uniform-random episodes, plus 20 episodes from each of 5 **oracle SAC checkpoints**. The checkpoints are picked from the oracle `log.csv` at eval success ≈ 0, 0.25, 0.5, 0.75 and 1.0 from the 20k checkpoints. Any `--gt_policy_ckpts` are added on top, for example the RRC-SAC runs themselves. Random alone almost never succeeds, so the checkpoints provide both classes for the AUC.
- **Success separation**: AUC of predicted episode return, successful vs failed, reported with class counts.
- **Saturation**: share of |r_m| > 0.95, per member, on held-out and off-distribution data.
- **Disagreement**: mean per-step std across standardized members, held-out vs off-distribution.
- **Label consistency**, two checks:
  1. The generator's own rule. For pref, corr, demo, scalar and credit, `rl_sum` is recomputed with the oracle critic on all val acts plus 1,000 train acts. For cum_estop, U_E (γ=0.99, endpoint V) is recomputed for 200 acts per branch, re-simulating from the pool `state` for endpoint obs. Stop rows must prefer the higher-U_E arm, and no-stop rows must pass the acceptance margin. **This must come out at about 1.0**, apart from a little MC noise.
  2. The true-reward partial-return accuracy, which is this model's ceiling.

  If check 1 is clearly below 1.0, I stop and investigate before going on. The check also confirms which oracle labeled the data; see assumption 2.

### 3.5 SAC on the learned reward (`research/reward_learning/envs.py`, `scripts/train_rrc_sac.py`)
- **Config**: `configs/rrc_sac/sac_oracle.yaml` is a verbatim copy of the oracle config with `env` templated. At startup the driver asserts that it equals the task's oracle `config.yaml` with env, seed and paths ignored. Hyperparameters, budget and eval are never edited per reward model.
- **Driver**: it mirrors `scripts/train.py` and `Config.get_trainer()` exactly (`Config.load` → seed → `config.save(path)` → `parse` → `get_model`). The only change is `Trainer(model, train_env_fn=wrapped, eval_env_fn=shadowed, **trainer_kwargs)`. It uses the same AsyncEnv runner, replay buffer, checkpointing and resume through `final_model.pt`.
- **`LearnedRewardWrapper`** (gym.Wrapper, **training env only**):
  - Replaces the reward with `a·R̂(obs_t, clip(a_t)) + b`, where `obs_t` is the raw obs **before** the step and `R̂` is the standardized ensemble reward.
  - Built inside the AsyncEnv subprocess from a module-level factory and a checkpoint path, so it can be pickled. It runs on CPU with 1 torch thread.
  - Forwards `_max_episode_steps`, and keeps `info["discount"]` and `info["success"]`.
  - Writes `info["true_reward"]` and `info["learned_reward"]`. AsyncEnv drops `info`, so the wrapper also appends one row per **training episode** to `train_episodes.csv`, with the global env step, true return, learned return, success and length.
  - Modes: `--reward ckpt` (RRC), `--reward true` (the oracle path, which keeps the env reward), `--reward untrained` (a random-init ensemble, the negative control).
- **`ShadowRewardWrapper`** (eval env): keeps the true reward and adds `info["learned_reward"]` and `info["learned_reward_std"]`, the per-step std across the 3 standardized members. Eval therefore logs `eval/reward` (true), `eval/success`, **`eval/learned_reward`** and `eval/learned_reward_std` for free. Every 5th eval (every 50k steps) it also saves the 20 eval episodes to `eval_traj/step_<N>.npz`: obs, act, true r, learned r, per-member r and success. The training-env wrapper adds the per-episode mean member std to `train_episodes.csv` in the same way.
- **Reward scale**: research SAC doesn't normalize rewards, so this takes the affine branch. `a = σ_true` and `b = μ_true` are computed over the training-split (s,a) of that type's feedback data, giving the learned reward the same mean and std as the true dense reward (÷10). MetaWorld has no terminal states and timeouts bootstrap, so the shift and scale don't change the optimal policy. Matching the scale also keeps SAC's entropy temperature on the same footing as the oracle's. The branch and `(a, b)` are written to `reward_map.json`.
- `best_model.pt` is still selected on true eval reward, as in the oracle config. It doesn't affect training. I report final success (mean of the last 3 evals), not the best model.

### 3.6 Reward-hacking measurement (`research/reward_learning/hacking.py`, written to `hacking.json` per SAC run)
Off-policy SAC exploits reward-model errors fast. This section turns "learned return rises while true return doesn't" into numbers. Every metric is computed per run, on both eval episodes (from the log) and training episodes (from `train_episodes.csv`, binned by 10k steps).

- **Common scale.** Raw learned and true returns aren't comparable, so each is normalized by where two reference policies sit **under that same reward**. The references come from the §3.4 off-distribution cache, so they cost no extra simulation: the uniform-random policy (0) and the final oracle SAC checkpoint (1).
  ```
  L̃(t) = (L(t) − L_rand) / (L_oracle − L_rand)     # learned return, scored by this run's reward checkpoint
  T̃(t) = (T(t) − T_rand) / (T_oracle − T_rand)     # true return
  ```
  Both ratios are invariant to the affine reward map, so they can be compared across types and tasks. L̃ > 1 means the policy collects **more learned reward than the oracle policy does**.
- **Hacking gap** `G(t) = L̃(t) − T̃(t)`. It is reported at the end (mean of the last 3 evals), at its maximum over training, and as a curve. On the oracle path G is identically 0, which serves as a sanity check.
- **Overoptimization drop**: peak eval success minus final success, and the same for T̃. This catches the case where true performance rises and then falls while the policy keeps optimizing the learned reward.
- **Trend test over the second half of training (500k–1M steps)**: OLS slopes of L̃ and T̃ against step, with bootstrap 95% CIs over the evals. A run is **flagged as hacking** if all three hold:
  1. the L̃ slope is > 0, with its CI excluding 0;
  2. the T̃ slope is ≤ 0, or final success is ≤ 0.5 × oracle final success;
  3. the final G is > 0.25.

  The thresholds are fixed in config before the pilot and reported with the results.
- **Learned-vs-true agreement on the policy's own states**, from the saved `eval_traj` snapshots every 50k steps:
  - per-step Pearson and Spearman of learned r vs true r;
  - the episode-level correlation of learned vs true return.

  A correlation that falls as training goes on means SAC is moving into regions where the reward model is wrong.
- **Ensemble disagreement on visited states**: the mean member std on eval episodes, divided by the same quantity on the reward model's held-out feedback data (from `report.json`). A ratio well above 1 that grows over training is the out-of-distribution exploitation signature.
- **Best-NLL vs 3×-epoch checkpoints** (pilot): the same reward type and seeds, differing only in how long the reward model was fit. The pilot reports ΔG, Δfinal success, Δdisagreement ratio and Δflag rate between the two. A reward model fit past its validation optimum should be hacked more.
- **Plots** per run, one figure: L̃, T̃ and success vs step (train and eval), with G shaded, plus the disagreement ratio on a second axis.

---

## 4. Tests and sanity checks (before the pilot)
`tests/test_rrc_*.py`, pytest, CPU, tiny synthetic data unless marked:
1. Each type's loss matches a hand-computed value: pairwise BT, K-way, credit windows, discounted cum_estop with a shared prefix, soft-target ties.
2. **Equivalence with `reward_cpl.py`** at bias=1 with one member, on real records: RewardCPL (pref/corr/demo/scalar) and RewardCreditCPL. For cum_estop, `score_segments(..., discount=0.99)` with an r_θ scorer, which is exactly DiscountedCorrCPL's scoring.
3. Per-act weighting: duplicating one act's structures leaves the loss unchanged, and every act has equal weight.
4. Credit masking with variable window counts: padded windows get −inf and zero gradient, and the cumsum or convolution returns equal brute-force sums.
5. cum_estop grounding equals ARIC's scores on stored records. On stop rows the score difference equals γ^τ times the suffix difference, using the recovered `stop_index`.
6. Choice structures equal ARIC's on 20 sampled records per type (obs, act, chosen, mask, γ). Skipped if the files are missing.
7. Ensemble: members are independent, clipping is per member, and standardization and `load` round-trip.
8. (MuJoCo-marked) `--reward true` through the wrapper gives rewards bit-identical to the unwrapped env under the same seed. The wrapper pickles and runs under AsyncEnv. The shadow wrapper doesn't change eval reward. `train_episodes.csv` returns equal the env's.
9. Hacking metrics on synthetic curves:
   - learned = true gives G ≡ 0 and no flag;
   - learned rising while true is flat gives a flag;
   - the normalization is invariant to the affine map;
   - the slope bootstrap is reproducible under a fixed seed.

On CHPC:
- **Oracle path**: SAC through the wrapper with the true reward, 3 seeds on the pilot task. It must match the oracle curves within noise, against oracle seed 0 and the `oracle_sac_all` run, and its G must be 0.
- **Negative control**: SAC on an untrained ensemble, 3 seeds on the pilot task. It must stay near random-policy success, which is the extreme end of the hacking scale.

---

## 5. Execution on CHPC

All jobs run with `--partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1` and `CONDA_ENV_NAME=cpl_gpu DEVICE=cuda`. SSH from my sandbox to CHPC is blocked, so I hand you the scripts and you submit them. You then rsync `runs/rrc_sac/` back, and I write the reports.

**Packing.** rai-gpu-grn is only two H200 nodes (grn025/026). Each SAC job packs **4 runs on 1 GPU**, each with its own AsyncEnv subprocess, using `--cpus-per-task=16 --mem=64G --time=24:00:00`. The runs resume from `final_model.pt`, so a timeout just means resubmitting. A reward-training job trains all requested seeds for one (task, type) at about 5–15 min each.

| Stage | What | Jobs | Gate |
|---|---|---|---|
| A. preflight | `scripts/rrc_preflight.py` for every task. It checks file shapes, that 200 episodes are recovered, 100% arm→pool matching, the reward range, that `sac_oracle.yaml` equals each oracle `config.yaml`, and that oracle checkpoints are present. | 1 short job | all pass |
| B. local | Implement, run unit tests, and smoke-test on the local mirrors for button-press: reward training for 5 types (cum_estop isn't local) and a 30k-step SAC through the wrapper. | none (I do this) | tests green |
| C1. reward models | Off-distribution caches for 4 tasks. Ensembles for 6 types × 4 tasks at seed 0, plus `report.json` and a summary table. The pilot types {pref, cum_estop} on the pilot task are trained for seeds 0, 1 and 2 with `--pilot`, so every epoch is saved. | 4 + 24 + 2 | label consistency ≈ 1.0 |
| C2. pilot SAC | On the pilot task: {pref, cum_estop} × {best-NLL checkpoint, ≈3× epochs} × 3 seeds = 12 runs, plus the oracle path (3 seeds) and the negative control (3 seeds). | 5 (18 runs) | → report, then **stop** |

Pilot report (`docs/rrc_sac_pilot.md`):
- the reward diagnostics table;
- final success (mean of the last 3 evals, mean ± std over seeds) per arm;
- learned vs true return curves for training episodes and eval;
- the **§3.6 hacking table** per arm (final and max G, success and T̃ drop, slope-test flag rate, per-step learned-vs-true correlation over training, disagreement ratio);
- the best-NLL vs 3×-epoch deltas.

The best-NLL pilot runs use seeds 0–2 and the same config as the sweep, so the sweep reuses them rather than rerunning them.

### Commands (all submitted from the repo root on CHPC)
Runs go to `RUNS_ROOT=/scratch/general/vast/u1472210/rrc_sac`, the default in `slurm/rrc_common.sh`, because SAC checkpoints are about 170 MB per run. Every script resumes or skips finished work, so resubmitting is always safe.
```bash
cd /uufs/chpc.utah.edu/common/home/u1472210/cpl && git pull
SB="--partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1"
M=slurm/manifests/rrc_sac && mkdir -p $M slurm/logs
RUNS_ROOT=/scratch/general/vast/u1472210/rrc_sac

# A. preflight (one job, ~10 min): writes $RUNS_ROOT/_preflight/<task>.json
sbatch $SB --cpus-per-task=4 --mem=64G --time=1:00:00 slurm/rrc_preflight.sbatch

# C1a. off-distribution caches (4 jobs, ~2 min each)
python scripts/make_rrc_manifest.py --stage offdist --tasks all --out $M/offdist.tsv
OFF=$(sbatch --parsable $SB --array=1-4 --cpus-per-task=4 --mem=32G --time=2:00:00 slurm/rrc_offdist.sbatch $M/offdist.tsv)

# C1b. reward models + reports (25 jobs: 24 (task, type) + the untrained control; pilot types get seeds 0-2 and --pilot)
python scripts/make_rrc_manifest.py --stage reward --tasks all --types all --seeds 0 \
    --pilot_task mw_door-open-v2 --pilot_types pref cum_estop --pilot_seeds 0 1 2 --out $M/reward_c1.tsv
sbatch $SB --dependency=afterok:$OFF --array=1-25 --cpus-per-task=8 --mem=64G --time=4:00:00 \
    slurm/rrc_reward.sbatch $M/reward_c1.tsv
python scripts/rrc_reward_table.py --runs_root $RUNS_ROOT --out results/rrc_reward_diagnostics.md

# C2. pilot SAC on door-open: 18 runs, 4 per GPU job
python scripts/make_rrc_manifest.py --stage sac --tasks mw_door-open-v2 \
    --arms pref@best pref@3x cum_estop@best cum_estop@3x true_reward untrained --seeds 0 1 2 --pack 4 --out $M/pilot.tsv
sbatch $SB --array=1-5 --cpus-per-task=16 --mem=64G --time=24:00:00 slurm/rrc_sac.sbatch $M/pilot.tsv
python scripts/rrc_hacking.py --runs_root $RUNS_ROOT --tasks mw_door-open-v2          # hacking.json + hacking.png per run
python scripts/aggregate_rrc_sac.py --runs_root $RUNS_ROOT --tasks mw_door-open-v2 --out docs/rrc_sac_pilot.md
```
**pref/corr/demo run (current scope), chained with dependencies:**
```bash
cd /uufs/chpc.utah.edu/common/home/u1472210/cpl
SB="--partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1"
M=slurm/manifests/rrc_sac && mkdir -p $M slurm/logs
export RRC_TYPES="pref corr demo"
python3 scripts/make_rrc_manifest.py --stage offdist --tasks all --out $M/offdist.tsv
python3 scripts/make_rrc_manifest.py --stage reward --tasks all --types pref corr demo --seeds 0 1 2 \
    --pilot_task mw_door-open-v2 --pilot_types pref corr demo --pilot_seeds 0 1 2 --out $M/reward3.tsv          # 13 lines
python3 scripts/make_rrc_manifest.py --stage sac --tasks all --arms pref@best corr@best demo@best true_reward \
    --seeds 0 1 2 --pack 4 --out $M/sac3_main.tsv                                                              # 12 lines
python3 scripts/make_rrc_manifest.py --stage sac --tasks mw_door-open-v2 --arms pref@3x corr@3x demo@3x untrained \
    --seeds 0 1 2 --pack 4 --out $M/sac3_extra.tsv                                                             # 3 lines
cat $M/sac3_main.tsv $M/sac3_extra.tsv > $M/sac3.tsv
PRE=$(sbatch --parsable $SB --cpus-per-task=4 --mem=64G --time=1:00:00 slurm/rrc_preflight.sbatch)
OFF=$(sbatch --parsable $SB --dependency=afterok:$PRE --array=1-4 --cpus-per-task=4 --mem=32G --time=2:00:00 slurm/rrc_offdist.sbatch $M/offdist.tsv)
REW=$(sbatch --parsable $SB --dependency=afterok:$OFF --array=1-13 --cpus-per-task=8 --mem=64G --time=4:00:00 slurm/rrc_reward.sbatch $M/reward3.tsv)
SAC=$(sbatch --parsable $SB --dependency=afterok:$REW --array=1-15 --cpus-per-task=16 --mem=64G --time=24:00:00 slurm/rrc_sac.sbatch $M/sac3.tsv)
```

Single runs (local or interactive):
- `python scripts/train_rrc_reward.py --config configs/rrc_sac/<type>.yaml --task <task> --seeds 0 [--pilot|--untrained]`
- `python scripts/train_rrc_sac.py --task <task> --arm <type>@best|<type>@3x|true_reward|untrained --seed 0`

On the local machine, add `--overlay configs/rrc_sac/local.yaml` (data mirrors under `datasets/`) and `--runs_root runs/rrc_sac`. Add `--smoke --set trainer_kwargs.total_steps=3000 ...` for SAC smoke tests only; without `--smoke`, overrides are refused.

### Stage B results (local, 2026-10-20)
- 52 tests pass: `pytest tests/test_rrc_*.py tests/test_reward_cpl_k2.py tests/test_cum_estop_consistency.py`. That includes loss equivalence with ARIC's scoring code, choice structures matching ARIC's buffers on real rows, the wrapper being identical to the bare env in true-reward mode, and pilot `best` equal to normal `best`.
- door-open reward models, seed 0, from local mirrors, with the `best` export (cum_estop is CHPC-only):

  | type | epochs (stop) | val NLL | val acc | GT corr val | off-dist ep. corr | success AUC | partial-return acc | labeler-rule acc (local candidates) |
  |---|---|---|---|---|---|---|---|---|
  | pref | 18 | 0.203 | 0.921 | 0.67 | 0.93 | 0.996 | 0.739 | 0.855 |
  | corr | 45 | 0.131 | 0.950 | 0.91 | 0.93 | 0.997 | 0.666 | 0.960 (oracle_sac_all) |
  | demo | 108 | ≈0 | 1.000 | 0.40 | 0.80 | 0.956 | 0.912 | 0.999 |
  | scalar | 43 | 0.364 | 0.847 | 0.17 | 0.16 | 0.642 | 0.534 | 0.722 |
  | credit | 39 | 2.575 | 0.342 | −0.13 | 0.13 | 0.590 | 0.105 | 0.159 |
  | untrained | – | 0.682 | 0.668 | 0.26 | 0.24 | 0.593 | – | – |

  credit and scalar are where partial-return RRC on regret labels breaks down: true-return scoring reproduces only 11% / 53% of their labels. The pilot's 3× pref checkpoint is epoch 15, against a best epoch of 5.
- SAC smoke runs (4k steps) for `pref@best`, `pref@3x`, `true_reward` and `untrained` completed through AsyncEnv. `true_reward` gives G = 0.00 exactly. The aggregator, hacking plots and tables render.

---

## 6. Full sweep (prepared, not launched)
- 6 types × 4 tasks × **3 seeds (0, 1, 2)**. Seed s trains the reward ensemble with seed s and SAC with seed s. The split stays at `split_seed=0`, matching ARIC's fixed data subsample. Plus the true-reward baseline through the oracle path, 4 tasks × 3 seeds.
- That is **72 reward ensembles and 84 SAC runs**. The pilot already covers 6 of the RRC runs (door-open × {pref, cum_estop} @best) and 3 oracle-path runs, so 75 SAC runs are new. That is 24 reward jobs and about 19 packed SAC jobs (4 runs per GPU, about 3–9h each, up to 24h with resume).
- Commands (not to be launched until the pilot is reviewed):
  ```bash
  python scripts/make_rrc_manifest.py --stage reward --tasks all --types all --seeds 0 1 2 --out $M/sweep_reward.tsv   # 24 lines
  sbatch $SB --array=1-24 --cpus-per-task=8 --mem=64G --time=6:00:00 slurm/rrc_reward.sbatch $M/sweep_reward.tsv
  python scripts/make_rrc_manifest.py --stage sac --tasks all --types all --seeds 0 1 2 --pack 4 --out $M/sweep_sac.tsv  # 21 lines
  sbatch $SB --array=1-21 --cpus-per-task=16 --mem=64G --time=24:00:00 slurm/rrc_sac.sbatch $M/sweep_sac.tsv
  python scripts/rrc_hacking.py --runs_root $RUNS_ROOT
  python scripts/aggregate_rrc_sac.py --runs_root $RUNS_ROOT --out docs/rrc_sac_results.md
  ```
  Finished seeds and runs are skipped, so the pilot's reward models and its `@best` / `true_reward` SAC runs are reused.
- `scripts/aggregate_rrc_sac.py` reports, per task and type:
  - final success (mean of the last 3 evals) as **mean ± std over the 3 seeds**, plus IQM with a 95% stratified bootstrap CI for the cross-task aggregate. With 3 seeds per cell the per-cell CIs are wide, so mean ± std is the headline;
  - learning curves;
  - the reward diagnostics table;
  - the **§3.6 hacking table**: final and max G, success drop, flag rate out of 3 seeds, learned-vs-true per-step correlation at 1M steps, and disagreement ratio.

  Outputs go to Markdown, CSV and PNG.

---

## 7. Files to add (nothing existing is modified)
```
research/reward_learning/{__init__,data,rrc_loss,ensemble,train,diagnostics,envs,offdist,hacking}.py
configs/rrc_sac/{base,local,pref,corr,demo,scalar,credit,cum_estop}.yaml, configs/rrc_sac/sac_oracle.yaml, configs/rrc_sac/hacking.yaml
scripts/{rrc_preflight,train_rrc_reward,collect_rrc_offdist,train_rrc_sac,make_rrc_manifest,rrc_reward_table,rrc_hacking,aggregate_rrc_sac}.py
slurm/{rrc_preflight,rrc_offdist,rrc_reward,rrc_sac}.sbatch, slurm/rrc_common.sh
tests/test_rrc_{losses,equivalence,ensemble,data,env,hacking,train}.py
docs/rrc_sac_plan.md (this file), docs/rrc_sac_pilot.md (after the pilot)
```
Run layout:
- `runs/rrc_sac/<task>/<type>/seed<s>/reward/`: `best.pt`, `last.pt`, `epoch_XXX.pt`, `split.npz`, `metrics.csv`, `report.json`, `config.yaml`.
- `runs/rrc_sac/<task>/<type>/seed<s>/sac[_<ckpt>]/`: the research Trainer outputs (`log.csv`, tfevents, `model_*.pt` every 20k, `final_model.pt`), plus `train_episodes.csv`, `reward_map.json`, `eval_traj/step_<N>.npz` (every 50k steps), `hacking.json` and `hacking.png`.
- The oracle path lives under `<task>/true_reward/` and the negative control under `<task>/untrained/`.

---

## 8. Assumptions (correct any of these)
1. Tasks are mw_button-press-v2, mw_door-open-v2, mw_drawer-open-v2 and mw_plate-slide-v2.
2. The labeling oracle is chosen per type from `paths.label_oracle_candidates` (F1): `oracle_sac_1m` for demo and cum_estop, and most likely `oracle_sac_seeds/<env>/seed-1` for pref, corr, scalar and credit. Off-distribution GT policies, hacking references and the SAC config all come from `oracle_sac_1m`, the live oracle.
3. On CHPC the pool for these labels is `/scratch/.../mw_de_pool/<env>/pool.npz`, identical to the local mirror. Preflight asserts this.
4. Stored actions are the executed actions, clipped to ±(1−1e-5) by the buffers. SAC actions are squashed into (−1,1), and the reward model sees the clipped action.
5. A scalar act is one ARIC row. Grouping by `traj_id` is available as a flag.
6. ARIC's row-level `RandomState(0)` split is not reused, because it leaks across episodes.
7. ARIC's reward indexing is `(s_t, a_t) → r_t`, with the obs before the action. This matches the pool rollouts.
8. WandB is used only if `WANDB_API_KEY` is set. Tensorboard and CSV are always on.
9. Seeds: SAC seed s and reward-model seed s, for s ∈ {0, 1, 2}, in the pilot and the sweep. The oracle's own run is seed 0, so oracle-path seed 0 should land close to it. AsyncEnv timing makes runs not bit-reproducible.

## 9. Risks
- **Misspecification** (§0): partial-return RRC on regret labels. The partial-return ceiling column shows where this bites.
- **Narrow data**: about 200 episodes of one mediocre checkpoint per task (drawer-open pool success is 0.05), so SAC will leave that support quickly. The off-distribution correlation and disagreement columns are the early warning.
- **Reward hacking under SAC**: off-policy SAC exploits reward-model errors fast. This is measured, not just watched: §3.6 gives the normalized gap G, the overoptimization drop, the slope-test flag, learned-vs-true correlation on visited states, and ensemble disagreement. The pilot's best-NLL vs 3×-epoch comparison isolates how much reward-model overfitting adds.
- **Three seeds**: per-cell CIs are wide. Per-cell claims rest on mean ± std plus the flag rate out of 3, and pooled claims on the 4-task stratified bootstrap.
- **Cluster capacity**: 2 grn nodes. The SAC stage runs unchanged on any GPU partition if the queue is long.
