"""
Drives PPO + learned-reward for the reward-CPL baseline comparison, mirroring
train_RL_agent.py's structure but wired to a research/-trained reward checkpoint
(scripts/export_reward_checkpoint.py) instead of a LightningNetwork, on an env
that matches research/envs/metaworld.py's obs/reward/horizon convention instead
of raw MetaWorld.

None of exp_manager.py's core PPO loop is modified: env-name dispatch is added via
a targeted monkeypatch (install_research_metaworld_dispatch), the reward clamp and
KL-to-mu term live entirely in ResearchRewardFn, the live-policy handoff for the KL
term lives in a callback (KLRewardSyncCallback) appended to exp_manager.callbacks,
and the action-std floor lives in a custom SB3 policy class referenced from
hyperparams/ppo.yml's research-metaworld-* block.

Usage:
  python multi_type_feedback/train_RL_agent_research.py \
      --environment drawer-open-v2 \
      --reward-checkpoint runs/reward_demo/portable_reward.pt \
      --beta-kl 2.0 \
      --mu-policy-path agents/BC_research_demo_mw_drawer-open-v2/bc_policy.zip \
      --seed 0

Run with --beta-kl 0 (or omit --mu-policy-path) to get the no-KL variant. The spec
calls for reporting the beta_KL in {2, 5} sweep, not just the best value -- run
this script once per value and report both.
"""

import os

from stable_baselines3.common.policies import ActorCriticPolicy
from train_baselines.exp_manager import ExperimentManager
from train_baselines.kl_reward_callback import KLRewardSyncCallback
from train_baselines.research_metaworld_compat import RESEARCH_MW_PREFIX, install_research_metaworld_dispatch

from multi_type_feedback.research_reward_fn import ResearchRewardFn
from multi_type_feedback.utils import TrainingUtils


def main():
    parser = TrainingUtils.setup_base_parser()
    # NOTE: --environment is already registered by setup_base_parser() (default "HalfCheetah-v5",
    # matching train_RL_agent.py/train_bc.py's own convention of reusing it unmodified) -- pass
    # e.g. --environment drawer-open-v2, a bare MetaWorld task id (no "metaworld-"/"research-
    # metaworld-" prefix; that prefix is added below via RESEARCH_MW_PREFIX).
    parser.add_argument(
        "--reward-checkpoint",
        type=str,
        default=None,
        help="Portable reward checkpoint from scripts/export_reward_checkpoint.py. "
        "Omit to run the oracle sanity check (ground-truth env reward, no learned reward).",
    )
    parser.add_argument("--beta-kl", type=float, default=0.0, help="KL-to-mu coefficient. 0 disables the KL term.")
    parser.add_argument(
        "--mu-policy-path",
        type=str,
        default=None,
        help="BC-pretrained reference policy (from train_bc.py's bc_trainer.policy.save(...)), "
        "used both as the KL reference and, if provided, to initialize pi <- mu.",
    )
    parser.add_argument("--sparse", action="store_true", help="Use sparse (success-only) reward instead of dense.")
    parser.add_argument(
        "--save-folder", type=str, default="trained_agents", help="Folder for finished feedback RL agents."
    )
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    install_research_metaworld_dispatch(sparse=args.sparse)
    env_id = f"{RESEARCH_MW_PREFIX}{args.environment}"
    device = TrainingUtils.get_device()

    if args.beta_kl > 0.0 and args.mu_policy_path is None:
        print("[warning] --beta-kl > 0 but no --mu-policy-path given; the KL-to-mu term will not fire.")

    reward_fn = None
    if args.reward_checkpoint is not None:
        reward_fn = ResearchRewardFn(
            reward_checkpoint_path=args.reward_checkpoint,
            kl_coeff=args.beta_kl,
            device=device,
        )

    mu_policy = None
    if args.mu_policy_path is not None:
        mu_policy = ActorCriticPolicy.load(args.mu_policy_path, device=device)
        mu_policy.set_training_mode(False)
        if reward_fn is not None:
            reward_fn.mu_policy = mu_policy

    run_name = f"RL_research_{args.environment}_beta{args.beta_kl}_seed{args.seed}"
    TrainingUtils.setup_wandb_logging(run_name, args, wandb_project_name=args.wandb_project_name)

    exp_manager = ExperimentManager(
        args,
        "ppo",
        env_id,
        os.path.join(args.save_folder, run_name),
        tensorboard_log=f"runs/{run_name}",
        seed=args.seed,
        log_interval=-1,
        reward_function=reward_fn,
        use_wandb_callback=True,
    )

    results = exp_manager.setup_experiment()
    if results is None:
        return
    model, _saved_hyperparams = results

    if mu_policy is not None:
        # init pi <- mu: requires the PPO policy_kwargs' net_arch (hyperparams/ppo.yml's
        # research-metaworld-* block) to match whatever net_arch train_bc.py's bc.BC used --
        # FloorStdActorCriticPolicy only overrides _get_action_dist_from_latent, so its
        # state_dict keys are identical to plain ActorCriticPolicy's when net_arch matches.
        model.policy.load_state_dict(mu_policy.state_dict())
        print(f"Initialized PPO policy from BC reference: {args.mu_policy_path}")

    if reward_fn is not None and args.beta_kl > 0.0 and mu_policy is not None:
        exp_manager.callbacks.append(KLRewardSyncCallback(reward_fn))

    exp_manager.learn(model)
    exp_manager.save_trained_model(model)


if __name__ == "__main__":
    main()
