"""
RRC (reward-rational-choice) reward models for the RRC + SAC baseline.

One reward ensemble is fit per feedback type on ARIC's own choice structures
(see docs/rrc_sac_plan.md), then handed to the oracle SAC config as a frozen
reward. Nothing here modifies ARIC/CPL training code.
"""

from .ensemble import EnsembleMLP, RewardEnsemble
from .rrc_loss import per_act_weights, rrc_nll, segment_returns, window_returns
