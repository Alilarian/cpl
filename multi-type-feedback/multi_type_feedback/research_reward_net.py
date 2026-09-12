"""
Dependency-free reconstruction of research/networks/mlp.py::ContinuousMLPCritic's
forward pass (ensemble_size=1 case only -- every reward_*.yaml Phase-1 config uses
ensemble_size=1), so a checkpoint exported by ../scripts/export_reward_checkpoint.py
can be loaded here without depending on the `research` package at all. research/ and
multi-type-feedback/ are kept in separate Python environments -- only weights cross
that boundary, never a live import.

Layer ordering must match research/networks/common.py::MLP exactly:
    Linear -> [Dropout] -> ReLU  (per hidden layer)  -> Linear (output)
"""

from typing import Dict, List

import torch
import torch.nn as nn


class ResearchRewardNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_layers: List[int] = (512, 512), dropout: float = 0.0):
        super().__init__()
        layers = []
        last_dim = obs_dim + act_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(last_dim, dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            last_dim = dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, action), dim=-1)
        return self.net(x).squeeze(-1)  # (N,) for 2D input, (N, T) for 3D input


def load_research_reward_net(checkpoint: Dict) -> ResearchRewardNet:
    """checkpoint is the dict saved by scripts/export_reward_checkpoint.py."""
    arch = checkpoint["arch"]
    net = ResearchRewardNet(arch["obs_dim"], arch["act_dim"], arch["hidden_layers"], arch["dropout"])
    # research/networks/mlp.py::ContinuousMLPCritic names its inner MLP "q", and
    # research/networks/common.py::MLP wraps its Sequential as "net" -- so the
    # exported reward-net state_dict keys are "q.net.<idx>.weight"/"...bias".
    prefix = "q.net."
    state_dict = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in checkpoint["state_dict"].items()}
    net.net.load_state_dict(state_dict)
    net.eval()
    return net
