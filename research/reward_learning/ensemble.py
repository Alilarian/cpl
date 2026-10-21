"""
Vectorized reward ensemble.

Every member is an independent MLP r_m(s, a) -> R, stored as batched weights of
shape (E, in, out) so all members run in one bmm. Members never share
parameters or minibatches: the trainer gives each member its own init seed and
its own per-epoch permutation, and gradient clipping is applied per member.

RewardEnsemble wraps the trained network with per-member standardization
(mu_m, sigma_m over that type's training (s, a)), so the reward handed to RL is
    r(s, a) = mean_m (r_m(s, a) - mu_m) / sigma_m
"""

import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn


class EnsembleLinear(nn.Module):
    def __init__(self, ensemble_size: int, in_features: int, out_features: int):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

    def reset_member(self, m: int, generator: torch.Generator) -> None:
        # Same distribution as nn.Linear's default init (kaiming_uniform with a=sqrt(5)
        # reduces to U(-1/sqrt(in), 1/sqrt(in)) for both weight and bias).
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            w = torch.rand(self.in_features, self.out_features, generator=generator) * 2 - 1
            b = torch.rand(1, self.out_features, generator=generator) * 2 - 1
            self.weight[m].copy_(w * bound)
            self.bias[m].copy_(b * bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, N, in) -> (E, N, out)
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleMLP(nn.Module):
    """E independent MLPs over concat(obs, action); returns per-member per-step rewards."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble_size: int = 3,
        hidden_layers: Sequence[int] = (256, 256, 256),
        output: str = "tanh",
        negative_slope: float = 0.01,
        seeds: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        assert output in ("tanh", "linear"), output
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_size = ensemble_size
        self.hidden_layers = list(hidden_layers)
        self.output = output
        self.negative_slope = negative_slope

        dims = [obs_dim + act_dim] + self.hidden_layers + [1]
        self.layers = nn.ModuleList(EnsembleLinear(ensemble_size, i, o) for i, o in zip(dims[:-1], dims[1:]))
        seeds = list(range(ensemble_size)) if seeds is None else list(seeds)
        assert len(seeds) == ensemble_size
        self.seeds = seeds
        for m, s in enumerate(seeds):
            gen = torch.Generator().manual_seed(int(s))
            for layer in self.layers:
                layer.reset_member(m, gen)

    def arch(self) -> Dict:
        return dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            ensemble_size=self.ensemble_size,
            hidden_layers=self.hidden_layers,
            output=self.output,
            negative_slope=self.negative_slope,
            seeds=self.seeds,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor, member_major: bool = False) -> torch.Tensor:
        """
        member_major=False: obs (*B, obs_dim) is shared by all members.
        member_major=True:  obs (E, *B, obs_dim), member m sees obs[m] (its own minibatch).
        act has the same leading shape as obs. Returns (E, *B) per-member per-step rewards.
        """
        assert obs.shape[-1] == self.obs_dim, f"obs dim {obs.shape[-1]} != {self.obs_dim}"
        assert act.shape[-1] == self.act_dim, f"act dim {act.shape[-1]} != {self.act_dim}"
        x = torch.cat((obs, act), dim=-1)
        E = self.ensemble_size
        if member_major:
            assert x.shape[0] == E, f"member-major input needs leading dim {E}, got {x.shape[0]}"
            lead = x.shape[1:-1]
            x = x.reshape(E, -1, x.shape[-1])
        else:
            lead = x.shape[:-1]
            x = x.reshape(1, -1, x.shape[-1]).expand(E, -1, -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.leaky_relu(x, self.negative_slope)
        x = x.squeeze(-1)
        if self.output == "tanh":
            x = torch.tanh(x)
        return x.reshape(E, *lead)


def clip_grad_norm_per_member(net: EnsembleMLP, max_norm: float) -> torch.Tensor:
    """Clip each member's gradient to max_norm independently. Returns (E,) pre-clip norms."""
    E = net.ensemble_size
    sq = None
    for layer in net.layers:
        for p in (layer.weight, layer.bias):
            if p.grad is None:
                continue
            g = p.grad.reshape(E, -1).pow(2).sum(dim=1)
            sq = g if sq is None else sq + g
    norms = sq.sqrt()
    scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    for layer in net.layers:
        for p in (layer.weight, layer.bias):
            if p.grad is not None:
                p.grad.mul_(scale.view(E, *([1] * (p.grad.dim() - 1))))
    return norms


class RewardEnsemble(nn.Module):
    """Frozen, standardized reward ensemble used by RL and diagnostics."""

    def __init__(self, net: EnsembleMLP, mu: Optional[torch.Tensor] = None, sigma: Optional[torch.Tensor] = None):
        super().__init__()
        self.net = net
        E = net.ensemble_size
        self.register_buffer("mu", torch.zeros(E) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sigma", torch.ones(E) if sigma is None else torch.as_tensor(sigma, dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        return self.mu.device

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def member_rewards(self, obs, act, chunk: int = 1 << 16) -> torch.Tensor:
        """Raw per-member rewards, (E, *B)."""
        obs, act = self._to_tensor(obs), self._to_tensor(act)
        lead = obs.shape[:-1]
        o = obs.reshape(-1, obs.shape[-1])
        a = act.reshape(-1, act.shape[-1])
        outs = [self.net(o[i : i + chunk], a[i : i + chunk]) for i in range(0, o.shape[0], chunk)]
        out = torch.cat(outs, dim=1) if outs else torch.zeros(self.net.ensemble_size, 0, device=self.device)
        return out.reshape(self.net.ensemble_size, *lead)

    @torch.no_grad()
    def standardized_members(self, obs, act) -> torch.Tensor:
        r = self.member_rewards(obs, act)
        shape = (-1,) + (1,) * (r.dim() - 1)
        return (r - self.mu.view(shape)) / self.sigma.view(shape)

    @torch.no_grad()
    def reward(self, obs, act) -> torch.Tensor:
        """Ensemble reward mean_m (r_m - mu_m) / sigma_m, shape (*B)."""
        return self.standardized_members(obs, act).mean(dim=0)

    @torch.no_grad()
    def reward_and_std(self, obs, act):
        z = self.standardized_members(obs, act)
        return z.mean(dim=0), z.std(dim=0, unbiased=False)

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        torch.save(
            dict(
                arch=self.net.arch(),
                state_dict=self.net.state_dict(),
                mu=self.mu.cpu(),
                sigma=self.sigma.cpu(),
                extra={} if extra is None else extra,
            ),
            path,
        )

    @classmethod
    def load(cls, path: str, device: Union[str, torch.device] = "cpu") -> "RewardEnsemble":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = EnsembleMLP(**ckpt["arch"])
        net.load_state_dict(ckpt["state_dict"])
        model = cls(net, ckpt["mu"], ckpt["sigma"])
        model.extra = ckpt.get("extra", {})
        model.eval()
        return model.to(device)
