"""
Train one RRC reward ensemble on one feedback type's choice structures.

Every member trains on the full training split with its own init seed and its own
per-epoch permutation of acts; members share nothing but the vectorized forward
pass. Each member's best held-out-NLL weights are kept; training stops when no
member has improved for `patience` epochs (or at max_epochs). Pilot runs freeze
`best` at that same early-stopping point, then keep training to 3x the ensemble-best
epoch so a checkpoint fit ~3x as long exists (exported as epoch3x).
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .data import ChoiceSet
from .ensemble import EnsembleMLP, RewardEnsemble, clip_grad_norm_per_member
from .rrc_loss import choice_correct, per_act_weights, rrc_nll, segment_returns, window_returns


class TorchChoiceData:
    """A ChoiceSet's tensors on one device, with option scoring for train (member-major) and eval."""

    def __init__(self, cs: ChoiceSet, device: torch.device):
        self.cs = cs
        self.kind = cs.kind
        self.gamma = cs.gamma
        self.window = cs.window
        f = lambda x: None if x is None else torch.as_tensor(np.ascontiguousarray(x), device=device)
        self.obs = f(cs.obs.astype(np.float32))
        self.act = f(cs.act.astype(np.float32))
        self.target = f(cs.target.astype(np.float32))
        self.step_mask = None if cs.step_mask is None else f(cs.step_mask.astype(np.float32))
        self.option_mask = None if cs.option_mask is None else f(cs.option_mask.astype(bool))
        self.device = device

    def __len__(self):
        return self.obs.shape[0]

    def _returns(self, r: torch.Tensor, step_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.kind == "windows":
            return window_returns(r, self.window, self.gamma)
        return segment_returns(r, self.gamma, step_mask)

    def member_scores(self, net: EnsembleMLP, idx: torch.Tensor) -> torch.Tensor:
        """idx: (E, B) rows per member -> (E, B, n_options), member m scored by member m only."""
        obs, act = self.obs[idx], self.act[idx]
        r = net(obs, act, member_major=True)
        mask = None if self.step_mask is None else self.step_mask[idx]
        return self._returns(r, mask)

    @torch.no_grad()
    def all_scores(self, net: EnsembleMLP, rows: Optional[torch.Tensor] = None, chunk: int = 512) -> torch.Tensor:
        """(E, N, n_options) scores of every member on every row (shared inputs)."""
        rows = torch.arange(len(self), device=self.device) if rows is None else rows
        outs = []
        for i in range(0, len(rows), chunk):
            b = rows[i : i + chunk]
            r = net(self.obs[b], self.act[b])
            mask = None if self.step_mask is None else self.step_mask[b]
            outs.append(self._returns(r, mask))
        return torch.cat(outs, dim=1)

    def option_mask_for(self, rows: torch.Tensor) -> Optional[torch.Tensor]:
        return None if self.option_mask is None else self.option_mask[rows]


@torch.no_grad()
def evaluate_split(data: TorchChoiceData, net: EnsembleMLP, rows: np.ndarray, act_ids: np.ndarray, beta: float) -> Dict:
    """Per-member and ensemble NLL (mean per act) and choice accuracy on `rows`."""
    if len(rows) == 0:
        return {}
    rows_t = torch.as_tensor(rows, device=data.device)
    scores = data.all_scores(net, rows_t)  # (E, n, K)
    target = data.target[rows_t]
    omask = data.option_mask_for(rows_t)
    w = torch.as_tensor(per_act_weights(act_ids[rows]), device=data.device)
    out = {}
    nll = rrc_nll(scores, target, omask, beta)  # (E, n)
    acc = choice_correct(scores, target, omask)
    ens_scores = scores.mean(dim=0)  # option returns are linear in r, so this = mean-member reward
    for m in range(scores.shape[0]):
        out[f"nll_m{m}"] = float((nll[m] * w).sum())
        out[f"acc_m{m}"] = float((acc[m] * w).sum())
    out["nll_ens"] = float((rrc_nll(ens_scores, target, omask, beta) * w).sum())
    out["acc_ens"] = float((choice_correct(ens_scores, target, omask) * w).sum())
    return out


class ActBatcher:
    """Batches of `batch_acts` acts, a separate permutation per member, padded to equal row count."""

    def __init__(self, train_rows: np.ndarray, act_ids: np.ndarray, batch_acts: int, ensemble_size: int, rng: np.random.RandomState):
        self.batch_acts = batch_acts
        self.E = ensemble_size
        self.rng = rng
        acts = act_ids[train_rows]
        self.uniq, inv = np.unique(acts, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        self.rows_sorted = train_rows[order]
        counts = np.bincount(inv, minlength=len(self.uniq))
        self.starts = np.concatenate([[0], np.cumsum(counts)])
        self.counts = counts
        self.identity = bool(np.all(counts == 1))

    @property
    def n_acts(self) -> int:
        return len(self.uniq)

    def epoch(self):
        perms = [self.rng.permutation(self.n_acts) for _ in range(self.E)]
        for b in range(0, self.n_acts, self.batch_acts):
            idx, wts = [], []
            for m in range(self.E):
                acts = perms[m][b : b + self.batch_acts]
                if self.identity:
                    rows = self.rows_sorted[acts]
                    w = np.full(len(acts), 1.0 / len(acts), np.float32)
                else:
                    rows = np.concatenate([self.rows_sorted[self.starts[a] : self.starts[a + 1]] for a in acts])
                    w = np.concatenate([np.full(self.counts[a], 1.0 / (self.counts[a] * len(acts)), np.float32) for a in acts])
                idx.append(rows)
                wts.append(w)
            width = max(len(r) for r in idx)
            I = np.zeros((self.E, width), np.int64)
            W = np.zeros((self.E, width), np.float32)
            for m in range(self.E):
                I[m, : len(idx[m])] = idx[m]
                W[m, : len(wts[m])] = wts[m]
            yield I, W


@torch.no_grad()
def standardization_stats(net: EnsembleMLP, obs: np.ndarray, act: np.ndarray, device, chunk: int = 1 << 16):
    E = net.ensemble_size
    s1 = torch.zeros(E, dtype=torch.float64, device=device)
    s2 = torch.zeros(E, dtype=torch.float64, device=device)
    n = 0
    for i in range(0, len(obs), chunk):
        o = torch.as_tensor(obs[i : i + chunk], device=device)
        a = torch.as_tensor(act[i : i + chunk], device=device)
        r = net(o, a).double()
        s1 += r.sum(1)
        s2 += (r * r).sum(1)
        n += r.shape[1]
    mu = s1 / n
    sigma = (s2 / n - mu * mu).clamp(min=0).sqrt().clamp(min=1e-6)
    return mu.float().cpu(), sigma.float().cpu()


def _state_copy(net: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def train_reward_ensemble(cs: ChoiceSet, split: Dict, cfg: Dict, out_dir: str, seed: int, device: str = "cpu",
                          pilot: bool = False, log=print, epoch_callback=None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)
    mcfg, tcfg = cfg["model"], cfg["train"]
    beta = float(cfg["rrc"].get("beta", 1.0))
    E = int(mcfg.get("ensemble_size", 3))
    torch.manual_seed(seed)
    net = EnsembleMLP(
        cs.obs.shape[-1], cs.act.shape[-1], ensemble_size=E, hidden_layers=mcfg.get("hidden_layers", (256, 256, 256)),
        output=mcfg.get("output", "tanh"), seeds=[seed * 100 + m for m in range(E)],
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=float(tcfg.get("lr", 3e-4)), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    data = TorchChoiceData(cs, device)
    train_rows, val_rows = split["train_idx"], split["val_idx"]
    batcher = ActBatcher(train_rows, cs.act_ids, int(tcfg.get("batch_acts", 128)), E, np.random.RandomState(seed))
    clip = float(tcfg.get("grad_clip", 1.0))
    max_epochs = int(tcfg.get("max_epochs", 200))
    patience = int(tcfg.get("patience", 10))
    save_every = int(tcfg.get("save_every", 25))
    pilot_max = int(tcfg.get("pilot_max_epochs", 600))

    epoch_dir = os.path.join(out_dir, "_epochs")
    if pilot:
        os.makedirs(epoch_dir, exist_ok=True)
    kept: Dict[str, Dict] = {}  # name -> state dict
    best_val = np.full(E, np.inf)
    best_epoch = np.zeros(E, np.int64)
    best_state = _state_copy(net)
    ens_best, ens_best_epoch = np.inf, 0
    since_improve = np.zeros(E, np.int64)
    es, pilot_target = None, None
    rows_csv: List[Dict] = []

    def log_epoch(epoch, grad_norms, dt):
        tr = evaluate_split(data, net, train_rows, cs.act_ids, beta)
        va = evaluate_split(data, net, val_rows, cs.act_ids, beta)
        row = dict(epoch=epoch, time=round(dt, 3))
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        for m in range(E):
            row[f"grad_norm_m{m}"] = float(grad_norms[m]) if grad_norms is not None else float("nan")
        rows_csv.append(row)
        if epoch_callback is not None:
            epoch_callback(row)  # e.g. WandB logging; one call per epoch, epoch 0 = before training
        return va

    va = log_epoch(0, None, 0.0)
    log(f"[rrc] epoch 0  val_nll_ens={va['nll_ens']:.4f} val_acc_ens={va['acc_ens']:.3f}")
    epoch = 0
    while True:
        epoch += 1
        t0 = time.time()
        net.train()
        gn = []
        for I, W in batcher.epoch():
            I_t = torch.as_tensor(I, device=device)
            W_t = torch.as_tensor(W, device=device)
            scores = data.member_scores(net, I_t)  # (E, B, K)
            target = data.target[I_t]
            omask = None if data.option_mask is None else data.option_mask[I_t]
            loss_rows = rrc_nll(scores, target, omask, beta)  # (E, B)
            loss = (loss_rows * W_t).sum()  # = sum over members of each member's per-act mean
            optim.zero_grad(set_to_none=True)
            loss.backward()
            gn.append(clip_grad_norm_per_member(net, clip).detach().cpu().numpy())
            optim.step()
        net.eval()
        va = log_epoch(epoch, np.mean(gn, axis=0), time.time() - t0)

        val_m = np.array([va[f"nll_m{m}"] for m in range(E)])
        improved = val_m < best_val
        state = None
        if improved.any():
            state = net.state_dict()
            for k in best_state:
                for m in np.nonzero(improved)[0]:
                    best_state[k][m].copy_(state[k][m])
        best_val = np.where(improved, val_m, best_val)
        best_epoch = np.where(improved, epoch, best_epoch)
        since_improve = np.where(improved, 0, since_improve + 1)
        if va["nll_ens"] < ens_best:
            ens_best, ens_best_epoch = va["nll_ens"], epoch
        if epoch % save_every == 0:
            kept[f"epoch_{epoch:03d}"] = dict(state=_state_copy(net), epochs=[epoch] * E)
        if pilot:
            torch.save(_state_copy(net), os.path.join(epoch_dir, f"epoch_{epoch:03d}.pt"))
        if epoch % 10 == 0 or epoch == 1:
            log(f"[rrc] epoch {epoch}  val_nll_ens={va['nll_ens']:.4f} acc_ens={va['acc_ens']:.3f} "
                f"best_epochs={best_epoch.tolist()} ens_best_epoch={ens_best_epoch}")

        stop = np.all(since_improve >= patience) or epoch >= max_epochs
        if stop and es is None:
            # Early-stopping point: `best` is frozen here in both modes, so a pilot run's `best`
            # export is the same checkpoint a normal (early-stopped) run produces.
            es = dict(epoch=epoch, best_state={k: v.clone() for k, v in best_state.items()},
                      best_epoch=best_epoch.copy(), best_val=best_val.copy(), ens_best_epoch=ens_best_epoch,
                      ens_best=ens_best)
            pilot_target = min(max(3 * ens_best_epoch, epoch), pilot_max)
        if es is not None and (not pilot or epoch >= pilot_target):
            break

    kept["last"] = dict(state=_state_copy(net), epochs=[epoch] * E)
    kept["best"] = dict(state=es["best_state"], epochs=es["best_epoch"].tolist())
    best_epoch, best_val, ens_best_epoch, ens_best = es["best_epoch"], es["best_val"], es["ens_best_epoch"], es["ens_best"]
    if pilot:
        e3 = int(min(round(3 * ens_best_epoch), epoch))
        kept["epoch3x"] = dict(state=torch.load(os.path.join(epoch_dir, f"epoch_{e3:03d}.pt"), map_location=device),
                               epochs=[e3] * E)
        kept["best_ens"] = dict(state=torch.load(os.path.join(epoch_dir, f"epoch_{ens_best_epoch:03d}.pt"), map_location=device),
                                epochs=[ens_best_epoch] * E)
        for f in os.listdir(epoch_dir):
            os.remove(os.path.join(epoch_dir, f))
        os.rmdir(epoch_dir)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        keys = list(rows_csv[-1].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_csv:
            w.writerow({k: r.get(k, "") for k in keys})

    obs_tr, act_tr, rew_tr = cs.subset(train_rows).valid_steps()
    summary = dict(
        type=cs.type, seed=seed, epochs_run=epoch, early_stop_epoch=int(es["epoch"]),
        best_epochs=best_epoch.tolist(), best_val_nll=best_val.tolist(),
        ens_best_epoch=int(ens_best_epoch), ens_best_val_nll=float(ens_best), pilot=pilot,
        n_train_acts=int(batcher.n_acts), n_val_rows=int(len(val_rows)),
        true_reward_mean=float(rew_tr.mean()), true_reward_std=float(rew_tr.std()),
        exports={},
    )
    for name, k in kept.items():
        net.load_state_dict(k["state"])
        mu, sigma = standardization_stats(net, obs_tr, act_tr, device)
        model = RewardEnsemble(net, mu, sigma)
        extra = dict(
            type=cs.type, gamma=cs.gamma, beta=beta, seed=seed, export=name, member_epochs=list(map(int, k["epochs"])),
            mu_true=summary["true_reward_mean"], sigma_true=summary["true_reward_std"], data=cs.meta.get("path"),
        )
        model.save(os.path.join(out_dir, f"{name}.pt"), extra=extra)
        summary["exports"][name] = dict(member_epochs=extra["member_epochs"])
    net.load_state_dict(kept["best"]["state"])
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def export_untrained(cs: ChoiceSet, split: Dict, cfg: Dict, out_dir: str, seed: int, device: str = "cpu") -> str:
    """Negative control: a random-init ensemble, standardized on the training data like a trained one."""
    os.makedirs(out_dir, exist_ok=True)
    mcfg = cfg["model"]
    E = int(mcfg.get("ensemble_size", 3))
    net = EnsembleMLP(cs.obs.shape[-1], cs.act.shape[-1], ensemble_size=E, hidden_layers=mcfg.get("hidden_layers", (256, 256, 256)),
                      output=mcfg.get("output", "tanh"), seeds=[seed * 100 + m for m in range(E)]).to(device)
    obs_tr, act_tr, rew_tr = cs.subset(split["train_idx"]).valid_steps()
    mu, sigma = standardization_stats(net, obs_tr, act_tr, torch.device(device))
    path = os.path.join(out_dir, "best.pt")
    RewardEnsemble(net, mu, sigma).save(path, extra=dict(
        type="untrained", stats_from=cs.type, gamma=cs.gamma, seed=seed, export="best", member_epochs=[0] * E,
        mu_true=float(rew_tr.mean()), sigma_true=float(rew_tr.std()), data=cs.meta.get("path"),
    ))
    return path
