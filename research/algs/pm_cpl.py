"""
PointMass CPL variants for the 2D PointMass navigation environment.

Each class extends a base CPL variant with a ``validation_extras`` override
that saves a policy vector-field snapshot at every eval step.  The snapshot
shows mean-action arrows over the [0,1]² arena, optionally overlaid on V*(s)
from a precomputed VI table.

Classes
-------
PointMassCPL      — pref feedback   (extends CPL)
PointMassDemoCPL  — corr / demo / seq_estop feedback  (extends DemoCPL)
PointMassEstopCPL — estop feedback  (extends EstopCPL)

Extra alg_kwargs (all classes)
-------------------------------
advantage_npz : str, optional
    Path to advantage_vi_*.npz from compute_advantage_vi.py.
    When given, V*(s) is drawn as a background heatmap.
vis_grid_n : int
    Grid side length for the quiver plot (default 30 → 900 arrows).
"""

import importlib.util
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

from .cpl import CPL, DemoCPL, EstopCPL


# ---------------------------------------------------------------------------
# Module loading helpers (avoid triggering MetaWorld via research.__init__)
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_once(dotted_name: str, rel_path: str):
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared visualization mixin
# ---------------------------------------------------------------------------

class _PointMassVizMixin:
    """
    Vector-field visualization for PointMass — mixin shared by all PM variants.

    MRO note: place before the CPL base class so __init__ is called first,
    passing remaining kwargs through super() to the CPL base.
    """

    def __init__(
        self,
        *args,
        advantage_npz: Optional[str] = None,
        vis_grid_n: int = 30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._advantage_npz = advantage_npz
        self._vis_grid_n    = vis_grid_n
        self._avi           = None
        self._pm_env        = None

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------

    def _env(self):
        if self._pm_env is None:
            pm = _load_once("research.envs.pointmass",
                            "research/envs/pointmass.py")
            self._pm_env = pm.PointMassGymEnv()
        return self._pm_env

    def _vi(self):
        if self._avi is None and self._advantage_npz is not None:
            avi_mod = _load_once("research.utils.advantage_vi",
                                 "research/utils/advantage_vi.py")
            self._avi = avi_mod.AdvantageVI.load(self._advantage_npz)
        return self._avi

    # ------------------------------------------------------------------
    # Vector-field snapshot
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _snapshot(self, path: str, step: int) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        env  = self._env()
        avi  = self._vi()
        N    = self._vis_grid_n
        xs   = np.linspace(0.0, 1.0, N)
        GX, GY = np.meshgrid(xs, xs, indexing="ij")
        states = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)

        obs_t   = torch.from_numpy(states).to(self.device)
        obs_enc = self.network.encoder(obs_t)
        out     = self.network.actor(obs_enc)

        if isinstance(out, torch.distributions.Distribution):
            mean_act = out.base_dist.loc if hasattr(out, "base_dist") else out.mean
        else:
            mean_act = out

        mean_act = mean_act.cpu().numpy()
        DX  = mean_act[:, 0].reshape(N, N)
        DY  = mean_act[:, 1].reshape(N, N)
        mag = np.hypot(DX, DY)

        fig, ax = plt.subplots(figsize=(5, 5))

        if avi is not None:
            im = ax.imshow(
                avi.V.T, origin="lower", extent=[0, 1, 0, 1],
                cmap="RdYlGn", aspect="equal",
                interpolation="nearest", alpha=0.55,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V*(s)")

        ax.quiver(
            GX, GY, DX, DY, mag.ravel(),
            cmap="plasma", scale=30, width=0.003, alpha=0.85,
        )

        ax.add_patch(plt.Circle(env.goal, env.goal_dist_thresh,
                                color="green", alpha=0.35, zorder=5))
        ax.plot(*env.goal, "g^", ms=8, zorder=6)
        ax.add_patch(plt.Circle(env.trap, env.trap_dist_thresh,
                                color="red", alpha=0.35, zorder=5))
        ax.plot(*env.trap, "rx", ms=8, mew=2, zorder=6)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Policy vector field  (step {step:,})", fontsize=10)
        ax.legend(
            handles=[
                mpatches.Patch(color="green", alpha=0.5, label="goal"),
                mpatches.Patch(color="red",   alpha=0.5, label="trap"),
            ],
            fontsize=7, loc="upper right",
        )

        frames_dir = os.path.join(path, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        fname = os.path.join(frames_dir, f"policy_{step:07d}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return fname

    # ------------------------------------------------------------------
    # Trainer hook
    # ------------------------------------------------------------------

    def validation_extras(self, path: str, step: int) -> Dict:
        try:
            fname = self._snapshot(path, step)
        except Exception as exc:
            print(f"[PointMass] vector-field snapshot failed at step {step}: {exc}")
            return {}

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"policy_vector_field": wandb.Image(fname)}, step=step)
        except ImportError:
            pass

        return {}


# ---------------------------------------------------------------------------
# Concrete PointMass CPL classes
# ---------------------------------------------------------------------------

class PointMassCPL(_PointMassVizMixin, CPL):
    """CPL for pairwise preference feedback on PointMass. Dataset: PMFeedbackBuffer."""


class PointMassDemoCPL(_PointMassVizMixin, DemoCPL):
    """CPL for corr / demo / seq_estop feedback on PointMass. Dataset: CorrBuffer or DemoBuffer."""


class PointMassEstopCPL(_PointMassVizMixin, EstopCPL):
    """CPL for non-sequential e-stop feedback on PointMass. Dataset: EstopBuffer."""
