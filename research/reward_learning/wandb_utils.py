"""
Optional WandB logging for RRC reward training (SAC runs use scripts/train.py::try_wandb_setup).

Project: $WANDB_PROJECT (default rrc-sac-2m), entity: $WANDB_ENTITY (setup_shell.sh exports a7a7).
A run is only created when a WandB login / API key exists or WANDB_MODE is offline/disabled;
otherwise everything is skipped and training proceeds unchanged.
"""

import os
from typing import Dict, Optional

DEFAULT_PROJECT = "rrc-sac-2m"


def flatten(d: Dict, prefix: str = "", sep: str = ".") -> Dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        elif isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, dict) for x in v):
            for i, x in enumerate(v):
                out.update(flatten(x, f"{key}{sep}{i}", sep))
        else:
            out[key] = v
    return out


def init_run(name: str, group: str, config: Dict, job_type: str, run_dir: str):
    """Returns a wandb run, or None if wandb is unavailable / not logged in."""
    try:
        import wandb
    except ImportError:
        return None
    mode = os.getenv("WANDB_MODE", "")
    if mode not in ("offline", "disabled") and not (os.getenv("WANDB_API_KEY") or wandb.api.api_key):
        print("[wandb] no API key / login found -- skipping reward-training logging")
        return None
    try:
        run = wandb.init(
            entity=os.getenv("WANDB_ENTITY") or None,
            project=os.getenv("WANDB_PROJECT", DEFAULT_PROJECT),
            name=name, group=group, job_type=job_type,
            config=flatten(config), dir=os.getenv("WANDB_DIR", run_dir),
        )
    except Exception as e:  # never let logging break training
        print(f"[wandb] init failed ({e}) -- continuing without it")
        return None
    if run is not None:
        with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
            f.write(run.id)
    return run


def finish(run, summary: Optional[Dict] = None) -> None:
    if run is None:
        return
    try:
        if summary:
            run.summary.update({k: v for k, v in flatten(summary).items() if isinstance(v, (int, float, str, bool))})
        run.finish()
    except Exception as e:
        print(f"[wandb] finish failed ({e})")
