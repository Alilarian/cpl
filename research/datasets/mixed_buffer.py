import math
from typing import Any, Dict, List, Optional

import gym
import torch


class MixedFeedbackBuffer(torch.utils.data.IterableDataset):
    """
    Wraps N feedback-type sub-datasets and yields one named sub-batch per
    component every step, e.g.:

        {"credit_assignment": {"obs": ..., "action": ..., "label": ...},
         "demo":              {"obs": ..., "action": ..., "label": ...}}

    This works for *any* mix of feedback types because every existing buffer
    class (DemoBuffer, CorrBuffer, PMCreditAssignmentBuffer, ...) already
    yields the same tensor contract:

        obs    : (B, K, T, obs_dim)
        action : (B, K, T, act_dim)
        label  : (B,)  index of the preferred/chosen candidate among the K

    (demo/pref/corr/scalar always set label=0; credit_assignment sets a
    variable per-row chosen index.) MixedCPL (research/algs/mixed_cpl.py)
    scores every component with the same shared loss regardless of which
    type it is, so mixing in a new feedback type is a config change here
    (append to `components`), not new code in either this class or MixedCPL.

    Each component keeps its own K, so sub-batches are NOT concatenated into
    one tensor (their K/T can differ) -- they're kept as separate named
    entries, and MixedCPL runs the shared encoder/actor on each separately.

    "Randomly pick N rows from the existing dataset" (as opposed to a nested
    top-B prefix, which is what plain `capacity` gives you) is handled by the
    `subsample_n`/`subsample_seed` kwargs that every one of these buffer
    classes already implements -- this class does not re-implement sampling,
    it just instantiates each component's own buffer with whatever
    dataset_kwargs (path, subsample_n, subsample_seed, ...) are given.

    Args:
        batch_size : total rows across all components per yielded step. Each
                     component's own sub-batch size is round(batch_size *
                     weight), with the last-listed component absorbing the
                     rounding remainder so the sizes sum exactly to
                     batch_size.
        components : list of
                        name           : str,   used as the sub-batch's key
                        dataset_class  : str,   resolved via
                                         research.datasets (e.g. "DemoBuffer")
                        weight         : float, this component's share of
                                         batch_size (need not sum to 1;
                                         normalized internally)
                        dataset_kwargs : dict passed straight through to the
                                         underlying buffer class (path,
                                         subsample_n, subsample_seed, ...)

    The overall iterator runs for as many steps as the *largest* (by weight)
    component would take to complete one epoch at its own sub-batch size;
    every other component's sub-iterator auto-restarts (reshuffling) via
    StopIteration whenever it's exhausted first -- the same "restart when
    exhausted" pattern DemoCPL._next_bc_batch already uses for its bc_pool
    loader.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        batch_size: int = 96,
        components: Optional[List[Dict[str, Any]]] = None,
    ):
        assert components, "MixedFeedbackBuffer requires at least one component."

        # Lazy import: this module lives inside research.datasets itself, so
        # importing the package at module load time would be circular.
        import research.datasets as datasets_module

        weights = [float(c["weight"]) for c in components]
        total_weight = sum(weights)
        assert total_weight > 0, "component weights must sum to a positive number."
        normalized = [w / total_weight for w in weights]

        sub_batch_sizes = [round(batch_size * w) for w in normalized[:-1]]
        sub_batch_sizes.append(batch_size - sum(sub_batch_sizes))  # remainder -> last component
        assert all(s > 0 for s in sub_batch_sizes), (
            f"batch_size={batch_size} too small for component weights {normalized} "
            f"-- some component would get a sub-batch size of 0."
        )

        self.names = [c["name"] for c in components]
        self.datasets = []
        self.dataset_sizes = []
        for component, sub_batch_size in zip(components, sub_batch_sizes):
            dataset_class = getattr(datasets_module, component["dataset_class"])
            dataset_kwargs = dict(component.get("dataset_kwargs", {}))
            dataset_kwargs["batch_size"] = sub_batch_size
            dataset = dataset_class(observation_space, action_space, **dataset_kwargs)
            self.datasets.append(dataset)
            self.dataset_sizes.append(len(dataset))

        # The component with the largest share of the mix drives the epoch
        # length (steps per __iter__ call); smaller components cycle faster
        # and restart (reshuffling) as needed to fill in every step.
        self._primary_idx = max(range(len(normalized)), key=lambda i: normalized[i])
        self.batch_size = batch_size

    def __len__(self):
        return self.dataset_sizes[self._primary_idx]

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        primary_dataset = self.datasets[self._primary_idx]
        num_steps = math.ceil(len(primary_dataset) / primary_dataset.batch_size)

        for _ in range(num_steps):
            sub_batches = {}
            for i, name in enumerate(self.names):
                try:
                    sub_batches[name] = next(iterators[i])
                except StopIteration:
                    iterators[i] = iter(self.datasets[i])
                    sub_batches[name] = next(iterators[i])
            yield sub_batches
