"""FLAME defense implementation.

This module implements a simplified version of the FLAME algorithm for
malicious update detection and model aggregation in federated learning.

The algorithm performs the following steps:

1. **Clustering** of client updates using HDBSCAN with cosine similarity.
2. **Median norm clipping** of client updates.
3. **Aggregation** of updates that belong to the main cluster.
4. **Noise addition** to the aggregated global model to improve privacy.

The implementation follows the pseudocode provided in the task
description.  It is intended as a reference implementation rather than a
fully‑featured production component.
"""

from __future__ import annotations

from typing import Dict, List

import torch

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None


class FlameDefense:
    """Apply the FLAME defense to a set of client model weights.

    Parameters
    ----------
    global_model: ``torch.nn.Module``
        The model that will be updated in‑place with the aggregated client
        updates.
    defense: bool, optional
        If ``False`` the class will still aggregate the client updates but
        will skip all FLAME‑specific defences.
    """

    def __init__(self, global_model: torch.nn.Module, defense: bool = True) -> None:
        self.global_model = global_model
        self.conf = {"defense": "flame" if defense else None}

    @staticmethod
    def _flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in state.values()])

    def defend(self, clients_weight: List[Dict[str, torch.Tensor]]) -> None:
        """Run the FLAME algorithm and update ``self.global_model`` in‑place."""
        if not clients_weight:
            return

        if hdbscan is None:
            raise ImportError("hdbscan is required for FlameDefense")

        # ----- 1. clustering -----
        weight_vectors = [self._flatten_state(s) for s in clients_weight]
        clients_weight_total = torch.stack(weight_vectors).double()
        num_clients = clients_weight_total.shape[0]

        cluster = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=num_clients // 2 + 1,
            min_samples=1,
            allow_single_cluster=True,
        )
        cluster.fit(clients_weight_total)

        # ----- 2. median norm clipping -----
        euclidean = torch.norm(clients_weight_total, p=2, dim=1)
        med = torch.median(euclidean)
        for i, data in enumerate(clients_weight):
            gamma = med / euclidean[i]
            gamma = torch.clamp(gamma, max=1.0)
            for name, params in data.items():
                params.data = (params.data * gamma).to(params.data.dtype)

        # ----- 3. aggregation -----
        weight_accumulator: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, dtype=torch.double)
            for name, param in self.global_model.state_dict().items()
        }

        num_in = 0
        for i, data in enumerate(clients_weight):
            if self.conf["defense"] == "flame" and cluster.labels_[i] == 0:
                num_in += 1
                for name, params in data.items():
                    weight_accumulator[name].add_(params.double())

        self.model_aggregate(weight_accumulator, max(num_in, 1))

        # ----- 4. noise addition -----
        if self.conf["defense"] == "flame":
            lamda = 0.000012
            for name, param in self.global_model.named_parameters():
                if "bias" in name or "bn" in name:
                    continue
                std = lamda * med * param.data.std()
                noise = torch.normal(0, std, size=param.size(), device=param.device)
                param.data.add_(noise)

    def model_aggregate(self, weight_accumulator: Dict[str, torch.Tensor], num: int) -> None:
        """Aggregate ``weight_accumulator`` into ``self.global_model``."""
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] / float(num)
            if data.dtype != update_per_layer.dtype:
                data.add_(update_per_layer.to(data.dtype))
            else:
                data.add_(update_per_layer)
