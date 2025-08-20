from typing import Dict, List, Any
import torch

from .base import AggregationStrategy

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None


class Strategy(AggregationStrategy):
    """FLAME aggregation strategy with clipping, clustering and noise."""

    name = "flame"

    def aggregate(
        self,
        states: List[Dict[str, torch.Tensor]],
        weights: List[float],
        *,
        base_sd: Dict[str, torch.Tensor],
        meta: Any,
    ) -> Dict[str, torch.Tensor]:
        if not states:
            return base_sd
        if hdbscan is None:
            raise ImportError("hdbscan is required for FLAME aggregation")

        # ----- 1. clustering -----
        vecs = [torch.cat([p.reshape(-1).to(torch.float64) for p in sd.values()]) for sd in states]
        matrix = torch.stack(vecs)
        num_clients = matrix.shape[0]
        cluster = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=num_clients // 2 + 1,
            min_samples=1,
            allow_single_cluster=True,
        )
        cluster.fit(matrix)

        # ----- 2. median norm clipping -----
        euclidean = torch.norm(matrix, p=2, dim=1)
        med = torch.median(euclidean)
        clipped_states: List[Dict[str, torch.Tensor]] = []
        for i, state in enumerate(states):
            gamma = torch.clamp(med / euclidean[i], max=1.0)
            clipped = {name: (param * gamma).to(param.dtype) for name, param in state.items()}
            clipped_states.append(clipped)

        # ----- 3. aggregation -----
        weight_acc = {
            name: torch.zeros_like(param, dtype=torch.double)
            for name, param in base_sd.items()
        }
        total_weight = 0.0
        for i, state in enumerate(clipped_states):
            if cluster.labels_[i] == 0:
                w = float(weights[i]) if i < len(weights) else 1.0
                total_weight += w
                for name, param in state.items():
                    weight_acc[name].add_(param.to(torch.double) * w)

        if total_weight == 0.0:
            agg_sd = {name: base_sd[name].clone() for name in base_sd.keys()}
        else:
            agg_sd = {
                name: (weight_acc[name] / total_weight).to(base_sd[name].dtype)
                for name in base_sd.keys()
            }

        # ----- 4. noise addition -----
        lamda = 0.000012
        for name, param in agg_sd.items():
            if "bias" in name or "bn" in name:
                continue
            std = lamda * med * param.std()
            if float(std) > 0.0:
                noise = torch.normal(0, std, size=param.size(), device=param.device)
                agg_sd[name] = param + noise.to(param.dtype)

        meta["cluster_labels"] = cluster.labels_.tolist()
        return agg_sd
