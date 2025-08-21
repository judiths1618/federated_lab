from typing import Dict, List, Any, Tuple
import torch
import numpy as np
from .base import AggregationStrategy, _weighted_average

class Strategy(AggregationStrategy):
    """FLAME aggregation with simple k-means clustering for anomaly detection."""
    name = "flame"

    @staticmethod
    def _flatten_delta(sd: Dict[str, torch.Tensor]) -> np.ndarray:
        vecs = []
        for v in sd.values():
            if torch.is_tensor(v):
                vecs.append(v.detach().reshape(-1).cpu().numpy())
        if not vecs:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(vecs).astype(np.float32)

    @staticmethod
    def _kmeans2(x: np.ndarray, iters: int = 10) -> np.ndarray:
        if len(x) <= 1:
            return np.zeros(len(x), dtype=int)
        rng = np.random.default_rng(0)
        centers = x[rng.choice(len(x), 2, replace=False)]
        labels = np.zeros(len(x), dtype=int)
        for _ in range(iters):
            dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            for j in range(2):
                pts = x[labels == j]
                if len(pts) > 0:
                    centers[j] = pts.mean(axis=0)
        return labels

    def aggregate(self, states: List[Dict[str, torch.Tensor]], weights: List[float], *, base_sd: Dict[str, torch.Tensor], meta: Any) -> Tuple[Dict[str, torch.Tensor], Dict[int, int]]:
        node_ids = meta.get("node_ids", list(range(len(states))))
        deltas = []
        for sd in states:
            delta = {k: sd[k] - base_sd[k] for k in base_sd.keys()}
            deltas.append(self._flatten_delta(delta))
        x = np.vstack(deltas) if deltas else np.zeros((0, 1), dtype=np.float32)
        labels = self._kmeans2(x)
        counts = np.bincount(labels, minlength=2)
        benign_label = int(np.argmax(counts))
        cluster_map = {nid: int(lbl != benign_label) for nid, lbl in zip(node_ids, labels)}
        benign_states = [sd for sd, lbl in zip(states, labels) if lbl == benign_label]
        benign_weights = [w for w, lbl in zip(weights, labels) if lbl == benign_label]
        agg = _weighted_average(benign_states if benign_states else states,
                                benign_weights if benign_states else weights)
        return agg, cluster_map
