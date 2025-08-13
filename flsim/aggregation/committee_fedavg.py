from typing import Dict, List, Any
import torch
from .base import AggregationStrategy, _weighted_average
class Strategy(AggregationStrategy):
    name='committee_fedavg'
    def aggregate(self, states: List[Dict[str, torch.Tensor]], weights: List[float], *, base_sd, meta: Any):
        com=set(meta.get('committee_ids', set())); node_ids=meta.get('node_ids', list(range(len(states))))
        ps,pw=[],[]
        for sd,w,nid in zip(states, weights, node_ids):
            if nid in com: ps.append(sd); pw.append(w)
        if not ps: ps,pw=states,weights
        return _weighted_average(ps,pw)
