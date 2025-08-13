from typing import Dict, List, Any
import torch
from .base import AggregationStrategy, _weighted_average
class Strategy(AggregationStrategy):
    name='fedavg'
    def aggregate(self, states: List[Dict[str, torch.Tensor]], weights: List[float], *, base_sd, meta: Any):
        return _weighted_average(states, weights)
