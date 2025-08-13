from typing import Dict, List, Any
import torch
from .base import AggregationStrategy
class Strategy(AggregationStrategy):
    name='trimmed_mean'
    def aggregate(self, states: List[Dict[str, torch.Tensor]], weights: List[float], *, base_sd, meta: Any):
        r=float(meta.get('trim_ratio',0.1)); r=max(0.0, min(0.49, r)); out={}; N=len(states)
        for k in states[0].keys():
            stack=torch.stack([sd[k].flatten().to(torch.float64) for sd in states], dim=0)
            low=int(N*r); high=N-int(N*r); sorted_vals,_=torch.sort(stack, dim=0); kept=sorted_vals[low:high,:] if high>low else sorted_vals
            mean=kept.mean(dim=0); out[k]=mean.reshape_as(states[0][k]).to(states[0][k].dtype)
        return out
