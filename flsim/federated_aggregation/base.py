from typing import Dict, List, Any
import torch
class AggregationStrategy: 
    name='base'
    def aggregate(self, states: List[Dict[str, torch.Tensor]], weights: List[float], *, base_sd: Dict[str, torch.Tensor], meta: Any): raise NotImplementedError
def _zeros_like(sd: Dict[str, torch.Tensor], dtype=torch.float64): return {k: torch.zeros_like(v, dtype=dtype) for k,v in sd.items()}
def _weighted_average(states, weights):
    wsum=float(sum(weights)); agg=_zeros_like(states[0], dtype=torch.float64)
    for sd,w in zip(states,weights):
        wf=float(w)/wsum if wsum>0 else 1.0/max(1,len(states))
        for k in agg: agg[k]+=sd[k].to(torch.float64)*wf
    return {k:v.to(states[0][k].dtype) for k,v in agg.items()}
