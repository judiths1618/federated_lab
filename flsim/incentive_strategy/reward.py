"""Reward calculation utilities."""
import numpy as _np
# def calculate_reward(contribution: float, base_reward: float = 1.0) -> float:
#     """Simple proportional reward calculator."""

def calculate_reward(self, node, avg_rep: float) -> float:
    if not getattr(node,'contrib_history',None) or (node.contrib_history[-1]==0): return 0.0

    avg_stake=float(_np.mean([getattr(n,'stake',0.0) for n in self.nodes])) if self.nodes else 0.0
    effective_stake=min(getattr(node,'stake',0.0), 3.0*avg_stake)
    recent=getattr(node,'contrib_history', [])[-5:]; hist=0.0
    for t,c in enumerate(reversed(recent)): hist+=float(c)*(self.cfg.hist_decay_factor**t)
    reputations=[float(getattr(n,'reputation',0.0)) for n in self.nodes]; diversity=jain_fairness(reputations)
    node_rep=float(getattr(node,'reputation',0.0)); alpha=1.0/(1.0+_np.exp(-(avg_rep-node_rep)/50.0))*float(getattr(self.cfg,'stake_weight',0.4)); beta=1.0-alpha
    in_committee=False
    if self.committee_history:
        last=self.committee_history[-1]; node_id=getattr(node,'id', getattr(node,'cfg',type('C',(),{})()).__dict__.get('node_id',-1)); in_committee = node_id in last
    committee_bonus=20.0*diversity if in_committee else 0.0
    total_stake=sum(float(getattr(n,'stake',0.0)) for n in self.nodes)+1e-8
    total_contrib=sum(float(getattr(n,'contrib_history',[0.0])[-1]) for n in self.nodes)+1e-8
    reward=((alpha*self.cfg.base_reward*(effective_stake/total_stake)+beta*self.cfg.base_reward*(hist/total_contrib))*diversity+committee_bonus)
    return float(max(reward,0.0))
# return contribution * base_reward

__all__ = ["calculate_reward"]
