from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Optional, Sequence, Dict
import numpy as np
from .reputation import ReputationIncentives, ReputationCfg, jain_fairness, sigmoid
def gini(x: Sequence[float]) -> float:
    arr=np.asarray(x,float); 
    if arr.size==0: return 0.0
    arr=np.clip(arr,0.0,None); 
    if np.allclose(arr,0): return 0.0
    arr_sorted=np.sort(arr); n=arr.size; cum=np.cumsum(arr_sorted)
    return float((n+1-2*np.sum(cum)/cum[-1])/n)
@dataclass
class RoundSimCfg:
    committee_size:int=10; rep_exponent:float=1.0; hist_decay_factor:float=0.9; base_reward:float=100.0; stake_weight:float=0.4
    gamma:float=0.1; X_c:float=1.0; C_min:float=0.0; C_max:float=10.0; stake_penalty_factor:float=0.05; rep_penalty_factor:float=0.10
class RoundRunner:
    def __init__(self, nodes: List[object], malicious_nodes: Optional[List[object]], cfg: RoundSimCfg, *, detect_malicious: Optional[Callable[[object], bool]]=None):
        self.nodes=nodes; self.malicious_nodes=set(malicious_nodes or []); self.cfg=cfg; self.current_round=0
        rep_cfg=ReputationCfg(committee_size=cfg.committee_size, rep_exponent=cfg.rep_exponent, hist_decay_factor=cfg.hist_decay_factor, base_reward=cfg.base_reward, stake_weight=cfg.stake_weight)
        self.incent=ReputationIncentives(nodes, rep_cfg)
        self.committee=[]; self.committee_history=[]; self.penalty_history=[]; self.reputation_history=[]; self.reward_history=[]; self.gini_history=[]; self.fairness_history=[]; self.detection_history=[]; self.utility_history=[]; self.profit_history=[]; self.profit_per_node_per_round=[]
        self._detect_malicious_fn=detect_malicious
    def detect_malicious(self, node) -> bool:
        if self._detect_malicious_fn is None: return False
        try: return bool(self._detect_malicious_fn(node))
        except Exception: return False
    def run_round(self, contrib_scores: Dict[int,float]):
        self.incent.begin_round(self.current_round); self.committee=self.incent.select_committee(num_strata=3)
        self.committee_history.append([getattr(n,'id', getattr(n,'cfg',type('C',(),{})()).__dict__.get('node_id',-1)) for n in self.committee])
        avg_rep=float(np.mean([getattr(n,'reputation',0.0) for n in self.nodes])) if self.nodes else 0.0
        rewards=[]; penalties_round=[]; detected=0; detected_ids=[]; detected_map={}
        for node in self.nodes:
            node_id=getattr(node,'id', getattr(node,'cfg',type('C',(),{})()).__dict__.get('node_id',-1))
            contrib=float(contrib_scores.get(node_id,0.0))
            if not hasattr(node,'contrib_history'): node.contrib_history=[]
            node.contrib_history.append(contrib)
            is_mal=self.detect_malicious(node) or (node in self.malicious_nodes)
            if is_mal:
                delta_stake=float(self.cfg.stake_penalty_factor)*float(getattr(node,'stake',0.0))
                delta_rep=min(float(self.cfg.rep_penalty_factor)*float(getattr(node,'reputation',0.0)), float(getattr(node,'reputation',0.0))/2.0)
                node.stake=max(float(getattr(node,'stake',0.0))-delta_stake,0.0); node.reputation=max(float(getattr(node,'reputation',0.0))-delta_rep,0.0)
                node.violations=int(getattr(node,'violations',0))+1; penalties_round.append({'stake':delta_stake,'reputation':delta_rep}); detected+=1; detected_ids.append(node_id); detected_map[node_id]=True
            else:
                _=self.incent.update_reputation(node, contrib); detected_map[node_id]=detected_map.get(node_id, False)
            reward=self.incent.calculate_reward(node, avg_rep); node.total_reward=float(getattr(node,'total_reward',0.0))+reward; rewards.append(reward); node.participation=int(getattr(node,'participation',0))+1
        self.penalty_history.append(penalties_round); self.reputation_history.append([float(getattr(n,'reputation',0.0)) for n in self.nodes]); self.reward_history.append(rewards)
        self.gini_history.append(gini(rewards)); self.fairness_history.append(jain_fairness(rewards)); self.detection_history.append(detected)
        round_utils=[]; profits_this_round=[]
        for node,reward in zip(self.nodes,rewards):
            node_id=getattr(node,'id', getattr(node,'cfg',type('C',(),{})()).__dict__.get('node_id',-1)); contrib=float(contrib_scores.get(node_id,0.0)); tau=1.0
            compliance=float(np.exp(-int(getattr(node,'violations',0)))); cost=0.5*float(self.cfg.gamma)*(contrib**2)
            util = reward*compliance - float(self.cfg.stake_weight)*float(getattr(node,'stake',0.0))*(1.0-compliance) - cost; round_utils.append(float(util))
            norm=(contrib - float(self.cfg.C_min))/max(1e-8, float(self.cfg.C_max)-float(self.cfg.C_min)); q=sigmoid(norm); V=float(self.cfg.X_c)/tau*q
            profit = (V - reward)*compliance + float(self.cfg.stake_weight)*float(getattr(node,'stake',0.0))*(1.0-compliance); profits_this_round.append(float(profit))
        self.utility_history.append(round_utils); total_profit=float(sum(profits_this_round)); self.profit_per_node_per_round.append(profits_this_round); self.profit_history.append(total_profit)
        self.incent.end_round(); self.current_round+=1
        return {'detected_ids': detected_ids, 'detected_map': detected_map, 'rewards': rewards}
