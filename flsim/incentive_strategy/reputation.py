from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
def sigmoid(x: float) -> float: return 1.0/(1.0+np.exp(-x))
def softmax(x): x=np.asarray(x,float); x=x-np.max(x); e=np.exp(x); s=e.sum(); return e/(s if s>0 else 1.0)
def jain_fairness(vals: Sequence[float]) -> float:
    vals=np.asarray(list(vals),float); s=vals.sum(); 
    if s<=0: return 0.0
    s2=float((vals*vals).sum())+1e-8; n=float(len(vals)); jf=(s*s)/(n*s2); mean=s/n; return float(jf*(1.0/(1.0+np.exp(-mean/10.0))))
@dataclass
class ReputationCfg:
    committee_size:int=10; rep_exponent:float=1.0; hist_decay_factor:float=0.9; base_reward:float=100.0; stake_weight:float=0.4
class ReputationIncentives:
    def __init__(self, nodes: List[object], cfg: ReputationCfg):
        self.nodes=nodes; self.cfg=cfg; self.committee_history=[]; self.current_round=0; self.X_c=1.0; self.X_s=1.0
    def select_committee(self, num_strata: int = 3):
        import numpy as _np
        sorted_nodes=sorted(self.nodes, key=lambda x: x.reputation, reverse=True)
        strata_size=max(1, len(sorted_nodes)//max(1,num_strata))
        strata=[sorted_nodes[i*strata_size:(i+1)*strata_size] for i in range(num_strata)]
        if len(sorted_nodes)%num_strata!=0: strata[-1].extend(sorted_nodes[num_strata*strata_size:])
        selected=[]; quotas=[self.cfg.committee_size//num_strata]*num_strata; rem=self.cfg.committee_size-sum(quotas)
        for i in range(rem): quotas[i%num_strata]+=1
        for stratum, quota in zip(strata, quotas):
            cand=[n for n in stratum if getattr(n,'cooldown',0)<=0]
            if not cand or quota<=0: continue
            k=max(1, min(quota, len(cand)))
            if len(cand)==1: idxs=[0]
            else:
                rep=_np.array([float(getattr(n,'reputation',0.0)) for n in cand], float); p=_np.exp(rep**1.0); p/=p.sum()
                idxs=list(_np.random.choice(len(cand), size=k, replace=False, p=p))
            selected.extend([cand[i] for i in idxs])
        if len(selected)<self.cfg.committee_size:
            need=self.cfg.committee_size-len(selected); cand=[n for n in self.nodes if getattr(n,'cooldown',0)<=0 and n not in selected]
            if len(cand)>=need:
                if len(cand)==1: idxs=[0]
                else:
                    rep=_np.array([float(getattr(n,'reputation',0.0)) for n in cand], float); p=_np.exp(rep**1.0); p/=p.sum()
                    idxs=list(_np.random.choice(len(cand), size=need, replace=False, p=p))
                selected.extend([cand[i] for i in idxs])
            elif len(cand)>0: selected.extend(cand)
        for n in self.nodes:
            in_c=n in selected
            if hasattr(n,'update_committee_status'): n.update_committee_status(in_c)
            cd=getattr(n,'cooldown',0); n.cooldown=max(0,cd-1); 
            if in_c: n.cooldown=max(n.cooldown,1)
        self.committee_history.append([getattr(n,'id', getattr(n,'cfg',type('C',(),{})()).__dict__.get('node_id',-1)) for n in selected])
        return selected
    def calculate_reward(self, node, avg_rep: float) -> float:
        if not getattr(node,'contrib_history',None) or (node.contrib_history[-1]==0): return 0.0
        import numpy as _np
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
    def update_reputation(self, node, contribution: float) -> float:
        import numpy as _np
        age=max(0,int(getattr(node,'participation',0))); age_factor=1.0-1.0/(1.0+age/100.0); delta=0.88+0.07*age_factor
        contrib_quality=1.0/(1.0+_np.exp(- (float(contribution)-0.0)/max(1e-8, (10.0-0.0))))
        hist=getattr(node,'contrib_history', []); stability=1.0-(float(_np.std(hist[-5:]))/5.0) if len(hist)>=5 else 0.8
        rep_prev=float(getattr(node,'reputation',0.0)); new_rep=rep_prev*delta + contrib_quality*1.0 + stability*1.0
        cap=500.0 if self.current_round>50 else 300.0; new_rep=float(_np.clip(new_rep,0.0,cap)); node.reputation=new_rep; return new_rep
    def begin_round(self, r:int): self.current_round=r
    def end_round(self): pass
