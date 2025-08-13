from .eval import evaluate_state_dict
import os
import math
from typing import Dict, List

import numpy as np
import torch



_AGG_DATASET_DEFAULT='mnist'
_AGG_MODEL_DEFAULT=None
class Aggregator:
    """
    FedAvg aggregator + contribution scoring (align/acc/loss/norm with robust min-max).

    Contribution for round r is computed *after* aggregation using:
      score = 0.4 * align + 0.3 * acc + 0.2 * (1 - norm(loss)) + 0.1 * norm(update_norm)
    where each component is normalized within the round using an IQR-based min-max.
    - align: cosine similarity between each client's delta and aggregation direction
             (new_global - prev_global), scaled from [-1,1] -> [0,1], then robust min-max.
    - acc:   client's reported train accuracy for the round, robust min-max.
    - loss:  lower is better, so we robust-minmax(loss) and then take (1 - loss_norm).
    - norm:  L2 of the client's realized delta (realized - base), robust min-max.

    Rewards are then computed based on your previously specified economic logic.
    """

    def __init__(
        self,
        ipfs,
        contract,
        nodes,
        save_dir: str,
        *,
        reward_rate: float,
        penalize_negative: bool,
        clip_min: float,
        clip_max: float,
        base_reward: float,
        stake_weight: float,
        committee_size: int,
        hist_decay_factor: float,
    ) -> None:
        self.ipfs = ipfs
        self.contract = contract
        self.nodes = nodes
        self.save_dir = save_dir

        self.reward_rate = reward_rate
        self.penalize_negative = penalize_negative
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.base_reward = base_reward
        self.stake_weight = stake_weight
        self.committee_size = committee_size
        self.hist_decay_factor = hist_decay_factor

        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "updates"), exist_ok=True)

        # Contribution weights
        self.W_ALIGN, self.W_ACC, self.W_LOSS, self.W_NORM = 0.4, 0.3, 0.2, 0.1

    # ----------------- helpers -----------------
        # fallback defaults for dataset/model names
    dataset_name = _AGG_DATASET_DEFAULT
    model_name = _AGG_MODEL_DEFAULT

    def _apply_update(self, base, update, update_type):
        if update_type == "delta":
            return {k: base[k] + update[k] for k in base.keys()}
        return update

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _jain_fairness(vals: List[float]) -> float:
        if not vals:
            return 0.0
        s = float(sum(vals))
        if s == 0:
            return 0.0
        s2 = float(sum(v * v for v in vals)) + 1e-8
        n = float(len(vals))
        jf = (s * s) / (n * s2)
        mean = s / n
        # light coupling to mean reputation to avoid degenerate distributions
        return jf * (1.0 / (1.0 + math.exp(-mean / 10.0)))

    def _calc_committee(self) -> List[int]:
        N = len(self.nodes)
        K = min(self.committee_size, N)
        sel = sorted(self.nodes, key=lambda n: getattr(n, "reputation", 10.0), reverse=True)[:K]
        return [n.cfg.node_id for n in sel]

    def _calculate_reward(self, node, avg_rep: float, in_committee: bool) -> float:
        stakes = [getattr(n, "stake", 100.0) for n in self.nodes]
        avg_stake = sum(stakes) / max(1, len(stakes))
        effective_stake = min(getattr(node, "stake", 100.0), 3.0 * avg_stake)

        hist = getattr(node, "contrib_history", [])[-5:]
        hist_contrib = 0.0
        for t, c in enumerate(reversed(hist)):
            hist_contrib += float(c) * (self.hist_decay_factor ** t)

        reputations = [getattr(n, "reputation", 10.0) for n in self.nodes]
        diversity_bonus = self._jain_fairness(reputations)

        node_rep = getattr(node, "reputation", 10.0)
        alpha = self._sigmoid((avg_rep - node_rep) / 50.0) * self.stake_weight
        beta = 1.0 - alpha
        committee_bonus = 20.0 * diversity_bonus if in_committee else 0.0

        total_stake = sum(stakes) + 1e-8
        contribs_curr = [(n.contrib_history[-1] if n.contrib_history else 0.0) for n in self.nodes]
        total_contrib = sum(contribs_curr) + 1e-8

        reward = (
            (alpha * self.base_reward * (effective_stake / total_stake)
             + beta * self.base_reward * (hist_contrib / total_contrib))
            * diversity_bonus
            + committee_bonus
        )
        reward_clipped = max(self.clip_min, min(self.clip_max, reward))
        return self.reward_rate * (reward_clipped if self.penalize_negative else max(0.0, reward_clipped))

    @staticmethod
    def _flatten_sd(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
        vecs = []
        for v in sd.values():
            if torch.is_tensor(v):
                vecs.append(v.detach().reshape(-1).to(torch.float32))
        if not vecs:
            return torch.zeros(1)
        return torch.cat(vecs, dim=0)

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() == 0 or b.numel() == 0:
            return float("nan")
        na = torch.linalg.norm(a)
        nb = torch.linalg.norm(b)
        if float(na) == 0.0 or float(nb) == 0.0:
            return float("nan")
        return float((a @ b) / (na * nb))

    @staticmethod
    def _robust_minmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(float)
        if x.size == 0:
            return x
        q1, q3 = np.nanpercentile(x, [25, 75])
        if not np.isfinite(q1) or not np.isfinite(q3) or q3 <= q1:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            if not np.isfinite(xmin) or xmax <= xmin:
                return np.zeros_like(x)
            return np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0)
        return np.clip((x - q1) / (q3 - q1), 0.0, 1.0)

    # ----------------- core -----------------
    def fedavg_weighted(self, states, weights):
        wsum = float(sum(weights))
        agg = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in states[0].items()}
        for sd, w in zip(states, weights):
            wf = float(w) / wsum if wsum > 0 else 1.0 / max(1, len(states))
            for k in agg:
                agg[k] += sd[k].to(torch.float64) * wf
        return {k: v.to(states[0][k].dtype) for k, v in agg.items()}

    def aggregate_round(self, r: int, base_cid: str):
        subs = self.contract.get_round_submissions(r)
        base_sd = self.ipfs.load(base_cid)

        realized_states: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        metrics_map: Dict[int, Dict] = {}

        # For contribution scoring features
        tmp_nodes, tmp_norms, tmp_aligns, tmp_accs, tmp_losses = [], [], [], [], []

        # 1) First pass: realize client states and collect raw features
        for nid, (mcid, mtcid, updtype) in subs.items():
            upd = self.ipfs.load(mcid)
            mt = self.ipfs.load(mtcid)
            realized = self._apply_update(base_sd, upd, updtype)
            realized_states.append(realized)
            weights.append(mt.get("samples", 1))

            metrics_map[nid] = {
                **mt,
                "update_cid": mcid,
                "metrics_cid": mtcid,
                "update_type": updtype,
            }

            # delta = realized - base (vectorize for norm & later alignment)
            delta = {k: realized[k] - base_sd[k] for k in base_sd.keys()}
            delta_vec = self._flatten_sd(delta)

            tmp_nodes.append(nid)
            tmp_norms.append(float(torch.linalg.norm(delta_vec)))
            tmp_accs.append(float(mt.get("acc", float("nan"))))
            tmp_losses.append(float(mt.get("loss", float("nan"))))
            tmp_aligns.append(float("nan"))  # fill after agg direction known

        # 2) FedAvg aggregate -> new global
        agg_sd = self.fedavg_weighted(realized_states, weights)
        new_cid = self.ipfs.save(agg_sd)
        torch.save(agg_sd, os.path.join(self.save_dir, "models", f"global_round_{r}.pt"))
        self.contract.set_global_model(r + 1, new_cid)

        # 3) Aggregation direction for this round
        agg_delta = {k: agg_sd[k] - base_sd[k] for k in base_sd.keys()}
        agg_vec = self._flatten_sd(agg_delta)

        
        # 4.5) Evaluate realized local models on validation set to verify claimed accuracy
        eval_accs = []
        for realized in realized_states:
            try:
                ev = float(evaluate_state_dict(realized))
            except Exception:
                ev = float("nan")
            eval_accs.append(ev)
# 4) Fill alignment now that we know agg_vec
        for i in range(len(tmp_nodes)):
            realized = realized_states[i]
            delta = {k: realized[k] - base_sd[k] for k in base_sd.keys()}
            delta_vec = self._flatten_sd(delta)
            tmp_aligns[i] = self._cosine(delta_vec, agg_vec)

        # 5) Robust normalization within round and compute contribution score
        align_01 = np.array([(a + 1.0) / 2.0 for a in tmp_aligns], dtype=float)  # [-1,1] -> [0,1]
        norm_n = self._robust_minmax(np.array(tmp_norms))
        align_n = self._robust_minmax(align_01)
        acc_n = self._robust_minmax(np.array(tmp_accs))
        loss_01 = self._robust_minmax(np.array(tmp_losses))
        loss_good = 1.0 - loss_01
        scores = self.W_ALIGN * align_n + self.W_ACC * acc_n + self.W_LOSS * loss_good + self.W_NORM * norm_n

        # 6) Commit contribution & compute rewards
        contrib_map: Dict[int, float] = {}
        reward_map: Dict[int, float] = {}
        node_by_id = {n.cfg.node_id: n for n in self.nodes}
        avg_rep = sum(getattr(n, "reputation", 10.0) for n in self.nodes) / max(1, len(self.nodes))
        committee_ids = set(self._calc_committee())

        for i, nid in enumerate(tmp_nodes):
            score = float(scores[i]) if i < len(scores) else float("nan")
            self.contract.set_contribution(r, nid, score)
            contrib_map[nid] = score

            claimed = float(metrics_map.get(nid, {}).get('acc', float('nan')))
            evalacc = float(eval_accs[i]) if i < len(eval_accs) else float('nan')
            self.contract.set_features(r, nid, claimed_acc=claimed, eval_acc=evalacc, align=float(tmp_aligns[i]) if i < len(tmp_aligns) else float('nan'), update_norm=float(tmp_norms[i]) if i < len(tmp_norms) else float('nan'), loss=float(tmp_losses[i]) if i < len(tmp_losses) else float('nan'), acc=float(tmp_accs[i]) if i < len(tmp_accs) else float('nan'))

            node = node_by_id.get(nid)
            rew = self._calculate_reward(node, avg_rep, nid in committee_ids) if node is not None else 0.0
            self.contract.add_reward(r, nid, rew)
            reward_map[nid] = rew

        return new_cid, metrics_map, contrib_map, reward_map
