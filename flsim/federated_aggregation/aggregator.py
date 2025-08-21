import importlib
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..incentive_strategy import ReputationIncentives, ReputationCfg
from ..evaluation import reconstruct_state

class Aggregator:
    """
    set committee + flame + FedAvg aggregator + contribution scoring (align/acc/loss/norm with robust min-max).

    1. calculate the contribution for each node based on the alignment, accuracy, loss, and norm of their updates.
    2. set committee based on the sampling strategy (e.g., reputation-based).
    3. malicious detection using FLAME.
    4. apply the reward and penalty logic based on the contributions, committee status, and behaviours for begnin and malicious nodes.
    5. aggregate the updates using FLAME
    6. save the aggregated model and update the global state.
    
    Contribution scoring:
    Contribution for round r is computed *after* aggregation using:
      score = 0.4 * align + 0.3 * acc + 0.2 * (1 - norm(loss)) + 0.1 * norm(update_norm)
    where each component is normalized within the round using an IQR-based min-max.
    - align: cosine similarity between each client's delta and aggregation direction
             (new_global - prev_global), scaled from [-1,1] -> [0,1], then robust min-max.
    - acc:   client's reported train accuracy for the round, robust min-max. claimed by the client.
    # - loss:  lower is better, so we robust-minmax(loss) and then take (1 - loss_norm).
    - norm:  L2 of the client's realized delta (realized - base), robust min-max.

    Rewards, penalties, and reputations are then computed based on your previously specified economic logic.
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
        base_reward: float,
        stake_weight: float,
        committee_size: int,
        hist_decay_factor: float,
        strategy_name: str = "fedavg",
        dataset_name: str = "mnist",
        model_name: Optional[str] = None,
        eval_loaders: Optional[List[DataLoader]] = None,
    ) -> None:
        self.ipfs = ipfs
        self.contract = contract
        self.nodes = nodes
        self.save_dir = save_dir

        self.reward_rate = reward_rate
        self.penalize_negative = penalize_negative
        self.base_reward = base_reward
        self.stake_weight = stake_weight
        self.committee_size = committee_size
        self.hist_decay_factor = hist_decay_factor
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.eval_loaders = eval_loaders

        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "updates"), exist_ok=True)

        self.strategy_name = strategy_name
        mod = importlib.import_module(f"flsim.federated_aggregation.{strategy_name}")
        self.strategy = getattr(mod, "Strategy")()

        # reputation-based incentives helper
        cfg = ReputationCfg(
            committee_size=committee_size,
            hist_decay_factor=hist_decay_factor,
            base_reward=base_reward,
            stake_weight=stake_weight,
        )
        self.incent = ReputationIncentives(self.nodes, cfg)

        # Contribution weights
        # self.W_ALIGN, self.W_ACC, self.W_LOSS, self.W_NORM = 0.4, 0.3, 0.2, 0.1
        self.W_ALIGN, self.W_ACC, self.W_NORM = 0.4, 0.3, 0.3

    # ----------------- helpers -----------------
    def _apply_update(self, base, update, update_type):
        if update_type == "delta":
            return {k: base[k] + update[k] for k in base.keys()}
        return update

    # @staticmethod
    # def _sigmoid(x: float) -> float:
    #     return 1.0 / (1.0 + math.exp(-x))

    # @staticmethod
    # def _jain_fairness(vals: List[float]) -> float:
    #     if not vals:
    #         return 0.0
    #     s = float(sum(vals))
    #     if s == 0:
    #         return 0.0
    #     s2 = float(sum(v * v for v in vals)) + 1e-8
    #     n = float(len(vals))
    #     jf = (s * s) / (n * s2)
    #     mean = s / n
    #     # light coupling to mean reputation to avoid degenerate distributions
    #     return jf * (1.0 / (1.0 + math.exp(-mean / 10.0)))

    def _calc_committee(self) -> List[int]:
        """Fallback committee selection based on node reputations.

        When the contract layer cannot provide a committee, we select the top
        ``committee_size`` nodes ranked by their current reputation.  Each node
        object is expected to expose ``cfg.node_id`` and an optional
        ``reputation`` attribute.  Nodes without an explicit reputation default
        to ``10.0`` which mirrors the behaviour in the rest of the system.

        Returns a list of node ids chosen for the committee.
        """

        N = len(self.nodes)
        K = min(self.committee_size, N)
        # Sort nodes by reputation (descending) and take the top-K.
        selected = sorted(
            self.nodes, key=lambda n: getattr(n, "reputation", 10.0), reverse=True
        )[:K]
        return [n.cfg.node_id for n in selected]

    # def _calculate_reward(self, node, avg_rep: float, in_committee: bool) -> float:
    #     """Compute the reward for ``node`` based on stake, history and committee status."""

    #     stakes = [getattr(n, "stake", 10.0) for n in self.nodes]
    #     avg_stake = sum(stakes) / max(1, len(stakes))
    #     effective_stake = min(getattr(node, "stake", 10.0), 3.0 * avg_stake)

    #     hist_contrib = self._historical_contribution(node)
    #     print(f"Node {node.cfg.node_id} history contribution: {hist_contrib:.4f}")

    #     reputations = [getattr(n, "reputation", 10.0) for n in self.nodes]
    #     diversity_bonus = self._jain_fairness(reputations)

    #     node_rep = getattr(node, "reputation", 10.0)
    #     alpha = self._sigmoid((avg_rep - node_rep) / 50.0) * self.stake_weight
    #     beta = 1.0 - alpha
    #     committee_bonus = 20.0 * diversity_bonus if in_committee else 0.0

    #     total_contrib = sum((n.contrib_history[-1] if getattr(n, "contrib_history", []) else 0.0) for n in self.nodes)
    #     total_contrib = total_contrib if abs(total_contrib) > 1e-8 else 1.0
    #     total_stake = sum(stakes) or 1.0

    #     reward = (
    #         (alpha * self.base_reward * (effective_stake / total_stake)
    #          + beta * self.base_reward * (hist_contrib / total_contrib))
    #         * diversity_bonus
    #         + committee_bonus
    #     )
    #     print(f"Node {node.cfg.node_id} reward before clipping: {reward:.4f}")
    #     reward *= self.reward_rate
    #     if self.penalize_negative:
    #         reward = max(0.0, reward)
    #     return reward

    # def _historical_contribution(self, node, window: int = 5) -> float:
    #     """Return decayed historical contribution for ``node`` over last ``window`` rounds."""
    #     hist = getattr(node, "contrib_history", []) or []
    #     recent = hist[-window:]
    #     return sum(float(c) * (self.hist_decay_factor ** t) for t, c in enumerate(reversed(recent)))

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
        """Compute cosine similarity between two flattened tensors."""

        if a.numel() == 0 or b.numel() == 0:
            return float("nan")
        na = torch.linalg.norm(a)
        nb = torch.linalg.norm(b)
        if float(na) == 0.0 or float(nb) == 0.0:
            return float("nan")
        return float((a @ b) / (na * nb))

    @staticmethod
    def _robust_minmax(x: np.ndarray) -> np.ndarray:
        """Robust min-max normalisation using the IQR.

        Values are scaled to the ``[0, 1]`` range using the interquartile range
        (25th to 75th percentile).  If the IQR is degenerate we fall back to
        classical min-max scaling.  Non-finite results yield a zero array.
        """

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

    @staticmethod
    def fedavg_weighted(
        states: List[Dict[str, torch.Tensor]], weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Standard weighted FedAvg aggregation.

        ``states`` is a list of model state dictionaries. ``weights`` contains
        the corresponding sample counts (or other weighting factors).  The
        method returns the weighted average state dictionary.
        """

        if not states:
            return {}
        total = float(sum(weights)) or 1.0
        agg = {k: torch.zeros_like(v) for k, v in states[0].items()}
        for sd, w in zip(states, weights):
            scale = float(w) / total
            for k, v in sd.items():
                agg[k] += v * scale
        return agg

    def aggregate_round(self, r: int, base_cid: str):
        """Aggregate updates for round ``r`` and settle incentives."""
        self.incent.begin_round(r)

        subs = self.contract.get_round_submissions(r)
        base_sd = self.ipfs.load(base_cid)

        realized_states: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        metrics_map: Dict[int, Dict] = {}

        node_by_id = {n.cfg.node_id: n for n in self.nodes}

        tmp_nodes: List[int] = []
        tmp_norms: List[float] = []
        tmp_aligns: List[float] = []
        tmp_accs: List[float] = []
        # tmp_losses: List[float] = []

        # 1) reconstruct local models & collect metrics
        for nid, (mcid, mtcid, updtype) in subs.items():
            upd = self.ipfs.load(mcid)          # 可能是 delta 或完整 state
            mt  = self.ipfs.load(mtcid)
            updtype = (updtype or "delta").lower()

            # --- 重构全量本地模型 ---
            realized = reconstruct_state(base_sd, upd, updtype)
            realized_states.append(realized)

            # 训练样本权重
            samples = mt.get("samples", 1)
            weights.append(float(samples) if isinstance(samples, (int, float)) else 1.0)

            metrics_map[nid] = {
                **mt,
                "update_cid": mcid,
                "metrics_cid": mtcid,
                "update_type": updtype,
            }

            # 计算 delta（用于范数/对齐）
            common = realized.keys() & base_sd.keys()
            delta = {k: realized[k] - base_sd[k] for k in common if torch.is_tensor(realized[k]) and torch.is_tensor(base_sd[k])}
            delta_vec = self._flatten_sd(delta)

            tmp_nodes.append(nid)
            tmp_norms.append(float(torch.linalg.norm(delta_vec)))
            tmp_accs.append(float(mt.get("acc", float("nan"))))
            # tmp_losses.append(float(mt.get("loss", float("nan"))))
            tmp_aligns.append(float("nan"))  # placeholder; filled after aggregation

        # ---------- 2) set committee ----------
        try:
            committee_ids = self.contract.set_committee(r, self.nodes)
        except TypeError:
            fallback = self._calc_committee()
            maybe = self.contract.set_committee(r, fallback)
            committee_ids = maybe if isinstance(maybe, list) else fallback
        print(f"[Round {r}] Committee IDs: {committee_ids}")

        # Ensure committee_ids is a set for fast membership checks
        if not isinstance(committee_ids, set):
            committee_ids = set(committee_ids)
        self.incent.committee_history.append(sorted(committee_ids))
        print(f"[Round {r}] Committee selected: {sorted(committee_ids)}")

        # ---------- 3) evaluate reconstructed models ----------
        # if realized_states:
        #     eval_accs = evaluate_many_state_dicts(
        #         realized_states,
        #         dataset=self.dataset_name,
        #         model_hint=self.model_name,
        #         max_workers=4,
        #         loaders=self.eval_loaders,
        #     )
        # else:
        #     agg_sd = base_sd
        #     new_cid = self.ipfs.save(agg_sd)
        #     self.contract.set_global_model(r + 1, new_cid)
        #     self.contract.settle_round(r)
        #     return new_cid, {}, {}, {}

        # ---------- 3) USING FLAME to detect poisoned model nodes, and defense the aggregation ----------
        meta = {"node_ids": tmp_nodes, "committee_ids": list(committee_ids)}
        if hasattr(self.strategy, "aggregate"):
            # Use the FLAME strategy to aggregate and detect malicious nodes
            agg_sd = self.strategy.aggregate(
                states=realized_states,
                weights=weights,
                base_sd=base_sd,
                meta=meta,
            )
        else:
            # Fallback to FedAvg if no custom strategy is defined
            agg_sd = self.fedavg_weighted(realized_states, weights)

        # ---------- 4) save aggregated model ----------
        new_cid = self.ipfs.save(agg_sd)
        torch.save(agg_sd, os.path.join(self.save_dir, "models", f"global_round_{r}.pt"))
        self.contract.set_global_model(r + 1, new_cid)

        # record cluster labels if provided by the FLAME strategy
        labels = meta.get("cluster_labels", [])
        flame_malicious: List[int] = []
        flame_benign: List[int] = []
        if labels and len(labels) == len(tmp_nodes):
            for nid, label in zip(tmp_nodes, labels):
                metrics_map[nid]["cluster_label"] = int(label)
                is_mal = int(label != 0)
                metrics_map[nid]["is_malicious"] = is_mal
                (flame_malicious if is_mal else flame_benign).append(nid)

            gt_malicious = [
                nid
                for nid in tmp_nodes
                if getattr(node_by_id[nid].behavior, "is_malicious", False)
            ]
            gt_set = set(gt_malicious)
            det_set = set(flame_malicious)
            tp = sorted(det_set & gt_set)
            fp = sorted(det_set - gt_set)
            fn = sorted(gt_set - det_set)
            print(
                f"[Round {r}] FLAME flagged malicious nodes: {sorted(flame_malicious)}"
            )
            print(
                f"[Round {r}] FLAME flagged benign nodes: {sorted(flame_benign)}"
            )
            print(
                f"[Round {r}] Ground-truth malicious nodes: {sorted(gt_malicious)}"
            )
            print(f"[Round {r}] FLAME analysis: TP={tp}, FP={fp}, FN={fn}")

        # ---------- 5) alignment ----------
        agg_delta = {k: agg_sd[k] - base_sd[k] for k in base_sd.keys()}
        agg_vec = self._flatten_sd(agg_delta)
        for i in range(len(tmp_nodes)):
            delta_i = {k: realized_states[i][k] - base_sd[k] for k in base_sd.keys()}
            delta_vec_i = self._flatten_sd(delta_i)
            tmp_aligns[i] = self._cosine(delta_vec_i, agg_vec)

        # ---------- 6) contribution scoring ----------
        align_01  = np.array([(a + 1.0) / 2.0 for a in tmp_aligns], dtype=float)
        norm_n    = self._robust_minmax(np.array(tmp_norms))
        align_n   = self._robust_minmax(align_01)
        acc_n     = self._robust_minmax(np.array(tmp_accs))
        # loss_01   = self._robust_minmax(np.array(tmp_losses))
        # loss_good = 1.0 - loss_01
        # scores = self.W_ALIGN * align_n + self.W_ACC * acc_n + self.W_LOSS * loss_good + self.W_NORM * norm_n
        
        scores = self.W_ALIGN * align_n + self.W_ACC * acc_n + self.W_NORM * norm_n

        # ---------- 7) commit & reward ----------
        contrib_map: Dict[int, float] = {}
        reward_map: Dict[int, float] = {}
        avg_rep = sum(getattr(n, "reputation", 10.0) for n in self.nodes) / max(1, len(self.nodes))

        for i, nid in enumerate(tmp_nodes):
            score = float(scores[i]) if i < len(scores) else float("nan")
            self.contract.set_contribution(r, nid, score)       # contribution for round r
            contrib_map[nid] = score

            node = node_by_id.get(nid)
            if node is not None:
                if not hasattr(node, "contrib_history") or node.contrib_history is None:
                    node.contrib_history = []
                node.contrib_history.append(score)

            claimed = float(metrics_map.get(nid, {}).get("acc", float("nan")))
            # evalacc = float(eval_accs[i]) if i < len(eval_accs) else float("nan")
            # print(f"Node {nid} claimed={claimed:.4f}, eval={evalacc:.4f}, score={score:.4f}")
            # self.contract.set_features(r, nid, claimed_acc=claimed, eval_acc=evalacc)

            rew = self.incent.calculate_reward(node, avg_rep) if node is not None else 0.0
            if rew < 0.0 and self.penalize_negative:
                rew = 0.0
            self.contract.add_reward(r, nid, rew * self.reward_rate)

            # Update the node's reputation based on contribution
            if node is not None:
                new_rep = self.contract.update_reputation(nid, score, current_round=r)
                node.reputation = new_rep
                print(f"Node {nid} new reputation: {new_rep:.4f}")

            reward_map[nid] = rew

        # ---------- 8) settle ----------
        self.contract.settle_round(r)

        print(f"[Round {r}] committee={sorted(committee_ids)} | avg_rep={avg_rep:.4f}")
        print(f"[Round {r}] contribs={contrib_map}")
        print(f"[Round {r}] rewards={reward_map}")

        self.incent.end_round()
        return new_cid, metrics_map, contrib_map, reward_map
