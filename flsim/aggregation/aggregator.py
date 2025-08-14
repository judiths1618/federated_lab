import importlib
import os
import math
from typing import Dict, List, Optional

import numpy as np
import torch

from ..eval import evaluate_reconstructed_batch, evaluate_many_state_dicts, reconstruct_state

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
        base_reward: float,
        stake_weight: float,
        committee_size: int,
        hist_decay_factor: float,
        strategy_name: str = "fedavg",
        dataset_name: str = "mnist",
        model_name: Optional[str] = None,
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

        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "updates"), exist_ok=True)

        self.strategy_name = strategy_name
        mod = importlib.import_module(f"flsim.aggregation.{strategy_name}")
        self.strategy = getattr(mod, "Strategy")()

        # Contribution weights
        self.W_ALIGN, self.W_ACC, self.W_LOSS, self.W_NORM = 0.4, 0.3, 0.2, 0.1

    # ----------------- helpers -----------------
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
        stakes = [getattr(n, "stake", 10.0) for n in self.nodes]
        avg_stake = sum(stakes) / max(1, len(stakes))
        effective_stake = min(getattr(node, "stake", 10.0), 3.0 * avg_stake)

        hist = getattr(node, "contrib_history", [])
        # 取最近 5 轮（包含当前轮，已在下面 append）
        recent = hist[-5:] if hist else []
        hist_contrib = 0.0
        for t, c in enumerate(reversed(recent)):
            hist_contrib += float(c) * (self.hist_decay_factor ** t)
        print(f"Node {node.cfg.node_id} history contribution: {hist_contrib:.4f}")

        reputations = [getattr(n, "reputation", 10.0) for n in self.nodes]
        diversity_bonus = self._jain_fairness(reputations)

        node_rep = getattr(node, "reputation", 10.0)
        alpha = self._sigmoid((avg_rep - node_rep) / 50.0) * self.stake_weight
        beta = 1.0 - alpha
        committee_bonus = 20.0 * diversity_bonus if in_committee else 0.0

        # 历史贡献求和避免为 0
        total_contrib = sum((n.contrib_history[-1] if getattr(n, "contrib_history", []) else 0.0) for n in self.nodes)
        total_contrib = total_contrib if abs(total_contrib) > 1e-8 else 1.0
        total_stake = sum(stakes)
        total_stake = total_stake if total_stake > 1e-8 else 1.0

        reward = (
            (alpha * self.base_reward * (effective_stake / total_stake)
             + beta * self.base_reward * (hist_contrib / total_contrib))
            * diversity_bonus
            + committee_bonus
        )
        print(f"Node {node.cfg.node_id} reward before clipping: {reward:.4f}")
        return reward
        # reward_clipped = max(self.clip_min, min(self.clip_max, reward))
        # return self.reward_rate * (reward_clipped if self.penalize_negative else max(0.0, reward_clipped))

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

    def aggregate_round(self, r: int, base_cid: str):
        # subs = self.contract.get_round_submissions(r)
        # # base_sd 是第 r 轮开始时的全局模型，用于还原每个客户端的本地模型
        # base_sd = self.ipfs.load(base_cid)
        # 保存 base 方便之后结合各节点的 delta 重建其训练后的模型
        # torch.save(base_sd, os.path.join(self.save_dir, "models", f"global_round_{r}_base.pt"))

        subs = self.contract.get_round_submissions(r)
        base_sd = self.ipfs.load(base_cid)

        realized_states, weights, metrics_map = [], [], []
        tmp_nodes, tmp_norms, tmp_aligns, tmp_accs, tmp_losses = [], [], [], [], []

        # metrics_map = {}

        realized_states: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        metrics_map: Dict[int, Dict] = {}

        # for scoring
        tmp_nodes: List[int] = []
        tmp_norms: List[float] = []
        tmp_aligns: List[float] = []
        tmp_accs: List[float] = []
        tmp_losses: List[float] = []
        deltas: List[Dict[str, torch.Tensor]] = []   # keep raw updates for eval
        upd_types: List[str] = []

        # 1) 还原本地模型 & 收集特征
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
            tmp_losses.append(float(mt.get("loss", float("nan"))))
            tmp_aligns.append(float("nan"))  # 等聚合方向出来后再算

        # ---------- 2) committee ----------
        try:
            committee_ids = self.contract.set_committee(r, self.nodes)
        except TypeError:
            fallback = self._calc_committee()
            maybe = self.contract.set_committee(r, fallback)
            committee_ids = maybe if isinstance(maybe, list) else fallback
        committee_ids = set(committee_ids or [])

        # ---------- 3) eval reconstructed models (base + delta) ----------
  
        # if realized_states:
        #     # 按 update_type 分组评测：delta -> base+delta，state -> 直接评测
        #     eval_accs = [float("nan")] * len(deltas)
        #     delta_idx = [i for i, t in enumerate(upd_types) if t == "delta"]
        #     state_idx = [i for i, t in enumerate(upd_types) if t != "delta"]
        #     if delta_idx:
        #         delta_updates = [deltas[i] for i in delta_idx]
        #         delta_accs = evaluate_reconstructed_batch(
        #             base_state=base_sd,
        #             deltas=delta_updates,
        #             dataset=self.dataset_name,
        #             model_hint=self.model_name,
        #             max_workers=4,
        #         )
        #         for i, acc in zip(delta_idx, delta_accs):
        #             eval_accs[i] = acc
        #     if state_idx:
        #         state_updates = [deltas[i] for i in state_idx]
        #         state_accs = evaluate_many_state_dicts(
        #             state_updates,
        #             dataset=self.dataset_name,
        #             model_hint=self.model_name,
        #             max_workers=4,
        #         )
        #         for i, acc in zip(state_idx, state_accs):
        #             eval_accs[i] = acc
        # else:
        #     # 无提交：直接维持 base
        #     agg_sd = base_sd
        #     new_cid = self.ipfs.save(agg_sd)
        #     self.contract.set_global_model(r + 1, new_cid)
        #     self.contract.settle_round(r)
        #     return new_cid, {}, {}, {}

        # 3) 评测重构后的本地模型
        if realized_states:
            eval_accs = evaluate_many_state_dicts(
                realized_states,
                dataset=self.dataset_name,
                model_hint=self.model_name,
                max_workers=4,
            )
        else:
            agg_sd = base_sd
            new_cid = self.ipfs.save(agg_sd)
            self.contract.set_global_model(r + 1, new_cid)
            self.contract.settle_round(r)
            return new_cid, {}, {}, {}

        # ---------- 4) aggregate -> new global ----------
        meta = {"node_ids": tmp_nodes, "committee_ids": list(committee_ids)}
        if hasattr(self, "strategy") and hasattr(self.strategy, "aggregate"):
            agg_sd = self.strategy.aggregate(realized_states, weights, base_sd=base_sd, meta=meta)
        else:
            agg_sd = self.fedavg_weighted(realized_states, weights)

        new_cid = self.ipfs.save(agg_sd)
        torch.save(agg_sd, os.path.join(self.save_dir, "models", f"global_round_{r}.pt"))
        self.contract.set_global_model(r + 1, new_cid)

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
        loss_01   = self._robust_minmax(np.array(tmp_losses))
        loss_good = 1.0 - loss_01
        scores = self.W_ALIGN * align_n + self.W_ACC * acc_n + self.W_LOSS * loss_good + self.W_NORM * norm_n

        # ---------- 7) commit & reward ----------
        contrib_map: Dict[int, float] = {}
        reward_map: Dict[int, float] = {}
        node_by_id = {n.cfg.node_id: n for n in self.nodes}
        avg_rep = sum(getattr(n, "reputation", 10.0) for n in self.nodes) / max(1, len(self.nodes))

        for i, nid in enumerate(tmp_nodes):
            score = float(scores[i]) if i < len(scores) else float("nan")
            self.contract.set_contribution(r, nid, score)
            contrib_map[nid] = score

            node = node_by_id.get(nid)
            if node is not None:
                if not hasattr(node, "contrib_history") or node.contrib_history is None:
                    node.contrib_history = []
                node.contrib_history.append(score)

            claimed = float(metrics_map.get(nid, {}).get("acc", float("nan")))
            evalacc = float(eval_accs[i]) if i < len(eval_accs) else float("nan")
            print(f"Node {nid} claimed={claimed:.4f}, eval={evalacc:.4f}, score={score:.4f}")
            self.contract.set_features(r, nid, claimed_acc=claimed, eval_acc=evalacc)

            in_committee = (nid in committee_ids)
            rew = self._calculate_reward(node, avg_rep, in_committee) if node is not None else 0.0
            self.contract.add_reward(r, nid, rew)
            reward_map[nid] = rew

        # ---------- 8) settle ----------
        self.contract.settle_round(r)

        print(f"[Round {r}] committee={sorted(committee_ids)} | avg_rep={avg_rep:.4f}")
        print(f"[Round {r}] contribs={contrib_map}")
        print(f"[Round {r}] rewards={reward_map}")

        return new_cid, metrics_map, contrib_map, reward_map
