from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
import math
import random  # NEW: use random.random()
import numpy as np  # 确保文件顶部已导入

# --------------------------- helpers ---------------------------

def _softmax(x):
    # numeric-stable softmax without scipy
    if not x:
        return []
    m = max(x)
    exps = [math.exp(v - m) for v in x]
    s = sum(exps) or 1.0
    return [v / s for v in exps]

def _get_node_attr(node: Any, name: str, default: float = 0.0):
    """
    Robustly fetch an attribute from a LocalNode or dict-like record.
    - supports node.reputation / node.cooldown / node.cfg.node_id
    - supports dict["reputation"] / dict["cooldown"] / dict["node_id"]
    """
    # dict-like
    if isinstance(node, dict):
        if name == "node_id":
            # prefer 'node_id' then 'id'
            return int(node.get("node_id", node.get("id", default)))
        return node.get(name, default)
    # object-like
    if name == "node_id":
        # try cfg.node_id → id
        try:
            return int(getattr(getattr(node, "cfg", None), "node_id"))
        except Exception:
            return int(getattr(node, "id", default))
    # plain attr
    try:
        return getattr(node, name)
    except Exception:
        return default


def _sigmoid_scalar(x: float) -> float:
    # 数值稳定版 sigmoid（标量）
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# --------------------------- params ---------------------------

@dataclass
class IncentiveParams:
    base_reward: float = 100.0
    stake_weight: float = 0.4
    hist_decay: float = 0.9
    rep_exponent: float = 1.0
    committee_bonus: float = 0.10
    reward_rate: float = 1.0
    penalize_negative: bool = False
    stake_penalty_factor: float = 0.02
    rep_gain_factor: float = 0.5
    rep_penalty_factor: float = 0.5
    detect_score_thresh: float = 0.05
    mal_eval_diff_thresh: float = 0.15
    # optional but useful knobs
    committee_size: int = 10
    committee_cooldown: int = 1  # rounds a selected node must wait before rejoining
    # —— 误判保护 & 检测（如果你已加过可以忽略）——
    mal_min_gap: float = 0.05
    mal_allowed_base: float = 0.05
    mal_allowed_slope: float = 0.20
    warmup_rounds: int = 1

    # —— 声誉更新权重（按你的公式里的 X_c、X_s）——
    X_c: float = 10.0   # 贡献质量项权重（你可以按实际分布调小/调大）
    X_s: float = 5.0    # 稳定性项权重

    # —— 动态上限参数（> cap_round 用 late 上限，否则用 early 上限）——
    rep_cap_early: float = 300.0
    rep_cap_late: float = 500.0
    rep_cap_round: int = 50


# --------------------------- storage sims ---------------------------

class IPFSSim:
    def __init__(self):
        self.storage: Dict[str, object] = {}
        self.counter = 0
    def save(self, obj) -> str:
        h = f"Qm{self.counter:08d}"
        self.storage[h] = obj
        self.counter += 1
        return h
    def load(self, h: str):
        return self.storage[h]

class ContractSim:
    def __init__(self):
        # submissions & artifacts
        self.records: Dict[int, Dict[int, Tuple[str, str, str]]] = {}
        self.global_models: Dict[int, str] = {}
        self.manifests: Dict[int, str] = {}

        # economics
        self.contribs: Dict[int, Dict[int, float]] = {}
        self.rewards: Dict[int, Dict[int, float]] = {}
        self.balances: Dict[int, float] = {}

        # participants / committees
        self.nodes: Dict[int, Dict[str, float]] = {}        # node_id -> {stake, reputation, cooldown, ...}
        self.committees: Dict[int, List[int]] = {}          # round -> committee node ids
        self.committee_history: List[List[int]] = []        # optional history

        # telemetry & detection
        self.features: Dict[int, Dict[int, Dict[str, float]]] = {}  # round -> node_id -> {feature:value}
        self.mal_detected: Dict[int, Dict[int, int]] = {}            # round -> node_id -> 0/1
        self.penalties: Dict[int, Dict[int, float]] = {}             # round -> node_id -> penalty amount

        # incentive params
        self.params = IncentiveParams()

    # ---------------- basic storage ----------------

    def submit_model(self, round_idx: int, node_id: int, model_cid: str, metrics_cid: str, update_type: str):
        self.records.setdefault(round_idx, {})[node_id] = (model_cid, metrics_cid, update_type)

    def get_round_submissions(self, round_idx: int):
        return self.records.get(round_idx, {})

    def set_global_model(self, round_idx: int, cid: str):
        self.global_models[round_idx] = cid

    def get_latest_global(self):
        if not self.global_models:
            return (-1, "")
        r = max(self.global_models.keys())
        return (r, self.global_models[r])

    def set_round_manifest(self, round_idx: int, cid: str):
        self.manifests[round_idx] = cid

    # ---------------- economics helpers ----------------

    def set_contribution(self, round_idx: int, node_id: int, score: float):
        self.contribs.setdefault(int(round_idx), {})[int(node_id)] = float(score)

    def add_reward(self, round_idx: int, node_id: int, amount: float):
        """
        累计本轮奖励到 rewards[r][nid]。
        注意：这里**不**更新 balances；统一在 settle_round() 中根据最终 reward 入账到余额。
        """
        r = int(round_idx); n = int(node_id)
        self.rewards.setdefault(r, {})
        self.rewards[r][n] = self.rewards[r].get(n, 0.0) + float(amount)

    def get_balance(self, node_id: int) -> float:
        return float(self.balances.get(int(node_id), 0.0))

    # ---------------- config / registry ----------------

    def set_incentive_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)

    def register_node(self, node_id: int, stake: float, reputation: float):
        self.nodes[int(node_id)] = {
            "stake": float(stake),
            "reputation": float(reputation),
            "cooldown": 0.0,
            "contrib_history": [],
            "participation": 0,   # <— 新增：参与轮数统计
        }

    def update_reputation(self, node_id: int, contribution: float, *, current_round: int) -> float:
        """Update a node's reputation using a robust decay and stability model."""
        nid = int(node_id)
        if nid not in self.nodes:
            return 0.0

        info = self.nodes[nid]
        rep = float(info.get("reputation", 0.0))
        history = list(info.get("contrib_history", []))
        participation = int(info.get("participation", 0))

        # dynamic decay factor based on participation count
        age_factor = 1.0 - 1.0 / (1.0 + (participation / 100.0))
        delta = 0.88 + 0.07 * age_factor

        # contribution quality via sigmoid normalization
        contrib_base = float(getattr(self.params, "contrib_base", 0.0))
        contrib_thre = float(getattr(self.params, "contrib_thre", 1.0))
        denom = max(1e-8, (contrib_thre - contrib_base))
        contrib_quality = _sigmoid_scalar((float(contribution) - contrib_base) / denom)

        # stability over recent history
        if len(history) >= 5:
            stability = 1.0 - (float(np.std(history[-5:], dtype=float)) / 5.0)
        else:
            stability = 0.8

        X_c = float(self.params.X_c)
        X_s = float(self.params.X_s)
        new_rep = (rep * delta) + (contrib_quality * X_c) + (stability * X_s)

        cap_round = int(self.params.rep_cap_round)
        rep_cap = float(self.params.rep_cap_late if current_round > cap_round else self.params.rep_cap_early)
        new_rep = float(np.clip(new_rep, 0.0, rep_cap))

        self.nodes[nid]["reputation"] = new_rep
        return new_rep
    # ---------------- committee selection (refined) ----------------
    def set_committee(self, round_idx: int, nodes: List[Any], *, num_strata: int = 3) -> List[int]:
        """
        Select committee members using stratified sampling based on reputation & cooldown.
        Returns a list of selected node IDs.
        """
        committee_size = int(getattr(self.params, "committee_size", 10))
        rep_exponent = float(getattr(self.params, "rep_exponent", 1.0))

        # robust projection of nodes into simple records
        proj = []
        for n in nodes:
            nid = int(_get_node_attr(n, "node_id", -1))
            rep = float(_get_node_attr(n, "reputation", 0.0))
            cd  = float(_get_node_attr(n, "cooldown", 0.0))
            if nid >= 0:
                proj.append({"node_id": nid, "reputation": rep, "cooldown": cd})

        if not proj:
            self.committees[int(round_idx)] = []
            self.committee_history.append([])
            return []

        # sort by reputation desc and build strata
        proj.sort(key=lambda r: r["reputation"], reverse=True)
        num_strata = max(1, int(num_strata))
        strata_size = max(1, len(proj) // num_strata)
        strata: List[List[Dict[str, float]]] = [
            proj[i * strata_size : (i + 1) * strata_size] for i in range(num_strata)
        ]
        # push remainder to last stratum
        remainder_start = num_strata * strata_size
        if remainder_start < len(proj):
            strata[-1].extend(proj[remainder_start:])

        # initial quota per stratum
        quotas = [committee_size // num_strata] * num_strata
        rem = committee_size - sum(quotas)
        for i in range(rem):
            quotas[i % num_strata] += 1

        selected: List[Dict[str, float]] = []

        # first pass: each stratum by quota, reputation-weighted softmax
        for stratum, quota in zip(strata, quotas):
            candidates = [r for r in stratum if r["cooldown"] <= 0]
            if not candidates or quota <= 0:
                continue
            take = min(quota, len(candidates))
            if len(candidates) == 1:
                chosen = [candidates[0]]
            else:
                probs = _softmax([c["reputation"] ** rep_exponent for c in candidates])
                idxs = list(range(len(candidates)))
                chosen = []
                for _ in range(take):
                    s = sum(probs) or 1.0
                    r = random.random() * s
                    acc = 0.0
                    pick = 0
                    for j, p in enumerate(probs):
                        acc += p
                        if r <= acc:
                            pick = j
                            break
                    chosen.append(candidates[idxs[pick]])
                    # remove picked (without full renorm is ok since we draw w.r.t current sum)
                    probs.pop(pick); idxs.pop(pick)
                    if not probs:
                        break
            selected.extend(chosen)

        # second pass: top-up if needed, across all remaining eligible
        if len(selected) < committee_size:
            need = committee_size - len(selected)
            selected_ids = set(r["node_id"] for r in selected)
            eligible = [r for r in proj if r["cooldown"] <= 0 and r["node_id"] not in selected_ids]
            if eligible:
                if len(eligible) <= need:
                    selected.extend(eligible)
                else:
                    probs = _softmax([e["reputation"] ** rep_exponent for e in eligible]) if len(eligible) > 1 else [1.0]
                    idxs = list(range(len(eligible)))
                    for _ in range(need):
                        s = sum(probs) or 1.0
                        rr = random.random() * s
                        acc = 0.0
                        pick = 0
                        for j, p in enumerate(probs):
                            acc += p
                            if rr <= acc:
                                pick = j
                                break
                        selected.append(eligible[idxs[pick]])
                        probs.pop(pick); idxs.pop(pick)
                        if not probs:
                            break

        committee_ids = [int(r["node_id"]) for r in selected][:committee_size]
        self.committees[int(round_idx)] = committee_ids
        self.committee_history.append(committee_ids)

        # decrement existing cooldowns
        for info in self.nodes.values():
            info["cooldown"] = max(0.0, float(info.get("cooldown", 0.0)) - 1.0)

        cd_rounds = int(getattr(self.params, "committee_cooldown", 1))
        for nid in committee_ids:
            if nid in self.nodes:
                self.nodes[nid]["cooldown"] = float(cd_rounds)

        return committee_ids

    # ---------------- features & settlement ----------------

    def set_features(self, round_idx: int, node_id: int, **kwargs):
        rec = dict(kwargs)
        self.features.setdefault(int(round_idx), {})[int(node_id)] = {
            k: float(v) for k, v in rec.items() if isinstance(v, (int, float))
        }

    def settle_round(self, round_idx: int) -> Dict[int, float]:
        """
        Finalize incentives for round `round_idx`:
        - Adaptive malicious detection using claimed/eval accuracy gaps
        - Warmup (no penalties)
        - Rewards settlement into balances
        - Stake and reputation updates (with user-provided robust rep update)
        Returns the per-node reward map for this round.
        """
        r = int(round_idx)

        # DO NOT clear rewards: aggregator has already written to self.rewards[r][nid]
        rewards_r = self.rewards.setdefault(r, {})
        self.mal_detected[r] = {}
        self.penalties[r] = {}

        contribs_r = self.contribs.get(r, {})
        if not contribs_r:
            # No contributions; nothing to settle
            return dict(rewards_r)

        feats_r = self.features.get(r, {})

        # --- Build adaptive threshold (IQR) over non-negative diffs: max(0, claimed - eval) ---
        diffs = []
        for nid, f in feats_r.items():
            cl = float(f.get("claimed_acc", float("nan")))
            ev = float(f.get("eval_acc", float("nan")))
            if math.isfinite(cl) and math.isfinite(ev):
                diffs.append(max(0.0, cl - ev))

        base_tau = float(getattr(self.params, "mal_eval_diff_thresh", 0.15))
        if len(diffs) >= 4:
            vals = np.asarray(diffs, dtype=float)
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = max(1e-8, float(q3 - q1))
            adaptive_tau = float(q3 + 1.5 * iqr)
        else:
            adaptive_tau = float("nan")

        pre_taus = [float(f.get("pre_tau")) for f in feats_r.values() if "pre_tau" in f]
        pre_tau = pre_taus[0] if pre_taus else float("nan")

        tau_candidates = [base_tau]
        if math.isfinite(pre_tau):
            tau_candidates.append(pre_tau)
        if math.isfinite(adaptive_tau):
            tau_candidates.append(adaptive_tau)
        final_tau = max(tau_candidates)

        mal_min_gap = float(getattr(self.params, "mal_min_gap", 0.05))
        a = float(getattr(self.params, "mal_allowed_base", 0.05))
        b = float(getattr(self.params, "mal_allowed_slope", 0.20))
        warmup = int(getattr(self.params, "warmup_rounds", 1))
        detect_score_thresh = float(getattr(self.params, "detect_score_thresh", 0.05))

        # --- Per-node settlement ---
        for nid, info in self.nodes.items():
            # Participation count (drives age/delta in update_reputation)
            if nid in contribs_r:
                info["participation"] = int(info.get("participation", 0)) + 1

            score = float(contribs_r.get(nid, 0.0))
            feat = feats_r.get(nid, {})

            claimed = float(feat.get("claimed_acc", float("nan")))
            evalacc = float(feat.get("eval_acc", float("nan")))

            # Suspicious decision:
            #  - only non-negative diff matters: diff = max(0, claimed - eval)
            #  - threshold = max( final_tau (IQR/default/pre_tau), mal_min_gap, a + b*(1-claimed) )
            if math.isfinite(claimed) and math.isfinite(evalacc):
                diff = max(0.0, claimed - evalacc)
                allowed_rel = a + b * max(0.0, 1.0 - claimed)
                tau = max(final_tau, mal_min_gap, allowed_rel)
                suspicious = (diff > tau)
            else:
                suspicious = (score < detect_score_thresh)

            self.mal_detected[r][nid] = int(bool(suspicious))

            # --- Warmup: mark only, no penalties; still settle rewards & update reputation ---
            if r < warmup:
                reward = float(rewards_r.get(nid, 0.0))
                # stake grows by reward; reward goes to balance once per round
                info["stake"] = float(info["stake"]) + reward
                self.balances[nid] = self.balances.get(nid, 0.0) + reward
                # reputation via robust rule
                self.update_reputation(nid, max(0.0, score), current_round=r)
                self.penalties[r][nid] = 0.0
                continue

            if suspicious:
                # Penalize stake multiplicatively; zero out reward for this round
                old_stake = float(info["stake"])
                stake_pen = old_stake * float(getattr(self.params, "stake_penalty_factor", 0.02))
                info["stake"] = max(0.0, old_stake - stake_pen)
                # Reputation multiplicative penalty
                rep_pen = float(getattr(self.params, "rep_penalty_factor", 0.5))
                punished_rep = float(info.get("reputation", 0.0)) * (1.0 - rep_pen)
                info["reputation"] = max(0.0, punished_rep)

                self.penalties[r][nid] = float(stake_pen)
                rewards_r[nid] = 0.0  # override any aggregator-written reward
                # No balance credit on malicious nodes
            else:
                # Good node: credit reward, grow stake, update reputation using your robust updater
                reward = float(rewards_r.get(nid, 0.0))
                info["stake"] = float(info["stake"]) + reward
                self.balances[nid] = self.balances.get(nid, 0.0) + reward

                # Reputation update per your formula (uses participation, recent stability, etc.)
                self.update_reputation(nid, max(0.0, score), current_round=r)

                self.penalties[r][nid] = 0.0

        # (Optional) small committee reputation boost per round:
        # self.apply_committee_reputation_bonus(r)

        return dict(rewards_r)
