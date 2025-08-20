"""Our contract implementation built on top of BaseContract with FLAME detection."""

from collections import Counter
from typing import Dict, List, Optional, Set

import torch

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

from flsim.evaluation import reconstruct_state
from .base import BaseContract


class OurContract(BaseContract):
    """Contract that filters malicious nodes using the FLAME defence."""

    def __init__(self, ipfs: Optional[object] = None) -> None:
        super().__init__()
        self.ipfs = ipfs

    def set_ipfs(self, ipfs: object) -> None:
        """Allow late binding of the IPFS simulator used to fetch updates."""
        self.ipfs = ipfs

    @staticmethod
    def _flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in state.values() if torch.is_tensor(p)])

    def _flame_detect(self, updates: List[Dict[str, torch.Tensor]]) -> Set[int]:
        """Return indices of updates flagged as malicious by FLAME clustering."""
        if hdbscan is None or not updates:
            return set()
        matrix = torch.stack([self._flatten_state(u).double() for u in updates])
        cluster = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=matrix.shape[0] // 2 + 1,
            min_samples=1,
            allow_single_cluster=True,
        )
        cluster.fit(matrix)
        labels = cluster.labels_
        counts = Counter(l for l in labels if l >= 0)
        main = counts.most_common(1)[0][0] if counts else -1
        return {i for i, l in enumerate(labels) if l != main}

    def settle_round(self, round_idx: int) -> Dict[int, float]:
        r = int(round_idx)
        rewards_r = self.rewards.setdefault(r, {})
        self.mal_detected[r] = {}
        self.penalties[r] = {}

        contribs_r = self.contribs.get(r, {})
        if not contribs_r:
            return dict(rewards_r)

        if self.ipfs is None:
            return super().settle_round(round_idx)

        subs = self.records.get(r, {})
        base_cid = self.global_models.get(r, "")
        base_state = self.ipfs.load(base_cid) if base_cid else {}

        updates: List[Dict[str, torch.Tensor]] = []
        node_ids: List[int] = []
        for nid, (mcid, _mtcid, updtype) in subs.items():
            try:
                upd = self.ipfs.load(mcid)
                realized = reconstruct_state(base_state, upd, updtype)
                updates.append(realized)
                node_ids.append(int(nid))
            except Exception:
                continue

        mal_indices = self._flame_detect(updates)
        malicious_ids = {node_ids[i] for i in mal_indices}

        warmup = int(getattr(self.params, "warmup_rounds", 1))
        stake_pen = float(getattr(self.params, "stake_penalty_factor", 0.02))
        rep_pen = float(getattr(self.params, "rep_penalty_factor", 0.5))

        for nid, info in self.nodes.items():
            if nid in contribs_r:
                info["participation"] = int(info.get("participation", 0)) + 1

            score = float(contribs_r.get(nid, 0.0))
            reward = float(rewards_r.get(nid, 0.0))

            if r < warmup or nid not in malicious_ids:
                info["stake"] = float(info["stake"]) + reward
                self.balances[nid] = self.balances.get(nid, 0.0) + reward
                self.update_reputation(nid, max(0.0, score), current_round=r)
                self.penalties[r][nid] = 0.0
                self.mal_detected[r][nid] = 0
            else:
                old_stake = float(info["stake"])
                pen_amt = old_stake * stake_pen
                info["stake"] = max(0.0, old_stake - pen_amt)
                rep = float(info.get("reputation", 0.0))
                info["reputation"] = max(0.0, rep * (1.0 - rep_pen))
                rewards_r[nid] = 0.0
                self.penalties[r][nid] = float(pen_amt)
                self.mal_detected[r][nid] = 1

        return dict(rewards_r)


__all__ = ["OurContract"]
