from typing import Dict, Tuple

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
        self.records: Dict[int, Dict[int, Tuple[str, str, str]]] = {}
        self.global_models: Dict[int, str] = {}
        self.manifests: Dict[int, str] = {}
        self.contribs: Dict[int, Dict[int, float]] = {}
        self.rewards: Dict[int, Dict[int, float]] = {}
        self.balances: Dict[int, float] = {}
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
    def set_contribution(self, round_idx: int, node_id: int, score: float):
        self.contribs.setdefault(round_idx, {})[node_id] = float(score)
    def add_reward(self, round_idx: int, node_id: int, amount: float):
        self.rewards.setdefault(round_idx, {})[node_id] = float(amount)
        self.balances[node_id] = self.balances.get(node_id, 0.0) + float(amount)
    def get_balance(self, node_id: int) -> float:
        return self.balances.get(node_id, 0.0)
