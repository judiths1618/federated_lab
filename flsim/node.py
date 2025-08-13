from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .models import LinearMNIST
from .attacks import AttackBehavior, make_behavior

@dataclass
class NodeConfig:
    node_id: int
    batch_size: int = 64
    lr: float = 0.05
    epochs: int = 1

class LocalNode:
    def __init__(self, cfg: NodeConfig, ds, keys: Tuple[str, str], ipfs, contract,
                 upload_delta: bool = False, save_updates: bool = False, save_dir: str = "./res",
                 init_stake: float = 100.0, init_reputation: float = 10.0,
                 behavior: Optional[AttackBehavior] = None):
        self.cfg = cfg
        self.ipfs = ipfs
        self.contract = contract
        self.img_key, self.label_key = keys
        self.loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
        self.num_samples = len(ds)
        self.upload_delta = upload_delta
        self.save_updates = save_updates
        self.save_dir = str(save_dir)
        self.stake = init_stake
        self.reputation = init_reputation
        self.participation = 0
        self.contrib_history: List[float] = []
        self.last_delta_norm: float = 0.0
        self.behavior = behavior or make_behavior("none")

    def _compute_delta(self, new_sd: Dict[str, torch.Tensor], base_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (new_sd[k] - base_sd[k]).to(dtype=torch.float32) for k in new_sd.keys()}

    def train_one_round(self, round_idx: int, global_model_cid: str):
        base_sd = self.ipfs.load(global_model_cid)
        model = LinearMNIST(); model.load_state_dict(base_sd); model.train()
        opt = optim.SGD(model.parameters(), lr=self.cfg.lr)
        crit = nn.CrossEntropyLoss()

        total_loss, correct, total = 0.0, 0, 0
        for _ in range(self.cfg.epochs):
            for batch in self.loader:
                xb, yb = batch[self.img_key], batch[self.label_key]
                if isinstance(xb, list): xb = torch.stack(xb)
                if isinstance(yb, list): yb = torch.tensor(yb)
                if self.behavior.label_flip:
                    yb = 9 - yb  # MNIST label flip example
                opt.zero_grad(); out = model(xb)
                loss = crit(out, yb); loss.backward(); opt.step()
                total_loss += float(loss) * xb.size(0)
                _, pred = out.max(1)
                total += yb.size(0); correct += int((pred == yb).sum())

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)

        updated_sd = model.state_dict()
        if self.upload_delta:
            update_obj, update_type = self._compute_delta(updated_sd, base_sd), "delta"
        else:
            update_obj, update_type = updated_sd, "state_dict"

        # mutate update (poisoning)
        update_obj = self.behavior.mutate_update(update_obj, base_sd, updated_sd, update_type)

        # ---------- 新：仅记录原始特征，贡献改到 Aggregator ----------
        if update_type == "delta":
            delta_norm = math.sqrt(
                sum(float((v.float() ** 2).sum()) for v in update_obj.values() if torch.is_tensor(v))
            )
        else:
            # 若上传的是 state_dict，这里用 (updated - base) 的范数
            delta_norm = math.sqrt(
                sum(float(((updated_sd[k] - base_sd[k]).float() ** 2).sum()) for k in updated_sd.keys())
            )

        self.last_delta_norm = float(delta_norm)
        self.last_loss = float(avg_loss)
        self.last_acc = float(acc)
        self.participation += 1

        rep_loss, rep_acc = self.behavior.mutate_metrics(avg_loss, acc)

        model_cid = self.ipfs.save(update_obj)
        metrics_cid = self.ipfs.save({"loss": rep_loss, "acc": rep_acc, "samples": self.num_samples})

        # Optional: save update *.pt locally for inspection
        if self.save_updates:
            fname = os.path.join(
                self.save_dir,
                "updates",
                f"round_{round_idx}_node_{self.cfg.node_id}_{update_type}.pt",
            )
            torch.save(update_obj, fname)
        self.contract.submit_model(round_idx, self.cfg.node_id, model_cid, metrics_cid, update_type)
        return rep_loss, rep_acc, update_type, model_cid, metrics_cid
