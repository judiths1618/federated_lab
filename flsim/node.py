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
from .eval import evaluate_state_dict, evaluate_reconstructed_single  # if not already imported

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    # print(f"Test loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return loss, accuracy
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
        # Save base (CPU tensors) to models/; include node id to avoid clobbering
        models_dir = os.path.join(self.save_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        base_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in base_sd.items()}
        base_fname = os.path.join(models_dir, f"base_round_{round_idx}_node_{self.cfg.node_id}.pt")
        if not os.path.exists(base_fname):  # avoid overwriting if called multiple times
            torch.save(base_cpu, base_fname)

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

        # 2.1 Evaluate the updated FULL model (before any mutation), on test/val
        # Choose dataset/model_hint; fall back if not present on cfg
        dataset_name = getattr(getattr(self, "cfg", None), "dataset_name", None) or \
                    getattr(getattr(self, "data_cfg", None), "name", None) or "mnist"
        # model_hint   = getattr(getattr(self, "cfg", None), "model_name", None) or "linear-mnist"

        print(dataset_name, model)
        eval_acc_updated = evaluate_state_dict(
            updated_sd,
            dataset=dataset_name,
            model_hint=model,
            # base_state=None, update_type='state' by default
        )
        # keep it for logging/inspection
        self.last_eval_acc_updated = float(eval_acc_updated)
        print(f"Node {self.cfg.node_id} round {round_idx} updated model, acc: {acc}, avg_loss: {avg_loss}, eval acc: {eval_acc_updated:.4f}")
        
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

        # add evaluation process 

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
