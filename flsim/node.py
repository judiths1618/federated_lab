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
from .evaluation import evaluate_state_dict, evaluate_reconstructed_single  # if not already imported


def test(net, testloader, device, img_key: str = "img", label_key: str = "label"):
    """Validate the model on the test set and return average loss and accuracy.

    The dataloader may yield batches as dictionaries with arbitrary keys
    (e.g. "image"/"label" or "img"/"label"). To remain agnostic to the
    underlying dataset, the keys are parameterised and default to the common
    "img"/"label" pair.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch[img_key], batch[label_key]
            if isinstance(images, list):
                images = torch.stack(images)
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total else 0.0
    loss = loss_sum / total if total else 0.0
    return loss, accuracy


@dataclass
class NodeConfig:
    node_id: int
    batch_size: int = 64
    lr: float = 0.05
    epochs: int = 1


class LocalNode:
    def __init__(
        self,
        cfg: NodeConfig,
        train_ds,
        test_ds,
        keys: Tuple[str, str],
        ipfs,
        contract,
        upload_delta: bool = False,
        save_updates: bool = False,
        save_dir: str = "./res",
        init_stake: float = 100.0,
        init_reputation: float = 10.0,
        behavior: Optional[AttackBehavior] = None,
    ):
        self.cfg = cfg
        self.ipfs = ipfs
        self.contract = contract
        self.img_key, self.label_key = keys
        self.loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        self.testloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        self.num_samples = len(train_ds)
        self.upload_delta = upload_delta
        self.save_updates = save_updates
        self.save_dir = str(save_dir)
        self.stake = init_stake
        self.reputation = init_reputation
        self.participation = 0
        self.contrib_history: List[float] = []
        self.last_delta_norm: float = 0.0
        self.behavior = behavior or make_behavior("none")

    def _compute_delta(
        self, new_sd: Dict[str, torch.Tensor], base_sd: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {k: (new_sd[k] - base_sd[k]).to(dtype=torch.float32) for k in new_sd.keys()}

    def train_one_round(self, round_idx: int, global_model_cid: str):
        base_sd = self.ipfs.load(global_model_cid)
        model = LinearMNIST()
        model.load_state_dict(base_sd)
        model.train()
        opt = optim.SGD(model.parameters(), lr=self.cfg.lr)
        crit = nn.CrossEntropyLoss()

        for _ in range(self.cfg.epochs):
            for batch in self.loader:
                xb, yb = batch[self.img_key], batch[self.label_key]
                if isinstance(xb, list):
                    xb = torch.stack(xb)
                if isinstance(yb, list):
                    yb = torch.tensor(yb)
                if self.behavior.label_flip:
                    yb = 9 - yb  # MNIST label flip example
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()

        # evaluate on held-out test set
        test_loss, test_acc = test(model, self.testloader, "cpu", self.img_key, self.label_key)

        updated_sd = model.state_dict()

        if self.upload_delta:
            update_obj, update_type = self._compute_delta(updated_sd, base_sd), "delta"
        else:
            update_obj, update_type = updated_sd, "state_dict"

        update_obj = self.behavior.mutate_update(update_obj, base_sd, updated_sd, update_type)

        if update_type == "delta":
            delta_norm = math.sqrt(
                sum(
                    float((v.float() ** 2).sum())
                    for v in update_obj.values()
                    if torch.is_tensor(v)
                )
            )
        else:
            delta_norm = math.sqrt(
                sum(
                    float(((updated_sd[k] - base_sd[k]).float() ** 2).sum())
                    for k in updated_sd.keys()
                )
            )

        self.last_delta_norm = float(delta_norm)
        self.last_loss = float(test_loss)
        self.last_acc = float(test_acc)
        self.participation += 1

        rep_loss, rep_acc = self.behavior.mutate_metrics(test_loss, test_acc)

        model_cid = self.ipfs.save(update_obj)
        metrics_cid = self.ipfs.save(
            {"loss": rep_loss, "acc": rep_acc, "samples": self.num_samples}
        )

        if self.save_updates:
            fname = os.path.join(
                self.save_dir,
                "updates",
                f"round_{round_idx}_node_{self.cfg.node_id}_{update_type}.pt",
            )
            torch.save(update_obj, fname)

        self.contract.submit_model(
            round_idx, self.cfg.node_id, model_cid, metrics_cid, update_type
        )
        return rep_loss, rep_acc, update_type, model_cid, metrics_cid
