from dataclasses import dataclass
from typing import Dict
import torch
import random

@dataclass
class AttackBehavior:
    strategy: str = "none"
    scale: float = 10.0
    noise_std: float = 0.1
    spoof_mode: str = "none"  # none|high_acc|low_acc|random
    label_flip: bool = False
    is_malicious: bool = False

    def mutate_update(self, update_obj: Dict[str, torch.Tensor], base_sd: Dict[str, torch.Tensor], updated_sd: Dict[str, torch.Tensor], update_type: str):
        if not self.is_malicious or self.strategy == "none":
            return update_obj
        upd = {}
        if self.strategy in ("signflip", "scaling", "gaussian"):
            # Work on delta. If state_dict, convert to delta w.r.t base, mutate, then return same type as input
            if update_type == "state_dict":
                delta = {k: (updated_sd[k] - base_sd[k]).to(torch.float32) for k in updated_sd.keys()}
            else:
                delta = {k: v.to(torch.float32) for k, v in update_obj.items()}
            if self.strategy == "signflip":
                factor = -abs(self.scale)
                delta = {k: v * factor for k, v in delta.items()}
            elif self.strategy == "scaling":
                delta = {k: v * float(self.scale) for k, v in delta.items()}
            elif self.strategy == "gaussian":
                delta = {k: v + torch.randn_like(v) * float(self.noise_std) for k, v in delta.items()}
            if update_type == "state_dict":
                upd = {k: base_sd[k] + delta[k] for k in base_sd.keys()}
            else:
                upd = delta
            return upd
        return update_obj

    def mutate_metrics(self, loss: float, acc: float):
        if not self.is_malicious or self.spoof_mode == "none":
            return loss, acc
        if self.spoof_mode == "high_acc":
            return max(0.0, loss * 0.5), min(1.0, acc + 0.2)
        if self.spoof_mode == "low_acc":
            return loss * 1.5, max(0.0, acc - 0.3)
        if self.spoof_mode == "random":
            return max(0.0, loss * random.uniform(0.5, 1.5)), min(1.0, max(0.0, acc + random.uniform(-0.3, 0.3)))
        return loss, acc

def make_behavior(strategy: str, scale: float = 10.0, noise_std: float = 0.1, spoof_mode: str = "none") -> AttackBehavior:
    b = AttackBehavior()
    b.is_malicious = strategy != "none"
    b.strategy = strategy
    b.scale = scale
    b.noise_std = noise_std
    b.spoof_mode = spoof_mode
    b.label_flip = (strategy == "label_flip")
    return b
