import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from typing import Optional, Sequence, List
import concurrent.futures

from .models import LinearMNIST

@torch.no_grad()
def evaluate_global(cidr: str, ipfs) -> float:
    test_set = MNIST(root="./data", train=False, download=True, transform=T.ToTensor())
    loader = DataLoader(test_set, batch_size=256, shuffle=False)
    model = LinearMNIST()
    model.load_state_dict(ipfs.load(cidr))
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        out = model(xb); _, pred = out.max(1)
        total += yb.size(0); correct += int((pred == yb).sum())
    return correct/max(1,total)


class SimpleCIFAR10(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- helpers ----------------

def _make_dataset(dataset: str):
    ds = (dataset or 'mnist').lower()
    if ds == 'cifar10':
        test = CIFAR10(root='./data', train=False, download=True, transform=T.ToTensor())
        return DataLoader(test, batch_size=256, shuffle=False, num_workers=2, pin_memory=False), 'cifar10'
    test = MNIST(root='./data', train=False, download=True, transform=T.ToTensor())
    return DataLoader(test, batch_size=256, shuffle=False, num_workers=2, pin_memory=False), 'mnist'


def _make_model(model_hint: Optional[str], dataset_kind: str):
    if model_hint:
        h = model_hint.lower()
        if 'cifar' in h or 'resnet' in h:
            return SimpleCIFAR10()
        if 'linear' in h or 'mnist' in h:
            return LinearMNIST()
    return LinearMNIST() if dataset_kind == 'mnist' else SimpleCIFAR10()
    # return model_hint


def _apply_update(base_state: dict, update: dict, update_type: str):
    """
    Reconstruct realized local model:
      - update_type == 'delta': realized = base + delta
      - otherwise: realized = update (treated as full state)
    """
    if update_type == 'delta':
        realized = {}
        for k, v in base_state.items():
            realized[k] = v + update.get(k, torch.zeros_like(v))
        for k, v in update.items():
            if k not in realized:
                realized[k] = v
        return realized
    return update

def reconstruct_state(base_state: dict, update_obj: dict, update_type: str = "delta") -> dict:
    """
    按客户端逻辑重构全量本地模型：
      - update_type == 'delta': realized = base + delta
      - update_type == 'state': realized = update_obj
    对缺键/新增键做了容错处理。
    """
    ut = (update_type or "delta").lower()
    if ut == "state":
        return update_obj

    # 默认 delta：realized = base + delta
    realized = {}
    # 先把 base 的键都放入
    for k, bv in base_state.items():
        dv = update_obj.get(k, None)
        if torch.is_tensor(bv) and torch.is_tensor(dv) and bv.shape == dv.shape:
            realized[k] = bv + dv
        elif torch.is_tensor(bv) and dv is None:
            realized[k] = bv.clone()
        else:
            # 非 tensor 或 shape 不匹配，保留 base
            realized[k] = bv

    # 处理 delta 里“新增的键”
    for k, dv in update_obj.items():
        if k not in realized:
            realized[k] = dv if torch.is_tensor(dv) else dv

    return realized
# ---------------- public APIs ----------------

@torch.no_grad()
def evaluate_global(cidr: str, ipfs) -> float:
    test_set = MNIST(root="./data", train=False, download=True, transform=T.ToTensor())
    loader = DataLoader(test_set, batch_size=256, shuffle=False)
    model = LinearMNIST()
    model.load_state_dict(ipfs.load(cidr))
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        out = model(xb)
        _, pred = out.max(1)
        total += yb.size(0)
        correct += int((pred == yb).sum())
    return correct / max(1, total)


@torch.no_grad()
def evaluate_state_dict(
    state_dict: dict,
    *,
    dataset: str = 'mnist',
    model_hint: Optional[str] = None,
    device: str = 'cpu',
    base_state: Optional[dict] = None,
    update_type: str = 'delta',   # 'delta' | 'state'
) -> float:
    """
    Evaluate a single model state or delta.
    If base_state is provided and update_type == 'delta', reconstruct realized = base + delta before evaluating.
    """
    loader, kind = _make_dataset(dataset)
    model = _make_model(model_hint, kind).to(device)

    realized = _apply_update(base_state, state_dict, update_type) if (base_state is not None) else state_dict
    model.load_state_dict(realized, strict=False)
    model.eval()

    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        _, pred = out.max(1)
        total += yb.size(0)
        correct += int((pred == yb).sum())
    return correct / max(1, total)


@torch.no_grad()
def evaluate_many_state_dicts(
    state_dicts: Sequence[dict],
    *,
    dataset: str = 'mnist',
    model_hint: Optional[str] = None,
    device: str = 'cpu',
    max_workers: int = 4,
    base_state: Optional[dict] = None,
    update_type: str = 'state',   # 'delta' | 'state'
) -> List[float]:
    """
    Parallel evaluation of many updates.
    If base_state is given and update_type == 'delta', each update is realized as (base + delta) before evaluation.
    """
    def _one(sd):
        try:
            return float(
                evaluate_state_dict(
                    sd,
                    dataset=dataset,
                    model_hint=model_hint,
                    device=device,
                    base_state=base_state,
                    update_type=update_type,
                )
            )
        except Exception:
            return float('nan')

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, state_dicts))


# -------- new: explicit helpers for reconstructed (base + delta) evaluation --------

@torch.no_grad()
def evaluate_reconstructed_single(
    *,
    base_state: dict,
    update_delta: dict,
    dataset: str = "mnist",
    model_hint: Optional[str] = None,
    device: str = "cpu",
) -> float:
    """Realize local model as (base + delta) and return evaluated accuracy."""
    return evaluate_state_dict(
        update_delta,
        dataset=dataset,
        model_hint=model_hint,
        device=device,
        base_state=base_state,
        update_type="delta",
    )


@torch.no_grad()
def evaluate_reconstructed_batch(
    *,
    base_state: dict,
    deltas: Sequence[dict],
    dataset: str = "mnist",
    model_hint: Optional[str] = None,
    device: str = "cpu",
    max_workers: int = 4,
) -> List[float]:
    """Batch version of evaluate_reconstructed_single."""
    return evaluate_many_state_dicts(
        deltas,
        dataset=dataset,
        model_hint=model_hint,
        device=device,
        max_workers=max_workers,
        base_state=base_state,
        update_type="delta",
    )
