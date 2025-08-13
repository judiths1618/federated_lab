import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from typing import Optional, Sequence
import concurrent.futures

from .models import LinearMNIST

class SimpleCIFAR10(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((4,4)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128*4*4, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


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
        if 'cifar' in h or 'resnet' in h: return SimpleCIFAR10()
        if 'linear' in h or 'mnist' in h: return LinearMNIST()
    return LinearMNIST() if dataset_kind == 'mnist' else SimpleCIFAR10()


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

@torch.no_grad()
def evaluate_state_dict(state_dict, *, dataset: str = 'mnist', model_hint: Optional[str] = None, device: str = 'cpu') -> float:
    loader, kind = _make_dataset(dataset)
    model = _make_model(model_hint, kind).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb); _, pred = out.max(1)
        total += yb.size(0); correct += int((pred == yb).sum())
    return correct / max(1, total)

@torch.no_grad()
def evaluate_many_state_dicts(state_dicts: Sequence[dict], *, dataset: str = 'mnist', model_hint: Optional[str] = None, device: str = 'cpu', max_workers: int = 4) -> list[float]:
    def _one(sd):
        try: return float(evaluate_state_dict(sd, dataset=dataset, model_hint=model_hint, device=device))
        except Exception: return float('nan')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, state_dicts))
