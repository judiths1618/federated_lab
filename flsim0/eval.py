import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST
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


@torch.no_grad()
def evaluate_state_dict(state_dict) -> float:
    test_set = MNIST(root="./data", train=False, download=True, transform=T.ToTensor())
    loader = DataLoader(test_set, batch_size=256, shuffle=False)
    model = LinearMNIST()
    model.load_state_dict(state_dict)
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        out = model(xb); _, pred = out.max(1)
        total += yb.size(0); correct += int((pred == yb).sum())
    return correct/max(1,total)
