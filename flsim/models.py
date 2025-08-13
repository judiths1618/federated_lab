import torch
import torch.nn as nn

class LinearMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))
