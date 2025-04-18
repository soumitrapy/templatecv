import torch
import torch.nn as nn
import torch.nn.functional as F

class DefaultModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self,x):
        x = self.conv(x)
        return torch.flatten(x, start_dim=1)