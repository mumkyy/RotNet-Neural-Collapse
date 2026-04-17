import torch
import torch.nn as nn

class DownstreamHeadLinear(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x