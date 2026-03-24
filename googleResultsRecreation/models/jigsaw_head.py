import torch
import torch.nn as nn
import torch.nn.functional as F

class JigsawHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 4096, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(4096, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x.mean(dim=(2,3))  # global average → [B, num_classes]