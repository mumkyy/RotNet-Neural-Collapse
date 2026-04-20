import torch
import torch.nn as nn
import torch.nn.functional as F

class DownstreamHeadSmall(nn.Module):
    """
    3-conv downstream classifier head for frozen spatial features [B, C, H, W].

    Design goals:
    - stable with small batch sizes: GroupNorm instead of BatchNorm
    - bounded compute across different hook layers: adaptive pool to fixed size
    - deep enough to test learnability, but not absurd for Imagenette-160
    """

    def __init__(self, in_channels, num_classes, pooled_size=8, dropout=0.10):
        super().__init__()
        self.pooled_size = pooled_size

        # 3 conv layers total:
        # 1x1 stem
        # 3x3 body
        # 1x1 classifier
        self.conv1 = nn.Conv2d(in_channels, 192, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, 192)

        self.conv2 = nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, 96)

        self.conv3 = nn.Conv2d(96, num_classes, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout)

    def _block(self, x, conv, norm, use_dropout=False):
        x = conv(x)
        x = norm(x)
        x = self.relu(x)
        if use_dropout:
            x = self.drop(x)
        return x

    def forward(self, x):
        # normalize spatial size so early and late hooked layers are both feasible
        x = F.adaptive_avg_pool2d(x, (self.pooled_size, self.pooled_size))

        x = self._block(x, self.conv1, self.gn1)
        x = self._block(x, self.conv2, self.gn2, use_dropout=True)

        x = self.conv3(x)
        return x.mean(dim=(2, 3))  # [B, num_classes]