import torch
import torch.nn as nn
import torch.nn.functional as F

class DownstreamHead(nn.Module):
    """
    9-conv downstream classifier head for frozen spatial features [B, C, H, W].

    Design goals:
    - stable with small batch sizes: GroupNorm instead of BatchNorm
    - bounded compute across different hook layers: adaptive pool to fixed size
    - deep enough to test learnability, but not absurd for Imagenette-160
    """

    def __init__(self, in_channels, num_classes, pooled_size=8, dropout=0.10):
        super().__init__()
        self.pooled_size = pooled_size

        # 9 conv layers total:
        # 1 stem 1x1
        # 7 body 3x3
        # 1 classifier 1x1
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, 256)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, 256)

        self.conv3 = nn.Conv2d(256, 192, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, 192)

        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False)
        self.gn4 = nn.GroupNorm(32, 192)

        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False)
        self.gn5 = nn.GroupNorm(32, 128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.gn6 = nn.GroupNorm(32, 128)

        self.conv7 = nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False)
        self.gn7 = nn.GroupNorm(32, 96)

        self.conv8 = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False)
        self.gn8 = nn.GroupNorm(32, 96)

        self.conv9 = nn.Conv2d(96, num_classes, kernel_size=1, bias=True)

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
        x = self._block(x, self.conv2, self.gn2)
        x = self._block(x, self.conv3, self.gn3)
        x = self._block(x, self.conv4, self.gn4, use_dropout=True)
        x = self._block(x, self.conv5, self.gn5)
        x = self._block(x, self.conv6, self.gn6)
        x = self._block(x, self.conv7, self.gn7, use_dropout=True)
        x = self._block(x, self.conv8, self.gn8)

        x = self.conv9(x)
        return x.mean(dim=(2, 3))  # [B, num_classes]