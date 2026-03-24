import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(x, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))


class BasicResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        stride=1,
        activation_fn=nn.ReLU,
        normalization=nn.BatchNorm2d,
    ):
        super().__init__()

        self.activation = activation_fn()

        # Shortcut path
        if stride > 1 or filters != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                normalization(filters),
            )
        else:
            self.shortcut = nn.Identity()

        # Main path
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters // 4,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = normalization(filters // 4)

        self.conv2 = nn.Conv2d(
            in_channels=filters // 4,
            out_channels=filters // 4,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=0,  
        )
        self.bn2 = normalization(filters // 4)

        self.conv3 = nn.Conv2d(
            in_channels=filters // 4,
            out_channels=filters,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = normalization(filters)

    def forward(self, x):
        x_shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = fixed_padding(x, kernel_size=3)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + x_shortcut
        x = self.activation(x)

        return x