import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(x, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))


class BottleneckV1(nn.Module):
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
            padding=0,
            bias=False,
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


class BottleneckV2(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        stride=1,
        activation_fn=nn.ReLU,
        normalization=nn.BatchNorm2d,
        no_shortcut=False,
    ):
        super().__init__()
        self.no_shortcut = no_shortcut
        self.activation = activation_fn()

        self.bn0 = normalization(in_channels)

        if stride > 1 or filters != in_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

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
            padding=0,
            bias=False,
        )

        self.bn2 = normalization(filters // 4)

        self.conv3 = nn.Conv2d(
            in_channels=filters // 4,
            out_channels=filters,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        identity = x

        x = self.bn0(x)
        x = self.activation(x)

        if isinstance(self.shortcut, nn.Identity):
            x_shortcut = identity
        else:
            x_shortcut = self.shortcut(x)

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.activation(x)
        x = fixed_padding(x, kernel_size=3)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)

        if self.no_shortcut:
            return x
        return x + x_shortcut


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_layers=(3, 4, 6, 3),
        strides=(2, 2, 2),
        num_classes=1000,
        filters_factor=4,
        include_root_block=True,
        root_conv_size=7,
        root_conv_stride=2,
        root_pool_size=3,
        root_pool_stride=2,
        activation_fn=nn.ReLU,
        last_relu=True,
        normalization=nn.BatchNorm2d,
        global_pool=True,
        mode="v2",
    ):
        super().__init__()

        if mode not in ("v1", "v2"):
            raise ValueError(f"Unknown Resnet mode: {mode}")

        if mode == "v1" and not last_relu:
            raise ValueError("last_relu is always True (implicitly) in the v1 mode.")

        self.mode = mode
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.last_relu = last_relu
        self.activation = activation_fn()

        unit = BottleneckV2 if mode == "v2" else BottleneckV1

        filters = 16 * filters_factor
        current_channels = in_channels

        self.include_root_block = include_root_block
        if include_root_block:
            self.root_conv_size = root_conv_size
            self.root_pool_size = root_pool_size

            self.root_conv = nn.Conv2d(
                in_channels=current_channels,
                out_channels=filters,
                kernel_size=root_conv_size,
                stride=root_conv_stride,
                padding=0, 
                bias=False,
            )
            current_channels = filters

            if mode == "v1":
                self.root_bn = normalization(filters)

            self.root_pool = nn.MaxPool2d(
                kernel_size=root_pool_size,
                stride=root_pool_stride,
                padding=0,   
            )

        
        num_layers = list(num_layers)
        stage_strides = list(strides)

        filters *= 4  # first bottleneck stage output channels

        self.block1, current_channels = self._make_stage(
            unit, current_channels, filters, num_layers[0], stride=1,
            activation_fn=activation_fn, normalization=normalization
        )

        filters *= 2
        self.block2, current_channels = self._make_stage(
            unit, current_channels, filters, num_layers[1], stride=stage_strides[0],
            activation_fn=activation_fn, normalization=normalization
        )

        filters *= 2
        self.block3, current_channels = self._make_stage(
            unit, current_channels, filters, num_layers[2], stride=stage_strides[1],
            activation_fn=activation_fn, normalization=normalization
        )

        filters *= 2
        self.block4, current_channels = self._make_stage(
            unit, current_channels, filters, num_layers[3], stride=stage_strides[2],
            activation_fn=activation_fn, normalization=normalization
        )

        if mode == "v2":
            self.postnorm = normalization(current_channels)

        if num_classes:
            self.classifier = nn.Conv2d(
                in_channels=current_channels,
                out_channels=num_classes,
                kernel_size=1,
                bias=True,
            )
        else:
            self.classifier = None

    def _make_stage(
        self,
        unit,
        in_channels,
        filters,
        num_units,
        stride,
        activation_fn,
        normalization,
    ):
        layers = []
        layers.append(
            unit(
                in_channels=in_channels,
                filters=filters,
                stride=stride,
                activation_fn=activation_fn,
                normalization=normalization,
            )
        )
        for _ in range(num_units - 1):
            layers.append(
                unit(
                    in_channels=filters,
                    filters=filters,
                    stride=1,
                    activation_fn=activation_fn,
                    normalization=normalization,
                )
            )
        return nn.Sequential(*layers), filters

    def forward(self, x, return_endpoints=False):
        end_points = {}

        if self.include_root_block:
            x = fixed_padding(x, kernel_size=self.root_conv_size)
            x = self.root_conv(x)

            if self.mode == "v1":
                x = self.root_bn(x)
                x = self.activation(x)

            x = fixed_padding(x, kernel_size=self.root_pool_size)
            x = self.root_pool(x)
            end_points["after_root"] = x

        x = self.block1(x)
        end_points["block1"] = x

        x = self.block2(x)
        end_points["block2"] = x

        x = self.block3(x)
        end_points["block3"] = x

        x = self.block4(x)
        end_points["block4"] = x

        if self.mode == "v2":
            x = self.postnorm(x)
            if self.last_relu:
                x = self.activation(x)

        if self.global_pool:
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            end_points["pre_logits"] = x.squeeze(-1).squeeze(-1)
        else:
            end_points["pre_logits"] = x

        if self.classifier is not None:
            logits = self.classifier(x)
            if self.global_pool:
                logits = logits.squeeze(-1).squeeze(-1)
            end_points["logits"] = logits
            if return_endpoints:
                return logits, end_points
            return logits
        else:
            if return_endpoints:
                return end_points["pre_logits"], end_points
            return end_points["pre_logits"]


def resnet50(
    in_channels=3,
    num_classes=1000,
    mode="v2",
    **kwargs,
):
    return ResNet(
        in_channels=in_channels,
        num_layers=(3, 4, 6, 3),
        num_classes=num_classes,
        mode=mode,
        **kwargs,
    )