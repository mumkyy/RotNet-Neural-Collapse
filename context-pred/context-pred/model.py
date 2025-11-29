import torch
import torch.nn as nn

class AlexNetwork(nn.Module):
    def __init__(self, num_classes=8, patch_dim=32, aux_logits=False,out_feat_keys=None, **kwargs):
        super(AlexNetwork, self).__init__()

        self.patch_dim = patch_dim

        if out_feat_keys is None:
            out_feat_keys = ['conv1','conv2','conv3','conv4','conv5','conv6','conv7','fc6']
        self.out_feat_keys = set(out_feat_keys)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32,96),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.GroupNorm(96,384),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(96, 384),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(96, 384),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 16x16 -> 8x8

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(96, 384),
        )
            
        self.conv6 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(96, 384),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(64, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        
        # compute spatial size after 3x 2x2 pools
        S_out = max(1, self.patch_dim // 8)   # floor(P/8)

        in_dim_fc6 = 256 * S_out * S_out

        self.fc6 = nn.Sequential(
            nn.Linear(in_dim_fc6, 4096),
            nn.ReLU(inplace=True),
            nn.LayerNorm(4096),
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.cnn = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7
        )

    def extract_features(self, x, out_keys=None):
        """
        Run the backbone and return a dict of named features.
        out_keys: list like ['conv1','conv4','fc6'] or None for default.
        """
        if out_keys is None:
            out_keys = self.out_feat_keys
        else:
            out_keys = set(out_keys)

        feats = {}

        x = self.conv1(x)
        if 'conv1' in out_keys:
            feats['conv1'] = x

        x = self.conv2(x)
        if 'conv2' in out_keys:
            feats['conv2'] = x

        x = self.conv3(x)
        if 'conv3' in out_keys:
            feats['conv3'] = x

        x = self.conv4(x)
        if 'conv4' in out_keys:
            feats['conv4'] = x

        x = self.conv5(x)
        if 'conv5' in out_keys:
            feats['conv5'] = x

        x = self.conv6(x)
        if 'conv6' in out_keys:
            feats['conv6'] = x

        x = self.conv7(x)
        if 'conv7' in out_keys:
            feats['conv7'] = x

        # fc6 on top
        if 'fc6' in out_keys:
            flat = x.view(x.size(0), -1)
            feats['fc6'] = self.fc6(flat)
        return feats


    def forward_once(self, x):
        feats = self.extract_features(x, out_keys=['fc6'])
        return feats['fc6']

    def forward(self, uniform_patch, random_patch):
        output_fc6_uniform = self.forward_once(uniform_patch)
        output_fc6_random = self.forward_once(random_patch)
        output = torch.cat((output_fc6_uniform, output_fc6_random), 1)
        output = self.fc(output)
        return output, output_fc6_uniform, output_fc6_random
    

class AlexClassifier(nn.Module):
    """
    Supervised classifier on top of a pretrained AlexNetwork backbone.
    - head_feat_key: which layer to use as features.
    - input_size: spatial size S of the input (assumed square SxS).
    """
    def __init__(self,
                 num_classes=10,
                 backbone_ckpt=None,
                 freeze_backbone=True,
                 head_feat_key="fc6",
                 input_size=32,
                 backbone_num_classes=8,
                 backbone_patch_dim=32):
        super().__init__()

        assert head_feat_key in ["conv1","conv2","conv3","conv4","conv5","conv6","conv7","fc6"]
        self.head_feat_key = head_feat_key
        self.input_size = input_size

        self.backbone = AlexNetwork(num_classes=backbone_num_classes,patch_dim=backbone_patch_dim)

        if backbone_ckpt is not None:
            ckpt = torch.load(backbone_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            self.backbone.load_state_dict(state, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # compute feature dim analytically from input_size and layer
        S = input_size
        if head_feat_key == "fc6":
            # fc6 is only defined for final 4x4 → S must be 32
            assert S == 32, "fc6 head only valid for 32x32 inputs in this architecture."
            in_dim = 4096
        elif head_feat_key == "conv7":
            S7 = S // 8        # after 3 pools
            in_dim = 256 * S7 * S7
        elif head_feat_key in ["conv4", "conv5", "conv6"]:
            S4 = S // 4        # after 2 pools
            in_dim = 384 * S4 * S4
        elif head_feat_key in ["conv2", "conv3"]:
            S2 = S // 2        # after 1 pool
            in_dim = 384 * S2 * S2
        else:  # conv1
            in_dim = 96 * S * S

        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        feats = self.backbone.extract_features(x, out_keys=[self.head_feat_key])
        h = feats[self.head_feat_key]
        if h.dim() > 2:
            h = h.flatten(1)
        return self.head(h)
