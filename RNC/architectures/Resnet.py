import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Standard ResNet Basic Block
# We keep this strictly modular so the 'forward' loop in the main class
# captures the output AFTER the residual addition.
class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(ResNetBasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

class ResNet34_NIN_Style(nn.Module):
    def __init__(self, opt):
        super(ResNet34_NIN_Style, self).__init__()

        num_classes = opt['num_classes']
        
        # ResNet-34 Configurations
        # We treat the Stem as Stage 1, and the 4 ResNet layers as Stages 2-5
        # This creates 5 distinct "convX" containers in your feature map.
        self.in_planes = 64
        block_counts = [3, 4, 6, 3] # Standard ResNet34 depths
        
        # Initialize lists for your custom block structure
        # +2 for: Stem (1) + ResNetLayers (4) + Classifier (1 extra appended later)
        blocks = [nn.Sequential() for _ in range(5)]

        # --- Stage 1: The Stem (conv1) ---
        # Note: We 7x7group these into one sequential so 'conv1' keys refer to the stem output
        blocks[0].add_module('Stem_Conv', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        blocks[0].add_module('Stem_BN', nn.BatchNorm2d(64))
        blocks[0].add_module('Stem_ReLU', nn.ReLU(inplace=True))
        blocks[0].add_module('Stem_MaxPool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # --- Stages 2-5: The ResNet Layers (conv2, conv3, conv4, conv5) ---
        planes_list = [64, 128, 256, 512]
        strides_list = [1, 2, 2, 2]

        for stage_idx, (num_blocks, planes, stride) in enumerate(zip(block_counts, planes_list, strides_list)):
            # We map ResNet layers to blocks[1] through blocks[4]
            current_block_idx = stage_idx + 1 
            
            strides = [stride] + [1]*(num_blocks-1)
            for i, s in enumerate(strides):
                # We name them 'block0', 'block1' etc.
                # Your keys will look like: 'conv2.block0', 'conv3.block5'
                blocks[current_block_idx].add_module(f'block{i}', ResNetBasicBlock(self.in_planes, planes, s))
                self.in_planes = planes * ResNetBasicBlock.expansion

        # --- Final Classifier Block ---
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling', GlobalAveragePooling())
        blocks[-1].add_module('Classifier', nn.Linear(512 * ResNetBasicBlock.expansion, num_classes))

        # --- EXACT Copy of your NIN Feature Extraction Logic ---
        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = []
        
        # Naming logic: conv1 (stem), conv2..5 (layers)
        num_stages = len(self._feature_blocks) - 1 # Should be 5
        for s in range(num_stages):
            for child_name, _ in blocks[s].named_children():
                self.all_feat_names.append(f'conv{s+1}.{child_name}')
            self.all_feat_names.append(f'conv{s+1}')
        
        self.all_feat_names.append('penult')
        self.all_feat_names.append('classifier')

        self._feature_name_map = {
            "classifier": self._feature_blocks[-1].Classifier, 
            "penult" : self._feature_blocks[-1].GlobalAveragePooling
        }
        
        for i, block in enumerate(self._feature_blocks[:-1]):
            self._feature_name_map[f"conv{i+1}"] = block
            self._feature_name_map.update({f"conv{i+1}.{n}": m for n, m in block.named_children()})

    def _parse_out_keys_arg(self, out_feat_keys):
        out_feat_keys = [self.all_feat_names[-1]] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(f'Feature with name {key} does not exist. Existing features: {self.all_feat_names}.')
            elif key in out_feat_keys[:f]:
                raise ValueError(f'Duplicate output feature key: {key}.')
        return out_feat_keys

    def forward(self, x, out_feat_keys=None):
        out_feat_keys = self._parse_out_keys_arg(out_feat_keys)
        out_index = {key: i for i, key in enumerate(out_feat_keys)}
        out_feats = [None] * len(out_feat_keys)

        feat = x
        num_stages = len(self._feature_blocks) - 1

        for s in range(num_stages):
            for child_name, child in self._feature_blocks[s].named_children():
                feat = child(feat)
                key = f'conv{s+1}.{child_name}'
                idx = out_index.get(key)
                if idx is not None:
                    out_feats[idx] = feat
            
            # Capture end of stage output
            key = f'conv{s+1}'
            idx = out_index.get(key)
            if idx is not None:
                out_feats[idx] = feat
        
        gap = self._feature_blocks[-1].GlobalAveragePooling
        fc  = self._feature_blocks[-1].Classifier

        pen = gap(feat)
        idx = out_index.get('penult')
        if idx is not None:
            out_feats[idx] = pen

        logits = fc(pen)
        idx = out_index.get('classifier')
        if idx is not None:
            out_feats[idx] = logits

        return out_feats[0] if len(out_feats) == 1 else out_feats

def create_model(opt):
    return ResNet34_NIN_Style(opt)

# --- Verification Block ---
if __name__ == '__main__':
    # 1. Setup
    opt = {'num_classes': 10}
    net = create_model(opt)
    
    # 2. Print available keys to ensure they match your expectation
    print(f"Total defined keys: {len(net.all_feat_names)}")
    # Example keys you will see:
    # 'conv1.Stem_Conv' (Stem)
    # 'conv2.block0'    (First ResNet block of Layer 1)
    # 'conv5.block2'    (Last ResNet block of Layer 4)
    print("Sample keys:", net.all_feat_names[:5], "...", net.all_feat_names[-5:])

    # 3. Dummy Forward Pass
    x = torch.randn(1, 3, 160, 160) # Imagenette size
    
    # Test grabbing specific ResNet blocks
    target_keys = ['conv2.block0', 'conv5.block2', 'classifier']
    outputs = net(x, out_feat_keys=target_keys)
    
    for k, o in zip(target_keys, outputs):
        print(f"Key: {k} | Output Shape: {o.shape}")