import torch
import torch.nn as nn

class JigsawHeadLinear(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.3):
        super().__init__()

        h1 = max(1024, in_dim)
        h2 = max(768, in_dim // 2)
        h3 = max(512, in_dim // 4)
        h4 = max(256, in_dim // 8)
        h5 = max(128, in_dim // 16)
        h6 = max(64,  in_dim // 32)

        self.lin1 = nn.Linear(in_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.lin2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.lin3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.lin4 = nn.Linear(h3, h4)
        self.bn4 = nn.BatchNorm1d(h4)
        self.lin5 = nn.Linear(h4, h5)
        self.bn5 = nn.BatchNorm1d(h5)
        self.lin6 = nn.Linear(h5, h6)
        self.bn6 = nn.BatchNorm1d(h6)
        self.lin7 = nn.Linear(h6, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def block(self, x, lin, bn, use_dropout=False):
        x = lin(x)
        x = bn(x)
        x = self.relu(x)
        if use_dropout:
            x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.block(x, self.lin1, self.bn1)
        x = self.block(x, self.lin2, self.bn2)
        x = self.block(x, self.lin3, self.bn3)
        x = self.block(x, self.lin4, self.bn4, use_dropout=True)
        x = self.block(x, self.lin5, self.bn5)
        x = self.block(x, self.lin6, self.bn6, use_dropout=True)
        x = self.lin7(x)
        return x