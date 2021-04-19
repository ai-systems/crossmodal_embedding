import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNet(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.5):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        # x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x
