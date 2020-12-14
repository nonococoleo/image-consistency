import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyModel(nn.Module):
    """
    Consistency Model
    """

    def __init__(self, feature_dim):
        super(ConsistencyModel, self).__init__()
        self.layer1 = nn.Linear(feature_dim, feature_dim * 2)
        self.layer2 = nn.Linear(feature_dim * 2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.dropout(x, 0.5)
        x = F.relu(x)
        x = self.layer2(x)

        return torch.sigmoid(x)
