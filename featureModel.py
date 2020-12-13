import torch.nn.functional as F
import torch.nn as nn
import torch


# two layer MLP to get the consistency score for each pair of patches
class FeatureModel(nn.Module):
    def __init__(self, exif_dim):
        super(FeatureModel, self).__init__()
        self.layer1 = nn.Linear(exif_dim, exif_dim * 2)
        self.layer2 = nn.Linear(exif_dim * 2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.dropout(x, 0.5)
        x = F.relu(x)
        x = self.layer2(x)

        return torch.sigmoid(x)
