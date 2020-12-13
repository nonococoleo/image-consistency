import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictExifModel(nn.Module):
    def __init__(self, encoder, n_features, middle_dim, labels, partitions):
        super(PredictExifModel, self).__init__()

        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(n_features, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, labels * partitions),
        )
        self.labels = labels
        self.partitions = partitions

    def forward(self, x):
        x = self.encoder(x)
        x = F.dropout(x, 0.5)
        x = self.projector(x)

        return x.view(x.size(0), self.partitions, self.labels)
