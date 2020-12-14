import torch
import torch.nn as nn


class CompareExifModel(nn.Module):
    """
    Comparison model
    """

    def __init__(self, encoder, n_features, projection_dim):
        super(CompareExifModel, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        self.projector = nn.Sequential(
            nn.Linear(2 * self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, 40 * projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(40 * projection_dim, 10 * projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(10 * projection_dim, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        c = torch.cat((h_i, h_j), 1)
        z = self.projector(c)
        return torch.sigmoid(z)
