# ==============================================================================
# INDS BENCHMARK MODULE: BASELINE MODELS FOR SCIENTIFIC FAIRNESS
# AUTHOR: S.Wattanasuphinyakit Author
# COMPONENTS: Standard Neural ODE, PatchTST (Transformer-based Baseline)
# ==============================================================================

import torch
import torch.nn as nn
from .inds_core import NeuralODEVectorField

class BaselineNeuralODE(nn.Module):
    """
    Standard Neural Ordinary Differential Equation
    """

    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, latent_dim)
        self.vf = NeuralODEVectorField(latent_dim, seed_offset=12)
        self.decoder = nn.Linear(latent_dim, in_dim)

    def forward(self, x, steps=10, dt=0.1):
        z = self.encoder(x)
        for _ in range(steps):
            z = z + self.vf(z) * dt
        return self.decoder(z), torch.tensor(0.0, device=x.device)


class PatchTSTBaseline(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.patch_layer = nn.Linear(in_dim, 128)
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )
        self.head = nn.Linear(128, in_dim)

    def forward(self, x, steps=10):
        features = torch.relu(self.patch_layer(x))
        if features.dim() == 2:
            features = features.unsqueeze(1)
        out = self.transformer_block(features)
        return self.head(out.squeeze(1)), torch.tensor(0.0, device=x.device)