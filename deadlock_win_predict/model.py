"""Small MLP + hero embeddings for team-0 win prediction."""

from __future__ import annotations

import torch
from torch import nn


class WinPredictor(nn.Module):
    """
    Concatenate normalized aggregate features with mean-pooled hero embeddings.

    Designed for M4 Pro / MPS: ~0.3–1M params at default sizes.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_heroes: int = 128,
        hero_emb_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hero_emb = nn.Embedding(num_heroes, hero_emb_dim, padding_idx=0)
        in_dim = feature_dim + hero_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor, hero_ids: torch.Tensor) -> torch.Tensor:
        # hero_ids: (B, 12) — clamp unknowns into embedding table
        hero_ids = hero_ids.clamp(min=0, max=self.hero_emb.num_embeddings - 1)
        pooled = self.hero_emb(hero_ids).mean(dim=1)
        x = torch.cat([features, pooled], dim=-1)
        return self.mlp(x).squeeze(-1)
