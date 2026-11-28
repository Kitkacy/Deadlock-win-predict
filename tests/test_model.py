"""Offline tests for the baseline model."""

from __future__ import annotations

import torch

from deadlock_win_predict.model import WinPredictor


def test_win_predictor_forward_shape():
    model = WinPredictor(feature_dim=112, num_heroes=64, hero_emb_dim=16, hidden=64)
    features = torch.randn(4, 112)
    hero_ids = torch.randint(0, 64, (4, 12))
    logits = model(features, hero_ids)
    assert logits.shape == (4,)
