"""Train WinPredictor on processed parquet splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import MatchSnapshotDataset
from .model import WinPredictor

logger = logging.getLogger(__name__)


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for batch in loader:
        features = batch["features"].to(device)
        hero_ids = batch["hero_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(features, hero_ids)
        loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
        correct += int((preds == labels).sum().item())
        probs_all.append(probs.detach().cpu())
        labels_all.append(labels.detach().cpu())
    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "auc": 0.5}
    y = torch.cat(labels_all)
    p = torch.cat(probs_all)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "auc": _auc(y, p),
    }


def _auc(y: torch.Tensor, p: torch.Tensor) -> float:
    """Trapezoidal ROC-AUC without sklearn."""
    y = y.double()
    p = p.double()
    if y.min() == y.max():
        return 0.5
    desc = torch.argsort(p, descending=True)
    y_sorted = y[desc]
    tps = torch.cumsum(y_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_sorted, dim=0)
    tps = torch.cat([torch.tensor([0.0], dtype=torch.double), tps])
    fps = torch.cat([torch.tensor([0.0], dtype=torch.double), fps])
    tpr = tps / tps[-1].clamp_min(1e-12)
    fpr = fps / fps[-1].clamp_min(1e-12)
    return float(torch.trapz(tpr, fpr).item())


def train_baseline(
    processed_dir: Path,
    *,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    snapshot_s: int | None = None,
    device_name: str = "auto",
    num_heroes: int = 128,
    hero_emb_dim: int = 32,
    hidden: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = pick_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MatchSnapshotDataset(
        processed_dir / "train.parquet",
        processed_dir / "normalization.json",
        snapshot_s=snapshot_s,
    )
    val_path = processed_dir / "val.parquet"
    val_ds = MatchSnapshotDataset(
        val_path,
        processed_dir / "normalization.json",
        snapshot_s=snapshot_s,
    )
    if len(train_ds) == 0:
        raise ValueError("train split is empty")

    feature_dim = int(train_ds[0]["features"].shape[0])
    model = WinPredictor(
        feature_dim,
        num_heroes=num_heroes,
        hero_emb_dim=hero_emb_dim,
        hidden=hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            hero_ids = batch["hero_ids"].to(device)
            labels = batch["label"].to(device)
            # Side-swap augmentation: flip team aggregates is hard post-hoc;
            # approximate by randomly inverting labels 50% with feature sign-flip
            # on the precomputed diff block is unsafe without layout knowledge.
            # Keep simple: no aug here; document team-swap for a later pass.
            opt.zero_grad(set_to_none=True)
            logits = model(features, hero_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(labels)
            n += len(labels)
        train_loss = running / max(n, 1)
        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None and len(val_ds) > 0:
            val_metrics = evaluate(model, val_loader, device)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["loss"] <= best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "feature_dim": feature_dim,
                        "num_heroes": num_heroes,
                        "hero_emb_dim": hero_emb_dim,
                        "hidden": hidden,
                        "snapshot_s": snapshot_s,
                        "epoch": epoch,
                        "val": val_metrics,
                    },
                    best_path,
                )
        else:
            torch.save(
                {
                    "model": model.state_dict(),
                    "feature_dim": feature_dim,
                    "num_heroes": num_heroes,
                    "hero_emb_dim": hero_emb_dim,
                    "hidden": hidden,
                    "snapshot_s": snapshot_s,
                    "epoch": epoch,
                },
                best_path,
            )
        history.append(metrics)
        logger.info("epoch %s %s", epoch, metrics)

    summary = {
        "device": str(device),
        "feature_dim": feature_dim,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "best_checkpoint": str(best_path),
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
