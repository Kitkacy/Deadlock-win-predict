"""Build parquet datasets, splits, normalization, and a torch Dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import DEFAULT_SNAPSHOT_SECONDS, extract_match_rows, feature_names
from .fetch import is_cached, read_raw_match

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional until train time
    torch = None  # type: ignore
    Dataset = object  # type: ignore


def build_rows_from_raw(
    raw_dir: Path,
    match_ids: list[int],
    snapshots_s: tuple[int, ...] | list[int] = DEFAULT_SNAPSHOT_SECONDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing = 0
    for mid in match_ids:
        if not is_cached(raw_dir, mid):
            missing += 1
            continue
        match = read_raw_match(raw_dir, mid)
        rows.extend(extract_match_rows(match, snapshots_s=snapshots_s))
    if missing:
        logger.warning("Skipped %s match ids with no raw cache", missing)
    logger.info("Built %s snapshot rows from %s matches", len(rows), len(match_ids) - missing)
    return rows


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        records.append(
            {
                "match_id": r["match_id"],
                "snapshot_s": r["snapshot_s"],
                "label_team0_wins": r["label_team0_wins"],
                "duration_s": r["duration_s"],
                "start_time": r["start_time"],
                "game_mode": r["game_mode"],
                "match_mode": r["match_mode"],
                "features": r["features"].tolist(),
                "hero_ids": r["hero_ids"].tolist(),
                "assigned_lanes": r["assigned_lanes"].tolist(),
                "per_player_stats": r["per_player_stats"].tolist(),
                "item_counts": r["item_counts"].tolist(),
            }
        )
    return pd.DataFrame.from_records(records)


def split_match_ids(
    match_ids: list[int],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    ids = sorted(set(int(m) for m in match_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": ids, "val": [], "test": []}
    if n == 2:
        return {"train": [ids[0]], "val": [], "test": [ids[1]]}
    # For tiny sets, keep at least one match in val and test when possible.
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * (1.0 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = 1
        n_test = max(0, n - 2)
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def compute_normalization(train_df: pd.DataFrame) -> dict[str, Any]:
    feats = np.stack(train_df["features"].to_numpy())
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "feature_names": feature_names(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "n_train_rows": int(len(train_df)),
        "n_train_matches": int(train_df["match_id"].nunique()),
    }


def write_dataset(
    processed_dir: Path,
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = rows_to_frame(rows)
    if df.empty:
        raise ValueError("No feature rows to write")

    splits = split_match_ids(df["match_id"].unique().tolist(), seed=seed)
    (processed_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    paths: dict[str, Path] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, ids in splits.items():
        id_set = set(ids)
        part = df[df["match_id"].isin(id_set)].reset_index(drop=True)
        frames[name] = part
        path = processed_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        paths[name] = path
        logger.info("%s: %s rows, %s matches -> %s", name, len(part), part["match_id"].nunique(), path)

    norm = compute_normalization(frames["train"])
    norm_path = processed_dir / "normalization.json"
    norm_path.write_text(json.dumps(norm, indent=2))
    paths["normalization"] = norm_path

    meta = {
        "n_rows": int(len(df)),
        "n_matches": int(df["match_id"].nunique()),
        "feature_dim": int(len(df.iloc[0]["features"])),
        "snapshots_s": sorted(df["snapshot_s"].unique().tolist()),
        "feature_names": feature_names(),
    }
    meta_path = processed_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    paths["meta"] = meta_path
    return paths


class MatchSnapshotDataset(Dataset):  # type: ignore[misc]
    """PyTorch Dataset over a split parquet produced by write_dataset."""

    def __init__(
        self,
        parquet_path: str | Path,
        normalization_path: str | Path | None = None,
        *,
        snapshot_s: int | None = None,
        normalize: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for MatchSnapshotDataset")
        self.df = pd.read_parquet(parquet_path)
        if snapshot_s is not None:
            self.df = self.df[self.df["snapshot_s"] == snapshot_s].reset_index(drop=True)
        self.mean = None
        self.std = None
        if normalize:
            norm_path = (
                Path(normalization_path)
                if normalization_path
                else Path(parquet_path).with_name("normalization.json")
            )
            if norm_path.exists():
                norm = json.loads(norm_path.read_text())
                self.mean = np.asarray(norm["mean"], dtype=np.float32)
                self.std = np.asarray(norm["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        features = np.asarray(row["features"], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std
        return {
            "features": torch.from_numpy(features),
            "hero_ids": torch.tensor(row["hero_ids"], dtype=torch.long),
            "label": torch.tensor(row["label_team0_wins"], dtype=torch.float32),
            "snapshot_s": torch.tensor(row["snapshot_s"], dtype=torch.long),
            "match_id": torch.tensor(row["match_id"], dtype=torch.long),
        }
