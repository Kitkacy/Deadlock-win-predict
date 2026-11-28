"""CLI entrypoints for discover / fetch / build / train / smoke."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .api_client import DeadlockAPIClient
from .dataset import MatchSnapshotDataset, build_rows_from_raw, write_dataset
from .discover import discover_bulk, discover_recent, load_match_ids, save_match_ids
from .features import DEFAULT_SNAPSHOT_SECONDS
from .fetch import fetch_matches
from .train import train_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("deadlock_win_predict")


def _parse_snapshots(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deadlock win-predict dataset pipeline")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--api-key", default=os.environ.get("DEADLOCK_API_KEY"))
    p.add_argument(
        "--min-interval",
        type=float,
        default=2.1,
        help="Min seconds between API calls (IP bulk limit ~30/min)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Discover match IDs")
    d.add_argument("--count", type=int, default=100)
    d.add_argument("--source", choices=("bulk", "recent"), default="bulk")
    d.add_argument("--game-mode", default="normal")
    d.add_argument("--match-mode", default="ranked")
    d.add_argument("--min-duration-s", type=int, default=900)
    d.add_argument("--min-average-badge", type=int, default=None)
    d.add_argument("--out", type=Path, default=None)

    f = sub.add_parser("fetch", help="Fetch and cache raw match metadata")
    f.add_argument("--match-ids", type=Path, required=True)
    f.add_argument("--batch-size", type=int, default=15)
    f.add_argument("--force", action="store_true")

    b = sub.add_parser("build", help="Extract features and write parquet splits")
    b.add_argument("--match-ids", type=Path, required=True)
    b.add_argument(
        "--snapshots",
        default=",".join(str(x) for x in DEFAULT_SNAPSHOT_SECONDS),
        help="Comma-separated snapshot seconds (API grid: 180-step to 900, then 300)",
    )
    b.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train", help="Train baseline MLP on processed parquet")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--snapshot-s", type=int, default=None, help="Optional single-tick filter")
    t.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    t.add_argument("--out-dir", type=Path, default=None)

    s = sub.add_parser("smoke", help="Tiny live end-to-end run (discover+fetch+build[+train])")
    s.add_argument("--count", type=int, default=5)
    s.add_argument(
        "--snapshots",
        default="360,900",
        help="Use shorter default for smoke so short matches still contribute",
    )
    s.add_argument("--source", choices=("bulk", "recent"), default="bulk")
    s.add_argument("--batch-size", type=int, default=5)
    s.add_argument("--skip-train", action="store_true")

    sub.add_parser("show-defaults", help="Print recommended snapshot defaults")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir: Path = args.data_dir
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    ids_path = data_dir / "match_ids.json"

    if args.cmd == "show-defaults":
        print("DEFAULT_SNAPSHOT_SECONDS", DEFAULT_SNAPSHOT_SECONDS)
        print("Reason: API samples every 180s until 900s, then every 300s. 5 min (300) is missing.")
        return 0

    if args.cmd == "train":
        out_dir = args.out_dir or (data_dir / "models" / "baseline")
        summary = train_baseline(
            processed_dir,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            snapshot_s=args.snapshot_s,
            device_name=args.device,
        )
        logger.info("Train done: %s", summary.get("best_checkpoint"))
        return 0

    if args.cmd == "build":
        ids = load_match_ids(args.match_ids)
        snaps = _parse_snapshots(args.snapshots)
        rows = build_rows_from_raw(raw_dir, ids, snapshots_s=snaps)
        paths = write_dataset(processed_dir, rows, seed=args.seed)
        for k, v in paths.items():
            logger.info("%s -> %s", k, v)
        return 0

    with DeadlockAPIClient(api_key=args.api_key, min_interval_s=args.min_interval) as client:
        if args.cmd == "discover":
            out = args.out or ids_path
            if args.source == "recent":
                ids = discover_recent(client, limit=args.count)
            else:
                ids = discover_bulk(
                    client,
                    target_count=args.count,
                    game_mode=args.game_mode,
                    match_mode=args.match_mode,
                    min_duration_s=args.min_duration_s,
                    min_average_badge=args.min_average_badge,
                )
            save_match_ids(out, ids)
            logger.info("Wrote %s match ids to %s", len(ids), out)
            return 0

        if args.cmd == "fetch":
            ids = load_match_ids(args.match_ids)
            fetch_matches(client, ids, raw_dir, batch_size=args.batch_size, force=args.force)
            return 0

        if args.cmd == "smoke":
            out = ids_path
            if args.source == "recent":
                ids = discover_recent(client, limit=args.count)
            else:
                ids = discover_bulk(
                    client,
                    target_count=args.count,
                    game_mode="normal",
                    match_mode="ranked",
                    min_duration_s=900,
                )
            save_match_ids(out, ids)
            fetch_matches(client, ids, raw_dir, batch_size=args.batch_size)
            snaps = _parse_snapshots(args.snapshots)
            rows = build_rows_from_raw(raw_dir, ids, snapshots_s=snaps)
            if not rows:
                logger.error("Smoke produced 0 rows; try --source recent or lower snapshots")
                return 1
            paths = write_dataset(processed_dir, rows, seed=42)
            ds = MatchSnapshotDataset(paths["train"], paths["normalization"])
            sample = ds[0]
            logger.info(
                "Smoke OK: %s rows, feature_dim=%s, sample label=%s",
                len(rows),
                sample["features"].shape[0],
                float(sample["label"]),
            )
            if not args.skip_train:
                summary = train_baseline(
                    processed_dir,
                    out_dir=data_dir / "models" / "smoke",
                    epochs=3,
                    batch_size=8,
                    device_name="cpu",
                )
                logger.info("Smoke train OK: %s", summary.get("best_checkpoint"))
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
