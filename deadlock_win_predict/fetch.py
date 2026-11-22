"""Fetch and cache raw match metadata (resumable)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .api_client import DeadlockAPIClient
from .discover import iter_chunks

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint.json"


def raw_match_path(raw_dir: Path, match_id: int) -> Path:
    # Shard by last 3 digits to avoid huge single directories.
    shard = f"{match_id % 1000:03d}"
    return raw_dir / shard / f"{match_id}.json"


def load_checkpoint(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / CHECKPOINT_NAME
    if not path.exists():
        return {"completed": [], "failed": []}
    return json.loads(path.read_text())


def save_checkpoint(raw_dir: Path, checkpoint: dict[str, Any]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = raw_dir / f".{CHECKPOINT_NAME}.tmp"
    tmp.write_text(json.dumps(checkpoint, indent=2))
    tmp.replace(raw_dir / CHECKPOINT_NAME)


def is_cached(raw_dir: Path, match_id: int) -> bool:
    return raw_match_path(raw_dir, match_id).exists()


def write_raw_match(raw_dir: Path, match_id: int, payload: dict[str, Any]) -> None:
    path = raw_match_path(raw_dir, match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(path)


def read_raw_match(raw_dir: Path, match_id: int) -> dict[str, Any]:
    return json.loads(raw_match_path(raw_dir, match_id).read_text())


def fetch_matches(
    client: DeadlockAPIClient,
    match_ids: list[int],
    raw_dir: Path,
    *,
    batch_size: int = 15,
    include_objectives: bool = True,
    include_player_items: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """
    Pull match metadata via bulk endpoint, cache one JSON file per match.

    Skips IDs already on disk unless force=True. Checkpoint tracks completed/failed.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(raw_dir)
    completed = set(int(x) for x in checkpoint.get("completed", []))
    failed = set(int(x) for x in checkpoint.get("failed", []))

    pending = []
    for mid in match_ids:
        if not force and (mid in completed or is_cached(raw_dir, mid)):
            completed.add(mid)
            continue
        pending.append(mid)

    logger.info(
        "Fetch: %s total, %s cached/done, %s pending",
        len(match_ids),
        len(match_ids) - len(pending),
        len(pending),
    )

    for chunk in tqdm(list(iter_chunks(pending, batch_size)), desc="fetch batches"):
        try:
            rows = client.bulk_metadata(
                match_ids=chunk,
                include_info=True,
                include_more_info=True,
                include_objectives=include_objectives,
                include_player_info=True,
                include_player_stats=True,
                include_player_items=include_player_items,
                limit=len(chunk),
            )
        except Exception as exc:  # noqa: BLE001 — resume-friendly
            logger.error("Bulk fetch failed for %s: %s", chunk, exc)
            failed.update(chunk)
            checkpoint["failed"] = sorted(failed)
            checkpoint["completed"] = sorted(completed)
            save_checkpoint(raw_dir, checkpoint)
            continue

        got = {int(r["match_id"]): r for r in rows if r.get("match_id") is not None}
        for mid in chunk:
            if mid in got:
                write_raw_match(raw_dir, mid, got[mid])
                completed.add(mid)
                failed.discard(mid)
            else:
                # Fallback to single-match endpoint once.
                try:
                    single = client.match_metadata(mid, disable_steam=True)
                    # Single endpoint nests under match_info in some clients;
                    # normalize to the flat bulk shape when possible.
                    payload = single.get("match_info", single)
                    if "match_id" not in payload:
                        payload = {**single, "match_id": mid}
                    write_raw_match(raw_dir, mid, payload)
                    completed.add(mid)
                    failed.discard(mid)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Single fetch failed for %s: %s", mid, exc)
                    failed.add(mid)

        checkpoint["completed"] = sorted(completed)
        checkpoint["failed"] = sorted(failed)
        save_checkpoint(raw_dir, checkpoint)

    return checkpoint
