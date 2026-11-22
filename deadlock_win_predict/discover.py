"""Discover match IDs via bulk metadata filters or recently-fetched."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from .api_client import DeadlockAPIClient

logger = logging.getLogger(__name__)


def discover_recent(client: DeadlockAPIClient, limit: int | None = None) -> list[int]:
    rows = client.recently_fetched()
    ids = [int(r["match_id"]) for r in rows if r.get("match_id") is not None]
    if limit is not None:
        ids = ids[:limit]
    return ids


def discover_bulk(
    client: DeadlockAPIClient,
    *,
    target_count: int,
    batch_limit: int = 1000,
    game_mode: str | None = "normal",
    match_mode: str | None = "ranked",
    min_duration_s: int | None = 900,
    min_average_badge: int | None = None,
    max_average_badge: int | None = None,
    min_unix_timestamp: int | None = None,
    max_unix_timestamp: int | None = None,
    is_low_pri_pool: bool | None = False,
    is_new_player_pool: bool | None = False,
    order_by: str = "match_id",
    order_direction: str = "desc",
) -> list[int]:
    """Page backward through bulk metadata (info-only) until target_count IDs collected.

    Paging uses ``max_match_id`` and therefore defaults to ``order_by=match_id``.
    """
    collected: list[int] = []
    seen: set[int] = set()
    cursor_max_id: int | None = None

    while len(collected) < target_count:
        params: dict[str, Any] = {
            "include_info": True,
            "include_player_info": False,
            "include_player_stats": False,
            "limit": min(batch_limit, target_count - len(collected) + 50),
            "order_by": order_by,
            "order_direction": order_direction,
            "game_mode": game_mode,
            "match_mode": match_mode,
            "min_duration_s": min_duration_s,
            "min_average_badge": min_average_badge,
            "max_average_badge": max_average_badge,
            "min_unix_timestamp": min_unix_timestamp,
            "max_unix_timestamp": max_unix_timestamp,
            "is_low_pri_pool": is_low_pri_pool,
            "is_new_player_pool": is_new_player_pool,
        }
        if cursor_max_id is not None:
            params["max_match_id"] = cursor_max_id - 1

        batch = client.bulk_metadata(**params)
        if not batch:
            logger.info("Discovery exhausted (empty batch); got %s ids", len(collected))
            break

        batch_min = None
        for row in batch:
            mid = row.get("match_id")
            if mid is None:
                continue
            mid = int(mid)
            if batch_min is None or mid < batch_min:
                batch_min = mid
            if mid in seen:
                continue
            seen.add(mid)
            collected.append(mid)
            if len(collected) >= target_count:
                break

        if batch_min is None:
            break
        if cursor_max_id is not None and batch_min >= cursor_max_id:
            logger.warning("Discovery cursor did not advance; stopping")
            break
        cursor_max_id = batch_min
        logger.info("Discovered %s / %s match ids", len(collected), target_count)

    return collected[:target_count]


def save_match_ids(path: Path, match_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(match_ids, indent=2))


def load_match_ids(path: Path) -> list[int]:
    return [int(x) for x in json.loads(path.read_text())]


def iter_chunks(items: list[int], size: int) -> Iterator[list[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
