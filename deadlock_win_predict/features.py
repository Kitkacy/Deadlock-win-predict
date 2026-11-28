"""Extract snapshot features at configurable in-match timestamps."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# API cadence: 180s ticks until 900s, then 300s. Default aligns to that grid.
DEFAULT_SNAPSHOT_SECONDS = (360, 900, 1800)

# Compact per-player numeric fields taken from a stats sample.
STAT_FIELDS = (
    "net_worth",
    "level",
    "kills",
    "deaths",
    "assists",
    "denies",
    "creep_kills",
    "neutral_kills",
    "possible_creeps",
    "player_damage",
    "player_damage_taken",
    "boss_damage",
    "player_healing",
    "teammate_healing",
    "max_health",
    "weapon_power",
    "tech_power",
    "ability_points",
)

TEAM_LABELS = {
    "Team0": 0,
    "Team1": 1,
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
}


def coerce_unix_seconds(value: Any) -> int:
    """Accept unix ints or ClickHouse DateTime strings like '2026-11-22 21:42:37'."""
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        try:
            from datetime import datetime, timezone

            # Bulk metadata often returns naive UTC timestamps.
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            return 0
    return 0


def winning_team_to_label(winning_team: Any) -> int | None:
    if winning_team is None:
        return None
    if winning_team in TEAM_LABELS:
        return TEAM_LABELS[winning_team]
    if isinstance(winning_team, str) and winning_team.lower() in ("team0", "team_0"):
        return 0
    if isinstance(winning_team, str) and winning_team.lower() in ("team1", "team_1"):
        return 1
    try:
        return int(winning_team)
    except (TypeError, ValueError):
        return None


def player_team(player: dict[str, Any]) -> int | None:
    return winning_team_to_label(player.get("team"))


def expected_tick_tolerance(target_s: int) -> int:
    """Max age (seconds) allowed when snapping to the nearest earlier API tick."""
    # Cadence is 180s until 900, then 300s. Allow one missed interval of slack.
    return 360 if target_s <= 900 else 600


def select_stats_sample(
    stats: list[dict[str, Any]],
    target_s: int,
    *,
    max_staleness_s: int | None = None,
) -> dict[str, Any] | None:
    """Return the latest sample with time_stamp_s <= target_s (exact tick preferred)."""
    if not stats:
        return None
    if max_staleness_s is None:
        max_staleness_s = expected_tick_tolerance(target_s)
    best = None
    best_ts = -1
    for sample in stats:
        ts = sample.get("time_stamp_s")
        if ts is None:
            continue
        ts = int(ts)
        if ts <= target_s and ts >= best_ts:
            best = sample
            best_ts = ts
    if best is None:
        return None
    if target_s - best_ts > max_staleness_s:
        return None
    return best


def items_owned_at(items: list[dict[str, Any]], target_s: int) -> list[dict[str, Any]]:
    owned = []
    for item in items or []:
        bought = item.get("game_time_s")
        if bought is None or int(bought) > target_s:
            continue
        sold = item.get("sold_time_s") or 0
        if int(sold) and 0 < int(sold) <= target_s:
            continue
        owned.append(item)
    return owned


def objectives_taken_by_t(objectives: list[dict[str, Any]], target_s: int) -> tuple[int, int]:
    """Count objectives destroyed by each team at or before t."""
    counts = [0, 0]
    for obj in objectives or []:
        destroyed = obj.get("destroyed_time_s")
        if destroyed is None or int(destroyed) <= 0 or int(destroyed) > target_s:
            continue
        team = winning_team_to_label(obj.get("team"))
        if team in (0, 1):
            counts[team] += 1
    return counts[0], counts[1]


def _stat_vec(sample: dict[str, Any]) -> np.ndarray:
    return np.array([float(sample.get(f) or 0) for f in STAT_FIELDS], dtype=np.float32)


def extract_snapshot_row(
    match: dict[str, Any],
    snapshot_s: int,
    *,
    require_exact_tick: bool = False,
) -> dict[str, Any] | None:
    """
    Build one training row for a match at snapshot_s.

    Returns None if the match is too short, missing label, or missing samples.
    """
    duration = int(match.get("duration_s") or 0)
    if duration < snapshot_s:
        return None

    label = winning_team_to_label(match.get("winning_team"))
    if label not in (0, 1):
        return None

    players = match.get("players") or []
    if len(players) < 8:
        return None

    team_players: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    hero_ids = np.zeros(12, dtype=np.int64)
    lanes = np.zeros(12, dtype=np.int64)
    per_player = np.zeros((12, len(STAT_FIELDS)), dtype=np.float32)
    item_counts = np.zeros(12, dtype=np.float32)
    slot = 0

    for player in players:
        team = player_team(player)
        if team not in (0, 1):
            continue
        stats = player.get("stats") or []
        sample = select_stats_sample(stats, snapshot_s)
        if sample is None:
            return None
        if require_exact_tick and int(sample.get("time_stamp_s") or -1) != snapshot_s:
            # Allow end-of-match sample only when equal duration; otherwise skip.
            if int(sample.get("time_stamp_s") or -1) != duration:
                return None

        if slot < 12:
            hero_ids[slot] = int(player.get("hero_id") or 0)
            lanes[slot] = int(player.get("assigned_lane") or 0)
            per_player[slot] = _stat_vec(sample)
            item_counts[slot] = float(len(items_owned_at(player.get("items") or [], snapshot_s)))
            team_players[team].append(
                {
                    "slot": slot,
                    "sample": sample,
                    "hero_id": int(player.get("hero_id") or 0),
                    "item_count": item_counts[slot],
                }
            )
            slot += 1

    if len(team_players[0]) < 3 or len(team_players[1]) < 3:
        return None

    def team_sum(team: int) -> np.ndarray:
        idxs = [p["slot"] for p in team_players[team]]
        return per_player[idxs].sum(axis=0)

    sum0 = team_sum(0)
    sum1 = team_sum(1)
    diff = sum0 - sum1
    mean0 = sum0 / max(len(team_players[0]), 1)
    mean1 = sum1 / max(len(team_players[1]), 1)

    obj0, obj1 = objectives_taken_by_t(match.get("objectives") or [], snapshot_s)

    badge0 = float(match.get("average_badge_team0") or 0)
    badge1 = float(match.get("average_badge_team1") or match.get("average_badge") or 0)

    # Sorted net worth within team (order-invariant strength profile), pad to 6.
    def sorted_nw(team: int) -> np.ndarray:
        vals = sorted(
            (float(p["sample"].get("net_worth") or 0) for p in team_players[team]),
            reverse=True,
        )
        out = np.zeros(6, dtype=np.float32)
        out[: min(6, len(vals))] = vals[:6]
        return out

    feature_parts = [
        sum0,
        sum1,
        diff,
        mean0,
        mean1,
        sorted_nw(0),
        sorted_nw(1),
        np.array(
            [
                float(snapshot_s),
                float(duration),
                badge0,
                badge1,
                badge0 - badge1,
                float(obj0),
                float(obj1),
                float(obj0 - obj1),
                float(sum(p["item_count"] for p in team_players[0])),
                float(sum(p["item_count"] for p in team_players[1])),
            ],
            dtype=np.float32,
        ),
    ]
    features = np.concatenate(feature_parts).astype(np.float32)

    return {
        "match_id": int(match["match_id"]),
        "snapshot_s": int(snapshot_s),
        "label_team0_wins": int(label == 0),
        "duration_s": duration,
        "start_time": coerce_unix_seconds(match.get("start_time")),
        "game_mode": str(match.get("game_mode") or ""),
        "match_mode": str(match.get("match_mode") or ""),
        "features": features,
        "hero_ids": hero_ids,
        "assigned_lanes": lanes,
        "per_player_stats": per_player,
        "item_counts": item_counts,
        "feature_dim": int(features.shape[0]),
    }


def feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("sum0", "sum1", "diff", "mean0", "mean1"):
        names.extend(f"{prefix}_{f}" for f in STAT_FIELDS)
    names.extend(f"sorted_nw0_{i}" for i in range(6))
    names.extend(f"sorted_nw1_{i}" for i in range(6))
    names.extend(
        [
            "snapshot_s",
            "duration_s",
            "badge0",
            "badge1",
            "badge_diff",
            "obj0",
            "obj1",
            "obj_diff",
            "items0",
            "items1",
        ]
    )
    return names


def extract_match_rows(
    match: dict[str, Any],
    snapshots_s: tuple[int, ...] | list[int] = DEFAULT_SNAPSHOT_SECONDS,
) -> list[dict[str, Any]]:
    rows = []
    for t in snapshots_s:
        row = extract_snapshot_row(match, int(t))
        if row is not None:
            rows.append(row)
    return rows
