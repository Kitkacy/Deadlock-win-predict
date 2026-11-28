"""Offline unit tests for feature extraction."""

from __future__ import annotations

from deadlock_win_predict.features import (
    coerce_unix_seconds,
    extract_snapshot_row,
    feature_names,
    select_stats_sample,
)


def _sample(ts: int, net_worth: int, **extra):
    base = {
        "time_stamp_s": ts,
        "net_worth": net_worth,
        "level": 10,
        "kills": 1,
        "deaths": 0,
        "assists": 2,
        "denies": 0,
        "creep_kills": 5,
        "neutral_kills": 0,
        "possible_creeps": 6,
        "player_damage": 1000,
        "player_damage_taken": 800,
        "boss_damage": 0,
        "player_healing": 0,
        "teammate_healing": 0,
        "max_health": 1000,
        "weapon_power": 10,
        "tech_power": 10,
        "ability_points": 4,
    }
    base.update(extra)
    return base


def _player(team: str, hero_id: int, stats, items=None):
    return {
        "account_id": hero_id * 100,
        "team": team,
        "hero_id": hero_id,
        "assigned_lane": 1,
        "player_slot": hero_id,
        "stats": stats,
        "items": items or [],
    }


def test_select_stats_sample_nearest_leq():
    stats = [_sample(180, 1), _sample(360, 2), _sample(540, 3)]
    assert select_stats_sample(stats, 360)["net_worth"] == 2
    assert select_stats_sample(stats, 400)["net_worth"] == 2
    assert select_stats_sample(stats, 100) is None


def test_coerce_unix_seconds():
    assert coerce_unix_seconds(100) == 100
    assert coerce_unix_seconds("2026-11-22 21:42:37") > 0
    assert coerce_unix_seconds(None) == 0


def test_extract_snapshot_row_shape():
    stats = [_sample(180, 5000), _sample(360, 12000), _sample(900, 30000)]
    players = [_player("Team0", i + 1, stats) for i in range(6)]
    players += [_player("Team1", i + 10, stats) for i in range(6)]
    match = {
        "match_id": 42,
        "duration_s": 2000,
        "winning_team": "Team0",
        "start_time": "2026-11-22 21:42:37",
        "game_mode": "Normal",
        "match_mode": "Ranked",
        "average_badge_team0": 50,
        "average_badge_team1": 52,
        "objectives": [
            {"destroyed_time_s": 200, "team": "Team0"},
            {"destroyed_time_s": 500, "team": "Team1"},
        ],
        "players": players,
    }
    row = extract_snapshot_row(match, 360)
    assert row is not None
    assert row["label_team0_wins"] == 1
    assert row["features"].shape[0] == len(feature_names())
    assert row["hero_ids"].shape == (12,)
    # Samples only go to 900; 1800 is too stale relative to the 300s late-game cadence.
    assert extract_snapshot_row(match, 1800) is None
