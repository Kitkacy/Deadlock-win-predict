# Deadlock-win-predict

Build a training database for a PyTorch model that predicts which team wins a
[Deadlock](https://store.steampowered.com/app/1422450/Deadlock/) match from the
in-match state at a few timestamps.

Data comes from the community [Deadlock API](https://api.deadlock-api.com)
(`GET /v1/matches/metadata` bulk endpoint).

## Snapshot times (API limitation)

Player `stats` are **not** available at arbitrary minutes. Live metadata shows:

- every **180 s** from **3 → 15 min** (`180, 360, …, 900`)
- then every **300 s** (`1200, 1500, 1800, …`) plus match end

So **5 / 15 / 30 min** becomes **6 / 15 / 30 min** (`360, 900, 1800`). There is
no 300 s tick. Defaults in code match that grid.

Scope / sizing notes (matches, rate limits, M4 Pro training): see the project
doc `dataset-scope-plan.md` in the Cursor project store (or ask the maintainer).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional: set `DEADLOCK_API_KEY` for higher quotas. Without a key, the client
throttles to ~30 req/min (bulk IP limit).

## Pipeline usage

All commands write under `data/` by default (`data/match_ids.json`, `data/raw/`,
`data/processed/`).

### 1. Discover match IDs

```bash
deadlock-dataset discover --count 50000 --game-mode normal --match-mode ranked \
  --min-duration-s 1800
```

Or use recently ingested matches (good for smoke tests):

```bash
deadlock-dataset discover --source recent --count 50
```

### 2. Fetch + cache raw metadata (resumable)

```bash
deadlock-dataset fetch --match-ids data/match_ids.json --batch-size 15
```

- Uses bulk `include_player_stats` / `items` / `objectives`
- Caches one JSON per match under `data/raw/<shard>/<match_id>.json`
- Checkpoint in `data/raw/checkpoint.json`; re-runs skip cached IDs
- Retries with backoff on 429 / 5xx; honors `Retry-After`

### 3. Build PyTorch-ready parquet

```bash
deadlock-dataset build --match-ids data/match_ids.json --snapshots 360,900,1800
```

Writes:

- `data/processed/train.parquet` / `val.parquet` / `test.parquet` (split **by match**)
- `data/processed/normalization.json` (mean/std from **train** only)
- `data/processed/splits.json`, `meta.json`

### 4. Smoke run (live API)

```bash
deadlock-dataset smoke --count 5 --snapshots 360,900
```

### Load in PyTorch

```python
from deadlock_win_predict.dataset import MatchSnapshotDataset
from torch.utils.data import DataLoader

ds = MatchSnapshotDataset(
    "data/processed/train.parquet",
    "data/processed/normalization.json",
    snapshot_s=900,  # optional filter
)
loader = DataLoader(ds, batch_size=256, shuffle=True)
batch = next(iter(loader))
# batch["features"], batch["hero_ids"], batch["label"]
```

Feature vector = team sums / means / diffs of core stats, sorted net-worth
profiles, badge + objective + item-count summaries, plus `hero_ids` (12 slots)
for an embedding table.

## Rate limits (bulk metadata)

From the OpenAPI description (subject to change upstream):

| Scope | Quota |
| ----- | ----- |
| IP | 30 req/min |
| API key | 30 / 10 s, 2000 / h |
| Global | 600 req/min |

Keep `batch-size` around **10–25** when requesting full stats (≈350–450 KB/match).

## Layout

```
deadlock_win_predict/
  api_client.py   # throttled HTTP + retries
  discover.py     # match ID discovery
  fetch.py        # cache + checkpoint
  features.py     # snapshot feature extraction
  dataset.py      # parquet + torch Dataset
  cli.py
```
