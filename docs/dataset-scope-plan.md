# Dataset scope plan — Deadlock win-rate predictor

**Audience:** Wincenty Wensker  
**Repo:** Kitkacy/Deadlock-win-predict  
**API:** [deadlock-api.com](https://api.deadlock-api.com) (community; OpenAPI snapshot in `internal/deadlock-api.json`)  
**Train target:** Apple M4 Pro, 24 GB unified memory (MPS)

## Headline numbers

| Item | Recommendation |
| ---- | -------------- |
| Matches (v1 “somewhat good”) | **50 000** ranked Normal games (floor **20 000**, stretch **100 000**) |
| Training rows | ~**100 000–150 000** (one row per match × reachable snapshot) |
| Snapshots | **6 / 15 / 30 min** (`360 / 900 / 1800` s) — *not* 5/15/30 |
| Feature vector / row | ~**120–180** floats (team aggregates + diffs + hero embeddings inputs) |
| Raw cache size | ~**15–25 GB** for 50 k matches with full `stats` + `items` |
| Feature parquet | ~**0.5–1.5 GB** |
| Pull wall time (no API key) | ~**2.5–4 h** at bulk IP limit |
| Pull wall time (with API key) | ~**30–60 min** (watch payload size / 503 load shedding) |
| Model | Small **MLP + hero embeddings** (~**0.3–1 M** params); optional tiny Transformer later |
| Train RAM / time on M4 Pro | comfortably fits 24 GB; **minutes–tens of minutes** per epoch set |

---

## 1. What the API actually gives (timestamp constraint)

Per-player `stats` in match metadata is a **time series of snapshots**, not a continuous stream.

From live probes of `GET /v1/matches/metadata?include_player_stats=true` (Normal / Ranked):

- Samples start at **180 s**, then every **180 s until 900 s** (3 → 15 min).
- After 15 min, samples continue every **300 s** (20, 25, 30, … min), plus a final sample at match end.
- Example timestamps: `180, 360, 540, 720, 900, 1200, 1500, 1800, …, duration_s`.

So the initial ask of **~5 / 15 / 30 min** must be adjusted:

| Desired | Exact API tick? | Use instead |
| ------- | --------------- | ----------- |
| 5 min (300 s) | **No** | **6 min (360 s)** — closest post-laning tick |
| 15 min (900 s) | Yes | **900 s** |
| 30 min (1800 s) | Yes (if match lasts) | **1800 s**; skip row if `duration_s < 1800` |

Optional extra ticks that align cleanly: **3 / 9 / 12 / 20 / 25 min** (`180 / 540 / 720 / 1200 / 1500`).

Do **not** rely on demo SQL (`/v1/matches/demo/query`) for v1: async, heavy, and rate-limited via salts/Steam. Metadata bulk is the right source.

---

## 2. How many matches for a “somewhat good” model

Task: binary **team-0 wins?** from mid-match state (separate heads or one model conditioned on `t`).

| Scale | Matches | Expectation | Fits user time + M4 Pro? |
| ----- | ------- | ----------- | ------------------------ |
| Smoke / debug | 100–1 000 | Pipeline validation only | Yes |
| Minimum useful | **20 000** | Better than badge/net-worth heuristics; noisy heroes | Yes |
| **Recommended v1** | **50 000** | “Somewhat good” MLP; stable val AUC / Brier | Yes |
| Stronger | 100 000–200 000 | Better rare-hero / late-patch coverage | Pull becomes the bottleneck |

**Why ~50 k:** ~12 heroes/side → combinatorial draft space is large; mid-match **economy/KDA/objectives** carry most of the signal. 50 k matches × ~2–3 timestamps ≈ **100–150 k rows** is enough for a compact NN without drowning an M4 Pro. Past ~100 k, returns diminish unless you add richer features (items graph, paths) or patch-aware splits.

**Rank filter:** prefer **Ranked + Normal**, `min_average_badge` around **Archon+** (exact badge ints from `/v1/assets/ranks`) to cut new-player / bot noise. Exclude low-pri / new-player pools. Drop matches with abandons before the snapshot.

**Patch / time filter:** pull a **recent contiguous window** (e.g. last 2–4 weeks) so meta is coherent; store `start_time` for later patch splits.

---

## 3. Per-match fields to keep

### Match-level (label + context)

- `match_id`, `start_time`, `duration_s`
- `winning_team` / `match_outcome` → label `y = 1` if Team0 wins
- `match_mode`, `game_mode`
- `average_badge_team0`, `average_badge_team1` (and match average)
- `objectives` (destroyed times / team) — encode “objectives taken by `t`”
- `banned_hero_ids` when present
- Flags: `not_scored`, low-pri / new-player / high-skill parties (filter or features)

### Per-player @ timestamp `t` (from `stats` sample nearest ≤ `t`, or exact tick)

Core (high value):

- `hero_id`, `team`, `assigned_lane`, `player_slot`
- `net_worth`, `level`, `kills`, `deaths`, `assists`
- `creep_kills`, `neutral_kills`, `denies`, `possible_creeps`
- `player_damage`, `player_damage_taken`, `boss_damage`
- `player_healing` / `teammate_healing`, `max_health`, `weapon_power`, `tech_power`

Items @ `t` (from `items` where `game_time_s ≤ t` and not sold yet):

- counts by tier/slot **or** top‑k item IDs → multi-hot / hashed
- v1 can start with **item count + total cost proxy (`net_worth_at_buy` sum)** to keep vectors small

### Aggregates to feed the model (recommended)

Prefer **team sums + team diffs** (Team0 − Team1), not raw 12-player concatenation as the only input:

- Team net worth, K/D/A, damage, healing, levels
- Diff features (strongest single signal class in MOBAs)
- Sorted-within-team net worth (order-invariant)
- Hero IDs → **embedding table** (shared), looked up per slot then pooled (sum/mean/attention)
- Lane matchups optional later

Keep **per-player tensors** in parquet for ablation, but train v1 on aggregates + hero IDs.

---

## 4. Storage & pull-time budget

### Endpoints

| Step | Endpoint | Role |
| ---- | -------- | ---- |
| Discover IDs | `GET /v1/matches/metadata` with filters, `include_info=true` only, high `limit` | Cheap ID listing |
| Or | `GET /v1/matches/recently-fetched` | Last ~10 min only; good for smoke |
| Pull full rows | Same bulk URL with `include_player_info`, `include_player_stats`, `include_player_items`, `include_objectives` | Primary dataset source |
| Fallback | `GET /v1/matches/{id}/metadata` | Single-match; use if bulk omits a field |

**Bulk rate limits (from OpenAPI description):** IP **30 req/min**; API key **30/10s + 2000/h**; global **600/min**.  
Expect **429** and **503 + Retry-After** under load shedding — client must backoff.

**Payload size:** Normal ranked match with stats+items+objectives ≈ **350–450 KB** JSON.  
Use **`limit=10–25`** per stats-inclusive request (not 1000).

### Estimates @ 50 000 matches

| | No API key (30/min) | With API key (~3/s careful) |
| - | ------------------- | --------------------------- |
| Requests @ `limit=20` | 2 500 | 2 500 |
| Wall time | ~**85 min** ideal; budget **2.5–4 h** with backoff | ~**20–40 min** ideal; budget **~1 h** |
| Raw on-disk cache | **~18–22 GB** | same |
| Feature parquet (3 ticks, compressed) | **~0.5–1.5 GB** | same |

Disk tip: cache raw **NDJSON/JSON per batch** under `data/raw/`; build features into `data/processed/*.parquet`. Re-runs must be **cache-first**.

---

## 5. Model recommendation (M4 Pro / 24 GB)

**v1:** feedforward net

- Inputs: team aggregate + diff vector (~80–120 dims) + hero embedding lookups (12 × `d_h`, `d_h=32–64`) pooled
- Hidden: 2–3 layers, width 128–256, dropout 0.1–0.3
- Params: **~0.3–1 M** — trivial for MPS; batch 1024–4096 fits easily
- Loss: BCEWithLogits; metrics: accuracy, AUC, Brier, calibration by minute mark
- Train separate models per `t` **or** one model with a learned/`t` scalar feature (start with **per-`t` heads or three models** — simpler analysis)

**Avoid for v1:** large Transformers over full tick histories, demo-path CNNs, or huge item graphs — little gain vs pull cost on 24 GB if features are already team diffs.

**Augmentation:** randomly swap Team0/Team1 and flip the label to double effective N and kill side bias.

---

## 6. Pipeline requirements (implementation checklist)

1. Discover match IDs with filters (mode, badge, duration ≥ max snapshot, time window).
2. Bulk-fetch with throttling, exponential backoff on 429/5xx, honor `Retry-After`.
3. Disk cache of raw responses; checkpoint progress; resume without re-download.
4. Configurable snapshot list (default `360,900,1800`).
5. Feature extraction → parquet + `normalization.json` (train-split mean/std only).
6. Match-level train/val/test split (e.g. 80/10/10); never split rows of the same match across sets.
7. `torch.utils.data.Dataset` reading parquet / memmap tensors.

---

## 7. Suggested phased pull

1. **Smoke:** 20–50 matches (done via pipeline CLI).
2. **Pilot:** 2 000 matches → train tiny MLP, sanity-check AUC ≫ 0.5 at 15/30 min.
3. **v1 dataset:** 50 000 matches.
4. Expand only if val metrics plateau for the wrong reasons (capacity), not data bugs.

---

## References

- OpenAPI: `https://api.deadlock-api.com/openapi.json` (local copy: project store `internal/deadlock-api.json`)
- Bulk metadata: `GET /v1/matches/metadata`
- Stats cadence also reflected in analytics docs (180 s to 900 s, then 300 s)
- Clients: https://github.com/deadlock-api/openapi-clients
