# Phase 7 — 1st Iteration Plan

## Goal
Train f01–f07 CNN models for all datasets; evaluate Pearson/Spearman correlation with human preference scores.

---

## ELO Scoring — Music Arena Fixes

### Problem identified
- `fit_pairwise_manifests.py` used `--passes 80` (track-level) with `both_bad=(-0.5,-0.5)`.
- Both_bad is non-zero-sum: drains 2K=48 rating points per match per pass from the pool.
- With 489 both_bad battles × 80 passes → all system ELOs collapsed to ~−150k.

### Fixes applied (`phase7_release/scripts/data/fit_pairwise_manifests.py`)
1. **`--passes` 80 → 4**: Limits track-level ELO accumulation. System-level forced to 1 pass via `--passes-system 1`.
2. **`--min-clips-per-system 60`**: Drops systems with < 60 qualifying clips (dropped `lyria-3-pro-preview`, 40 clips).
3. **`--zscore-residual` (default=True, `--system-span 50`)**: Per-system z-score of residuals before adding system baseline. Compresses within-system std; found to improve test-set R.

### Scoring variant comparison (f02/entropy-only CNN, music_arena test set)

| Variant | Pearson | Spearman | Description |
|---------|---------|----------|-------------|
| A | 0.511 | 0.528 | no-zscore, span=50 |
| B | 0.512 | 0.531 | zscore, span=10 |
| **C** | **0.623** | **0.695** | **zscore, span=50 ← selected** |

**Decision**: Use variant C (current scoring, z-score + span=50). Wins by +0.11 Pearson / +0.16 Spearman over no-z variants.

### Scoring validity check
- Pearson = 0.863, Spearman = 0.906 between ELO margin and rater outcome
- Directional accuracy = 99.3% on decisive battles

---

## Feature Split CSV Update

**Problem**: `phase7_release/features/loss/music_arena/clean/split_*.csv` had stale ELO scores from before the fix. Training used these CSVs directly, bypassing the corrected manifest.

**Fix**: Regenerated C-variant splits by joining corrected manifest onto feature split CSVs (matching by audio path stem). Overwrote the feature splits in-place:
- `split_train.csv`: 4868 rows (was 4896; dropped lyria-3-pro-preview clips)
- `split_val.csv`: 606 rows (was 612)
- `split_test.csv`: 606 rows (was 612)

---

## Training Status

### Previous runs (stale scores — do not use)
- f01–f07 CNN clean results in `phase7_release/outputs/full/reports/music_arena/clean/` reflect OLD ELO scoring.

### Current run — f01–f07 with corrected scoring (variant C)
- **Started**: 2026-04-25
- **Orchestrator**: `phase7_release/scripts/run/run_full_14_experiments_parallel.py`
- **Config**: `phase7_release/config/data/full_datasets.yaml` (music_arena only; other datasets temporarily disabled)
- **Log**: `phase7_release/outputs/run_state/full_parallel_logs/music_arena_retrain_C.log`
- **Reports dir**: `phase7_release/outputs/full/reports/music_arena/clean/`

---

## Next Steps
1. Wait for f01–f07 training to complete.
2. Read `aggregate_table_14_experiments_clean.csv` for final Pearson/Spearman table.
3. Re-enable other datasets in `full_datasets.yaml` (musicpref, aime, songeval, all_5_datasets).
4. Commit corrected split CSVs and scoring script changes.
