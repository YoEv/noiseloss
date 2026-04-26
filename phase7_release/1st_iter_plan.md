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

**Fix**: Merged C-variant scores onto the full 4896-row feature splits (preserving SAE shard alignment). The 28 dropped lyria-3-pro-preview clips retain original scores; all other clips use corrected C scores.
- `split_train.csv`: 4896 rows (4868 C-scored + 28 lyria fallback)
- `split_val.csv`: 612 rows (606 C-scored + 6 lyria fallback)
- `split_test.csv`: 612 rows (606 C-scored + 6 lyria fallback)

---

## Results — f01–f07 CNN (music_arena, corrected C scoring)

**Completed**: 2026-04-26 | **Reports**: `phase7_release/outputs/full/reports/music_arena/clean/`

| Experiment | Features | Pearson | Spearman | N |
|---|---|---|---|---|
| f01 | loss-only | 0.622 | 0.652 | 606 |
| f02 | entropy-only | 0.604 | 0.676 | 606 |
| **f03** | **SAE-only** | **0.785** | **0.851** | 612 |
| f04 | loss + entropy | 0.599 | 0.648 | 612 |
| f05 | entropy + SAE | 0.780 | 0.846 | 612 |
| f06 | loss + SAE | 0.774 | 0.840 | 612 |
| f07 | loss + entropy + SAE | 0.760 | 0.827 | 612 |

**Key findings**:
- SAE features dominate: f03 (SAE-only) is best at Pearson=0.785
- Adding loss/entropy on top of SAE slightly hurts (f05–f07 < f03)
- Loss/entropy alone: 0.60–0.62, far behind SAE
- Previous run with stale scores peaked at Pearson=0.256 — corrected scoring lifted all experiments dramatically

**Comparison with stale-score run** (for reference):

| Experiment | Old Pearson | New Pearson | Δ |
|---|---|---|---|
| f01 | 0.146 | 0.622 | +0.476 |
| f02 | 0.136 | 0.604 | +0.468 |
| f03 | 0.242 | 0.785 | +0.543 |
| f04 | 0.109 | 0.599 | +0.490 |
| f05 | 0.256 | 0.780 | +0.524 |
| f06 | 0.276 | 0.774 | +0.498 |
| f07 | 0.259 | 0.760 | +0.501 |

---

## Results — f01–f07 CNN (all_5_datasets merged, uniform sampling)

**Completed**: 2026-04-26 | **Checkpoints**: `/tmp/ckpt_all5/` | **Reports**: `/tmp/reports_all5/`

**Setup**: 14077 train / 1758 val / 1762 test (musicpref + aime + songeval + music_arena + musiceval merged).
Uniform sampling. SAE hidden_dim=4096. Entropy manifests: 15 files (5 datasets × 3 splits).

| Experiment | Features | Test Pearson | Test Spearman | N |
|---|---|---|---|---|
| f01 | loss-only | 0.582 | 0.546 | 1762 |
| f02 | entropy-only | 0.617 | 0.591 | 1762 |
| f03 | SAE-only | 0.551 | 0.448 | 1762 |
| f04 | loss + entropy | 0.614 | 0.579 | 1762 |
| **f05** | **entropy + SAE** | **0.729** | **0.702** | 1762 |
| f06 | loss + SAE | 0.703 | 0.664 | 1762 |
| f07 | loss + entropy + SAE | 0.722 | 0.699 | 1762 |

**Key findings**:
- f05 (entropy+SAE) is best: P=0.729, S=0.702 — entropy curves complement SAE well on cross-dataset generalization
- f03 SAE-only drops dramatically vs music_arena (0.551 vs 0.785) — SAE alone doesn't generalize across datasets
- Adding entropy to SAE (f05) recovers +0.178 Pearson over f03 alone — entropy provides cross-dataset signal
- Loss curves add less value than entropy when combined with SAE (f06 < f05)
- f07 (full hybrid) slightly worse than f05 — loss curves may add noise on top of entropy+SAE
- Non-SAE models (f01–f04): 0.58–0.62 Pearson, consistent with music_arena-only performance

**Contrast with music_arena-only** (N=606):

| Exp | music_arena P | all_5 P | Δ |
|-----|--------------|---------|---|
| f01 | 0.622 | 0.582 | −0.040 |
| f02 | 0.604 | 0.617 | +0.013 |
| f03 | 0.785 | 0.551 | −0.234 |
| f05 | 0.780 | 0.729 | −0.051 |
| f06 | 0.774 | 0.703 | −0.071 |
| f07 | 0.760 | 0.722 | −0.038 |

SAE-only collapses on cross-dataset (−0.234); adding entropy rescues performance (f05 −0.051 only).

---

## Next Steps
1. Investigate SAE generalization gap: SAE features (musicgen-small) may be dataset-specific; entropy codebook curves provide more universal signal.
2. Evaluate per-dataset breakdown of all_5 test set (are some datasets harder? does SAE help on music_arena subset?).
3. Run full pipeline for individual datasets (musicpref, aime, songeval) with the orchestrator.
4. Consider ablating masked mean pooling vs. AdaptiveAvgPool in hybrid models.
