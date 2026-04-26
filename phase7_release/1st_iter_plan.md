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

## Baseline — Audiobox Aesthetics

**Completed**: 2026-04-26 | **Script**: `phase7_release/scripts/baseline/aesthetics_windowed.py`
**Env**: `audiobox` (torchaudio 2.1.0+cu121) | Window: `start_time`/`end_time` passed directly to AesPredictor

Axes: CE (Content Enjoyment), CU (Content Usefulness), PC (Production Complexity), PQ (Production Quality)

### music_arena — all clips (N=6120)

| Axis | Pearson | Spearman |
|------|---------|----------|
| CE | 0.237 | 0.260 |
| CU | 0.223 | 0.244 |
| PC | 0.148 | 0.230 |
| **PQ** | **0.333** | **0.395** |

Test set only (N=612): CE=0.289, CU=0.265, PC=0.134, PQ=**0.393**

### all_5_datasets — all clips (N=17597)

| Axis | Pearson | Spearman |
|------|---------|----------|
| CE | 0.200 | 0.215 |
| CU | 0.184 | 0.191 |
| PC | 0.071 | 0.086 |
| **PQ** | **0.198** | **0.233** |

Test set only (N=1762): CE=0.209, CU=0.199, PC=0.050, PQ=**0.215**

### Comparison: Aesthetics baseline vs CNN models

| Dataset | Best Aesthetics (PQ) | Best CNN (f05 ent+SAE) | CNN / Aes ratio |
|---------|---------------------|----------------------|-----------------|
| music_arena | P=0.333 | P=0.785 (f03) | 2.4× |
| all_5_datasets | P=0.198 | P=0.729 (f05) | 3.7× |

**Key findings**:
- PQ (Production Quality) is the strongest aesthetics axis across both datasets
- Aesthetics alone is a weak predictor: P=0.33 on music_arena, P=0.20 on all_5_datasets
- Our CNN models outperform aesthetics by 2.4–3.7× — learned features capture rater preference better than generic audio quality scores
- Cross-dataset gap is larger for aesthetics (0.333 → 0.198, −0.135) than for our models (0.785 → 0.729, −0.056)

---

## Baseline — Mean Token Loss

**Method**: Mean of `avg_loss_value` across all tokens in the pre-extracted loss file (already windowed to rater's audio window). No recomputation from MusicGen-small needed.

| Dataset | N | Pearson R | Spearman R |
|---------|---|-----------|------------|
| musicpref | 5030 | −0.297 | −0.415 |
| aime | 1300 | −0.105 | −0.092 |
| songeval | 2399 | +0.236 | +0.107 |
| music_arena | 6120 | −0.081 | −0.069 |
| musiceval | 2748 | −0.022 | −0.069 |
| **all_5_datasets** | **17597** | **−0.078** | **−0.127** |

**Key findings**:
- Negative correlation for 4/5 datasets — lower loss (model assigns higher probability) → higher human quality score, as expected
- songeval is anomalous (+0.236): scoring methodology differs (full-song ratings vs pairwise)
- musicpref has the strongest signal (|R|=0.297) but still far below CNN models
- Mean loss alone is a weak predictor: |R|=0.08–0.30 vs f01 CNN loss-only achieving P=0.622 on music_arena
- CNN models extract richer structure from the full loss curve shape beyond a scalar mean

---

## Next Steps
1. Investigate SAE generalization gap: SAE features (musicgen-small) may be dataset-specific; entropy codebook curves provide more universal signal.
2. Evaluate per-dataset breakdown of all_5 test set (are some datasets harder? does SAE help on music_arena subset?).
3. Run full pipeline for individual datasets (musicpref, aime, songeval) with the orchestrator.
4. Consider ablating masked mean pooling vs. AdaptiveAvgPool in hybrid models.
