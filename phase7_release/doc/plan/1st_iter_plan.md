# 1st-Iter Plan — Port the New Scoring & Extraction from `phase7_local4server` into `phase7_release`

## 0. Goals and Constraints

- **Goal**: Replace the two things `phase7_release` got wrong — the **scoring rules** and the **30 s feature extraction** — and only run the 7 clean × CNN experiment families, once per single dataset plus once on `all_5_datasets`.
- **Hard constraints**:
  1. Every existing **path and filename in `phase7_release` stays byte-identical** (CSV names, manifest paths, config keys, feature directories, splits directories, CLI arguments).
  2. Only **replace the implementation of existing files**; do not introduce a new directory layout.
  3. `scripts/run/run_full_14_experiments_parallel.sh --splits clean` overwrites the old results in one shot.
  4. **MusicEval is untouched** — scoring, splits, features, checkpoints and reports all remain as-is; nothing is re-run.
  5. **Completely remove** transformer and noisy: 7 families × 1 backbone × 1 splits-variant = 7 experiments per dataset.

---

## 1. Scoring Rule Replacement (new kernel, same filenames)

### 1.1 Import the Elo library

- Add `phase7_release/lib/elo_scoring.py`: lift `fit_elo`, `fit_elo_grouped`, `map_to_mos_range`, `track_score_from_system_elo` (with `target_fn`) from `phase7_local4server/training/elo.py`.
- **Delete** `phase7_release/lib/pairwise_relu_scores.py` (no legacy kept).

### 1.2 First, fix MusicPref scoring locally to "musicality-only"

Before starting the port, edit `phase7_local4server/training/build_musicpref_scores.py`: drop the fidelity axis, drop the `--w_musicality / --w_fidelity` arguments, and feed the single-axis `track_m` straight into `map_to_mos_range`. This keeps the scoring rules in `local4server` and `release` aligned.

### 1.3 Swap the kernel of `scripts/data/fit_pairwise_manifests.py`

**Filenames, CLI names, output CSV paths and columns all stay unchanged** — only the algorithm changes:

| Existing CLI (kept combinations) | Existing output path (unchanged) | New implementation |
|---|---|---|
| `--dataset musicpref --head musicality` | `data/manifests/pairwise_relu/musicpref_musicality_1to5.csv` | Elo: system Elo (7 systems) + `track_score_from_system_elo` + robust MOS onto `[1, 5]` (validated locally). |
| `--dataset aime --head music_quality` | `data/manifests/pairwise_relu/aime_music_quality_1to5.csv` | `build_aime_scores.py`'s `system_elo + 400·logit_Laplace(p̂)` → robust MOS; also writes per-track `begin_s, end_s` columns into the same file (backward-compatible). |
| `--dataset musicarena` | `data/manifests/pairwise_relu/musicarena_1to5.csv` | `build_musicarena_scores.py`'s system Elo + 4-outcome context-aware soft target + system-span=50 compression + robust MOS; also writes a `listen_sec` column. |

**Head combinations to remove entirely** (both code branches and old artifacts):

- CLI: drop `fidelity` and `text_audio_alignment` from `--head`'s `choices`, and remove the `load_musicpref_pairs` fidelity branch and the `load_aime_pairs` "Text-Audio Alignment" question-type branch.
- Old artifacts: `rm` `data/manifests/pairwise_relu/musicpref_fidelity_1to5.csv` and `data/manifests/pairwise_relu/aime_text_audio_alignment_1to5.csv` (already listed in the §3.1 cleanup).

### 1.4 Changes to `scripts/data/gen_full_splits.py`

All output paths and columns (`score, audio_path, token_loss_path`) **stay unchanged**; only the `score` source changes:

| dataset | current | new |
|---|---|---|
| `musicpref` | `(musicality + fidelity) / 2` | directly from `musicpref_musicality_1to5.csv` (single axis) |
| `aime` | `(music_quality + text_audio_alignment) / 2` | directly from `aime_music_quality_1to5.csv` (Music Quality only) |
| `music_arena` | `musicarena_1to5.csv` single column | unchanged (but underlying algorithm replaced) |
| `songeval` | 5 dims × 4 annotators averaged | **only** the 4 annotators' `Musicality` averaged |
| `musiceval` | native 5-point | **kept exactly as-is** |

The `begin_s, end_s` / `listen_sec` columns that AIME / MusicArena need are written into their respective 1to5 CSVs by `fit_pairwise_manifests.py`, and `gen_full_splits.py` propagates them as extra columns in `full_splits/<ds>/{train,val,test}.csv` (downstream scripts that only read `score, audio_path, token_loss_path` are unaffected).

### 1.5 `merge_all_datasets.py`

Filename, output path and columns **all unchanged**; a single re-run overwrites `full_splits/all_5_datasets/*.csv` with the new upstream scores.

---

## 2. Feature Extraction Replacement (new behaviour, same CLI surface)

### 2.1 Common changes (applied to all three extractors; default behaviour stays backward-compatible)

- `scripts/features/extract_entropy_curves.py`
- `scripts/features/extract_sae_features_musicdiscovery.py`
- `scripts/features/extract_loss_curves.py` + `scripts/loss/extract_per_token_loss.py`

New options (off by default):
- `--chunk-sec FLOAT` (default 0) — chunked-inference window length.
- `--pool-full-song` / `--no-pool-full-song` (default False) — whether to uniform-pool the concatenated long sequence to `fixed-time-steps`.
- `--max-audio-sec FLOAT` (replaces the hard-coded `MAX_AUDIO_SECONDS=30`; default still 30).
- Optional `begin_s, end_s` columns in the split CSV are read as a rater-aligned window (when present).

The SAE extractor also gets a pre-existing bug fixed: zero-padding for short wavs is replaced with **truncate-only, no padding** (matching `local4server`).

### 2.2 Per-dataset extraction settings (CLI flags injected via config)

| dataset | flags | effect |
|---|---|---|
| `musicpref` | use defaults | 30 s clip (already validated locally) |
| `aime` | `--max-audio-sec 10` + read `begin_s/end_s` | exact 10 s survey window per row |
| `music_arena` | `--chunk-sec 30 --pool-to-frames 1500 --max-audio-sec 180` + read `listen_sec` | rater-aligned ≤ 180 s → 30 s chunks → uniform-pool to 1500 frames |
| `songeval` | `--chunk-sec 30 --pool-full-song --max-audio-sec 0` | full song → 30 s chunks → uniform-pool to 1500 frames |
| `all_5_datasets` | per-row source column picks the matching rule above | per-subset fidelity preserved under merge |

Loss / entropy / SAE **must be passed the same flag set**, otherwise the three time axes won't align to 1500 frames and the hybrid channel-stacking breaks.

### 2.3 Paths

All feature / manifest CSV / shard directory paths **stay unchanged**:
- `phase7_release/outputs/full/features/entropy/<dataset>/clean/`
- `phase7_release/outputs/full/features/sae/<dataset>/clean/`
- `phase7_release/outputs/full/features/loss/<dataset>/clean/`

The old AIME / MusicPref / MusicArena features under those directories are **overwritten in place** by this iteration (same dataset, same path, same filename). The `musiceval/` subtree is **never touched** (see §3).

---

## 3. One-shot Script (overwrites the old results)

Reuse the existing `scripts/run/run_full_14_experiments_parallel.{sh,py}` + `run_musiceval_14_experiments.py` — **no new entry point** — and do only three things:

1. **Prepend a scoring stage to the parallel runner** (run once before `run_full_14_experiments_parallel.py` spawns dataset sub-jobs):
   ```bash
   python scripts/data/fit_pairwise_manifests.py --dataset musicpref   --head musicality
   python scripts/data/fit_pairwise_manifests.py --dataset aime        --head music_quality
   python scripts/data/fit_pairwise_manifests.py --dataset musicarena
   python scripts/data/gen_full_splits.py          # overwrites full_splits/{musicpref,aime,music_arena,songeval}/*
   python scripts/data/merge_all_datasets.py       # overwrites full_splits/all_5_datasets/*
   ```
   MusicEval's `full_splits/musiceval/*` is **not rebuilt**.

   Note: **SongEval is intentionally absent from the pair-fit list above** because its scores were never pairwise to begin with — they are per-song annotations. `gen_full_splits.py` reads the labels directly from `raw_hf/SongEval/metadata.jsonl`; the change is to switch from averaging all five `[Coherence, Musicality, Memorability, Clarity, Naturalness]` dimensions to averaging only `Musicality`. That is a local edit inside `gen_full_splits.py`; no new script is needed. SongEval **features** do still have to be re-extracted because the audio window changes from a 30 s prefix to full-song chunking + pooling, but that is handled by the runner via `extract_flags`, not by the scoring stage.

2. **Per-dataset `extract_flags`** are added to each dataset entry in `config/data/full_datasets.yaml`; `run_full_14_experiments_parallel.py::_build_jobs` writes them into the per-job merged config; `run_musiceval_14_experiments.py` forwards them to the three prep steps `prep_loss_features / prep_entropy_features / prep_sae_features`.

3. **Training**: the 7-family CNN suite (see §4), using the existing `seq_len=1500`.

### 3.1 Overwrite strategy (paths unchanged)

Use the runner's `--reset-state` flag, or manually delete the corresponding `state.json` under `run_state`. Whether each artifact actually needs to be cleaned is detailed in the matrix in §6.3; here we just list every path that **will or may be overwritten**:
- Scoring CSVs to be overwritten: `data/manifests/pairwise_relu/{musicpref_musicality,aime_music_quality,musicarena}_1to5.csv`.
- Splits to be overwritten: `data/full_splits/{musicpref,aime,music_arena,songeval,all_5_datasets}/*.csv`.
- Features to be overwritten (only AIME / MusicArena / SongEval truly need re-extraction; MusicPref's window is unchanged and can be reused; `all_5` reuses per-DB features and is not extracted separately): `outputs/full/features/{entropy,sae,loss}/{aime,music_arena,songeval}/clean/**`.
- Training artifacts to be overwritten (all 5 datasets are retrained): `outputs/full/{checkpoints,plots,reports,logs}/{musicpref,aime,music_arena,songeval,all_5_datasets}/clean/**`.
- To be **removed** (never regenerated): `data/manifests/pairwise_relu/{musicpref_fidelity,aime_text_audio_alignment}_1to5.csv`.
- **Never touched**: anything under `musiceval/`, `data/splits/musiceval/*`, `data/full_splits/musiceval/*`, `outputs/full/*/musiceval/**`, `reports/musiceval/**`.

---

## 4. Remove Transformer and Noisy (full slim-down across code + config)

### 4.1 Delete transformer

- Directory removals: `phase7_release/training/curve_transformer/`, `phase7_release/training/hybrid/train_transformer_pool.py`, `train_transformer.py`.
- Config removals: `config/model/loss_curve_transformer.yaml`; the transformer sections in `config/model/hybrid.yaml` and `config/training/hybrid.yaml`; the three entries `curve_transformer / hybrid_transformer_pool / hybrid_transformer` in `config/training/defaults`; every `*transformer*` file under `config/model/experiments/` and `config/training/experiments/`.
- Runner slim-down:
  - `run_musiceval_14_experiments.py`: drop the `transformer_step_names` set, the `--skip-transformer` argument, every `f0x_*_transformer` Step, and the transformer-related batch-size / prefetch arguments.
  - `run_full_14_experiments_parallel.py`: no transformer-specific fields to drop (it only forwards via the runner), but verify no stale flags remain.
- Documentation: `doc/plan/experiment_code_map.md` shrinks to 7 entries (CNN only); `doc/plan/phase7_release_plan.md` changes "7 × 2 = 14" to "7 × 1 = 7"; `scripts/eval/eval_14_experiments.py` keeps its name but its internal enumeration lists CNN families only.

### 4.2 Delete noisy

- Config removals: `config/paths.yaml::data.splits.noisy`; every `splits.noisy` block in `config/data/full_datasets.yaml`; any `*_noisy.yaml` if present.
- Data-dir cleanup: delete `data/splits/musiceval/*_noisy.csv` and `data/full_splits/*/*_noisy.csv`.
- Runner slim-down:
  - `scripts/data/preprocess_data.py::_musiceval_copy_splits`: keep only the `clean` mapping.
  - `scripts/data/gen_full_splits.py`: remove the `.to_csv(..._noisy.csv)` line.
  - `scripts/data/merge_all_datasets.py`: the `splits` list keeps only `train/val/test`.
  - `run_full_14_experiments_parallel.sh/.py`: remove the `SPLITS` / `--splits` noisy branch (default clean); make `--splits` non-optional or drop it.
  - `run_musiceval_14_experiments.py`: remove every branch reachable via `--splits noisy`; delete `noisy_flag` / `use_noisy_splits`.
  - `scripts/features/*` / `scripts/loss/*`: hard-code `--splits` to `"clean"` or remove it altogether.
- Documentation: remove the "noisy-labelled version" instructions block from `phase7_release_plan.md`.

### 4.3 New 7-experiment matrix

Keep the CNN versions of the 7 families: `f01_loss_only_cnn`, `f02_entropy_only_cnn`, `f03_sae_only_cnn`, `f04_loss_entropy_cnn`, `f05_entropy_sae_cnn`, `f06_loss_sae_cnn`, `f07_loss_entropy_sae_cnn`. Total workload = `5 datasets × 7 + merged 1 × 7 = 42 CNN runs`. MusicEval never triggers.

---

## 5. What is NOT Changed (explicit list)

- All CNN code and CLI under `training/{loss_curve,entropy_curve,hybrid}/*`.
- `lib/repro/*` (dataset / nets / metrics / data_paths / scaling).
- Every field in `config/paths.yaml` except `data.splits.noisy`.
- `config/model/{loss_curve_cnn,hybrid,sae,sae_sparse_autoencoder,sae_verifier,verifier}.yaml` (CNN / SAE sections kept, transformer sections deleted).
- MusicEval scoring, splits, features, checkpoints, reports, logs.
- The 14-experiment runner's run_tag / state / lock mechanism.
- `scripts/baseline/*`, `scripts/eval/*`, `scripts/analysis/*` (the CNN-related portions).

---

## 6. Execution Order

1. **Phase A (code; waits for review)**
   1. Edit `phase7_local4server/training/build_musicpref_scores.py` to drop fidelity; run once to verify.
   2. Add `phase7_release/lib/elo_scoring.py`; delete `phase7_release/lib/pairwise_relu_scores.py`.
   3. Rewrite `scripts/data/fit_pairwise_manifests.py` (filenames / CLI / output paths unchanged; kernel replaced with Elo; `fidelity / text_audio_alignment` branches now raise and exit).
   4. Rewrite `scripts/data/gen_full_splits.py` (score source per §1.4).
   5. Update all three feature extractors' CLI (add chunk / pool / max-audio-sec / `begin_s`, `end_s` support with defaults unchanged); fix the SAE zero-padding bug.
   6. Add `extract_flags` to each dataset entry in `config/data/full_datasets.yaml`; wire them through the runners.
   7. Clean up transformer / noisy (every deletion listed in §4).

2. **Phase B (run on server and overwrite)** — detailed in §6.2.

---

## 6.1 The Two Runners' Division of Labor (must be clear)

| runner | role | processes | GPU usage | typical scenario |
|---|---|---|---|---|
| `scripts/run/run_musiceval_14_experiments.py` | **Single dataset**, runs all 7 steps serially: `prep_loss_features / prep_sae_features / prep_entropy_features / baselines (optional) / f01..f07 CNN / eval_14_experiments`. | 1 | 1 GPU by default (pinned via external `CUDA_VISIBLE_DEVICES`); every step runs sequentially. | the small-scale MusicEval gate; or used as a sub-process by the full parallel runner. |
| `scripts/run/run_full_14_experiments_parallel.py` | **Multi-dataset orchestrator**. Reads `config/data/full_datasets.yaml`, enumerates datasets via `execution.include_scopes`; probes `nvidia-smi` for usable GPUs; launches one `run_musiceval_14_experiments.py` sub-process per dataset, pinning `CUDA_VISIBLE_DEVICES=<gpu_id>`; concurrency ≤ `gpu_parallel.max_concurrent_jobs`. | N (one per dataset) | 1 GPU per child, up to 8 concurrent. | full-scale: 5 single datasets + merged. |

**On the server you must use the parallel runner**: on 8×H100, running the 5 single datasets sequentially leaves 7 cards idle most of the time; parallelism collapses the full-scale wall time down to roughly `single-dataset time × ceil(5 / concurrency)`. **The small-scale MusicEval gate does not need parallelism** (single dataset, single GPU, and this iteration does not rerun it anyway).

---

## 6.2 Phase B — One-shot Server Plan

Only operate on the 5 datasets in the **clean × CNN × 7** matrix (`musicpref / aime / songeval / music_arena / all_5_datasets`); MusicEval is not touched.

> **Entry point**: `phase7_release/scripts/run/1st_iter.sh`
> Internally executes §6.2.1 → §6.2.4 in order; supports `--skip-{clean,scoring,singles,merged}` and `--dry-run`.
> Overlays are written to `phase7_release/outputs/run_state/1st_iter/full_datasets_{single,merged}.yaml`; `config/data/full_datasets.yaml` is **not mutated**.
>
> **The AIME audio filename fix (Option B) has been split out of 1st iter** into a separate entry point `phase7_release/scripts/run/2nd_iter.sh`; see §7.

### 6.2.1 Step-0: Clean Old Artifacts (runs locally or on server; only deletes, does not produce)

**Targets only the 4 affected single DBs + `all_5`**, and runs exactly once. Execute from `PROJECT_ROOT`:

```bash
# 1. Old scoring manifests (delete the ones the new Elo rules will overwrite + the two that are retired outright)
rm -f phase7_release/data/manifests/pairwise_relu/{musicpref_musicality,aime_music_quality,musicarena}_1to5.csv
rm -f phase7_release/data/manifests/pairwise_relu/{musicpref_fidelity,aime_text_audio_alignment}_1to5.csv

# 2. Old full_splits (MusicEval retained)
for ds in musicpref aime music_arena songeval all_5_datasets; do
  rm -rf "phase7_release/data/full_splits/${ds}"
done

# 3. Old features (only AIME / MusicArena / SongEval / all_5 need cleaning; MusicPref's 30 s features are unchanged and can be reused)
for ds in aime music_arena songeval all_5_datasets; do
  rm -rf "phase7_release/outputs/full/features/loss/${ds}"    \
         "phase7_release/outputs/full/features/entropy/${ds}" \
         "phase7_release/outputs/full/features/sae/${ds}"
done

# 4. Old training artifacts + run_state (5 datasets)
for ds in musicpref aime music_arena songeval all_5_datasets; do
  rm -rf "phase7_release/outputs/full/checkpoints/${ds}" \
         "phase7_release/outputs/full/plots/${ds}"       \
         "phase7_release/outputs/full/reports/${ds}"     \
         "phase7_release/outputs/full/logs/${ds}"
  rm -f  "phase7_release/outputs/run_state/full14_${ds}_clean.state.json" \
         "phase7_release/outputs/run_state/full14_${ds}_clean.lock.json"
done
```

The `musiceval/` subtree, `phase7_release/data/splits/musiceval/`, and `outputs/**/musiceval/` are all **left alone**.

### 6.2.2 Step-1: Re-score and Re-split (CPU-only; seconds to a few minutes)

This step is **not** driven by the parallel runner; it is a prerequisite that must finish first:

```bash
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicpref --head musicality \
  --out phase7_release/data/manifests/pairwise_relu/musicpref_musicality_1to5.csv
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset aime --head music_quality \
  --out phase7_release/data/manifests/pairwise_relu/aime_music_quality_1to5.csv
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicarena \
  --out phase7_release/data/manifests/pairwise_relu/musicarena_1to5.csv

conda run -n torch21 python phase7_release/scripts/data/gen_full_splits.py
conda run -n torch21 python phase7_release/scripts/data/merge_all_datasets.py
```

SongEval is intentionally not in the `fit_pairwise_manifests` list: `gen_full_splits.py` reads `Musicality` averages straight from `raw_hf/SongEval/metadata.jsonl` to form `score`.

Outputs (all overwrite the old files):
- `data/manifests/pairwise_relu/{musicpref_musicality,aime_music_quality,musicarena}_1to5.csv`
- `data/full_splits/{musicpref,aime,music_arena,songeval}/{train,val,test}.csv`
- `data/full_splits/all_5_datasets/{train,val,test}.csv`

### 6.2.3 Step-2: Full-Scale Parallel Execution (one GPU per dataset)

The parallel runner already schedules all 5 datasets by default; on an 8×H100 server it can run up to 8 concurrent jobs (5 active in this iteration). **Key caveat: `all_5_datasets` must reuse per-DB features and therefore has to be scheduled in a second pass separate from the 4 single DBs** (see §6.3 for why):

```bash
# (a) First pass: run the 4 single DBs concurrently — feature extraction + 7 CNN trainings
FULL_CFG=phase7_release/config/data/full_datasets.yaml
#  -> Edit full_datasets.yaml so execution.include_scopes is temporarily
#     ["large_scale_single"], or pass --full-config pointing at an overlay copy.

conda run -n torch21 python phase7_release/scripts/run/run_full_14_experiments_parallel.py \
  --splits clean

# Once (a) finishes, run (b): the merged job. Prerequisite: per-DB features exist.
#  -> Flip include_scopes back to ["large_scale_merged"] and set
#     execution.skip_feature_extract to true (no re-extraction; all_5 reads
#     token_loss_path directly from per-DB feature directories).
conda run -n torch21 python phase7_release/scripts/run/run_full_14_experiments_parallel.py \
  --splits clean
```

> Each dataset child shows a tqdm progress bar; logs go to `outputs/run_state/full_parallel_logs/<scope>_<dataset>_clean.log`, and the summary to `outputs/run_state/full_parallel_summary_clean.csv`.

### 6.2.4 Step-3: Aggregation and Sanity Check

The parallel runner automatically runs `scripts/eval/eval_14_experiments.py` as the last step of each dataset's child process; it aggregates the 7 `f0?_*_cnn_clean_test_scores.csv` files into:

- `outputs/full/reports/<dataset>/clean/tables/aggregate_table_14_experiments_clean.{csv,md}`
- Scatter plots under `outputs/full/reports/<dataset>/clean/plots/`.

Recommended server-side sanity check:

```bash
# All 5 datasets should have one summary + 7 test-score CSVs
for ds in musicpref aime music_arena songeval all_5_datasets; do
  ls phase7_release/outputs/full/reports/${ds}/clean/tables/aggregate_table_14_experiments_clean.md || echo "MISSING ${ds}"
  ls phase7_release/outputs/full/reports/${ds}/clean/f0?_*_cnn_clean_test_scores.csv | wc -l
done
```

---

## 6.3 Exact Re-extraction / Retraining Matrix

This expands the one-line note Phase B §3 used to have.

| dataset | scoring changed? | audio window changed? | re-extract loss/entropy/sae? | retrain 7 CNNs? |
|---|---|---|---|---|
| `musiceval` | no | no | **no** (subtree left alone) | **no** |
| `musicpref` | yes (Elo × musicality-only) | no (still 30 s clips) | **not strictly needed**: `extract_flags` match the old rule; but Step-0 does not wipe these features, and the parallel runner's extract step will fast-skip them (manifest may be rewritten). | yes (scores changed) |
| `aime` | yes (system Elo + logit winrate) | yes (10 s rater window via `begin_s/end_s`) | **yes** | yes |
| `music_arena` | yes (system Elo + 4-outcome soft target) | yes (rater ≤ 180 s → 30 s chunks → pool 1500) | **yes** | yes |
| `songeval` | yes (Musicality mean only) | yes (full song → 30 s chunks → pool 1500) | **yes** | yes |
| `all_5_datasets` | yes (inherited from the 4 above) | features are **reused** directly from per-DB feature directories | **no** (no re-extraction; see §6.3.1) | yes |

### 6.3.1 `all_5_datasets` Features Must Be *Reused*, Not *Re-extracted*

The merged entry in `full_datasets.yaml` has an empty `extract_flags: {}` — if you actually invoked the extractors on the merged split, the SongEval / MusicArena full-song / long-clip data would be processed with the **default 30 s, no-chunk** global rule, which is wrong. The correct procedure:

1. First finish §6.2.3 (a): per-DB extract populates `outputs/full/features/{loss,entropy,sae}/<dataset>/clean/`.
2. When training the merged dataset, **do not invoke extractors**: set `skip_feature_extract: true` in `full_datasets.yaml::execution` and invoke the parallel runner a second time (§6.2.3 (b)).
3. Every row's `token_loss_path` in `all_5_datasets/{train,val,test}.csv` is already in `<source>/<file>` form (the output convention of `gen_full_splits.py`), so the merged runner's `token_loss_root` must point at `outputs/full/features/loss/` (**the parent**) rather than `.../loss/all_5_datasets/clean/`. The current `run_full_14_experiments_parallel.py::_build_jobs` writes a per-dataset subdirectory even for `all_5_datasets`; a **small patch** is needed: when `name == "all_5_datasets"`, set `token_loss_root / features_entropy / sae.output_dir` to the shared parent directory and force `--skip-feature-extract`.

This patch is not part of Phase A; the recommendation is to land it after Phase A's 7 items finish and before Phase B runs — complexity < 30 lines.

---

## 7. 2nd Iter — AIME Audio Filename Fix (Option B)

> **Entry point**: `phase7_release/scripts/run/2nd_iter.sh`
> Only run this after the 1st iter completes, when the server shows AIME correlation stuck around ~0.3.
> Touches only AIME + `all_5_datasets`; the other four DBs (musicpref / musicarena / songeval / musiceval) are **not touched**.

### 7.1 Root Cause

- The old `phase7_release/scripts/data/hf_ingest_smoke.py` named WAVs by `row_index`: `AIME2025_0.wav`, `AIME2025_1.wav`, ....
- `phase7_release/scripts/data/aime_join_survey.py` keys tracks by the HF `track_1_id / track_2_id` (integer, zero-padded to 5 digits, e.g. `"05331"`); the AIME manifest produced by `fit_pairwise_manifests.py` is therefore indexed by `track_id`.
- When `gen_full_splits.py::_build_aime` reads that manifest, pandas silently coerces `"05331"` back to the integer `5331`, so the constructed `audio_path` becomes `AIME2025_5331.wav`.
- The two naming schemes (`<row_index>` vs `<track_id>`) are in different coordinate systems: they only accidentally line up when HF `disco-eth/AIME` happens to be sorted by `id` and the `id` has no leading zeros. Otherwise labels and audio decouple, the model learns from a random audio-label pairing, and Pearson hovers around 0.3.

### 7.2 Option B (the code changes already landed in this repo)

| file | change |
|---|---|
| `phase7_release/scripts/data/hf_ingest_smoke.py` | Uses `row["id"]` as the filename stem (fallback order `id → item_id → track_id → track_id_str`), producing `AIME2025_<id>.wav`; the master CSV records both `id` and `row_index` for diagnostics. |
| `phase7_release/scripts/data/gen_full_splits.py::_build_aime` | `pd.read_csv(..., dtype={"track_id": str})` prevents pandas from coercing `"05331"` back to `5331`. A new `_aime_audio_path()` helper first tries the 5-digit zfill form (`AIME2025_05331.wav`) and falls back to the bare integer (`AIME2025_5331.wav`), so both HF id conventions resolve. `_aime_token_loss_path()` reuses whichever stem actually hit, keeping loss/entropy/SAE filenames aligned with the audio. |

> **Other DBs are unaffected by this bug**: MusicPref / MusicArena / SongEval / MusicEval all carry `audio_path` directly in their manifests and do not rely on an implicit `row_index → filename` mapping; no re-extraction is needed.

### 7.3 Server-side Execution Order (orchestrated by `2nd_iter.sh`)

Supports `--skip-{ingest,clean,scoring,singles,merged}` and `--dry-run`. Overlays go to `phase7_release/outputs/run_state/2nd_iter/full_datasets_{single_aime_only,merged}.yaml`; `config/data/full_datasets.yaml` is **not mutated**.

1. **Stage 0: Re-ingest AIME audio (id-aware)**

   ```bash
   rm -rf phase7_release/datasets/aime/audio
   mkdir -p phase7_release/datasets/aime/audio
   conda run -n torch21 python phase7_release/scripts/data/hf_ingest_smoke.py \
     --repo disco-eth/AIME --source-tag AIME2025 \
     --out-audio-dir phase7_release/datasets/aime/audio \
     --master-csv phase7_release/data/manifests/master_index.csv \
     --max-samples 0
   ```

2. **Stage 1: Clean only AIME + all_5 artifacts** (manifest / full_splits / features / checkpoints / reports / run_state); the other three single-DB artifacts stay.
3. **Stage 2: Re-score AIME → `gen_full_splits` → `merge_all_datasets`** (only AIME's score is new; the other DBs' manifests are unchanged and `gen_full_splits` is idempotent for them).
4. **Stage 3a: parallel runner, with only AIME `enabled: true` in the `large_scale_single` overlay** and the others `enabled: false`; extract + train AIME.
5. **Stage 3b: parallel runner, `large_scale_merged` + `skip_feature_extract: true`**, trains the all_5 CNNs (depends on the §6.3.1 patch).
6. **Stage 4 sanity check**: AIME + all_5 summary tables + 7 `f0?_*_cnn_clean_test_scores.csv` each.

### 7.4 Common Entry Points

```bash
# Full 2nd-iter delta:
bash phase7_release/scripts/run/2nd_iter.sh

# Audio already re-ingested; only retrain AIME + all_5:
bash phase7_release/scripts/run/2nd_iter.sh --skip-ingest

# Only re-run AIME single (skip merged):
bash phase7_release/scripts/run/2nd_iter.sh --skip-merged

# Only re-run merged all_5 (AIME already trained):
bash phase7_release/scripts/run/2nd_iter.sh --skip-ingest --skip-clean --skip-scoring --skip-singles
```
