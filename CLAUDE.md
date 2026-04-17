# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Phase7 Release — a unified ML training and analysis system for audio quality assessment / music preference prediction. The experiment matrix is **7 feature families × 2 backbones = 14 experiments per dataset track**.

All runnable code lives under `phase7_release/`. An external git submodule at `external/musicdiscovery` provides SAE feature extraction (MusicGen-small, 16384-dim).

## Running the Pipeline

```bash
# Main entry point — orchestrates all 14 training runs
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
```

Key environment variables (auto-detected from cwd; override as needed):

```bash
export PROJECT_ROOT=/absolute/path/to/noiseloss
export TORCH_ENV=torch21                    # conda env for training
export MUSICDISCOVERY_ENV=torch21           # conda env for SAE extraction (same as TORCH_ENV when python>=3.10)
export DATASET=musiceval
export SPLITS=clean                         # clean | noisy
export SKIP_TRANSFORMER=1                   # skip transformer runs
export SKIP_SAE=1                           # skip SAE feature extraction
```

Individual steps (each also callable directly):

```bash
# Feature extraction
python phase7_release/scripts/features/extract_loss_curves.py --config phase7_release/config/paths.yaml
python phase7_release/scripts/features/extract_entropy_curves.py --config phase7_release/config/paths.yaml
python phase7_release/scripts/features/extract_sae_features_musicdiscovery.py --config phase7_release/config/paths.yaml

# Training a single model
python phase7_release/training/loss_curve/train.py --config phase7_release/config/paths.yaml --splits clean
python phase7_release/training/hybrid/train_cnn.py --config phase7_release/config/paths.yaml

# Post-training analysis
python phase7_release/scripts/analysis/segment_score_curves.py --config phase7_release/config/paths.yaml --splits clean

# Evaluation
python phase7_release/scripts/eval/eval_14_experiments.py --config phase7_release/config/paths.yaml
```

There is no formal test or lint framework.

## Architecture

### Configuration

`phase7_release/config/paths.yaml` is the single runtime config file — it sets data paths, `seq_len` (1500), SAE dims (16384), and training hyperparameters for every family/backbone.

`phase7_release/config/model/experiments/registry.yaml` indexes the 14 experiment configs across 7 families (`f01`–`f07`) × 2 backbones (CNN, Transformer).

### Feature Families

| Family | Inputs |
|--------|--------|
| f01 | Loss curve only |
| f02 | Entropy curve only |
| f03 | SAE features only |
| f04 | Loss + Entropy |
| f05 | Entropy + SAE |
| f06 | Loss + SAE |
| f07 | Loss + Entropy + SAE (full hybrid) |

### Core Models (`phase7_release/lib/repro/nets.py`)

- `LossCurveCNN` — 1D Conv × 2 → MaxPool → FC
- `TransformerEncoderRegressor` — Transformer encoder + regression head (cls or mean pooling)
- `CNN1DReduceThenTransformer` — strided Conv1d to reduce seq_len, then Transformer
- `HybridTwoTowerModel` (in `training/hybrid/train_cnn.py`) — loss tower + SAE tower → fusion head

### Data Contract

Manifest CSVs (under `phase7_release/data/manifests/`) must contain:
- `score` (or `score_bt_1to5` / `score_mse_1to5`) — regression target
- `audio_path` — relative to `PROJECT_ROOT`
- `token_loss_path` — pickled/pt per-token loss sequence
- `entropy_curve_path` — optional, for entropy families
- `sae_feature_path` — optional, for SAE families

### Key Library (`phase7_release/lib/repro/`)

| File | Role |
|------|------|
| `nets.py` | All model class definitions |
| `loss_dataset.py` | `LossCurveDataset` — loads token loss sequences |
| `entropy_dataset.py` | `EntropyCurveDataset` — loads entropy codebook curves |
| `curve_channels.py` | Codebook packing/unpacking (handles masked tokens) |
| `data_paths.py` | Auto-detects `PROJECT_ROOT`; resolves relative paths |
| `metrics.py` | Pearson and Spearman correlation evaluation |
| `scaling.py` | Label scaling (BT vs MSE modes) |

### Orchestrator (`phase7_release/scripts/run/run_musiceval_14_experiments.py`)

Subprocess manager that spawns training runs with process locking, conda env activation, and state tracking. Called by the `.sh` wrapper.

## Environments

Conda-based; no `requirements.txt`. Two active environments:
- `torch21` — PyTorch 2.1, training, feature extraction (loss/entropy/SAE), evaluation. musicdiscovery deps installed here (transformers pinned at 4.38.1 by audiocraft 1.3.0).
- `audiobox` — Audiobox-aesthetics baseline

> `musicdiscovery310` is only created by `setup_sae_musicdiscovery.sh` when `torch21` is Python <3.10. Current setup uses Python 3.10, so no separate env is needed.

## Deployment

The `server-deploy` branch is lean by design (no artifacts). `.gitignore` excludes `outputs/`, `features/`, `datasets/`, and audio files. Clone to a GPU server, set `PROJECT_ROOT`, and run the orchestrator shell script.

Design documentation is in `phase7_release/doc/plan/phase7_release_plan.md`.
