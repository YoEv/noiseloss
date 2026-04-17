# Data Contract

This folder stores lightweight manifests and documentation for release usage.

## Required split files

- `manifests/train.csv`
- `manifests/val.csv`
- `manifests/test.csv`
- Optional noisy labels:
  - `manifests/train_noisy.csv`
  - `manifests/val_noisy.csv`
  - `manifests/test_noisy.csv`

## Required columns

For training and evaluation pipelines, each row should include:

- `score`
- `audio_path`
- `token_loss_path`

## Precomputed SAE features

Feature files are expected in the legacy exp11 location by default:

- `sae_features_{split}.npy`
- `sae_features_{split}_meta.pt`

Optional old SAE variant:

- `sae_features_{split}_oldsae.npy`
- `sae_features_{split}_oldsae_meta.pt`
