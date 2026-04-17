# Phase7 Release

This folder provides a clean, server-friendly implementation for Phase7 workflows.
All runnable code paths are self-contained under `phase7_release/`.

## Core workflows

1. SAE feature extraction (musicdiscovery backend)
2. Loss/Entropy/Hybrid training pipelines
3. Segment-level RNN scoring
4. MusicEval 14-experiment unified run
5. Baselines (mean loss and audiobox-aesthetics)

## Structure

- `config/`: release YAML configs (`paths.yaml`, `data/`, `model/`, `training/`, `analysis/`)
- `scripts/`: data prep, feature extraction, evaluation, baseline entrypoints
- `training/`: model training entrypoints
- `analysis/`: analysis entrypoints (segment-level scoring)
- `data/`: manifests and data contract docs
- `outputs/`: default output location for this release layer

## Experiment matrix policy

- Small-scale experiments run on `MusicEval`.
- Large-scale experiments include:
  - 5 single-dataset tracks,
  - 1 merged 5-dataset track.
- Each track runs 7 feature families with 2 backbones (`CNN`, `Transformer`).

## Usage

```bash
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
```

Experiment matrix (7 families × CNN/Transformer) is indexed in `phase7_release/config/model/experiments/registry.yaml`.

## Notes

- This release layer focuses on stable orchestration and path unification.
- Runtime defaults are managed by `phase7_release/config/paths.yaml`.
