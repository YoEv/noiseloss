**[中文 README](README.md)** · **[English README](README.eng.md)**
# NoiseLoss

**NoiseLoss** probes how music language models react when coherence is broken.
We inject structured perturbations (noise, deletions, shuffles), trace token-level loss and gradients, and map the dynamics of **Peak → Assimilation → Recovery**.

> When noise enters, loss spikes.
> When adaptation follows, loss falls.
> Between them lies the trace of forgetting.

- **Project**: https://noiseloss.github.io
- **Paper (PDF)**: https://noiseloss.github.io/paper/paper.pdf

## Overview

This repository contains the code and scripts used to:
- Generate perturbed and reference audio/token sequences for Music LLMs.
- Compute per-token **likelihood (NLL)** and extract **gradients** on activations/parameters.
- Visualize layer/time behaviors showing *context amnesia* under localized noise.

Our experiments reveal that **likelihood can deceptively improve** after the model adapts to injected noise, calling for a rethink of likelihood-based evaluation in Music LLMs.

## Repository Layout

```
noiseloss/
├── dataselect/        # Data selection and sampling scripts (MIDI/audio filtering, rhythm/duration selection, etc.)
├── Dataset/           # Dataset materials (local placement; recommend .gitignore)
├── external/          # Third-party dependencies/external tool wrappers or submodules
├── Loss/              # Scripts and artifacts related to loss comparison (legacy directory, retained)
├── losscal/           # Loss calculation/calibration scripts (legacy directory, retained)
├── LossPlot/          # Chart scripts or finished plots directly related to loss themes (legacy directory, retained)
├── plot/              # General plotting scripts and results (line plots, box plots, histograms, etc.)
├── processors/        # Experiment processing entry points (perturbation, generation, measurement, phased scripts, etc.)
└── tools/             # Utility tools (transcoding, tokenizer, EnCodec/FFmpeg/SOX, etc.)
```

## Experimental Phases

**Phase 1–4**
Basic perturbations and position/length scanning: fixed and random injection, different noise colors, replacement and shuffling.

**Phase 5**
Regionalized analysis (*Peak / Assimilation / Recovery*):
- White/pink/brown noise, token deletion and order shuffling
- Conditional/unconditional generation comparison
- Statistical distributions (box plots, violin plots), cross-track aggregation

**Phase 6 (Current)**
Gradient protocols:
- Per-token forward + per-token backpropagation, extracting **∇activation** and **∇parameter**
- Module-wise (Attention Q/K/V/O, FFN, LayerNorm) and layer-wise heatmaps
- Observing gradient flux evolution from spike to adaptation to recovery

## Quick Start


**1) Regionalized Loss Curves (Phase 5)**
```bash
python processors/phase5/phase5_loss_curve_3area.py
```

**2) Loss Comparison Visualization on Timeline**

```bash
python plot/plot_loss_time_in.py
```

**3) Unconditional / Conditional Comparison (Examples)**

```bash
python plot/plot_comparison_unconditional.py
python plot/plot_conditional.py
```

**4) Common Tools (Examples)**

```bash
# Batch audio transcoding
python tools/convert_audio_ffmpeg.py --in_dir Dataset/raw --out_dir Dataset/wav --sr 32000

# Calculate EnCodec receptive field
python tools/encodec_receptive_field_calculator.py
```

## Notes & Practices

* **Right shift**: Per-token NLL needs to align with the next token (`logits[..., t]` vs `target[..., t+1]`).
* **Per-token backpropagation**: Use `retain_graph=True`; tokens outside the window can be subsampled for acceleration.
* **Hook points**: Use `retain_grad()` at residual/intermediate layer outputs to read **∇h**.
* **Parameter bucketing**: Attention (Q/K/V/O), FFN (W1/W2/bias), LayerNorm (γ/β) are counted separately; store both layer×module and per-head statistics.
* **Reproducible experiments**: Save `meta.json` (Git commit hash, configuration, package versions) and store matrix drafts as `npy/parquet` files.


## Citation

If you reference or build upon this work, please cite:

```
@inproceedings{Li2025NoiseLowersLoss,
  title   = {When Noise Lowers the Loss: Rethinking Likelihood-Based Evaluation in Music LLMs},
  author  = {Xiaosha Li and Chun Liu and Ziyu Wang},
  year    = {2025},
  note    = {Georgia Institute of Technology; ByteDance Inc.; Courant Institute of Mathematical Sciences, NYU; MBZUAI},
  url     = {https://noiseloss.github.io/paper/paper.pdf}
}
```

**Title**
**WHEN NOISE LOWERS THE LOSS: RETHINKING LIKELIHOOD-BASED EVALUATION IN MUSIC LLMS**

**Authors**
Xiaosha Li¹, Chun Liu², Ziyu Wang³⁴
¹ Georgia Institute of Technology
² ByteDance Inc.
³ Courant Institute of Mathematical Sciences, New York University
⁴ Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Abu Dhabi, UAE
Contact: `xiaosha@gatech.edu`, `chun.liu@bytedance.com`, `ziyu.wang@nyu.edu`


## License

This repository is for research and academic use.
For other uses, please contact the authors.

> Loss can reveal more than accuracy:
> it tells when a model listens, when it forgets, and when it learns to live with noise.

