# NoiseLoss

**NoiseLoss** probes how music language models react when coherence is disrupted.
We inject structured perturbations—noise, deletions, and shuffles—trace token-level loss and gradients, and map the dynamics of **Peak → Assimilation → Recovery**.

> When noise enters, loss spikes.
> When adaptation follows, loss falls.
> Between them lies the trace of forgetting.

* **Project**: [noiseloss.github.io](https://noiseloss.github.io)
* **Paper (PDF)**: [noiseloss.github.io/paper/paper.pdf](https://noiseloss.github.io/paper/paper.pdf)

---

## Overview

This repository contains the code and scripts used to:

* Generate perturbed and reference audio/token sequences for Music LLMs.
* Compute per-token **likelihood (NLL)** and extract **gradients** on activations and parameters.
* Visualize layer- and time-wise behaviors to reveal *context amnesia* under localized noise.

Our experiments show that **likelihood can deceptively improve** after a model adapts to injected noise—calling for a rethink of likelihood-based evaluation in Music LLMs.

---

## Repository Layout

```
noiseloss/
├── dataselect/        # Data selection and sampling scripts (MIDI/audio filtering, rhythm/length selection, etc.)
├── Dataset/           # Dataset materials (local only; recommended to .gitignore)
├── external/          # Wrappers or submodules for third-party dependencies/tools
├── Loss/              # Legacy scripts and artifacts related to loss comparison
├── losscal/           # Legacy scripts for loss computation/calibration
├── LossPlot/          # Legacy directory for loss-related figures and plotting scripts
├── plot/              # General plotting scripts and results (line, box, histogram, etc.)
├── processors/        # Experimental pipelines (perturbation, generation, measurement, phase-by-phase scripts)
└── tools/             # Utility tools (transcoding, tokenizer, EnCodec/FFmpeg/SOX, etc.)
```

---

## Experimental Phases

**Phase 1–4**
Basic perturbations and position/length scanning: fixed vs. random injections, noise color variations, token replacement, and order shuffling.

**Phase 5**
Regional analysis (*Peak / Assimilation / Recovery*):

* White / pink / brown noise, token deletion and shuffling
* Conditional vs. unconditional generation
* Distribution statistics (box/violin plots), cross-piece aggregation

**Phase 6 (Current)**
Gradient protocol:

* Per-token forward and backward passes, extracting **∇activation** and **∇parameter**
* Heatmaps by module (Attention Q/K/V/O, FFN, LayerNorm) and by layer
* Observing gradient flux from spike → adaptation → recovery

---

## Quick Start

**1) Regionalized loss curve (Phase 5)**

```bash
python processors/phase5/phase5_loss_curve_3area.py
```

**2) Loss comparison over time**

```bash
python plot/plot_loss_time_in.py
```

**3) Unconditional / Conditional comparison (example)**

```bash
python plot/plot_comparison_unconditional.py
python plot/plot_conditional.py
```

**4) Common utilities (examples)**

```bash
# Batch audio transcoding
python tools/convert_audio_ffmpeg.py --in_dir Dataset/raw --out_dir Dataset/wav --sr 32000

# Calculate EnCodec receptive field
python tools/encodec_receptive_field_calculator.py
```

---

## Notes & Practices

* **Right shift**: Per-token NLL must align with the next token (`logits[..., t]` vs. `target[..., t+1]`).
* **Per-token backward**: Use `retain_graph=True`; subsample tokens outside the target window for efficiency.
* **Hook points**: Use `retain_grad()` at residual/intermediate outputs to capture **∇h**.
* **Parameter bucketing**: Track separately for Attention (Q/K/V/O), FFN (W1/W2/bias), and LayerNorm (γ/β); store both layer×module and per-head statistics.
* **Reproducibility**: Save `meta.json` (Git hash, config, package versions) and all result matrices as `npy`/`parquet`.

---

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

---

## License

This repository is intended for research and academic use.
For other uses, please contact the authors.

> Loss can reveal more than accuracy:
> it tells when a model listens, when it forgets, and when it learns to live with noise.
