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
├── dataselect/        # 数据筛选与抽样脚本（MIDI/音频过滤、节奏/时长选择等）
├── Dataset/           # 数据集素材（本地放置；建议 .gitignore）
├── external/          # 第三方依赖/外部工具的封装或子模块
├── Loss/              # 与损失比较相关的脚本与产物（历史目录，保留）
├── losscal/           # 计算/校准损失的脚本（历史目录，保留）
├── LossPlot/          # 与损失主题直接相关的图表脚本或成品图（历史目录，保留）
├── plot/              # 通用绘图脚本与结果（折线、箱线、直方图等）
├── processors/        # 实验处理入口（扰动、生成、度量、phase 分阶段脚本等）
└── tools/             # 实用工具（转码、tokenizer、EnCodec/FFmpeg/SOX 等）
```

## Experimental Phases

**Phase 1–4**  
基础扰动与位置/长度扫描：固定与随机注入、不同噪声颜色、替换与置乱。

**Phase 5**  
区域化分析（*Peak / Assimilation / Recovery*）：  
- 白/粉/棕噪、token 删除与顺序打乱  
- 条件/非条件生成对比  
- 统计分布（箱线、提琴图）、跨曲目聚合

**Phase 6（当前）**  
梯度协议：  
- 逐 token 前向 + 逐 token 反传，提取 **∇activation** 与 **∇parameter**  
- 分模块（Attention Q/K/V/O、FFN、LayerNorm）与分层热力图  
- 观察从尖峰到适应再到恢复的梯度通量演化

## Quick Start


**1) 区域化损失曲线（Phase 5）**
```bash
python processors/phase5/phase5_loss_curve_3area.py
````

**2) 时间轴上的损失对比可视化**

```bash
python plot/plot_loss_time_in.py
```

**3) Unconditional / Conditional 对比（示例）**

```bash
python plot/plot_comparison_unconditional.py
python plot/plot_conditional.py
```

**4) 常用工具（示例）**

```bash
# 音频批量转码
python tools/convert_audio_ffmpeg.py --in_dir Dataset/raw --out_dir Dataset/wav --sr 32000

# 计算 EnCodec receptive field
python tools/encodec_receptive_field_calculator.py
```

## Notes & Practices

* **Right shift**：per-token NLL 需与下一 token 对齐（`logits[..., t]` 对 `target[..., t+1]`）。
* **逐 token 反传**：使用 `retain_graph=True`，对窗口外 token 可子采样以加速。
* **Hook 点**：在 residual/中间层输出处 `retain_grad()` 以读取 **∇h**。
* **参数分桶**：Attention（Q/K/V/O）、FFN（W1/W2/偏置）、LayerNorm（γ/β）分别统计；既存逐层×模块，也存逐头。
* **复现实验**：保存 `meta.json`（Git 提交哈希、配置、包版本）、把矩阵底稿以 `npy/parquet` 落盘。


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

