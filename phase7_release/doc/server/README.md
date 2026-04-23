# Phase7 Release — Server 执行指南

从零开始复现完整流程，按 Phase A → B → C → D → E 推进。

**子文档：**
- [setup.md](setup.md) — 硬件要求、代码拉取、Conda 环境、外部依赖（§1–4）
- [data_prep.md](data_prep.md) — 数据下载与预处理（§5）
- [run.md](run.md) — 执行流程、断点续跑、自检与常见问题（§6–9）

---

## 实验矩阵

| 范围 | 数据集 | 入口脚本 | 实验数 |
|---|---|---|---|
| 小规模闸门 | MusicEval | `run_musiceval_14_experiments.sh` | 7 |
| 大规模单库 | musicpref / aime / songeval / music_arena | `run_full_14_experiments_parallel.sh` | 4 × 7 = 28 |
| 大规模合库 | `all_5_datasets` | `run_full_14_experiments_parallel.sh` | 7 |
| Segment 分析 | 复用 MusicEval 权重 | `run_segment_rnn_analysis.sh` | 1 RNN |

- Backbone 固定为 `cnn`（Transformer 已移除）。
- Split 固定为 `clean`（noisy 已移除）。
- 脚本 / 汇总表文件名沿用 `14_experiments`，不做重命名。

## PROJECT_ROOT 解析

优先级：`--project-root` CLI 参数 > `$PROJECT_ROOT` 环境变量 > 自动探测（脚本位置向上推导）。无需改任何代码，直接跑脚本即可。

```bash
# 默认（自动探测）
cd /home/cliu/wk/noiseloss
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh

# 显式覆盖
export PROJECT_ROOT=/any/other/path
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
```

---

## TL;DR — 完整执行顺序

```bash
# 1. 系统依赖
sudo apt-get install -y git git-lfs curl build-essential ffmpeg libsndfile1 && git lfs install

# 2. 代码
git clone <repo> noiseloss && cd noiseloss
git submodule update --init --recursive
export PROJECT_ROOT="$(pwd)"

# 3. Conda 环境（torch21 + audiobox）
bash phase7_release/scripts/run/setup_environments.sh
# SAE deps（装入 torch21）+ checkpoint
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh

# 4. HuggingFace 登录
hf auth login

# 5. 数据下载
bash phase7_release/scripts/run/download_hf_datasets.sh
# 音频落地 + 清单生成 + split 生成 → 详见 data_prep.md

# 6. 小规模闸门（Phase B）
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh

# 7. Segment 分析（Phase C，可选）
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh

# 8. 大规模（Phase D，前置：data/full_splits/ 已生成）
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh
```

产出主表：
- 小规模：`outputs/reports/musiceval/clean/eval_14_experiments_summary_clean.csv`
- 大规模：`outputs/full/reports/<dataset>/clean/eval_14_experiments_summary_clean.csv`
