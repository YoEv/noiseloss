# Phase7 Release — 执行与运维

所有入口在 `phase7_release/scripts/run/`。每个脚本**幂等 + 可断点续跑**：step 完成状态写入 `phase7_release/outputs/run_state/<run_tag>.state.json`，重跑会自动跳过已完成步骤。

推荐顺序：A → B → C → D → E。

---

## Phase A — 一次性环境准备

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/setup_environments.sh       # torch21 + audiobox
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh # SAE deps + checkpoint（装入 torch21）
```

验证：

```bash
conda env list | grep -E "torch21|audiobox"
conda run -n torch21 python -c "import torch; print(torch.cuda.device_count())"
ls external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/
```

---

## Phase B — MusicEval 7 实验（小规模闸门）

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
```

脚本依次执行：数据校验 → loss/SAE/entropy 特征提取 → baselines → f01~f07（CNN）→ 汇总评估。
Transformer 骨干、`--splits noisy` 以及 `--skip-transformer` 开关都已移除；目录 / 文件名沿用 `14_experiments` 不变。

常用开关：

```bash
--skip-sae                  # 跳过 f03/f05/f06/f07 的 SAE 部分
--no-with-aesthetics        # 跳过 audiobox baseline（省一个环境）
--rerun-step <step_name>    # 强制重跑某个 step
--reset-state               # 彻底重置状态
```

**通过条件**：`outputs/reports/musiceval/clean/eval_14_experiments_summary_clean.csv` 存在，7 个 `f0?_*_cnn_clean_test_scores.csv` 齐全，无 step 处于 `running`。

预期产出结构：

```text
outputs/
  run_state/musiceval14_musiceval_clean.{state,lock}.json
  features/{loss,entropy,sae}/musiceval/clean/
  checkpoints/musiceval/clean/<run_name>/best.pth
  reports/musiceval/clean/eval_14_experiments_summary_clean.csv
  reports/musiceval/clean/eval_14_experiments_plots_clean/
```

---

## Phase C — Segment-level RNN 分析（可选）

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh                 # loss-only
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh --with-entropy  # loss + entropy
```

产出：

```text
outputs/checkpoints/segment/segment_rnn_clean_<mode>.pth
outputs/reports/segment_curves/*.csv
```

---

## Phase D — Full-scale 单库 + 合库（并发调度）

前置：`data/full_splits/` 已按 [data_prep.md](data_prep.md) 生成齐全。

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh
```

默认启用 4 个单库（`musicpref / aime / songeval / music_arena`）+ 合库（`all_5_datasets`）= **5 × 7 = 35 个主实验/次**。

**GPU 调度机制**：探测满足 `min_free_memory_mb=60000`、`max_utilization_pct≤35` 的卡，最多 8 路并发（均可在 `config/data/full_datasets.yaml → gpu_parallel` 调整）。

```bash
# 干跑（不真实启动）
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh --dry-run

# 重跑某个失败数据集
rm phase7_release/outputs/run_state/full14_<dataset>_clean.state.json
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh
```

产出：`outputs/full/reports/<dataset>/clean/eval_14_experiments_summary_clean.csv`

---

## Phase E — 汇总与交付

```text
deliverables/
  summary_musiceval_clean.csv
  summary_full_<dataset>_clean.csv   # dataset ∈ {musicpref,aime,songeval,music_arena,all_5_datasets}
  plots/      # 从 outputs/reports/**/eval_14_experiments_plots_*/ 汇总
  segment_curves/
  logs/       # run_state/*.state.json + timelines/*.csv
```

---

## 断点续跑与锁机制

- **续跑**：默认开启，step 级别记录 `completed_commands`，重启自动跳过。
- **锁**：`outputs/run_state/<run_tag>.lock.json` 防止同一 run-tag 并发启动。进程被 `kill -9` 后手动删除 lock 文件即可。
- **重跑单个 step**：`--rerun-step <step_name>`
- **彻底重置**：`--reset-state`

---

## 快速自检清单

```bash
cd "${PROJECT_ROOT}"
git submodule status
conda env list | grep -E "torch21|audiobox"
conda run -n torch21 python -c "import torch; print(torch.cuda.device_count())"
ls external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/
head -n 2 phase7_release/data/splits/musiceval/test.csv
ls phase7_release/data/full_splits/*/test.csv 2>/dev/null
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
```

任一条不通过，先修好再进入 Phase B/D。

---

## 常见问题

**`RuntimeError: Run lock exists for tag=...`**
确认无同名 Python 进程后：`rm phase7_release/outputs/run_state/<run_tag>.lock.json`

**`No eligible GPUs found`（Phase D）**
当前 GPU 不满足阈值。用 `nvidia-smi` 确认占用情况；可在 `config/data/full_datasets.yaml` 调低 `min_free_memory_mb`。

**SAE checkpoint 下载失败**
确保 S3 bucket 可访问（需境外 IP）；离线时先在跳板机下载 3 个文件后 rsync 到 `external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/`。

**`audiocraft` 与 `musicdiscovery` 依赖冲突**
`setup_sae_musicdiscovery.sh` 已内置 patch；不要手动 `pip install -r external/musicdiscovery/requirements.txt`。

**音频路径找不到**
MusicEval → 见 [data_prep.md §5.9](data_prep.md#59-musiceval-小规模音频路径特别说明)；大规模数据集 → 检查 `datasets/<name>/audio/` 是否齐全（[data_prep.md §5.3](data_prep.md#53-把音频落地到-datasetsnamedios层-2)）。

**OOM（hybrid CNN 模型）**
追加参数：`--hybrid-cnn-batch-size 8 --hybrid-cnn-sae-only-batch-size 16 --hybrid-num-workers 4 --hybrid-prefetch-factor 4`
