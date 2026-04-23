# 1st-Iter Plan — 把 `phase7_local4server` 的新打分与新提取并入 `phase7_release`

## 0. 目标与约束

- **目标**：替换 `phase7_release` 里错误的**打分规则**与**30 s 提取**两件事；只做 clean × CNN 的 7 个家族，每个 dataset 独立 + `all_5_datasets` 合库。
- **硬约束**：
  1. `phase7_release` 所有既有**路径与文件名完全保持不变**（CSV 名字、manifest 路径、config key、特征目录、splits 目录、CLI 参数）；
  2. 仅**替换已有文件的实现**，不新增新目录结构；
  3. `scripts/run/run_full_14_experiments_parallel.sh --splits clean` 一键覆盖旧结果；
  4. **不动 MusicEval**（打分、splits、特征、checkpoints、reports 全部保留现状，不重跑）；
  5. **彻底移除** transformer 和 noisy：7 家族 × 1 backbone × 1 splits-variant，共 7 个实验/库。

---

## 1. 打分规则替换（只换内核，不动文件名）

### 1.1 Elo 库迁入

- 新增 `phase7_release/lib/elo_scoring.py`：从 `phase7_local4server/training/elo.py` 整体搬过来（`fit_elo`, `fit_elo_grouped`, `map_to_mos_range`, `track_score_from_system_elo` with `target_fn`）。
- **删除** `phase7_release/lib/pairwise_relu_scores.py`（不再保留 legacy）。

### 1.2 先在 local 把 MusicPref 打分改成"只用 musicality"

在开始迁移前先改 `phase7_local4server/training/build_musicpref_scores.py`：移除 fidelity 轴、移除 `--w_musicality / --w_fidelity` 参数、`track_m` 单轴直接 `map_to_mos_range`。这样 local 与 release 两边的打分规则一致。

### 1.3 `scripts/data/fit_pairwise_manifests.py` 内核替换

**文件名、CLI 名、输出 CSV 路径与列全部保持不变**，只换算法：

| 现有 CLI（保留可用的组合） | 现有输出路径（不改） | 新实现 |
|---|---|---|
| `--dataset musicpref --head musicality` | `data/manifests/pairwise_relu/musicpref_musicality_1to5.csv` | Elo：系统 Elo（7 systems）+ `track_score_from_system_elo` + robust MOS 到 `[1,5]`（本地已验证） |
| `--dataset aime --head music_quality` | `data/manifests/pairwise_relu/aime_music_quality_1to5.csv` | `build_aime_scores.py` 的 `system_elo + 400·logit_Laplace(p̂)` → robust MOS；同时把每条 track 的 `begin_s,end_s` 写到同文件新增列（CSV 向后兼容） |
| `--dataset musicarena` | `data/manifests/pairwise_relu/musicarena_1to5.csv` | `build_musicarena_scores.py` 的系统 Elo + 4-outcome context-aware 软目标 + system-span=50 压缩 + robust MOS；同时把 `listen_sec` 写到 CSV 新增列 |

**被彻底移除的 head 组合**（代码分支与旧产物都要清）：

- CLI 层：从 `--head` 的 `choices` 里删掉 `fidelity` 与 `text_audio_alignment`，并删除 `load_musicpref_pairs` 对 fidelity 列、`load_aime_pairs` 对 "Text-Audio Alignment" question_type 的分支。
- 旧产物：`data/manifests/pairwise_relu/musicpref_fidelity_1to5.csv` 与 `data/manifests/pairwise_relu/aime_text_audio_alignment_1to5.csv` 直接 `rm`（§3.1 清理清单里已列出）。

### 1.4 `scripts/data/gen_full_splits.py` 改动

所有 output 路径与列（`score, audio_path, token_loss_path`）**完全不变**；只改 score 的来源：

| dataset | 现状 | 改为 |
|---|---|---|
| `musicpref` | `(musicality + fidelity) / 2` | 直接用 `musicpref_musicality_1to5.csv`（单轴） |
| `aime` | `(music_quality + text_audio_alignment) / 2` | 直接用 `aime_music_quality_1to5.csv`（仅 Music Quality） |
| `music_arena` | `musicarena_1to5.csv` 单列 | 不变（但底层算法已换） |
| `songeval` | 5 维 × 4 标注平均 | **仅** 4 标注者的 `Musicality` 平均 |
| `musiceval` | 原生 5 分 | **保留原规则，完全不动** |

AIME / MusicArena 需要的 `begin_s,end_s` / `listen_sec` 由 `fit_pairwise_manifests.py` 写进各自的 1to5 CSV，`gen_full_splits.py` 转写到对应 `full_splits/<ds>/{train,val,test}.csv` 的额外列（不影响下游只读 `score/audio_path/token_loss_path` 的脚本）。

### 1.5 `merge_all_datasets.py`

文件名、输出路径、列**全部不变**；只因上游 score 变了，重跑一次就覆盖旧的 `full_splits/all_5_datasets/*.csv`。

---

## 2. 特征提取替换（只换行为，不换 CLI 形状）

### 2.1 通用改动（同时加到 3 个提取器，默认行为保持向后兼容）

- `scripts/features/extract_entropy_curves.py`
- `scripts/features/extract_sae_features_musicdiscovery.py`
- `scripts/features/extract_loss_curves.py` + `scripts/loss/extract_per_token_loss.py`

新增（默认关闭）：
- `--chunk-sec FLOAT`（默认 0）—— 分块推理窗口长度
- `--pool-full-song` / `--no-pool-full-song`（默认 False）—— 是否把分块拼出的长序列 uniform-pool 到 `fixed-time-steps`
- `--max-audio-sec FLOAT`（替换硬编码 `MAX_AUDIO_SECONDS=30`，默认仍 30）
- 从 split CSV 里的可选列 `begin_s,end_s` 读取 rater-aligned 窗口（若列存在）

SAE 提取器另外修一个现有 bug：短 wav 的零填充改成**只截不补**（和 local4server 一致）。

### 2.2 每个 dataset 的提取配置（CLI flags 由 config 注入）

| dataset | flags | 效果 |
|---|---|---|
| `musicpref` | 沿用默认 | 30 s 片段（本地已跑通） |
| `aime` | `--max-audio-sec 10` + 读 `begin_s/end_s` | 按 survey 精确 10 s 窗口 |
| `music_arena` | `--chunk-sec 30 --pool-to-frames 1500 --max-audio-sec 180` + 读 `listen_sec` | rater-aligned ≤180 s → 30 s 块 → uniform-pool 1500 帧 |
| `songeval` | `--chunk-sec 30 --pool-full-song --max-audio-sec 0` | 整曲 → 30 s 块 → uniform-pool 1500 帧 |
| `all_5_datasets` | 按样本 `source` 列逐条选上面对应规则 | 合库时每个子库特征保真 |

loss / entropy / SAE **必须传同一组 flags**，保证三路时间长度对齐到 1500 帧（hybrid 拼通道才能 work）。

### 2.3 路径

所有 features / manifest CSV / shard 目录路径**完全不变**：
- `phase7_release/outputs/full/features/entropy/<dataset>/clean/`
- `phase7_release/outputs/full/features/sae/<dataset>/clean/`
- `phase7_release/outputs/full/features/loss/<dataset>/clean/`

AIME / MusicPref / MusicArena 在这三个目录下的旧特征在本次迭代里会**被新特征直接覆盖**（同 dataset 同路径同文件名）。`musiceval/` 子目录**不会被触发覆盖**（见第 3 节）。

---

## 3. 一键脚本（覆盖旧结果）

复用现有 `scripts/run/run_full_14_experiments_parallel.{sh,py}` + `run_musiceval_14_experiments.py`，**不新建入口**，只做三件事：

1. **在 parallel runner 前插入一段打分步骤**（在 `run_full_14_experiments_parallel.py` 启动 dataset 子任务之前跑一次）：
   ```bash
   python scripts/data/fit_pairwise_manifests.py --dataset musicpref   --head musicality
   python scripts/data/fit_pairwise_manifests.py --dataset aime        --head music_quality
   python scripts/data/fit_pairwise_manifests.py --dataset musicarena
   python scripts/data/gen_full_splits.py          # 覆盖 full_splits/{musicpref,aime,music_arena,songeval}/*
   python scripts/data/merge_all_datasets.py       # 覆盖 full_splits/all_5_datasets/*
   ```
   MusicEval 的 `full_splits/musiceval/*` **不重建**。

   注：**SongEval 不在上面的 pair-fit 列表里**，因为它本来就是每曲直接标注（不是 pairwise）。它的 label 由 `gen_full_splits.py` 直接从 `raw_hf/SongEval/metadata.jsonl` 解析，现在要把 `[Coherence, Musicality, Memorability, Clarity, Naturalness]` 全平均改成只取 `Musicality` 平均，逻辑在 `gen_full_splits.py` 内部调整即可，不需要新脚本。SongEval 的**特征**因为音频窗口从 30 s prefix 变成全曲分块 + pool，仍然需要重抽（由 runner 按 `extract_flags` 处理，不属于打分这一步）。

2. **per-dataset 的 `extract_flags`** 加到 `config/data/full_datasets.yaml` 的每个 dataset entry；`run_full_14_experiments_parallel.py` 的 `_build_jobs` 把 flags 写进 per-job merged config；`run_musiceval_14_experiments.py` 在 `prep_loss_features / prep_entropy_features / prep_sae_features` 三个 step 里透传。

3. **训练部分**：7 家族 CNN（见第 4 节），沿用现有 `seq_len=1500`。

### 3.1 覆盖策略（不改路径）

直接用 runner 的 `--reset-state` 或手动清掉对应 `run_state` 下的 state.json。每一条的"是否真的需要清"详见 §6.3 的矩阵；这里只列出"会被覆盖 / 可能被覆盖"的路径：
- 被覆盖的打分 CSV：`data/manifests/pairwise_relu/{musicpref_musicality,aime_music_quality,musicarena}_1to5.csv`
- 被覆盖的 splits：`data/full_splits/{musicpref,aime,music_arena,songeval,all_5_datasets}/*.csv`
- 被覆盖的特征（只 AIME / MusicArena / SongEval 真正需要重提；MusicPref 窗口未变可沿用；all_5 复用单库特征不独立抽）：`outputs/full/features/{entropy,sae,loss}/{aime,music_arena,songeval}/clean/**`
- 被覆盖的训练产物（5 个 dataset 全部重训）：`outputs/full/{checkpoints,plots,reports,logs}/{musicpref,aime,music_arena,songeval,all_5_datasets}/clean/**`
- 被**清理**（不再生成）的：`data/manifests/pairwise_relu/{musicpref_fidelity,aime_text_audio_alignment}_1to5.csv`
- **不动**：`musiceval/` 子目录下的任何东西、`data/splits/musiceval/*`、`data/full_splits/musiceval/*`、`outputs/full/*/musiceval/**`、`reports/musiceval/**`。

---

## 4. 清 transformer 与 noisy（代码 + config 全面瘦身）

### 4.1 删除 transformer

- 目录删除：`phase7_release/training/curve_transformer/`、`phase7_release/training/hybrid/train_transformer_pool.py`、`train_transformer.py`。
- Config 删除：`config/model/loss_curve_transformer.yaml`；`config/model/hybrid.yaml` 和 `config/training/hybrid.yaml` 里的 transformer 段；`config/training/defaults` 下 `curve_transformer / hybrid_transformer_pool / hybrid_transformer` 三项；`config/model/experiments/`、`config/training/experiments/` 下所有 `*transformer*` 文件。
- Runner 瘦身：
  - `run_musiceval_14_experiments.py`：删除 `transformer_step_names` 集合、`--skip-transformer` 参数、`f0x_*_transformer` 所有 Step、transformer 相关 batch-size / prefetch 参数。
  - `run_full_14_experiments_parallel.py`：删除 `--torch-env` 外只和 transformer 有关的字段（目前没有专门字段，就是通过 runner 传）。
- 文档：`doc/plan/experiment_code_map.md` 改成 7 条（只留 `_cnn`）；`doc/plan/phase7_release_plan.md` 里 "7 × 2 = 14" 相关表述改成 "7 × 1 = 7"；`scripts/eval/eval_14_experiments.py` 与其命名保留，但内部枚举只列 CNN 家族。

### 4.2 删除 noisy

- Config 删除：`config/paths.yaml::data.splits.noisy`；`config/data/full_datasets.yaml` 里所有 `splits.noisy` 段；任何 `*_noisy.yaml` 若存在。
- 数据目录清理：`data/splits/musiceval/*_noisy.csv`、`data/full_splits/*/*_noisy.csv` 删除。
- Runner 瘦身：
  - `scripts/data/preprocess_data.py::_musiceval_copy_splits` 只保留 `clean` 映射。
  - `scripts/data/gen_full_splits.py`：去掉 `.to_csv(..._noisy.csv)` 这行。
  - `scripts/data/merge_all_datasets.py`：`splits` 列表只留 `train/val/test`。
  - `run_full_14_experiments_parallel.sh/.py`：去掉 `SPLITS` / `--splits` 的 noisy 分支（默认 clean），更改 `--splits` 参数为不可选或直接删除。
  - `run_musiceval_14_experiments.py`：去掉 `--splits noisy` 能走的所有分支；`noisy_flag` / `use_noisy_splits` 相关变量删除。
  - `scripts/features/*` / `scripts/loss/*`：`--splits` 参数改成常量 `"clean"` 或完全删除。
- 文档：`phase7_release_plan.md` 的 "noisy 标签版" 指令块删除。

### 4.3 新 7 实验矩阵

保留 CNN 版本的 7 家族：`f01_loss_only_cnn`, `f02_entropy_only_cnn`, `f03_sae_only_cnn`, `f04_loss_entropy_cnn`, `f05_entropy_sae_cnn`, `f06_loss_sae_cnn`, `f07_loss_entropy_sae_cnn`。总工作量 = `5 库 × 7 + 合库 1 × 7 = 42 个 CNN 实验`。MusicEval 不触发。

---

## 5. 不改的东西（显式清单）

- 所有 `training/{loss_curve,entropy_curve,hybrid}/*` 的 CNN 代码与 CLI
- `lib/repro/*`（dataset / nets / metrics / data_paths / scaling）
- `config/paths.yaml` 除 `data.splits.noisy` 外的其它字段
- `config/model/{loss_curve_cnn,hybrid,sae,sae_sparse_autoencoder,sae_verifier,verifier}.yaml`（CNN / SAE 段保留，transformer 段删）
- MusicEval 的打分、splits、特征、checkpoints、reports、logs
- 14 实验 runner 的 run_tag / state / lock 机制
- `scripts/baseline/*`、`scripts/eval/*`、`scripts/analysis/*`（CNN 相关部分）

---

## 6. 执行顺序

1. **阶段 A（代码，等 review 通过）**
   1. 改 `phase7_local4server/training/build_musicpref_scores.py` 去掉 fidelity；跑一次确认。
   2. 新增 `phase7_release/lib/elo_scoring.py`；删除 `phase7_release/lib/pairwise_relu_scores.py`。
   3. 改写 `scripts/data/fit_pairwise_manifests.py`（文件名、CLI、输出路径不变；内核换 Elo；`fidelity / text_audio_alignment` 分支改为报错退出）。
   4. 改写 `scripts/data/gen_full_splits.py`（score 来源按 1.4）。
   5. 改 3 个特征抽取器的 CLI（新增 chunk/pool/max-audio-sec/begin_s-end_s 支持，默认不变）；修 SAE 零填充 bug。
   6. `config/data/full_datasets.yaml` 每个 dataset 加 `extract_flags`；runner 透传。
   7. 清理 transformer / noisy（第 4 节全部删除动作）。

2. **阶段 B（上 server 覆盖跑）** —— 详见 §6.2 的脚本 plan。

---

## 6.1 两个 runner 的分工（必须搞清）

| runner | 角色 | 进程数 | GPU 占用 | 典型场景 |
|---|---|---|---|---|
| `scripts/run/run_musiceval_14_experiments.py` | **单数据集** 执行全部 7 个 step：`prep_loss_features / prep_sae_features / prep_entropy_features / baselines（可选） / f01..f07 CNN / eval_14_experiments` | 1 | 默认 1 张（由外部 `CUDA_VISIBLE_DEVICES` 指定），所有 step 串行跑 | MusicEval 小规模闸门；或被 full parallel runner 作为子进程调起 |
| `scripts/run/run_full_14_experiments_parallel.py` | **多数据集编排器**：读 `config/data/full_datasets.yaml`，按 `execution.include_scopes` 枚举 dataset；`nvidia-smi` 探测可用 GPU，每个 dataset 起一个 `run_musiceval_14_experiments.py` 子进程，`CUDA_VISIBLE_DEVICES=<gpu_id>` 钉在一张卡上；并发数 ≤ `gpu_parallel.max_concurrent_jobs` | N（每 dataset 1 个） | 每子进程 1 张卡，最多并发 8 | full-scale 5 单库 + 合库 |

**server 上必须用 parallel runner**：8×H100 场景下，5 单库顺序跑会让 7 张卡常年空闲；并行能把整个 full-scale 时间压到 ~单库耗时 × ceil(5/并发度)。**小规模 MusicEval 不用并行**（单库、单 GPU、且本次完全不重跑）。

---

## 6.2 阶段 B —— server 上的一键运行 plan

只操作 **clean × CNN × 7** 的 5 个 dataset（`musicpref / aime / songeval / music_arena / all_5_datasets`）；MusicEval 全程不动。

> **一键入口**：`phase7_release/scripts/run/1st_iter.sh`
> 内部按 §6.2.1 → §6.2.4 顺序依次执行；支持 `--skip-{clean,scoring,singles,merged}` 和 `--dry-run`。
> overlay 写到 `phase7_release/outputs/run_state/1st_iter/full_datasets_{single,merged}.yaml`，**不改** `config/data/full_datasets.yaml`。

### 6.2.1 Step-0 清旧产物（本地 / server 皆可；只清不生成）

**只针对受影响的 4 个单库 + all_5**，且只清这一次。脚本执行在 `PROJECT_ROOT`：

```bash
# 1. 旧打分 manifest（只清会被新 Elo 规则覆盖的几份 + 彻底废弃的两份）
rm -f phase7_release/data/manifests/pairwise_relu/{musicpref_musicality,aime_music_quality,musicarena}_1to5.csv
rm -f phase7_release/data/manifests/pairwise_relu/{musicpref_fidelity,aime_text_audio_alignment}_1to5.csv

# 2. 旧 full_splits（MusicEval 保留）
for ds in musicpref aime music_arena songeval all_5_datasets; do
  rm -rf "phase7_release/data/full_splits/${ds}"
done

# 3. 旧特征（仅 AIME/MusicArena/SongEval/all_5 需要清；MusicPref 的 30 s 特征数值不变可复用）
for ds in aime music_arena songeval all_5_datasets; do
  rm -rf "phase7_release/outputs/full/features/loss/${ds}"    \
         "phase7_release/outputs/full/features/entropy/${ds}" \
         "phase7_release/outputs/full/features/sae/${ds}"
done

# 4. 旧训练产物 + run_state（5 个 dataset）
for ds in musicpref aime music_arena songeval all_5_datasets; do
  rm -rf "phase7_release/outputs/full/checkpoints/${ds}" \
         "phase7_release/outputs/full/plots/${ds}"       \
         "phase7_release/outputs/full/reports/${ds}"     \
         "phase7_release/outputs/full/logs/${ds}"
  rm -f  "phase7_release/outputs/run_state/full14_${ds}_clean.state.json" \
         "phase7_release/outputs/run_state/full14_${ds}_clean.lock.json"
done
```

`musiceval/` 子目录、`phase7_release/data/splits/musiceval/`、`outputs/**/musiceval/` 全部 **不动**。

### 6.2.2 Step-1 重打分 + 重切 split（CPU 单机，无 GPU，秒级-分钟级）

这一步 **不进 parallel runner**，是 full runner 启动前的前置步骤：

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

SongEval 不在 `fit_pairwise_manifests` 列表里：`gen_full_splits.py` 直接从 `raw_hf/SongEval/metadata.jsonl` 读取 Musicality 均值作为 `score`。

产出（全部覆盖旧文件）：
- `data/manifests/pairwise_relu/{musicpref_musicality,aime_music_quality,musicarena}_1to5.csv`
- `data/full_splits/{musicpref,aime,music_arena,songeval}/{train,val,test}.csv`
- `data/full_splits/all_5_datasets/{train,val,test}.csv`

### 6.2.3 Step-2 full-scale 并发执行（每个 dataset 一张 GPU）

parallel runner 默认就把 5 个 dataset 全部排上；在 server 8×H100 下会并发最多 8 个任务（本次实际 5 个）。**关键：`all_5_datasets` 因为要复用单库特征，需和 4 个单库分两次调度**（原因见 §6.3）：

```bash
# (a) 先并发 4 个单库：特征抽取 + 7 个 CNN 训练
FULL_CFG=phase7_release/config/data/full_datasets.yaml
#  -> 编辑 full_datasets.yaml，将 execution.include_scopes 临时改成
#     ["large_scale_single"]，或用下面 --full-config 指向一份覆盖版。

conda run -n torch21 python phase7_release/scripts/run/run_full_14_experiments_parallel.py \
  --splits clean

# 等 (a) 完成后再跑 (b)：合库；前提是单库特征都写好了。
#  -> 把 include_scopes 改回 ["large_scale_merged"]，并把
#     execution.skip_feature_extract 改成 true（不重复抽取；all_5 的训练直接
#     从单库的 features 目录读 token_loss_path）。
conda run -n torch21 python phase7_release/scripts/run/run_full_14_experiments_parallel.py \
  --splits clean
```

> 每个 dataset 的子进程还会显示 tqdm 进度条；日志写到 `outputs/run_state/full_parallel_logs/<scope>_<dataset>_clean.log`，汇总写到 `outputs/run_state/full_parallel_summary_clean.csv`。

### 6.2.4 Step-3 汇总与自检

parallel runner 自动为每个 dataset 在子进程最后一步跑 `scripts/eval/eval_14_experiments.py`；它会把 7 个 `f0?_*_cnn_clean_test_scores.csv` 汇总成：

- `outputs/full/reports/<dataset>/clean/tables/aggregate_table_14_experiments_clean.{csv,md}`
- `outputs/full/reports/<dataset>/clean/plots/` 下的散点图

server 侧推荐的自检：

```bash
# 5 个 dataset 都有 summary + 7 条 test_scores
for ds in musicpref aime music_arena songeval all_5_datasets; do
  ls phase7_release/outputs/full/reports/${ds}/clean/tables/aggregate_table_14_experiments_clean.md || echo "MISSING ${ds}"
  ls phase7_release/outputs/full/reports/${ds}/clean/f0?_*_cnn_clean_test_scores.csv | wc -l
done
```

---

## 6.3 重提特征 / 重训练 的确切矩阵

这是 1st_iter_plan 之前 `阶段 B` 第 3 步一句话带过的东西，显式展开。

| dataset | 打分是否变 | 音频窗口是否变 | 需要重提 loss/entropy/sae？ | 需要重训 7 个 CNN？ |
|---|---|---|---|---|
| `musiceval` | 否 | 否 | **否**（整个子目录不动） | **否** |
| `musicpref` | 是（Elo × musicality-only） | 否（仍 30 s 截取） | **实际不需要**，extract_flags 与旧规则一致；但 Step-0 既然没清特征，parallel runner 的 extract step 会以 skip-if-exists 形式快速跳过（最多 manifest 重写） | 是（分数变了） |
| `aime` | 是（系统 Elo + logit winrate） | 是（10 s rater 窗口 via `begin_s/end_s`） | **是** | 是 |
| `music_arena` | 是（系统 Elo + 4-outcome 软目标） | 是（rater ≤180 s → 30 s chunk → pool 1500） | **是** | 是 |
| `songeval` | 是（只取 Musicality 均值） | 是（全曲 → 30 s chunk → pool 1500） | **是** | 是 |
| `all_5_datasets` | 是（由上面 4 个变化传导） | 特征直接 **复用** 各单库的特征目录 | **否**（不重新抽；见 §6.3.1） | 是 |

### 6.3.1 `all_5_datasets` 的特征必须"复用"而非"重抽"

`full_datasets.yaml` 里合库的 `extract_flags: {}` 是空的 —— 如果对合库跑一次 extractor，会用 **默认 30 s / 无 chunk** 的全局规则处理 SongEval / MusicArena 的整曲 / 长片段数据，结果是错的。正确做法：

1. 先完成 §6.2.3 (a)：单库 extract 把 `outputs/full/features/{loss,entropy,sae}/<dataset>/clean/` 写满。
2. 合库训练时 **不再调 extractor**：在 `full_datasets.yaml::execution` 把 `skip_feature_extract` 设 `true` 再跑第二次 parallel runner（§6.2.3 (b)）。
3. `all_5_datasets/{train,val,test}.csv` 里每行的 `token_loss_path` 已经是 `<source>/<file>` 形态（`gen_full_splits.py` 的输出约定），合库 runner 的 `token_loss_root` 需要指向 `outputs/full/features/loss/`（**parent**）而不是 `.../loss/all_5_datasets/clean/`。当前 `run_full_14_experiments_parallel.py::_build_jobs` 对 `all_5_datasets` 也会写子目录，需要一个 **小补丁**：对 `name == "all_5_datasets"` 时把 `token_loss_root / features_entropy / sae.output_dir` 设成单库共用的 parent 目录，并强制 `--skip-feature-extract`。

这步代码补丁没在阶段 A 里包含，建议在阶段 A 7 步完成后、阶段 B 之前单独做；复杂度 < 30 行改动。

---
