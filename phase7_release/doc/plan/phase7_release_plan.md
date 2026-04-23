# Phase7 Release — 系统化实施方案

> 约定：`phase7_release` 内所有脚本 / Python 文件会**自动把当前仓库根作为项目根**（脚本由 `$(dirname $0)/../../..` 推导，Python 由 `__file__` 向上寻找 `phase7_release/` 的父目录）。
> 需要覆盖时 `export PROJECT_ROOT=<your_path>`；文档示例里的 `cd "${PROJECT_ROOT}"` 会立即采用它。

## 1) 目标与范围

本方案用于 `phase7_release` 的收官搭建，目标是形成一套可长期复用的训练与分析系统，而不是一次性脚本集合。

- 小规模实验：全部在 `MusicEval` 完成（作为系统闸门）。
- 大规模实验：  
  1) 5 个数据库分别独立实验；  
  2) 5 个数据库合并实验。

系统要求：

- 数据、特征、训练、分析模块解耦；
- 同一实验矩阵可复用于小规模/单库/合库；
- pairwise 数据统一提供 BT 与 MSE 两套 1-5 分，便于对比。

---

## 2) 实验矩阵（固定）

每个数据范围都跑 7 个特征组合，骨干固定为 `CNN`，数据 split 固定为 `clean`：

1. Loss Curve
2. Entropy Curve
3. SAE
4. Loss Curve + Entropy
5. Entropy + SAE
6. Loss Curve + SAE
7. Loss Curve + Entropy + SAE

即：

- 每个数据范围 = `7` 个实验；
- 小规模（MusicEval）= 7；
- 大规模单库（5 库）= `5 x 7 = 35`；
- 大规模合库 = 7；
- 全阶段总计 = 49 个主实验。

> Transformer 骨干与 `noisy` split 已从 release 中完全移除；仅保留 `LossCurveCNN` 一个骨干、`clean` 一个 split variant。

---

## 3) 数据系统设计

## 3.1 数据源

- MusicPref 2025
- AIME 2025 + AIME-survey
- MusicEval 2025
- SongEval 2025
- MusicArena 2025

> 现状：数据下载已具备。系统中仍保留下载入口与校验步骤，保证从 0 可重跑。

## 3.2 数据分层

建议固定三层：

1. `raw_hf/`：原始快照（可选全量保留）
2. `datasets/`：统一音频落地与标准化后数据
3. `data/manifests/`：训练与分析唯一入口（CSV）

训练与分析脚本只读 manifests，不直接读 raw 源结构。

## 3.3 标签统一与重标定

对仅有 `win/lose/tie/both_bad` 的库（MusicPref / AIME / MusicArena），统一使用
`phase7_release/lib/elo_scoring.py` 中的 Elo 拟合，再按各库规则映射到 1-5 分
（单列 `score`）。具体规则：

- **MusicPref**：`musicality`-only Elo。不使用 fidelity、对齐等其它轴。
- **AIME**：12 对 Music Quality 的系统 Elo 叠加 `logit(Laplace-smoothed winrate)`
  残差，并记录每条样本的 10 秒 survey 窗口（`begin_s`/`end_s`）。
- **MusicArena**：系统 Elo + 上下文软目标（`alpha=0.5`，`system_span=50`），
  四种 outcome 一起考虑（`BOTH_BAD` 严格低于 `TIE`），按 rater 对齐的音频片段
  （最多 180 秒）训练。

对原生标量分数据：

- **MusicEval**：打分与特征均保持不动。
- **SongEval**：仅使用 4 位打分者的 `Musicality` 平均值，按全曲分块 + 均匀池化到
  1500 帧的 feature 形状。

## 3.4 Manifest 规范（核心契约）

主清单至少包含：

- `source`
- `audio_path`
- `token_loss_path`
- `score`（单列 1-5）

按需扩展：

- `begin_s`/`end_s`（AIME / MusicArena 的 rater 窗口）
- `listen_sec_used`
- `entropy_curve_path`
- `sae_feature_path`
- `length`
- `split`

路径统一为相对 `project_root`。

---

## 4) 特征系统设计

## 4.1 Loss / Entropy

- Loss curve：来自 per-token loss 序列；
- Entropy curve：来自对应 token 分布熵序列；
- 两者在样本维度严格对齐（同一 `audio_path`、同长度或可对齐长度）。

## 4.2 SAE（统一标准）

SAE 提取统一使用 `musicdiscovery` 的 MusicGen-small 方案：

- 仓库：<https://github.com/PapayaResearch/musicdiscovery>
- 放置：`external/musicdiscovery`
- 管理：git submodule + 固定 revision
- 后续语义解释（concept labeling / clustering）也使用同一体系，避免方法漂移。

## 4.3 三类特征的组合策略

7 个组合都从同一 manifest 索引样本，组合只改变输入通道，不改变样本集合定义，确保可比性。

---

## 5) 训练系统设计

## 5.1 配置分层

`phase7_release/config` 结构按职责拆分：

- `paths.yaml`：运行路径、输出目录、SAE 参数
- `data/`：split 策略、标签列选择、缩放设置
- `data/full_datasets.yaml`：full-scale 数据集 split 映射 + GPU 并发配置（仅大规模）
- `model/`：网络结构（按 7 组合 x 2 backbone）
- `training/`：优化参数（lr、batch、epoch 等）
- `analysis/`：分析参数（窗口、片段长度、聚类设置等）

## 5.2 运行标识体系

每次实验建议统一三元标识：

- `family`（f01~f07）
- `backbone`（固定 `cnn`）
- `dataset_scope`（musiceval / single_db / merged_5db）

用于日志目录、checkpoint 命名、汇总表主键。

---

## 6) 分析系统设计

## 6.1 进一步打分实验（曲线化评分）

基于已训练打分模型，新增片段级评分实验：

- 模型：RNN；
- 输入：测试集切片段（滑窗）；
- 输出：每首曲子的 segment score curve；
- 对齐：与同曲 loss/entropy 曲线做时序关联。

实现脚本（phase7_release 内）：

- `analysis/segment/train_rnn.py`：RNN 片段级回归训练（主线）
- `analysis/segment/predict_curves.py`：导出逐曲 segment score curve（CSV）
- `scripts/run/run_segment_rnn_analysis.sh`：RNN 训练+测试曲线导出一键入口

## 6.2 可解释性分析

两条主线：

1. loss-entropy 关系模式（例如 HL/LH）是否与片段评分变化一致；
2. SAE feature 聚类后是否能得到稳定的 synthetic 语义解释，并和评分段落对应。

输出应包含：

- 模式统计表（按数据源/模型分层）；
- 代表片段可视化；
- SAE 聚类语义摘要。

---

## 7) 执行阶段（推荐顺序）

### Phase A — 基础准备

1. 固化配置分层与实验 ID 规则；
2. 数据 manifests 校验通过；
3. SAE 外部依赖（submodule）固定版本。

### Phase B — 小规模闸门（MusicEval）

1. 跑完整 7 个 CNN 实验；
2. 验证训练、评估、日志、汇总链路；
3. 跑片段评分与基础可解释性分析。

通过条件：7 个实验可复现、指标可汇总、分析脚本可跑通。

### Phase C — 大规模单库（5 库）

每库跑 7 个 CNN 实验，形成库内最佳组合与库间差异比较。  
执行方式：在 server 上先做 GPU 探测，然后使用并发调度（8xH100 对应最多 8 并发任务，受可用卡与阈值控制）。

### Phase D — 大规模合库（5 库融合）

在 source-aware split 下跑 7 个 CNN 实验，评估跨库泛化与稳健性。  
执行方式：复用同一 GPU 并发调度器，和单库任务共享卡池。

### Phase E — 汇总发布

统一输出主结果表、分析图、可解释性报告与复现实验说明。

---

## 8) 交付物

1. 本计划文档（系统设计版）
2. 完整实验矩阵配置（7 × cnn × clean）
3. 数据预处理与 Elo / 原生标量分打分流水线
4. `external/musicdiscovery` 接入与版本锁定
5. 小规模 + 大规模（单库/合库）结果与分析报告
6. `doc/plan/experiment_code_map.md`（7 实验与脚本的一一映射）

---

## 9) 质量闸门（必须满足）

- 同一实验在不同机器可重复启动（路径契约一致）；
- 单列 `score`（Elo-1to5 或原生）可训练、可评估、可汇总；
- 片段评分曲线可生成且可对齐 loss/entropy；
- SAE 聚类解释能回溯到样本与片段；
- 所有结果可由 `family × backbone × dataset_scope` 三元唯一定位。

---

## 10) Server 执行脚本（给他人直接跑）

为保证“只执行脚本就能跑”，在 `phase7_release/scripts/run/` 提供三条入口：

1. `setup_sae_musicdiscovery.sh`  
   - 安装/初始化 SAE 依赖（`external/musicdiscovery`）；
   - 若 `torch21` 为 Python<3.10，则自动创建并使用独立环境 `musicdiscovery310` 安装依赖；
   - 若 `torch21` 已是 Python>=3.10，则直接在 `torch21` 安装；
   - 这是所有 SAE 与 exp13 训练前置步骤。
2. `run_musiceval_14_experiments.sh`
   - 串行执行 MusicEval 的 7 个主实验（7 组合 × CNN）；
   - 每个实验步骤显示 `tqdm` 时间进度条（按秒更新，含累计耗时）；
   - 自动执行统一评估，输出每个实验的散点图与 Pearson/Spearman 汇总表；
   - 注意：脚本 / 目录 / 汇总表文件名沿用 `14_experiments`，不做重命名，只是实际步骤减到 7 个。
3. `run_segment_rnn_analysis.sh`
   - 基于 train/val 训练 segment-level RNN；
   - 对 test 输出曲线化打分结果（每段预测分）；
   - 支持 loss-only 或 loss+entropy 输入。
4. `run_full_14_experiments_parallel.sh`
   - 用于 full-scale（5 单库 + 合库）任务；
   - 先探测可用 GPU，再按数据集并发提交 `run_musiceval_14_experiments.py` 子任务；
   - 并发参数（GPU 白名单、阈值、最大并发）在 `config/data/full_datasets.yaml` 配置；
   - `MusicEval` 小规模默认不走并发调度。

### 10.1 先装 SAE（必须）

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh
```

可选环境变量：

```bash
PROJECT_ROOT="<your_path>" TORCH_ENV=torch21 \
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh
```

### 10.2 MusicEval 统一运行（7 个 CNN 实验）

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
```

仅支持 `--splits clean`（默认）；noisy 分支已移除。

### 10.3 Segment-level RNN 分析链路

loss-only（默认）：

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh
```

loss+entropy：

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh --with-entropy
```

### 10.4 Full-scale 并发运行（8xH100）

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh
```

`--splits` 参数只接受 `clean`（默认）。

关键点：

- 并发前自动执行 `nvidia-smi` 探测；
- 只调度满足阈值（空闲显存/利用率）的 GPU；
- 每个 dataset 一个独立子进程，使用独立生成的 config 与输出目录，避免互相覆盖；
- 失败任务写入 summary，支持按 dataset 级别重跑。

---

## 11) 脚本内聚化重构（全部放在 phase7_release）

目标：`run_musiceval_14_experiments.sh` 统一执行 7 个 CNN 实验并自动评估。  
原则：保留 exp12/exp13 训练细节与默认超参，仅把核心逻辑迁移到 `phase7_release`；文件名沿用 `14_experiments`，不做重命名。

### 11.1 目录结构（执行版）

```text
phase7_release/
  lib/
    repro/
      data_paths.py        # split 与路径解析（clean）
      loss_dataset.py      # Loss curve 2/3 通道 dataset
      nets.py              # LossCurveCNN（唯一骨干）
      metrics.py           # Pearson/Spearman + scatter
      scaling.py           # affine / quantile rescale
      elo_scoring.py       # Elo 拟合工具（MusicPref / AIME / MusicArena）
      audio_window.py      # rater 对齐裁剪 + 分块 + 均匀池化的共享工具


  scripts/
    baseline/
      mean_loss.py         # exp12 step2（原 compute_mean_loss）
      aesthetics.py        # exp12 step1（可选）
    eval/
      eval_rescaled.py     # exp12 step4（原 run_rescale_and_correlations）
    features/
      extract_sae_features.py
      extract_sae_features_musicdiscovery.py
      extract_entropy_curves.py
    run/
      setup_sae_musicdiscovery.sh
      run_musiceval_14_experiments.sh
      run_segment_rnn_analysis.sh

  training/
    loss_curve/
      train.py             # exp12 step3 训练
      predict.py           # exp12 step3 test 推理
    entropy_curve/
      train.py             # entropy-only cnn 训练
      predict.py           # entropy-only cnn 推理
    hybrid/
      dataset.py           # exp13 HybridPrecomputedDataset
      train_cnn.py         # exp13 hybrid cnn（f03/f05/f06/f07）
  analysis/
    segment/
      train_rnn.py         # segment-level rnn 训练
      predict_curves.py    # segment score curve 导出
```

### 11.2 运行脚本阶段映射（小规模）

`run_musiceval_14_experiments.sh` 固定按下列顺序调用：

1. 特征提取（SAE + entropy，可按参数跳过）
2. 7 个主实验（7 组合 × CNN）
3. `scripts/eval/eval_14_experiments.py`（统一散点图 + Pearson/Spearman 汇总，文件名沿用 `14_experiments`）

`run_segment_rnn_analysis.sh` 固定按下列顺序调用：

1. `analysis/segment/train_rnn.py`
2. `analysis/segment/predict_curves.py`

`run_musiceval_14_experiments.sh` 固定在 7 个实验完成后调用：

1. `scripts/eval/eval_14_experiments.py`
   - 逐实验输出相关性散点图；
   - 汇总 Pearson / Spearman 到总表（CSV + MD）。

### 11.3 兼容与验收

- `phase7_release/scripts/run/run_musiceval_14_experiments.sh` 中不得出现 `experiments/phase7/.../*.py` 调用。
- 同一命令行参数语义保持与 exp12/exp13 一致（如 `--splits`, `--run-name`）；`--use_noisy_splits` / `--skip-transformer` 已移除。
- 迁移后以 MusicEval 小规模链路做一次端到端 smoke 验收。
- 训练关键参数（epoch、patience、lr、weight_decay、lr_decay_factor、lr_decay_patience、min_lr）统一记录在 `config/paths.yaml` 的 `training.defaults`（仅 CNN 两类：`loss_curve` / `hybrid_cnn`）。
