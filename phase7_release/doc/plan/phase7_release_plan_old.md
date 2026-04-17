# Phase7 Release — 数据与特征流水线计划（Server 全链路）

## 目标

在**服务器**上对 `phase7_release/` 做一次**端到端试跑**：从五套评测数据库的获取与落盘、路径规范化、划分（train/val/test）、per-token loss 导出，到 SAE / 下游用的 embedding（含与 loss 曲线对齐的 temporal embedding 等）全部可复现。试跑验证通过后：

- **删除**本地下载的音频、缓存、体积大的中间产物（per-token CSV、`.npy`、checkpoint 等可按约定清理）。
- **保留**目录骨架、`.gitkeep`、manifest 的**列名与相对路径写法**、配置模板与脚本入口，使公开 release **不捆绑数据**，只描述「相对 `project_root` 的布局与格式」。

本计划以 `phase7_release/` 为**唯一运行根目录**：`phase7_release/data/README.md` 中的 manifest 契约（`score`, `audio_path`, `token_loss_path`）保留，同时把核心算法脚本复制到 `phase7_release` 内，避免 server 侧依赖 `experiments/phase7/...` 与 `losscal/...`。

---

## 非目标

- 不在本计划中承诺完成全部训练超参搜索；焦点是**数据与特征链路**可跑通、路径可移植。
- 不删除历史实验目录；历史目录仅作为“复制来源”，迁移完成后不再作为运行依赖。

---

## 五套数据库（2025 评测向）

| 名称 | 获取方式 | 备注 |
|------|-----------|------|
| **MusicPref 2025** | Hugging Face: [i-need-sleep/musicprefs](https://huggingface.co/datasets/i-need-sleep/musicprefs) | 音频在 `data/train-*-of-00012.parquet`；人类偏好见根目录 **`human_preference.csv`**（与 `audio.path` 按文件名对齐）。 |
| **AIME 2025** | 音频：[disco-eth/AIME](https://huggingface.co/datasets/disco-eth/AIME)；人类偏好：[disco-eth/AIME-survey](https://huggingface.co/datasets/disco-eth/AIME-survey)（pairwise，`track-*-id` 与音频 `id` 对齐） | 音频 parquet **无标量分**；打分在问卷子集。 |
| **MusicEval 2025** | 官方站: [AISHELL MusicEval / AISHELL_7A](https://www.aishelltech.com/AISHELL_7A) | **通常需注册/协议**；不适合纯脚本无交互下载。计划中单独列为「人工步骤 + 本机/服务器 rsync」。 |
| **SongEval 2025** | Hugging Face: [ASLP-lab/SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval) | 与 MusicPref 类似，统一落到 `datasets/phase7/...` 约定根下。 |
| **MusicArena 2025** | Hugging Face: [music-arena/music-arena-dataset](https://huggingface.co/datasets/music-arena/music-arena-dataset) | 可能为多模态/对话+音频；需定义「用于 loss 评测的一条样本」粒度（单段 wav + 一个标量 score）。 |

**统一原则**：进入流水线前都要有一张 **master 表**（或等价 JSONL），至少包含 `source`、`audio_path`（建议相对 `project_root`）。标签列可多列保留；若下游需要单列 `score`，仅在具体实验配置里聚合，不作为数据准备默认行为。

### Pairwise 打分（exp10 MusicPrefs v2 对齐）

凡 **win / lose / tie** 或等价序关系，统一采用 v2 思路（ReLU + margin + tie 平方）：

- a 胜：`L = ReLU(m + s_b - s_a)`；b 胜对称；
- 平局：`L_tie = (s_a - s_b)^2`；
- `margin m` 作为配置超参（如 0.2）。

**MusicPref**：`musicality` 与 `fidelity` 分两路，不混。  
**AIME**：按 `question-type` 分两路（Music Quality / Text-Audio Alignment）。  
**MusicArena**：`A/B/TIE/BOTH_BAD` 独立规则版本。

---

## 目录与路径约定（Release 可移植）

```text
phase7_release/
  raw_hf/
    MusicPref/
    AIME/
    AIME-survey/
    SongEval/
    MusicArena/
  datasets/
    musicprefs/
    aime/
    musiceval/
    songeval/
    music_arena/
  data/
    manifests/
    README.md
  outputs/
    per_token_losses/
    checkpoints/
    logs/
    reports/
```

Manifest 关键列：

- `score`
- `audio_path`
- `token_loss_path`

---

## 流水线阶段（Server 执行顺序）

### 0) 核心算法复制到 `phase7_release`（先做）

把 server 运行所需核心算法放进 `phase7_release`，避免强依赖 `experiments/` 与 `losscal/`：

1. per-token loss 抽取（来自 `losscal/loss_cal_small.py`）
2. hybrid index 构建（`create_hybrid_index` 对齐逻辑）
3. split 构建（替换委托 exp11 的 prepare）
4. SAE 特征与 temporal embedding 脚本迁入 release

---

### 1) 下载与标准化（Ingest）

- HF 四套：统一脚本入口下载，支持 smoke（`--max_samples`）。
- MusicEval：人工下载 + 目录结构校验。
- 每库先产出一份 manifest 草稿。

### 2) 路径规范化

- 全部 `audio_path` 转为相对 `project_root` 的 POSIX 路径。
- 统一采样率/扩展名（按模型要求）。
- 记录丢弃原因（缺文件、坏音频、空标签）。

### 3) Master -> Hybrid index

- 生成 `master_index.csv`（含 `source/path/score`）。
- 导出 per-token loss 后绑定 `token_loss_path`，得到 `hybrid_index.csv`。

### 4) Split

- source-aware 切分 train/val/test。
- 输出 `phase7_release/data/manifests/{train,val,test}.csv`。

### 5) Per-token loss extraction

- 批量导出 token loss csv 到 `outputs/per_token_losses/`。

### 6) Embedding extraction

- SAE 特征：`sae_features_{split}.npy` 与 `*_meta.pt`
- temporal embedding（可选第二阶段）

### 7) 训练/评估入口验证

- 先用轻量 baseline 验证链路（例如 mean-loss）。

---

## 在 `torch21` 环境中执行（复制即用）

```bash
conda activate torch21
export PHASE7_PROJECT_ROOT="/home/evev/noiseloss"
cd "$PHASE7_PROJECT_ROOT"
export PYTHONPATH="$PHASE7_PROJECT_ROOT"
export HF_HOME="${PHASE7_PROJECT_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_DATASETS_CACHE"
```

### HF 认证

```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
huggingface-cli whoami
```

### 0.6 下载 HF 数据到 `phase7_release/raw_hf`

```bash
mkdir -p "$PHASE7_PROJECT_ROOT/phase7_release/raw_hf"

huggingface-cli download "i-need-sleep/musicprefs" --repo-type dataset \
  --local-dir "$PHASE7_PROJECT_ROOT/phase7_release/raw_hf/MusicPref"
huggingface-cli download "disco-eth/AIME" --repo-type dataset \
  --local-dir "$PHASE7_PROJECT_ROOT/phase7_release/raw_hf/AIME"
huggingface-cli download "disco-eth/AIME-survey" --repo-type dataset \
  --local-dir "$PHASE7_PROJECT_ROOT/phase7_release/raw_hf/AIME-survey"
huggingface-cli download "ASLP-lab/SongEval" --repo-type dataset \
  --local-dir "$PHASE7_PROJECT_ROOT/phase7_release/raw_hf/SongEval"
huggingface-cli download "music-arena/music-arena-dataset" --repo-type dataset \
  --local-dir "$PHASE7_PROJECT_ROOT/phase7_release/raw_hf/MusicArena" \
  --include "battle_data/**"
```

### 0.7 AIME 问卷关联（survey + audio）

```bash
python "$PHASE7_PROJECT_ROOT/phase7_release/scripts/data/aime_join_survey.py" \
  --survey-parquet "$PHASE7_PROJECT_ROOT/phase7_release/raw_hf/AIME-survey/data/train-00000-of-00001.parquet" \
  --out-dir "$PHASE7_PROJECT_ROOT/phase7_release/data/manifests"
```

产物：

- `aime_pairwise_survey.csv`
- `aime_track_winrate.csv`
- `aime_track_winrate_by_question_type.csv`

### 0.8 MusicPref 对齐说明

- `human_preference.csv` 使用 `audio_a/audio_b` 与 parquet 的 `audio.path` 对齐。
- `musicality` 与 `fidelity` 两列独立建模，不合并。

### MusicEval 2025（按旧版流程）

[OpenData 入口](https://opendata.aishelltech.com/aishell-7a) 在 AISHELL-7A 下提供 Netdisk（`mab.to` -> myairbridge，浏览器交互）与 Google Drive 镜像。命令行推荐使用 Drive 文件夹：

```bash
mkdir -p "$PHASE7_PROJECT_ROOT/phase7_release/datasets/musiceval/audio"
mkdir -p "$PHASE7_PROJECT_ROOT/phase7_release/datasets/musiceval/metadata"

pip install gdown
cd "$PHASE7_PROJECT_ROOT/phase7_release/datasets/musiceval"
gdown --folder "https://drive.google.com/drive/folders/1vQJrmhu7sn4r-KALndm4MQQjhaaItLWN" -O . --remaining-ok
# 得到 MusicEval-full.zip，解压后目录为 MusicEval-full/

unzip -q -o MusicEval-full.zip
# 注意：wav/ 常为只读目录位，使用 cp 不要 mv
cp -a MusicEval-full/wav/*.wav audio/
cp -a MusicEval-full/metadata MusicEval-full/person_mos MusicEval-full/sets MusicEval-full/system_mos metadata/
cp -a MusicEval-full/prompt_info.txt MusicEval-full/demo_prompt_info.txt MusicEval-full/README.md metadata/
```

生成路径+评分表（MOS1/MOS2 分开保留）：

```bash
python - <<'PY'
import os, csv
root = os.environ["PHASE7_PROJECT_ROOT"]
mos_path = os.path.join(root, "phase7_release/datasets/musiceval/metadata/sets/total_mos_list.txt")
out = os.path.join(root, "phase7_release/data/manifests/musiceval_paths_scores.csv")
rows = []
with open(mos_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        fn, a, b = line.split(",")
        rows.append({
            "source": "MusicEval2025",
            "path": "phase7_release/datasets/musiceval/audio/" + fn,
            "score_musical_impression": float(a),
            "score_text_alignment": float(b),
        })
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Wrote", out, len(rows))
PY
```

---

## 清理策略（保留布局）

删除重资产：

- `datasets/**` 音频
- `outputs/**` npy/pt/checkpoint/per-token csv
- HF 缓存

保留：

- `config/**`
- `scripts/**`
- `training/**`
- manifest 表头与格式契约

---

## 交付物

1. `phase7_release/doc/plan/phase7_release_plan.md`（本文档）
2. `phase7_release/scripts/data/` 下载+预处理+索引脚本
3. `phase7_release/config/paths.yaml` 与分层配置
4. `phase7_release/data/README.md` 数据契约说明
5. server 端 smoke test 记录
