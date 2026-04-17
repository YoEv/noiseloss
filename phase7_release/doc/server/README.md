 # Phase7 Release — Server 执行指南

本文档面向**服务器端**的完整复现与执行流程。本地（单机）测试已经完成，下述步骤假定是在一台干净的 GPU 服务器（推荐 8×H100）上**从 0 开始**（含数据下载 + 预处理 + 训练 + 评估 + 分析）。

设计与 `phase7_release/doc/plan/phase7_release_plan.md` 完全对应，按 Phase A → B → C → D → E 推进。

## 0. 实验矩阵一览（对齐 plan §2、§7）

| 范围 | 数据集 | 入口脚本 | 单次实验数 |
|---|---|---|---|
| 小规模闸门 | MusicEval | `run_musiceval_14_experiments.sh` | `7 × 2 = 14` |
| 大规模单库 | musicpref / aime / songeval / music_arena（`full_datasets.yaml` 中 `enabled: true`） | `run_full_14_experiments_parallel.sh` | `4 × 14 = 56` |
| 大规模合库 | `all_5_datasets` | `run_full_14_experiments_parallel.sh` | `1 × 14 = 14` |
| Segment 分析 | 复用 MusicEval 训好的权重 | `run_segment_rnn_analysis.sh` | 1 RNN + 曲线导出 |

> `full_datasets.yaml` 默认把 `musiceval` 设为 `enabled: false`，由小规模 runner 处理，避免与大规模重复；如需让 full-scale 也跑 MusicEval，把该 `enabled` 改为 `true` 即可（plan §2 的 `5 × 14` 口径）。
>
> 每一次执行要按 `--splits clean` 与 `--splits noisy` 各跑一遍，得到两套 label 版本。

所有命令默认的项目根目录是 `/home/evev/noiseloss`。如果服务器路径不同，请在每条命令前显式 `export PROJECT_ROOT=<your_path>`。

---

## 1. 服务器硬件与系统要求

- **GPU**：8×H100（80 GB）；full-scale 并发调度需要单卡 ≥60 GB 空闲显存。小规模 MusicEval 只需要 1 张可用卡。
- **CPU / 内存**：≥32 核 / ≥256 GB（hybrid 训练 `num_workers=8, prefetch_factor=8`）。
- **磁盘**：≥2 TB 可用（音频 + loss / entropy / SAE 特征）。
- **操作系统**：Linux（与 `torch==2.1.0 + CUDA` 兼容的发行版均可；已在 Ubuntu 22.04 验证）。
- **CUDA 驱动**：支持 `torch 2.1.0 + cu118/cu121` 的驱动（nvidia-smi 可见即可）。

系统级依赖（一次性安装）：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ca-certificates build-essential ffmpeg libsndfile1
git lfs install
```

Conda 安装（若未装）：

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
```

---

## 2. 拉取代码

```bash
cd /home
sudo mkdir -p evev && sudo chown "$USER":"$USER" evev
cd evev
git clone <your_repo_url> noiseloss
cd noiseloss

git submodule update --init --recursive
```

代码库根目录结构如下（只列关键部分）：

```text
noiseloss/
  datasets/                 # 原始/中间音频（大文件，不进 git，见 §5）
  external/                 # 第三方子模块：musicdiscovery / audiobox-aesthetics / audiocraft ...
  experiments/phase7/loss_eval_experiments/  # 小规模 loss/entropy CSV 来源（见 §5）
  phase7_release/           # 发行层：本次服务器执行的唯一入口
    config/paths.yaml
    config/data/full_datasets.yaml
    data/splits/musiceval/  # MusicEval 小规模 split CSV（随仓库）
    data/full_splits/       # 大规模 5 库 split CSV（由 §5 生成或复制）
    scripts/run/            # 全部服务器入口脚本
    doc/                    # 本文档所在位置
    outputs/                # 运行期自动生成（checkpoints / reports / features ...）
```

---

## 3. Conda 环境

总共需要 3 个 conda 环境，职责严格分工：

| 环境名               | Python   | 用途                                                                 |
| -------------------- | -------- | -------------------------------------------------------------------- |
| `torch21`            | 3.8–3.12 | Loss/Entropy 特征提取、全部训练与评估、baseline mean_loss            |
| `musicdiscovery310`  | 3.10     | 仅用于 SAE 特征提取（musicdiscovery + MusicGen-small 管线）          |
| `audiobox`           | 3.10     | 仅用于 audiobox-aesthetics baseline 推理                             |

> `torch21` 若本身就是 Python ≥3.10，理论上可与 `musicdiscovery310` 合并，但 `setup_sae_musicdiscovery.sh` 会自动探测并处理，**不需要手动合并**。

### 3.1 `torch21`（主训练环境）

```bash
conda create -y -n torch21 python=3.10
conda activate torch21

pip install --upgrade pip
pip install \
  torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install \
  numpy==1.26.4 pandas==2.2.* scipy==1.13.* scikit-learn==1.4.* \
  pyyaml==6.* tqdm matplotlib seaborn \
  soundfile==0.12.* librosa==0.10.* \
  transformers==4.41.* accelerate==0.30.* safetensors \
  tensorboard einops
```

验证：

```bash
conda run -n torch21 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### 3.2 `musicdiscovery310`（SAE 特征提取）

**不要手动装**。`setup_sae_musicdiscovery.sh`（§4.1）会自动：

1. 初始化 `external/musicdiscovery` submodule；
2. 如果 `torch21` 是 Python<3.10，则创建 `musicdiscovery310` 并装上 patch 过的依赖（解决 audiocraft 1.3.0 与 musicdiscovery 上游 pin 的冲突）；
3. 从 S3 下载配置中指定的 SAE checkpoint（`external/musicdiscovery_checkpoints/...`）。

### 3.3 `audiobox`（aesthetics baseline）

仅当需要跑 `baseline_aesthetics` 步骤时才需要；默认开启。

```bash
conda create -y -n audiobox python=3.10
conda activate audiobox
pip install --upgrade pip
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e external/audiobox-aesthetics
pip install soundfile librosa tqdm pandas pyyaml
```

如果不想跑 aesthetics，在 §6 的执行命令上追加 `--no-with-aesthetics` 即可完全跳过这个环境。

---

## 4. 外部依赖与 Checkpoint

### 4.1 SAE（必须，一次性）

#### 4.1.0 预备：解决 PyAV (`av==11.0.0`) 源码编译失败

`musicdiscovery` 的依赖中包含 `audiocraft==1.3.0`，后者 pin 了 `av==11.0.0`。当 pip 在目标机器上找不到匹配的 PyAV wheel（比如服务器 glibc / Python / pip 组合较新或较旧）时，会回退到源码编译，这时就会出现：

```text
Package 'libavformat', required by 'virtual:world', not found
Package 'libavcodec', required by 'virtual:world', not found
...
ERROR: Failed to build 'av' when getting requirements to build wheel
```

原因：机器没有 FFmpeg 的 dev 头文件 / `pkg-config`。在跑 §4.1.1 之前，**二选一**先执行下面的命令，把这个问题消掉：

**Option A（推荐）：通过 conda-forge 装 FFmpeg dev 头 + pkg-config**

> 原因：把 `libav*` 头文件和 `pkg-config` 直接装进 `torch21` env，PyAV 走正常源码编译也能成功，对后续调用 ffmpeg 的脚本（aesthetics、切片、重采样等）同样有用。

```bash
conda install -n torch21 -c conda-forge -y 'ffmpeg=6.*' pkg-config
```

**Option B：强制使用 PyAV 预编译 wheel，跳过源码编译**

> 原因：不动系统也不装 FFmpeg dev，只要 PyPI 上有对应 Python/平台的 `av==11.0.0` wheel 就能直接用。缺点：如果后续其他包也要 `libav*`，还得补装 ffmpeg。

```bash
conda run -n torch21 python -m pip install --only-binary=:all: av==11.0.0
```

执行完 Option A 或 Option B 之后，再跑 §4.1.1 的主安装脚本，即可绕过 `av` 编译阶段。

#### 4.1.1 安装 musicdiscovery + 下载 SAE checkpoint

```bash
cd /home/evev/noiseloss
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh
```

该脚本会：

- 初始化 / 添加 `external/musicdiscovery` submodule；
- 按 `phase7_release/config/paths.yaml` 里 `sae.musicdiscovery_checkpoint_dir` 的路径，从 `https://music-discovery-sae-checkpoints.s3.amazonaws.com` 下载 `cfg.json`、`sae_weights.safetensors`、`sparsity.safetensors`；
- 安装 musicdiscovery 的 Python 依赖（必要时自动创建 `musicdiscovery310`）。

成功后应看到：

```text
external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/
  cfg.json
  sae_weights.safetensors
  sparsity.safetensors
```

> 注：远端 S3 公开 bucket 上 `layer_12` 可能不存在，脚本会自动回退到 `known_layers=[1,5,11,17,21]` 中离 11（即 `layer_12 − 1`）最近的一层，并下载对应 checkpoint。运行结束时会打印 `selected checkpoint prefix`，请核对。

### 4.2 MusicGen-small（loss/entropy 特征提取）

`transformers` 首次调用 `MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")` 会自动下载约 1.5 GB 到 `~/.cache/huggingface`。如果服务器离线，需手动在能联网的机器上拉好后 rsync 过去。

### 4.3 audiobox-aesthetics checkpoint（可选）

`baseline_aesthetics` 第一次调用 `AesPredictor()` 会自动下载权重；若服务器离线，参考 `external/audiobox-aesthetics/README.md` 手动放置。

---

## 5. 数据准备（从 0 开始）

本节对齐 `phase7_release/doc/plan/phase7_release_plan.md` §3（数据系统设计）的三层结构：

```text
phase7_release/
  raw_hf/<Dataset>/          # 层 1：HF 原始快照（只读，不进 git）
  datasets/<dataset>/audio/  # 层 2：统一落地的音频（wav / mp3 → wav）
  data/manifests/            # 层 3a：带 score_bt_1to5 / score_mse_1to5 的主清单
  data/splits/musiceval/     # 层 3b-a：MusicEval 小规模 split（随仓库）
  data/full_splits/<dataset>/# 层 3b-b：大规模 split（由本节生成）
```

训练与分析脚本**只读 `data/splits/` 与 `data/full_splits/` 下的 CSV**，不直接读 `raw_hf/`。

所有步骤都在 `torch21` 环境中跑（需要额外 `datasets`、`huggingface_hub`）：

```bash
conda activate torch21
pip install datasets==2.21.0 huggingface_hub
```

### 5.1 数据源与 HF 仓库

| 库 | HF repo | 类型 | 用途 |
|---|---|---|---|
| MusicEval 2025 | `BAAI/MusicEval` | 原生 MOS | 小规模闸门 |
| MusicPref 2025 | `i-need-sleep/musicprefs` | pairwise | 大规模单库 |
| AIME 2025 | `disco-eth/AIME` | 音频 | 大规模单库（配 survey） |
| AIME-survey 2025 | `disco-eth/AIME-survey` | pairwise | AIME 的 label |
| SongEval 2025 | `ASLP-lab/SongEval` | 5 维 Likert | 大规模单库（原生分） |
| MusicArena 2025 | `gneubig/music-arena-public` | pairwise（battle JSON） | 大规模单库 |

> 凡是**离线服务器**，都要先在跳板机上 `huggingface-cli download <repo> --repo-type dataset --local-dir phase7_release/raw_hf/<Dataset>` 下载好后 rsync 到目标机；在线服务器则可直接在服务器上 `huggingface-cli login` + `download`。

### 5.2 HF 原始快照（层 1）

```bash
cd /home/evev/noiseloss

# 如需访问 gated 仓库
huggingface-cli login

# MusicPref
huggingface-cli download i-need-sleep/musicprefs \
  --repo-type dataset --local-dir phase7_release/raw_hf/MusicPref

# AIME 音频（~60 GB，parquet 编码）
huggingface-cli download disco-eth/AIME \
  --repo-type dataset --local-dir phase7_release/raw_hf/AIME

# AIME 配套 survey（pairwise 标签）
huggingface-cli download disco-eth/AIME-survey \
  --repo-type dataset --local-dir phase7_release/raw_hf/AIME-survey

# SongEval（含 mp3/ 与 metadata.jsonl）
huggingface-cli download ASLP-lab/SongEval \
  --repo-type dataset --local-dir phase7_release/raw_hf/SongEval

# MusicArena（battle_data/*.json + audio_files/*.wav）
huggingface-cli download gneubig/music-arena-public \
  --repo-type dataset --local-dir phase7_release/raw_hf/MusicArena
```

确认后应该有：

```text
phase7_release/raw_hf/
  AIME/data/train-*.parquet
  AIME-survey/data/train-*.parquet
  MusicPref/data/train-*.parquet
  MusicPref/human_preference.csv
  SongEval/mp3/*.mp3 + metadata.jsonl
  MusicArena/audio_files/... + battle_data/**/*.json
```

### 5.3 把音频落地到 `datasets/<name>/audio/`（层 2）

对每个库，把 parquet / mp3 / zip 里的音频统一导出成 wav（或保留 wav），以 **item id 作为文件名**，落到 `phase7_release/datasets/<name>/audio/`。这步有两条通用路径：

#### 5.3.1 可直接用 HF datasets 迭代的仓库

```bash
# MusicPref（parquet 内嵌 audio）
conda run -n torch21 python phase7_release/scripts/data/hf_ingest_smoke.py \
  --repo i-need-sleep/musicprefs \
  --source-tag MusicPref2025 \
  --out-audio-dir phase7_release/datasets/musicprefs/audio \
  --master-csv phase7_release/data/manifests/master_index.csv \
  --max-samples 0   # 0 = 导全量；调试阶段设 8 跑通链路

# AIME（parquet 内嵌 audio）
conda run -n torch21 python phase7_release/scripts/data/hf_ingest_smoke.py \
  --repo disco-eth/AIME \
  --source-tag AIME2025 \
  --out-audio-dir phase7_release/datasets/aime/audio \
  --master-csv phase7_release/data/manifests/master_index.csv \
  --max-samples 0

# SongEval（本地 mp3 可直接 ffmpeg 批转 wav）
find phase7_release/raw_hf/SongEval/mp3 -name '*.mp3' -print0 | \
  xargs -0 -n1 -P8 -I{} bash -c '
    out="phase7_release/datasets/songeval/audio/$(basename "{}" .mp3).wav"
    mkdir -p "$(dirname "$out")"
    [[ -f "$out" ]] || ffmpeg -y -loglevel error -i "{}" -ac 1 -ar 32000 "$out"
  '
```

#### 5.3.2 MusicArena（battle JSON 中 `audio_a / audio_b` 相对路径）

```bash
# 直接把原 wav 同步到 datasets/ 下，保持相对结构
rsync -av phase7_release/raw_hf/MusicArena/audio_files/ \
          phase7_release/datasets/music_arena/audio/
```

#### 5.3.3 MusicEval

MusicEval 官方 HF 仓库 `BAAI/MusicEval`（`MusicEval-full.zip` ≈ 1.76 GB）：

```bash
huggingface-cli download BAAI/MusicEval --repo-type dataset \
  --local-dir phase7_release/raw_hf/MusicEval
unzip phase7_release/raw_hf/MusicEval/MusicEval-full.zip \
  -d phase7_release/datasets/musiceval/
```

### 5.4 生成主清单 + BT / MSE 双缩放（层 3a）

按 `plan.md` §3.3 / §3.4 的规范，每个库要产出至少这五列：`source, audio_path, token_loss_path, score_bt_1to5, score_mse_1to5`。

> `token_loss_path` 只是一个稳定的**唯一字符串键**，`extract_loss_curves.py` 会用它做 MD5 去生成实际的 `.csv` 文件。直接填 `audio_path` 相同值或者 `"<dataset>/<item_id>"` 即可。

#### 5.4.1 pairwise 库（MusicPref / AIME / MusicArena）→ BT + MSE

用 `fit_pairwise_manifests.py` 产 1–5 分（ReLU hinge；与 `preprocess_data.py` 里的 BT/MSE 对应）：

```bash
cd /home/evev/noiseloss

# MusicPref：两个 head（musicality / fidelity），产两份 1-5 分
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicpref --head musicality \
  --out phase7_release/data/manifests/pairwise_relu/musicpref_musicality_1to5.csv
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicpref --head fidelity \
  --out phase7_release/data/manifests/pairwise_relu/musicpref_fidelity_1to5.csv

# AIME：两个 head（music_quality / text_audio_alignment）
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset aime --head music_quality \
  --out phase7_release/data/manifests/pairwise_relu/aime_music_quality_1to5.csv
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset aime --head text_audio_alignment \
  --out phase7_release/data/manifests/pairwise_relu/aime_text_audio_alignment_1to5.csv

# MusicArena：基于 battle JSON
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicarena \
  --out phase7_release/data/manifests/pairwise_relu/musicarena_1to5.csv
```

再用 `preprocess_data.py pairwise-scale-and-split` 走一遍 **BT + MSE 双缩放** 并切 `train/val/test`。入参是**已展开的 pairwise 对 CSV**（列 `item_a, item_b, outcome`，可通过上面 `fit_pairwise_manifests.py` 的中间数据或 `aime_join_survey.py` 得到）：

```bash
# 以 AIME 为例：先把 AIME-survey 的 parquet 转成带 outcome 的 pairwise CSV
conda run -n torch21 python phase7_release/scripts/data/aime_join_survey.py \
  --survey-parquet phase7_release/raw_hf/AIME-survey/data/train-00000-of-00001.parquet \
  --out-dir phase7_release/data/manifests

# 然后对该 pairwise CSV 同时计算 BT 与 MSE 1-5 分，并按 8/1/1 随机切分
bash phase7_release/scripts/run/run_data_preprocess.sh pairwise \
  phase7_release/data/manifests/aime_pairwise_survey.csv \
  phase7_release/data/manifests/aime_bt_mse_1to5.csv \
  phase7_release/data/full_splits/aime \
  track_1_id_str track_2_id_str answer
```

> `run_data_preprocess.sh pairwise` 产出的 split 只含 `item_id, score_bt_1to5, score_mse_1to5`。后续还要补 `audio_path`、`token_loss_path` 两列（见 §5.5）。MusicPref 与 MusicArena 按同样方法执行。

#### 5.4.2 原生 MOS 库（MusicEval / SongEval）

- **MusicEval**：已经是 1–5 实分，不做 BT/MSE，直接在 `data/splits/musiceval/` 使用现成 CSV。
- **SongEval**：5 维 Likert，每行在 `metadata.jsonl` 里是 `{id, dim1..dim5}`。按 plan §3.3 的策略**保留原分**，取 5 维均值作为 `score`：

```bash
conda run -n torch21 python - <<'PY'
import json, pandas as pd, os
rows = []
with open("phase7_release/raw_hf/SongEval/metadata.jsonl") as f:
    for line in f:
        r = json.loads(line)
        score = sum(r[k] for k in r if k.startswith("dim")) / 5.0
        rows.append({
            "source": "SongEval2025",
            "item_id": r["id"],
            "score": score,
            "audio_path": f"phase7_release/datasets/songeval/audio/{r['id']}.wav",
            "token_loss_path": f"songeval/{r['id']}",
        })
df = pd.DataFrame(rows)
os.makedirs("phase7_release/data/manifests", exist_ok=True)
df.to_csv("phase7_release/data/manifests/songeval_native_1to5.csv", index=False)
print(f"wrote {len(df)} rows")
PY
```

### 5.5 生成训练用 split（层 3b）

大规模 runner 只认以下目录（定义在 `config/data/full_datasets.yaml`）：

```text
phase7_release/data/full_splits/{musicpref,aime,songeval,music_arena,all_5_datasets}/
  train.csv  val.csv  test.csv
  train_noisy.csv  val_noisy.csv  test_noisy.csv
```

每个 CSV 至少要有 `score, audio_path, token_loss_path`（可附 `length`、`source`）。推荐用下面的 helper 补齐并切分（随机 80/10/10，`seed=42`）：

```bash
conda run -n torch21 python - <<'PY'
import os, sys, numpy as np, pandas as pd
from pathlib import Path

# === 按库改这里 ===
ds_name = "aime"                     # musicpref / aime / songeval / music_arena
manifest = "phase7_release/data/manifests/aime_bt_mse_1to5.csv"   # 或 songeval_native_1to5.csv
audio_dir = "phase7_release/datasets/aime/audio"
label_col = "score_bt_1to5"          # BT 版；改 "score_mse_1to5" 或 "score" 得到其它版本
use_native = False                    # SongEval/MusicEval 设为 True 直接用 score 列

out_dir = Path(f"phase7_release/data/full_splits/{ds_name}")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(manifest)
if use_native:
    df["score"] = df["score"].astype(float)
else:
    df["score"] = df[label_col].astype(float)

# 注入 audio_path（以 item_id 命名的 wav）与 token_loss_path 唯一键
if "audio_path" not in df.columns:
    df["audio_path"] = df["item_id"].map(lambda x: f"{audio_dir}/{x}.wav")
df["audio_path"] = df["audio_path"].map(lambda p: p if os.path.isabs(p)
                                         else os.path.abspath(p))
if "token_loss_path" not in df.columns:
    df["token_loss_path"] = df["item_id"].map(lambda x: f"{ds_name}/{x}")

df = df[df["audio_path"].map(os.path.exists)].reset_index(drop=True)
rng = np.random.default_rng(42)
perm = rng.permutation(len(df))
n_tr = int(0.8 * len(df)); n_va = int(0.1 * len(df))
splits = {
    "train": df.iloc[perm[:n_tr]],
    "val":   df.iloc[perm[n_tr:n_tr+n_va]],
    "test":  df.iloc[perm[n_tr+n_va:]],
}
cols = ["score", "audio_path", "token_loss_path"]
for k, part in splits.items():
    part[cols].to_csv(out_dir / f"{k}.csv", index=False)
    # noisy 版：此处先与 clean 相同；若有独立 noisy 标签再覆盖
    part[cols].to_csv(out_dir / f"{k}_noisy.csv", index=False)
    print(k, len(part))
PY
```

> `noisy` split 如果没有单独的 noisy 标签，就直接复用 clean 版本（两份文件内容相同），只是让 `--splits noisy` 这条管线能跑通。如果有单独的 noisy 标注文件（例如把 MusicPref 的 tie 全部转为噪声对），改写这一段即可。

### 5.6 合库 `all_5_datasets`（merged）

按 plan §3.4 / §7 Phase D 的 **source-aware split** 要求：每库各自切好 `train/val/test` 后，**按 split 维度纵向拼接**，确保同一条样本不会跨 split 泄漏：

```bash
conda run -n torch21 python - <<'PY'
import pandas as pd, os
from pathlib import Path
names = ["musicpref", "aime", "songeval", "music_arena", "musiceval"]
splits = ["train", "val", "test", "train_noisy", "val_noisy", "test_noisy"]
out = Path("phase7_release/data/full_splits/all_5_datasets")
out.mkdir(parents=True, exist_ok=True)
for s in splits:
    parts = []
    for n in names:
        p = Path(f"phase7_release/data/full_splits/{n}/{s}.csv")
        if p.exists():
            df = pd.read_csv(p); df["source"] = n; parts.append(df)
    if not parts: continue
    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(out / f"{s}.csv", index=False)
    print(s, len(merged))
PY
```

### 5.7 快速自检

```bash
# 每个 split 都应存在且非空
for d in musicpref aime songeval music_arena all_5_datasets; do
  for s in train val test train_noisy val_noisy test_noisy; do
    f="phase7_release/data/full_splits/$d/$s.csv"
    [[ -s "$f" ]] && echo "ok  $f  $(wc -l < "$f") lines" || echo "MISSING $f"
  done
done

# 抽查 100 条 audio_path 是否真的存在
conda run -n torch21 python - <<'PY'
import pandas as pd, os, random
for n in ["musicpref","aime","songeval","music_arena","all_5_datasets"]:
    df = pd.read_csv(f"phase7_release/data/full_splits/{n}/train.csv")
    miss = sum(1 for p in random.sample(df["audio_path"].tolist(), min(100,len(df)))
               if not os.path.exists(p))
    print(n, "missing_audio=", miss, "/100")
PY
```

### 5.8 Loss / Entropy / SAE 特征

**不用预先抽**：`run_musiceval_14_experiments.sh`（小规模）和 `run_full_14_experiments_parallel.sh`（大规模）都会在首个 step 自动调用：

- `scripts/features/extract_loss_curves.py`（`torch21`，MusicGen-small 抽 per-token loss）
- `scripts/features/extract_entropy_curves.py`（`torch21`，抽分布熵）
- `scripts/features/extract_sae_features.py`（`musicdiscovery310`，抽 SAE）

产出分别落到 `outputs/features/{loss,entropy,sae}/<dataset>/<splits>/{train,val,test}/`，并在 `run_state/` 下写对应 manifest。

> 这几步一次跑完后**可复用**：再次启动同一 `run_tag` 时会跳过已存在的 `.csv/.npy` 文件，不会重跑 MusicGen。

### 5.9 MusicEval 小规模的音频路径（特别说明）

`phase7_release/data/splits/musiceval/*.csv` 的 `audio_path` 与 `token_loss_path` 是**绝对路径**（`/home/evev/noiseloss/...`）。两种处理方案：

- **方案 A（推荐）**：服务器项目根保持 `/home/evev/noiseloss`，同步音频与 per-token loss 源：

```bash
rsync -av --progress \
  /home/evev/noiseloss/datasets/Phase5_2/MusicEval-full/wav/ \
  <server>:/home/evev/noiseloss/datasets/Phase5_2/MusicEval-full/wav/

rsync -av --progress \
  /home/evev/noiseloss/experiments/phase7/loss_eval_experiments/exp10_large_scale_eval/results/per_token_losses/wav_tokens/ \
  <server>:/home/evev/noiseloss/experiments/phase7/loss_eval_experiments/exp10_large_scale_eval/results/per_token_losses/wav_tokens/
```

- **方案 B**：项目根不同，重新生成 split 后把 `experiments/phase7/loss_eval_experiments/exp11_large_scale_replication/1_data_preparation/` 目录同步过来，再执行：

```bash
bash phase7_release/scripts/run/run_data_preprocess.sh musiceval
```

等价于：

```bash
conda run -n torch21 python phase7_release/scripts/data/preprocess_data.py \
  --config phase7_release/config/paths.yaml \
  musiceval-copy-splits
```

---

## 6. 执行流程

所有入口都在 `phase7_release/scripts/run/`。每个脚本都是 **幂等 + 可断点续跑**：同一 `run-tag` 会把每个 step 的完成状态写入 `phase7_release/outputs/run_state/<run_tag>.state.json`，中途 Ctrl+C 或宕机后重跑就会跳过已完成的步骤。

推荐顺序：A → B → C → D → E。

### Phase A — 一次性环境准备

```bash
cd /home/evev/noiseloss
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh
```

验证：

```bash
conda env list | grep -E "torch21|musicdiscovery310|audiobox"
ls external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/
```

### Phase B — MusicEval 14 实验（小规模闸门）

```bash
cd /home/evev/noiseloss
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
```

如需跑 noisy 标签版：

```bash
SPLITS=noisy bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
# 等价：bash .../run_musiceval_14_experiments.sh --splits noisy
```

脚本内部会依次执行（约 20 个 step，按 `tqdm` 展示每一步耗时）：

1. `prep_data_splits`：把 `config/paths.yaml` 指向的 split CSV 复制/校验到 `phase7_release/data/splits/musiceval/`；
2. `prep_loss_features`：MusicGen-small 前向抽 per-token loss（train/val/test）；
3. `prep_sae_features`：musicdiscovery SAE 特征（train/val/test，`musicdiscovery310` 环境）；
4. `prep_entropy_features`：per-token 分布熵曲线（train/val/test）；
5. `baseline_aesthetics`、`baseline_mean_loss`、`baseline_rescaled_eval`；
6. `f01_*` ~ `f07_*`：7 组合 × {CNN, Transformer} = 14 个训练 + 预测 step；
7. `eval_14_experiments`：汇总每个 family 的 Pearson/Spearman + 散点图。

常用开关：

```bash
# 跳过 transformer 分支（只跑 CNN），适合算力紧张时先拿到一组曲线
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --skip-transformer

# 跳过 SAE 相关实验（f03/f05/f06/f07 的 SAE 输入部分）
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --skip-sae

# 不跑 audiobox aesthetics baseline（省一个环境）
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --no-with-aesthetics

# 强制从头再跑某个 step
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --rerun-step f02_entropy_only_cnn

# 彻底重置状态（删除 run_state 下对应文件）
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --reset-state
```

预期输出（`phase7_release/outputs/` 下）：

```text
outputs/
  run_state/
    musiceval14_musiceval_clean.state.json      # 续跑状态
    musiceval14_musiceval_clean.lock.json        # 运行锁
    runtime_configs/musiceval14_*.yaml           # 每次启动固化的 runtime 配置
    runtime_splits/musiceval14_*/{train,val,test}.csv
    timelines/musiceval14_*.csv                  # 每 step 耗时
  features/
    entropy/musiceval/clean/{train,val,test}/*.npy
    loss/musiceval/clean/{train,val,test}/*.csv
  features/sae/musiceval/clean/...               # musicdiscovery 产出
  checkpoints/musiceval/clean/<run_name>/best.pth
  logs/musiceval/clean/<run_name>/tensorboard/
  reports/musiceval/clean/
    f01_loss_only_cnn_clean_test_scores.csv
    f01_loss_only_transformer_clean_test_scores.csv
    ...
    aesthetics_scores_{train,val,test}[_noisy].csv
    mean_loss_scores_*.csv
    eval_14_experiments_summary_clean.csv        # 主结果表
    eval_14_experiments_plots_clean/*.png         # 散点图
```

**Phase B 通过条件**：`eval_14_experiments_summary_clean.csv` 存在、14 个 `f0?_*_test_scores.csv` 齐全、Pearson/Spearman 数值合理、无 step 处于 `running` 状态。

### Phase C — Segment-level RNN 分析（可选，但推荐）

```bash
cd /home/evev/noiseloss
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh                 # loss-only
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh --with-entropy  # loss + entropy
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh --use-noisy     # noisy 标签
```

产出：

```text
phase7_release/outputs/checkpoints/segment/segment_rnn_<splits>_<mode>.pth
phase7_release/outputs/reports/segment_curves/*.csv
```

### Phase D — Full-scale 单库 + 合库（并发调度）

前置：§5.5 / §5.6 产出的 `phase7_release/data/full_splits/` 齐全，对应音频也已落到 `phase7_release/datasets/<name>/audio/`。

当前 `config/data/full_datasets.yaml` 默认启用 4 个单库（`musicpref / aime / songeval / music_arena`）+ 1 个合库（`all_5_datasets`）；`musiceval` 在此被禁用以避免与 Phase B 重复。因此一次 `--splits clean` 会产生 **4 × 14 + 1 × 14 = 70 个主实验**；`--splits noisy` 再跑一遍 → 共 140。

```bash
cd /home/evev/noiseloss
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh --splits clean
# noisy：
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh --splits noisy
```

运行机制（`run_full_14_experiments_parallel.py`）：

1. 读取 `phase7_release/config/data/full_datasets.yaml`；
2. 遍历 `datasets.large_scale_single` + `datasets.large_scale_merged` 中 `enabled: true` 的条目；
3. 为每个数据集生成一份独立的 runtime config（`outputs/run_state/full_generated_configs/*.yaml`），输出路径自动按 `outputs/full/.../<dataset>/<splits>/` 隔离，互不覆盖；
4. 调用 `nvidia-smi` 探测满足 `min_free_memory_mb=60000` 且 `max_utilization_pct<=35` 的 GPU；
5. 每个数据集分配 1 张卡，以 `CUDA_VISIBLE_DEVICES=<idx>` 启动一个 `run_musiceval_14_experiments.py` 子进程（`--no-prepare-splits`），每 `probe_interval_sec=20` 秒轮询；
6. 最多 `max_concurrent_jobs=8` 并发；失败/成功统一写入 `outputs/run_state/full_parallel_summary_<splits>.csv`，每任务日志在 `outputs/run_state/full_parallel_logs/*.log`。

> 合库 `all_5_datasets` 走的就是和单库一样的 14-step 流程，只是输入 CSV 是 5 个库纵向拼接后的结果（见 §5.6），因此天然具备 source-aware 的切分（每个 `source` 列的 test 都保留在同一 split），满足 plan §7 Phase D 的要求。

GPU 阈值、并发上限、GPU 白名单都在 `config/data/full_datasets.yaml → gpu_parallel` 下调整。

```yaml
gpu_parallel:
  enabled: true
  max_concurrent_jobs: 8
  gpu_ids: []              # 空 = 自动检测所有可见 GPU
  detection:
    min_free_memory_mb: 60000
    max_utilization_pct: 35
    probe_interval_sec: 20
```

干跑（只打印要提交的任务，不真实启动）：

```bash
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh --splits clean --dry-run
```

重跑某个失败的数据集（只需重置该数据集对应的 run_state）：

```bash
rm phase7_release/outputs/run_state/full14_<dataset>_<splits>.state.json
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh --splits clean
```

### Phase E — 汇总与交付

1. 从 `phase7_release/outputs/reports/musiceval/<splits>/eval_14_experiments_summary_<splits>.csv` 拿小规模主表；
2. 从 `phase7_release/outputs/full/reports/<dataset>/<splits>/eval_14_experiments_summary_<splits>.csv` 拿每个大规模数据集的主表；
3. 使用同一列命名（`family, backbone, pearson, spearman, dataset, splits`）合并成最终结果表；
4. 交付目录建议：

```text
deliverables/
  summary_musiceval_clean.csv
  summary_musiceval_noisy.csv
  summary_full_<dataset>_clean.csv
  summary_full_<dataset>_noisy.csv
  plots/                            # 从 outputs/reports/**/eval_14_experiments_plots_*/ 汇总
  segment_curves/                   # Phase C 产出
  logs/                             # run_state/*.state.json + timelines/*.csv 摘要
```

---

## 7. 断点续跑与锁机制

- **续跑**：默认开启（`--resume`）。每个 step 以命令级别记录 `completed_commands`，重新启动会自动跳过已完成的命令。
- **锁**：`outputs/run_state/<run_tag>.lock.json` 记录运行进程的 PID；同一 run-tag 的两个实例不会并发启动。脚本正常退出或异常时锁会被释放；如果进程被 `kill -9`，手动删除 lock 文件即可。
- **强制重跑单个 step**：

```bash
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --rerun-step prep_sae_features
```

- **彻底重置**：

```bash
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --reset-state
```

---

## 8. 快速自检清单

在服务器上按顺序执行以下自检：

```bash
cd /home/evev/noiseloss

# 1. git & submodules
git status
git submodule status

# 2. conda 环境
conda env list | grep -E "torch21|musicdiscovery310|audiobox"
conda run -n torch21 python -c "import torch; print(torch.cuda.device_count())"

# 3. SAE checkpoint
ls external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/

# 4. MusicEval split
head -n 2 phase7_release/data/splits/musiceval/test.csv
head -n 2 phase7_release/data/splits/musiceval/test.csv | awk -F, 'NR==2{system("ls -l "$2)}'

# 5. full datasets split（若跑大规模）
ls phase7_release/data/full_splits/*/test.csv 2>/dev/null

# 6. GPU
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
```

任一条不通过，先修好再进入 Phase B/D。

---

## 9. 常见问题

- **`RuntimeError: Run lock exists for tag=...`**：上次进程没释放锁。确认无同名 Python 进程后 `rm phase7_release/outputs/run_state/<run_tag>.lock.json`。
- **`No eligible GPUs found`（仅 Phase D）**：当前 GPU 都不满足 `min_free_memory_mb / max_utilization_pct` 阈值。先 `nvidia-smi` 确认是否有其它任务在跑；若仅需较小显存可在 `config/data/full_datasets.yaml` 调低 `min_free_memory_mb`。
- **SAE checkpoint 下载失败**：请确保 S3 bucket 可访问（境外 IP 友好）；若服务器无外网，先在堡垒机下载好 3 个文件后 rsync 到 `external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/`。
- **`audiocraft` 与 `musicdiscovery` 依赖冲突**：`setup_sae_musicdiscovery.sh` 已内置 patch，不要手动 `pip install -r external/musicdiscovery/requirements.txt`。
- **音频路径找不到**：若是 MusicEval，见 §5.9 的方案 A/B；若是大规模数据集，回到 §5.3 / §5.7 抽查 `datasets/<name>/audio/` 是否齐全。
- **显存不够导致 OOM（hybrid 模型）**：在 `run_musiceval_14_experiments.sh` 命令后追加：

```bash
--hybrid-cnn-batch-size 8 --hybrid-transformer-batch-size 6 \
--hybrid-cnn-sae-only-batch-size 16 --hybrid-num-workers 4 --hybrid-prefetch-factor 4
```

---

## 10. 最小执行摘要（TL;DR）

从 0 开始的完整顺序（对应 plan §7 Phase A → E）：

```bash
# === Phase A：环境 ===
# 0) 系统依赖
sudo apt-get install -y git git-lfs curl build-essential ffmpeg libsndfile1 && git lfs install

# 1) 代码 + 子模块
cd /home/evev && git clone <repo> noiseloss && cd noiseloss && git submodule update --init --recursive

# 2) 主 conda 环境
conda create -y -n torch21 python=3.10 && conda activate torch21
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pandas pyyaml tqdm matplotlib soundfile librosa transformers==4.41.* safetensors tensorboard einops scikit-learn
pip install datasets==2.21.0 huggingface_hub

# 3) SAE + musicdiscovery310 + audiobox
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh

# === 数据（§5）===
# 4) HF 原始快照
huggingface-cli login
for repo in \
  "i-need-sleep/musicprefs:MusicPref" \
  "disco-eth/AIME:AIME" \
  "disco-eth/AIME-survey:AIME-survey" \
  "ASLP-lab/SongEval:SongEval" \
  "gneubig/music-arena-public:MusicArena" \
  "BAAI/MusicEval:MusicEval"; do
  IFS=":" read -r hf local <<< "$repo"
  huggingface-cli download "$hf" --repo-type dataset \
    --local-dir "phase7_release/raw_hf/$local"
done

# 5) 音频落地 datasets/<name>/audio/ + 主清单 + BT/MSE 双缩放（§5.3~§5.4）
# 按 §5.3/§5.4 的命令分别对 MusicPref / AIME / SongEval / MusicArena 执行。

# 6) 生成 full_splits/<dataset>/{train,val,test[,*_noisy]}.csv（§5.5），
#    再生成合库 all_5_datasets（§5.6），自检通过（§5.7）。

# 7) 同步 MusicEval 的音频与 per-token-loss（§5.9 方案 A 最简）

# === Phase B：小规模闸门 ===
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh
bash phase7_release/scripts/run/run_musiceval_14_experiments.sh --splits noisy

# === Phase C：Segment 分析 ===
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh
bash phase7_release/scripts/run/run_segment_rnn_analysis.sh --with-entropy

# === Phase D：大规模（4 单库 + 1 合库，每次 70 个主实验）===
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh --splits clean
bash phase7_release/scripts/run/run_full_14_experiments_parallel.sh --splits noisy

# === Phase E：按 §6 Phase E 汇总 deliverables/ ===
```

结束后：

- 小规模主表：`phase7_release/outputs/reports/musiceval/<splits>/eval_14_experiments_summary_<splits>.csv`
- 大规模主表：`phase7_release/outputs/full/reports/<dataset>/<splits>/eval_14_experiments_summary_<splits>.csv`
- 所有 `<dataset> ∈ {musicpref, aime, songeval, music_arena, all_5_datasets}`
