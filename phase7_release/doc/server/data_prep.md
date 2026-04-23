# Phase7 Release — 数据准备

三层结构（对齐 `doc/plan/phase7_release_plan.md` §3）：

```text
phase7_release/
  raw_hf/<Dataset>/          # 层 1：HF 原始快照（只读，不进 git）
  datasets/<dataset>/audio/  # 层 2：统一落地的音频
  data/manifests/            # 层 3a：带 score_bt_1to5 / score_mse_1to5 的主清单
  data/splits/musiceval/     # 层 3b-a：MusicEval 小规模 split（随仓库）
  data/full_splits/<dataset>/# 层 3b-b：大规模 split（由本文生成）
```

训练脚本只读 `data/splits/` 与 `data/full_splits/` 下的 CSV，不直接读 `raw_hf/`。

---

## 5.1 数据源

| 库 | HF repo | 类型 | 用途 |
|---|---|---|---|
| MusicEval 2025 | `BAAI/MusicEval` | 原生 MOS | 小规模闸门 |
| MusicPref 2025 | `i-need-sleep/musicprefs` | pairwise | 大规模单库 |
| AIME 2025 | `disco-eth/AIME` | 音频 | 大规模单库（配 survey） |
| AIME-survey 2025 | `disco-eth/AIME-survey` | pairwise | AIME 的 label |
| SongEval 2025 | `ASLP-lab/SongEval` | 5 维 Likert | 大规模单库（原生分） |
| MusicArena 2025 | `music-arena/music-arena-dataset` | pairwise（battle JSON） | 大规模单库 |

前置：确保已完成 HF 登录（见 [setup.md §4](setup.md#4-huggingface-登录)）。离线服务器先在跳板机上下载后 rsync 过去。

---

## 5.2 下载 HF 原始快照（层 1）

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/download_hf_datasets.sh
```

（脚本会提示 `huggingface-cli login`，然后顺序下载全部 6 个仓库。）

下载完成后应有：

```text
phase7_release/raw_hf/
  AIME/data/train-*.parquet
  AIME-survey/data/train-*.parquet
  MusicPref/data/train-*.parquet  MusicPref/human_preference.csv
  SongEval/mp3/*.mp3 + metadata.jsonl
  MusicArena/audio_files/... + battle_data/**/*.json
  MusicEval/MusicEval-full.zip
```

---

## 5.3 把音频落地到 `datasets/<name>/audio/`（层 2）

### 5.3.1 HF datasets 可迭代的仓库

```bash
# MusicPref（parquet 内嵌 audio）
conda run -n torch21 python phase7_release/scripts/data/hf_ingest_smoke.py \
  --repo i-need-sleep/musicprefs --source-tag MusicPref2025 \
  --out-audio-dir phase7_release/datasets/musicprefs/audio \
  --master-csv phase7_release/data/manifests/master_index.csv \
  --max-samples 0

# AIME（parquet 内嵌 audio）
conda run -n torch21 python phase7_release/scripts/data/hf_ingest_smoke.py \
  --repo disco-eth/AIME --source-tag AIME2025 \
  --out-audio-dir phase7_release/datasets/aime/audio \
  --master-csv phase7_release/data/manifests/master_index.csv \
  --max-samples 0

# SongEval（mp3 → wav）
find phase7_release/raw_hf/SongEval/mp3 -name '*.mp3' -print0 | \
  xargs -0 -n1 -P8 -I{} bash -c '
    out="phase7_release/datasets/songeval/audio/$(basename "{}" .mp3).wav"
    mkdir -p "$(dirname "$out")"
    [[ -f "$out" ]] || ffmpeg -y -loglevel error -i "{}" -ac 1 -ar 32000 "$out"
  '
```

### 5.3.2 MusicArena

```bash
rsync -av phase7_release/raw_hf/MusicArena/audio_files/ \
          phase7_release/datasets/music_arena/audio/
```

### 5.3.3 MusicEval

```bash
unzip phase7_release/raw_hf/MusicEval/MusicEval-full.zip \
  -d phase7_release/datasets/musiceval/
```

---

## 5.4 生成主清单 + BT / MSE 双缩放（层 3a）

每个库要产出：`source, audio_path, token_loss_path, score_bt_1to5, score_mse_1to5`。

> `token_loss_path` 是唯一字符串键（`extract_loss_curves.py` 用它做 MD5 命名文件），直接填 `audio_path` 或 `"<dataset>/<item_id>"` 均可。

### 5.4.1 pairwise 库（MusicPref / AIME / MusicArena）

```bash
cd "${PROJECT_ROOT}"

# MusicPref（musicality / fidelity 两个 head）
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicpref --head musicality \
  --out phase7_release/data/manifests/pairwise_relu/musicpref_musicality_1to5.csv
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicpref --head fidelity \
  --out phase7_release/data/manifests/pairwise_relu/musicpref_fidelity_1to5.csv

# AIME（music_quality / text_audio_alignment 两个 head）
conda run -n torch21 python phase7_release/scripts/data/aime_join_survey.py \
  --survey-parquet phase7_release/raw_hf/AIME-survey/data/train-00000-of-00001.parquet \
  --out-dir phase7_release/data/manifests
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset aime --head music_quality \
  --out phase7_release/data/manifests/pairwise_relu/aime_music_quality_1to5.csv
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset aime --head text_audio_alignment \
  --out phase7_release/data/manifests/pairwise_relu/aime_text_audio_alignment_1to5.csv

# MusicArena
conda run -n torch21 python phase7_release/scripts/data/fit_pairwise_manifests.py \
  --dataset musicarena \
  --out phase7_release/data/manifests/pairwise_relu/musicarena_1to5.csv
```

BT + MSE 双缩放并切 train/val/test（以 AIME 为例）：

```bash
bash phase7_release/scripts/run/run_data_preprocess.sh pairwise \
  phase7_release/data/manifests/aime_pairwise_survey.csv \
  phase7_release/data/manifests/aime_bt_mse_1to5.csv \
  phase7_release/data/full_splits/aime \
  track_1_id_str track_2_id_str answer
```

MusicPref / MusicArena 按同样方式执行。

### 5.4.2 原生 MOS 库（MusicEval / SongEval）

- **MusicEval**：直接使用 `data/splits/musiceval/` 中的现成 CSV（见 §5.9）。
- **SongEval**：取 5 维均值作为 `score`：

```bash
conda run -n torch21 python - <<'PY'
import json, pandas as pd, os
rows = []
with open("phase7_release/raw_hf/SongEval/metadata.jsonl") as f:
    for line in f:
        r = json.loads(line)
        score = sum(r[k] for k in r if k.startswith("dim")) / 5.0
        rows.append({
            "source": "SongEval2025", "item_id": r["id"], "score": score,
            "audio_path": f"phase7_release/datasets/songeval/audio/{r['id']}.wav",
            "token_loss_path": f"songeval/{r['id']}",
        })
df = pd.DataFrame(rows)
os.makedirs("phase7_release/data/manifests", exist_ok=True)
df.to_csv("phase7_release/data/manifests/songeval_native_1to5.csv", index=False)
print(f"wrote {len(df)} rows")
PY
```

---

## 5.5 生成训练用 split（层 3b）

大规模 runner 读取：

```text
phase7_release/data/full_splits/{musicpref,aime,songeval,music_arena,all_5_datasets}/
  train.csv  val.csv  test.csv
```

每个 CSV 需包含 `score, audio_path, token_loss_path`。用下面的 helper 补齐并按 80/10/10 切分（`seed=42`）：

```bash
conda run -n torch21 python - <<'PY'
import os, numpy as np, pandas as pd
from pathlib import Path

ds_name  = "aime"       # 按库修改：musicpref / aime / songeval / music_arena
manifest = "phase7_release/data/manifests/aime_bt_mse_1to5.csv"
audio_dir = "phase7_release/datasets/aime/audio"
label_col = "score_bt_1to5"   # 或 "score_mse_1to5" / "score"
use_native = False             # SongEval/MusicEval 直接用 score 列时设 True

out_dir = Path(f"phase7_release/data/full_splits/{ds_name}")
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(manifest)
df["score"] = df["score" if use_native else label_col].astype(float)
if "audio_path" not in df.columns:
    df["audio_path"] = df["item_id"].map(lambda x: f"{audio_dir}/{x}.wav")
df["audio_path"] = df["audio_path"].map(lambda p: p if os.path.isabs(p) else os.path.abspath(p))
if "token_loss_path" not in df.columns:
    df["token_loss_path"] = df["item_id"].map(lambda x: f"{ds_name}/{x}")
df = df[df["audio_path"].map(os.path.exists)].reset_index(drop=True)
rng = np.random.default_rng(42)
perm = rng.permutation(len(df))
n_tr = int(0.8 * len(df)); n_va = int(0.1 * len(df))
splits = {"train": df.iloc[perm[:n_tr]], "val": df.iloc[perm[n_tr:n_tr+n_va]], "test": df.iloc[perm[n_tr+n_va:]]}
cols = ["score", "audio_path", "token_loss_path"]
for k, part in splits.items():
    part[cols].to_csv(out_dir / f"{k}.csv", index=False)
    print(k, len(part))
PY
```

---

## 5.6 合库 `all_5_datasets`

每库各自切好后，按 split 纵向拼接（source-aware split，不跨 split 泄漏）：

```bash
conda run -n torch21 python - <<'PY'
import pandas as pd
from pathlib import Path
names = ["musicpref", "aime", "songeval", "music_arena", "musiceval"]
splits = ["train", "val", "test"]
out = Path("phase7_release/data/full_splits/all_5_datasets")
out.mkdir(parents=True, exist_ok=True)
for s in splits:
    parts = []
    for n in names:
        p = Path(f"phase7_release/data/full_splits/{n}/{s}.csv")
        if p.exists():
            df = pd.read_csv(p); df["source"] = n; parts.append(df)
    if not parts: continue
    pd.concat(parts, ignore_index=True).to_csv(out / f"{s}.csv", index=False)
    print(s, sum(len(p) for p in parts))
PY
```

---

## 5.7 快速自检

```bash
# split 文件齐全且非空
for d in musicpref aime songeval music_arena all_5_datasets; do
  for s in train val test; do
    f="phase7_release/data/full_splits/$d/$s.csv"
    [[ -s "$f" ]] && echo "ok  $f" || echo "MISSING $f"
  done
done

# 抽查 audio_path 是否存在
conda run -n torch21 python - <<'PY'
import pandas as pd, os, random
for n in ["musicpref","aime","songeval","music_arena","all_5_datasets"]:
    df = pd.read_csv(f"phase7_release/data/full_splits/{n}/train.csv")
    miss = sum(1 for p in random.sample(df["audio_path"].tolist(), min(100,len(df))) if not os.path.exists(p))
    print(n, "missing_audio=", miss, "/100")
PY
```

---

## 5.8 Loss / Entropy / SAE 特征

**无需手动提取**：两个 runner 脚本会在首个 step 自动调用特征提取：

- `extract_loss_curves.py`（`torch21`）
- `extract_entropy_curves.py`（`torch21`）
- `extract_sae_features.py`（`musicdiscovery310`）

产出落到 `outputs/features/{loss,entropy,sae}/<dataset>/<splits>/`，已存在的文件会被跳过（可复用）。

---

## 5.9 MusicEval 小规模音频路径（特别说明）

`data/splits/musiceval/*.csv` 里的 `audio_path` 是 dev 机器上的绝对路径，拷到 server 后无法直接使用。三种处理方案：

**方案 A（推荐）**：重写路径 + 同步文件

```bash
bash phase7_release/scripts/run/run_data_preprocess.sh musiceval
```

**方案 B**：先从 dev 机器 rsync 音频与 per-token loss，再执行方案 A：

```bash
# 在 dev 机器上执行
rsync -av --progress "${SRC_ROOT}/datasets/Phase5_2/MusicEval-full/wav/" \
  "<server>:${DST_ROOT}/datasets/Phase5_2/MusicEval-full/wav/"
rsync -av --progress \
  "${SRC_ROOT}/experiments/phase7/loss_eval_experiments/exp10_large_scale_eval/results/per_token_losses/wav_tokens/" \
  "<server>:${DST_ROOT}/experiments/phase7/loss_eval_experiments/exp10_large_scale_eval/results/per_token_losses/wav_tokens/"
```

然后在 server 上跑一次方案 A。

**方案 C**：重新生成（把 `exp11_large_scale_replication/1_data_preparation/` 目录同步过来后跑方案 A）。
