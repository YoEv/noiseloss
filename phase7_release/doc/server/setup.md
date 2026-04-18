# Phase7 Release — 环境配置

## 1. 服务器硬件与系统要求

- **GPU**：8×H100（80 GB）；full-scale 并发调度需要单卡 ≥60 GB 空闲显存。小规模 MusicEval 只需要 1 张可用卡。
- **CPU / 内存**：≥32 核 / ≥256 GB（hybrid 训练 `num_workers=8, prefetch_factor=8`）。
- **磁盘**：≥2 TB 可用（音频 + loss / entropy / SAE 特征）。
- **操作系统**：Linux（Ubuntu 22.04 已验证）；`torch==2.1.0 + CUDA` 兼容的发行版均可。
- **CUDA 驱动**：支持 `torch 2.1.0 + cu118/cu121` 的驱动（`nvidia-smi` 可见即可）。

系统级依赖（一次性）：

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
git clone <your_repo_url> noiseloss
cd noiseloss
git submodule update --init --recursive
```

关键目录：

```text
noiseloss/
  datasets/                 # 原始/中间音频（大文件，不进 git）
  external/                 # 第三方子模块
  phase7_release/           # 发行层：唯一入口
    config/paths.yaml
    config/data/full_datasets.yaml
    data/splits/musiceval/  # MusicEval 小规模 split CSV（随仓库）
    data/full_splits/       # 大规模 5 库 split CSV（由数据准备生成）
    scripts/run/            # 全部入口脚本
    outputs/                # 运行期自动生成
```

---

## 3. Conda 环境

需要 2 个 conda 环境：

| 环境名 | Python | 用途 |
|--------|--------|------|
| `torch21` | 3.10 | 训练、特征提取（loss/entropy/SAE）、评估、baseline mean_loss |
| `audiobox` | 3.10 | audiobox-aesthetics baseline 推理 |

> `musicdiscovery310` 是历史备用名：`setup_sae_musicdiscovery.sh` 仅在 `torch21` 为 Python<3.10 时才创建单独的 `musicdiscovery310` 环境。当前 `torch21` 是 3.10，**SAE 依赖直接装进 `torch21`**，无需第三个环境。

用脚本一键创建两个环境：

```bash
bash phase7_release/scripts/run/setup_environments.sh
```

也可手动按下面步骤创建。

### 3.1 `torch21`（主训练 + SAE 提取环境）

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
  tensorboard einops \
  datasets==2.21.0 huggingface_hub
```

> 注意：运行 `setup_sae_musicdiscovery.sh` 后，musicdiscovery 依赖会追加安装到 `torch21`，`transformers` 会被降级至 4.38.1（audiocraft 1.3.0 的要求）。

验证：

```bash
conda run -n torch21 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### 3.2 `audiobox`（aesthetics baseline）

```bash
conda create -y -n audiobox python=3.10
conda activate audiobox
pip install --upgrade pip
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e external/audiobox-aesthetics
pip install soundfile librosa tqdm pandas pyyaml
```

如不需要 aesthetics baseline，执行时追加 `--no-with-aesthetics` 跳过。

---

## 4. HuggingFace CLI 安装与登录

`hf` CLI 由 `huggingface_hub` 提供，已随 `torch21` 环境安装。将其加入系统 PATH 以便在任意 shell 中使用：

```bash
echo 'export PATH="/home/cliu/miniconda3/envs/torch21/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

> 如果 conda 安装在其他路径，将 `/home/cliu/miniconda3` 替换为实际路径（`conda info --base` 可查）。

数据下载（§5）需要 HF 账号和访问令牌：

```bash
hf auth login
# 按提示粘贴 HF token（https://huggingface.co/settings/tokens）
# Read 权限即可；访问 gated 仓库需先在 HF 网页上 Accept 对应协议
```

验证：

```bash
hf auth whoami
```

离线服务器无法直接登录时，在能联网的机器上完成登录，然后把 `~/.cache/huggingface/token` rsync 到服务器同路径。

---

## 5. 外部依赖与 Checkpoint

### 5.1 SAE（必须，一次性）

#### 5.1.0 预备：解决 PyAV 编译失败

`musicdiscovery` 依赖 `audiocraft==1.3.0`，后者 pin 了 `av==11.0.0`。若目标机器没有 FFmpeg dev 头文件，pip 会回退到源码编译并报错：

```text
Package 'libavformat', required by 'virtual:world', not found
ERROR: Failed to build 'av' when getting requirements to build wheel
```

**二选一**，在运行 §5.1.1 之前先执行：

**Option A（推荐）**：通过 conda-forge 装 FFmpeg dev 头（对后续 ffmpeg 调用也有益）

```bash
conda install -n torch21 -c conda-forge -y 'ffmpeg=6.*' pkg-config
```

**Option B**：强制使用预编译 wheel

```bash
conda run -n torch21 python -m pip install --only-binary=:all: av==11.0.0
```

#### 5.1.1 安装 musicdiscovery + 下载 SAE checkpoint

```bash
cd "${PROJECT_ROOT}"
bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh
```

脚本会初始化 submodule、下载 SAE checkpoint（从 S3）、安装 Python 依赖（必要时创建 `musicdiscovery310`）。

成功后应看到：

```text
external/musicdiscovery_checkpoints/sae-4_k_32_layer_12/facebook/musicgen-small/
  cfg.json  sae_weights.safetensors  sparsity.safetensors
```

> 若 S3 上 `layer_12` 不存在，脚本会自动回退到最近已知层并打印 `selected checkpoint prefix`，请核对。

### 5.2 MusicGen-small

首次运行特征提取时，`transformers` 会自动下载约 1.5 GB 到 `~/.cache/huggingface`。离线服务器需提前 rsync 过来。

### 5.3 audiobox-aesthetics checkpoint（可选）

首次调用 `AesPredictor()` 时自动下载。离线时参考 `external/audiobox-aesthetics/README.md` 手动放置。
