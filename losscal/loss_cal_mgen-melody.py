import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import pandas as pd
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes
from audiocraft.modules.conditioners import WavCondition
from tqdm import tqdm

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
    model.lm = model.lm.to(dtype=torch.float32)
    model.compression_model = model.compression_model.to(dtype=torch.float32)

    compression_ratio = model.compression_model.frame_rate / model.sample_rate
    print(f"压缩比率: {compression_ratio:.4f} (每音频秒生成{model.compression_model.frame_rate}个tokens)")

    return model

def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reduction="mean"
    ):
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    
    if reduction == "none":
        # Return per-token losses
        ce_per_token = torch.zeros((K, T), device=targets.device)
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            
            # Compute cross entropy for all positions
            ce_all = F.cross_entropy(logits_k, targets_k, reduction='none').view(B, T)
            # Apply mask and store
            ce_per_token[k, :] = ce_all[0] * mask[0, k, :].float()
        return ce_per_token
    else:
        # Original mean reduction
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        ce = ce / K
        return ce, ce_per_codebook

def save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path=None):
    """
    将per-token loss保存为CSV文件，只保存平均值（与 loss_cal_small.py 一致）
    """
    if token_output_dir is None:
        return

    # 扁平化命名：使用 relative_path 的目录部分 + 基础文件名，路径分隔符替换为下划线
    if relative_path:
        rel_dir = os.path.dirname(relative_path)
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        if rel_dir:
            filename = f"{rel_dir.replace(os.sep, '_')}_{base_filename}"
        else:
            filename = base_filename
    else:
        filename = os.path.splitext(os.path.basename(audio_path))[0]

    avg_csv_file = os.path.join(token_output_dir, f"{filename}_tokens_avg.csv")
    os.makedirs(token_output_dir, exist_ok=True)

    try:
        # 转为 numpy（断开计算图以防 requires_grad）
        ce_np = ce_per_token.detach().cpu().numpy()

        # 仅保存每个 token 在四个 codebook 的平均值（K 维求均值）
        avg_records = []
        if ce_np.ndim == 3:  # (B, K, T)
            B, K, T = ce_np.shape
            for b in range(B):
                for t in range(T):
                    token_avg = float(ce_np[b, :, t].mean())
                    avg_records.append({'token_position': t, 'avg_loss_value': token_avg})
        elif ce_np.ndim == 2:  # (K, T)
            K, T = ce_np.shape
            for t in range(T):
                token_avg = float(ce_np[:, t].mean())
                avg_records.append({'token_position': t, 'avg_loss_value': token_avg})
        elif ce_np.ndim == 1:  # (T,)
            T = ce_np.shape[0]
            for t in range(T):
                avg_records.append({'token_position': t, 'avg_loss_value': float(ce_np[t])})

        if avg_records:
            avg_df = pd.DataFrame(avg_records)
            avg_df.to_csv(avg_csv_file, index=False)
            print(f"保存了 {len(avg_records)} 条平均值记录到 {avg_csv_file}")
            overall_mean = avg_df['avg_loss_value'].mean()
            print(f"总体平均loss: {overall_mean:.6f}")
            return overall_mean
        else:
            print(f"没有数据可保存: {filename}")
            return None
    except Exception as e:
        print(f"保存CSV时出错 {filename}: {str(e)}")
        return None


def process_single_audio(audio_path, model, device, loss_type="cross_entropy", reduction="mean", token_output_dir=None, relative_path=None):
    # 读取和预处理音频
    audio_read = model.sample_rate
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)  # 转为单声道
    wav = wav.unsqueeze(0).to(device)  # [1, 1, T]
    
    # 使用compression model进行tokenization
    with torch.no_grad():
        encoded_frames = model.compression_model.encode(wav)
        
    # 准备输入和目标tokens
    input_tokens = encoded_frames[0]  # [B, K, T]
    target_tokens = input_tokens.clone()
    
    # 创建mask
    B, K, T = input_tokens.shape
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    
    # 创建空的条件 - 避免数据泄露
    empty_wav = torch.zeros(1, 1, 1, device=device)  # 空的wav张量
    conditions = [ConditioningAttributes(
        text={'description': None},
        wav={'self_wav': WavCondition(
            wav=empty_wav,
            length=torch.tensor([1], device=device),
            sample_rate=torch.tensor([model.sample_rate], device=device)
        )}
    )]
    
    # 模型前向传播
    with torch.autocast(device_type='cuda', enabled=False):
        logits = model.lm.forward(input_tokens, conditions)
    
    # 计算loss
    logits = logits.reshape(-1, logits.size(-1))  # [B*K*T, vocab_size]
    target_tokens = target_tokens.reshape(-1)  # [B*K*T]
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(logits, target_tokens)  # [B*K*T]
    losses = losses.reshape(B, K, T)  # [B, K, T]
    
    # 应用mask并计算平均loss
    mask_expanded = mask.unsqueeze(1).expand(-1, K, -1)  # [B, K, T]
    masked_losses = losses * mask_expanded.float()
    total_loss = masked_losses.sum() / mask_expanded.sum()
    
    # 直接使用masked_losses作为per-token loss，取第一个batch和第一个codebook
    per_token_losses = masked_losses[0, 0, :]  # [T] - 取第一个batch的第一个codebook
    
    # 保存结果
    if token_output_dir:
        save_tokens_as_csv(per_token_losses, audio_path, token_output_dir, relative_path)
    
    return total_loss.item()

def process_audio_directory(audio_dir, model, device, output_file=None, loss_type="cross_entropy", reduction="mean", token_output_dir=None):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), audio_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        raise ValueError(f"目录 {audio_dir} 中没有WAV/MP3文件")

    print(f"\n开始处理 {len(audio_files)} 个音频 (损失类型: {loss_type})")
    results = []
    error_files = [] 

    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"结果将保存到: {output_file}")

    for rel_filename in tqdm(audio_files, desc="处理进度"):
        audio_path = os.path.join(audio_dir, rel_filename)
        loss = process_single_audio(audio_path, model, device, loss_type, reduction, token_output_dir, rel_filename)
        if loss is not None:
            results.append((rel_filename, loss))
            if out_file:
                out_file.write(f"{rel_filename}: {loss:.8f}\n")
                out_file.flush()  # Ensure data is written immediately
        else:
            error_files.append(rel_filename)

    if out_file:
        out_file.close()

    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"处理失败的文件数量: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n处理失败的 {len(error_files)} 个文件已保存到: {error_file_path}")

    return results, error_files

def get_missing_files(audio_dir, results_file):
    all_audio_files = set(f for f in os.listdir(audio_dir)
                         if f.lower().endswith(('.wav', '.mp3')))

    processed_files = set()
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename = line.split(':', 1)[0].strip()
                    processed_files.add(filename)
    except FileNotFoundError:
        print(f"结果文件 {results_file} 不存在，将处理所有文件")

    missing_files = all_audio_files - processed_files
    return list(missing_files)

def process_missing_files(audio_dir, results_file, loss_type="cross_entropy"):
    """处理缺失的文件（使用CPU）"""
    missing_files = get_missing_files(audio_dir, results_file)

    if not missing_files:
        print("没有缺失的文件需要处理")
        return

    print(f"发现 {len(missing_files)} 个未处理的文件，将使用CPU处理 (损失类型: {loss_type})")

    print("\n未处理的文件列表:")
    for filename in missing_files:
        print(f"{filename}")
    print("\n开始处理这些文件...\n")

    device = "cpu"
    print(f"Using device: {device}")
    model = load_model(device)

    with open(results_file, 'a') as out_file:
        for filename in tqdm(missing_files, desc="CPU处理进度"):
            audio_path = os.path.join(audio_dir, filename)
            loss = process_single_audio(audio_path, model, device, loss_type)
            if loss is not None:
                out_file.write(f"{filename}: {loss:.4f}\n")
                out_file.flush()  # Ensure data is written immediately

def main():
    parser = argparse.ArgumentParser(description="计算音频文件的loss值")
    parser.add_argument("--audio_dir", type=str, default="pitch_out_wav",
                        help="音频文件目录")
    parser.add_argument("--output_file", type=str, default="results.txt",
                        help="输出结果文件路径")
    parser.add_argument("--process_missing", action="store_true",
                        help="处理缺失的文件（使用CPU）")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU处理所有文件")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "mse"], default="cross_entropy",
                        help="损失函数类型: cross_entropy或mse")
    parser.add_argument("--reduction", type=str, choices=["mean", "none"], default="mean", 
                        help="Loss reduction类型: mean或none (保存per-token losses)")
    parser.add_argument("--token_output_dir", type=str, default=None, 
                        help="当reduction=none时保存per-token losses的目录")

    args = parser.parse_args()
    
    if args.reduction == "none" and not args.token_output_dir:
        args.token_output_dir = "token_losses_melody"
        print(f"使用默认token输出目录: {args.token_output_dir}")

    # Redirect stdout to file if output_file is specified and not processing missing files
    original_stdout = None
    if args.output_file and not args.process_missing:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        if args.process_missing and args.output_file:
            # Process missing files on CPU
            process_missing_files(args.audio_dir, args.output_file, args.loss_type)
        else:
            # Normal processing
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")

            model = load_model(device)

            assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"

            results, error_files = process_audio_directory(
                args.audio_dir, model, device, args.output_file, args.loss_type, args.reduction, args.token_output_dir
            )

            if not args.output_file:
                print("\n处理结果:")
                for filename, loss in results:
                    print(f"{filename}: {loss:.4f}")

    except Exception as e:
        print(f"错误发生: {str(e)}")
    finally:
        if 'device' in locals() and device == 'cuda':
            torch.cuda.empty_cache()
        print("显存已清理")

        # Restore stdout if redirected
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()