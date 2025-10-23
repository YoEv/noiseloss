import torch
import torchaudio
import os
import argparse
import sys
import librosa 
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import functional as F
import numpy as np
import pandas as pd

def setup_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
    model = model.to(device)
    model.eval()
    
    compression_ratio = model.config.audio_encoder.frame_rate / model.config.audio_encoder.sampling_rate
    print(f"压缩比率: {compression_ratio:.4f} (每音频秒生成{model.config.audio_encoder.frame_rate}个tokens)")
    return model, processor

def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reduction="mean"
    ):
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    
    if reduction == "none":
        # 返回每个codebook每个时间步的loss，形状为(B, K, T)
        ce_per_token = torch.zeros_like(targets, dtype=torch.float32)
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            
            ce_flat = F.cross_entropy(logits_k, targets_k, reduction="none")
            ce_masked = ce_flat * mask_k.float()
            ce_per_token[:, k, :] = ce_masked.view(B, T)
        
        return ce_per_token
    else:
        # 原有的平均计算逻辑
        ce = 0
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            
            ce_k = F.cross_entropy(logits_k, targets_k, reduction="none")
            ce_k = ce_k * mask_k.float()  # 应用mask
            ce += ce_k.sum() / mask_k.float().sum().clamp(min=1)  # 避免除零
        
        return ce / K

def save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path=None):
    """
    将per-token loss保存为CSV文件，只保存平均值（与 loss_cal_small.py 一致）
    """
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
        ce_np = ce_per_token.detach().cpu().numpy()
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
    """保存per-token losses到CSV文件"""
    if token_output_dir is None:
        return
    
    os.makedirs(token_output_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 如果有相对路径，创建对应的子目录结构
    if relative_path and relative_path != '.':
        csv_subdir = os.path.join(token_output_dir, relative_path)
        os.makedirs(csv_subdir, exist_ok=True)
        csv_path = os.path.join(csv_subdir, f"{base_name}_per_token.csv")
    else:
        csv_path = os.path.join(token_output_dir, f"{base_name}_per_token.csv")
    
    # ce_per_token shape: (B, K, T) -> (K, T) for single batch
    if ce_per_token.dim() == 3:
        ce_per_token = ce_per_token.squeeze(0)  # Remove batch dimension
    
    K, T = ce_per_token.shape
    
    # 创建DataFrame
    data = []
    for t in range(T):
        for k in range(K):
            token_index = t * K + k  # 全局token索引
            token_id = k  # codebook索引作为token_id
            nll = ce_per_token[k, t].item()
            data.append({
                'token_index': token_index,
                'token_id': token_id,
                'nll': nll
            })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Per-token losses保存到: {csv_path}")

def process_single_audio(audio_path, model, processor, device, reduction="mean", token_output_dir=None, relative_path=None):
    """处理单个音频文件"""
    try:
        # 读取音频
        if device == "cuda":
            audio_data, sample_rate = audio_read(audio_path)
            audio_data = audio_data.mean(dim=0, keepdim=True)
            if sample_rate != 32000:
                audio_data = torchaudio.functional.resample(audio_data, sample_rate, 32000)
            audio_data = audio_data.to(device)
            valid_token_length = min(audio_data.shape[-1] // 320, 1500)
        else:
            audio_data, sample_rate = librosa.load(audio_path, sr=32000, mono=True)
            audio_data = torch.tensor(audio_data).unsqueeze(0).to(device)
            valid_token_length = min(audio_data.shape[-1] // 320, 1500)
        
        with torch.no_grad():
            # 编码音频
            encoded = model.audio_encoder.encode(audio_data.unsqueeze(0))
            tokens = encoded.audio_codes.long()
            
        B, C, K, T = tokens.shape
        input_tokens = tokens[:, :, :, :-1].contiguous()
        target_tokens = tokens[:, :, :, 1:].contiguous()
        
        # 创建mask
        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                mask[b, c, :, :valid_token_length] = True
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                input_tokens_reshaped = input_tokens.view(B, C * K, -1)
                model.decoder = model.decoder.to(dtype=torch.float32)
                outputs = model.decoder(input_tokens_reshaped)
                output_logits = outputs.logits
            
            logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
            
            if reduction == "none":
                # 计算per-token loss
                ce_per_token = _compute_cross_entropy(
                    logits.view(B * C, K, -1, output_logits.size(-1)), 
                    target_tokens.view(B * C, K, -1), 
                    mask.view(B * C, K, -1),
                    reduction=reduction)
                
                if token_output_dir:
                    # 保存为CSV并返回平均loss
                    avg_loss = save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path)
                    return avg_loss if avg_loss is not None else ce_per_token.mean().item()
                else:
                    return ce_per_token.mean().item()
            else:
                # 计算平均loss
                ce, ce_per_codebook = _compute_cross_entropy(
                    logits.view(B * C, K, -1, output_logits.size(-1)), 
                    target_tokens.view(B * C, K, -1), 
                    mask.view(B * C, K, -1),
                    reduction=reduction)
                return ce.item()
            
    except Exception as e:
        print(f"处理文件 {audio_path} 时出错: {str(e)}")
        return None

def process_audio_directory(audio_dir, model, processor, device, output_file=None, reduction="mean", token_output_dir=None):
    """处理音频目录中的所有文件"""
    results = []
    error_files = []
    
    # 收集所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(os.path.join(root, file), audio_dir)
                audio_files.append((full_path, relative_path))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 处理每个文件
    for audio_path, relative_path in tqdm(audio_files, desc="处理音频文件"):
        loss = process_single_audio(audio_path, model, processor, device, reduction, token_output_dir, relative_path)
        
        if loss is not None:
            filename = os.path.basename(audio_path)
            results.append((filename, loss))
            print(f"{filename}: {loss:.4f}")
        else:
            error_files.append(audio_path)
    
    # 保存结果到文件
    if output_file and results:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("文件名\t损失值\n")
            for filename, loss in results:
                f.write(f"{filename}\t{loss:.6f}\n")
        print(f"结果已保存到: {output_file}")
    
    if error_files:
        print(f"\n处理失败的文件 ({len(error_files)}个):")
        for error_file in error_files:
            print(f"  - {error_file}")
    
    return results, error_files

def main():
    parser = argparse.ArgumentParser(description="计算音频文件的per token loss (使用musicgen-medium模型)")
    parser.add_argument("--audio_dir", type=str, required=True, help="音频文件目录")
    parser.add_argument("--output_file", type=str, default="results_medium.txt", help="输出结果文件路径")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU处理所有文件")
    parser.add_argument("--reduction", type=str, choices=["mean", "none"], default="mean", help="Loss reduction类型: mean或none (保存per-token losses)")
    parser.add_argument("--token_output_dir", type=str, default=None, help="当reduction=none时保存per-token losses的目录")
    
    args = parser.parse_args()

    if args.reduction == "none" and not args.token_output_dir:
        args.token_output_dir = "token_losses"
        print(f"使用默认token输出目录: {args.token_output_dir}")

    original_stdout = None
    if args.output_file:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")

    model, processor = load_model(device)

    assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"

    results, error_files = process_audio_directory(
        args.audio_dir, model, processor, device, args.output_file, args.reduction, args.token_output_dir
    )

    if not args.output_file:
        print("\n处理结果:")
        for filename, loss in results:
            print(f"{filename}: {loss:.4f}")

    if 'device' in locals() and device == 'cuda':
        torch.cuda.empty_cache()
    print("GPU内存已清理")

    if original_stdout:
        sys.stdout.close()
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()