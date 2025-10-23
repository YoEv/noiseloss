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
from collections import defaultdict

def setup_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
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
    将per-token loss保存为CSV文件，只保存平均值
    """
    if relative_path:
        # 使用相对路径信息，将路径分隔符替换为下划线避免文件系统问题
        rel_dir = os.path.dirname(relative_path)
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        if rel_dir:
            # 将路径分隔符替换为下划线，创建唯一的文件名
            filename = f"{rel_dir.replace(os.sep, '_')}_{base_filename}"
        else:
            filename = base_filename
    else:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    avg_csv_file = os.path.join(token_output_dir, f"{filename}_tokens_avg.csv")
    os.makedirs(token_output_dir, exist_ok=True)
    
    try:
        # 转换为numpy数组
        ce_np = ce_per_token.cpu().numpy()
        
        # 计算每个token位置的平均值
        avg_records = []
        
        if ce_np.ndim == 3:  # Shape: (B, K, T)
            B, K, T = ce_np.shape
            for b in range(B):
                for t in range(T):
                    # 计算该token位置上所有codebook的平均值
                    token_avg = float(ce_np[b, :, t].mean())
                    avg_records.append({
                        'token_position': t,
                        'avg_loss_value': token_avg
                    })
        elif ce_np.ndim == 2:  # Shape: (K, T)
            K, T = ce_np.shape
            for t in range(T):
                # 计算该token位置上所有codebook的平均值
                token_avg = float(ce_np[:, t].mean())
                avg_records.append({
                    'token_position': t,
                    'avg_loss_value': token_avg
                })
        
        # 保存平均值数据
        if avg_records:
            avg_df = pd.DataFrame(avg_records)
            avg_df.to_csv(avg_csv_file, index=False)
            print(f"保存了 {len(avg_records)} 条平均值记录到 {avg_csv_file}")
            
            # 计算并返回总体平均loss
            overall_mean = avg_df['avg_loss_value'].mean()
            print(f"总体平均loss: {overall_mean:.6f}")
            
            return overall_mean
        else:
            print(f"没有数据可保存: {filename}")
            return None
            
    except Exception as e:
        print(f"保存CSV时出错 {filename}: {str(e)}")
        return None

def process_single_audio(audio_path, model, processor, device, reduction="mean", token_output_dir=None, relative_path=None):
    try:
        if device == "cuda":
            audio, sr = audio_read(audio_path)
            audio = audio.mean(dim=0, keepdim=True)
            if sr != 32000:
                audio = torchaudio.functional.resample(audio, sr, 32000)
            audio = audio.to(device)
            valid_token_length = min(audio.shape[-1] // 320, 1500)
        else:
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
            valid_token_length = min(audio.shape[-1] // 320, 1500)

        with torch.no_grad():
            encoded = model.audio_encoder.encode(audio.unsqueeze(0)) 
            tokens = encoded.audio_codes.long()

        B, C, K, T = tokens.shape
        input_tokens = tokens[:, :, :, :-1].contiguous()
        target_tokens = tokens[:, :, :, 1:].contiguous()

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
        print(f"处理音频时出错 {os.path.basename(audio_path)}: {str(e)}")
        return None

def parse_filename(filename):
    """
    解析文件名，提取曲子名称和token长度
    例如: 1217199_mozart-rondo-alla-turca_short30_preview_cut14s_shuffle_tk10.wav
    返回: (曲子名称, token长度)
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    # 提取token长度
    token_length = None
    for part in parts:
        if part.startswith('tk'):
            try:
                token_length = int(part[2:])
                break
            except ValueError:
                continue
    
    # 提取曲子名称（去掉token相关的后缀）
    song_parts = []
    for part in parts:
        if part in ['shuffle'] or part.startswith('tk'):
            break
        song_parts.append(part)
    
    song_name = '_'.join(song_parts)
    
    return song_name, token_length

def process_shutter_shuffle_directory(audio_dir, model, processor, device, output_dir="shutter_shuffle_results"):
    """
    处理shutter_shuffle_order目录，按曲子和token长度组织结果
    """
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
    
    if not audio_files:
        raise ValueError(f"No WAV files found in {audio_dir}")

    print(f"Processing {len(audio_files)} audio files")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    token_output_dir = os.path.join(output_dir, "per_token")
    os.makedirs(token_output_dir, exist_ok=True)
    
    # 按曲子组织结果
    results_by_song = defaultdict(dict)
    all_results = []
    error_files = []
    
    # 处理每个文件
    for filename in tqdm(audio_files, desc="Processing"):
        audio_path = os.path.join(audio_dir, filename)
        song_name, token_length = parse_filename(filename)
        
        # 计算average loss
        avg_loss = process_single_audio(audio_path, model, processor, device, reduction="mean")
        
        # 计算per-token loss并保存CSV
        per_token_loss = process_single_audio(audio_path, model, processor, device, 
                                            reduction="none", token_output_dir=token_output_dir, 
                                            relative_path=filename)
        
        if avg_loss is not None and per_token_loss is not None:
            results_by_song[song_name][token_length] = {
                'filename': filename,
                'avg_loss': avg_loss,
                'per_token_avg': per_token_loss
            }
            all_results.append((filename, song_name, token_length, avg_loss, per_token_loss))
        else:
            error_files.append(filename)
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, "summary_results.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Shutter Shuffle Order Loss Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        for song_name in sorted(results_by_song.keys()):
            f.write(f"Song: {song_name}\n")
            f.write("-" * 30 + "\n")
            
            song_results = results_by_song[song_name]
            for token_length in sorted(song_results.keys()):
                result = song_results[token_length]
                f.write(f"  Token Length {token_length:3d}: Avg Loss = {result['avg_loss']:.6f}\n")
            f.write("\n")
    
    # 保存详细CSV结果
    csv_file = os.path.join(output_dir, "detailed_results.csv")
    df_data = []
    for filename, song_name, token_length, avg_loss, per_token_avg in all_results:
        df_data.append({
            'filename': filename,
            'song_name': song_name,
            'token_length': token_length,
            'avg_loss': avg_loss,
            'per_token_avg': per_token_avg
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    
    print(f"\n处理完成!")
    print(f"汇总结果保存到: {summary_file}")
    print(f"详细CSV结果保存到: {csv_file}")
    print(f"Per-token CSV文件保存到: {token_output_dir}")
    
    if error_files:
        error_file_path = os.path.join(output_dir, "processing_errors.txt")
        with open(error_file_path, 'w') as f:
            f.write(f"Failed files count: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n{len(error_files)} failed files saved to: {error_file_path}")
    
    return results_by_song, all_results, error_files

def main():
    parser = argparse.ArgumentParser(description="使用musicgen-small模型计算shutter_shuffle_order音频文件的loss值")
    parser.add_argument("--audio_dir", type=str, default="shutter_shuffle_random", help="音频文件目录")
    parser.add_argument("--output_dir", type=str, default="shutter_shuffle_random_results", help="输出结果目录")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU处理所有文件")
    
    args = parser.parse_args()

    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")

    model, processor = load_model(device)

    assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"

    results_by_song, all_results, error_files = process_shutter_shuffle_directory(
        args.audio_dir, model, processor, device, args.output_dir
    )

    if device == 'cuda':
        torch.cuda.empty_cache()
    print("GPU内存已清理")

if __name__ == "__main__":
    main()