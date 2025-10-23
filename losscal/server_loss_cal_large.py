import os
import sys
import torch
import torchaudio
import argparse
import gc
from tqdm import tqdm
import tempfile
import shutil
import soundfile as sf
import pandas as pd
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio import audio_read
from torch.nn import functional as F
from transformers import AutoProcessor, MusicgenForConditionalGeneration
#from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device, offload_folder=None, use_8bit=False, use_4bit=False):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    
    # Standard loading for large model
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large", use_safetensors=False)
    model = model.to(device)
    model.decoder = model.decoder.to(dtype=torch.float32)
    model.audio_encoder = model.audio_encoder.to(dtype=torch.float32)
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

def process_single_audio(audio_path, model, processor, device, reduction="mean", token_output_dir=None, relative_path=None, max_audio_length=None, chunk_size=None):
    if device == "cuda":
        audio, sr = audio_read(audio_path)
        audio = audio.mean(dim=0, keepdim=True)
        if sr != 32000:
            audio = torchaudio.functional.resample(audio, sr, 32000)
        audio = audio.to(device)
        valid_token_length = min(audio.shape[-1] // 320, 1500)
    else:
        import librosa
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        audio = torch.tensor(audio).unsqueeze(0).to(device)
        valid_token_length = min(audio.shape[-1] // 320, 1500)

    # Apply max audio length if specified
    if max_audio_length is not None and audio.shape[-1] > max_audio_length:
        audio = audio[..., :max_audio_length]
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

def process_audio_directory(audio_dir, model, processor, device, output_file=None, reduction="mean", token_output_dir=None):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), audio_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in {audio_dir}")

    print(f"Processing {len(audio_files)} audio files")
    results = []
    error_files = [] 

    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"Results will be saved to: {output_file}")

    for rel_filename in tqdm(audio_files, desc="Processing"):
        audio_path = os.path.join(audio_dir, rel_filename)
        try:
            loss = process_single_audio(audio_path, model, processor, device, reduction, token_output_dir, rel_filename)
            if loss is not None:
                results.append((rel_filename, loss))
                if out_file:
                    out_file.write(f"{rel_filename}: {loss:.8f}\n")
                    out_file.flush()
            else:
                error_files.append(rel_filename)
        except Exception as e:
            print(f"处理音频时出错 {os.path.basename(audio_path)}: {str(e)}")
            error_files.append(rel_filename)

    if out_file:
        out_file.close()

    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Failed files count: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n{len(error_files)} failed files saved to: {error_file_path}")

    return results, error_files

def process_phase5_datasets():
    """
    处理+Dataset/Phase5_1中的所有子目录
    """
    print("\n=== 处理 +Dataset/Phase5_1 中的数据集 ===")
    
    base_dir = "."
    input_base = os.path.join(base_dir, "+Dataset/Phase5_1")
    output_base = os.path.join(base_dir, "+Loss/Phase5_1")
    
    if not os.path.exists(input_base):
        print(f"错误: 输入目录 {input_base} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(output_base, exist_ok=True)
    
    # 获取所有子目录
    subdirs = []
    for item in os.listdir(input_base):
        item_path = os.path.join(input_base, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    
    subdirs.sort()
    print(f"找到 {len(subdirs)} 个子目录需要处理:")
    for subdir in subdirs:
        print(f"  - {subdir}")
    
    # 设置设备和加载模型
    device = setup_device()
    print(f"使用设备: {device}")
    model, processor = load_model(device)
    
    # 处理每个子目录
    for subdir in subdirs:
        print(f"\n=== 处理目录: {subdir} ===")
        
        input_dir = os.path.join(input_base, subdir)
        
        # 检查目录是否存在音频文件
        audio_count = 0
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.wav', '.mp3')):
                    audio_count += 1
        
        if audio_count == 0:
            print(f"警告: 目录 {input_dir} 中没有找到音频文件，跳过")
            continue
        
        print(f"找到 {audio_count} 个音频文件")
        
        # 创建输出目录名称: 原目录名 + _token_loss_large
        output_dir = os.path.join(output_base, f"{subdir}_token_loss_large")
        output_file = os.path.join(output_dir, "results.txt")
        per_token_dir = os.path.join(output_dir, "per_token")
        
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(per_token_dir, exist_ok=True)
        
        # 运行loss计算
        print("开始计算 per token loss...")
        
        results, error_files = process_audio_directory(
            input_dir, model, processor, device, 
            output_file, "none", per_token_dir
        )
        
        if results:
            print(f"✓ 完成: {output_dir}")
            print(f"  处理文件: {len(results)} 个")
            if error_files:
                print(f"  失败文件: {len(error_files)} 个")
        else:
            print(f"✗ 失败: {output_dir}")
        
        # 清理GPU内存
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        print("清理GPU内存...")

def process_original_datasets():
    """
    处理dataset_ori中的原始数据集
    """
    print("\n=== 处理 dataset_ori 中的原始数据集 ===")
    
    base_dir = "."
    dataset_ori_base = os.path.join(base_dir, "dataset_ori")
    output_base = os.path.join(base_dir, "+Loss/Phase5_1")
    
    original_datasets = {
        "noise_color_ori": os.path.join(dataset_ori_base, "noise_color_ori"),
        "asap_ori": os.path.join(dataset_ori_base, "asap_ori"),
        "ShutterStock_32k_ori": os.path.join(dataset_ori_base, "ShutterStock_32k_ori"),
        "Unconditional_ori": os.path.join(dataset_ori_base, "Unconditional_ori")
    }
    
    os.makedirs(output_base, exist_ok=True)
    
    device = setup_device()
    print(f"使用设备: {device}")
    model, processor = load_model(device)
    
    for dataset_name, dataset_path in original_datasets.items():
        print(f"\n=== 处理数据集: {dataset_name} ===")
        print(f"数据集路径: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"警告: 数据集目录不存在: {dataset_path}")
            continue
        
        audio_count = 0
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith(('.wav', '.mp3')):
                    audio_count += 1
        
        if audio_count == 0:
            print(f"警告: 目录 {dataset_path} 中没有找到音频文件，跳过")
            continue
        
        print(f"找到 {audio_count} 个音频文件")
        
        output_dir = os.path.join(output_base, f"{dataset_name}_large_token_loss_medium")
        per_token_dir = os.path.join(output_dir, "per_token")
        os.makedirs(per_token_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, "results.txt")
        log_file = os.path.join(output_dir, "results.txt.log")
        
        print(f"输出目录: {output_dir}")
        print("开始计算...")
        
        results, error_files = process_audio_directory(
            dataset_path, model, processor, device,
            results_file, "none", per_token_dir
        )
        
        if results:
            print(f"✓ 完成")
            print(f"  处理文件: {len(results)} 个")
            if error_files:
                print(f"  失败文件: {len(error_files)} 个")
            
            if os.path.exists(per_token_dir):
                csv_count = len([f for f in os.listdir(per_token_dir) if f.endswith('.csv')])
                print(f"  生成CSV文件: {csv_count} 个")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    result_lines = len(f.readlines())
                print(f"  结果文件行数: {result_lines}")
        else:
            print(f"✗ 失败")
        
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        print("清理GPU内存...")

def main():
    """
    主函数：整合处理Phase5_1数据集和原始数据集
    """
    print("=== 开始批量计算Phase5_1数据的per token loss (Large模型) ===")
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    # 处理+Dataset/Phase5_1中的数据集
    process_phase5_datasets()
    
    # 处理dataset_ori中的原始数据集
    process_original_datasets()
    
    # 计算总耗时
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n=== 所有处理完成! ===")
    print(f"总耗时: {total_duration:.0f}秒 ({total_duration/60:.1f}分钟)")
    print("输出目录: +Loss/Phase5_1")
    
    # 生成处理摘要
    output_base = "+Loss/Phase5_1"
    if os.path.exists(output_base):
        print("\n=== 输出目录结构预览 ===")
        subdirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
        subdirs.sort()
        print(f"生成的输出目录数: {len(subdirs)}")
        for subdir in subdirs[:10]:  # 只显示前10个
            print(f"  - {subdir}")
        if len(subdirs) > 10:
            print(f"  ... 还有 {len(subdirs) - 10} 个目录")
    
    print("\n脚本执行完成！")

if __name__ == "__main__":
    main()

