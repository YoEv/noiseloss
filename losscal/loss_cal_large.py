import os
import sys
import torch
import torchaudio
import argparse
import gc
from tqdm import tqdm
from datasets import load_dataset
import tempfile
import shutil
import soundfile as sf
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
    
    # Use model offloading if specified
    if offload_folder and device == "cuda":
        print(f"使用模型分片加载到: {offload_folder}")
        # Create directory if it doesn't exist
        os.makedirs(offload_folder, exist_ok=True)
        
        # Load with offloading to disk
        with init_empty_weights():
            model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
        
        model = load_checkpoint_and_dispatch(
            model, 
            "facebook/musicgen-large", 
            device_map="auto",
            offload_folder=offload_folder,
            offload_state_dict=True
        )
    elif use_8bit and device == "cuda":
        print("使用8位量化加载模型")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-large", 
            device_map="auto", 
            load_in_8bit=True
        )
    elif use_4bit and device == "cuda":
        print("使用4位量化加载模型")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-large", 
            device_map="auto", 
            load_in_4bit=True
        )
    else:
        # Standard loading
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
        model = model.to(device)
        model.decoder = model.decoder.to(dtype=torch.float32)
        model.audio_encoder = model.audio_encoder.to(dtype=torch.float32)
    
    compression_ratio = model.config.audio_encoder.frame_rate / model.config.audio_encoder.sampling_rate
    print(f"压缩比率: {compression_ratio:.4f} (每音频秒生成{model.config.audio_encoder.frame_rate}个tokens)")

    return model, processor

def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ):

    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
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
    # 计算各码本平均交叉熵
    ce = ce / K
    return ce, ce_per_codebook

# 在process_single_audio函数中启用混合精度
# 在函数参数中添加batch_size和max_audio_length参数
def process_single_audio(audio_path, model, processor, device, loss_type="cross_entropy", max_audio_length=None, chunk_size=None):
    try:
        audio, sr = audio_read(audio_path)

        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if max_audio_length is not None and audio.shape[-1] > max_audio_length:
            audio = audio[..., :max_audio_length]
            print(f"音频已截断至 {max_audio_length/sr:.2f} 秒")

        audio = audio.to(device)
        original_length = audio.shape[-1]

        # Calculate frame rate from model config
        frame_rate = model.config.audio_encoder.frame_rate
        valid_token_length = int((original_length / sr) * frame_rate) - 1

        # Process in chunks if specified
        if chunk_size and valid_token_length > chunk_size and device == "cuda":
            print(f"将音频分成多个块处理，每块 {chunk_size} 个token")
            total_loss = 0
            num_chunks = 0
            
            for start_idx in range(0, valid_token_length, chunk_size):
                end_idx = min(start_idx + chunk_size, valid_token_length)
                chunk_length = end_idx - start_idx
                
                # Calculate audio sample indices
                audio_start = int(start_idx * sr / frame_rate)
                audio_end = int((end_idx + 1) * sr / frame_rate)  # +1 for the target token
                audio_chunk = audio[..., audio_start:audio_end]
                
                # Process chunk
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=device=="cuda"):
                        encoded = model.audio_encoder.encode(audio_chunk.unsqueeze(0))
                        tokens = encoded.audio_codes.long()
                        
                        B, C, K, T = tokens.shape
                        input_tokens = tokens[:, :, :, :-1].contiguous()
                        target_tokens = tokens[:, :, :, 1:].contiguous()
                        
                        # Create mask for valid tokens
                        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
                        for b in range(B):
                            for c in range(C):
                                mask[b, c, :, :chunk_length] = True
                        
                        # Calculate loss
                        input_tokens_reshaped = input_tokens.view(B, C * K, -1)
                        outputs = model.decoder(input_tokens_reshaped)
                        output_logits = outputs.logits
                        
                        if loss_type == "cross_entropy":
                            logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
                            ce, _ = _compute_cross_entropy(
                                logits.view(B * C, K, -1, output_logits.size(-1)), 
                                target_tokens.view(B * C, K, -1), 
                                mask.view(B * C, K, -1))
                            total_loss += ce.item() * chunk_length
                        else:  # MSE loss
                            sample_logits = output_logits.view(B, C, K, -1, output_logits.size(-1))[0, 0, :, :chunk_length, :].reshape(-1, output_logits.size(-1))
                            sample_target = target_tokens[0, 0, :, :chunk_length].reshape(-1)
                            target_one_hot = torch.zeros_like(sample_logits)
                            target_one_hot.scatter_(1, sample_target.unsqueeze(1), 1.0)
                            loss = torch.nn.functional.mse_loss(
                                torch.softmax(sample_logits, dim=1),
                                target_one_hot,
                                reduction='mean'
                            )
                            total_loss += loss.item() * chunk_length
                
                num_chunks += 1
                # Clear cache after each chunk
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Average loss by total tokens
            return total_loss / valid_token_length
        else:
            # Original processing for smaller files
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=device=="cuda"):
                    encoded = model.audio_encoder.encode(audio.unsqueeze(0))
                    tokens = encoded.audio_codes.long()
                    
                    B, C, K, T = tokens.shape
                    input_tokens = tokens[:, :, :, :-1].contiguous()
                    target_tokens = tokens[:, :, :, 1:].contiguous()
                    
                    mask = torch.zeros_like(target_tokens, dtype=torch.bool)
                    for b in range(B):
                        for c in range(C):
                            mask[b, c, :, :valid_token_length] = True
                    
                    input_tokens_reshaped = input_tokens.view(B, C * K, -1)
                    outputs = model.decoder(input_tokens_reshaped)
                    output_logits = outputs.logits
                
                if loss_type == "cross_entropy":
                    logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
                    ce, _ = _compute_cross_entropy(
                        logits.view(B * C, K, -1, output_logits.size(-1)), 
                        target_tokens.view(B * C, K, -1), 
                        mask.view(B * C, K, -1))
                    return ce.item()
                else:  # MSE loss
                    sample_logits = output_logits.view(B, C, K, -1, output_logits.size(-1))[0, 0, :, :valid_token_length, :].reshape(-1, output_logits.size(-1))
                    sample_target = target_tokens[0, 0, :, :valid_token_length].reshape(-1)
                    target_one_hot = torch.zeros_like(sample_logits)
                    target_one_hot.scatter_(1, sample_target.unsqueeze(1), 1.0)
                    loss = torch.nn.functional.mse_loss(
                        torch.softmax(sample_logits, dim=1),
                        target_one_hot,
                        reduction='mean'
                    )
                    return loss.item()

    except Exception as e:
        print(f"处理 {os.path.basename(audio_path)} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 修改process_audio_directory函数以分批处理文件
def process_audio_directory(audio_dir, model, processor, device, output_file=None, loss_type="cross_entropy", batch_size=1, max_audio_length=None, chunk_size=None):
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"目录 {audio_dir} 中没有WAV/MP3文件")

    print(f"\n开始处理 {len(audio_files)} 个音频 (损失类型: {loss_type})")
    results = []
    error_files = []

    # 打开输出文件
    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"结果将保存到: {output_file}")

    # 分批处理文件
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}")
        
        for filename in tqdm(batch_files, desc="处理进度"):
            audio_path = os.path.join(audio_dir, filename)
            loss = process_single_audio(audio_path, model, processor, device, loss_type, max_audio_length, chunk_size)
            
            if loss is not None:
                results.append((filename, loss))
                if out_file:
                    out_file.write(f"{filename}: {loss:.8f}\n")
                    out_file.flush()
            else:
                error_files.append(filename)
        
        # 每批处理完后清理缓存
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            print("批次处理完成，已清理显存")

    # 关闭输出文件
    if out_file:
        out_file.close()

    # 处理错误文件的代码保持不变
    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"处理失败的文件数量: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n处理失败的 {len(error_files)} 个文件已保存到: {error_file_path}")

    return results, error_files

def get_missing_files(audio_dir, results_file):
    # Get all audio files in directory
    all_audio_files = set(f for f in os.listdir(audio_dir)
                         if f.lower().endswith(('.wav', '.mp3')))

    # Get processed files from results file
    processed_files = set()
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename = line.split(':', 1)[0].strip()
                    processed_files.add(filename)
    except FileNotFoundError:
        print(f"结果文件 {results_file} 不存在，将处理所有文件")

    # Return files that are in all_audio_files but not in processed_files
    missing_files = all_audio_files - processed_files
    return list(missing_files)

def process_missing_files(audio_dir, results_file, loss_type="cross_entropy"):
    """处理缺失的文件（使用CPU）"""
    missing_files = get_missing_files(audio_dir, results_file)

    if not missing_files:
        print("没有缺失的文件需要处理")
        return

    print(f"发现 {len(missing_files)} 个未处理的文件，将使用CPU处理 (损失类型: {loss_type})")

    # Print all missing filenames before processing
    print("\n未处理的文件列表:")
    for filename in missing_files:
        print(f"{filename}")
    print("\n开始处理这些文件...\n")

    # Load model on CPU
    device = "cpu"
    print(f"Using device: {device}")
    model, processor = load_model(device)

    # Open results file in append mode
    with open(results_file, 'a') as out_file:
        for filename in tqdm(missing_files, desc="CPU处理进度"):
            audio_path = os.path.join(audio_dir, filename)
            loss = process_single_audio(audio_path, model, processor, device, loss_type)
            if loss is not None:
                out_file.write(f"{filename}: {loss:.4f}\n")
                out_file.flush()  # Ensure data is written immediately

def process_huggingface_dataset(dataset_name, model, processor, device, output_file=None, loss_type="cross_entropy", batch_size=1, max_audio_length=None, chunk_size=None):
    """处理Hugging Face数据集"""
    print(f"正在加载数据集: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # 假设数据集有'train'分割，如果不同请调整
    if 'train' in dataset:
        data_split = dataset['train']
    else:
        # 如果没有train分割，使用第一个可用的分割
        data_split = dataset[list(dataset.keys())[0]]
    
    print(f"数据集加载完成，共有 {len(data_split)} 个样本")
    
    results = []
    error_files = []
    
    # 打开输出文件
    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"结果将保存到: {output_file}")
    
    # 创建临时目录来存储音频文件
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 分批处理数据集
        for i in range(0, len(data_split), batch_size):
            batch_data = data_split[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(data_split) + batch_size - 1)//batch_size}")
            
            for idx, sample in enumerate(tqdm(batch_data, desc="处理进度")):
                # 获取音频数据和文件名
                if 'audio' in sample:
                    audio_data = sample['audio']
                    # 生成文件名，不添加_ori后缀
                    if 'path' in sample:
                        original_filename = os.path.basename(sample['path'])
                        filename_without_ext = os.path.splitext(original_filename)[0]
                        filename = f"{filename_without_ext}.wav"
                    else:
                        filename = f"sample_{i+idx}.wav"
                    
                    # 保存音频到临时文件
                    temp_audio_path = os.path.join(temp_dir, filename)
                    
                    # 如果音频数据是字典格式（包含array和sampling_rate）
                    if isinstance(audio_data, dict) and 'array' in audio_data:
                        sf.write(temp_audio_path, audio_data['array'], audio_data['sampling_rate'])
                    else:
                        # 如果是其他格式，尝试直接保存
                        print(f"警告: 音频数据格式未知，跳过样本 {filename}")
                        continue
                    
                    # 处理音频文件
                    loss = process_single_audio(temp_audio_path, model, processor, device, loss_type, max_audio_length, chunk_size)
                    
                    if loss is not None:
                        results.append((filename, loss))
                        if out_file:
                            out_file.write(f"{filename}: {loss:.8f}\n")
                            out_file.flush()
                    else:
                        error_files.append(filename)
                    
                    # 删除临时文件
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            
            # 每批处理完后清理缓存
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                print("批次处理完成，已清理显存")
    
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        # 关闭输出文件
        if out_file:
            out_file.close()
    
    # 处理错误文件
    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"处理失败的文件数量: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n处理失败的 {len(error_files)} 个文件已保存到: {error_file_path}")
    
    return results, error_files

def main():
    parser = argparse.ArgumentParser(description="计算音频文件的loss值 (使用musicgen-large模型)")
    parser.add_argument("--audio_dir", type=str, default="None",
                        help="音频文件目录")
    parser.add_argument("--dataset_name", type=str, default="Dev4IC/garbage-dataset",
                        help="Hugging Face数据集名称")
    parser.add_argument("--output_file", type=str, default="loss_large.txt",
                        help="输出结果文件路径")
    parser.add_argument("--process_missing", action="store_true",
                        help="处理缺失的文件（使用CPU）")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU处理所有文件")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "mse"], default="cross_entropy",
                        help="损失函数类型: cross_entropy或mse")
    parser.add_argument("--max_audio_length", type=int, default=None,
                        help="处理的最大音频长度（采样点数），用于减少内存使用")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="批处理大小，默认为1")
    parser.add_argument("--offload_folder", type=str, default=None,
                        help="模型分片存储文件夹，用于减少GPU内存使用")
    parser.add_argument("--use_8bit", action="store_true",
                        help="使用8位量化加载模型，减少内存使用")
    parser.add_argument("--use_4bit", action="store_true",
                        help="使用4位量化加载模型，进一步减少内存使用")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="处理长音频时的分块大小（token数），用于减少内存使用")

    args = parser.parse_args()

    # Redirect stdout to file if output_file is specified and not processing missing files
    original_stdout = None
    if args.output_file and not args.process_missing:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        if args.process_missing and args.output_file:
            # Process missing files on CPU
            process_missing_files(args.audio_dir, args.output_file, args.loss_type)
        elif args.dataset_name:
            # Process Hugging Face dataset
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")

            # 先加载模型
            model, processor = load_model(device, args.offload_folder, args.use_8bit, args.use_4bit)
            
            results, error_files = process_huggingface_dataset(
                args.dataset_name, model, processor, device, args.output_file, args.loss_type,
                batch_size=args.batch_size, max_audio_length=args.max_audio_length, chunk_size=args.chunk_size
            )
            
            if not args.output_file:
                print("\n处理结果:")
                for filename, loss in results:
                    print(f"{filename}: {loss:.4f}")
        else:
            # Normal directory processing
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")

            model, processor = load_model(device, args.offload_folder, args.use_8bit, args.use_4bit)

            assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"

            results, error_files = process_audio_directory(
                args.audio_dir, model, processor, device, args.output_file, args.loss_type,
                batch_size=args.batch_size, max_audio_length=args.max_audio_length, chunk_size=args.chunk_size
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

def process_huggingface_dataset(dataset_name, model, processor, device, output_file=None, loss_type="cross_entropy", batch_size=1, max_audio_length=None, chunk_size=None):
    """处理Hugging Face数据集"""
    print(f"正在加载数据集: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # 假设数据集有'train'分割，如果不同请调整
    if 'train' in dataset:
        data_split = dataset['train']
    else:
        # 如果没有train分割，使用第一个可用的分割
        data_split = dataset[list(dataset.keys())[0]]
    
    print(f"数据集加载完成，共有 {len(data_split)} 个样本")
    
    results = []
    error_files = []
    
    # 打开输出文件
    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"结果将保存到: {output_file}")
    
    # 创建临时目录来存储音频文件
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 分批处理数据集
        for i in range(0, len(data_split), batch_size):
            batch_data = data_split[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(data_split) + batch_size - 1)//batch_size}")
            
            for idx, sample in enumerate(tqdm(batch_data, desc="处理进度")):
                # 获取音频数据和文件名
                if 'audio' in sample:
                    audio_data = sample['audio']
                    # 生成文件名，不添加_ori后缀
                    if 'path' in sample:
                        original_filename = os.path.basename(sample['path'])
                        filename_without_ext = os.path.splitext(original_filename)[0]
                        filename = f"{filename_without_ext}.wav"
                    else:
                        filename = f"sample_{i+idx}.wav"
                    
                    # 保存音频到临时文件
                    temp_audio_path = os.path.join(temp_dir, filename)
                    
                    # 如果音频数据是字典格式（包含array和sampling_rate）
                    if isinstance(audio_data, dict) and 'array' in audio_data:
                        sf.write(temp_audio_path, audio_data['array'], audio_data['sampling_rate'])
                    else:
                        # 如果是其他格式，尝试直接保存
                        print(f"警告: 音频数据格式未知，跳过样本 {filename}")
                        continue
                    
                    # 处理音频文件
                    loss = process_single_audio(temp_audio_path, model, processor, device, loss_type, max_audio_length, chunk_size)
                    
                    if loss is not None:
                        results.append((filename, loss))
                        if out_file:
                            out_file.write(f"{filename}: {loss:.8f}\n")
                            out_file.flush()
                    else:
                        error_files.append(filename)
                    
                    # 删除临时文件
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            
            # 每批处理完后清理缓存
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                print("批次处理完成，已清理显存")
    
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        # 关闭输出文件
        if out_file:
            out_file.close()
    
    # 处理错误文件
    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"处理失败的文件数量: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n处理失败的 {len(error_files)} 个文件已保存到: {error_file_path}")
    
    return results, error_files