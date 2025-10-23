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

"""
# Unconditional generation (like original small version)
python loss_cal_small_prmpt.py --audio_dir /path/to/audio --output_file results.txt

# Text-conditioned generation
python loss_cal_small_prmpt.py --audio_dir /path/to/audio --output_file results.txt --text_prompt "classical piano music"

# Time interval analysis with text prompt
python loss_cal_small_prmpt.py --audio_dir /path/to/audio --time_intervals --text_prompt "upbeat electronic music"
"""

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model = model.to(device)
    model.eval()
    
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
    ce = ce / K
    return ce, ce_per_codebook

def process_single_audio(audio_path, model, processor, device, loss_type="cross_entropy", text_prompt=None):
    try:
        result = audio_read(audio_path)
        if len(result) == 2:
            audio, sr = result
        else:
            audio, sr = result[0], result[1]

        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to(device)
        original_length = audio.shape[-1]

        frame_rate = model.config.audio_encoder.frame_rate
        valid_token_length = int((original_length / sr) * frame_rate) - 1

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
            input_tokens_reshaped = input_tokens.squeeze(1)  # 移除C维度，变为 [B, K, T]
            
            if text_prompt:
                inputs = processor(
                    text=[text_prompt],
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Text-conditioned generation
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=input_tokens_reshaped,
                    return_dict=True
                )
            else:
                # Unconditional generation
                outputs = model(
                    decoder_input_ids=input_tokens_reshaped,
                    return_dict=True
                )
            
            output_logits = outputs.logits

        if loss_type == "cross_entropy":
            if output_logits.dim() == 3:  # [K, T, vocab_size]
                output_logits = output_logits.unsqueeze(0)  # [1, K, T, vocab_size]
            
            target_tokens_reshaped = target_tokens.squeeze(1)  # [B, K, T]
            mask_reshaped = mask.squeeze(1)  # [B, K, T]
            
            assert output_logits.shape[:-1] == target_tokens_reshaped.shape, f"形状不匹配: {output_logits.shape[:-1]} vs {target_tokens_reshaped.shape}"
            
            ce, ce_per_codebook = _compute_cross_entropy(
                output_logits, 
                target_tokens_reshaped, 
                mask_reshaped
            )
            return ce.item()
        else:  # MSE loss
            if output_logits.dim() == 3:  # [K, T, vocab_size]
                sample_logits = output_logits[:, :valid_token_length, :].reshape(-1, output_logits.size(-1))
            else:  # [B, K, T, vocab_size]
                sample_logits = output_logits[0, :, :valid_token_length, :].reshape(-1, output_logits.size(-1))
            
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
        return None

def process_audio_directory(audio_dir, model, processor, device, output_file=None, loss_type="cross_entropy", text_prompt=None):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"在目录 {audio_dir} 中未找到音频文件")
        return
    
    results = []
    error_files = []
    
    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"结果将保存到: {output_file}")
    
    for audio_file in tqdm(audio_files, desc="处理音频文件"):
        try:
            loss = process_single_audio(audio_file, model, processor, device, loss_type, text_prompt)
            if loss is not None:
                results.append((audio_file, loss))
                if out_file:
                    out_file.write(f"{os.path.basename(audio_file)}: {loss:.8f}\n")
                    out_file.flush()
            else:
                error_files.append(audio_file)
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错: {str(e)}")
            error_files.append(audio_file)
    
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
    all_audio_files = set()
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                all_audio_files.add(file)

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

def process_missing_files(audio_dir, results_file, loss_type="cross_entropy", text_prompt=None):
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
    model, processor = load_model(device)

    with open(results_file, 'a') as out_file:
        for filename in tqdm(missing_files, desc="CPU处理进度"):
            audio_path = os.path.join(audio_dir, filename)
            loss = process_single_audio(audio_path, model, processor, device, loss_type, text_prompt)
            if loss is not None:
                out_file.write(f"{filename}: {loss:.4f}\n")
                out_file.flush()

def process_audio_at_time_intervals(audio_path, model, processor, device, interval_percent=1, loss_type="cross_entropy", text_prompt=None):
    """
    Process audio and calculate loss at different time intervals (e.g., 0%-1%, 1%-2%, etc.)
    Returns a list of (start_percent, end_percent, loss) tuples
    """
    audio, sr = librosa.load(audio_path, sr=32000)
    
    duration = len(audio) / sr
    
    results = []
    
    num_intervals = int(100 / interval_percent)
    
    for i in range(num_intervals):
        start_percent = i * interval_percent
        end_percent = (i + 1) * interval_percent
        
        start_time = (start_percent / 100.0) * duration
        end_time = (end_percent / 100.0) * duration
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]
        
        if len(audio_segment) < sr * 0.1:  # Less than 0.1 seconds
            continue
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(audio_segment).unsqueeze(0).to(device)
        
        with torch.no_grad():
            encoded = model.audio_encoder.encode(audio_tensor.unsqueeze(0))
            tokens = encoded.audio_codes.long()
        
        B, C, K, T = tokens.shape
        input_tokens = tokens[:, :, :, :-1].contiguous()
        target_tokens = tokens[:, :, :, 1:].contiguous()
        
        valid_token_length = T - 1
        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                mask[b, c, :, :valid_token_length] = True
        
        with torch.no_grad():
            input_tokens_reshaped = input_tokens.squeeze(1)
            
            if text_prompt:
                inputs = processor(
                    text=[text_prompt],
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=input_tokens_reshaped,
                    return_dict=True
                )
            else:
                outputs = model(
                    decoder_input_ids=input_tokens_reshaped,
                    return_dict=True
                )
            
            output_logits = outputs.logits
        
        if loss_type == "cross_entropy":
            if output_logits.dim() == 3:
                output_logits = output_logits.unsqueeze(0)
            
            target_tokens_reshaped = target_tokens.squeeze(1)
            mask_reshaped = mask.squeeze(1)
            
            ce, ce_per_codebook = _compute_cross_entropy(
                output_logits, 
                target_tokens_reshaped, 
                mask_reshaped
            )
            loss_value = ce.item()
        else:  # MSE loss
            if output_logits.dim() == 3:
                sample_logits = output_logits[:, :valid_token_length, :].reshape(-1, output_logits.size(-1))
            else:
                sample_logits = output_logits[0, :, :valid_token_length, :].reshape(-1, output_logits.size(-1))
            
            sample_target = target_tokens[0, 0, :, :valid_token_length].reshape(-1)
            target_one_hot = torch.zeros_like(sample_logits)
            target_one_hot.scatter_(1, sample_target.unsqueeze(1), 1.0)
            loss = torch.nn.functional.mse_loss(
                torch.softmax(sample_logits, dim=1),
                target_one_hot,
                reduction='mean'
            )
            loss_value = loss.item()
        
        results.append((start_percent, end_percent, loss_value))
    
    return results

def save_time_interval_losses(audio_dir, output_file, interval_percent=1, loss_type="cross_entropy", device=None, text_prompt=None):
    """
    Process all audio files and save loss values at time intervals
    """
    if device is None:
        device = setup_device(False)
    
    print(f"Using device: {device}")
    model, processor = load_model(device)
    
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        raise ValueError(f"目录 {audio_dir} 中没有音频文件")
    
    print(f"\n开始处理 {len(audio_files)} 个音频文件的时间区间损失 (区间: {interval_percent}%, 损失类型: {loss_type})")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("filename,start_percent,end_percent,loss\n")
        
        for audio_file in tqdm(audio_files, desc="处理进度"):
            interval_losses = process_audio_at_time_intervals(audio_file, model, processor, device, interval_percent, loss_type, text_prompt)
            
            for start_percent, end_percent, loss_value in interval_losses:
                f.write(f"{os.path.basename(audio_file)},{start_percent},{end_percent},{loss_value:.8f}\n")
            
            f.flush()
    
    print(f"\n时间区间损失结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="计算音频文件的loss值 (使用musicgen-small模型，支持文本条件)")
    parser.add_argument("--audio_dir", type=str, default="pitch_out_wav",
                        help="音频文件目录")
    parser.add_argument("--output_file", type=str, default="results_small_prompt.txt",
                        help="输出结果文件路径")
    parser.add_argument("--process_missing", action="store_true",
                        help="处理缺失的文件（使用CPU）")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU处理所有文件")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "mse"], default="cross_entropy",
                        help="损失函数类型: cross_entropy或mse")
    parser.add_argument("--time_intervals", action="store_true",
                        help="计算时间区间损失 (如0%%-1%%, 1%%-2%%等)")
    parser.add_argument("--interval_percent", type=int, default=1,
                        help="时间区间百分比 (默认1%%)")
    parser.add_argument("--interval_output_file", type=str, default="time_interval_losses_small_prompt.csv",
                        help="时间区间损失输出文件路径")
    parser.add_argument("--text_prompt", type=str, default=None,
                        help="文本提示词，用于条件生成 (例如: 'classical piano music')")
    
    args = parser.parse_args()

    # Redirect stdout to file if output_file is specified and not processing missing files or time percentages
    original_stdout = None
    if args.output_file and not args.process_missing and not args.time_intervals:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        if args.time_intervals:
            device = setup_device(args.force_cpu)
            save_time_interval_losses(args.audio_dir, args.interval_output_file, args.interval_percent, args.loss_type, device, args.text_prompt)
        elif args.process_missing and args.output_file:
            process_missing_files(args.audio_dir, args.output_file, args.loss_type, args.text_prompt)
        else:
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")
            if args.text_prompt:
                print(f"Text prompt: {args.text_prompt}")

            model, processor = load_model(device)

            assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"

            results, error_files = process_audio_directory(
                args.audio_dir, model, processor, device, args.output_file, args.loss_type, args.text_prompt
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