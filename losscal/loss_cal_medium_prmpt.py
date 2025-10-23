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

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
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
            
            #inputs to model including text_prompt as inputs['input_ids'] and audio_tokens as input_tokens_reshaped
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
    for audio_file in tqdm(audio_files, desc="处理音频文件"):
        try:
            loss = process_single_audio(audio_file, model, processor, device, loss_type, text_prompt)
            results.append((audio_file, loss))
            # print(f"{audio_file}: {loss:.6f}")
        except Exception as e:
            tqdm.write(f"处理文件 {audio_file} 时出错: {e}")
            continue
    
    if output_file:
        with open(output_file, 'w') as f:
            for audio_file, loss in results:
                f.write(f"{audio_file}: {loss:.6f}\n")
        print(f"结果已保存到 {output_file}")
    
    return results

def process_audio_at_time_intervals(audio_path, model, processor, device, interval_percent=1, loss_type="cross_entropy", text_prompt=None):
    result = audio_read(audio_path)
    if len(result) == 2:
        audio, sr = result
    else:
        audio, sr = result[0], result[1]

    if audio.dim() > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    audio = audio.to(device)
    total_length = audio.shape[-1]
    duration_seconds = total_length / sr
    
    interval_seconds = duration_seconds * (interval_percent / 100)
    interval_samples = int(interval_seconds * sr)
    
    results = []
    
    for start_sample in range(0, total_length - interval_samples + 1, interval_samples):
        end_sample = start_sample + interval_samples
        audio_segment = audio[:, start_sample:end_sample]
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        try:
            temp_audio_path = f"/tmp/temp_audio_{start_sample}_{end_sample}.wav"
            torchaudio.save(temp_audio_path, audio_segment.cpu(), sr)
            
            loss = process_single_audio(temp_audio_path, model, processor, device, loss_type, text_prompt)
            results.append((start_time, end_time, loss))
            
            os.remove(temp_audio_path)
            
            print(f"时间段 {start_time:.2f}s - {end_time:.2f}s: {loss:.6f}")
        except Exception as e:
            print(f"处理时间段 {start_time:.2f}s - {end_time:.2f}s 时出错: {e}")
            continue
    
    return results

def save_time_interval_losses(audio_dir, output_file, interval_percent=1, loss_type="cross_entropy", device=None, text_prompt=None):
    if device is None:
        device = setup_device()
    
    model, processor = load_model(device)
    
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"在目录 {audio_dir} 中未找到音频文件")
        return
    
    with open(output_file, 'w') as f:
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            f.write(f"文件: {audio_file}\n")
            try:
                results = process_audio_at_time_intervals(
                    audio_file, model, processor, device, interval_percent, loss_type, text_prompt
                )
                for start_time, end_time, loss in results:
                    f.write(f"  {start_time:.2f}s - {end_time:.2f}s: {loss:.6f}\n")
            except Exception as e:
                f.write(f"  错误: {e}\n")
            f.write("\n")
    
    print(f"时间间隔损失结果已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="使用 MusicGen 计算音频损失 (Hugging Face 版本)")
    parser.add_argument("--audio_path", type=str, help="音频文件路径")
    parser.add_argument("--audio_dir", type=str, help="音频目录路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", 
                       choices=["cross_entropy", "mse"], help="损失类型")
    parser.add_argument("--text_prompt", type=str, help="文本提示")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--interval_percent", type=float, default=1, 
                       help="时间间隔百分比 (用于时间间隔处理)")
    parser.add_argument("--time_intervals", action="store_true", 
                       help="是否按时间间隔处理音频")
    
    args = parser.parse_args()
    
    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")
    
    model, processor = load_model(device)
    
    if args.time_intervals:
        if args.audio_dir:
            save_time_interval_losses(
                args.audio_dir, args.output_file, args.interval_percent, 
                args.loss_type, device, args.text_prompt
            )
        elif args.audio_path:
            results = process_audio_at_time_intervals(
                args.audio_path, model, processor, device, 
                args.interval_percent, args.loss_type, args.text_prompt
            )
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(f"文件: {args.audio_path}\n")
                    for start_time, end_time, loss in results:
                        f.write(f"  {start_time:.2f}s - {end_time:.2f}s: {loss:.6f}\n")
        else:
            print("错误: 使用时间间隔模式时必须指定 --audio_path 或 --audio_dir")
    elif args.audio_dir:
        process_audio_directory(
            args.audio_dir, model, processor, device, 
            args.output_file, args.loss_type, args.text_prompt
        )
    elif args.audio_path:
        loss = process_single_audio(
            args.audio_path, model, processor, device, 
            args.loss_type, args.text_prompt
        )
        print(f"损失: {loss:.6f}")
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"{args.audio_path}: {loss:.6f}\n")
    else:
        print("错误: 必须指定 --audio_path 或 --audio_dir")

if __name__ == "__main__":
    main()