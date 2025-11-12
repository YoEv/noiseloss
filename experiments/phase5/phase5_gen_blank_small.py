import torch
import torchaudio
import os
import argparse
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.models import MusicGen
import librosa
import json

def setup_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"

def load_model(device):
    try:
        # Prefer passing device at load time (supported by newer interfaces)
        model = MusicGen.get_pretrained('facebook/musicgen-small', device=str(device))
    except TypeError:
        # Backward compatibility: load first, then set device
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        if hasattr(model, 'set_device'):
            model.set_device(device)
        else:
            print("Warning: Current MusicGen version does not support set_device/passing device; running on CPU")
    # Note: Do not call model.to(device); this object has no .to method
    return model

def generate_missing_segment(original_audio_path, output_path, model, device,
                           context_duration=5.0, generation_length_tokens=50,
                           temperature=0.7, top_k=250, top_p=0.0):
    """
    Generate a missing audio segment
    
    Args:
        original_audio_path: path to the original full audio
        output_path: output path for the generated segment
        model: MusicGen model
        device: device
        context_duration: context duration (seconds)
        generation_length_tokens: number of tokens to generate
        temperature: generation temperature
        top_k: top-k sampling
        top_p: top-p sampling
    """
    try:
        # Load original audio
        print(f"Loading original audio: {original_audio_path}")
        audio, sr = audio_read(original_audio_path)
        
        # Ensure mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample to 32kHz
        if sr != 32000:
            audio = torchaudio.functional.resample(audio, sr, 32000)
            sr = 32000
        
        # Take the first context_duration seconds as conditioning
        context_samples = int(context_duration * sr)
        if audio.shape[-1] < context_samples:
            print(f"Warning: audio shorter than {context_duration}s, using the whole audio as context")
            context_audio = audio
        else:
            context_audio = audio[:, :context_samples]
        
        print(f"Context audio length: {context_audio.shape[-1]/sr:.2f}s")
        
        # Compute generation duration
        tokens_per_sec = 32000 // 320  # ~100 tokens/sec
        generation_duration = generation_length_tokens / tokens_per_sec
        total_duration = context_duration + generation_duration
        
        print(f"Will generate {generation_length_tokens} tokens ({generation_duration:.2f}s)")
        print(f"Total output duration: {total_duration:.2f}s")
        
        # 设置生成参数
        model.set_generation_params(
            duration=total_duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_sampling=True
        )
        
        # Generate audio
        print("Start generation...")
        with torch.no_grad():
            generated = model.generate_continuation(
                context_audio.unsqueeze(0).to(device),
                prompt_sample_rate=sr,
                progress=True
            )
        
        # Extract generated part (remove context)
        generated_audio = generated.squeeze(0).cpu()
        context_samples_out = int(context_duration * sr)
        
        if generated_audio.shape[-1] > context_samples_out:
            generated_segment = generated_audio[:, context_samples_out:]
        else:
            print("Warning: generated audio too short, using the entire result")
            generated_segment = generated_audio
        
        print(f"Generated segment length: {generated_segment.shape[-1]/sr:.2f}s")
        
        # 保存生成的片段
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio_write(output_path.replace('.wav', ''), generated_segment, sr, format='wav')
        
        print(f"Generated audio segment saved to: {output_path}")
        
        return {
            'original_audio_path': original_audio_path,
            'generated_segment_path': output_path,
            'context_duration': context_duration,
            'generation_duration': generation_segment.shape[-1] / sr,
            'generation_length_tokens': generation_length_tokens,
            'actual_generated_samples': generated_segment.shape[-1]
        }
        
    except Exception as e:
        print(f"Error generating audio segment {original_audio_path}: {str(e)}")
        return None

def fill_audio_gap(original_audio_path, generated_segment_path, output_path,
                  gap_start_sec=5.0, gap_length_tokens=50):
    """
    Insert the generated segment into the original audio gap
    
    Args:
        original_audio_path: path to the original full audio
        generated_segment_path: path to the generated segment
        output_path: output path for the filled audio
        gap_start_sec: gap start position (seconds)
        gap_length_tokens: gap length (tokens)
    """
    # 加载原始音频和生成片段
    original_audio, sr1 = audio_read(original_audio_path)
    generated_segment, sr2 = audio_read(generated_segment_path)
    
    # 确保采样率一致
    if sr1 != sr2:
        generated_segment = torchaudio.functional.resample(generated_segment, sr2, sr1)
    
    # 确保单声道
    if original_audio.shape[0] > 1:
        original_audio = original_audio.mean(dim=0, keepdim=True)
    if generated_segment.shape[0] > 1:
        generated_segment = generated_segment.mean(dim=0, keepdim=True)
    
    # 计算填充位置
    gap_start_samples = int(gap_start_sec * sr1)
    tokens_per_sec = sr1 // 320
    gap_length_samples = int((gap_length_tokens / tokens_per_sec) * sr1)
    gap_end_samples = gap_start_samples + gap_length_samples
    
    print(f"原始音频长度: {original_audio.shape[-1]/sr1:.2f}秒")
    print(f"生成片段长度: {generated_segment.shape[-1]/sr1:.2f}秒")
    print(f"填充位置: {gap_start_sec:.2f}秒 - {gap_end_samples/sr1:.2f}秒")
    
    # 调整生成片段长度以匹配空白长度
    if generated_segment.shape[-1] > gap_length_samples:
        generated_segment = generated_segment[:, :gap_length_samples]
    elif generated_segment.shape[-1] < gap_length_samples:
        # 如果生成片段较短，进行填充或重复
        padding_needed = gap_length_samples - generated_segment.shape[-1]
        padding = torch.zeros(generated_segment.shape[0], padding_needed)
        generated_segment = torch.cat([generated_segment, padding], dim=-1)
    
    # 构建填充后的音频
    before_gap = original_audio[:, :gap_start_samples]
    after_gap = original_audio[:, gap_end_samples:]
    
    filled_audio = torch.cat([before_gap, generated_segment, after_gap], dim=-1)
    
    print(f"填充后音频长度: {filled_audio.shape[-1]/sr1:.2f}秒")
    
    # 保存填充后的音频
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_write(output_path.replace('.wav', ''), filled_audio, sr1, format='wav')
    
    print(f"填充后的音频已保存到: {output_path}")
    
    return {
        'filled_audio_path': output_path,
        'original_length': original_audio.shape[-1] / sr1,
        'filled_length': filled_audio.shape[-1] / sr1,
        'gap_start_sec': gap_start_sec,
        'gap_length_sec': gap_length_samples / sr1
    }

def process_directory(original_dir, removed_dir, output_segments_dir, output_filled_dir,
                     model, device, context_duration=5.0, generation_length_tokens=50,
                     gap_start_sec=5.0, overwrite=False, **generation_kwargs):
    """
    Batch process directory
    """
    # Ensure output directories exist (including subdirs)
    from pathlib import Path
    Path(output_segments_dir).mkdir(parents=True, exist_ok=True)
    Path(output_filled_dir).mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for root, dirs, files in os.walk(original_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), original_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        print(f"No audio files found in {original_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load existing results (for resume/merge)
    results_file = os.path.join(output_filled_dir, 'generation_results.json')
    existing_results = []
    processed_set = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
            for r in existing_results:
                if isinstance(r, dict) and 'file_path' in r:
                    processed_set.add(r['file_path'])
            print(f"Found existing results: {len(existing_results)} entries, will merge and resume")
        except Exception as e:
            print(f"Warning: cannot read existing results file {results_file}, will regenerate. Error: {e}")

    results = []
    skipped = 0
    continued_fill = 0

    for rel_path in audio_files:
        original_path = os.path.join(original_dir, rel_path)
        segment_output_path = os.path.join(output_segments_dir, rel_path)
        filled_output_path = os.path.join(output_filled_dir, rel_path)

        # Ensure subdirs exist
        Path(os.path.dirname(segment_output_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(filled_output_path)).mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {rel_path}")

        # 如果已存在填充结果且不覆盖，整首直接跳过
        if (not overwrite) and os.path.exists(filled_output_path):
            print("Filled audio already exists, skip this file")
            skipped += 1
            continue

        gen_result = None
        fill_result = None

        # 生成缺失片段：若无覆盖，且已有片段文件存在，则跳过生成阶段
        if (not overwrite) and os.path.exists(segment_output_path):
            print("Existing generated segment detected, skip generation and try filling directly")
            continued_fill += 1
        else:
            gen_result = generate_missing_segment(
                original_path, segment_output_path, model, device,
                context_duration, generation_length_tokens, **generation_kwargs
            )

        # 只要不存在填充结果（或选择覆盖），就尝试填充
        if overwrite or (not os.path.exists(filled_output_path)):
            fill_result = fill_audio_gap(
                original_path, segment_output_path, filled_output_path,
                gap_start_sec, generation_length_tokens
            )

        # 汇总结果（只有在填充成功时才记录一条完整结果）
        if fill_result:
            result = {**(gen_result if isinstance(gen_result, dict) else {}), **fill_result, 'file_path': rel_path}
            results.append(result)

    # 合并已有与本次结果（按 file_path 去重，后者覆盖前者）
    merged = {}
    for r in existing_results:
        if isinstance(r, dict) and 'file_path' in r:
            merged[r['file_path']] = r
    for r in results:
        if isinstance(r, dict) and 'file_path' in r:
            merged[r['file_path']] = r
    merged_list = list(merged.values())

    # 保存处理结果
    with open(results_file, 'w') as f:
        json.dump(merged_list, f, indent=2)
    
    print(f"\nThis run completed {len(results)} files, skipped {skipped} finished, continued filling {continued_fill} directly.")
    print(f"Processing results saved to: {results_file} (total {len(merged_list)} entries)")
    return merged_list

def main():
    parser = argparse.ArgumentParser(description="Generate missing audio segments and fill gaps")
    parser.add_argument("--original_dir", type=str, default="MusicEval_Small_ori",
                       help="Directory of original full audio")
    parser.add_argument("--removed_dir", type=str, default="MusicEval_Small_removed_50tks",
                       help="Directory of audio after segment removal (optional, for validation)")
    parser.add_argument("--output_segments_dir", type=str, default="MusicEval_Small_generated_segments",
                       help="Output directory for generated segments")
    parser.add_argument("--output_filled_dir", type=str, default="MusicEval_Small_filled",
                       help="Output directory for filled audio")
    parser.add_argument("--context_duration", type=float, default=5.0,
                       help="Context duration (seconds)")
    parser.add_argument("--generation_length_tokens", type=int, default=50,
                       help="Number of tokens to generate")
    parser.add_argument("--gap_start_sec", type=float, default=5.0,
                       help="Gap start position (seconds)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=250,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.0,
                       help="Top-p sampling")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing intermediate/final results, force recompute")
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device(args.force_cpu)
    print(f"Device: {device}")
    
    # Load model
    model = load_model(device)
    
    # Process audio
    generation_kwargs = {
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p
    }
    
    results = process_directory(
        args.original_dir, args.removed_dir, args.output_segments_dir, args.output_filled_dir,
        model, device, args.context_duration, args.generation_length_tokens,
        args.gap_start_sec, overwrite=args.overwrite, **generation_kwargs
    )
    
    print(f"\nBatch processing complete, processed {len(results)} files")

if __name__ == "__main__":
    main()