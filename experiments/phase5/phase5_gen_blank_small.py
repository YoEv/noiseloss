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
        # 优先在加载时传入 device（新版本接口支持）
        model = MusicGen.get_pretrained('facebook/musicgen-small', device=str(device))
    except TypeError:
        # 兼容旧版本接口：先加载，再设置设备
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        if hasattr(model, 'set_device'):
            model.set_device(device)
        else:
            print("警告: 当前 MusicGen 版本不支持 set_device/传入 device，将继续在CPU上运行")
    # 注意：不要再调用 model.to(device)，该对象没有 .to 方法
    return model

def generate_missing_segment(original_audio_path, output_path, model, device,
                           context_duration=5.0, generation_length_tokens=50,
                           temperature=0.7, top_k=250, top_p=0.0):
    """
    生成缺失的音频片段
    
    Args:
        original_audio_path: 原始完整音频路径
        output_path: 生成片段的输出路径
        model: MusicGen模型
        device: 设备
        context_duration: 上下文时长（秒）
        generation_length_tokens: 生成的token数量
        temperature: 生成温度
        top_k: top-k采样
        top_p: top-p采样
    """
    try:
        # 加载原始音频
        print(f"加载原始音频: {original_audio_path}")
        audio, sr = audio_read(original_audio_path)
        
        # 确保单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 重采样到32kHz
        if sr != 32000:
            audio = torchaudio.functional.resample(audio, sr, 32000)
            sr = 32000
        
        # 提取前context_duration秒作为conditioning
        context_samples = int(context_duration * sr)
        if audio.shape[-1] < context_samples:
            print(f"警告: 音频长度不足{context_duration}秒，使用全部音频作为上下文")
            context_audio = audio
        else:
            context_audio = audio[:, :context_samples]
        
        print(f"上下文音频长度: {context_audio.shape[-1]/sr:.2f}秒")
        
        # 计算生成时长
        tokens_per_sec = 32000 // 320  # 约100 tokens/sec
        generation_duration = generation_length_tokens / tokens_per_sec
        total_duration = context_duration + generation_duration
        
        print(f"将生成 {generation_length_tokens} tokens ({generation_duration:.2f}秒)")
        print(f"总输出时长: {total_duration:.2f}秒")
        
        # 设置生成参数
        model.set_generation_params(
            duration=total_duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_sampling=True
        )
        
        # 生成音频
        print("开始生成...")
        with torch.no_grad():
            generated = model.generate_continuation(
                context_audio.unsqueeze(0).to(device),
                prompt_sample_rate=sr,
                progress=True
            )
        
        # 提取生成的部分（去掉上下文部分）
        generated_audio = generated.squeeze(0).cpu()
        context_samples_out = int(context_duration * sr)
        
        if generated_audio.shape[-1] > context_samples_out:
            generated_segment = generated_audio[:, context_samples_out:]
        else:
            print("警告: 生成的音频长度不足，使用全部生成结果")
            generated_segment = generated_audio
        
        print(f"生成片段长度: {generated_segment.shape[-1]/sr:.2f}秒")
        
        # 保存生成的片段
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio_write(output_path.replace('.wav', ''), generated_segment, sr, format='wav')
        
        print(f"生成的音频片段已保存到: {output_path}")
        
        return {
            'original_audio_path': original_audio_path,
            'generated_segment_path': output_path,
            'context_duration': context_duration,
            'generation_duration': generation_segment.shape[-1] / sr,
            'generation_length_tokens': generation_length_tokens,
            'actual_generated_samples': generated_segment.shape[-1]
        }
        
    except Exception as e:
        print(f"生成音频片段时出错 {original_audio_path}: {str(e)}")
        return None

def fill_audio_gap(original_audio_path, generated_segment_path, output_path,
                  gap_start_sec=5.0, gap_length_tokens=50):
    """
    将生成的片段填入原始音频的空白处
    
    Args:
        original_audio_path: 原始完整音频路径
        generated_segment_path: 生成片段路径
        output_path: 填充后音频的输出路径
        gap_start_sec: 空白开始位置（秒）
        gap_length_tokens: 空白长度（tokens）
    """
    try:
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
        
    except Exception as e:
        print(f"填充音频时出错: {str(e)}")
        return None

def process_directory(original_dir, removed_dir, output_segments_dir, output_filled_dir,
                     model, device, context_duration=5.0, generation_length_tokens=50,
                     gap_start_sec=5.0, overwrite=False, **generation_kwargs):
    """
    批量处理目录
    """
    # 确保输出目录存在（含子目录）
    from pathlib import Path
    Path(output_segments_dir).mkdir(parents=True, exist_ok=True)
    Path(output_filled_dir).mkdir(parents=True, exist_ok=True)

    # 找到所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(original_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), original_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        print(f"在 {original_dir} 中未找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 载入已有的处理结果（用于续跑合并）
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
            print(f"检测到已有结果: {len(existing_results)} 条，将进行合并并断点续跑")
        except Exception as e:
            print(f"警告: 无法读取已有结果文件 {results_file}，将重新生成。错误: {e}")

    results = []
    skipped = 0
    continued_fill = 0

    for rel_path in audio_files:
        original_path = os.path.join(original_dir, rel_path)
        segment_output_path = os.path.join(output_segments_dir, rel_path)
        filled_output_path = os.path.join(output_filled_dir, rel_path)

        # 确保子目录存在
        Path(os.path.dirname(segment_output_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(filled_output_path)).mkdir(parents=True, exist_ok=True)
        
        print(f"\n处理: {rel_path}")

        # 如果已存在填充结果且不覆盖，整首直接跳过
        if (not overwrite) and os.path.exists(filled_output_path):
            print("已存在填充后的音频，跳过该文件")
            skipped += 1
            continue

        gen_result = None
        fill_result = None

        # 生成缺失片段：若无覆盖，且已有片段文件存在，则跳过生成阶段
        if (not overwrite) and os.path.exists(segment_output_path):
            print("检测到已存在生成片段，跳过生成阶段，直接尝试填充")
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
    
    print(f"\n本次新增完成 {len(results)} 个文件，跳过已完成 {skipped} 个，直接继续填充 {continued_fill} 个。")
    print(f"处理结果已保存到: {results_file}（总计 {len(merged_list)} 条）")
    return merged_list

def main():
    parser = argparse.ArgumentParser(description="生成缺失的音频片段并填充")
    parser.add_argument("--original_dir", type=str, default="MusicEval_Small_ori",
                       help="原始完整音频目录")
    parser.add_argument("--removed_dir", type=str, default="MusicEval_Small_removed_50tks",
                       help="删除片段后的音频目录（可选，用于验证）")
    parser.add_argument("--output_segments_dir", type=str, default="MusicEval_Small_generated_segments",
                       help="生成片段输出目录")
    parser.add_argument("--output_filled_dir", type=str, default="MusicEval_Small_filled",
                       help="填充后音频输出目录")
    parser.add_argument("--context_duration", type=float, default=5.0,
                       help="上下文时长（秒）")
    parser.add_argument("--generation_length_tokens", type=int, default=50,
                       help="生成的token数量")
    parser.add_argument("--gap_start_sec", type=float, default=5.0,
                       help="空白开始位置（秒）")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--top_k", type=int, default=250,
                       help="top-k采样")
    parser.add_argument("--top_p", type=float, default=0.0,
                       help="top-p采样")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的中间/最终结果，强制重算")
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(device)
    
    # 处理音频
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
    
    print(f"\n批量处理完成，共处理 {len(results)} 个文件")

if __name__ == "__main__":
    main()