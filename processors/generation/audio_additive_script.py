import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random
import argparse

def load_audio(file_path, target_sr=32000):
    audio, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    return audio, target_sr

def normalize_channels(audio1, audio2):
    if audio1.shape[0] == 1 and audio2.shape[0] == 2:
        audio2 = audio2.mean(dim=0, keepdim=True)
    elif audio1.shape[0] == 2 and audio2.shape[0] == 1:
        audio2 = audio2.repeat(2, 1)
    
    return audio1, audio2

def additive_synthesis(base_audio, add_audio, percentage, sr=32000, segment_min=0.01, segment_max=0.05):
    """使用加法合成，随机选择短片段进行叠加
    
    Args:
        base_audio: 基础音频 (Beethoven)
        add_audio: 要添加的音频 (ShutterStock)
        percentage: 添加的总时长百分比 (5-80)
        sr: 采样率
        segment_min: 最小片段长度 (秒)
        segment_max: 最大片段长度 (秒)
    """
    base_audio, add_audio = normalize_channels(base_audio, add_audio)
    
    min_length = min(base_audio.shape[-1], add_audio.shape[-1])
    base_trimmed = base_audio[..., :min_length].clone()
    add_trimmed = add_audio[..., :min_length]
    
    total_duration = min_length / sr  # 总时长（秒）
    add_duration = total_duration * (percentage / 100.0)  # 要添加的时长（秒）
    
    segment_min_samples = int(segment_min * sr)
    segment_max_samples = int(segment_max * sr)
    add_samples_total = int(add_duration * sr)
    
    print(f"  添加 {percentage}% 的音频片段 (约 {add_duration:.2f}秒)")
    
    added_samples = 0
    segments_added = 0
    
    while added_samples < add_samples_total:
        segment_length = random.randint(segment_min_samples, segment_max_samples)
        
        segment_length = min(segment_length, add_samples_total - added_samples)
        
        max_start_pos = min_length - segment_length
        if max_start_pos <= 0:
            break
            
        base_start = random.randint(0, max_start_pos)
        
        add_max_start = add_trimmed.shape[-1] - segment_length
        if add_max_start <= 0:
            break
            
        add_start = random.randint(0, add_max_start)
        
        base_segment = base_trimmed[..., base_start:base_start + segment_length]
        add_segment = add_trimmed[..., add_start:add_start + segment_length]
        
        mix_ratio = 0.3 
        mixed_segment = base_segment + mix_ratio * add_segment
        
        mixed_segment = torch.clamp(mixed_segment, -1.0, 1.0)
        
        base_trimmed[..., base_start:base_start + segment_length] = mixed_segment
        
        added_samples += segment_length
        segments_added += 1
    
    print(f"  总共添加了 {segments_added} 个片段")
    
    return base_trimmed

def create_additive_dataset():
    base_file = "/home/evev/asap-dataset/Dataset/Phase2_Stage2_asap_lmd/ASAP_100_rhythm_wav/Beethoven_016_midi_score_short.mid_ori.wav"

    add_file = "/home/evev/asap-dataset/Dataset/Phase3_Stage1/ShutterStock_20_reordered_token/1217199_mozart-rondo-alla-turca_short30_preview_cut14s_reordered_50p_token.mp3"
    
    output_dir = "/home/evev/asap-dataset/Additive_Synthesis_Output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    percentages = list(range(5, 85, 5))  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    
    print(f"开始加法合成处理...")
    print(f"基础音频: {os.path.basename(base_file)}")
    print(f"添加音频: {os.path.basename(add_file)}")
    print(f"总共将生成 {len(percentages)} 首合成音频")
    print(f"百分比范围: {percentages[0]}% - {percentages[-1]}%")

    print("\n加载基础音频文件...")
    base_audio, sr = load_audio(base_file)
    
    print("加载添加音频文件...")
    add_audio, _ = load_audio(add_file)
    
    add_basename = os.path.basename(add_file).replace('.mp3', '').replace('_token', '')
    
    total_files = 0
    
    for percentage in tqdm(percentages, desc="生成加法合成音频"):
        
        synthesized_audio = additive_synthesis(
            base_audio, 
            add_audio, 
            percentage, 
            sr=sr
        )
        
        output_filename = f"beethoven_016_additive_{percentage}p_{add_basename}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        sf.write(output_path, synthesized_audio.numpy().T, sr)
        total_files += 1
    
    print(f"\n加法合成完成！总共生成了 {total_files} 首合成音频")
    print(f"输出目录: {output_dir}")
    
    print("\n生成的文件:")
    for percentage in percentages:
        filename = f"beethoven_016_additive_{percentage}p_{add_basename}.wav"
        print(f"├── {filename}")
    
    print("\n文件命名格式:")
    print("beethoven_016_additive_[百分比]p_[ShutterStock文件名].wav")
    print(f"例如: beethoven_016_additive_25p_{add_basename}.wav")

def create_multiple_files_additive_dataset():
    """为多个ShutterStock文件创建加法合成数据集"""
    base_file = "/home/evev/asap-dataset/Dataset/Phase2_Stage2_asap_lmd/ASAP_100_rhythm_wav/Beethoven_016_midi_score_short.mid_ori.wav"
    target_dir = "/home/evev/asap-dataset/Dataset/Phase3_Stage1/ShutterStock_20_reordered_token"
    output_dir = "/home/evev/asap-dataset/Additive_Synthesis_Multiple_Output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    selected_files = [
        "1217199_mozart-rondo-alla-turca_short30_preview_cut14s_reordered_50p_token.mp3",
        "1228481_piano-andante_short30_preview_cut14s_reordered_30p_token.mp3",
        "510410_chopin---raindrops-(solo-piano-arrangement)_short30_preview_cut14s_reordered_80p_token.mp3"
    ]
    
    percentages = list(range(5, 85, 5))
    
    print(f"开始多文件加法合成处理...")
    print(f"基础音频: {os.path.basename(base_file)}")
    print(f"选择的ShutterStock文件数量: {len(selected_files)}")
    print(f"每个文件的百分比数量: {len(percentages)}")
    print(f"总共将生成: {len(selected_files)} × {len(percentages)} = {len(selected_files) * len(percentages)} 首合成音频")
    
    print("\n加载基础音频文件...")
    base_audio, sr = load_audio(base_file)
    
    total_files = 0
    
    for add_filename in selected_files:
        add_file_path = os.path.join(target_dir, add_filename)
        
        print(f"\n处理文件: {add_filename}")
        
        try:
            add_audio, _ = load_audio(add_file_path)
            
            add_basename = add_filename.replace('.mp3', '').replace('_token', '')
            
            for percentage in tqdm(percentages, desc=f"处理 {add_basename[:30]}..."):
                try:
                    synthesized_audio = additive_synthesis(
                        base_audio, 
                        add_audio, 
                        percentage, 
                        sr=sr
                    )
                    
                    output_filename = f"beethoven_016_additive_{percentage}p_{add_basename}.wav"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    sf.write(output_path, synthesized_audio.numpy().T, sr)
                    total_files += 1
                    
                except Exception as e:
                    print(f"  处理 {percentage}% 时出错: {e}")
                    
        except Exception as e:
            print(f"加载文件 {add_filename} 时出错: {e}")
    
    print(f"\n多文件加法合成完成！总共生成了 {total_files} 首合成音频")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='音频加法合成脚本')
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single',
                        help='选择模式: single (单个文件) 或 multiple (多个文件)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        create_additive_dataset()
    else:
        create_multiple_files_additive_dataset()