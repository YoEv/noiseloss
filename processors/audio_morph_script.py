import os
import torch
import torchaudio
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import soundfile as sf
from tqdm import tqdm
import argparse

def load_audio(file_path, target_sr=32000):
    """加载音频文件并重采样到目标采样率"""
    audio, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    return audio, target_sr

def audio_morph_with_timing(source_audio, target_audio, morph_start_time, morph_factor=0.6, sr=32000):
    """在指定时间点开始进行音频morphing"""
    # 统一通道数处理
    # 如果源音频是单声道，目标音频是立体声，将目标音频转为单声道
    if source_audio.shape[0] == 1 and target_audio.shape[0] == 2:
        target_audio = target_audio.mean(dim=0, keepdim=True)  # 立体声转单声道
    # 如果源音频是立体声，目标音频是单声道，将目标音频复制为立体声
    elif source_audio.shape[0] == 2 and target_audio.shape[0] == 1:
        target_audio = target_audio.repeat(2, 1)  # 单声道转立体声
    
    # 计算开始morphing的样本点
    start_sample = int(morph_start_time * sr)
    
    # 确保两个音频长度相同
    min_length = min(source_audio.shape[-1], target_audio.shape[-1])
    source_trimmed = source_audio[..., :min_length]
    target_trimmed = target_audio[..., :min_length]
    
    # 创建morphed音频
    morphed = source_trimmed.clone()
    
    # 从指定时间点开始进行morphing
    if start_sample < min_length:
        morph_section = source_trimmed[..., start_sample:]
        target_section = target_trimmed[..., start_sample:]
        
        # 线性插值morphing
        morphed_section = (1 - morph_factor) * morph_section + morph_factor * target_section
        morphed[..., start_sample:] = morphed_section
    
    return morphed

def create_morph_dataset():
    """创建完整的morph数据集"""
    # 文件路径
    source_file = "/home/evev/asap-dataset/Dataset/Phase2_Stage2_asap_lmd/ASAP_100_rhythm_wav/Beethoven_016_midi_score_short.mid_ori.wav"
    target_dir = "/home/evev/asap-dataset/Dataset/Phase3_Stage1/ShutterStock_20_reordered_token"
    output_dir = "/home/evev/asap-dataset/Morphed_Audio_Output"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 时间组设置（从第2、4、6、8秒开始morph）
    time_groups = {
        "2s": 2.0,
        "4s": 4.0,
        "6s": 6.0,
        "8s": 8.0
    }
    
    # Loss值分类（基于之前的分析，每组选择12首）
    loss_categories = {
        "5-6": [
            "510410_chopin---raindrops-(solo-piano-arrangement)_short30_preview_cut14s_reordered_80p_token.mp3",
            "510410_chopin---raindrops-(solo-piano-arrangement)_short30_preview_cut14s_reordered_30p_token.mp3",
            "1251571_a-long-night_short30_preview_cut14s_reordered_80p_token.mp3",
            "510410_chopin---raindrops-(solo-piano-arrangement)_short30_preview_cut14s_reordered_5p_token.mp3",
            "1251571_a-long-night_short30_preview_cut14s_reordered_50p_token.mp3",
            "1250854_hugs-are-the-best-medicine_short30_preview_cut14s_reordered_80p_token.mp3",
            "1251571_a-long-night_short30_preview_cut14s_reordered_30p_token.mp3",
            "1250854_hugs-are-the-best-medicine_short30_preview_cut14s_reordered_50p_token.mp3",
            "510410_chopin---raindrops-(solo-piano-arrangement)_short30_preview_cut14s_reordered_10p_token.mp3",
            "510410_chopin---raindrops-(solo-piano-arrangement)_short30_preview_cut14s_reordered_50p_token.mp3",
            "1250854_hugs-are-the-best-medicine_short30_preview_cut14s_reordered_30p_token.mp3",
            "1251571_a-long-night_short30_preview_cut14s_reordered_10p_token.mp3"
        ],
        "6-7": [
            "1228481_piano-andante_short30_preview_cut14s_reordered_5p_token.mp3",
            "1228481_piano-andante_short30_preview_cut14s_reordered_50p_token.mp3",
            "433499_fashionable_short30_preview_cut14s_reordered_10p_token.mp3",
            "1234504_put-it-there_short30_preview_cut14s_reordered_10p_token.mp3",
            "1217199_mozart-rondo-alla-turca_short30_preview_cut14s_reordered_30p_token.mp3",
            "1219263_connected-through-hope_short30_preview_cut14s_reordered_80p_token.mp3",
            "1220765_les-antique_short30_preview_cut14s_reordered_10p_token.mp3",
            "1228481_piano-andante_short30_preview_cut14s_reordered_10p_token.mp3",
            "1217199_mozart-rondo-alla-turca_short30_preview_cut14s_reordered_10p_token.mp3",
            "1228481_piano-andante_short30_preview_cut14s_reordered_80p_token.mp3",
            "1242936_the-drive-for-resolution_short30_preview_cut14s_reordered_50p_token.mp3",
            "1219263_connected-through-hope_short30_preview_cut14s_reordered_5p_token.mp3"
        ],
        "7-8": [
            "1219906_spreading-your-wings_short30_preview_cut14s_reordered_30p_token.mp3",
            "1236170_with-the-greatest-will_short30_preview_cut14s_reordered_30p_token.mp3",
            "1256122_one-last-time_short30_preview_cut14s_reordered_30p_token.mp3",
            "1219906_spreading-your-wings_short30_preview_cut14s_reordered_50p_token.mp3",
            "1236170_with-the-greatest-will_short30_preview_cut14s_reordered_50p_token.mp3",
            "1256122_one-last-time_short30_preview_cut14s_reordered_50p_token.mp3",
            "1236170_with-the-greatest-will_short30_preview_cut14s_reordered_80p_token.mp3",
            "1256250_shouting-it-out_short30_preview_cut14s_reordered_80p_token.mp3",
            "1219906_spreading-your-wings_short30_preview_cut14s_reordered_10p_token.mp3",
            "1256250_shouting-it-out_short30_preview_cut14s_reordered_50p_token.mp3",
            "1219906_spreading-your-wings_short30_preview_cut14s_reordered_80p_token.mp3",
            "1236170_with-the-greatest-will_short30_preview_cut14s_reordered_5p_token.mp3"
        ]
    }
    
    # 为每个loss范围和时间组合创建输出目录
    for loss_range in loss_categories.keys():
        for time_suffix in time_groups.keys():
            dir_name = f"loss_{loss_range}_morph_{time_suffix}"
            os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    print("开始音频morphing处理...")
    print(f"总共将生成 {len(loss_categories)} × {len(time_groups)} × 12 = {len(loss_categories) * len(time_groups) * 12} 首morphed曲子")
    
    # 加载源音频
    print("加载源音频文件...")
    source_audio, sr = load_audio(source_file)
    
    total_files = 0
    
    # 遍历每个loss范围
    for loss_range, target_files in loss_categories.items():
        print(f"\n处理loss范围 {loss_range}...")
        
        # 遍历每个时间组
        for time_suffix, morph_start_time in time_groups.items():
            print(f"  处理时间组 {time_suffix} (从第{morph_start_time}秒开始morph)...")
            
            output_subdir = f"loss_{loss_range}_morph_{time_suffix}"
            
            # 处理该组的12首曲子
            for i, target_file in enumerate(tqdm(target_files[:12], desc=f"Loss {loss_range} - {time_suffix}")):
                target_path = os.path.join(target_dir, target_file)
                
                # 生成输出文件名，包含所有相关信息
                base_name = target_file.replace('_token.mp3', '')
                output_filename = f"beethoven_morph_to_{base_name}_loss{loss_range}_start{time_suffix}.wav"
                output_path = os.path.join(output_dir, output_subdir, output_filename)
                
                try:
                    # 加载目标音频
                    target_audio, _ = load_audio(target_path)
                    
                    # 执行带时间控制的morphing
                    morphed_audio = audio_morph_with_timing(
                        source_audio, 
                        target_audio, 
                        morph_start_time, 
                        morph_factor=0.6, 
                        sr=sr
                    )
                    
                    # 保存morphed音频
                    sf.write(output_path, morphed_audio.numpy().T, sr)
                    total_files += 1
                    
                except Exception as e:
                    print(f"处理 {target_file} (时间组 {time_suffix}) 时出错: {e}")
    
    print(f"\n音频morphing完成！总共生成了 {total_files} 首morphed曲子")
    print(f"输出目录: {output_dir}")
    print("\n文件夹结构:")
    for loss_range in loss_categories.keys():
        for time_suffix in time_groups.keys():
            print(f"├── loss_{loss_range}_morph_{time_suffix}/ (12首曲子)")
    
    print("\n文件命名格式:")
    print("beethoven_morph_to_[目标曲子名]_loss[范围]_start[时间].wav")
    print("例如: beethoven_morph_to_1228481_piano-andante_short30_preview_cut14s_reordered_5p_loss6-7_start2s.wav")

if __name__ == "__main__":
    create_morph_dataset()