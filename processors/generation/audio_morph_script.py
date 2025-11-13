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
    audio, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    return audio, target_sr

def audio_morph_with_timing(source_audio, target_audio, morph_start_time, morph_factor=0.6, sr=32000):
    if source_audio.shape[0] == 1 and target_audio.shape[0] == 2:
        target_audio = target_audio.mean(dim=0, keepdim=True)
    elif source_audio.shape[0] == 2 and target_audio.shape[0] == 1:
        target_audio = target_audio.repeat(2, 1)
    
    start_sample = int(morph_start_time * sr)
    
    min_length = min(source_audio.shape[-1], target_audio.shape[-1])
    source_trimmed = source_audio[..., :min_length]
    target_trimmed = target_audio[..., :min_length]
    
    morphed = source_trimmed.clone()
    
    if start_sample < min_length:
        morph_section = source_trimmed[..., start_sample:]
        target_section = target_trimmed[..., start_sample:]
        
        morphed_section = (1 - morph_factor) * morph_section + morph_factor * target_section
        morphed[..., start_sample:] = morphed_section
    
    return morphed

def create_morph_dataset():
    source_file = "/home/evev/asap-dataset/Dataset/Phase2_Stage2_asap_lmd/ASAP_100_rhythm_wav/Beethoven_016_midi_score_short.mid_ori.wav"
    target_dir = "/home/evev/asap-dataset/Dataset/Phase3_Stage1/ShutterStock_20_reordered_token"
    output_dir = "/home/evev/asap-dataset/Morphed_Audio_Output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    time_groups = {
        "2s": 2.0,
        "4s": 4.0,
        "6s": 6.0,
        "8s": 8.0
    }
    
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
    
    for loss_range in loss_categories.keys():
        for time_suffix in time_groups.keys():
            dir_name = f"loss_{loss_range}_morph_{time_suffix}"
            os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    print("开始音频morphing处理...")
    print(f"总共将生成 {len(loss_categories)} × {len(time_groups)} × 12 = {len(loss_categories) * len(time_groups) * 12} 首morphed曲子")
    
    print("加载源音频文件...")
    source_audio, sr = load_audio(source_file)
    
    total_files = 0
    
    for loss_range, target_files in loss_categories.items():
        print(f"\n处理loss范围 {loss_range}...")
        
        for time_suffix, morph_start_time in time_groups.items():
            print(f"  处理时间组 {time_suffix} (从第{morph_start_time}秒开始morph)...")
            
            output_subdir = f"loss_{loss_range}_morph_{time_suffix}"
            
            for i, target_file in enumerate(tqdm(target_files[:12], desc=f"Loss {loss_range} - {time_suffix}")):
                target_path = os.path.join(target_dir, target_file)
                
                base_name = target_file.replace('_token.mp3', '')
                output_filename = f"beethoven_morph_to_{base_name}_loss{loss_range}_start{time_suffix}.wav"
                output_path = os.path.join(output_dir, output_subdir, output_filename)
                
                target_audio, _ = load_audio(target_path)
                
                morphed_audio = audio_morph_with_timing(
                    source_audio, 
                    target_audio, 
                    morph_start_time, 
                    morph_factor=0.6, 
                    sr=sr
                )
                
                sf.write(output_path, morphed_audio.numpy().T, sr)
                total_files += 1
    
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