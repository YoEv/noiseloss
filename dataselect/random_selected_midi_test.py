import os
import pandas as pd
from pretty_midi import PrettyMIDI
import random
from pathlib import Path

# 配置路径
ASAP_BASE = "/Volumes/MCastile/Castile/HackerProj/asap-dataset"  # 替换为您的ASAP根目录
OUTPUT_DIR = "./selected_midis"       # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载元数据
meta = pd.read_csv(os.path.join(ASAP_BASE, "metadata.csv"))

# 筛选符合条件的片段
good_pieces = []
for _, row in meta.iterrows():
    try:
        midi_path = os.path.join(ASAP_BASE, row['midi_score'])
        midi = PrettyMIDI(midi_path)
        
        if 16 <= len(midi.get_downbeats()) <= 32 and \
           80 <= midi.estimate_tempo() <= 140 and \
           len(midi.key_signature_changes) == 0:
            good_pieces.append(row)
    except Exception as e:
        print(f"跳过 {row['midi_score']} (错误: {str(e)})")
        continue

# 随机选择100个
selected = random.sample(good_pieces, min(100, len(good_pieces)))

# 保存选中的MIDI（含裁剪）
for i, row in enumerate(selected):
    try:
        # 原始MIDI路径
        src_midi = os.path.join(ASAP_BASE, row['midi_score'])
        
        # 新文件名 (保留作曲家信息)
        composer_clean = row['composer'].replace(" ", "_")
        output_name = f"{composer_clean}_{i:03d}_{os.path.basename(src_midi)}"
        dest_midi = os.path.join(OUTPUT_DIR, output_name)
        
        # 方案A：直接复制完整MIDI
        # import shutil
        # shutil.copy2(src_midi, dest_midi)
        
        # 方案B：裁剪片段（推荐）
        midi = PrettyMIDI(src_midi)
        
        # 自动检测乐段边界（示例：取前32小节）
        downbeats = midi.get_downbeats()
        end_time = downbeats[32] if len(downbeats) > 32 else midi.get_end_time()
        
        # 创建裁剪后的MIDI
        cropped_midi = PrettyMIDI()
        for instrument in midi.instruments:
            new_instr = instrument.__class__(program=instrument.program)
            new_instr.notes = [n for n in instrument.notes if n.start < end_time]
            cropped_midi.instruments.append(new_instr)
        
        # 保存裁剪版本
        cropped_midi.write(dest_midi)
        print(f"已保存: {output_name} (时长: {end_time:.2f}s)")
        
    except Exception as e:
        print(f"处理失败 {row['midi_score']}: {str(e)}")

print(f"\n完成！已保存 {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mid')])} 个MIDI到 {OUTPUT_DIR}")