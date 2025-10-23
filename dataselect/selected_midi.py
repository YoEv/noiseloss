import os
import pandas as pd
from pretty_midi import PrettyMIDI, Instrument
import random
import warnings
from pathlib import Path

# 统一裁剪为32小节，符合条件的midi
# 禁用特定警告（处理无效MIDI文件的元数据问题）
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")

# 配置路径
ASAP_BASE = "/Volumes/MCastile/Castile/HackerProj/asap-dataset"  # 替换为您的实际路径
OUTPUT_DIR = "./selected_midis_cropped"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_validate_midi(midi_path):
    """安全加载MIDI文件并验证基本结构"""
    try:
        midi = PrettyMIDI(midi_path)
        # 基础验证
        if len(midi.instruments) == 0:
            raise ValueError("No instruments found")
        if len(midi.instruments[0].notes) < 10:  # 至少包含10个音符
            raise ValueError("Too few notes")
        return midi
    except Exception as e:
        print(f"Invalid MIDI {midi_path}: {str(e)}")
        return None

def crop_midi(midi, start_beat=0, num_bars=32):
    """智能裁剪MIDI片段"""
    downbeats = midi.get_downbeats()
    if len(downbeats) < num_bars + 1:
        return midi  # 不足指定小节则返回完整

    # 计算起止时间（音乐小节为单位）
    start_time = downbeats[start_beat]
    end_time = downbeats[start_beat + num_bars]

    # 创建新MIDI对象并转移属性
    new_midi = PrettyMIDI()
    new_midi.time_signature_changes = [
        ts for ts in midi.time_signature_changes
        if ts.time < end_time
    ]
    new_midi.key_signature_changes = [
        ks for ks in midi.key_signature_changes
        if ks.time < end_time
    ]

    # 处理每个乐器轨道
    for instr in midi.instruments:
        new_instr = Instrument(
            program=instr.program,
            is_drum=instr.is_drum,
            name=instr.name
        )
        # 裁剪音符（包含跨越边界的音符）
        new_instr.notes = [
            note for note in instr.notes
            if note.start < end_time and note.end > start_time
        ]
        if new_instr.notes:  # 只保留有音符的轨道
            new_midi.instruments.append(new_instr)

    return new_midi

# 主处理流程
if __name__ == "__main__":
    # 加载元数据
    meta = pd.read_csv(os.path.join(ASAP_BASE, "metadata.csv"))

    # 筛选有效片段
    valid_pieces = []
    for _, row in meta.iterrows():
        midi_path = os.path.join(ASAP_BASE, row['midi_score'])
        if load_and_validate_midi(midi_path):
            valid_pieces.append(row)

    # 随机选择100个（或全部不足时）
    selected = random.sample(valid_pieces, min(100, len(valid_pieces)))

    # 处理并保存MIDI
    success_count = 0
    for idx, row in enumerate(selected):
        try:
            midi = load_and_validate_midi(os.path.join(ASAP_BASE, row['midi_score']))
            if not midi:
                continue

            # 裁剪MIDI（前32小节）
            cropped = crop_midi(midi, num_bars=32)

            # 生成输出文件名
            composer = row['composer'].replace(" ", "_")
            title = Path(row['midi_score']).stem
            output_path = os.path.join(OUTPUT_DIR, f"{composer}_{idx:03d}_{title}.mid")

            # 保存并验证
            cropped.write(output_path)
            if os.path.exists(output_path):
                success_count += 1
                print(f"✅ Saved: {output_path} ({len(cropped.instruments[0].notes)} notes)")
            else:
                print(f"❌ Failed to save: {output_path}")

        except Exception as e:
            print(f"⚠️ Error processing {row['midi_score']}: {str(e)}")

    # 生成元数据报告
    report = {
        "original_count": len(selected),
        "success_count": success_count,
        "output_dir": os.path.abspath(OUTPUT_DIR)
    }
    print("\n" + "="*50)
    print(f"处理完成！成功保存 {success_count}/{len(selected)} 个MIDI文件")
    print(f"输出目录: {report['output_dir']}")
    print("="*50)