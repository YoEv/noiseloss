import os
import random
import copy
from pretty_midi import PrettyMIDI, note_number_to_name
from tqdm import tqdm
from collections import defaultdict

# 配置常量
MAX_NOTE_GAP = 0.15  # 合并为同一和弦的最大时间差（秒）
MIN_CHORD_NOTES = 3   # 视为和弦的最小音符数
DIATONIC_SCALE = [0,2,4,5,7,9,11]  # 大调音阶半音偏移
CHORD_TONES = [0,4,7]  # 大三和弦组成音

class MidiModifier:
    def __init__(self):
        self.log = []
    
    def _cluster_chords(self, midi):
        """改进的和弦检测算法"""
        chords = []
        notes = sorted(midi.instruments[0].notes, key=lambda x: x.start)
        
        current_chord = []
        for note in notes:
            if current_chord and (note.start - current_chord[-1].start) > MAX_NOTE_GAP:
                if len(current_chord) >= MIN_CHORD_NOTES:
                    chords.append(current_chord)
                current_chord = []
            current_chord.append(note)
        
        if len(current_chord) >= MIN_CHORD_NOTES:
            chords.append(current_chord)
        
        return chords

    def _validate_pitch(self, midi, original_pitch, new_pitch, chord_root):
        """增强的调性验证"""
        # 获取主调
        if midi.key_signature_changes:
            key = midi.key_signature_changes[0].key_number % 12
        else:
            key = 0  # 默认C大调
        
        # 检查是否在调式音阶内
        scale_degrees = [(key + offset) % 12 for offset in DIATONIC_SCALE]
        if (new_pitch % 12) in scale_degrees:
            return True
        
        # 检查是否为和弦内音
        chord_degree = (new_pitch - chord_root) % 12
        if chord_degree in CHORD_TONES:
            return True
        
        return False

    def modify_single_note(self, midi_path, output_path):
        """执行单音修改并保存"""
        try:
            # 加载并克隆原始MIDI
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = copy.deepcopy(orig_midi)
            
            # 分析和弦
            chords = self._cluster_chords(mod_midi)
            if not chords:
                return False
            
            # 随机选择一个和弦
            selected_chord = random.choice(chords)
            chord_root = min(n.pitch for n in selected_chord) % 12
            
            # 随机选择一个音符修改
            target_note = random.choice(selected_chord)
            original_pitch = target_note.pitch
            
            # 生成合规的八度变化
            for _ in range(20):  # 最多尝试20次
                new_pitch = original_pitch + random.choice([-12, 12])
                if self._validate_pitch(mod_midi, original_pitch, new_pitch, chord_root):
                    target_note.pitch = new_pitch
                    # 记录修改
                    self.log.append({
                        'action': 'single_note',
                        'time': target_note.start,
                        'original': f"{note_number_to_name(original_pitch)} ({original_pitch})",
                        'modified': f"{note_number_to_name(new_pitch)} ({new_pitch})"
                    })
                    break
            else:  # 未找到合规修改
                return False
            
            # 保存修改后的MIDI
            mod_midi.write(output_path)
            return True
            
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            return False

    def modify_voicing(self, midi_path, output_path):
        """执行和弦排列修改并保存"""
        try:
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = copy.deepcopy(orig_midi)
            
            chords = self._cluster_chords(mod_midi)
            if not chords:
                return False
            
            # 随机选择一个和弦
            selected_chord = random.choice(chords)
            chord_notes = sorted(selected_chord, key=lambda x: x.pitch)
            
            # 确保至少有三个音符
            if len(chord_notes) < 3:
                return False
            
            # 记录原始状态
            original_pitches = [n.pitch for n in chord_notes]
            
            # 执行转位操作
            inversion_type = random.choice(['root', 'first', 'second'])
            
            if inversion_type == 'first':
                # 第一转位：最低音升八度
                chord_notes[0].pitch += 12
            elif inversion_type == 'second':
                # 第二转位：次低音升八度
                if len(chord_notes) > 1:
                    chord_notes[1].pitch += 12
            
            # 重新排序并避免声部交叉
            chord_notes.sort(key=lambda x: x.pitch)
            for i in range(1, len(chord_notes)):
                if chord_notes[i].pitch < chord_notes[i-1].pitch + 3:
                    chord_notes[i].pitch = chord_notes[i-1].pitch + 3
            
            # 记录修改
            self.log.append({
                'action': 'voicing',
                'time': chord_notes[0].start,
                'original': [note_number_to_name(p) for p in original_pitches],
                'modified': [note_number_to_name(n.pitch) for n in chord_notes]
            })
            
            mod_midi.write(output_path)
            return True
            
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            return False

    def process_directory(self, input_dir, output_dir):
        """批量处理目录"""
        os.makedirs(output_dir, exist_ok=True)
        midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
        
        success = 0
        for fname in tqdm(midi_files, desc="Processing MIDIs"):
            in_path = os.path.join(input_dir, fname)
            
            # Action 1: 单音修改
            action1_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_action1.mid")
            if self.modify_single_note(in_path, action1_path):
                success += 1
            
            # Action 2: 排列修改
            action2_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_action2.mid")
            if self.modify_voicing(in_path, action2_path):
                success += 1
        
        print(f"\n成功处理 {success}/{len(midi_files)*2} 个修改")
        return self.log

# 使用示例
if __name__ == "__main__":
    modifier = MidiModifier()
    log = modifier.process_directory(
        input_dir="./selected_midis_cropped",
        output_dir="./modified_output"
    )
    
    # 打印修改记录
    print("\n=== 详细修改记录 ===")
    for entry in log:
        if entry['action'] == 'single_note':
            print(f"单音修改 @ {entry['time']:.2f}s")
            print(f"  Original: {entry['original']}")
            print(f"  Modified: {entry['modified']}")
        else:
            print(f"和弦排列修改 @ {entry['time']:.2f}s")
            print(f"  Original: {entry['original']}")
            print(f"  Modified: {entry['modified']}")