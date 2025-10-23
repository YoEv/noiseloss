import os
import random
import copy
import json
from datetime import datetime
from pretty_midi import PrettyMIDI, note_number_to_name
from tqdm import tqdm

# 常量定义
MAX_NOTE_GAP = 0.15
MIN_CHORD_NOTES = 3
SAFE_PITCH_RANGE = (12, 115)  # 安全修改范围

class MidiModifier:
    def __init__(self, output_dir):
        self.log = []
        self.error_log = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化日志文件
        self.log_file = open(os.path.join(output_dir, "process_log.txt"), "w", encoding="utf-8")
        self._log_message(f"初始化处理器，输出目录：{output_dir}")

    def __del__(self):
        self.log_file.close()

    def _log_message(self, message, error=False):
        """统一日志记录"""
        log_entry = f"[{datetime.now().isoformat()}] {'ERROR' if error else 'INFO'} - {message}"
        self.log_file.write(log_entry + "\n")
        if error:
            self.error_log.append(log_entry)

    def _clamp_pitch(self, pitch):
        """强制限制音高范围"""
        return max(0, min(127, pitch))

    def _deep_clone_midi(self, midi):
        """安全克隆MIDI对象"""
        new_midi = PrettyMIDI()
        # 克隆元数据
        new_midi.time_signature_changes = copy.deepcopy(midi.time_signature_changes)
        new_midi.key_signature_changes = copy.deepcopy(midi.key_signature_changes)
        # 克隆乐器轨道
        for instr in midi.instruments:
            new_instr = copy.deepcopy(instr)
            new_instr.notes = [copy.deepcopy(n) for n in instr.notes]  # 深度复制音符
            new_midi.instruments.append(new_instr)
        return new_midi

    def _validate_midi(self, midi):
        """最终音高验证"""
        for instr in midi.instruments:
            for note in instr.notes:
                if not (0 <= note.pitch <= 127):
                    return False
        return True

    def modify_single_note(self, midi_path, output_path):
        """安全单音修改"""
        try:
            # 加载并克隆
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = self._deep_clone_midi(orig_midi)
            
            # 获取可修改音符（带安全范围过滤）
            candidate_notes = [
                n for instr in mod_midi.instruments 
                for n in instr.notes 
                if SAFE_PITCH_RANGE[0] <= n.pitch <= SAFE_PITCH_RANGE[1]
            ]
            if not candidate_notes:
                return False

            # 随机选择并修改
            target_note = random.choice(candidate_notes)
            original_pitch = target_note.pitch
            new_pitch = self._clamp_pitch(original_pitch + random.choice([-12, 12]))
            
            # 应用修改
            target_note.pitch = new_pitch

            # 最终验证
            if not self._validate_midi(mod_midi):
                raise ValueError("最终验证发现非法音高")

            # 保存
            mod_midi.write(output_path)
            self.log.append({
                'file': os.path.basename(output_path),
                'action': 'single_note',
                'original': original_pitch,
                'modified': new_pitch
            })
            return True
            
        except Exception as e:
            self._log_message(f"处理失败 {midi_path}: {str(e)}", error=True)
            return False

    def modify_voicing(self, midi_path, output_path):
        """安全和弦排列修改"""
        try:
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = self._deep_clone_midi(orig_midi)
            
            # 获取可修改和弦（带安全范围过滤）
            candidate_notes = [
                n for instr in mod_midi.instruments 
                for n in instr.notes 
                if SAFE_PITCH_RANGE[0] <= n.pitch <= SAFE_PITCH_RANGE[1]
            ]
            if len(candidate_notes) < 3:
                return False

            # 随机选择3个音符组成和弦
            chord_notes = random.sample(candidate_notes, 3)
            original_pitches = [n.pitch for n in chord_notes]

            # 执行转位操作
            chord_notes.sort(key=lambda x: x.pitch)
            inversion_type = random.choice(['root', 'first', 'second'])
            
            # 应用转位并限制音高
            if inversion_type == 'first':
                chord_notes[0].pitch = self._clamp_pitch(chord_notes[0].pitch + 12)
            elif inversion_type == 'second':
                chord_notes[1].pitch = self._clamp_pitch(chord_notes[1].pitch + 12)

            # 重新排序并防止声部交叉
            chord_notes.sort(key=lambda x: x.pitch)
            for i in range(1, len(chord_notes)):
                if chord_notes[i].pitch < chord_notes[i-1].pitch + 3:
                    chord_notes[i].pitch = self._clamp_pitch(chord_notes[i-1].pitch + 3)

            # 最终验证
            if not self._validate_midi(mod_midi):
                raise ValueError("最终验证发现非法音高")

            # 保存
            mod_midi.write(output_path)
            self.log.append({
                'file': os.path.basename(output_path),
                'action': 'voicing',
                'original': original_pitches,
                'modified': [n.pitch for n in chord_notes]
            })
            return True
            
        except Exception as e:
            self._log_message(f"处理失败 {midi_path}: {str(e)}", error=True)
            return False

    def process_directory(self, input_dir):
        success = 0
        total = 0
        midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]

        for fname in tqdm(midi_files, desc="处理进度"):
            in_path = os.path.join(input_dir, fname)
            base_name = os.path.splitext(fname)[0]
            total += 2  # 每个文件尝试两种修改

            # 单音修改
            out1 = os.path.join(self.output_dir, f"{base_name}_action1.mid")
            success += int(self.modify_single_note(in_path, out1))

            # 和弦修改
            out2 = os.path.join(self.output_dir, f"{base_name}_action2.mid")
            success += int(self.modify_voicing(in_path, out2))

        # 保存结构化日志
        with open(os.path.join(self.output_dir, "modifications.json"), "w") as f:
            json.dump(self.log, f, indent=2)
        
        self._log_message(f"处理完成！成功率：{success}/{total} ({success/total:.1%})")
        return success

# 使用示例
if __name__ == "__main__":
    processor = MidiModifier(output_dir="./final_output")
    result = processor.process_directory("./selected_midis_cropped")