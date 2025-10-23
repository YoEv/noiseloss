import os
import random
import copy
import json
from datetime import datetime
from pretty_midi import PrettyMIDI, note_number_to_name
from tqdm import tqdm

# 常量定义
MAX_NOTE_GAP = 0.3
MIN_CHORD_NOTES = 3
DIATONIC_SCALE = [0,2,4,5,7,9,11]
CHORD_TONES = [0,4,7]
SAFE_PITCH_RANGE = (24, 103)  # 新增安全音高范围

class MidiModifier:
    def __init__(self):
        self.log = []
        self.error_log = []
        self.current_process = ""  # 新增当前处理状态记录

    def _save_error(self, fname, error, action):
        """统一错误记录方法"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "file": fname,
            "action": action,
            "error": str(error)
        }
        self.error_log.append(error_entry)

    def _clamp_pitch(self, pitch):
        """音高安全钳制方法"""
        return max(0, min(127, pitch))

    def _deep_clone_midi(self, midi):
        """改进的深克隆方法"""
        new_midi = PrettyMIDI()
        new_midi.time_signature_changes = copy.deepcopy(midi.time_signature_changes)
        new_midi.key_signature_changes = copy.deepcopy(midi.key_signature_changes)
        new_midi.lyrics = copy.deepcopy(midi.lyrics)
        for instr in midi.instruments:
            new_instr = copy.deepcopy(instr)
            # 深度复制每个音符
            new_instr.notes = [copy.deepcopy(n) for n in instr.notes]
            new_midi.instruments.append(new_instr)
        return new_midi

    def _cluster_chords(self, midi):
        """带音高过滤的和弦检测"""
        chords = []
        notes = [n for n in midi.instruments[0].notes 
                if SAFE_PITCH_RANGE[0] <= n.pitch <= SAFE_PITCH_RANGE[1]]  # 新增音高过滤
        current_chord = []
        for note in sorted(notes, key=lambda x: x.start):
            if current_chord and (note.start - current_chord[-1].start) > MAX_NOTE_GAP:
                if len(current_chord) >= MIN_CHORD_NOTES:
                    chords.append(current_chord)
                current_chord = []
            current_chord.append(note)
        if len(current_chord) >= MIN_CHORD_NOTES:
            chords.append(current_chord)
        return chords

    def _validate_pitch(self, midi, original_pitch, new_pitch, chord_root):
        """带音高钳制的验证方法"""
        new_pitch = self._clamp_pitch(new_pitch)  # 新增钳制
        key = midi.key_signature_changes[0].key_number % 12 if midi.key_signature_changes else 0
        scale_degrees = [(key + offset) % 12 for offset in DIATONIC_SCALE]
        chord_degree = (new_pitch - chord_root) % 12
        return (new_pitch % 12) in scale_degrees or chord_degree in CHORD_TONES

    def modify_single_note(self, midi_path, output_path):
        self.current_process = f"正在处理单音修改: {os.path.basename(midi_path)}"
        try:
            orig_midi = PrettyMIDI(midi_path)
            if not orig_midi.instruments:
                return False

            mod_midi = self._deep_clone_midi(orig_midi)
            chords = self._cluster_chords(mod_midi)
            if not chords:
                return False

            selected_chord = random.choice(chords)
            chord_root = min(n.pitch for n in selected_chord) % 12
            target_note = random.choice(selected_chord)
            original_pitch = target_note.pitch

            valid_modified = False
            for _ in range(40):
                new_pitch = self._clamp_pitch(original_pitch + random.choice([-12, 12]))
                if self._validate_pitch(mod_midi, original_pitch, new_pitch, chord_root):
                    target_note.pitch = new_pitch
                    valid_modified = True
                    break

            if valid_modified:
                # 最终音高验证
                for note in mod_midi.instruments[0].notes:
                    if not (0 <= note.pitch <= 127):
                        raise ValueError(f"非法音高值: {note.pitch}")

                mod_midi.write(output_path)
                self.log.append({
                    'file': os.path.basename(output_path),
                    'action': 'single_note',
                    'time': target_note.start,
                    'original': f"{note_number_to_name(original_pitch)} ({original_pitch})",
                    'modified': f"{note_number_to_name(new_pitch)} ({new_pitch})",
                    'status': 'success'
                })
                return True
            return False
        except Exception as e:
            self._save_error(os.path.basename(midi_path), e, 'single_note')
            return False

    def modify_voicing(self, midi_path, output_path):
        self.current_process = f"正在处理和弦排列: {os.path.basename(midi_path)}"
        try:
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = self._deep_clone_midi(orig_midi)
            chords = self._cluster_chords(mod_midi)
            if not chords:
                return False

            selected_chord = random.choice(chords)
            chord_notes = sorted(selected_chord, key=lambda x: x.pitch)
            if len(chord_notes) < 3:
                return False

            original_pitches = [n.pitch for n in chord_notes]
            inversion_type = random.choice(['root', 'first', 'second'])

            # 应用音高钳制
            if inversion_type == 'first':
                chord_notes[0].pitch = self._clamp_pitch(chord_notes[0].pitch + 12)
            elif inversion_type == 'second' and len(chord_notes) > 1:
                chord_notes[1].pitch = self._clamp_pitch(chord_notes[1].pitch + 12)

            # 防止声部交叉并钳制音高
            chord_notes.sort(key=lambda x: x.pitch)
            for i in range(1, len(chord_notes)):
                if chord_notes[i].pitch < chord_notes[i - 1].pitch + 3:
                    chord_notes[i].pitch = self._clamp_pitch(chord_notes[i - 1].pitch + 3)

            # 最终音高验证
            for note in mod_midi.instruments[0].notes:
                if not (0 <= note.pitch <= 127):
                    raise ValueError(f"非法音高值: {note.pitch}")

            mod_midi.write(output_path)
            self.log.append({
                'file': os.path.basename(output_path),
                'action': 'voicing',
                'time': chord_notes[0].start,
                'original': [note_number_to_name(p) for p in original_pitches],
                'modified': [note_number_to_name(n.pitch) for n in chord_notes],
                'status': 'success'
            })
            return True
        except Exception as e:
            self._save_error(os.path.basename(midi_path), e, 'voicing')
            return False

    def _save_logs(self, output_dir):
        """带时间戳的日志保存方法"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            'process_info': {
                'start_time': timestamp,
                'total_files': len(self.log) + len(self.error_log),
                'success_rate': f"{len(self.log)/(len(self.log)+len(self.error_log)):.1%}"
            },
            'modifications': self.log,
            'errors': self.error_log
        }
        error_file = os.path.join(output_dir, f"error_log_{timestamp}.json")
        log_path = os.path.join(output_dir, f"process_log_{timestamp}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(self.error_log, f, ensure_ascii=False, indent=2)

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
        success = 0
        total_attempts = 0

        for fname in midi_files:  # 移除tqdm以消除终端输出
            in_path = os.path.join(input_dir, fname)
            base_name = os.path.splitext(fname)[0]

            # 保存原始文件
            orig_out = os.path.join(output_dir, f"{base_name}_original.mid")
            try:
                orig_midi = PrettyMIDI(in_path)
                orig_midi.write(orig_out)
            except Exception as e:
                self._save_error(fname, e, 'original_save')
                continue

            # 单音修改尝试
            out1 = os.path.join(output_dir, f"{base_name}_action1.mid")
            try:
                if self.modify_single_note(in_path, out1):
                    success += 1
                total_attempts += 1
            except Exception as e:
                self.error_log.append({
                    'file': fname,
                    'error': str(e),
                    'action': 'single_note'
                })


            # 和弦修改尝试
            out2 = os.path.join(output_dir, f"{base_name}_action2.mid")
            try:
                if self.modify_voicing(in_path, out2):
                    success += 1
                total_attempts += 1
            except Exception as e:
                self.error_log.append({
                    'file': fname,
                    'error': str(e),
                    'action': 'voicing'
                })
        print(f"\n✅ 成功修改 {success}/{total_attempts} 个文件")
        self._save_logs(output_dir)
        return self.log

if __name__ == "__main__":
    modifier = MidiModifier()
    modifier.process_directory(
        input_dir="./selected_midis_cropped",
        output_dir="./final_modified_output_test1"
    )