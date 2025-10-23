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
SAFE_PITCH_RANGE = (24, 103)

class MidiModifier:
    def __init__(self):
        self.log = []
        self.error_log = []
        self.current_process = ""
        self.processed_files = 0

    def _save_error(self, fname, error, action):
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "file": fname,
            "action": action,
            "error": str(error),
            "details": {
                "processing_stage": self.current_process,
                "file_count": self.processed_files
            }
        }
        self.error_log.append(error_entry)

    def _clamp_pitch(self, pitch):
        return max(0, min(127, pitch))

    def _deep_clone_midi(self, midi):
        new_midi = PrettyMIDI()
        new_midi.time_signature_changes = copy.deepcopy(midi.time_signature_changes)
        new_midi.key_signature_changes = copy.deepcopy(midi.key_signature_changes)
        new_midi.lyrics = copy.deepcopy(midi.lyrics)
        for instr in midi.instruments:
            new_instr = copy.deepcopy(instr)
            new_instr.notes = [copy.deepcopy(n) for n in instr.notes]
            new_midi.instruments.append(new_instr)
        return new_midi

    def _cluster_chords(self, midi):
        """改进版和弦检测：支持多轨道合并分析"""
        # 合并所有轨道的有效音符
        all_notes = []
        for instr in midi.instruments:
            valid_notes = [n for n in instr.notes 
                          if SAFE_PITCH_RANGE[0] <= n.pitch <= SAFE_PITCH_RANGE[1]]
            all_notes.extend(valid_notes)
        
        # 时间窗口检测算法改进
        chords = []
        sorted_notes = sorted(all_notes, key=lambda x: x.start)
        current_window = []
        last_end = -1
        
        for note in sorted_notes:
            # 检测时间重叠或邻近
            if current_window and note.start > max(n.end for n in current_window) + MAX_NOTE_GAP:
                if len(current_window) >= MIN_CHORD_NOTES:
                    chords.append(current_window)
                current_window = []
            
            # 检测时间连续性
            if last_end < 0 or note.start - last_end <= MAX_NOTE_GAP:
                current_window.append(note)
                last_end = max(last_end, note.end)
            else:
                if len(current_window) >= MIN_CHORD_NOTES:
                    chords.append(current_window)
                current_window = [note]
                last_end = note.end
        
        if len(current_window) >= MIN_CHORD_NOTES:
            chords.append(current_window)
        return chords

    def _validate_pitch(self, midi, original_pitch, new_pitch, chord_root):
        new_pitch = self._clamp_pitch(new_pitch)
        key = midi.key_signature_changes[0].key_number % 12 if midi.key_signature_changes else 0
        scale_degrees = [(key + offset) % 12 for offset in DIATONIC_SCALE]
        chord_degree = (new_pitch - chord_root) % 12
        return (new_pitch % 12) in scale_degrees or chord_degree in CHORD_TONES

    def modify_single_note(self, midi_path, output_path):
        self.current_process = f"Processing: {os.path.basename(midi_path)}"
        try:
            orig_midi = PrettyMIDI(midi_path)
            if not orig_midi.instruments:
                self._save_error(os.path.basename(midi_path), "No instruments found", 'single_note')
                return False

            mod_midi = self._deep_clone_midi(orig_midi)
            chords = self._cluster_chords(mod_midi)
            if not chords:
                self._save_error(os.path.basename(midi_path), "No valid chords detected", 'single_note')
                return False

            selected_chord = random.choice([c for c in chords if len(c) >= 3])
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
            self._save_error(os.path.basename(midi_path), "No valid pitch after 40 attempts", 'single_note')
            return False
        except Exception as e:
            self._save_error(os.path.basename(midi_path), e, 'single_note')
            return False

    def modify_voicing(self, midi_path, output_path):
        self.current_process = f"Processing: {os.path.basename(midi_path)}"
        try:
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = self._deep_clone_midi(orig_midi)
            chords = self._cluster_chords(mod_midi)
            if not chords:
                self._save_error(os.path.basename(midi_path), "No valid chords detected", 'voicing')
                return False

            selected_chord = random.choice([c for c in chords if len(c) >= 3])
            chord_notes = sorted(selected_chord, key=lambda x: x.pitch)
            original_pitches = [n.pitch for n in chord_notes]
            inversion_type = random.choice(['root', 'first', 'second'])

            if inversion_type == 'first':
                chord_notes[0].pitch = self._clamp_pitch(chord_notes[0].pitch + 12)
            elif inversion_type == 'second' and len(chord_notes) > 1:
                chord_notes[1].pitch = self._clamp_pitch(chord_notes[1].pitch + 12)

            chord_notes.sort(key=lambda x: x.pitch)
            for i in range(1, len(chord_notes)):
                if chord_notes[i].pitch < chord_notes[i-1].pitch + 3:
                    chord_notes[i].pitch = self._clamp_pitch(chord_notes[i-1].pitch + 3)

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            'process_info': {
                'start_time': timestamp,
                'total_files': len(self.log) + len(self.error_log),
                'success_rate': f"{len(self.log)/(len(self.log)+len(self.error_log)):.1%}",
                'error_details': {
                    'no_instruments': sum(1 for e in self.error_log if "No instruments" in e['error']),
                    'no_chords': sum(1 for e in self.error_log if "No valid chords" in e['error']),
                    'invalid_pitch': sum(1 for e in self.error_log if "invalid pitch" in str(e['error']))
                }
            },
            'modifications': self.log,
            'errors': self.error_log
        }
        log_path = os.path.join(output_dir, f"process_log_{timestamp}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
        success = 0
        total = len(midi_files) * 2

        with tqdm(total=total, desc="Processing Files", unit="op") as pbar:
            for fname in midi_files:
                self.processed_files += 1
                in_path = os.path.join(input_dir, fname)
                base_name = os.path.splitext(fname)[0]

                # 保存原始文件
                orig_out = os.path.join(output_dir, f"{base_name}_original.mid")
                try:
                    PrettyMIDI(in_path).write(orig_out)
                except Exception as e:
                    self._save_error(fname, e, 'original_save')
                    pbar.update(2)
                    continue

                # 单音修改
                out1 = os.path.join(output_dir, f"{base_name}_action1.mid")
                if self.modify_single_note(in_path, out1):
                    success += 1
                pbar.update(1)

                # 和弦修改
                out2 = os.path.join(output_dir, f"{base_name}_action2.mid")
                if self.modify_voicing(in_path, out2):
                    success += 1
                pbar.update(1)

        print(f"\nSuccess: {success}/{total} | Errors: {len(self.error_log)}")
        self._save_logs(output_dir)
        return self.log

if __name__ == "__main__":
    modifier = MidiModifier()
    modifier.process_directory(
        input_dir="./selected_midis_cropped",
        output_dir="./final_modified_output"
    )