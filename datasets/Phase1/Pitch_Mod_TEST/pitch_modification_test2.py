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
DIATONIC_SCALE = [0,2,4,5,7,9,11]
CHORD_TONES = [0,4,7]

class MidiModifier:
    def __init__(self):
        self.log = []
        self.error_log = []

    def _deep_clone_midi(self, midi):
        """彻底深克隆MIDI对象"""
        new_midi = PrettyMIDI()
        new_midi.time_signature_changes = copy.deepcopy(midi.time_signature_changes)
        new_midi.key_signature_changes = copy.deepcopy(midi.key_signature_changes)
        new_midi.lyrics = copy.deepcopy(midi.lyrics)
        for instr in midi.instruments:
            new_instr = copy.deepcopy(instr)
            new_midi.instruments.append(new_instr)
        return new_midi

    def _cluster_chords(self, midi):
        """分析和弦簇"""
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
        """判断音高是否合法"""
        key = midi.key_signature_changes[0].key_number % 12 if midi.key_signature_changes else 0
        scale_degrees = [(key + offset) % 12 for offset in DIATONIC_SCALE]
        chord_degree = (new_pitch - chord_root) % 12
        return (new_pitch % 12) in scale_degrees or chord_degree in CHORD_TONES

    def _get_editable_notes(self, midi):
        return [note for instr in midi.instruments for note in instr.notes]

    def modify_single_note(self, midi_path, output_path):
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

            for _ in range(20):
                new_pitch = original_pitch + random.choice([-12, 12])
                if self._validate_pitch(mod_midi, original_pitch, new_pitch, chord_root):
                    target_note.pitch = new_pitch
                    self.log.append({
                        'file': os.path.basename(output_path),
                        'action': 'single_note',
                        'time': target_note.start,
                        'original': f"{note_number_to_name(original_pitch)} ({original_pitch})",
                        'modified': f"{note_number_to_name(new_pitch)} ({new_pitch})"
                    })
                    mod_midi.write(output_path)
                    return True
            return False
        except Exception as e:
            print(f"处理错误 {midi_path}: {str(e)}")
            return False

    def modify_voicing(self, midi_path, output_path):
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

            if inversion_type == 'first':
                chord_notes[0].pitch += 12
            elif inversion_type == 'second' and len(chord_notes) > 1:
                chord_notes[1].pitch += 12

            chord_notes.sort(key=lambda x: x.pitch)
            for i in range(1, len(chord_notes)):
                if chord_notes[i].pitch < chord_notes[i - 1].pitch + 3:
                    chord_notes[i].pitch = chord_notes[i - 1].pitch + 3

            self.log.append({
                'file': os.path.basename(output_path),
                'action': 'voicing',
                'time': chord_notes[0].start,
                'original': [note_number_to_name(p) for p in original_pitches],
                'modified': [note_number_to_name(n.pitch) for n in chord_notes]
            })
            mod_midi.write(output_path)
            return True
        except Exception as e:
            print(f"处理错误 {midi_path}: {str(e)}")
            return False

    def _save_logs(self, output_dir):
        """保存日志到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"modification_log_{timestamp}.json")
        error_file = os.path.join(output_dir, f"error_log_{timestamp}.json")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, ensure_ascii=False, indent=2)
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(self.error_log, f, ensure_ascii=False, indent=2)

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
        success = 0
        total_attempts = 0

        for fname in tqdm(midi_files, desc="批量处理中"):
            in_path = os.path.join(input_dir, fname)
            base_name = os.path.splitext(fname)[0]

            # 保存原始文件
            orig_out = os.path.join(output_dir, f"{base_name}_original.mid")
            try:
                orig_midi = PrettyMIDI(in_path)
                orig_midi.write(orig_out)
            except Exception as e:
                self.error_log.append({
                    'file': fname,
                    'error': str(e),
                    'action': 'original_save'
                })
                continue

            # 尝试修改单音
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

            # 尝试修改和弦排列
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

# 示例调用
if __name__ == "__main__":
    modifier = MidiModifier()
    logs = modifier.process_directory(
        input_dir="./selected_midis_cropped",
        output_dir="./final_modified_output"
    )

    print("\n🎵 === 修改日志 ===")
    for entry in logs:
        print(f"\n文件: {entry['file']}")
        print(f"时间: {entry['time']:.2f}s")
        if entry['action'] == 'single_note':
            print(f"  单音修改: {entry['original']} → {entry['modified']}")
        else:
            print(f"  和弦排列修改:")
            print(f"    原始: {entry['original']}")
            print(f"    修改: {entry['modified']}")
