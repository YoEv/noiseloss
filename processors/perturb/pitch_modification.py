import os
import random
import copy
import json
from datetime import datetime
from pretty_midi import PrettyMIDI, note_number_to_name
from tqdm import tqdm

MAX_NOTE_GAP = 0.3
MIN_CHORD_NOTES = 3
DIATONIC_SCALE = [0,2,4,5,7,9,11]
CHORD_TONES = [0,4,7]
SAFE_PITCH_RANGE = (24, 103)
TRACK_SELECTION_THRESHOLD = 5

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
                "file_count": self.processed_files,
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

    def _get_main_track(self, midi):
        return max(midi.instruments, key=lambda x: len(x.notes), default=None)

    def _cluster_chords(self, midi):
        main_track = self._get_main_track(midi)
        if not main_track:
            return []

        all_notes = []
        for instr in midi.instruments:
            all_notes.extend([n for n in instr.notes
                            if SAFE_PITCH_RANGE[0] <= n.pitch <= SAFE_PITCH_RANGE[1]])

        chords = []
        current_chord = []
        for note in sorted(all_notes, key=lambda x: x.start):
            if current_chord:
                time_gap = note.start - current_chord[-1].start
                pitch_gap = abs(note.pitch - current_chord[-1].pitch)

                if time_gap > MAX_NOTE_GAP or pitch_gap > 12:
                    if len(current_chord) >= MIN_CHORD_NOTES:
                        chords.append(current_chord)
                    current_chord = []
            current_chord.append(note)

        if len(current_chord) >= MIN_CHORD_NOTES:
            chords.append(current_chord)

        if not chords:
            temp_chords = []
            time_windows = [n.start for n in all_notes]
            for t in time_windows:
                window_notes = [n for n in all_notes
                            if t - 0.1 <= n.start <= t + 0.1]
                if len(window_notes) >= MIN_CHORD_NOTES:
                    temp_chords.append(window_notes)
            chords = [c for c in temp_chords if len(c) >= MIN_CHORD_NOTES]

        return chords

    def _find_dense_region(self, chords):
        if not chords:
            return []

        chord_durations = [(c[0].start, c[-1].end) for c in chords]

        regions = []
        current_start, current_end = chord_durations[0]
        for start, end in chord_durations[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                regions.append((current_start, current_end))
                current_start, current_end = start, end
        regions.append((current_start, current_end))

        longest = max(regions, key=lambda x: x[1]-x[0])
        return [c for c in chords if c[0].start >= longest[0] and c[-1].end <= longest[1]]

    def _validate_pitch(self, midi, original_pitch, new_pitch, chord_root):
        new_pitch = self._clamp_pitch(new_pitch)
        key = midi.key_signature_changes[0].key_number % 12 if midi.key_signature_changes else 0
        scale_degrees = [(key + offset) % 12 for offset in DIATONIC_SCALE]
        chord_degree = (new_pitch - chord_root) % 12
        return ((new_pitch % 12) in scale_degrees) or \
            (chord_degree in CHORD_TONES) or \
            (abs(new_pitch - original_pitch) % 12 == 0)

    def modify_single_note(self, midi_path, output_path):
        self.current_process = f"Single note modification: {os.path.basename(midi_path)}"
        try:
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = self._deep_clone_midi(orig_midi)

            all_chords = self._cluster_chords(mod_midi)
            dense_chords = self._find_dense_region(all_chords)

            selected_chord = None
            if dense_chords:
                selected_chord = random.choice(dense_chords)
            elif all_chords:
                selected_chord = random.choice(all_chords)
            else:
                all_notes = sorted(mod_midi.instruments[0].notes, key=lambda x: x.start)
                last_notes = all_notes[-3:]
                if len(last_notes) >= 3:
                    selected_chord = last_notes
                else:
                    self._save_error(os.path.basename(midi_path), "Final fallback failed", 'single_note')
                    return False

            if not orig_midi.instruments:
                self._save_error(os.path.basename(midi_path), "No instrument tracks found", 'single_note')
                return False

            chord_root = min(n.pitch for n in selected_chord) % 12
            target_note = random.choice(selected_chord)
            original_pitch = target_note.pitch

            valid_modified = False
            for _ in range(60):
                new_pitch = self._clamp_pitch(original_pitch + random.choice([-12, 12]))
                if self._validate_pitch(mod_midi, original_pitch, new_pitch, chord_root):
                    target_note.pitch = new_pitch
                    valid_modified = True
                    break

            if valid_modified:
                for note in mod_midi.instruments[0].notes:
                    if not (0 <= note.pitch <= 127):
                        raise ValueError(f"Invalid pitch value: {note.pitch}")

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
            self._save_error(os.path.basename(midi_path), "No valid pitch found after 60 attempts", 'single_note')
            return False
        except Exception as e:
            self._save_error(os.path.basename(midi_path), e, 'single_note')
            print(f"❌ {os.path.basename(midi_path)} single note modification failed: {str(e)}")
            return False

    def modify_voicing(self, midi_path, output_path):
        self.current_process = f"Voicing modification: {os.path.basename(midi_path)}"
        try:
            orig_midi = PrettyMIDI(midi_path)
            mod_midi = self._deep_clone_midi(orig_midi)
            chords = self._cluster_chords(mod_midi)

            if not chords:
                self._save_error(os.path.basename(midi_path), "No valid chords detected", 'voicing')
                return False

            selected_chord = random.choice(chords)
            chord_notes = sorted(selected_chord, key=lambda x: x.pitch)
            if len(chord_notes) < 3:
                self._save_error(os.path.basename(midi_path), "Not enough chord notes (min 3)", 'voicing')
                return False

            original_pitches = [n.pitch for n in chord_notes]
            inversion_type = random.choice(['root', 'first', 'second'])

            if inversion_type == 'first':
                chord_notes[0].pitch = self._clamp_pitch(chord_notes[0].pitch + 12)
            elif inversion_type == 'second' and len(chord_notes) > 1:
                chord_notes[1].pitch = self._clamp_pitch(chord_notes[1].pitch + 12)

            chord_notes.sort(key=lambda x: x.pitch)
            for i in range(1, len(chord_notes)):
                if chord_notes[i].pitch < chord_notes[i - 1].pitch + 3:
                    chord_notes[i].pitch = self._clamp_pitch(chord_notes[i - 1].pitch + 3)

            for note in mod_midi.instruments[0].notes:
                if not (0 <= note.pitch <= 127):
                    raise ValueError(f"Invalid pitch value: {note.pitch}")

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
            print(f"❌ {os.path.basename(midi_path)} voicing modification failed: {str(e)}")
            return False

    def _save_logs(self, output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            'process_info': {
                'start_time': timestamp,
                'total_files': len(self.log) + len(self.error_log),
                'success_rate': f"{len(self.log)/(len(self.log)+len(self.error_log)):.1%}",
                'error_details': {
                    'no_instruments': sum(1 for e in self.error_log if "No instrument tracks found" in e['error']),
                    'no_chords': sum(1 for e in self.error_log if "No valid chords detected" in e['error']),
                    'invalid_pitch': sum(1 for e in self.error_log if "Invalid pitch value" in e['error'])
                }
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
        total = len(midi_files) * 2
        total_attempts = 0

        with tqdm(total=total, desc="Processing Files", unit="op") as pbar:
            for fname in midi_files:
                self.processed_files += 1
                in_path = os.path.join(input_dir, fname)
                base_name = os.path.splitext(fname)[0]

                orig_out = os.path.join(output_dir, f"{base_name}_original.mid")
                try:
                    orig_midi = PrettyMIDI(in_path)
                    orig_midi.write(orig_out)
                except Exception as e:
                    self._save_error(fname, e, 'original_save')
                    pbar.update(2)
                    continue

                out1 = os.path.join(output_dir, f"{base_name}_action1.mid")
                if self.modify_single_note(in_path, out1):
                    success += 1
                pbar.update(1)

                out2 = os.path.join(output_dir, f"{base_name}_action2.mid")
                if self.modify_voicing(in_path, out2):
                    success += 1
                pbar.update(1)

                pbar.set_postfix(file=fname[:15], success=success, errors=len(self.error_log))

        print(f"\nSuccess: {success}/{total} | Errors: {len(self.error_log)}")
        print(f"\n✅ Successfully modified {success}/{total_attempts} files")
        print(f"⚠️ Failed operations: {len(self.error_log)}")
        self._save_logs(output_dir)
        return self.log

if __name__ == "__main__":
    modifier = MidiModifier()
    modifier.process_directory(
        input_dir="./selected_midis_cropped",
        output_dir="./final_modified_output_test4_evev_ssh"
    )