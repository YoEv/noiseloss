import os
import pandas as pd
from pretty_midi import PrettyMIDI, Instrument
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")

ASAP_BASE = "/Volumes/MCastile/Castile/HackerProj/asap-dataset"
OUTPUT_DIR = "./selected_midis_cropped"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_validate_midi(midi_path):
    midi = PrettyMIDI(midi_path)
    if len(midi.instruments) == 0:
        raise ValueError("No instruments found")
    if len(midi.instruments[0].notes) < 10:
        raise ValueError("Too few notes")
    return midi

def crop_midi(midi, start_beat=0, num_bars=32):
    downbeats = midi.get_downbeats()
    if len(downbeats) < num_bars + 1:
        return midi

    start_time = downbeats[start_beat]
    end_time = downbeats[start_beat + num_bars]

    new_midi = PrettyMIDI()
    new_midi.time_signature_changes = [
        ts for ts in midi.time_signature_changes
        if ts.time < end_time
    ]
    new_midi.key_signature_changes = [
        ks for ks in midi.key_signature_changes
        if ks.time < end_time
    ]

    for instr in midi.instruments:
        new_instr = Instrument(
            program=instr.program,
            is_drum=instr.is_drum,
            name=instr.name
        )
        new_instr.notes = [
            note for note in instr.notes
            if note.start < end_time and note.end > start_time
        ]
        if new_instr.notes:
            new_midi.instruments.append(new_instr)

    return new_midi

if __name__ == "__main__":
    meta = pd.read_csv(os.path.join(ASAP_BASE, "metadata.csv"))

    valid_pieces = []
    for _, row in meta.iterrows():
        midi_path = os.path.join(ASAP_BASE, row['midi_score'])
        if load_and_validate_midi(midi_path):
            valid_pieces.append(row)

    selected = random.sample(valid_pieces, min(100, len(valid_pieces)))

    success_count = 0
    for idx, row in enumerate(selected):
        midi = load_and_validate_midi(os.path.join(ASAP_BASE, row['midi_score']))
        if not midi:
            continue

        cropped = crop_midi(midi, num_bars=32)

        composer = row['composer'].replace(" ", "_")
        title = Path(row['midi_score']).stem
        output_path = os.path.join(OUTPUT_DIR, f"{composer}_{idx:03d}_{title}.mid")

        cropped.write(output_path)
        if os.path.exists(output_path):
            success_count += 1
            print(f"Saved: {output_path} ({len(cropped.instruments[0].notes)} notes)")
        else:
            print(f"Failed to save: {output_path}")

    report = {
        "original_count": len(selected),
        "success_count": success_count,
        "output_dir": os.path.abspath(OUTPUT_DIR)
    }
    print("\n" + "="*50)
    print(f"Saved {success_count}/{len(selected)} MIDI files")
    print(f"Output to: {report['output_dir']}")
    print("="*50)
