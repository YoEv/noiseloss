import os
import pandas as pd
from pretty_midi import PrettyMIDI
import random
from pathlib import Path

ASAP_BASE = "/Volumes/MCastile/Castile/HackerProj/asap-dataset"
OUTPUT_DIR = "./selected_midis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

meta = pd.read_csv(os.path.join(ASAP_BASE, "metadata.csv"))

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
        print(f"Skip {row['midi_score']} (error: {str(e)})")
        continue

selected = random.sample(good_pieces, min(100, len(good_pieces)))

for i, row in enumerate(selected):
    src_midi = os.path.join(ASAP_BASE, row['midi_score'])
    
    composer_clean = row['composer'].replace(" ", "_")
    output_name = f"{composer_clean}_{i:03d}_{os.path.basename(src_midi)}"
    dest_midi = os.path.join(OUTPUT_DIR, output_name)
    
    # import shutil
    # shutil.copy2(src_midi, dest_midi)
    
    midi = PrettyMIDI(src_midi)
    
    downbeats = midi.get_downbeats()
    end_time = downbeats[32] if len(downbeats) > 32 else midi.get_end_time()
    
    cropped_midi = PrettyMIDI()
    for instrument in midi.instruments:
        new_instr = instrument.__class__(program=instrument.program)
        new_instr.notes = [n for n in instrument.notes if n.start < end_time]
        cropped_midi.instruments.append(new_instr)
    
    cropped_midi.write(dest_midi)
    print(f"Saved: {output_name} (Length: {end_time:.2f}s)")

print(f"\nComplete！Saved {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mid')])} MIDI to {OUTPUT_DIR}")
