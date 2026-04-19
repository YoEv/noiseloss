#!/usr/bin/env python3
"""Fix audio_path in musicpref manifests: map original filenames to downloaded MusicPref2025_{i}.wav."""
import os
import pandas as pd
from pathlib import Path

parquets = sorted(Path("phase7_release/raw_hf/MusicPref/data").glob("*.parquet"))
mapping = {}
i = 0
for p in parquets:
    df = pd.read_parquet(p)
    for _, row in df.iterrows():
        orig = row["audio"]["path"]
        mapping[orig] = f"MusicPref2025_{i}.wav"
        i += 1

print(f"Built mapping for {len(mapping)} items")
print("sample:", list(mapping.items())[:4])

audio_dir = "phase7_release/datasets/musicprefs/audio"
for head in ("musicality", "fidelity"):
    csv_path = f"phase7_release/data/manifests/pairwise_relu/musicpref_{head}_1to5.csv"
    df = pd.read_csv(csv_path)
    df["audio_path"] = df["filename"].map(
        lambda fn: os.path.abspath(f"{audio_dir}/{mapping[fn]}") if fn in mapping else ""
    )
    df.to_csv(csv_path, index=False)
    missing = (df["audio_path"] == "").sum()
    print(f"{head}: updated {len(df)} rows, {missing} unmapped")
