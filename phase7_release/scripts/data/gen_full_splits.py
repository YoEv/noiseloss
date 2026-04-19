#!/usr/bin/env python3
"""Generate train/val/test splits for all pairwise datasets and SongEval."""
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def make_splits(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["score", "audio_path", "token_loss_path"]
    before = len(df)
    df = df[df["audio_path"].map(os.path.exists)].reset_index(drop=True)
    print(f"  {out_dir.name}: {len(df)}/{before} rows with existing audio")
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    n_tr = int(0.8 * len(df))
    n_va = int(0.1 * len(df))
    splits = {
        "train": df.iloc[perm[:n_tr]],
        "val": df.iloc[perm[n_tr:n_tr + n_va]],
        "test": df.iloc[perm[n_tr + n_va:]],
    }
    for k, part in splits.items():
        part[cols].to_csv(out_dir / f"{k}.csv", index=False)
        part[cols].to_csv(out_dir / f"{k}_noisy.csv", index=False)
        print(f"    {k}: {len(part)}")


def main():
    # AIME: average music_quality + text_audio_alignment
    mq = pd.read_csv("phase7_release/data/manifests/pairwise_relu/aime_music_quality_1to5.csv")
    ta = pd.read_csv("phase7_release/data/manifests/pairwise_relu/aime_text_audio_alignment_1to5.csv")
    aime = pd.DataFrame({
        "score": (mq["score_1to5"].values + ta["score_1to5"].values) / 2,
        "audio_path": mq["track_id"].map(
            lambda x: os.path.abspath(f"phase7_release/datasets/aime/audio/AIME2025_{x}.wav")
        ),
        "token_loss_path": mq["track_id"].map(lambda x: f"aime/AIME2025_{x}"),
    })
    make_splits(aime, "phase7_release/data/full_splits/aime")

    # MusicPref: average musicality + fidelity
    mu = pd.read_csv("phase7_release/data/manifests/pairwise_relu/musicpref_musicality_1to5.csv")
    fi = pd.read_csv("phase7_release/data/manifests/pairwise_relu/musicpref_fidelity_1to5.csv")
    mp = pd.DataFrame({
        "score": (mu["score_1to5"].values + fi["score_1to5"].values) / 2,
        "audio_path": mu["audio_path"].map(os.path.abspath),
        "token_loss_path": mu["filename"].map(lambda x: f"musicpref/{x}"),
    })
    make_splits(mp, "phase7_release/data/full_splits/musicpref")

    # MusicArena
    ma = pd.read_csv("phase7_release/data/manifests/pairwise_relu/musicarena_1to5.csv")
    ma_df = pd.DataFrame({
        "score": ma["score_1to5"].values,
        "audio_path": ma["audio_path"].map(os.path.abspath),
        "token_loss_path": ma["audio_path"].map(lambda p: f"music_arena/{Path(p).stem}"),
    })
    make_splits(ma_df, "phase7_release/data/full_splits/music_arena")

    # SongEval: mean of 5 dims across all annotators
    rows = []
    with open("phase7_release/raw_hf/SongEval/metadata.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fname = Path(r["file_name"]).stem
            dims = [
                a[k]
                for a in r["annotation"]
                for k in ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]
            ]
            rows.append({
                "score": sum(dims) / len(dims),
                "audio_path": os.path.abspath(f"phase7_release/datasets/songeval/audio/{fname}.wav"),
                "token_loss_path": f"songeval/{fname}",
            })
    make_splits(pd.DataFrame(rows), "phase7_release/data/full_splits/songeval")

    print("Done.")


if __name__ == "__main__":
    main()
