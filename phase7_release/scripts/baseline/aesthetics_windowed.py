"""
Audiobox Aesthetics baseline with rater-aligned audio windowing.

Reads split CSVs (score, audio_path, begin_s, end_s).
- Passes start_time/end_time directly to AesPredictor (no temp file trimming).
- Missing audio files are skipped with a warning.
- Outputs one CSV per split: aesthetics_scores_{split}.csv
  columns: score, audio_path, begin_s, end_s, CE, CU, PC, PQ
- Prints Pearson/Spearman per split.

Usage (audiobox conda env):
  python aesthetics_windowed.py --config <paths.yaml> --out-dir <dir>
  python aesthetics_windowed.py --csv <split.csv> --out-dir <dir> --split-name test
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from phase7_release.lib.repro.data_paths import get_exp11_split_paths

AXES = ["CE", "CU", "PC", "PQ"]


def score_split(predictor, df, split_name, batch_size):
    items = []
    skipped = 0
    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        if not os.path.isfile(audio_path):
            skipped += 1
            continue
        begin_s = row.get("begin_s", None)
        end_s = row.get("end_s", None)
        use_window = (
            begin_s is not None
            and end_s is not None
            and np.isfinite(float(begin_s))
            and np.isfinite(float(end_s))
        )
        entry = {"path": audio_path}
        if use_window:
            entry["start_time"] = float(begin_s)
            entry["end_time"] = float(end_s)
        items.append({
            "pred_input": entry,
            "score": float(row["score"]),
            "audio_path": audio_path,
            "begin_s": begin_s,
            "end_s": end_s,
        })

    if skipped:
        print(f"  [{split_name}] Skipped {skipped} rows (missing audio)")
    print(f"  [{split_name}] Scoring {len(items)} clips …")

    all_scores = []
    for i in tqdm(range(0, len(items), batch_size), desc=split_name):
        batch = [it["pred_input"] for it in items[i: i + batch_size]]
        out = predictor.forward(batch)
        for r in out:
            all_scores.append([r[ax] for ax in AXES])

    out_df = pd.DataFrame({
        "score":      [it["score"]      for it in items],
        "audio_path": [it["audio_path"] for it in items],
        "begin_s":    [it["begin_s"]    for it in items],
        "end_s":      [it["end_s"]      for it in items],
    })
    for j, ax in enumerate(AXES):
        out_df[ax] = [s[j] for s in all_scores]
    return out_df


def print_correlations(df, split_name):
    scores = df["score"].values
    for ax in AXES:
        vals = df[ax].values
        mask = np.isfinite(scores) & np.isfinite(vals)
        if mask.sum() < 2:
            continue
        p = pearsonr(scores[mask], vals[mask])[0]
        s = spearmanr(scores[mask], vals[mask])[0]
        print(f"  [{split_name}] {ax}: Pearson={p:.4f}  Spearman={s:.4f}  n={mask.sum()}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str)
    group.add_argument("--csv", type=str)
    parser.add_argument("--split-name", type=str, default="test")
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    from audiobox_aesthetics.infer import AesPredictor
    os.makedirs(args.out_dir, exist_ok=True)
    predictor = AesPredictor(checkpoint_pth=args.ckpt, data_col="path")
    print(f"Loaded AesPredictor")

    if args.config:
        split_paths = get_exp11_split_paths(args.config, splits=args.splits)
        split_map = {name: pd.read_csv(path) for name, path in split_paths.items()}
    else:
        split_map = {args.split_name: pd.read_csv(args.csv)}

    for split_name, df in split_map.items():
        out_df = score_split(predictor, df, split_name, args.batch_size)
        out_path = os.path.join(args.out_dir, f"aesthetics_scores_{split_name}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path} ({len(out_df)} rows)")
        print_correlations(out_df, split_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
