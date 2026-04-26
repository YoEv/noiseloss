"""
Compute Pearson/Spearman correlations between Audiobox Aesthetics scores
and human ground-truth scores, for each dataset's test split.

Ground-truth source:
  reports/<dataset>/clean/f01_loss_only_cnn_clean_test_scores.csv
  (contains Elo-derived continuous scores in the 'score' column)

audio_id extraction:
  musiceval  - stem of audio_path basename  (e.g. 'audiomos2025-track1-S017_P121')
  songeval   - stem of audio_path basename  (e.g. '1403')
  aime       - integer from 'AIME2025_{id}.wav'   (e.g. 5081)
  musicpref  - integer from 'MusicPref2025_{id}.wav' (e.g. 3260)

Requires:
  phase7_release/outputs/aesthetics/<dataset>/aesthetics_scores.csv

Usage:
  python compute_aesthetics_correlations.py
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PH7_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT  = os.path.dirname(PH7_ROOT)

REPORTS_ROOT = os.path.join(REPO_ROOT, "reports")
OUT_BASE     = os.path.join(PH7_ROOT, "outputs", "aesthetics")

AXES = ["CE", "CU", "PC", "PQ"]

REF_CSV = "f01_loss_only_cnn_clean_test_scores.csv"   # any model works as ground-truth source


# ── audio_id extractors ───────────────────────────────────────────────────────

def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]

ID_EXTRACTORS = {
    "musiceval": lambda p: _stem(p),
    "songeval":  lambda p: _stem(p),
    "aime":      lambda p: int(re.search(r"AIME2025_(\d+)", p).group(1)),
    "musicpref": lambda p: int(re.search(r"MusicPref2025_(\d+)", p).group(1)),
}


def load_gt(dataset):
    """Return DataFrame with columns [audio_id, score] for the test split."""
    csv_path = os.path.join(REPORTS_ROOT, dataset, "clean", REF_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    extractor = ID_EXTRACTORS[dataset]
    df["audio_id"] = df["audio_path"].apply(extractor).astype(str)
    return df[["audio_id", "score"]]


# ── Correlation helper ────────────────────────────────────────────────────────

def correlate(gt_df, aes_df, axis):
    gt_df  = gt_df.copy();  gt_df["audio_id"]  = gt_df["audio_id"].astype(str)
    aes_df = aes_df.copy(); aes_df["audio_id"] = aes_df["audio_id"].astype(str)
    merged = gt_df.merge(aes_df[["audio_id", axis]], on="audio_id", how="inner")
    merged = merged.dropna(subset=["score", axis])
    n = len(merged)
    if n < 2:
        return float("nan"), float("nan"), n
    r,   _ = pearsonr( merged["score"], merged[axis])
    rho, _ = spearmanr(merged["score"], merged[axis])
    return round(r, 4), round(rho, 4), n


# ── Main ─────────────────────────────────────────────────────────────────────

DATASETS = ["musiceval", "songeval", "aime", "musicpref"]


def main():
    all_rows = []
    for ds in DATASETS:
        aes_path = os.path.join(OUT_BASE, ds, "aesthetics_scores.csv")
        if not os.path.exists(aes_path):
            print(f"[{ds}] aesthetics_scores.csv not found – skip")
            continue
        try:
            gt_df = load_gt(ds)
        except FileNotFoundError as e:
            print(f"[{ds}] GT not found: {e} – skip")
            continue

        aes_df = pd.read_csv(aes_path)
        print(f"\n[{ds}] GT test rows: {len(gt_df)}, Aesthetics rows: {len(aes_df)}")

        for axis in AXES:
            r, rho, n = correlate(gt_df, aes_df, axis)
            print(f"  Aesthetics-{axis}: Pearson={r:>7.4f}, Spearman={rho:>7.4f}, n={n}")
            all_rows.append({"dataset": ds, "method": f"Aesthetics-{axis}",
                              "pearson": r, "spearman": rho, "n": n})

    if not all_rows:
        print("No results to save.")
        return

    out_df = pd.DataFrame(all_rows)
    os.makedirs(OUT_BASE, exist_ok=True)

    csv_path = os.path.join(OUT_BASE, "correlations_summary.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    md_path = os.path.join(OUT_BASE, "correlations_summary.md")
    with open(md_path, "w") as f:
        f.write("| dataset | method | pearson | spearman | n |\n")
        f.write("|---------|--------|--------:|---------:|--:|\n")
        for _, row in out_df.iterrows():
            f.write(f"| {row['dataset']} | {row['method']} "
                    f"| {row['pearson']} | {row['spearman']} | {row['n']} |\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
