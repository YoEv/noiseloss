#!/usr/bin/env python3
"""
Fit per-track scores from non-numeric pairwise labels using hinge (ReLU) + tie loss,
then map to [1, 5] (MusicEval-style range). One independent run per --dataset/--head.

Does NOT touch datasets that are already MOS (MusicEval, SongEval per-dimension CSVs).

Examples:
  python fit_pairwise_manifests.py --dataset musicpref --head musicality \\
    --human-preference-csv raw_hf/MusicPref/human_preference.csv \\
    --out data/manifests/pairwise_relu/musicpref_musicality_1to5.csv

  python fit_pairwise_manifests.py --dataset aime --head music_quality \\
    --survey-parquet raw_hf/AIME-survey/data/train-00000-of-00001.parquet \\
    --out data/manifests/pairwise_relu/aime_music_quality_1to5.csv

  python fit_pairwise_manifests.py --dataset musicarena \\
    --battle-glob 'raw_hf/MusicArena/battle_data/**/*.json' \\
    --out data/manifests/pairwise_relu/musicarena_1to5.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# repo root = parent of phase7_release
_REPO = Path(__file__).resolve().parents[3]
_LIB = Path(__file__).resolve().parents[2] / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from pairwise_relu_scores import Outcome, Pair, fit_pairwise_scores, minmax_map_to_range


def _parse_musicpref_column(val: str) -> Outcome:
    m = str(val).strip().lower()
    if "a_wins" in m and "b_wins" not in m:
        return Outcome.WIN_A
    if "b_wins" in m:
        return Outcome.WIN_B
    return Outcome.TIE


def load_musicpref_pairs(csv_path: Path, head: str) -> List[Pair]:
    col = "musicality" if head == "musicality" else "fidelity"
    df = pd.read_csv(csv_path)
    pairs: List[Pair] = []
    for _, row in df.iterrows():
        a = str(row["audio_a"]).strip()
        b = str(row["audio_b"]).strip()
        o = _parse_musicpref_column(str(row[col]))
        pairs.append(Pair(a, b, o))
    return pairs


def load_aime_pairs(parquet_path: Path, head: str) -> List[Pair]:
    """head: music_quality | text_audio_alignment (matches question-type strings)."""
    want = "Music Quality" if head == "music_quality" else "Text-Audio Alignment"
    df = pd.read_parquet(parquet_path)
    # normalize column names (parquet may have hyphens)
    rename = {c: c.replace("-", "_") for c in df.columns}
    df = df.rename(columns=rename)
    qcol = "question_type" if "question_type" in df.columns else None
    if qcol is None:
        raise ValueError("Expected question_type / question-type in survey parquet")
    pairs: List[Pair] = []
    for _, row in df.iterrows():
        if str(row[qcol]).strip() != want:
            continue
        t1 = int(row["track_1_id"])
        t2 = int(row["track_2_id"])
        a = f"{t1:05d}"
        b = f"{t2:05d}"
        ans = int(row["answer"])
        o = Outcome.WIN_A if ans == 1 else Outcome.WIN_B
        pairs.append(Pair(a, b, o))
    return pairs


def load_musicarena_pairs(
    battle_glob: str,
    *,
    repo_root: Path,
    raw_prefix: str = "phase7_release/raw_hf/MusicArena",
) -> List[Pair]:
    """
    preference: A -> win A, B -> win B, TIE / BOTH_BAD -> tie loss (close scores).
    Item id = posix path under repo: raw_prefix + '/' + audio path from JSON.
    Skips rows with empty audio_a or audio_b.
    """
    paths = sorted(glob.glob(battle_glob, recursive=True))
    pairs: List[Pair] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            obj = json.load(f)
        pa = (obj.get("audio_a") or "").strip()
        pb = (obj.get("audio_b") or "").strip()
        if not pa or not pb:
            continue
        pref = str(obj.get("preference", "")).strip().upper()
        if pref == "A":
            o = Outcome.WIN_A
        elif pref == "B":
            o = Outcome.WIN_B
        elif pref in ("TIE", "BOTH_BAD"):
            o = Outcome.TIE
        else:
            continue
        ida = f"{raw_prefix}/{pa}".replace("\\", "/")
        idb = f"{raw_prefix}/{pb}".replace("\\", "/")
        pairs.append(Pair(ida, idb, o))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset",
        choices=("musicpref", "aime", "musicarena"),
        required=True,
    )
    ap.add_argument(
        "--head",
        choices=("musicality", "fidelity", "music_quality", "text_audio_alignment"),
        default=None,
        help="Required for musicpref / aime; omitted for musicarena",
    )
    ap.add_argument("--human-preference-csv", type=Path, help="MusicPref human_preference.csv")
    ap.add_argument("--survey-parquet", type=Path, help="AIME-survey parquet")
    ap.add_argument("--battle-glob", type=str, help="Glob for MusicArena battle JSON (quote for **)")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path (under phase7_release recommended)")
    ap.add_argument("--project-root", type=Path, default=None, help="Default: repo root inferred from script")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--squared-hinge", action="store_true")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low", type=float, default=1.0, help="Min after min–max map (MusicEval-style floor)")
    ap.add_argument("--high", type=float, default=5.0, help="Max after min–max map")
    ap.add_argument(
        "--musicpref-audio-subpath",
        default="phase7_release/datasets/musicprefs/audio",
        help="Prefix for audio_path column (relative to project root); {filename} for basename",
    )
    args = ap.parse_args()

    root = args.project_root or _REPO
    root = root.resolve()

    if args.dataset == "musicpref":
        if args.head not in ("musicality", "fidelity"):
            ap.error("musicpref requires --head musicality|fidelity")
        csv_p = args.human_preference_csv
        if csv_p is None:
            csv_p = root / "phase7_release/raw_hf/MusicPref/human_preference.csv"
        csv_p = csv_p.resolve()
        pairs = load_musicpref_pairs(csv_p, args.head)
        id_col = "filename"
        musicpref_prefix = args.musicpref_audio_subpath.strip("/")
    elif args.dataset == "aime":
        if args.head not in ("music_quality", "text_audio_alignment"):
            ap.error("aime requires --head music_quality|text_audio_alignment")
        pq = args.survey_parquet
        if pq is None:
            pq = root / "phase7_release/raw_hf/AIME-survey/data/train-00000-of-00001.parquet"
        pq = pq.resolve()
        pairs = load_aime_pairs(pq, args.head)
        id_col = "track_id"
        musicpref_prefix = None  # unused
    else:
        if args.head is not None:
            ap.error("musicarena does not take --head")
        bg = args.battle_glob
        if not bg:
            bg = str(root / "phase7_release/raw_hf/MusicArena/battle_data/**/*.json")
        pairs = load_musicarena_pairs(bg, repo_root=root)
        id_col = "audio_path"
        musicpref_prefix = None

    if not pairs:
        print("No pairs loaded; check paths and filters.", file=sys.stderr)
        sys.exit(1)

    raw_scores = fit_pairwise_scores(
        pairs,
        margin=args.margin,
        squared=args.squared_hinge,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    mos = minmax_map_to_range(raw_scores, low=args.low, high=args.high)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _source_tag() -> str:
        if args.dataset == "musicarena":
            return "MusicArena2025"
        if args.dataset == "musicpref":
            return f"MusicPref2025_{args.head}"
        return f"AIME2025_{args.head}"

    rows = []
    for k, v in sorted(mos.items()):
        row = {
            "source": _source_tag(),
            id_col: k,
            "score_1to5": round(v, 6),
        }
        if args.dataset == "musicpref":
            row["audio_path"] = f"{musicpref_prefix}/{k}".replace("//", "/")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows) from {len(pairs)} comparisons.")


if __name__ == "__main__":
    main()
