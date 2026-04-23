import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from phase7_release.analysis.segment.train_rnn import SegmentRNNRegressor, _read_entropy_map, _read_loss_curve
from phase7_release.lib.repro.data_paths import get_exp11_split_paths


def _segment_starts(length: int, win: int, hop: int) -> List[int]:
    if length <= 0:
        return []
    if length <= win:
        return [0]
    starts = []
    st = 0
    while st < length:
        starts.append(st)
        if st + win >= length:
            break
        st += hop
    return starts


def _build_segment_input(
    loss: np.ndarray,
    entropy: np.ndarray | None,
    st: int,
    win: int,
    input_mode: str,
) -> np.ndarray:
    ed = min(st + win, int(loss.shape[0]))
    l = loss[st:ed]
    if l.shape[0] < win:
        l = np.pad(l, (0, win - l.shape[0]), constant_values=0.0)
    if input_mode == "loss":
        return l[:, None].astype(np.float32)
    if entropy is None:
        raise ValueError("entropy is required when input_mode=loss_entropy")
    h = entropy[st:ed]
    h = np.nan_to_num(h, nan=0.0)
    if h.shape[0] < win:
        h = np.pad(h, (0, win - h.shape[0]), constant_values=0.0)
    return np.stack([l, h], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--entropy-manifest-csv", action="append", default=[])
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=50.0, help="curve frame rate. default 50 (~20ms)")
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="phase7_release/outputs/reports/segment_curves")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    input_mode = ckpt["input_mode"]
    segment_steps = ckpt["segment_steps"]
    hop_steps = ckpt["hop_steps"]
    if args.segment_seconds > 0:
        segment_steps = max(1, int(round(args.segment_seconds * args.fps)))
    if args.hop_seconds > 0:
        hop_steps = max(1, int(round(args.hop_seconds * args.fps)))

    model = SegmentRNNRegressor(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    split_csv = get_exp11_split_paths(args.config, splits=args.splits)[args.split]
    df = pd.read_csv(split_csv)
    if args.max_tracks > 0:
        df = df.head(args.max_tracks).copy()
    entropy_map: Dict[str, str] = _read_entropy_map(args.entropy_manifest_csv) if input_mode == "loss_entropy" else {}

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    for _, row in df.iterrows():
        token_loss_path = str(row["token_loss_path"])
        audio_path = str(row["audio_path"])
        score = float(row["score"])
        loss = _read_loss_curve(token_loss_path)
        if loss.size == 0:
            continue
        entropy = None
        if input_mode == "loss_entropy":
            ep = entropy_map.get(token_loss_path, "")
            if not ep or not os.path.isfile(ep):
                continue
            entropy = np.load(ep).astype(np.float32)
        starts = _segment_starts(int(loss.shape[0]), segment_steps, hop_steps)
        seg_preds = []
        with torch.no_grad():
            for st in starts:
                x = _build_segment_input(loss, entropy, st, segment_steps, input_mode)
                t = torch.tensor(x[None, ...], dtype=torch.float32, device=device)
                p = float(model(t).item())
                seg_preds.append(p)
        for i, p in enumerate(seg_preds):
            seg_start_step = starts[i]
            records.append(
                {
                    "audio_path": audio_path,
                    "token_loss_path": token_loss_path,
                    "track_score": score,
                    "segment_idx": i,
                    "segment_start_step": int(seg_start_step),
                    "segment_start_sec": float(seg_start_step / args.fps),
                    "segment_score_pred": p,
                }
            )

    out_csv = os.path.join(args.out_dir, f"segment_curve_{args.splits}_{args.split}_{input_mode}.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(records)} rows)")


if __name__ == "__main__":
    main()
