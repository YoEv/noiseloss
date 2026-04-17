import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from phase7_release.lib.repro.data_paths import get_exp11_split_paths, resolve_project_root
from phase7_release.lib.repro.entropy_dataset import EntropyCurveDataset, entropy_curve_collate_fn
from phase7_release.lib.repro.loss_dataset import LossCurveDataset, loss_curve_collate_fn
from phase7_release.lib.repro.nets import TransformerEncoderRegressor

MAX_LEN = 1500


def _build_test_dataset(test_csv, mode, entropy_manifest_csv):
    if mode == "loss":
        ds = LossCurveDataset(test_csv, max_len=MAX_LEN, entropy_manifest_csv=None)
        collate = loss_curve_collate_fn
        d_in = int(ds.output_channels)
    elif mode == "entropy":
        ds = EntropyCurveDataset(test_csv, max_len=MAX_LEN, entropy_manifest_csv=entropy_manifest_csv)
        collate = entropy_curve_collate_fn
        d_in = int(ds.output_channels)
    elif mode == "loss_entropy":
        ds = LossCurveDataset(test_csv, max_len=MAX_LEN, entropy_manifest_csv=entropy_manifest_csv)
        collate = loss_curve_collate_fn
        d_in = int(ds.output_channels)
    else:
        raise ValueError("mode must be loss/entropy/loss_entropy")
    return ds, collate, d_in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["loss", "entropy", "loss_entropy"])
    parser.add_argument("--entropy-manifest-csv", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--run-name", type=str, default="curve_transformer")
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_cfg = cfg.get("outputs", {})
    project_root = resolve_project_root(cfg)

    paths = get_exp11_split_paths(args.config, splits=args.splits)
    ds, collate_fn, d_in = _build_test_dataset(paths["test"], args.mode, args.entropy_manifest_csv)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    ckpt_dir = out_cfg.get("checkpoints", "phase7_release/outputs/checkpoints")
    ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(project_root, ckpt_dir)
    ckpt = os.path.join(ckpt_dir, f"{args.run_name}_best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderRegressor(seq_len=MAX_LEN, d_in=d_in, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, pool="mean").to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            if inputs.nelement() == 0:
                continue
            x = inputs.to(device).permute(0, 2, 1)
            out = model(x)
            preds.extend(out.cpu().numpy().ravel().tolist())

    test_df = pd.read_csv(paths["test"])
    out_df = test_df[["score", "audio_path"]].copy()
    out_df["pred"] = preds
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
