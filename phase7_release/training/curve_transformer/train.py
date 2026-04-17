import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from phase7_release.lib.repro.data_paths import get_exp11_split_paths
from phase7_release.lib.repro.entropy_dataset import EntropyCurveDataset, entropy_curve_collate_fn
from phase7_release.lib.repro.loss_dataset import LossCurveDataset, loss_curve_collate_fn
from phase7_release.lib.repro.nets import TransformerEncoderRegressor

MAX_LEN = 1500


def _build_dataset(paths, mode, entropy_manifest_csv):
    if mode == "loss":
        train_ds = LossCurveDataset(paths["train"], max_len=MAX_LEN, entropy_manifest_csv=None)
        val_ds = LossCurveDataset(paths["val"], max_len=MAX_LEN, entropy_manifest_csv=None)
        collate = loss_curve_collate_fn
        d_in = int(train_ds.output_channels)
    elif mode == "entropy":
        train_ds = EntropyCurveDataset(paths["train"], max_len=MAX_LEN, entropy_manifest_csv=entropy_manifest_csv)
        val_ds = EntropyCurveDataset(paths["val"], max_len=MAX_LEN, entropy_manifest_csv=entropy_manifest_csv)
        collate = entropy_curve_collate_fn
        d_in = int(train_ds.output_channels)
    elif mode == "loss_entropy":
        train_ds = LossCurveDataset(paths["train"], max_len=MAX_LEN, entropy_manifest_csv=entropy_manifest_csv)
        val_ds = LossCurveDataset(paths["val"], max_len=MAX_LEN, entropy_manifest_csv=entropy_manifest_csv)
        collate = loss_curve_collate_fn
        d_in = int(train_ds.output_channels)
    else:
        raise ValueError("mode must be loss/entropy/loss_entropy")
    return train_ds, val_ds, collate, d_in


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for inputs, targets in loader:
        if inputs.nelement() == 0:
            continue
        x = inputs.to(device).permute(0, 2, 1)
        y = targets.to(device).squeeze(-1)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


def _eval(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, t in loader:
            if inputs.nelement() == 0:
                continue
            x = inputs.to(device).permute(0, 2, 1)
            out = model(x)
            preds.extend(out.cpu().numpy().ravel())
            targets.extend(t.numpy().ravel())
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if len(preds) < 2:
        return float("nan"), float("nan")
    mse = float(np.mean((preds - targets) ** 2))
    corr = float(pearsonr(preds, targets)[0])
    return mse, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["loss", "entropy", "loss_entropy"])
    parser.add_argument("--entropy-manifest-csv", action="append", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr-decay-factor", type=float, default=None)
    parser.add_argument("--lr-decay-patience", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--run-name", type=str, default="curve_transformer")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg.get("training", {}).get("defaults", {}).get("curve_transformer", {})
    lr = float(args.lr if args.lr is not None else tcfg.get("lr", 1e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else tcfg.get("weight_decay", 1e-4))
    epochs = int(args.epochs if args.epochs is not None else tcfg.get("epochs", 100))
    batch_size = int(args.batch_size if args.batch_size is not None else tcfg.get("batch_size", 32))
    patience = int(args.patience if args.patience is not None else tcfg.get("patience", 10))
    lr_decay_factor = float(args.lr_decay_factor if args.lr_decay_factor is not None else tcfg.get("lr_decay_factor", 0.5))
    lr_decay_patience = int(args.lr_decay_patience if args.lr_decay_patience is not None else tcfg.get("lr_decay_patience", 3))
    min_lr = float(args.min_lr if args.min_lr is not None else tcfg.get("min_lr", 1e-6))

    paths = get_exp11_split_paths(args.config, splits=args.splits)
    train_ds, val_ds, collate_fn, d_in = _build_dataset(paths, args.mode, args.entropy_manifest_csv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    out_cfg = cfg.get("outputs", {})
    project_root = cfg.get("project_root", "/home/evev/noiseloss")
    ckpt_dir = out_cfg.get("checkpoints", "phase7_release/outputs/checkpoints")
    ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(project_root, ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.run_name}_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderRegressor(seq_len=MAX_LEN, d_in=d_in, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, pool="mean").to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_decay_factor, patience=lr_decay_patience, min_lr=min_lr
    )

    best = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        tr = _train_epoch(model, train_loader, criterion, optimizer, device)
        va, corr = _eval(model, val_loader, device)
        scheduler.step(va)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch+1}/{epochs} train_loss={tr:.4f} val_loss={va:.4f} val_corr={corr:.4f} lr={cur_lr:.2e}")
        if np.isfinite(va) and va < best:
            best = va
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved {ckpt_path}")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break


if __name__ == "__main__":
    main()
