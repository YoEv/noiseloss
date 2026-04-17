import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase7_release.lib.repro.nets import TransformerEncoderRegressor
from phase7_release.training.hybrid.dataset import HybridPrecomputedDataset
from phase7_release.lib.repro.data_paths import detect_project_root, get_exp11_split_paths

matplotlib.use("Agg")
DATA_ROOT = detect_project_root()
SHM_SAFETY_RATIO = 0.9


class ConcatProject(nn.Module):
    def __init__(self, curve_channels=2, sae_dim=16384, d_model=256):
        super().__init__()
        self.curve_channels = curve_channels
        self.proj = nn.Linear(curve_channels + sae_dim, d_model)

    def forward(self, loss_curve, sae_features):
        if self.curve_channels <= 0:
            return self.proj(sae_features)
        loss_t = loss_curve.permute(0, 2, 1)
        return self.proj(torch.cat([loss_t, sae_features], dim=-1))


class HybridTransformerModel(nn.Module):
    def __init__(self, seq_len, sae_dim, curve_channels=2, d_model=256, nhead=8, num_layers=4, pool="mean"):
        super().__init__()
        self.concat = ConcatProject(curve_channels=curve_channels, sae_dim=sae_dim, d_model=d_model)
        self.transformer = TransformerEncoderRegressor(seq_len=seq_len, d_in=d_model, d_model=d_model, nhead=nhead, num_layers=num_layers, pool=pool)

    def forward(self, loss_curve, sae_features):
        return self.transformer(self.concat(loss_curve, sae_features))


def _safe_loader_params(
    seq_len: int,
    sae_dim: int,
    curve_channels: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    if num_workers <= 0:
        return batch_size, 0, prefetch_factor
    try:
        st = os.statvfs("/dev/shm")
        shm_avail = int(st.f_frsize * st.f_bavail)
    except Exception:
        shm_avail = 0
    if shm_avail <= 0:
        return batch_size, num_workers, prefetch_factor
    sample_bytes = int(seq_len) * int(max(1, sae_dim + max(0, curve_channels))) * 4
    batch_bytes = int(batch_size) * sample_bytes
    safe_limit = int(shm_avail * SHM_SAFETY_RATIO)
    nw = max(0, int(num_workers))
    pf = max(2, int(prefetch_factor))
    est_inflight = batch_bytes * max(1, nw) * max(1, pf) * 2
    while nw > 0 and est_inflight > safe_limit:
        if pf > 2:
            pf -= 1
        else:
            nw -= 1
        est_inflight = batch_bytes * max(1, nw) * max(1, pf) * 2
    return batch_size, nw, pf


def _safe_corr(preds_list, scores_list):
    preds = np.asarray(preds_list, dtype=np.float64)
    scores = np.asarray(scores_list, dtype=np.float64)
    valid = np.isfinite(preds) & np.isfinite(scores)
    if valid.sum() < 2:
        return 0.0, 0.0
    preds, scores = preds[valid], scores[valid]
    try:
        p, _ = pearsonr(preds, scores)
        s, _ = spearmanr(preds, scores)
        return float(p) if np.isfinite(p) else 0.0, float(s) if np.isfinite(s) else 0.0
    except Exception:
        return 0.0, 0.0


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        loss_curve = batch["loss_curve"].to(device)
        sae_features = batch["sae_features"].to(device)
        scores = batch["score"].to(device).squeeze(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            preds = model(loss_curve, sae_features)
            loss = criterion(preds, scores)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    preds_list, scores_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            loss_curve = batch["loss_curve"].to(device)
            sae_features = batch["sae_features"].to(device)
            scores = batch["score"].to(device).squeeze(-1)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                preds = model(loss_curve, sae_features)
                loss = criterion(preds, scores)
            total += loss.item()
            preds_list.extend(preds.cpu().numpy())
            scores_list.extend(scores.cpu().numpy())
    p, s = _safe_corr(preds_list, scores_list)
    return total / len(loader), p, s, preds_list, scores_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-decay-factor", type=float, default=None)
    parser.add_argument("--lr-decay-patience", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--run_name", type=str, default="transformer")
    parser.add_argument("--sae_suffix", type=str, default="")
    parser.add_argument("--use_noisy_splits", action="store_true")
    parser.add_argument("--curve-mode", type=str, default="loss", choices=["loss", "entropy", "loss_entropy", "none"])
    parser.add_argument("--entropy-manifest-csv", action="append", default=None)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--num-workers", type=int, default=max(2, min(8, os.cpu_count() or 4)))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg.get("training", {}).get("defaults", {}).get("hybrid_transformer", {})
    epochs = int(args.epochs if args.epochs is not None else tcfg.get("epochs", 50))
    patience = int(args.patience if args.patience is not None else tcfg.get("patience", 10))
    default_batch_size = max(int(tcfg.get("batch_size", 8)), 12)
    batch_size = int(args.batch_size if args.batch_size is not None else default_batch_size)
    lr = float(args.lr if args.lr is not None else tcfg.get("lr", 1e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else tcfg.get("weight_decay", 1e-4))
    lr_decay_factor = float(args.lr_decay_factor if args.lr_decay_factor is not None else tcfg.get("lr_decay_factor", 0.5))
    lr_decay_patience = int(args.lr_decay_patience if args.lr_decay_patience is not None else tcfg.get("lr_decay_patience", 3))
    min_lr = float(args.min_lr if args.min_lr is not None else tcfg.get("min_lr", 1e-6))
    data_cfg = cfg.get("data", {})
    feature_roots = data_cfg.get("feature_roots", {})
    split_paths = get_exp11_split_paths(args.config, splits="noisy" if args.use_noisy_splits else "clean")
    sae_feature_dir = cfg.get("sae", {}).get("output_dir", feature_roots.get("sae_feature_root", "phase7_release/features/sae"))
    token_loss_root = feature_roots.get("token_loss_root", "")
    seq_len = cfg.get("seq_len", 1500)
    sae_dim = cfg.get("sae_hidden_dim", 16384)
    out_cfg = cfg.get("outputs", {})
    project_root = cfg.get("project_root", DATA_ROOT)
    plot_dir = out_cfg.get("plots", "phase7_release/outputs/plots")
    save_dir = out_cfg.get("checkpoints", "phase7_release/outputs/checkpoints")
    report_dir = out_cfg.get("reports", "phase7_release/outputs/reports")
    plot_dir = plot_dir if os.path.isabs(plot_dir) else os.path.join(project_root, plot_dir)
    save_dir = save_dir if os.path.isabs(save_dir) else os.path.join(project_root, save_dir)
    report_dir = report_dir if os.path.isabs(report_dir) else os.path.join(project_root, report_dir)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    def _make_ds(split_name, csv_path):
        return HybridPrecomputedDataset(
            split_name,
            DATA_ROOT,
            split_csv_path=csv_path,
            sae_feature_dir=sae_feature_dir,
            token_loss_root=token_loss_root,
            seq_len=seq_len,
            sae_dim=sae_dim,
            sae_variant_suffix=args.sae_suffix,
            curve_mode=args.curve_mode,
            entropy_manifest_csv=args.entropy_manifest_csv,
        )

    train_ds = _make_ds("train", split_paths["train"])
    val_ds = _make_ds("val", split_paths["val"])
    test_ds = _make_ds("test", split_paths["test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    req_workers = max(0, int(args.num_workers))
    req_prefetch = max(2, int(args.prefetch_factor))
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "3060" in gpu_name:
            if req_workers > 4 or req_prefetch > 2:
                print(
                    f"[gpu-profile] {gpu_name}: tune loader workers {req_workers}->4, prefetch {req_prefetch}->2"
                )
            req_workers = min(req_workers, 4)
            req_prefetch = min(req_prefetch, 2)
    batch_size, tuned_workers, tuned_prefetch = _safe_loader_params(
        seq_len=seq_len,
        sae_dim=int(train_ds.sae_dim),
        curve_channels=int(train_ds.curve_channels),
        batch_size=batch_size,
        num_workers=req_workers,
        prefetch_factor=req_prefetch,
    )
    if tuned_workers != int(args.num_workers) or tuned_prefetch != int(args.prefetch_factor):
        print(
            f"[loader-autotune] shm guard active: workers {args.num_workers}->{tuned_workers}, "
            f"prefetch {args.prefetch_factor}->{tuned_prefetch}"
        )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": tuned_workers,
        "pin_memory": device.type == "cuda",
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = tuned_prefetch
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = HybridTransformerModel(seq_len, int(train_ds.sae_dim), curve_channels=train_ds.curve_channels, pool=args.pool).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_decay_factor, patience=lr_decay_patience, min_lr=min_lr
    )
    criterion = nn.MSELoss()
    sae_tag = args.sae_suffix.lstrip("_") if args.sae_suffix else "newsae"
    run_tag = args.run_name
    if args.use_noisy_splits and (not run_tag.endswith("_noisy")):
        run_tag = f"{run_tag}_noisy"
    curve_tag = args.curve_mode
    ckpt_path = os.path.join(save_dir, f"exp13_hybrid_transformer_{curve_tag}_{sae_tag}_{run_tag}_best.pth")

    best_val_pearson = -1.0
    no_improve = 0
    epoch_pbar = tqdm(range(epochs), desc=f"{run_tag}-epochs", unit="ep", dynamic_ncols=True)
    for epoch in epoch_pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_loss, val_p, val_s, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_p if np.isfinite(val_p) else -1.0)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        epoch_pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            p=f"{val_p:.4f}",
            lr=f"{cur_lr:.2e}",
        )
        print(f"Epoch {epoch+1}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | ValP {val_p:.4f} | ValS {val_s:.4f} | lr={cur_lr:.2e}")
        if val_p > best_val_pearson:
            best_val_pearson = val_p
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Best saved: {ckpt_path}")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    epoch_pbar.close()

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_p, test_s, test_preds, test_scores = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss {test_loss:.4f} | Test Pearson {test_p:.4f} | Test Spearman {test_s:.4f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(test_scores, test_preds, alpha=0.5)
    plt.title(f"Hybrid Transformer ({curve_tag}+sae) [{sae_tag}] {run_tag}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.savefig(os.path.join(plot_dir, f"exp13_hybrid_transformer_{curve_tag}_{sae_tag}_{run_tag}_scatter.png"), dpi=150)
    plt.close()
    out_df = pd.DataFrame({"score": test_scores, "pred": test_preds})
    if "audio_path" in test_ds.df.columns:
        out_df["audio_path"] = test_ds.df["audio_path"].values[: len(out_df)]
    out_df = out_df[["score", "audio_path", "pred"]] if "audio_path" in out_df.columns else out_df[["score", "pred"]]
    out_csv = os.path.join(report_dir, f"{run_tag}_test_scores.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
