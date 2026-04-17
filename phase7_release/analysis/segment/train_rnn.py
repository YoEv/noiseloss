import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

from phase7_release.lib.repro.data_paths import get_exp11_split_paths


def _read_loss_curve(token_loss_path: str) -> np.ndarray:
    try:
        df = pd.read_csv(token_loss_path, usecols=[1], header=0)
        v = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values.astype(np.float32)
        return v
    except Exception:
        return np.zeros((0,), dtype=np.float32)


def _read_entropy_map(entropy_manifest_csvs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in entropy_manifest_csvs:
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            if pd.isna(row.get("entropy_curve_path", np.nan)):
                continue
            out[str(row["token_loss_path"])] = str(row["entropy_curve_path"])
    return out


def _segment_count(length: int, win: int, hop: int) -> int:
    if length <= 0:
        return 0
    if length <= win:
        return 1
    return 1 + (length - win + hop - 1) // hop


class SegmentDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        segment_steps: int,
        hop_steps: int,
        input_mode: str = "loss",
        entropy_map: Dict[str, str] | None = None,
        max_tracks: int = 0,
    ):
        self.df = pd.read_csv(split_csv)
        if max_tracks > 0:
            self.df = self.df.head(max_tracks).copy()
        self.segment_steps = segment_steps
        self.hop_steps = hop_steps
        self.input_mode = input_mode
        self.entropy_map = entropy_map or {}
        self.samples: List[Tuple[np.ndarray, float]] = []
        self._build()

    def _build(self):
        for _, row in self.df.iterrows():
            score = float(row["score"])
            token_loss_path = str(row["token_loss_path"])
            loss = _read_loss_curve(token_loss_path)
            if loss.size == 0:
                continue
            entropy = None
            if self.input_mode == "loss_entropy":
                ep = self.entropy_map.get(token_loss_path, "")
                if ep and os.path.isfile(ep):
                    entropy = np.load(ep).astype(np.float32)
                else:
                    continue
            n_seg = _segment_count(int(loss.shape[0]), self.segment_steps, self.hop_steps)
            for i in range(n_seg):
                st = i * self.hop_steps
                ed = min(st + self.segment_steps, int(loss.shape[0]))
                l = loss[st:ed]
                if l.shape[0] < self.segment_steps:
                    l = np.pad(l, (0, self.segment_steps - l.shape[0]), constant_values=0.0)
                if entropy is None:
                    x = l[:, None]
                else:
                    h = entropy[st:ed]
                    h = np.nan_to_num(h, nan=0.0)
                    if h.shape[0] < self.segment_steps:
                        h = np.pad(h, (0, self.segment_steps - h.shape[0]), constant_values=0.0)
                    x = np.stack([l, h], axis=-1)
                self.samples.append((x.astype(np.float32), score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


class SegmentRNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).cpu().numpy()
            preds.extend(p.tolist())
            targets.extend(y.numpy().ravel().tolist())
    if len(preds) < 2:
        return float("nan"), float("nan")
    return float(pearsonr(preds, targets)[0]), float(spearmanr(preds, targets)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--input-mode", type=str, default="loss", choices=["loss", "loss_entropy"])
    parser.add_argument("--entropy-manifest-csv", action="append", default=[])
    parser.add_argument("--segment-steps", type=int, default=500)
    parser.add_argument("--hop-steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="segment_rnn")
    parser.add_argument("--out-dir", type=str, default="phase7_release/outputs/checkpoints/segment")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = get_exp11_split_paths(args.config, splits=args.splits)
    entropy_map = _read_entropy_map(args.entropy_manifest_csv) if args.input_mode == "loss_entropy" else {}

    train_ds = SegmentDataset(
        split_csv=paths["train"],
        segment_steps=args.segment_steps,
        hop_steps=args.hop_steps,
        input_mode=args.input_mode,
        entropy_map=entropy_map,
        max_tracks=args.max_tracks,
    )
    val_ds = SegmentDataset(
        split_csv=paths["val"],
        segment_steps=args.segment_steps,
        hop_steps=args.hop_steps,
        input_mode=args.input_mode,
        entropy_map=entropy_map,
        max_tracks=args.max_tracks,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = 1 if args.input_mode == "loss" else 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmentRNNRegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = -1e9
    ckpt_path = os.path.join(args.out_dir, f"{args.run_name}_{args.splits}_{args.input_mode}.pth")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            optimizer.zero_grad(set_to_none=True)
            p = model(x)
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        train_loss = total / max(1, n)
        val_p, val_s = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_pearson={val_p:.4f} val_spearman={val_s:.4f}")
        score = -999.0 if not np.isfinite(val_p) else val_p
        if score > best_val:
            best_val = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "segment_steps": args.segment_steps,
                    "hop_steps": args.hop_steps,
                    "input_mode": args.input_mode,
                    "splits": args.splits,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path}")


if __name__ == "__main__":
    main()
