#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def infer_basename_from_csv(csv_path: str) -> str:
    fname = os.path.basename(csv_path)
    if fname.endswith("_activation_gradients.csv"):
        return fname[:-len("_activation_gradients.csv")]
    if fname.endswith("_gradient_statistics.csv"):
        return fname[:-len("_gradient_statistics.csv")]
    if fname.endswith("_gradient_heatmap.png"):
        return fname[:-len("_gradient_heatmap.png")]
    return os.path.splitext(fname)[0]

def load_activation_matrix(csv_path: str) -> np.ndarray:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop non-numeric columns like 'layer' if present
    lower_cols = [c.lower() for c in df.columns]
    drop_cols = [c for c, lc in zip(df.columns, lower_cols) if lc in ("layer", "layers")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Coerce to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # If first column looks like an index (0..N-1), drop it
    if df.shape[1] > 1:
        col0 = df.iloc[:, 0].values
        if np.all(~np.isnan(col0)):
            try:
                if np.allclose(col0, np.arange(len(col0)), atol=1e-6):
                    df = df.iloc[:, 1:]
            except Exception:
                pass

    mat = df.to_numpy(dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"Invalid CSV shape (expected 2D), got {mat.shape} from {csv_path}")
    if mat.size == 0:
        raise ValueError(f"CSV empty: {csv_path}")
    return mat

def compute_diff_and_plot(
    ori_csv: str,
    pert_csv: str,
    output_csv: str,
    output_png: str,
    title: str
):
    ori = load_activation_matrix(ori_csv)
    pert = load_activation_matrix(pert_csv)

    layers = min(ori.shape[0], pert.shape[0])
    tokens = min(ori.shape[1], pert.shape[1])

    if (ori.shape[0] != pert.shape[0]) or (ori.shape[1] != pert.shape[1]):
        print(f"[warn] Shape mismatch. Trimming to common shape: layers={layers}, tokens={tokens}")
        print(f"       ori={ori.shape}, pert={pert.shape}")

    ori_trim = ori[:layers, :tokens]
    pert_trim = pert[:layers, :tokens]
    diff = pert_trim - ori_trim

    # Save diff CSV
    pd.DataFrame(diff).to_csv(output_csv, index=False)

    # Plot diff heatmap (red = positive diff, blue = negative diff)
    vmax = float(np.nanmax(np.abs(diff))) if diff.size > 0 else 0.0
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1e-12  # avoid zero range colorbar

    # Auto figure size (guard against too-large)
    width = min(20.0, max(6.0, tokens / 20.0 + 4.0))
    height = min(12.0, max(4.0, layers / 10.0 + 3.0))

    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    im = ax.imshow(diff, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pert - ori", rotation=90)

    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("Layer")

    # Tick density control
    if tokens > 100:
        ax.set_xticks(np.linspace(0, tokens - 1, 6).astype(int))
    if layers > 50:
        ax.set_yticks(np.linspace(0, layers - 1, 6).astype(int))

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)

def parse_tk_levels(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    levels = []
    for p in parts:
        try:
            levels.append(int(p))
        except ValueError:
            raise ValueError(f"Invalid tk level: {p}")
    return levels

def resolve_path(p: str) -> str:
    """解析为绝对路径：
    - 若已是绝对路径，直接返回；
    - 否则以脚本所在目录的两级父目录（项目根 noiseloss）为基准拼接。
    """
    if os.path.isabs(p):
        return p
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return os.path.abspath(os.path.join(project_root, p))

def main():
    parser = argparse.ArgumentParser(description="Compute per-token gradient diffs (pert - ori) with red-blue heatmaps.")
    parser.add_argument("--ori_csv", required=True, help="Path to ori activation gradients CSV.")
    parser.add_argument("--perturb_root", default="experiments/phase6/results/perturb", help="Root dir containing tk* subfolders.")
    parser.add_argument("--output_root", default="experiments/phase6/results/diff", help="Root dir to write diff results.")
    parser.add_argument("--tk_levels", default="5,10,50,100,150,200", help="Comma-separated tk levels.")
    parser.add_argument("--basename", default=None, help="Base filename without suffix; inferred from --ori_csv if not provided.")
    args = parser.parse_args()

    # 统一解析为绝对路径
    ori_csv_abs = resolve_path(args.ori_csv)
    perturb_root_abs = resolve_path(args.perturb_root)
    output_root_abs = resolve_path(args.output_root)

    # 基名基于解析后的 ori 路径
    basename = args.basename or infer_basename_from_csv(ori_csv_abs)
    tk_levels = parse_tk_levels(args.tk_levels)

    os.makedirs(output_root_abs, exist_ok=True)

    for tk in tk_levels:
        pert_dir = os.path.join(perturb_root_abs, f"tk{tk}")
        pert_csv = os.path.join(pert_dir, f"{basename}_activation_gradients.csv")

        out_dir = os.path.join(output_root_abs, f"tk{tk}")
        os.makedirs(out_dir, exist_ok=True)

        out_csv = os.path.join(out_dir, f"{basename}_activation_gradients_diff.csv")
        out_png = os.path.join(out_dir, f"{basename}_gradient_diff_heatmap.png")

        title = f"{basename} | Diff Heatmap (pert - ori) | tk{tk}"
        try:
            compute_diff_and_plot(
                ori_csv=ori_csv_abs,
                pert_csv=pert_csv,
                output_csv=out_csv,
                output_png=out_png,
                title=title
            )
            print(f"[ok] tk{tk}: wrote {out_csv} and {out_png}")
        except FileNotFoundError as e:
            print(f"[skip] tk{tk}: {e}")
        except Exception as e:
            print(f"[error] tk{tk}: {e}")

if __name__ == "__main__":
    main()