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

def resolve_path(p: str) -> str:

    if os.path.isabs(p):
        return p
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return os.path.abspath(os.path.join(project_root, p))

def extract_perturbation_info(filename: str):
    if "rhythm" in filename:
        parts = filename.split("_")
        for part in parts:
            if part.startswith("r") and part[1:].isdigit():
                return "rhythm", part
    elif "velocity" in filename:
        parts = filename.split("_")
        for part in parts:
            if part.startswith("nt") and part[2:].isdigit():
                return "velocity", part
    return "unknown", "unknown"

def main():
    parser = argparse.ArgumentParser(description="Compute musical perturbation gradient diffs with red-blue heatmaps.")
    parser.add_argument("--ori_csv", default="experiments/phase6/results/ori_musical/Beethoven_010_midi_score_short.mid_ori_activation_gradients.csv", help="Path to ori activation gradients CSV.")
    parser.add_argument("--perturb_root", default="experiments/phase6/results/perturb_musical", help="Root dir containing perturbation files.")
    parser.add_argument("--output_root", default="experiments/phase6/results/diff_musical", help="Root dir to write diff results.")
    args = parser.parse_args()

    ori_csv_abs = resolve_path(args.ori_csv)
    perturb_root_abs = resolve_path(args.perturb_root)
    output_root_abs = resolve_path(args.output_root)

    if not os.path.exists(ori_csv_abs):
        print(f"Error: Original CSV not found: {ori_csv_abs}")
        return

    if not os.path.exists(perturb_root_abs):
        print(f"Error: Perturbation root directory not found: {perturb_root_abs}")
        return

    os.makedirs(output_root_abs, exist_ok=True)

    perturb_files = []
    for filename in os.listdir(perturb_root_abs):
        if filename.endswith("_activation_gradients.csv"):
            perturb_files.append(filename)

    print(f"Found {len(perturb_files)} perturbation files")

    rhythm_files = []
    velocity_files = []
    
    for filename in perturb_files:
        pert_type, pert_param = extract_perturbation_info(filename)
        if pert_type == "rhythm":
            rhythm_files.append((filename, pert_param))
        elif pert_type == "velocity":
            velocity_files.append((filename, pert_param))

    if rhythm_files:
        rhythm_dir = os.path.join(output_root_abs, "rhythm")
        os.makedirs(rhythm_dir, exist_ok=True)
        print(f"\nProcessing {len(rhythm_files)} rhythm perturbations...")
        
        for filename, param in sorted(rhythm_files):
            pert_csv = os.path.join(perturb_root_abs, filename)
            basename = infer_basename_from_csv(filename)
            
            out_csv = os.path.join(rhythm_dir, f"{basename}_activation_gradients_diff.csv")
            out_png = os.path.join(rhythm_dir, f"{basename}_gradient_diff_heatmap.png")
            
            title = f"Rhythm Perturbation {param} | Diff Heatmap (pert - ori)"
            
            try:
                compute_diff_and_plot(
                    ori_csv=ori_csv_abs,
                    pert_csv=pert_csv,
                    output_csv=out_csv,
                    output_png=out_png,
                    title=title
                )
                print(f"[ok] rhythm {param}: wrote {out_csv} and {out_png}")
            except FileNotFoundError as e:
                print(f"[skip] rhythm {param}: {e}")
            except Exception as e:
                print(f"[error] rhythm {param}: {e}")

    if velocity_files:
        velocity_dir = os.path.join(output_root_abs, "velocity")
        os.makedirs(velocity_dir, exist_ok=True)
        print(f"\nProcessing {len(velocity_files)} velocity perturbations...")
        
        for filename, param in sorted(velocity_files):
            pert_csv = os.path.join(perturb_root_abs, filename)
            basename = infer_basename_from_csv(filename)
            
            out_csv = os.path.join(velocity_dir, f"{basename}_activation_gradients_diff.csv")
            out_png = os.path.join(velocity_dir, f"{basename}_gradient_diff_heatmap.png")
            
            title = f"Velocity Perturbation {param} | Diff Heatmap (pert - ori)"
            
            try:
                compute_diff_and_plot(
                    ori_csv=ori_csv_abs,
                    pert_csv=pert_csv,
                    output_csv=out_csv,
                    output_png=out_png,
                    title=title
                )
                print(f"[ok] velocity {param}: wrote {out_csv} and {out_png}")
            except FileNotFoundError as e:
                print(f"[skip] velocity {param}: {e}")
            except Exception as e:
                print(f"[error] velocity {param}: {e}")

    print(f"\nProcessing completed!")
    print(f"Results saved in: {output_root_abs}")
    print(f"- Rhythm perturbations: {os.path.join(output_root_abs, 'rhythm')}")
    print(f"- Velocity perturbations: {os.path.join(output_root_abs, 'velocity')}")

if __name__ == "__main__":
    main()