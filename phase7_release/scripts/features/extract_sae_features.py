import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import pandas as pd
import torch
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(SCRIPTS_DIR), "config", "paths.yaml"),
    )
    parser.add_argument("--splits", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    known = parser.parse_args()
    with open(known.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project_root = cfg["project_root"]
    sae_cfg = cfg.get("sae", {})
    backend = sae_cfg.get("backend", "musicdiscovery")
    if backend != "musicdiscovery":
        raise ValueError("Only musicdiscovery backend is supported in phase7_release.")

    md_script = os.path.join(THIS_DIR, "extract_sae_features_musicdiscovery.py")
    checkpoint_dir = sae_cfg.get("musicdiscovery_checkpoint_dir", "")
    if not checkpoint_dir:
        raise ValueError("Missing sae.musicdiscovery_checkpoint_dir in config/paths.yaml")
    checkpoint_dir = checkpoint_dir if os.path.isabs(checkpoint_dir) else os.path.join(project_root, checkpoint_dir)
    required = ["cfg.json", "sae_weights.safetensors", "sparsity.safetensors"]
    missing = [f for f in required if not os.path.isfile(os.path.join(checkpoint_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing SAE checkpoint files in {checkpoint_dir}: {missing}. "
            "Run: bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh"
        )
    output_dir = sae_cfg.get("output_dir", "phase7_release/features/sae")
    output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(project_root, output_dir)
    split_map = cfg.get("data", {}).get("splits", {}).get(known.splits, {})
    split_csv = split_map.get(known.split, "")
    if split_csv:
        split_csv = split_csv if os.path.isabs(split_csv) else os.path.join(project_root, split_csv)
    out_npy = os.path.join(output_dir, f"sae_features_{known.split}.npy")
    out_meta = os.path.join(output_dir, f"sae_features_{known.split}_meta.pt")
    if split_csv and os.path.isfile(split_csv) and os.path.isfile(out_npy) and os.path.isfile(out_meta):
        try:
            df = pd.read_csv(split_csv)
            meta = torch.load(out_meta, map_location="cpu")
            lengths = meta.get("lengths", [])
            if isinstance(lengths, list) and len(lengths) == len(df):
                print(
                    f"[skip-existing] {known.split}: found complete SAE artifacts "
                    f"{out_npy} and {out_meta} (samples={len(lengths)})"
                )
                return 0
        except Exception:
            pass
    cmd = [
        "python",
        md_script,
        "--project-root",
        project_root,
        "--config",
        known.config,
        "--split",
        known.split,
        "--splits",
        known.splits,
        "--checkpoint-dir",
        checkpoint_dir,
        "--output-dir",
        output_dir,
    ]
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
