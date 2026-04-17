import os
from typing import Dict

import pandas as pd
import yaml


def _load_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_exp11_split_paths(config_path: str, splits: str = "clean") -> Dict[str, str]:
    cfg = _load_cfg(config_path)
    root = cfg["project_root"]
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("splits", {})
    if splits in split_cfg:
        chosen = split_cfg[splits]
        return {
            "train": os.path.join(root, chosen["train"]),
            "val": os.path.join(root, chosen["val"]),
            "test": os.path.join(root, chosen["test"]),
        }
    legacy_cfg = cfg.get("legacy", {})
    if "exp11_root" in legacy_cfg:
        base = os.path.join(root, legacy_cfg["exp11_root"], "1_data_preparation")
        if splits == "noisy":
            return {
                "train": os.path.join(base, "train_noisy.csv"),
                "val": os.path.join(base, "val_noisy.csv"),
                "test": os.path.join(base, "test_noisy.csv"),
            }
        return {
            "train": os.path.join(base, "train.csv"),
            "val": os.path.join(base, "val.csv"),
            "test": os.path.join(base, "test.csv"),
        }
    raise ValueError(f"No split paths configured for '{splits}' in {config_path}")


def load_exp11_splits(config_path: str, splits: str = "clean") -> Dict[str, pd.DataFrame]:
    paths = get_exp11_split_paths(config_path, splits=splits)
    return {k: pd.read_csv(v) for k, v in paths.items()}
