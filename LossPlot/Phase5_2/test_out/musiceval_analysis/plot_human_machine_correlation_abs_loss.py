#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_name(name: str) -> str:
    """
    规范化文件名用于匹配：
    - 只保留 basename
    - 去掉结尾冒号
    - 去掉扩展名（.wav/.mp3 等）
    - 转为小写
    """
    base = os.path.basename(str(name)).strip()
    if base.endswith(":"):
        base = base[:-1]
    # 去扩展名
    base = re.sub(r"\.(wav|mp3|flac|ogg)$", "", base, flags=re.IGNORECASE)
    return base.lower()

def load_human_scores(path: str) -> pd.Series:
    """
    读取人类打分文件：每行格式类似
    <filename>,<score1>,<score2>,... 或 <filename> <score1> <score2> ...
    返回：pd.Series(index=normalized_filename, values=avg_human_score)
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # 自动判断分隔符：优先逗号，否则空白
            parts = [p.strip() for p in (s.split(",") if "," in s else s.split())]
            if len(parts) < 2:
                continue

            raw_name = parts[0]
            fname = normalize_name(raw_name.rstrip(":"))

            scores = []
            for p in parts[1:]:
                p = p.strip().strip(",")
                try:
                    scores.append(float(p))
                except ValueError:
                    # 忽略非数字字段
                    pass
            if not scores:
                continue
            data[fname] = float(np.mean(scores))

    return pd.Series(data, name="human_mean").sort_index()

def load_abs_loss(path: str) -> pd.Series:
    """
    读取绝对损失文件：每行格式类似
    <filename>.wav: <value>
    返回：pd.Series(index=normalized_filename, values=abs_loss)
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 支持 “name: value” 或 “name.wav: value”
            m = re.match(r"^(.*?)\s*:\s*([+-]?\d+(?:\.\d+)?)$", line)
            if not m:
                continue
            raw_name, val = m.group(1), m.group(2)
            fname = normalize_name(raw_name)
            try:
                data[fname] = float(val)
            except ValueError:
                continue
    return pd.Series(data, name="abs_loss").sort_index()

def safe_pearson(x: pd.Series, y: pd.Series):
    xv = x.values.astype(float)
    yv = y.values.astype(float)
    if len(xv) < 2:
        return np.nan, np.nan
    # 常量序列的情况
    if np.isclose(np.std(xv), 0) or np.isclose(np.std(yv), 0):
        return np.nan, np.nan
    r = np.corrcoef(xv, yv)[0, 1]
    # p 值用近似（无外部包时无法精确计算），这里返回 NaN
    return float(r), np.nan

def safe_spearman(x: pd.Series, y: pd.Series):
    # 使用 pandas 排名后做 Pearson 即 Spearman
    if len(x) < 2:
        return np.nan, np.nan
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return safe_pearson(xr, yr)

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_and_save_scatter(x: pd.Series, y: pd.Series, out_png: str, title: str, xlabel: str, ylabel: str, r_text: str = ""):
    ensure_dir_for(out_png)

    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(x, y, s=25, alpha=0.8, edgecolors="k", linewidths=0.5)

    # 回归线
    try:
        coeffs = np.polyfit(x.values, y.values, 1)
        xx = np.linspace(float(np.min(x.values)), float(np.max(x.values)), 100)
        yy = coeffs[0] * xx + coeffs[1]
        plt.plot(xx, yy, color="red", linewidth=1.5, label="Linear fit")
    except Exception:
        pass

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if r_text:
        plt.legend(title=r_text, loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot correlation between human MOS (avg) and absolute loss.")
    parser.add_argument("--human_file", type=str, default="/home/evev/asap-dataset/test_out/musiceval_analysis/musiceval_small_human.txt")
    parser.add_argument("--abs_loss_file", type=str, default="/home/evev/asap-dataset/+Loss/Phase5_1_MusicEval_small/musiceval_ori_small/results.txt")
    parser.add_argument("--output_base", type=str, default="/home/evev/asap-dataset/test_out/musiceval_analysis/abs_loss",
                        help="Output base path (without extension). Will produce *_merged.csv, *_correlation_summary.csv, *_scatter.png")
    args = parser.parse_args()

    human = load_human_scores(args.human_file)
    abs_loss = load_abs_loss(args.abs_loss_file)

    # 合并（内连接）
    merged = pd.concat([human, abs_loss], axis=1, join="inner").dropna()
    merged = merged.rename_axis("filename").reset_index()

    # 保存合并数据
    merged_csv = f"{args.output_base}_merged.csv"
    ensure_dir_for(merged_csv)
    merged.to_csv(merged_csv, index=False)

    n = len(merged)
    if n == 0:
        print("No overlapping filenames. Please check filename normalization and input files.")
        print(f"Human count: {len(human)}, sample: {list(human.index[:5])}")
        print(f"Abs loss count: {len(abs_loss)}, sample: {list(abs_loss.index[:5])}")
        return

    px, pp = safe_pearson(merged["human_mean"], merged["abs_loss"])
    sx, sp = safe_spearman(merged["human_mean"], merged["abs_loss"])

    # 相关性汇总
    summary = pd.DataFrame([{
        "pair": "human_mean vs abs_loss",
        "n": n,
        "pearson_r": px,
        "pearson_p": pp,
        "spearman_r": sx,
        "spearman_p": sp,
    }])
    summary_csv = f"{args.output_base}_correlation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # 绘图
    r_text = f"Pearson r={px:.3f}, Spearman r={sx:.3f}" if not (np.isnan(px) and np.isnan(sx)) else ""
    scatter_png = f"{args.output_base}_scatter.png"
    plot_and_save_scatter(
        x=merged["human_mean"],
        y=merged["abs_loss"],
        out_png=scatter_png,
        title="Human MOS (avg) vs Absolute Loss",
        xlabel="Human MOS (avg)",
        ylabel="Absolute Loss",
        r_text=r_text
    )

    print(f"Merged items: {n}")
    print(f"Saved merged data to: {merged_csv}")
    print(f"Saved correlation summary to: {summary_csv}")
    print(f"Saved scatter plot to: {scatter_png}")

if __name__ == "__main__":
    main()