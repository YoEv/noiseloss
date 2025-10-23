import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 尝试使用 scipy 计算 p 值，如不可用则仅计算 r
try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

HUMAN_FILE = "test_out/musiceval_analysis/musiceval_small_human.txt"
MACHINE_FILE = "test_out/musiceval_analysis/bumped_area_statistics.csv"
OUT_DIR = "test_out/musiceval_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

def load_human_scores(path: str) -> pd.DataFrame:
    # 文件格式：filename,human1,human2（无表头）
    df = pd.read_csv(path, header=None, names=["filename", "human1", "human2"])
    # 清理可能的空格
    df["filename"] = df["filename"].astype(str).str.strip()
    # 移除.wav扩展名以便与机器评分文件匹配
    df["filename"] = df["filename"].str.replace('.wav', '', regex=False)
    df["human1"] = pd.to_numeric(df["human1"], errors="coerce")
    df["human2"] = pd.to_numeric(df["human2"], errors="coerce")
    df["human_mean"] = df[["human1", "human2"]].mean(axis=1)
    return df

def load_machine_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "noise_position" not in df.columns:
        def infer_pos(bs):
            try:
                b = float(bs)
            except Exception:
                return np.nan
            if abs(b - 245) <= 10:
                return "5s"
            if abs(b - 999) <= 10:
                return "20s"
            return np.nan
        df["noise_position"] = df.get("bumped_start", np.nan).apply(infer_pos)

    keep_cols = ["filename", "noise_position", "bumped_max"]
    df = df[keep_cols].copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["noise_position"] = df["noise_position"].astype(str).str.strip()
    df["bumped_max"] = pd.to_numeric(df["bumped_max"], errors="coerce")

    piv = df.pivot_table(index="filename",
                         columns="noise_position",
                         values="bumped_max",
                         aggfunc="mean")
    piv = piv.rename(columns={"5s": "machine_5s", "20s": "machine_20s"})
    piv["machine_mean"] = piv[["machine_5s", "machine_20s"]].mean(axis=1)
    piv = piv.reset_index()
    return piv

def safe_corr(x, y, method="pearson"):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return np.nan, np.nan 
    if method == "pearson":
        if SCIPY_AVAILABLE:
            r, p = pearsonr(df["x"], df["y"])
        else:
            r = np.corrcoef(df["x"], df["y"])[0, 1]
            p = np.nan
    else:
        if SCIPY_AVAILABLE:
            r, p = spearmanr(df["x"], df["y"])
        else:
            rx = df["x"].rank()
            ry = df["y"].rank()
            r = np.corrcoef(rx, ry)[0, 1]
            p = np.nan
    return r, p

def annotate_corr(ax, x, y, title):
    pr, pp = safe_corr(x, y, "pearson")
    sr, sp = safe_corr(x, y, "spearman")
    txt = f"Pearson r={pr:.3f}, p={pp:.3g}\nSpearman ρ={sr:.3f}, p={sp:.3g}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_title(title)

def main():
    human = load_human_scores(HUMAN_FILE)
    machine = load_machine_scores(MACHINE_FILE)
    
    print(f"Human data loaded: {len(human)} items")
    print(f"Machine data loaded: {len(machine)} items")
    print(f"Sample human filenames: {human['filename'].head(3).tolist()}")
    print(f"Sample machine filenames: {machine['filename'].head(3).tolist()}")

    # 只保留两边都存在的曲目
    df = pd.merge(human, machine, on="filename", how="inner")

    # 输出数据对齐的数量
    print(f"Merged items: {len(df)}")
    if len(df) == 0:
        print("No overlapping filenames between human and machine files.")
        return

    # 计算相关性并保存汇总
    rows = []
    pairs = [
        ("machine_5s",  "Human mean vs Machine (5s)"),
        ("machine_20s", "Human mean vs Machine (20s)"),
        ("machine_mean","Human mean vs Machine (mean of 5s & 20s)")
    ]
    for col, label in pairs:
        pr, pp = safe_corr(df["human_mean"], df[col], "pearson")
        sr, sp = safe_corr(df["human_mean"], df[col], "spearman")
        rows.append({
            "pair": label,
            "n": int(df[["human_mean", col]].dropna().shape[0]),
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp
        })
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_DIR, "correlation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    print(summary)

    # 绘图（3 个子图）
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    plot_cfgs = [
        ("machine_5s",  "Machine (bumped_max @5s)"),
        ("machine_20s", "Machine (bumped_max @20s)"),
        ("machine_mean","Machine (bumped_max mean)")
    ]
    for ax, (col, title) in zip(axes, plot_cfgs):
        sns.regplot(
            ax=ax,
            data=df,
            x="human_mean",
            y=col,
            scatter_kws={"alpha": 0.6, "s": 40, "edgecolor": "none"},
            line_kws={"color": "crimson"},
            ci=95
        )
        ax.set_xlabel("Human score (mean of 2 raters)")
        ax.set_ylabel("Machine score (bumped_max)")
        annotate_corr(ax, df["human_mean"], df[col], title)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "human_vs_machine_correlation.png")
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()