import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

# 配置
INPUT_FILE = 'Loss_Cal/rhythm_mod_loss_cal_ori_add.txt'
OUTPUT_DIR = 'Loss_Cal_Plot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义目标类别
CATEGORIES = ['_ori', 'rhythm1_4_step1_r30', 'rhythm1_4_step1_r40',
              'rhythm1_4_step2_r30', 'rhythm1_4_step2_r40']

def load_data():
    """加载数据并组织成结构化格式"""
    data = {}

    with open(INPUT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue

            # 提取文件名和loss值
            filename, loss = line.rsplit(':', 1)
            filename = filename.strip()
            try:
                loss = float(loss.strip())
            except ValueError:
                continue

            # 提取曲目名称(去掉_midi_score后面的部分)
            song_name = filename.split('_midi_score')[0]

            # 分类
            for cat in CATEGORIES:
                if cat in filename:
                    if song_name not in data:
                        data[song_name] = {}
                    data[song_name][cat] = loss
                    break

    return data

def calculate_differences(data):
    """计算每个曲目各节奏模式与原始版本的差异"""
    diff_data = []

    for song, values in data.items():
        if '_ori' not in values:
            continue  # 必须有原始版本才能比较

        ori_value = values['_ori']
        for cat in CATEGORIES[1:]:  # 跳过'_ori'
            if cat in values:
                diff = values[cat] - ori_value
                diff_data.append({
                    'Song': song,
                    'Category': cat,
                    'Difference': diff,
                    'Original': ori_value
                })

    return pd.DataFrame(diff_data)

def plot_diverging_dots(df):
    """绘制发散点图"""
    plt.figure(figsize=(12, 20))

    # 为每首歌曲创建y位置
    songs = sorted(df['Song'].unique())
    y_pos = np.arange(len(songs))

    # 为每个类别设置颜色和标记
    colors = {
        'rhythm1_4_step1_r30': '#1f77b4',
        'rhythm1_4_step1_r40': '#ff7f0e',
        'rhythm1_4_step2_r30': '#2ca02c',
        'rhythm1_4_step2_r40': '#d62728'
    }

    markers = {
        'rhythm1_4_step1_r30': 'o',
        'rhythm1_4_step1_r40': 's',
        'rhythm1_4_step2_r30': '^',
        'rhythm1_4_step2_r40': 'D'
    }

    # 绘制每个类别的点
    for cat in CATEGORIES[1:]:
        cat_data = df[df['Category'] == cat]
        # 确保数据顺序与y_pos一致
        sorted_data = cat_data.set_index('Song').reindex(songs)['Difference']
        plt.scatter(sorted_data.values, y_pos,
                   color=colors[cat], marker=markers[cat],
                   label=cat, alpha=0.7, s=50)

    # 添加中心线(原始版本)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # 添加95%和99%置信区间线
    overall_diff = df['Difference'].values
    ci95 = 1.96 * np.std(overall_diff) / np.sqrt(len(overall_diff))
    ci99 = 2.576 * np.std(overall_diff) / np.sqrt(len(overall_diff))

    plt.axvline(x=ci95, color='gray', linestyle=':', linewidth=1)
    plt.axvline(x=-ci95, color='gray', linestyle=':', linewidth=1)
    plt.axvline(x=ci99, color='gray', linestyle='-.', linewidth=1)
    plt.axvline(x=-ci99, color='gray', linestyle='-.', linewidth=1)

    # 设置y轴标签
    plt.yticks(y_pos, songs)
    plt.ylim(-1, len(songs))

    # 其他装饰
    plt.title('Performance Difference from Original by Song (n=100)')
    plt.xlabel('Difference from Original (Positive = Worse)')
    plt.ylabel('Song Name')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diverging_dots.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_diverging_boxplot(df):
    """绘制发散箱线图"""
    plt.figure(figsize=(10, 6))

    # 准备数据 - 每个类别的差异
    plot_data = [df[df['Category'] == cat]['Difference'].values for cat in CATEGORIES[1:]]

    # 绘制箱线图(水平)
    box = plt.boxplot(plot_data, vert=False, patch_artist=True,
                     labels=[cat.replace('rhythm1_4_', '') for cat in CATEGORIES[1:]])

    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 添加中心线
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # 添加95%和99%置信区间
    overall_diff = df['Difference'].values
    ci95 = 1.96 * np.std(overall_diff) / np.sqrt(len(overall_diff))
    ci99 = 2.576 * np.std(overall_diff) / np.sqrt(len(overall_diff))

    plt.axvline(x=ci95, color='gray', linestyle=':', linewidth=1)
    plt.axvline(x=-ci95, color='gray', linestyle=':', linewidth=1)
    plt.axvline(x=ci99, color='gray', linestyle='-.', linewidth=1)
    plt.axvline(x=-ci99, color='gray', linestyle='-.', linewidth=1)

    # 装饰
    plt.title('Distribution of Differences from Original (n=100 songs)')
    plt.xlabel('Difference from Original (Positive = Worse)')
    plt.ylabel('Rhythm Pattern')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diverging_boxplot.png'), dpi=300)
    plt.close()

def calculate_percentile_differences(df):
    """计算差异的百分位数"""
    results = {}
    for cat in CATEGORIES[1:]:
        cat_diff = df[df['Category'] == cat]['Difference']
        results[cat] = {
            'Mean Difference': cat_diff.mean(),
            'Median Difference': cat_diff.median(),
            '95% CI Lower': cat_diff.quantile(0.025),
            '95% CI Upper': cat_diff.quantile(0.975),
            '99% CI Lower': cat_diff.quantile(0.005),
            '99% CI Upper': cat_diff.quantile(0.995),
            'p-value (vs 0)': stats.ttest_1samp(cat_diff, 0).pvalue
        }

    return pd.DataFrame(results).T

if __name__ == "__main__":
    print("Loading data...")
    raw_data = load_data()

    print("\nCalculating differences...")
    diff_df = calculate_differences(raw_data)

    print("\nCalculating statistics...")
    stats_df = calculate_percentile_differences(diff_df)
    print("\nDifference Statistics:")
    print(stats_df.to_string(float_format="%.4f"))

    print("\nGenerating visualizations...")
    plot_diverging_dots(diff_df)
    plot_diverging_boxplot(diff_df)

    print("\nAnalysis complete. Results saved to:")
    print("- diverging_dots.png (详细歌曲级差异)")
    print("- diverging_boxplot.png (整体差异分布)")
    print(f"Output directory: {OUTPUT_DIR}")

