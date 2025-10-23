import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os

# 严格定义6个目标类别
TARGET_CATEGORIES = [
    'original_Salamander',
    'action1_Salamander',
    'action2_Salamander',
    'original_YDP',
    'action1_YDP',
    'action2_YDP'
]

def load_and_process(file_paths):
    """严格按6类别加载数据"""
    category_data = {category: [] for category in TARGET_CATEGORIES}

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
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

                    # 严格分类逻辑
                    for category in TARGET_CATEGORIES:
                        if category in filename:
                            category_data[category].append(loss)
                            break
                    else:
                        continue  # 非目标类别跳过
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # 验证数据完整性
    for cat, values in category_data.items():
        if len(values) == 0:
            print(f"Warning: No data found for {cat}")
        elif len(values) != 100:
            print(f"Warning: Expected 100 samples for {cat}, got {len(values)}")

    return category_data

def calculate_statistics(category_data):
    """计算各组的均值和方差"""
    stats_data = []
    for category in TARGET_CATEGORIES:
        values = np.array(category_data[category])
        stats_data.append({
            'Category': category,
            'Mean': np.mean(values),
            'Variance': np.var(values, ddof=1),  # 样本方差
            'Std Dev': np.std(values, ddof=1),    # 样本标准差
            'Instrument': 'Salamander' if 'Salamander' in category else 'YDP',
            'Action': 'original' if 'original' in category else
                     'action1' if 'action1' in category else 'action2'
        })

    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('Category')

def perform_statistical_tests(category_data):
    """执行统计检验"""
    # 初始化结果存储
    t_test_results = []

    # 对每种乐器单独比较
    for instrument in ['Salamander', 'YDP']:
        # 获取该乐器的三类数据
        original = category_data[f'original_{instrument}']
        action1 = category_data[f'action1_{instrument}']
        action2 = category_data[f'action2_{instrument}']

        # action1 vs original
        t1, p1 = stats.ttest_ind(action1, original)
        t_test_results.append({
            'Comparison': f"{instrument}: action1 vs original",
            't-statistic': t1,
            'p-value': p1,
            'Significant (97%)': p1 < 0.03
        })

        # action2 vs original
        t2, p2 = stats.ttest_ind(action2, original)
        t_test_results.append({
            'Comparison': f"{instrument}: action2 vs original",
            't-statistic': t2,
            'p-value': p2,
            'Significant (97%)': p2 < 0.03
        })

    return {
        'T-tests': pd.DataFrame(t_test_results)
    }

def plot_comparison(stats_df, category_data):
    """绘制6类别对比图"""
    plt.style.use('seaborn')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1. 均值比较柱状图（按乐器分组）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Salamander组
    sal_df = stats_df[stats_df.index.str.contains('Salamander')]
    ax1.bar(sal_df.index, sal_df['Mean'], yerr=sal_df['Std Dev'],
            color=colors[:3], capsize=5, alpha=0.7)
    ax1.set_title('Salamander Group (Mean ± SD)')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # YDP组
    ydp_df = stats_df[stats_df.index.str.contains('YDP')]
    ax2.bar(ydp_df.index, ydp_df['Mean'], yerr=ydp_df['Std Dev'],
            color=colors[3:], capsize=5, alpha=0.7)
    ax2.set_title('YDP Group (Mean ± SD)')
    ax2.set_ylabel('Loss Value')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'instrument_comparison.png'), dpi=300)


    # 2. 方差热力图
    plt.figure(figsize=(10, 6))
    variance_matrix = np.zeros((6, 6))
    for i, cat1 in enumerate(TARGET_CATEGORIES):
        for j, cat2 in enumerate(TARGET_CATEGORIES):
            # 使用F检验比较方差
            var1 = np.var(category_data[cat1], ddof=1)
            var2 = np.var(category_data[cat2], ddof=1)
            variance_matrix[i,j] = var1/var2 if var1 > var2 else var2/var1

    plt.imshow(variance_matrix, cmap='viridis')
    plt.colorbar(label='Variance Ratio (F)')
    plt.xticks(range(6), TARGET_CATEGORIES, rotation=45)
    plt.yticks(range(6), TARGET_CATEGORIES)
    plt.title('Variance Comparison Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variance_heatmap.png'), dpi=300)


    # 3. 分组箱线图
    plt.figure(figsize=(12, 6))
    box_data = [category_data[cat] for cat in TARGET_CATEGORIES]
    plt.boxplot(box_data, labels=TARGET_CATEGORIES, patch_artist=True)

    # 设置颜色（Salamander蓝色系，YDP红色系）
    for i, box in enumerate(plt.gca().artists):
        if i < 3:  # Salamander
            box.set_facecolor(f'C{i}')
        else:      # YDP
            box.set_facecolor(f'C{i-3}')
        box.set_alpha(0.6)

    plt.title('Loss Distribution Across 6 Categories')
    plt.ylabel('Loss Value')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_categories_boxplot.png'), dpi=300)


if __name__ == "__main__":
    # 输入文件路径
    input_files = [
        'Loss_Cal/pitch_mod_Salamander_loss_cal.txt',
        'Loss_Cal/pitch_mod_YDP_loss_cal.txt'
    ]
    output_dir = "Loss_Cal_Plot"  # ← 修改为你的目标路径
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录

    # 1. 数据加载
    print("Loading data...")
    data = load_and_process(input_files)

    # 2. 计算统计量
    print("\nCalculating statistics...")
    stats_df = calculate_statistics(data)
    print("\nDescriptive Statistics:")
    print(stats_df.to_string(float_format="%.4f"))

    # 3. 统计检验
    print("\nPerforming statistical tests...")
    test_results = perform_statistical_tests(data)

    print("\nT-test Comparisons (Same Action, Different Instruments):")
    print(test_results['T-tests'].to_string(index=False))

    # 4. 可视化
    print("\nGenerating visualizations...")
    plot_comparison(stats_df, data)

    print("\nAnalysis complete. Results saved to:")
    print("- instrument_comparison.png")
    print("- variance_heatmap.png")
    print("- all_categories_boxplot.png")