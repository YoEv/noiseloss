import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os
import re

# 设置全局样式
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'figure.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 300
})

def load_real_data():
    """从Loss_Cal文件夹加载真实数据"""
    categories = [
        '_ori',
        'rhythm1_4_step1_r40','pitch1_3_step1_dia80',
        'velocity1_1_step1_nt80','tempo1_1_step1_nt80']
    # categories = [
    #     '_ori',
    #     'tempo1_1_step1_nt10','tempo1_1_step1_nt30',
    #     'tempo1_1_step1_nt50','tempo1_1_step1_nt80']
    # categories = [
    #     '_ori',
    #     'velocity1_1_step1_nt10','velocity1_1_step1_nt30',
    #     'velocity1_1_step1_nt50','velocity1_1_step1_nt80']
    # categories = [
    #     '_ori', '_pitch1_1_step1_semi10', '_pitch1_1_step1_semi30',
    #     '_pitch1_1_step1_semi50', '_pitch1_1_step1_semi80',
    #     '_pitch1_2_step1_oct10','_pitch1_2_step1_oct30',
    #     '_pitch1_2_step1_oct50','_pitch1_2_step1_oct80',
    #     '_pitch1_3_step1_dia10','_pitch1_3_step1_dia30',
    #     '_pitch1_3_step1_dia50','_pitch1_3_step1_dia80']
    # categories = [
    #     '_ori', 'rhythm1_4_step1_r30', 'rhythm1_4_step1_r40',
    #     'rhythm1_4_step2_r30', 'rhythm1_4_step2_r40']
    data = {cat: [] for cat in categories}

    # 定义文件路径
    input_files = [
        os.path.join('Loss_Cal', 'loss_cal_multi_actions_compare.txt')
        # os.path.join('Loss_Cal', 'structure_mod_loss_cal.txt')
        # os.path.join('Loss_Cal', 'loss_cal_velocity.txt')
        # os.path.join('Loss_Cal', 'loss_cal_0427_fixed.txt') # pitch semi oct dia
        #os.path.join('Loss_Cal', 'rhythm_mod_loss_cal_ori_add.txt')
        #os.path.join('Loss_Cal', 'pitch_mod_Salamander_loss_cal.txt'),
        #os.path.join('Loss_Cal', 'pitch_mod_YDP_loss_cal.txt')
    ]

    # 检查文件是否存在
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    # 解析文件内容
    for file_path in input_files:
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

                # 分类到对应的类别
                matched = False
                for cat in categories:
                    if cat.replace('_', '') in filename.replace('_', ''):
                        data[cat].append(loss)
                        matched = True
                        break

                if not matched:
                    print(f"警告: 未分类的数据行: {filename}")

    # 验证数据完整性
    # for cat, values in data.items():
    #     if len(values) != 100:
    #         print(f"警告: 类别 {cat} 的数据量不足100个 (实际:{len(values)})")

    return data

# 2. 可视化函数
def plot_faceted_histograms(data, save_path):
    """方案1：分面直方图（1行5列）"""
    fig, axes = plt.subplots(1, 5, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (category, values) in enumerate(data.items()):
        ax = axes[idx]

        # 计算自动分箱（Freedman-Diaconis规则）
        q25, q75 = np.percentile(values, [25, 75])
        bin_width = 2 * (q75 - q25) * len(values) ** (-1/3)
        bins = round((max(values) - min(values)) / bin_width)

        ax.hist(values, bins=min(bins, 20), density=False,
                color=f'C{idx}', edgecolor='white', alpha=0.8)

        # 添加统计信息
        stats_text = (f'$\mu$ = {np.mean(values):.2f}\n'
                     f'$\sigma$ = {np.std(values):.2f}\n'
                     f'n = {len(values)}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))

        ax.set_title(category.replace('_', ' '), pad=12)
        ax.set_xlabel('Loss Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('Faceted Histograms of Loss Distributions', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'faceted_histograms.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_overlaid_pdfs(data, save_path):
    """方案2：叠加概率密度图"""
    plt.figure(figsize=(12, 7))

    # 计算全局坐标范围
    x_min = min(np.min(values) for values in data.values())
    x_max = max(np.max(values) for values in data.values())
    x_range = np.linspace(x_min - 1, x_max + 1, 500)

    # 为每组数据绘制KDE曲线
    for idx, (category, values) in enumerate(data.items()):
        kde = gaussian_kde(values)
        y = kde(x_range)
        plt.plot(x_range, y, lw=2.5,
                 label=category.replace('_', ' '))

        # 标记均值线
        mean = np.mean(values)
        plt.axvline(mean, color=f'C{idx}', linestyle='--', alpha=0.5)
        plt.text(mean, kde(mean)*1.05, f'{mean:.2f}',
                 ha='center', va='bottom', color=f'C{idx}')

    plt.title('Probability Density Functions Comparison', pad=20)
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.xlim(x_min - 0.5, x_max + 0.5)
    plt.savefig(os.path.join(save_path, 'overlaid_pdfs.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

# def plot_hist_pdf_combined(data, save_path):
#     """方案3：直方图与PDF组合图（分组展示）"""
#     # 按乐器类型分组
#     groups = {
#         'Salamander': [k for k in data.keys() if 'Salamander' in k],
#         'YDP': [k for k in data.keys() if 'YDP' in k]
#     }

#     for group_name, categories in groups.items():
#         plt.figure(figsize=(12, 6))

#         # 计算全局坐标范围
#         group_values = np.concatenate([data[cat] for cat in categories])
#         x_min, x_max = np.min(group_values), np.max(group_values)
#         x_range = np.linspace(x_min - 1, x_max + 1, 500)

#         for cat in categories:
#             values = data[cat]

#             # 绘制直方图（概率密度形式）
#             plt.hist(values, bins=15, density=True,
#                      alpha=0.3, label=f'{cat.replace("_", " ")} Hist')

#             # 绘制PDF曲线
#             kde = gaussian_kde(values)
#             plt.plot(x_range, kde(x_range), lw=2,
#                      label=f'{cat.replace("_", " ")} PDF')

#         plt.title(f'{group_name}: Histogram with PDF Overlay', pad=15)
#         plt.xlabel('Loss Value')
#         plt.ylabel('Density')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.3)
#         plt.savefig(os.path.join(save_path, f'hist_pdf_{group_name.lower()}.png'),
#                     bbox_inches='tight', dpi=300)
#         plt.close()

def calculate_category_statistics(data):
    """
    Calculate mean and standard deviation for each category.

    Args:
        data: Dictionary with categories as keys and lists of values as values

    Returns:
        DataFrame with statistics for each category
    """
    import pandas as pd

    stats_data = []
    for category, values in data.items():
        values_array = np.array(values)
        stats_data.append({
            'Category': category,
            'Mean': np.mean(values_array),
            'Std Dev': np.std(values_array, ddof=1),  # Sample standard deviation
            'Variance': np.var(values_array, ddof=1),  # Sample variance
            'Count': len(values_array),
            'Min': np.min(values_array),
            'Max': np.max(values_array)
        })

    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('Category')

def compare_categories(data, confidence_levels=[0.95, 0.99]):
    """
    Compare categories using t-tests at specified confidence levels.

    Args:
        data: Dictionary with categories as keys and lists of values as values
        confidence_levels: List of confidence levels (default: [0.95, 0.99])

    Returns:
        DataFrame with comparison results
    """
    import pandas as pd
    from scipy import stats

    categories = list(data.keys())
    comparison_results = []

    # Use the first category (usually '_ori') as reference
    reference_category = categories[0]
    reference_data = data[reference_category]

    for category in categories[1:]:  # Skip the reference category
        category_data = data[category]

        # Perform t-test
        t_stat, p_val = stats.ttest_ind(category_data, reference_data, equal_var=False)

        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(category_data) - np.mean(reference_data)
        pooled_std = np.sqrt((np.var(category_data, ddof=1) + np.var(reference_data, ddof=1)) / 2)
        cohen_d = mean_diff / pooled_std

        # Check significance at different confidence levels
        significant_95 = p_val < 0.05
        significant_99 = p_val < 0.01

        comparison_results.append({
            'Comparison': f"{category} vs {reference_category}",
            'Mean Difference': mean_diff,
            't-statistic': t_stat,
            'p-value': p_val,
            'Cohen\'s d': cohen_d,
            'Significant at 95%': significant_95,
            'Significant at 99%': significant_99
        })

    return pd.DataFrame(comparison_results)

def plot_category_means(stats_df, output_dir):
    """
    Create a bar plot showing means with error bars for each category.

    Args:
        stats_df: DataFrame with statistics for each category
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Create bar plot with error bars
    bars = plt.bar(stats_df.index, stats_df['Mean'], yerr=stats_df['Std Dev'],
                  capsize=5, alpha=0.7)

    # Highlight the first category (usually original)
    if len(bars) > 0:
        bars[0].set_color('#1f77b4')
        bars[0].set_alpha(1.0)

    plt.title('Mean Loss by Category (with Standard Deviation)')
    plt.ylabel('Loss Value')
    plt.xlabel('Category')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'category_means.png'), dpi=300)
    plt.close()

def main():
    # 创建输出目录
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载真实数据
        print("正在从Loss_Cal文件夹加载数据...")
        data = load_real_data()

        # 检查数据完整性
        print("\n数据加载完成，各类别数据量:")
        for cat, values in data.items():
            print(f"- {cat}: {len(values)}个样本")

        # 计算统计量
        print("\n计算统计量...")
        stats_df = calculate_category_statistics(data)
        print("\n描述性统计:")
        print(stats_df.to_string(float_format="%.4f"))

        # 执行类别比较
        print("\n执行类别比较...")
        comparison_results = compare_categories(data)
        print("\n比较结果 (95% 和 99% 置信区间):")
        print(comparison_results.to_string(index=False, float_format="%.4f"))

        # 生成可视化
        print("\n生成可视化图表...")
        plot_faceted_histograms(data, output_dir)
        plot_overlaid_pdfs(data, output_dir)
        plot_category_means(stats_df, output_dir)
        #plot_hist_pdf_combined(data, output_dir)

        print(f"\n所有图表已保存至: {os.path.abspath(output_dir)}")
        print("生成的文件包括:")
        print("- faceted_histograms.png (分面直方图)")
        print("- overlaid_pdfs.png (叠加PDF图)")
        print("- category_means.png (类别均值比较图)")
        #print("- hist_pdf_salamander.png (Salamander组合图)")
        #print("- hist_pdf_ydp.png (YDP组合图)")

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        print("请检查:")
        print("1. Loss_Cal文件夹是否存在")
        print("2. 文件中数据格式是否正确")

if __name__ == "__main__":
    main()