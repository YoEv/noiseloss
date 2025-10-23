import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re

def extract_additive_category(filename):
    """从文件名中提取additive合成类别信息"""
    # 匹配模式: beethoven_016_additive_{percentage}p_{source_file}.wav
    pattern = r'beethoven_016_additive_(\d+)p_(.+?)_short30_preview_cut14s_reordered_(\d+)p\.wav'
    match = re.search(pattern, filename)
    
    if match:
        percentage = match.group(1)  # e.g., "5", "10", "15"...
        source_file = match.group(2)  # e.g., "1217199_mozart-rondo-alla-turca"
        source_percentage = match.group(3)  # e.g., "30", "50", "80"
        
        # 返回两个分类：百分比和源文件
        return {
            'percentage_group': f"{percentage}p",
            'source_group': f"{source_file}_{source_percentage}p"
        }
    
    return None

def load_additive_data(file_path):
    """加载additive合成音频的loss数据"""
    percentage_data = {}
    source_data = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                
                filename, loss = line.rsplit(':', 1)
                filename = filename.strip()
                
                try:
                    loss = float(loss.strip())
                except ValueError:
                    continue
                
                # 提取类别
                categories = extract_additive_category(filename)
                if categories:
                    # 按百分比分组
                    percentage_group = categories['percentage_group']
                    if percentage_group not in percentage_data:
                        percentage_data[percentage_group] = []
                    percentage_data[percentage_group].append(loss)
                    
                    # 按源文件分组
                    source_group = categories['source_group']
                    if source_group not in source_data:
                        source_data[source_group] = []
                    source_data[source_group].append(loss)
                    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
    
    return percentage_data, source_data

def calculate_group_statistics(group_data):
    """计算每组的统计信息"""
    stats_data = []
    
    for category, values in group_data.items():
        values_array = np.array(values)
        stats_data.append({
            'Category': category,
            'Mean': np.mean(values_array),
            'Std_Dev': np.std(values_array, ddof=1),
            'Count': len(values_array),
            'Min': np.min(values_array),
            'Max': np.max(values_array)
        })
    
    return pd.DataFrame(stats_data)

def perform_anova_analysis(group_data):
    """执行方差分析"""
    # 将所有组数据放入一个列表
    all_data = []
    all_labels = []
    
    for category, values in group_data.items():
        if len(values) > 0:
            all_data.append(values)
            all_labels.append(category)
    
    if len(all_data) > 1:
        try:
            f_stat, p_value = stats.f_oneway(*all_data)
            return {
                'F_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'groups': all_labels
            }
        except Exception as e:
            print(f"ANOVA error: {e}")
    
    return None

def perform_tukey_hsd(group_data):
    """执行Tukey HSD多重比较"""
    # 准备数据
    all_values = []
    group_indices = []
    group_names = []
    
    for i, (category, values) in enumerate(group_data.items()):
        if len(values) > 0:
            all_values.extend(values)
            group_indices.extend([i] * len(values))
            group_names.append(category)
    
    if len(group_names) > 1:
        try:
            # 执行Tukey HSD测试
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey_results = pairwise_tukeyhsd(all_values, group_indices, alpha=0.05)
            
            # 格式化结果
            result_pairs = []
            for i, (group1, group2, reject) in enumerate(zip(tukey_results.groupsunique[tukey_results.pairindices[:,0]],
                                                         tukey_results.groupsunique[tukey_results.pairindices[:,1]],
                                                         tukey_results.reject)):
                result_pairs.append({
                    'group1': group_names[int(group1)],
                    'group2': group_names[int(group2)],
                    'significant': bool(reject),
                    'p_value': tukey_results.pvalues[i]
                })
            
            return result_pairs
        except Exception as e:
            print(f"Tukey HSD error: {e}")
    
    return None

def create_percentage_boxplot(percentage_data, output_dir):
    """创建按百分比分组的箱线图"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 按百分比排序
    def extract_percentage(key):
        return int(key.replace('p', ''))
    
    sorted_categories = sorted(percentage_data.keys(), key=extract_percentage)
    
    # 准备数据
    data_for_plot = [percentage_data[category] for category in sorted_categories]
    
    # 创建箱线图
    plt.figure(figsize=(15, 8))
    box = plt.boxplot(data_for_plot, labels=sorted_categories, patch_artist=True)
    
    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_for_plot)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Loss Values by Percentage of Additive Synthesis', fontsize=16)
    plt.xlabel('Percentage', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    
    # 添加均值点
    means = [np.mean(percentage_data[category]) for category in sorted_categories]
    plt.plot(range(1, len(sorted_categories) + 1), means, 'ro-', linewidth=2, markersize=8, label='Mean')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'additive_loss_by_percentage.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_source_boxplot(source_data, output_dir):
    """创建按源文件分组的箱线图"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取源文件名和百分比
    source_info = {}
    for key in source_data.keys():
        parts = key.split('_')
        percentage = parts[-1]  # 最后一部分是百分比
        source_name = '_'.join(parts[:-1])  # 前面部分是源文件名
        
        if source_name not in source_info:
            source_info[source_name] = []
        source_info[source_name].append(percentage)
    
    # 创建箱线图
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    source_names = list(source_info.keys())
    data_for_plot = []
    labels_for_plot = []
    
    for source_name in source_names:
        for percentage in sorted(source_info[source_name], key=lambda x: int(x.replace('p', ''))):
            key = f"{source_name}_{percentage}"
            if key in source_data:
                data_for_plot.append(source_data[key])
                labels_for_plot.append(f"{source_name}\n{percentage}")
    
    box = plt.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
    
    # 设置颜色 - 每个源文件一种颜色
    color_map = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(source_names)))
    
    for i, source_name in enumerate(source_names):
        color_map[source_name] = colors[i]
    
    for i, (patch, label) in enumerate(zip(box['boxes'], labels_for_plot)):
        source_name = label.split('\n')[0]
        patch.set_facecolor(color_map[source_name])
        patch.set_alpha(0.7)
    
    plt.title('Loss Values by Source File and Percentage', fontsize=16)
    plt.xlabel('Source File and Percentage', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'additive_loss_by_source.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 文件路径
    input_file = '/home/evev/asap-dataset/Loss/Phase3_2_3/loss_addi_morph_medium_prmpt_classical_music.txt'
    output_dir = 'Loss_Plot/Plot_Additive_prmpt_classical_music_Meduim'
    
    print("Loading additive synthesis loss data...")
    percentage_data, source_data = load_additive_data(input_file)
    
    if not percentage_data or not source_data:
        print("No data found in the input file.")
        return
    
    print(f"Found {len(percentage_data)} percentage categories and {len(source_data)} source categories with data.")
    
    # 计算统计信息
    print("\nCalculating statistics for percentage groups...")
    percentage_stats = calculate_group_statistics(percentage_data)
    
    print("\nCalculating statistics for source groups...")
    source_stats = calculate_group_statistics(source_data)
    
    # 显示统计结果
    print("\nPercentage Group Statistical Summary:")
    print(percentage_stats.sort_values(by='Category').to_string(index=False, float_format='%.6f'))
    
    print("\nSource Group Statistical Summary:")
    print(source_stats.to_string(index=False, float_format='%.6f'))
    
    # 执行方差分析
    print("\nPerforming ANOVA analysis for percentage groups...")
    percentage_anova = perform_anova_analysis(percentage_data)
    
    print("\nPerforming ANOVA analysis for source groups...")
    source_anova = perform_anova_analysis(source_data)
    
    # 执行Tukey HSD多重比较
    print("\nPerforming Tukey HSD analysis for percentage groups...")
    percentage_tukey = perform_tukey_hsd(percentage_data)
    
    print("\nPerforming Tukey HSD analysis for source groups...")
    source_tukey = perform_tukey_hsd(source_data)
    
    # 打印ANOVA结果
    if percentage_anova:
        print("\nANOVA Results for Percentage Groups:")
        print(f"F={percentage_anova['F_statistic']:.4f}, p={percentage_anova['p_value']:.6f}, "
              f"significant={percentage_anova['significant']}")
    
    if source_anova:
        print("\nANOVA Results for Source Groups:")
        print(f"F={source_anova['F_statistic']:.4f}, p={source_anova['p_value']:.6f}, "
              f"significant={source_anova['significant']}")
    
    # 创建可视化
    print("\nGenerating visualizations...")
    create_percentage_boxplot(percentage_data, output_dir)
    create_source_boxplot(source_data, output_dir)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    percentage_stats.to_csv(os.path.join(output_dir, 'percentage_statistics.csv'), index=False)
    source_stats.to_csv(os.path.join(output_dir, 'source_statistics.csv'), index=False)
    
    # 保存ANOVA结果
    with open(os.path.join(output_dir, 'anova_results.txt'), 'w') as f:
        f.write("ANOVA Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        if percentage_anova:
            f.write("Percentage Groups:\n")
            f.write(f"  F-statistic: {percentage_anova['F_statistic']:.6f}\n")
            f.write(f"  p-value: {percentage_anova['p_value']:.6f}\n")
            f.write(f"  Significant: {percentage_anova['significant']}\n")
            f.write(f"  Groups: {percentage_anova['groups']}\n\n")
        
        if source_anova:
            f.write("Source Groups:\n")
            f.write(f"  F-statistic: {source_anova['F_statistic']:.6f}\n")
            f.write(f"  p-value: {source_anova['p_value']:.6f}\n")
            f.write(f"  Significant: {source_anova['significant']}\n")
            f.write(f"  Groups: {source_anova['groups']}\n\n")
    
    # 保存Tukey HSD结果
    with open(os.path.join(output_dir, 'tukey_hsd_results.txt'), 'w') as f:
        f.write("Tukey HSD Multiple Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        if percentage_tukey:
            f.write("Percentage Groups:\n")
            for result in percentage_tukey:
                f.write(f"  {result['group1']} vs {result['group2']}: ")
                f.write(f"significant={result['significant']}, p={result['p_value']:.6f}\n")
            f.write("\n")
        
        if source_tukey:
            f.write("Source Groups:\n")
            for result in source_tukey:
                f.write(f"  {result['group1']} vs {result['group2']}: ")
                f.write(f"significant={result['significant']}, p={result['p_value']:.6f}\n")
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")
    print("Generated files:")
    print("- percentage_statistics.csv: 百分比组统计数据表")
    print("- source_statistics.csv: 源文件组统计数据表")
    print("- anova_results.txt: 方差分析结果")
    print("- tukey_hsd_results.txt: Tukey HSD多重比较结果")
    print("- additive_loss_by_percentage.png: 按百分比分组的箱线图")
    print("- additive_loss_by_source.png: 按源文件分组的箱线图")

if __name__ == "__main__":
    main()