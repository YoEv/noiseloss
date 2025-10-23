import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re

def read_rhythm_deletion_results(file_path):
    """读取rhythm deletion的loss结果"""
    results = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and ': ' in line:
                parts = line.split(': ')
                if len(parts) >= 2:
                    filename = parts[0]
                    loss_str = parts[1]
                    
                    try:
                        loss_value = float(loss_str)
                        
                        # 解析文件名以提取歌曲名和rhythm deletion类型
                        if '_ori_cut15s.wav' in filename or '.mid_ori_cut15s.wav' in filename:
                            # 原始文件
                            song_name = filename.replace('_ori_cut15s.wav', '').replace('.mid_ori_cut15s.wav', '')
                            category = 'ori'
                        else:
                            # rhythm deletion文件
                            match = re.search(r'(.+)_rhythm1_4_step1_r(\d+)_cut15s\.wav', filename)
                            if match:
                                song_name = match.group(1)
                                deletion_percent = int(match.group(2))
                                category = f'r{deletion_percent}'
                            else:
                                continue
                        
                        if song_name not in results:
                            results[song_name] = {}
                        
                        results[song_name][category] = loss_value
                        
                    except ValueError:
                        continue
    
    return results

def create_boxplot_data(rhythm_results):
    """创建用于boxplot的数据结构"""
    data = []
    
    for song_name, categories in rhythm_results.items():
        for category, loss_value in categories.items():
            data.append({
                'song': song_name,
                'category': category,
                'loss': loss_value
            })
    
    df = pd.DataFrame(data)
    return df

def create_dual_rhythm_deletion_boxplot(small_df, medium_df, output_path=None):
    """创建双模型rhythm deletion的boxplot比较图"""
    # 设置图形大小和样式
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_style("whitegrid")
    
    # 定义类别顺序
    category_order = ['ori', 'r10', 'r20', 'r30', 'r40']
    
    # 为small模型定义渐变绿色
    def get_green_colors(categories):
        colors = {}
        colors['ori'] = '#FFB3B3'  # 浅红色
        
        rhythm_categories = [cat for cat in categories if cat != 'ori']
        if rhythm_categories:
            # 生成从浅绿到深绿的渐变
            green_intensities = np.linspace(0.2, 0.8, len(rhythm_categories))
            for i, cat in enumerate(rhythm_categories):
                intensity = green_intensities[i]
                # RGB格式的绿色渐变
                colors[cat] = (1-intensity, 1, 1-intensity*0.5)
        return colors
    
    # 为medium模型定义渐变蓝色
    def get_blue_colors(categories):
        colors = {}
        colors['ori'] = '#FFB3B3'  # 浅红色
        
        rhythm_categories = [cat for cat in categories if cat != 'ori']
        if rhythm_categories:
            # 生成从浅蓝到深蓝的渐变
            blue_intensities = np.linspace(0.2, 0.8, len(rhythm_categories))
            for i, cat in enumerate(rhythm_categories):
                intensity = blue_intensities[i]
                # RGB格式的蓝色渐变
                colors[cat] = (1-intensity*0.5, 1-intensity, 1)
        return colors
    
    # 处理small模型数据
    small_available_categories = [cat for cat in category_order if cat in small_df['category'].unique()]
    small_df_filtered = small_df[small_df['category'].isin(small_available_categories)]
    small_colors = get_green_colors(small_available_categories)
    
    # 创建small模型的boxplot
    small_box_plot = ax1.boxplot(
        [small_df_filtered[small_df_filtered['category'] == cat]['loss'].values for cat in small_available_categories],
        labels=[cat.upper() if cat == 'ori' else f'{cat[1:]}%' for cat in small_available_categories],
        patch_artist=True,
        notch=True,
        showmeans=True
    )
    
    # 设置small模型颜色
    for i, category in enumerate(small_available_categories):
        small_box_plot['boxes'][i].set_facecolor(small_colors[category])
        small_box_plot['boxes'][i].set_alpha(0.7)
    
    # 设置small模型图形属性
    ax1.set_title('Small Model: Original vs Rhythm Deletion', fontsize=28, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=24)
    ax1.grid(False)
    
    # 处理medium模型数据
    medium_available_categories = [cat for cat in category_order if cat in medium_df['category'].unique()]
    medium_df_filtered = medium_df[medium_df['category'].isin(medium_available_categories)]
    medium_colors = get_blue_colors(medium_available_categories)
    
    # 创建medium模型的boxplot
    medium_box_plot = ax2.boxplot(
        [medium_df_filtered[medium_df_filtered['category'] == cat]['loss'].values for cat in medium_available_categories],
        labels=[cat.upper() if cat == 'ori' else f'{cat[1:]}%' for cat in medium_available_categories],
        patch_artist=True,
        notch=True,
        showmeans=True
    )
    
    # 设置medium模型颜色
    for i, category in enumerate(medium_available_categories):
        medium_box_plot['boxes'][i].set_facecolor(medium_colors[category])
        medium_box_plot['boxes'][i].set_alpha(0.7)
    
    # 设置medium模型图形属性
    ax2.set_title('Medium Model: Original vs Rhythm Deletion', fontsize=28, fontweight='bold')
    # ax2.set_xlabel('Rhythm Deletion Level', fontsize=24)
    # ax2.set_ylabel('Loss Value', fontsize=24)
    ax2.tick_params(axis='x', labelsize=28)
    ax2.tick_params(axis='y', labelsize=28)
    ax2.grid(False)
    
    # 添加统计信息到small模型
    small_stats_text = []
    for category in small_available_categories:
        cat_data = small_df_filtered[small_df_filtered['category'] == category]['loss']
        if len(cat_data) > 0:
            mean_val = cat_data.mean()
            std_val = cat_data.std()
            count = len(cat_data)
            label = category.upper() if category == 'ori' else f'{category[1:]}%'
            small_stats_text.append(f'{label}: μ={mean_val:.3f}, σ={std_val:.3f}, n={count}')
    
    # 添加统计信息到medium模型
    medium_stats_text = []
    for category in medium_available_categories:
        cat_data = medium_df_filtered[medium_df_filtered['category'] == category]['loss']
        if len(cat_data) > 0:
            mean_val = cat_data.mean()
            std_val = cat_data.std()
            count = len(cat_data)
            label = category.upper() if category == 'ori' else f'{category[1:]}%'
            medium_stats_text.append(f'{label}: μ={mean_val:.3f}, σ={std_val:.3f}, n={count}')
    
    # # 在图上添加统计信息
    # ax1.text(0.02, 0.98, '\n'.join(small_stats_text), transform=ax1.transAxes, fontsize=9,
    #          verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # ax2.text(0.02, 0.98, '\n'.join(medium_stats_text), transform=ax2.transAxes, fontsize=9,
    #          verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # 设置共享的轴标签
    fig.text(0.5, 0.01, 'Rhythm Deletion Level', ha='center', fontsize=28, fontweight='bold')
    fig.text(0.06, 0.5, 'Loss Value', va='center', rotation='vertical', fontsize=28, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.1)  # 为轴标签留出空间
    
    # 保存图形
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dual boxplot saved to: {output_path}")
    
    plt.show()
    
    return small_df_filtered, medium_df_filtered

def create_detailed_analysis(small_df, medium_df, output_dir):
    """创建详细的统计分析"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 计算统计信息
    small_stats = small_df.groupby('category')['loss'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
    medium_stats = medium_df.groupby('category')['loss'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
    
    # 保存统计信息
    stats_file = output_dir / 'dual_rhythm_deletion_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Dual Model Rhythm Deletion Loss Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Small Model Statistics:\n")
        f.write("-" * 25 + "\n")
        f.write(small_stats.to_string())
        f.write("\n\n")
        
        f.write("Medium Model Statistics:\n")
        f.write("-" * 25 + "\n")
        f.write(medium_stats.to_string())
        f.write("\n\n")
        
        # 添加模型间比较
        f.write("Model Comparison (Medium - Small):\n")
        f.write("-" * 35 + "\n")
        
        common_categories = set(small_df['category'].unique()) & set(medium_df['category'].unique())
        for category in sorted(common_categories):
            small_mean = small_df[small_df['category'] == category]['loss'].mean()
            medium_mean = medium_df[medium_df['category'] == category]['loss'].mean()
            diff = medium_mean - small_mean
            label = category.upper() if category == 'ori' else f'{category[1:]}% Deletion'
            f.write(f"{label}: {diff:+.4f}\n")
    
    print(f"Detailed analysis saved to: {stats_file}")
    return small_stats, medium_stats

def main():
    """主函数"""
    # 文件路径
    small_file = "+Loss/Phase3_2_1/loss_asap_rhythm_small.txt"
    medium_file = "+Loss/Phase3_2_1/loss_asap_rhythm_medium.txt"
    output_dir = "+Loss_Plot/Phase5_1/dual_rhythm_deletion_results"
    
    # 检查输入文件是否存在
    if not Path(small_file).exists():
        print(f"Error: Input file {small_file} not found!")
        return
    
    if not Path(medium_file).exists():
        print(f"Error: Input file {medium_file} not found!")
        return
    
    print("Reading rhythm deletion results...")
    small_results = read_rhythm_deletion_results(small_file)
    medium_results = read_rhythm_deletion_results(medium_file)
    
    if not small_results or not medium_results:
        print("No valid data found in the input files!")
        return
    
    print(f"Small model: Found data for {len(small_results)} songs")
    print(f"Medium model: Found data for {len(medium_results)} songs")
    
    # 创建DataFrame
    small_df = create_boxplot_data(small_results)
    medium_df = create_boxplot_data(medium_results)
    
    print(f"Small model categories: {sorted(small_df['category'].unique())}")
    print(f"Medium model categories: {sorted(medium_df['category'].unique())}")
    print(f"Small model data points: {len(small_df)}")
    print(f"Medium model data points: {len(medium_df)}")
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 创建双模型boxplot
    output_path = Path(output_dir) / "dual_rhythm_deletion_boxplot.png"
    small_df_filtered, medium_df_filtered = create_dual_rhythm_deletion_boxplot(small_df, medium_df, output_path)
    
    # 创建详细分析
    small_stats, medium_stats = create_detailed_analysis(small_df_filtered, medium_df_filtered, output_dir)
    
    print("\nAnalysis completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()