import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re

def read_original_results(file_path):
    """读取原始音频的loss结果（制表符分隔格式）"""
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过标题行
        for line in lines[1:]:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    filename = parts[0]
                    loss_str = parts[1]
                    # 提取基础文件名（去掉.wav后缀）
                    base_name = filename.replace('.wav', '')
                    results[base_name] = float(loss_str)
    return results

def read_shuffle_results(file_path):
    """读取shuffle音频的loss结果（制表符分隔格式）"""
    results = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过标题行
        for line in lines[1:]:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    filename = parts[0]
                    loss_str = parts[1]
                    
                    # 解析文件名以提取歌曲名和token长度
                    # 格式：songname_shuffle_random_tkXX.wav
                    match = re.search(r'(.+)_shuffle_random_tk(\d+)\.wav', filename)
                    if match:
                        song_name = match.group(1)
                        token_length = int(match.group(2))
                        loss_value = float(loss_str)
                        
                        if song_name not in results:
                            results[song_name] = {}
                        results[song_name][token_length] = loss_value
    
    return results

def create_boxplot_data(original_results, shuffle_results):
    """创建用于boxplot的数据结构"""
    # 从shuffle结果中获取所有可用的token长度
    all_token_lengths = set()
    for song_data in shuffle_results.values():
        all_token_lengths.update(song_data.keys())
    token_lengths = sorted(list(all_token_lengths))
    
    # 准备数据
    plot_data = []
    
    # 添加原始数据（作为参考）
    for song_name, original_loss in original_results.items():
        if song_name in shuffle_results:
            plot_data.append({
                'song': song_name,
                'token_length': 'Original',
                'loss': original_loss,
                'category': 'Original'
            })
    
    # 添加shuffle数据
    for song_name, token_data in shuffle_results.items():
        if song_name in original_results:
            for token_length in token_lengths:
                if token_length in token_data:
                    plot_data.append({
                        'song': song_name,
                        'token_length': f'tk{token_length}',
                        'loss': token_data[token_length],
                        'category': 'Shuffled'
                    })
    
    return pd.DataFrame(plot_data), token_lengths

def create_comparison_boxplot(df, token_lengths, output_path=None):
    """创建比较boxplot"""
    plt.figure(figsize=(16, 8))
    
    # 设置颜色 - ori为浅红色，其他为蓝色渐变（从浅到深）
    ori_color = '#FFB3B3'  # 浅红色
    # 生成蓝色渐变色（从浅蓝到深蓝）
    blue_colors = []
    n_tokens = len(token_lengths)
    for i in range(n_tokens):
        # 从浅蓝(#ADD8E6)到深蓝(#003366)的渐变
        intensity = 0.3 + (0.7 * i / max(1, n_tokens - 1))  # 0.3到1.0的强度
        blue_val = int(255 * (1 - intensity))
        blue_colors.append(f'#{blue_val:02x}{blue_val:02x}ff')
    
    colors = [ori_color] + blue_colors
    
    # 创建boxplot
    order = ['Original'] + [f'tk{tl}' for tl in token_lengths]
    
    box_plot = plt.boxplot([df[df['token_length'] == tl]['loss'].values for tl in order],
                          labels=order,
                          patch_artist=True,
                          notch=True,
                          showmeans=True)
    
    # 设置颜色
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置样式
    plt.title('Loss Comparison: Original vs Random Shuffled Audio\n(ShutterStock Dataset - Medium)', 
              fontsize=32, fontweight='bold', pad=20)
    plt.xlabel('Token Length', fontsize=28, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=28, fontweight='bold')
    plt.grid(False)
    plt.xticks(rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    
    # # 添加统计信息
    # stats_text = []
    # for tl in order:
    #     data = df[df['token_length'] == tl]['loss'].values
    #     if len(data) > 0:
    #         mean_val = np.mean(data)
    #         std_val = np.std(data)
    #         stats_text.append(f'{tl}: μ={mean_val:.2f}, σ={std_val:.2f}')
    
    # plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=8, 
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Boxplot saved to: {output_path}")
    
    return plt.gcf()

def create_detailed_analysis(df, token_lengths, output_dir):
    """创建详细分析"""
    # 计算统计数据
    stats_data = []
    
    order = ['Original'] + [f'tk{tl}' for tl in token_lengths]
    
    for tl in order:
        data = df[df['token_length'] == tl]['loss'].values
        if len(data) > 0:
            stats_data.append({
                'Token_Length': tl,
                'Count': len(data),
                'Mean': np.mean(data),
                'Std': np.std(data),
                'Min': np.min(data),
                'Max': np.max(data),
                'Median': np.median(data),
                'Q1': np.percentile(data, 25),
                'Q3': np.percentile(data, 75)
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    # 保存统计数据
    stats_file = output_dir / 'shuffle_random_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"Statistics saved to: {stats_file}")
    
    # 计算相对于原始的变化百分比
    if 'Original' in stats_df['Token_Length'].values:
        original_mean = stats_df[stats_df['Token_Length'] == 'Original']['Mean'].iloc[0]
        
        change_data = []
        for _, row in stats_df.iterrows():
            if row['Token_Length'] != 'Original':
                change_pct = ((row['Mean'] - original_mean) / original_mean) * 100
                change_data.append({
                    'Token_Length': row['Token_Length'],
                    'Loss_Increase_Percent': change_pct
                })
        
        change_df = pd.DataFrame(change_data)
        change_file = output_dir / 'shuffle_random_loss_changes.csv'
        change_df.to_csv(change_file, index=False)
        print(f"Loss changes saved to: {change_file}")
    
    return stats_df

def main():
    """主函数"""
    # 设置路径
    original_file = Path('/home/evev/asap-dataset/+Loss/Phase5_1/ShutterStock_32k_ori_medium/results.txt')
    shuffle_file = Path('/home/evev/asap-dataset/shutter_shuffle_random_results_medium/result.txt')
    output_dir = Path('/home/evev/asap-dataset/+Loss_Plot/Phase5_1/shutter_shuffle_random_results_medium')
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    print("Reading original results...")
    original_results = read_original_results(original_file)
    print(f"Found {len(original_results)} original songs")
    
    print("Reading shuffle results...")
    shuffle_results = read_shuffle_results(shuffle_file)
    print(f"Found {len(shuffle_results)} shuffled songs")
    
    # 创建数据框
    print("Creating comparison data...")
    df, token_lengths = create_boxplot_data(original_results, shuffle_results)
    print(f"Token lengths found: {token_lengths}")
    print(f"Total data points: {len(df)}")
    
    # 创建boxplot
    print("Creating boxplot...")
    output_plot = output_dir / 'shuffle_random_comparison_boxplot.png'
    fig = create_comparison_boxplot(df, token_lengths, output_plot)
    
    # 创建详细分析
    print("Creating detailed analysis...")
    stats_df = create_detailed_analysis(df, token_lengths, output_dir)
    
    # 保存完整数据
    data_file = output_dir / 'shuffle_random_complete_data.csv'
    df.to_csv(data_file, index=False)
    print(f"Complete data saved to: {data_file}")
    
    print("\n=== Analysis Summary ===")
    print(stats_df.to_string(index=False))
    
    print("\nAnalysis complete!")
    plt.show()

if __name__ == "__main__":
    main()