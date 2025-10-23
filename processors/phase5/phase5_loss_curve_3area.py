import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import glob
import re

# 配置参数
STABLE_LEN = 5
BUMPED_START_FIXED = 245

# 输出目录
OUTPUT_DIR = "test_out/loss_curve_3area"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据文件路径
ORIGINAL_FILE = "+Loss/Phase5_1/ShutterStock_32k_ori_medium/per_token/433499_fashionable_short30_preview_cut14s_tokens_avg.csv"
SHUFFLE_FILE = "shutter_shuffle_random_results_medium/433499_fashionable_short30_preview_cut14s_shuffle_random_tk70_tokens_avg.csv"

def load_csv_data(file_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path)
        return df['avg_loss_value'].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def detect_regions_ma(ori_loss, shuffle_loss, window_size=5):
    """检测三个区域：稳定区域、影响区域、恢复区域"""
    # 计算损失差异
    loss_diff = shuffle_loss - ori_loss
    
    # 使用移动平均平滑差异
    loss_diff_smooth = pd.Series(loss_diff).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # 计算阈值（基于标准差）
    threshold = np.std(loss_diff_smooth) * 0.5
    
    # 检测显著变化的区域
    significant_change = np.abs(loss_diff_smooth) > threshold
    
    # 找到连续的影响区域
    change_points = np.diff(significant_change.astype(int))
    start_points = np.where(change_points == 1)[0] + 1
    end_points = np.where(change_points == -1)[0] + 1
    
    # 处理边界情况
    if significant_change.iloc[0]:
        start_points = np.concatenate([[0], start_points])
    if significant_change.iloc[-1]:
        end_points = np.concatenate([end_points, [len(significant_change)]])
    
    # 合并相近的区域
    merged_regions = []
    if len(start_points) > 0 and len(end_points) > 0:
        for start, end in zip(start_points, end_points):
            if len(merged_regions) == 0 or start - merged_regions[-1][1] > 10:
                merged_regions.append([start, end])
            else:
                merged_regions[-1][1] = end
    
    # 分类区域
    stable_regions = []
    affected_regions = merged_regions
    recovery_regions = []
    
    # 在影响区域之间的区域视为恢复区域
    for i in range(len(affected_regions) - 1):
        recovery_start = affected_regions[i][1]
        recovery_end = affected_regions[i + 1][0]
        if recovery_end - recovery_start > 5:
            recovery_regions.append([recovery_start, recovery_end])
    
    # 开始和结束的稳定区域
    if len(affected_regions) > 0:
        if affected_regions[0][0] > 10:
            stable_regions.append([0, affected_regions[0][0]])
        if affected_regions[-1][1] < len(ori_loss) - 10:
            stable_regions.append([affected_regions[-1][1], len(ori_loss)])
    else:
        stable_regions.append([0, len(ori_loss)])
    
    return {
        'stable': stable_regions,
        'affected': affected_regions,
        'recovery': recovery_regions,
        'loss_diff': loss_diff,
        'loss_diff_smooth': loss_diff_smooth.values
    }

def analyze_regions(ori_loss, shuffle_loss, regions):
    """分析各区域的统计信息"""
    results = {}
    
    for region_type, region_list in regions.items():
        if region_type in ['loss_diff', 'loss_diff_smooth']:
            continue
            
        region_stats = []
        for start, end in region_list:
            ori_segment = ori_loss[start:end]
            shuffle_segment = shuffle_loss[start:end]
            diff_segment = shuffle_segment - ori_segment
            
            stats = {
                'start': start,
                'end': end,
                'length': end - start,
                'ori_mean': np.mean(ori_segment),
                'shuffle_mean': np.mean(shuffle_segment),
                'diff_mean': np.mean(diff_segment),
                'diff_std': np.std(diff_segment),
                'ori_std': np.std(ori_segment),
                'shuffle_std': np.std(shuffle_segment)
            }
            region_stats.append(stats)
        
        results[region_type] = region_stats
    
    return results

def plot_loss_curves_with_regions(ori_loss, shuffle_loss, regions, output_path):
    """绘制损失曲线和区域分析图"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # 颜色定义
    colors = {
        'stable': '#90EE90',      # 浅绿色
        'affected': '#FFB6C1',    # 浅粉色  
        'recovery': '#87CEEB'     # 浅蓝色
    }
    
    token_positions = np.arange(len(ori_loss))
    
    # 第一个子图：原始损失曲线
    ax1.plot(token_positions, ori_loss, 'b-', linewidth=1.5, label='Original Loss', alpha=0.8)
    ax1.set_title('Original Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 第二个子图：Shuffle损失曲线
    ax2.plot(token_positions, shuffle_loss, 'r-', linewidth=1.5, label='Shuffle Loss', alpha=0.8)
    ax2.set_title('Shuffle Loss Curve', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 第三个子图：对比分析
    ax3.plot(token_positions, ori_loss, 'b-', linewidth=1.5, label='Original', alpha=0.7)
    ax3.plot(token_positions, shuffle_loss, 'r-', linewidth=1.5, label='Shuffle', alpha=0.7)
    
    # 添加区域背景色
    for region_type, region_list in regions.items():
        if region_type in ['loss_diff', 'loss_diff_smooth']:
            continue
        
        color = colors.get(region_type, '#CCCCCC')
        for start, end in region_list:
            for ax in [ax1, ax2, ax3]:
                ax.axvspan(start, end, alpha=0.3, color=color, label=f'{region_type.title()} Region' if start == region_list[0][0] else "")
    
    ax3.set_title('Loss Comparison with Region Analysis', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Token Position', fontsize=12)
    ax3.set_ylabel('Loss Value', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves plot saved to: {output_path}")

def plot_difference_analysis(regions, output_path):
    """绘制损失差异分析图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 第一个子图：损失差异曲线
    token_positions = np.arange(len(regions['loss_diff']))
    ax1.plot(token_positions, regions['loss_diff'], 'g-', linewidth=1, alpha=0.6, label='Raw Difference')
    ax1.plot(token_positions, regions['loss_diff_smooth'], 'darkgreen', linewidth=2, label='Smoothed Difference')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Loss Difference (Shuffle - Original)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss Difference', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加区域背景色
    colors = {
        'stable': '#90EE90',
        'affected': '#FFB6C1',
        'recovery': '#87CEEB'
    }
    
    for region_type, region_list in regions.items():
        if region_type in ['loss_diff', 'loss_diff_smooth']:
            continue
        
        color = colors.get(region_type, '#CCCCCC')
        for start, end in region_list:
            ax1.axvspan(start, end, alpha=0.3, color=color, label=f'{region_type.title()} Region' if start == region_list[0][0] else "")
    
    # 第二个子图：区域统计
    region_names = []
    mean_diffs = []
    std_diffs = []
    
    for region_type, region_list in regions.items():
        if region_type in ['loss_diff', 'loss_diff_smooth']:
            continue
        
        for i, (start, end) in enumerate(region_list):
            region_names.append(f'{region_type.title()} {i+1}')
            diff_segment = regions['loss_diff'][start:end]
            mean_diffs.append(np.mean(diff_segment))
            std_diffs.append(np.std(diff_segment))
    
    if region_names:
        x_pos = np.arange(len(region_names))
        bars = ax2.bar(x_pos, mean_diffs, yerr=std_diffs, capsize=5, alpha=0.7)
        
        # 为不同类型的区域设置不同颜色
        for i, name in enumerate(region_names):
            if 'Stable' in name:
                bars[i].set_color('#90EE90')
            elif 'Affected' in name:
                bars[i].set_color('#FFB6C1')
            elif 'Recovery' in name:
                bars[i].set_color('#87CEEB')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(region_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Mean Loss Difference by Region', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Loss Difference', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Difference analysis plot saved to: {output_path}")

def save_analysis_results(regions, region_stats, output_dir):
    """保存分析结果到CSV文件"""
    # 保存区域信息
    region_data = []
    for region_type, region_list in regions.items():
        if region_type in ['loss_diff', 'loss_diff_smooth']:
            continue
        
        for i, (start, end) in enumerate(region_list):
            region_data.append({
                'region_type': region_type,
                'region_id': i + 1,
                'start_token': start,
                'end_token': end,
                'length': end - start
            })
    
    if region_data:
        region_df = pd.DataFrame(region_data)
        region_file = os.path.join(output_dir, 'regions_info.csv')
        region_df.to_csv(region_file, index=False)
        print(f"Region information saved to: {region_file}")
    
    # 保存统计信息
    stats_data = []
    for region_type, stats_list in region_stats.items():
        for stats in stats_list:
            stats['region_type'] = region_type
            stats_data.append(stats)
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_file = os.path.join(output_dir, 'region_statistics.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"Region statistics saved to: {stats_file}")

def main():
    """主函数"""
    print("Loading data...")
    
    # 加载数据
    ori_loss = load_csv_data(ORIGINAL_FILE)
    shuffle_loss = load_csv_data(SHUFFLE_FILE)
    
    if ori_loss is None or shuffle_loss is None:
        print("Error: Could not load data files")
        return
    
    if len(ori_loss) != len(shuffle_loss):
        print(f"Warning: Data length mismatch - Original: {len(ori_loss)}, Shuffle: {len(shuffle_loss)}")
        min_len = min(len(ori_loss), len(shuffle_loss))
        ori_loss = ori_loss[:min_len]
        shuffle_loss = shuffle_loss[:min_len]
    
    print(f"Data loaded successfully. Length: {len(ori_loss)} tokens")
    
    # 检测区域
    print("Detecting regions...")
    regions = detect_regions_ma(ori_loss, shuffle_loss)
    
    # 分析区域统计
    print("Analyzing regions...")
    region_stats = analyze_regions(ori_loss, shuffle_loss, regions)
    
    # 打印区域信息
    print("\n=== Region Analysis Results ===")
    for region_type, region_list in regions.items():
        if region_type in ['loss_diff', 'loss_diff_smooth']:
            continue
        print(f"\n{region_type.title()} Regions: {len(region_list)}")
        for i, (start, end) in enumerate(region_list):
            print(f"  Region {i+1}: tokens {start}-{end} (length: {end-start})")
    
    # 绘制图表
    print("\nGenerating plots...")
    
    # 损失曲线图
    curves_output = os.path.join(OUTPUT_DIR, '433499_loss_curves_3area.png')
    plot_loss_curves_with_regions(ori_loss, shuffle_loss, regions, curves_output)
    
    # 差异分析图
    diff_output = os.path.join(OUTPUT_DIR, '433499_difference_analysis.png')
    plot_difference_analysis(regions, diff_output)
    
    # 保存分析结果
    print("\nSaving analysis results...")
    save_analysis_results(regions, region_stats, OUTPUT_DIR)
    
    # 打印总结统计
    print("\n=== Summary Statistics ===")
    print(f"Total tokens: {len(ori_loss)}")
    print(f"Original loss - Mean: {np.mean(ori_loss):.4f}, Std: {np.std(ori_loss):.4f}")
    print(f"Shuffle loss - Mean: {np.mean(shuffle_loss):.4f}, Std: {np.std(shuffle_loss):.4f}")
    print(f"Mean difference (Shuffle - Original): {np.mean(regions['loss_diff']):.4f}")
    print(f"Std difference: {np.std(regions['loss_diff']):.4f}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()