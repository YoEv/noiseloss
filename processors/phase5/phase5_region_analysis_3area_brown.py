import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# 配置参数
NOISE_LENGTH = 150  # tk150
COMPROMISED_END = 403  # compromised区域结束位置
STABLE_LEN = 5  # 连续稳定点数量
BUMPED_START_FIXED = 245  # 250 - 1 - 4，噪音开始作用的地方

# 数据路径
ORIGINAL_PATH = "+Loss/Phase5_1/ShutterStock_32k_ori_small/per_token"
NOISE_PATH = "+Loss/Phase5_1/shutter_replace_noise_brown_at5_tk150_token_loss_small/per_token"
BROWN_REFERENCE_PATH = "+Loss/Phase5_1/noise_color_ori_small/per_token/brown_noise_12db_15.0s_tokens_avg.csv"
OUTPUT_DIR = "test_out/brown"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv_data(file_path):
    """
    加载CSV文件并返回avg_loss_value列
    
    Args:
        file_path: str, CSV文件路径
    
    Returns:
        np.array, avg_loss_value数据
    """
    try:
        df = pd.read_csv(file_path)
        return df['avg_loss_value'].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def detect_regions_ma(ori_loss, mix_loss):
    """
    检测三个区域：bumped, compromised, regression
    
    Args:
        ori_loss:   np.array, 原始音乐的 per-token loss
        mix_loss:   np.array, 注入噪音后的 per-token loss
    
    Returns:
        dict: 包含三个区域起始和结束位置的字典
    """
    diff = mix_loss - ori_loss
    baseline = np.mean(diff[:200])  # 使用前200个token作为baseline
    
    # Bumped区域：固定起始位置
    bumped_start = BUMPED_START_FIXED
    
    # 寻找bumped区域结束位置：连续STABLE_LEN个点都<=baseline
    bumped_end = None
    for i in range(bumped_start + STABLE_LEN, len(diff)):
        if all(diff[j] <= baseline for j in range(i - STABLE_LEN, i)):
            bumped_end = i - STABLE_LEN
            break
    
    # 如果没找到bumped结束点，使用默认值
    if bumped_end is None:
        bumped_end = bumped_start + 50  # 默认bumped区域长度
    
    # Compromised区域：从bumped结束到固定位置403
    compromised_start = bumped_end
    compromised_end = min(COMPROMISED_END, len(diff) - 1)
    
    # Regression区域：从compromised结束到序列末尾
    regression_start = compromised_end
    regression_end = len(diff) - 1
    
    return {
        'bumped': (bumped_start, bumped_end),
        'compromised': (compromised_start, compromised_end),
        'regression': (regression_start, regression_end),
        'baseline': baseline
    }

def plot_regions_analysis_with_brown_ref(ori_loss, mix_loss, brown_ref_loss, regions, filename, output_dir):
    """
    绘制区域分析图（包含brown noise reference）
    
    Args:
        ori_loss: np.array, 原始loss
        mix_loss: np.array, 混合loss
        brown_ref_loss: np.array, brown noise reference loss
        regions: dict, 区域信息
        filename: str, 文件名
        output_dir: str, 输出目录
    """
    diff = mix_loss - ori_loss
    
    # 确保所有数组长度一致
    min_len = len(ori_loss)
    if brown_ref_loss is not None:
        min_len = min(min_len, len(brown_ref_loss))
        brown_ref_truncated = brown_ref_loss[:min_len]
        ori_loss_truncated = ori_loss[:min_len]
        brown_diff = brown_ref_truncated - ori_loss_truncated
    else:
        brown_diff = None
        brown_ref_truncated = None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 上图：原始loss、混合loss和brown reference
    ax1.plot(ori_loss, label='Original Loss', alpha=0.7, color='blue', linewidth=1.5)
    ax1.plot(mix_loss, label='Mixed Loss (with brown noise)', alpha=0.7, color='red', linewidth=1.5)
    
    if brown_ref_truncated is not None:
        ax1.plot(brown_ref_truncated, label='Brown Noise Reference', alpha=0.7, color='brown', linewidth=1.5, linestyle='--')
    
    ax1.axhline(y=regions['baseline'] + np.mean(ori_loss[:200]), color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    # 标注区域
    bumped_start, bumped_end = regions['bumped']
    compromised_start, compromised_end = regions['compromised']
    regression_start, regression_end = regions['regression']
    
    ax1.axvspan(bumped_start, bumped_end, alpha=0.3, color='orange', label='Bumped')
    ax1.axvspan(compromised_start, compromised_end, alpha=0.3, color='red', label='Compromised')
    ax1.axvspan(regression_start, regression_end, alpha=0.3, color='green', label='Regression')
    
    ax1.set_title(f'Brown Noise Loss Analysis: {filename}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：差值曲线对比
    ax2.plot(diff, label='Loss Difference (Mixed - Original)', color='purple', alpha=0.8, linewidth=1.5)
    
    if brown_diff is not None:
        # 确保brown_diff长度与diff一致
        brown_diff_plot = brown_diff[:len(diff)]
        ax2.plot(brown_diff_plot, label='Brown Reference Difference', color='brown', alpha=0.8, linewidth=1.5, linestyle='--')
    
    ax2.axhline(y=regions['baseline'], color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    # 标注区域
    ax2.axvspan(bumped_start, bumped_end, alpha=0.3, color='orange', label='Bumped')
    ax2.axvspan(compromised_start, compromised_end, alpha=0.3, color='red', label='Compromised')
    ax2.axvspan(regression_start, regression_end, alpha=0.3, color='green', label='Regression')
    
    ax2.set_title('Loss Difference Analysis with Brown Noise Reference', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Loss Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{filename}_brown_regions_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存: {output_path}")

def analyze_single_file_with_brown_ref(ori_file, noise_file, brown_ref_loss, output_dir):
    """
    分析单个文件对（包含brown noise reference）
    
    Args:
        ori_file: str, 原始文件路径
        noise_file: str, 噪音文件路径
        brown_ref_loss: np.array, brown noise reference数据
        output_dir: str, 输出目录
    
    Returns:
        dict: 分析结果
    """
    # 加载数据
    ori_loss = load_csv_data(ori_file)
    mix_loss = load_csv_data(noise_file)
    
    if ori_loss is None or mix_loss is None:
        return None
    
    # 确保数据长度一致 - 选择最短的长度
    min_len = min(len(ori_loss), len(mix_loss))
    if brown_ref_loss is not None:
        min_len = min(min_len, len(brown_ref_loss))
    
    ori_loss = ori_loss[:min_len]
    mix_loss = mix_loss[:min_len]
    
    # 检测区域
    regions = detect_regions_ma(ori_loss, mix_loss)
    
    # 提取文件名用于图片标题
    filename = os.path.basename(ori_file).replace('_tokens_avg.csv', '')
    
    # 绘制图片（包含brown reference）
    plot_regions_analysis_with_brown_ref(ori_loss, mix_loss, brown_ref_loss, regions, filename, output_dir)
    
    # 计算区域统计信息
    diff = mix_loss - ori_loss
    bumped_start, bumped_end = regions['bumped']
    compromised_start, compromised_end = regions['compromised']
    regression_start, regression_end = regions['regression']
    
    # 计算brown reference的区域统计（如果存在）
    brown_stats = {}
    if brown_ref_loss is not None:
        brown_ref_truncated = brown_ref_loss[:min_len]
        brown_diff = brown_ref_truncated - ori_loss
        brown_stats = {
            'brown_bumped_avg_diff': np.mean(brown_diff[bumped_start:bumped_end]) if bumped_end > bumped_start else 0,
            'brown_compromised_avg_diff': np.mean(brown_diff[compromised_start:compromised_end]) if compromised_end > compromised_start else 0,
            'brown_regression_avg_diff': np.mean(brown_diff[regression_start:regression_end]) if regression_end > regression_start else 0,
        }
    
    result = {
        'filename': filename,
        'data_length': len(ori_loss),
        'regions': regions,
        'bumped_length': bumped_end - bumped_start,
        'compromised_length': compromised_end - compromised_start,
        'regression_length': regression_end - regression_start,
        'bumped_avg_diff': np.mean(diff[bumped_start:bumped_end]) if bumped_end > bumped_start else 0,
        'compromised_avg_diff': np.mean(diff[compromised_start:compromised_end]) if compromised_end > compromised_start else 0,
        'regression_avg_diff': np.mean(diff[regression_start:regression_end]) if regression_end > regression_start else 0,
        **brown_stats  # 添加brown noise统计
    }
    
    return result

def plot_summary_statistics_with_brown(results, output_dir):
    """
    绘制汇总统计图（包含brown noise对比）
    
    Args:
        results: list, 分析结果列表
        output_dir: str, 输出目录
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 区域长度分布
    bumped_lens = [r['bumped_length'] for r in results]
    compromised_lens = [r['compromised_length'] for r in results]
    regression_lens = [r['regression_length'] for r in results]
    
    ax1.hist([bumped_lens, compromised_lens, regression_lens], 
             label=['Bumped', 'Compromised', 'Regression'], 
             alpha=0.7, bins=20)
    ax1.set_title('Region Length Distribution', fontweight='bold')
    ax1.set_xlabel('Length (tokens)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 平均差值分布对比（包含brown reference）
    bumped_diffs = [r['bumped_avg_diff'] for r in results]
    compromised_diffs = [r['compromised_avg_diff'] for r in results]
    regression_diffs = [r['regression_avg_diff'] for r in results]
    
    # Brown noise数据（如果存在）
    brown_bumped_diffs = [r.get('brown_bumped_avg_diff', 0) for r in results if 'brown_bumped_avg_diff' in r]
    brown_compromised_diffs = [r.get('brown_compromised_avg_diff', 0) for r in results if 'brown_compromised_avg_diff' in r]
    brown_regression_diffs = [r.get('brown_regression_avg_diff', 0) for r in results if 'brown_regression_avg_diff' in r]
    
    ax2.hist([bumped_diffs, compromised_diffs, regression_diffs], 
             label=['Bumped (Mixed)', 'Compromised (Mixed)', 'Regression (Mixed)'], 
             alpha=0.7, bins=20, color=['orange', 'red', 'green'])
    
    if brown_bumped_diffs:
        ax2.hist([brown_bumped_diffs, brown_compromised_diffs, brown_regression_diffs], 
                 label=['Bumped (Brown Ref)', 'Compromised (Brown Ref)', 'Regression (Brown Ref)'], 
                 alpha=0.5, bins=20, color=['orange', 'red', 'green'], linestyle='--', histtype='step', linewidth=2)
    
    ax2.set_title('Average Loss Difference Distribution', fontweight='bold')
    ax2.set_xlabel('Average Loss Difference')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 箱线图 - 区域长度
    ax3.boxplot([bumped_lens, compromised_lens, regression_lens], 
                labels=['Bumped', 'Compromised', 'Regression'])
    ax3.set_title('Region Length Box Plot', fontweight='bold')
    ax3.set_ylabel('Length (tokens)')
    ax3.grid(True, alpha=0.3)
    
    # 箱线图 - 平均差值对比
    if brown_bumped_diffs:
        # 创建对比数据
        mixed_data = [bumped_diffs, compromised_diffs, regression_diffs]
        brown_data = [brown_bumped_diffs, brown_compromised_diffs, brown_regression_diffs]
        
        positions1 = [1, 3, 5]
        positions2 = [1.5, 3.5, 5.5]
        
        bp1 = ax4.boxplot(mixed_data, positions=positions1, widths=0.4, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', alpha=0.7))
        bp2 = ax4.boxplot(brown_data, positions=positions2, widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor='lightcoral', alpha=0.7))
        
        ax4.set_xticks([1.25, 3.25, 5.25])
        ax4.set_xticklabels(['Bumped', 'Compromised', 'Regression'])
        ax4.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Mixed with Brown', 'Brown Reference'], loc='upper right')
    else:
        ax4.boxplot([bumped_diffs, compromised_diffs, regression_diffs], 
                    labels=['Bumped', 'Compromised', 'Regression'])
    
    ax4.set_title('Average Loss Difference Box Plot Comparison', fontweight='bold')
    ax4.set_ylabel('Average Loss Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存汇总图
    summary_path = os.path.join(output_dir, "brown_regions_summary_statistics.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"汇总统计图已保存: {summary_path}")

def main():
    """
    主函数：批量分析所有文件（包含brown noise reference）
    """
    print("开始Brown Noise区域分析...")
    print(f"噪音长度: {NOISE_LENGTH}")
    print(f"Compromised区域结束位置: {COMPROMISED_END}")
    print(f"Bumped区域固定起始位置: {BUMPED_START_FIXED}")
    print(f"Brown Reference路径: {BROWN_REFERENCE_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("-" * 50)
    
    # 加载brown noise reference
    print("加载Brown Noise Reference...")
    brown_ref_loss = load_csv_data(BROWN_REFERENCE_PATH)
    if brown_ref_loss is None:
        print(f"警告: 无法加载Brown Noise Reference文件 {BROWN_REFERENCE_PATH}")
        print("将继续分析但不包含reference对比")
    else:
        print(f"成功加载Brown Noise Reference，数据长度: {len(brown_ref_loss)}")
    
    # 获取所有原始文件
    ori_files = glob.glob(os.path.join(ORIGINAL_PATH, "*.csv"))
    
    results = []
    successful_analyses = 0
    
    for ori_file in ori_files:
        # 构造对应的噪音文件路径
        filename = os.path.basename(ori_file)
        noise_file = os.path.join(NOISE_PATH, filename)
        
        if not os.path.exists(noise_file):
            print(f"警告: 找不到对应的噪音文件 {noise_file}")
            continue
        
        print(f"分析文件: {filename}")
        
        # 分析文件
        result = analyze_single_file_with_brown_ref(ori_file, noise_file, brown_ref_loss, OUTPUT_DIR)
        
        if result:
            results.append(result)
            successful_analyses += 1
            
            # 打印区域信息
            regions = result['regions']
            print(f"  Bumped: {regions['bumped'][0]}-{regions['bumped'][1]} (长度: {result['bumped_length']})")
            print(f"  Compromised: {regions['compromised'][0]}-{regions['compromised'][1]} (长度: {result['compromised_length']})")
            print(f"  Regression: {regions['regression'][0]}-{regions['regression'][1]} (长度: {result['regression_length']})")
            print(f"  平均差值 - Bumped: {result['bumped_avg_diff']:.4f}, Compromised: {result['compromised_avg_diff']:.4f}, Regression: {result['regression_avg_diff']:.4f}")
            
            # 打印brown reference统计（如果存在）
            if 'brown_bumped_avg_diff' in result:
                print(f"  Brown Ref差值 - Bumped: {result['brown_bumped_avg_diff']:.4f}, Compromised: {result['brown_compromised_avg_diff']:.4f}, Regression: {result['brown_regression_avg_diff']:.4f}")
        else:
            print(f"  分析失败")
        
        print()
    
    # 生成汇总统计
    if results:
        print("=" * 50)
        print("汇总统计:")
        print(f"总文件数: {len(ori_files)}")
        print(f"成功分析: {successful_analyses}")
        print(f"成功率: {successful_analyses/len(ori_files)*100:.1f}%")
        
        # 计算平均值
        avg_bumped_len = np.mean([r['bumped_length'] for r in results])
        avg_compromised_len = np.mean([r['compromised_length'] for r in results])
        avg_regression_len = np.mean([r['regression_length'] for r in results])
        
        avg_bumped_diff = np.mean([r['bumped_avg_diff'] for r in results])
        avg_compromised_diff = np.mean([r['compromised_avg_diff'] for r in results])
        avg_regression_diff = np.mean([r['regression_avg_diff'] for r in results])
        
        print(f"平均区域长度 - Bumped: {avg_bumped_len:.1f}, Compromised: {avg_compromised_len:.1f}, Regression: {avg_regression_len:.1f}")
        print(f"平均差值 - Bumped: {avg_bumped_diff:.4f}, Compromised: {avg_compromised_diff:.4f}, Regression: {avg_regression_diff:.4f}")
        
        # Brown reference统计
        if brown_ref_loss is not None and 'brown_bumped_avg_diff' in results[0]:
            avg_brown_bumped = np.mean([r['brown_bumped_avg_diff'] for r in results])
            avg_brown_compromised = np.mean([r['brown_compromised_avg_diff'] for r in results])
            avg_brown_regression = np.mean([r['brown_regression_avg_diff'] for r in results])
            print(f"Brown Ref平均差值 - Bumped: {avg_brown_bumped:.4f}, Compromised: {avg_brown_compromised:.4f}, Regression: {avg_brown_regression:.4f}")
        
        # 保存详细结果到CSV
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(OUTPUT_DIR, "brown_regions_analysis_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"详细结果已保存到: {results_csv_path}")
        
        # 生成汇总图
        plot_summary_statistics_with_brown(results, OUTPUT_DIR)
        
    else:
        print("没有成功分析的文件")

if __name__ == "__main__":
    main()