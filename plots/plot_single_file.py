import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import re
import argparse  # 添加argparse模块

# 配置
OUTPUT_DIR = 'Loss_Cal_Plot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_categories(file_paths):
    """提取所有唯一类别"""
    categories = set()
    original_category = '_ori'  # 定义原始类别名称
    categories.add(original_category)  # 确保原始类别始终存在

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                filename = line.rsplit(':', 1)[0].strip()

                # 跳过原始文件
                if '_ori' in filename:
                    continue

                # 使用正则表达式提取类别
                patterns = [
                    r'_(white|pink|blue|brown)_(\d+)db',  # 噪音类型和分贝
                    r'_(low|mid|high)_(\d+)p',            # 频率范围和百分比
                    r'_(fq_dele_random)_(\d+)p',          # 随机频率删除
                    r'(noise)_(\d+)',                     # 噪声和百分比
                    r'(randomnoise)_(\d+)',               # 随机噪声和百分比
                    r'_(velocity|pitch|structure|rhythm)(\d+)_(\d+)_step(\d+)_(\w+\d+)',  # 其他修改类型
                    r'_(velocity|pitch|structure|rhythm)'  # 简单修改类型
                ]

                for pattern in patterns:
                    match = re.search(pattern, filename)
                    if match:
                        if len(match.groups()) >= 2:
                            if match.group(1) in ['velocity', 'pitch', 'structure', 'rhythm'] and len(match.groups()) >= 5:
                                category = f"_{match.group(1)}{match.group(2)}_{match.group(3)}_step{match.group(4)}_{match.group(5)}"
                            else:
                                category = f"_{match.group(1)}_{match.group(2)}"
                        else:
                            category = f"_{match.group(1)}"
                        categories.add(category)
                        break

    # 将原始类别放在首位，其他类别按字母顺序排序
    sorted_categories = [original_category] + sorted([c for c in categories if c != original_category])
    return sorted_categories

def load_data(file_paths):
    """加载数据并组织成结构化格式"""
    categories = get_all_categories(file_paths)
    data = {}

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

                    # 提取曲目名称 - 改进的方法
                    # 检查是否包含_midi_score
                    if '_midi_score' in filename:
                        song_name = filename.split('_midi_score')[0]
                    else:
                        # 尝试从文件名中提取基本名称
                        # 移除所有已知的修改标记
                        song_name = filename
                        for pattern in [
                            r'_(white|pink|blue|brown)_\d+db',
                            r'_(low|mid|high)_\d+p',
                            r'_(fq_dele_random)_\d+p',
                            r'_noise_\d+',
                            r'_randomnoise_\d+',
                            r'_(velocity|pitch|structure|rhythm)\d+_\d+_step\d+_\w+\d+',
                            r'_(velocity|pitch|structure|rhythm)',
                            r'_ori'
                        ]:
                            song_name = re.sub(pattern, '', song_name)
                        
                    # 检查是否为原始文件
                    if '_ori' in filename:
                        if song_name not in data:
                            data[song_name] = {}
                        data[song_name]['_ori'] = loss
                        continue

                    # 匹配适当的类别
                    matched = False
                    for category in categories:
                        if category != '_ori' and category in filename:
                            if song_name not in data:
                                data[song_name] = {}
                            data[song_name][category] = loss
                            matched = True
                            break

                    # 如果没有匹配到任何类别，尝试使用正则表达式
                    if not matched:
                        patterns = [
                            r'_(white|pink|blue|brown)_(\d+)db',
                            r'_(low|mid|high)_(\d+)p',
                            r'_(fq_dele_random)_(\d+)p',
                            r'(noise)_(\d+)',
                            r'(randomnoise)_(\d+)',
                            r'_(velocity|pitch|structure|rhythm)'
                        ]

                        for pattern in patterns:
                            match = re.search(pattern, filename)
                            if match:
                                if len(match.groups()) == 2:
                                    matched_category = f"_{match.group(1)}_{match.group(2)}"
                                else:
                                    matched_category = f"_{match.group(1)}"

                                if matched_category in categories:
                                    if song_name not in data:
                                        data[song_name] = {}
                                    data[song_name][matched_category] = loss
                                    matched = True
                                    break
        except Exception as e:
            print(f"错误处理文件 {file_path}: {str(e)}")

    return data, categories

def calculate_differences(data, categories):
    """计算每个曲目各修改模式与原始版本的差异"""
    diff_data = {}

    for cat in categories[1:]:  # 跳过'_ori'
        cat_diff = []
        for song, values in data.items():
            if '_ori' not in values or cat not in values:
                continue  # 必须有原始版本和当前类别才能比较

            ori_value = values['_ori']
            diff = values[cat] - ori_value
            cat_diff.append({
                'Song': song,
                'Category': cat,
                'Difference': diff,
                'Original': ori_value,
                'Modified': values[cat]
            })
        
        # 将差异数据转换为DataFrame并按差异大小排序
        if cat_diff:
            df = pd.DataFrame(cat_diff)
            df = df.sort_values('Difference', ascending=False)
            diff_data[cat] = df

    return diff_data

def plot_category_comparison(diff_df, category, output_dir):
    """为单个类别绘制与原始版本的对比图"""
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    songs = diff_df['Song'].values
    diffs = diff_df['Difference'].values
    
    # 创建水平条形图
    bars = plt.barh(range(len(songs)), diffs, height=0.7, alpha=0.7)
    
    # 为正负值设置不同颜色
    for i, bar in enumerate(bars):
        if diffs[i] > 0:
            bar.set_color('#d62728')  # 红色表示性能变差
        else:
            bar.set_color('#2ca02c')  # 绿色表示性能改善
    
    # 添加中心线(原始版本)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # 添加95%和99%置信区间线
    ci95 = 1.96 * np.std(diffs) / np.sqrt(len(diffs))
    ci99 = 2.576 * np.std(diffs) / np.sqrt(len(diffs))
    
    plt.axvline(x=ci95, color='gray', linestyle=':', linewidth=1, label='95% CI')
    plt.axvline(x=-ci95, color='gray', linestyle=':', linewidth=1)
    plt.axvline(x=ci99, color='gray', linestyle='-.', linewidth=1, label='99% CI')
    plt.axvline(x=-ci99, color='gray', linestyle='-.', linewidth=1)
    
    # 设置y轴标签
    plt.yticks(range(len(songs)), songs)
    plt.ylim(-1, len(songs))
    
    # 其他装饰
    plt.title(f'Performance Difference: {category} vs Original')
    plt.xlabel('Difference from Original (Positive = Worse)')
    plt.ylabel('Song Name')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图片，文件名使用类别名
    safe_category = category.replace('/', '_').replace('\\', '_')
    plt.savefig(os.path.join(output_dir, f'{safe_category}_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_statistics(diff_data):
    """计算每个类别的统计数据"""
    stats_data = []
    
    for category, df in diff_data.items():
        diffs = df['Difference'].values
        stats_data.append({
            'Category': category,
            'Mean Difference': np.mean(diffs),
            'Median Difference': np.median(diffs),
            'Std Dev': np.std(diffs, ddof=1),
            'Min': np.min(diffs),
            'Max': np.max(diffs),
            '95% CI Lower': np.mean(diffs) - 1.96 * np.std(diffs) / np.sqrt(len(diffs)),
            '95% CI Upper': np.mean(diffs) + 1.96 * np.std(diffs) / np.sqrt(len(diffs)),
            'p-value (vs 0)': stats.ttest_1samp(diffs, 0).pvalue,
            'Count': len(diffs)
        })
    
    return pd.DataFrame(stats_data).set_index('Category')

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='处理MIDI损失数据并生成统计分析')
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='包含损失数据的输入文件路径')
    parser.add_argument('-o', '--output', default=OUTPUT_DIR,
                        help='图表和结果的输出目录')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 更新输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    for file_path in args.input:
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 - {file_path}")
            return
    
    print("加载数据...")
    data, categories = load_data(args.input)
    
    if not data:
        print("错误：未找到有效数据")
        return
    
    print(f"\n找到以下类别: {', '.join(categories)}")
    
    print("\n计算差异...")
    diff_data = calculate_differences(data, categories)
    
    if not diff_data:
        print("错误：无法计算差异，请确保数据中包含原始版本和修改版本")
        return
    
    print("\n计算统计数据...")
    stats_df = calculate_statistics(diff_data)
    print("\n差异统计:")
    print(stats_df.to_string(float_format="%.4f"))
    
    # 保存统计结果
    stats_df.to_csv(os.path.join(output_dir, 'statistical_results.csv'))
    
    print("\n生成可视化...")
    for category, df in diff_data.items():
        print(f"  处理类别: {category}")
        plot_category_comparison(df, category, output_dir)
    
    print(f"\n所有图表已保存至: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()