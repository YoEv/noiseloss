import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import glob
from matplotlib.colors import to_rgb
import argparse

# 基础马卡龙色彩配置 - 只保留white noise相关颜色
BASE_MACARON_COLORS = {
    'reference': '#7FB069',  # 深一点的马卡龙绿
    'white': '#F5F5DC',      # 浅马卡龙白
}

# 数据集和模型配置
DATASETS = ['asap', 'shutter', 'unconditional']
MODELS = ['small', 'medium', 'mgen-melody', 'large']
TOKEN_LENGTHS = [5, 10, 50, 100, 150, 200]

def generate_gradient_colors(base_color, num_steps=6):
    """生成从基础颜色到更深颜色的渐变色"""
    base_rgb = to_rgb(base_color)
    colors = []
    
    for i in range(num_steps):
        # 计算加深程度 (0.2 到 0.8)
        factor = 0.8 - (i * 0.6 / (num_steps - 1))
        darker_rgb = tuple(c * factor for c in base_rgb)
        colors.append(darker_rgb)
    
    return colors

# 生成white noise的渐变色
WHITE_GRADIENT_COLORS = generate_gradient_colors(BASE_MACARON_COLORS['white'])

def load_csv_files(directory):
    """加载目录下所有CSV文件并计算均值"""
    csv_files = glob.glob(os.path.join(directory, 'per_token', '*.csv'))
    if not csv_files:
        return None
    
    all_losses = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'avg_loss_value' in df.columns:
                all_losses.extend(df['avg_loss_value'].values)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    return np.array(all_losses) if all_losses else None

def collect_data_for_white_noise_dataset(dataset):
    """收集特定数据集的white noise数据"""
    data = {}
    
    for model in MODELS:
        data[model] = {}
        
        # 根据模型类型选择基础目录
        if model == 'large':
            base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1_Large/Phase5_1'
        else:
            base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1'
        
        # 收集reference数据（原始数据）
        if dataset == 'shutter':
            ref_dir = f'ShutterStock_32k_ori_{model}'
        elif dataset == 'unconditional':
            ref_dir = f'Unconditional_ori_{model}'
        else:  # asap
            ref_dir = f'{dataset}_ori_{model}'
        
        ref_path = os.path.join(base_dir, ref_dir)
        data[model]['reference'] = load_csv_files(ref_path)
        
        # 收集white noise数据
        for token_len in TOKEN_LENGTHS:
            noise_dir = f'{dataset}_replace_noise_white_at5_tk{token_len}_token_loss_{model}'
            noise_path = os.path.join(base_dir, noise_dir)
            data[model][f'tk{token_len}'] = load_csv_files(noise_path)
    
    return data

def collect_data_for_white_noise_model(model):
    """收集特定模型的white noise数据（所有数据集）"""
    data = {}
    
    for dataset in DATASETS:
        data[dataset] = {}
        
        # 根据模型类型选择基础目录
        if model == 'large':
            base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1_Large/Phase5_1'
        else:
            base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1'
        
        # 收集reference数据（原始数据）
        if dataset == 'shutter':
            ref_dir = f'ShutterStock_32k_ori_{model}'
        elif dataset == 'unconditional':
            ref_dir = f'Unconditional_ori_{model}'
        else:  # asap
            ref_dir = f'{dataset}_ori_{model}'
        
        ref_path = os.path.join(base_dir, ref_dir)
        data[dataset]['reference'] = load_csv_files(ref_path)
        
        # 收集white noise数据
        for token_len in TOKEN_LENGTHS:
            noise_dir = f'{dataset}_replace_noise_white_at5_tk{token_len}_token_loss_{model}'
            noise_path = os.path.join(base_dir, noise_dir)
            data[dataset][f'tk{token_len}'] = load_csv_files(noise_path)
    
    return data

def create_white_noise_chart_plot_by_dataset(dataset, data, output_dir, suffix):
    """创建按数据集分组的white noise chart plot（柱状图）- 无error bar，组内柱子紧贴"""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 计算均值
    means = []
    colors = []
    positions = []
    labels = []
    
    pos = 0
    group_positions = []
    bar_width = 0.8  # 柱子宽度
    
    for model_idx, model in enumerate(MODELS):
        group_start = pos
        
        # Reference数据
        ref_data = data[model]['reference']
        if ref_data is not None:
            means.append(np.mean(ref_data))
            colors.append(BASE_MACARON_COLORS['reference'])
            positions.append(pos)
            labels.append('Ref')
            pos += bar_width  # 紧贴排列
        
        # Token长度数据
        for token_idx, token_len in enumerate(TOKEN_LENGTHS):
            token_data = data[model][f'tk{token_len}']
            if token_data is not None:
                means.append(np.mean(token_data))
                colors.append(WHITE_GRADIENT_COLORS[token_idx])
                positions.append(pos)
                labels.append(f'{token_len}')
                pos += bar_width  # 紧贴排列
        
        group_positions.append((group_start, pos - bar_width))
        pos += 2  # 组间间隔保持较大
    
    # 创建柱状图（无error bar）
    if means:
        bars = ax.bar(positions, means, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=0.5, width=bar_width)
    
    # 设置Y轴从2开始
    y_min = 2
    y_max = max(means) * 1.2 if means else 3
    ax.set_ylim(y_min, y_max)
    
    # 设置X轴标签（只显示模型名）- 加大字体并加粗
    model_positions = [(start + end) / 2 for start, end in group_positions]
    ax.set_xticks(model_positions)
    ax.set_xticklabels([model.title() for model in MODELS], fontsize=16, fontweight='bold')
    
    # 设置Y轴刻度标签字体
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 添加组间分隔线
    for i in range(len(group_positions) - 1):
        sep_pos = (group_positions[i][1] + group_positions[i+1][0]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    # 创建图例 - 加大字体并加粗
    legend_elements = []
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=BASE_MACARON_COLORS['reference'], 
                                       alpha=0.7, edgecolor='black', label='Reference'))
    
    for token_idx, token_len in enumerate(TOKEN_LENGTHS):
        legend_elements.append(plt.Rectangle((0,0),1,1, 
                                           facecolor=WHITE_GRADIENT_COLORS[token_idx], 
                                           alpha=0.7, edgecolor='black', 
                                           label=f'tk{token_len}'))
    
    legend = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                      fontsize=14, prop={'weight': 'bold'})
    
    # 设置轴标签 - 加大字体并加粗
    ax.set_ylabel('Mean Per Token Loss', fontsize=18, fontweight='bold')
    ax.set_title(f'White Noise Chart Plot: {dataset.title()} Dataset', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'white_noise_chartplot_{dataset}{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_single_chart_subplot(ax, dataset, data, title):
    """在给定的子图上创建单个chart plot"""
    # 计算均值
    means = []
    colors = []
    positions = []
    labels = []
    
    pos = 0
    group_positions = []
    bar_width = 0.8  # 柱子宽度
    
    # Reference数据
    ref_data = data[dataset]['reference']
    if ref_data is not None:
        means.append(np.mean(ref_data))
        colors.append(BASE_MACARON_COLORS['reference'])
        positions.append(pos)
        labels.append('Ref')
        pos += bar_width  # 紧贴排列
    
    # Token长度数据
    for token_idx, token_len in enumerate(TOKEN_LENGTHS):
        token_data = data[dataset][f'tk{token_len}']
        if token_data is not None:
            means.append(np.mean(token_data))
            colors.append(WHITE_GRADIENT_COLORS[token_idx])
            positions.append(pos)
            labels.append(f'{token_len}')
            pos += bar_width  # 紧贴排列
    
    # 创建柱状图（无error bar）
    if means:
        bars = ax.bar(positions, means, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=0.5, width=bar_width)
    
    # 设置Y轴从2开始
    y_min = 2
    y_max = max(means) * 1.2 if means else 3
    ax.set_ylim(y_min, y_max)
    
    # 设置X轴标签 - 加大字体并加粗
    ax.set_xticks(positions)
    ax.set_xticklabels(['Ref'] + [str(tl) for tl in TOKEN_LENGTHS], fontsize=10, fontweight='bold')
    
    # 设置Y轴刻度标签字体
    ax.tick_params(axis='y', labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 设置轴标签 - 加大字体并加粗
    ax.set_ylabel('Mean Per Token Loss', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

def create_white_noise_chart_plot_by_model(model, data, output_dir, suffix):
    """创建按模型分组的white noise chart plot（柱状图）- 无error bar，组内柱子紧贴"""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 计算均值
    means = []
    colors = []
    positions = []
    labels = []
    
    pos = 0
    group_positions = []
    bar_width = 0.8  # 柱子宽度
    
    for dataset_idx, dataset in enumerate(DATASETS):
        group_start = pos
        
        # Reference数据
        ref_data = data[dataset]['reference']
        if ref_data is not None:
            means.append(np.mean(ref_data))
            colors.append(BASE_MACARON_COLORS['reference'])
            positions.append(pos)
            labels.append('Ref')
            pos += bar_width  # 紧贴排列
        
        # Token长度数据
        for token_idx, token_len in enumerate(TOKEN_LENGTHS):
            token_data = data[dataset][f'tk{token_len}']
            if token_data is not None:
                means.append(np.mean(token_data))
                colors.append(WHITE_GRADIENT_COLORS[token_idx])
                positions.append(pos)
                labels.append(f'{token_len}')
                pos += bar_width  # 紧贴排列
        
        group_positions.append((group_start, pos - bar_width))
        pos += 2  # 组间间隔保持较大
    
    # 创建柱状图（无error bar）
    if means:
        bars = ax.bar(positions, means, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=0.5, width=bar_width)
    
    # 设置Y轴从2开始
    y_min = 2
    y_max = max(means) * 1.2 if means else 3
    ax.set_ylim(y_min, y_max)
    
    # 设置X轴标签（只显示数据集名）- 加大字体并加粗
    dataset_positions = [(start + end) / 2 for start, end in group_positions]
    ax.set_xticks(dataset_positions)
    ax.set_xticklabels([dataset.title() for dataset in DATASETS], fontsize=16, fontweight='bold')
    
    # 设置Y轴刻度标签字体
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 添加组间分隔线
    for i in range(len(group_positions) - 1):
        sep_pos = (group_positions[i][1] + group_positions[i+1][0]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    # 创建图例 - 加大字体并加粗
    legend_elements = []
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=BASE_MACARON_COLORS['reference'], 
                                       alpha=0.7, edgecolor='black', label='Reference'))
    
    for token_idx, token_len in enumerate(TOKEN_LENGTHS):
        legend_elements.append(plt.Rectangle((0,0),1,1, 
                                           facecolor=WHITE_GRADIENT_COLORS[token_idx], 
                                           alpha=0.7, edgecolor='black', 
                                           label=f'tk{token_len}'))
    
    legend = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                      fontsize=14, prop={'weight': 'bold'})
    
    # 设置轴标签 - 加大字体并加粗
    ax.set_ylabel('Mean Per Token Loss', fontsize=18, fontweight='bold')
    ax.set_title(f'White Noise Chart Plot: {model.title()} Model', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'white_noise_chartplot_{model}{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_white_noise_chart_plot_combined(output_dir, suffix):
    """创建合并的white noise chart plot（1x12水平长图布局）"""
    # 创建1行12列的子图布局，图片非常宽
    fig, axes = plt.subplots(1, 12, figsize=(60, 8))
    fig.suptitle('White Noise Chart Plot: All Models Combined (Horizontal Layout)', fontsize=24, fontweight='bold')
    
    subplot_idx = 0
    
    # 为每个模型收集数据并创建子图
    for model_idx, model in enumerate(MODELS):
        print(f"处理 white noise - {model} model...")
        
        # 收集数据
        data = collect_data_for_white_noise_model(model)
        
        # 检查是否有有效数据
        has_data = False
        for dataset in DATASETS:
            if data[dataset]['reference'] is not None:
                has_data = True
                break
        
        if not has_data:
            print(f"警告: white noise - {model} 没有找到有效数据")
            continue
        
        # 为每个数据集创建子图（每个模型3个子图）
        for dataset_idx, dataset in enumerate(DATASETS):
            ax = axes[subplot_idx]
            
            # 创建单个数据集的数据
            single_dataset_data = {dataset: data[dataset]}
            
            # 在子图上绘制
            create_single_chart_subplot(ax, dataset, single_dataset_data, 
                                       f'{model.title()}\n{dataset.title()}')
            
            subplot_idx += 1
    
    # 创建全局图例
    legend_elements = []
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=BASE_MACARON_COLORS['reference'], 
                                       alpha=0.7, edgecolor='black', label='Reference'))
    
    for token_idx, token_len in enumerate(TOKEN_LENGTHS):
        legend_elements.append(plt.Rectangle((0,0),1,1, 
                                           facecolor=WHITE_GRADIENT_COLORS[token_idx], 
                                           alpha=0.7, edgecolor='black', 
                                           label=f'tk{token_len}'))
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=7, fontsize=14, prop={'weight': 'bold'})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(os.path.join(output_dir, f'white_noise_chartplot_combined{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate white noise chart plots')
    parser.add_argument('--group-by', choices=['dataset', 'model', 'combined'], default='dataset',
                       help='Group charts by dataset, model, or create combined plot (default: dataset)')
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = '/home/evev/asap-dataset/+Loss_Plot/Phase5_1/white_noise_chart_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.group_by == 'dataset':
        # 按数据集分组（原有逻辑）
        suffix = '_by_dataset'
        print("开始生成按数据集分组的white noise chart plot可视化图表...")
        
        # 为每个数据集生成white noise图表
        for dataset in DATASETS:
            print(f"处理 white noise - {dataset} dataset...")
            
            # 收集数据
            data = collect_data_for_white_noise_dataset(dataset)
            
            # 检查是否有有效数据
            has_data = False
            for model in MODELS:
                if data[model]['reference'] is not None:
                    has_data = True
                    break
            
            if not has_data:
                print(f"警告: white noise - {dataset} 没有找到有效数据")
                continue
            
            # 生成white noise chart plot
            create_white_noise_chart_plot_by_dataset(dataset, data, output_dir, suffix)
            
            print(f"完成 white noise - {dataset}")
        
        print(f"所有图表已保存到: {output_dir}")
        print("总共生成了3张white noise chart plot图（按数据集分组）")
        print("每幅图包含4个模型组（包括large），每组7列（1个reference + 6个token长度）")
    
    elif args.group_by == 'model':
        # 按模型分组（新增逻辑）
        suffix = '_by_model'
        print("开始生成按模型分组的white noise chart plot可视化图表...")
        
        # 为每个模型生成white noise图表
        for model in MODELS:
            print(f"处理 white noise - {model} model...")
            
            # 收集数据
            data = collect_data_for_white_noise_model(model)
            
            # 检查是否有有效数据
            has_data = False
            for dataset in DATASETS:
                if data[dataset]['reference'] is not None:
                    has_data = True
                    break
            
            if not has_data:
                print(f"警告: white noise - {model} 没有找到有效数据")
                continue
            
            # 生成white noise chart plot
            create_white_noise_chart_plot_by_model(model, data, output_dir, suffix)
            
            print(f"完成 white noise - {model}")
        
        print(f"所有图表已保存到: {output_dir}")
        print("总共生成了4张white noise chart plot图（按模型分组）")
        print("每幅图包含3个数据集组，每组7列（1个reference + 6个token长度）")
    
    else:  # group_by == 'combined'
        # 合并所有模型到一张水平长图（新增逻辑）
        suffix = '_by_model_combined'
        print("开始生成合并的white noise chart plot可视化图表...")
        
        # 生成合并的white noise chart plot
        create_white_noise_chart_plot_combined(output_dir, suffix)
        
        print(f"所有图表已保存到: {output_dir}")
        print("总共生成了1张合并的white noise chart plot图（1x12水平长图布局）")
        print("图表包含4个模型，每个模型3个数据集子图，水平排列成一条长图")

if __name__ == '__main__':
    main()