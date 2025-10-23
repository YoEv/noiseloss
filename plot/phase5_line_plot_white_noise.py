import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
import glob
from matplotlib.colors import to_rgb

# 基础马卡龙色彩配置 - 只保留white noise相关颜色
BASE_MACARON_COLORS = {
    'reference': '#4CAF50',  # 更深的绿色
    'white': '#E8E8E8',      # 更深的灰白色
    'asap': '#FFE082',       # 淡黄色
    'shutter': '#F8BBD9',    # 淡粉色
    'unconditional': '#B3E5FC'  # 淡蓝色
}

# 数据集和模型配置
DATASETS = ['asap', 'shutter', 'unconditional']
MODELS = ['small', 'medium', 'mgen-melody', 'large']  # 添加large模型
TOKEN_LENGTHS = [5, 10, 50, 100, 150, 200]
NOISE_COLORS = ['white']  # 只保留white noise

def generate_gradient_colors(base_color, num_steps=6):
    """生成更深的渐变色"""
    base_rgb = to_rgb(base_color)
    colors = []
    
    for i in range(num_steps):
        # 调整加深程度为更深 (0.4 到 0.9)
        factor = 0.9 - (i * 0.5 / (num_steps - 1))
        darker_rgb = tuple(c * factor for c in base_rgb)
        colors.append(darker_rgb)
    
    return colors

# 生成white noise的渐变色
GRADIENT_COLORS = {}
for noise_color in NOISE_COLORS:
    GRADIENT_COLORS[noise_color] = generate_gradient_colors(BASE_MACARON_COLORS[noise_color])

# 为每个数据集生成渐变色
DATASET_GRADIENT_COLORS = {}
for dataset in DATASETS:
    DATASET_GRADIENT_COLORS[dataset] = generate_gradient_colors(BASE_MACARON_COLORS[dataset])

def load_csv_files(directory):
    """加载指定目录下的CSV文件并返回loss值数组"""
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

def collect_data_for_model(model):
    """收集指定模型在所有数据集上的数据"""
    # 根据模型类型选择不同的基础目录
    if model == 'large':
        base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1_Large/Phase5_1'
    else:
        base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1'
    
    data = {}
    
    for dataset in DATASETS:
        data[dataset] = {}
        
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
        data[dataset]['noise'] = {}
        for noise_color in NOISE_COLORS:  # 只有white
            data[dataset]['noise'][noise_color] = {}
            for token_len in TOKEN_LENGTHS:
                noise_dir = f'{dataset}_replace_noise_{noise_color}_at5_tk{token_len}_token_loss_{model}'
                noise_path = os.path.join(base_dir, noise_dir)
                data[dataset]['noise'][noise_color][f'tk{token_len}'] = load_csv_files(noise_path)
    
    return data

def collect_data_for_dataset(dataset):
    """收集指定数据集在所有模型上的数据"""
    data = {}
    
    for model in MODELS:
        data[model] = {}
        
        # 根据模型类型选择不同的基础目录
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
        data[model]['noise'] = {}
        for noise_color in NOISE_COLORS:  # 只有white
            data[model]['noise'][noise_color] = {}
            for token_len in TOKEN_LENGTHS:
                noise_dir = f'{dataset}_replace_noise_{noise_color}_at5_tk{token_len}_token_loss_{model}'
                noise_path = os.path.join(base_dir, noise_dir)
                data[model]['noise'][noise_color][f'tk{token_len}'] = load_csv_files(noise_path)
    
    return data

def create_line_plot_by_model(model, data, output_dir, suffix):
    """创建按模型分组的折线图，X轴按token分组，每个子图显示一个数据集"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Line Plot: {model.title()} Model - White Noise Only', fontsize=16, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(DATASETS):
        ax = axes[dataset_idx]
        
        # 获取reference值
        ref_data = data[dataset]['reference']
        ref_mean = np.mean(ref_data) if ref_data is not None else None
        
        # 为white noise绘制折线
        noise_color = 'white'
        x_values = []
        y_values = []
        colors = []
        
        for token_idx, token_len in enumerate(TOKEN_LENGTHS):
            token_data = data[dataset]['noise'][noise_color][f'tk{token_len}']
            if token_data is not None:
                # X轴按token分组
                x_pos = token_idx * 5 + 2  # 在每组的中心位置
                x_values.append(x_pos)
                y_values.append(np.mean(token_data))
                colors.append(GRADIENT_COLORS[noise_color][token_idx])
        
        if x_values and y_values:
            # 绘制折线，连接white noise的不同token长度
            ax.plot(x_values, y_values, color=BASE_MACARON_COLORS[noise_color], 
                   linewidth=3, alpha=0.9, label='White Noise')
            
            # 绘制数据点，使用更深的渐变色
            for i, (x, y, color) in enumerate(zip(x_values, y_values, colors)):
                ax.scatter(x, y, color=color, s=140, alpha=1.0, 
                         edgecolors='white', linewidth=1.5, zorder=5)
        
        # 绘制reference虚线
        if ref_mean is not None:
            ax.axhline(y=ref_mean, color=BASE_MACARON_COLORS['reference'], 
                      linestyle='--', linewidth=3, alpha=0.9, 
                      label='Reference', zorder=3)
            
            # 添加reference标签
            ax.text(len(TOKEN_LENGTHS) * 5 * 0.8, ref_mean + 0.05, 'Reference', 
                   fontsize=12, fontweight='bold', 
                   color=BASE_MACARON_COLORS['reference'],
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=BASE_MACARON_COLORS['reference'], 
                           alpha=0.5))
        
        # 设置Y轴从2开始
        y_min = 2
        all_y_values = []
        for token_len in TOKEN_LENGTHS:
            token_data = data[dataset]['noise']['white'][f'tk{token_len}']
            if token_data is not None:
                all_y_values.append(np.mean(token_data))
        
        if ref_mean is not None:
            all_y_values.append(ref_mean)
        
        if all_y_values:
            y_max = max(all_y_values) * 1.1
            ax.set_ylim(y_min, y_max)
        
        # 设置X轴 - 按token分组
        x_max = len(TOKEN_LENGTHS) * 5 - 1
        ax.set_xlim(-0.5, x_max + 0.5)
        
        # 设置X轴标签 - 显示token组，标签在每组中心
        token_positions = [i * 5 + 2 for i in range(len(TOKEN_LENGTHS))]
        ax.set_xticks(token_positions)
        ax.set_xticklabels([f'{tk}tk' for tk in TOKEN_LENGTHS], fontsize=12, fontweight='bold')
        
        # 添加分组分隔线
        for i in range(1, len(TOKEN_LENGTHS)):
            ax.axvline(x=i * 5 - 0.5, color='lightgray', linestyle='-', alpha=0.3, linewidth=1)
        
        # 设置子图标题和标签
        ax.set_title(f'{dataset.title()} Dataset', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Token Length Groups', fontsize=12, fontweight='bold')
        if dataset_idx == 0:
            ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        # 添加图例（只在第一个子图添加）
        if dataset_idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f'line_plot_{model}_white_noise{suffix}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"保存图片: {filepath}")

def create_line_plot_by_dataset(dataset, data, output_dir, suffix):
    """创建按数据集分组的折线图，X轴按token分组，每个子图显示一个模型"""
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    fig.suptitle(f'Line Plot: {dataset.title()} Dataset - White Noise Only', fontsize=16, fontweight='bold')
    
    for model_idx, model in enumerate(MODELS):
        ax = axes[model_idx]
        
        # 获取reference值
        ref_data = data[model]['reference']
        ref_mean = np.mean(ref_data) if ref_data is not None else None
        
        # 为white noise绘制折线
        noise_color = 'white'
        x_values = []
        y_values = []
        colors = []
        
        for token_idx, token_len in enumerate(TOKEN_LENGTHS):
            token_data = data[model]['noise'][noise_color][f'tk{token_len}']
            if token_data is not None:
                # X轴按token分组
                x_pos = token_idx * 5 + 2  # 在每组的中心位置
                x_values.append(x_pos)
                y_values.append(np.mean(token_data))
                colors.append(GRADIENT_COLORS[noise_color][token_idx])
        
        if x_values and y_values:
            # 绘制折线，连接white noise的不同token长度
            ax.plot(x_values, y_values, color=BASE_MACARON_COLORS[noise_color], 
                   linewidth=3, alpha=0.9, label='White Noise')
            
            # 绘制数据点，使用更深的渐变色
            for i, (x, y, color) in enumerate(zip(x_values, y_values, colors)):
                ax.scatter(x, y, color=color, s=140, alpha=1.0, 
                         edgecolors='white', linewidth=1.5, zorder=5)
        
        # 绘制reference虚线
        if ref_mean is not None:
            ax.axhline(y=ref_mean, color=BASE_MACARON_COLORS['reference'], 
                      linestyle='--', linewidth=3, alpha=0.9, 
                      label='Reference', zorder=3)
            
            # 添加reference标签
            ax.text(len(TOKEN_LENGTHS) * 5 * 0.8, ref_mean + 0.05, 'Reference', 
                   fontsize=12, fontweight='bold', 
                   color=BASE_MACARON_COLORS['reference'],
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=BASE_MACARON_COLORS['reference'], 
                           alpha=0.5))
        
        # 设置Y轴从2开始
        y_min = 2
        all_y_values = []
        for token_len in TOKEN_LENGTHS:
            token_data = data[model]['noise']['white'][f'tk{token_len}']
            if token_data is not None:
                all_y_values.append(np.mean(token_data))
        
        if ref_mean is not None:
            all_y_values.append(ref_mean)
        
        if all_y_values:
            y_max = max(all_y_values) * 1.1
            ax.set_ylim(y_min, y_max)
        
        # 设置X轴 - 按token分组
        x_max = len(TOKEN_LENGTHS) * 5 - 1
        ax.set_xlim(-0.5, x_max + 0.5)
        
        # 设置X轴标签 - 显示token组，标签在每组中心
        token_positions = [i * 5 + 2 for i in range(len(TOKEN_LENGTHS))]
        ax.set_xticks(token_positions)
        ax.set_xticklabels([f'{tk}tk' for tk in TOKEN_LENGTHS], fontsize=12, fontweight='bold')
        
        # 添加分组分隔线
        for i in range(1, len(TOKEN_LENGTHS)):
            ax.axvline(x=i * 5 - 0.5, color='lightgray', linestyle='-', alpha=0.3, linewidth=1)
        
        # 设置子图标题和标签
        ax.set_title(f'{model.title()} Model', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Token Length Groups', fontsize=12, fontweight='bold')
        if model_idx == 0:
            ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        # 添加图例（只在第一个子图添加）
        if model_idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f'line_plot_{dataset}_white_noise{suffix}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"保存图片: {filepath}")

def collect_data_for_model_difference(model):
    """收集指定模型在所有数据集上的差值数据（相对于reference），包含方差信息和reference值"""
    # 根据模型类型选择不同的基础目录
    if model == 'large':
        base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1_Large/Phase5_1'
    else:
        base_dir = '/home/evev/asap-dataset/+Loss/Phase5_1'
    
    data = {}
    
    for dataset in DATASETS:
        data[dataset] = {}
        
        # 收集reference数据（原始数据）
        if dataset == 'shutter':
            ref_dir = f'ShutterStock_32k_ori_{model}'
        elif dataset == 'unconditional':
            ref_dir = f'Unconditional_ori_{model}'
        else:  # asap
            ref_dir = f'{dataset}_ori_{model}'
        
        ref_path = os.path.join(base_dir, ref_dir)
        ref_data = load_csv_files(ref_path)
        ref_mean = np.mean(ref_data) if ref_data is not None else None
        ref_std = np.std(ref_data) if ref_data is not None else None
        
        # 保存reference值
        data[dataset]['reference_value'] = ref_mean
        
        # 收集white noise差值数据和方差
        data[dataset]['noise_diff'] = {}
        for noise_color in NOISE_COLORS:  # 只有white
            data[dataset]['noise_diff'][noise_color] = {}
            for token_len in TOKEN_LENGTHS:
                noise_dir = f'{dataset}_replace_noise_{noise_color}_at5_tk{token_len}_token_loss_{model}'
                noise_path = os.path.join(base_dir, noise_dir)
                noise_data = load_csv_files(noise_path)
                
                if noise_data is not None and ref_mean is not None:
                    noise_mean = np.mean(noise_data)
                    noise_std = np.std(noise_data)
                    # 计算差值（noise - reference）
                    difference = noise_mean - ref_mean
                    # 计算合并方差（差值的标准差）
                    combined_std = np.sqrt(noise_std**2 + ref_std**2) if ref_std is not None else noise_std
                    data[dataset]['noise_diff'][noise_color][f'tk{token_len}'] = {
                        'mean': difference,
                        'std': combined_std
                    }
                else:
                    data[dataset]['noise_diff'][noise_color][f'tk{token_len}'] = None
    
    return data

def create_line_plot_combined(model, data, output_dir, suffix):
    """创建combined模式的折线图，显示相对于reference的差值"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Difference Plot: {model.title()} Model - White Noise vs Reference', fontsize=16, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(DATASETS):
        ax = axes[dataset_idx]
        
        # 为white noise绘制差值折线
        noise_color = 'white'
        x_values = []
        y_values = []
        colors = []
        
        for token_idx, token_len in enumerate(TOKEN_LENGTHS):
            difference = data[dataset]['noise_diff'][noise_color][f'tk{token_len}']
            if difference is not None:
                # X轴按token分组
                x_pos = token_idx * 5 + 2  # 在每组的中心位置
                x_values.append(x_pos)
                y_values.append(difference)
                colors.append(GRADIENT_COLORS[noise_color][token_idx])
        
        if x_values and y_values:
            # 绘制折线，连接white noise的不同token长度
            ax.plot(x_values, y_values, color=BASE_MACARON_COLORS[noise_color], 
                   linewidth=3, alpha=0.9, label='White Noise Difference')
            
            # 绘制数据点，使用更深的渐变色
            for i, (x, y, color) in enumerate(zip(x_values, y_values, colors)):
                ax.scatter(x, y, color=color, s=140, alpha=1.0, 
                         edgecolors='white', linewidth=1.5, zorder=5)
        
        # 绘制reference线（y=0）
        ax.axhline(y=0, color=BASE_MACARON_COLORS['reference'], 
                  linestyle='--', linewidth=3, alpha=0.9, 
                  label='Reference (0)', zorder=3)
        
        # 添加reference标签
        ax.text(len(TOKEN_LENGTHS) * 5 * 0.8, 0.1, 'Reference', 
               fontsize=12, fontweight='bold', 
               color=BASE_MACARON_COLORS['reference'],
               bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor=BASE_MACARON_COLORS['reference'], 
                       alpha=0.5))
        
        # 设置Y轴范围为-3到+1
        ax.set_ylim(-3, 1)
        
        # 设置X轴 - 按token分组
        x_max = len(TOKEN_LENGTHS) * 5 - 1
        ax.set_xlim(-0.5, x_max + 0.5)
        
        # 设置X轴标签 - 显示token组，标签在每组中心
        token_positions = [i * 5 + 2 for i in range(len(TOKEN_LENGTHS))]
        ax.set_xticks(token_positions)
        ax.set_xticklabels([f'{tk}tk' for tk in TOKEN_LENGTHS], fontsize=12, fontweight='bold')
        
        # 添加分组分隔线
        for i in range(1, len(TOKEN_LENGTHS)):
            ax.axvline(x=i * 5 - 0.5, color='lightgray', linestyle='-', alpha=0.3, linewidth=1)
        
        # 设置子图标题和标签
        ax.set_title(f'{dataset.title()} Dataset', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Token Length Groups', fontsize=12, fontweight='bold')
        if dataset_idx == 0:
            ax.set_ylabel('Loss Difference (vs Reference)', fontsize=12, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        # 添加图例（只在第一个子图添加）
        if dataset_idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f'line_plot_{model}_white_noise{suffix}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"保存图片: {filepath}")

def create_line_plot_combined_single(all_model_data, output_dir, suffix):
    """
    Create a combined line plot with all models in a single figure (1x4 subplots)
    """
    # Construct output path
    output_path = os.path.join(output_dir, f'white_noise_line_plot{suffix}.png')
    
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    fig.suptitle('Combined Line Plot: All Models - White Noise', fontsize=20, fontweight='bold')
    
    for model_idx, model in enumerate(MODELS):
        ax = axes[model_idx]
        ax.set_title(f'{model.title()} Model', fontsize=16, fontweight='bold')
        
        # 收集reference值用于显示
        reference_texts = []
        
        for dataset_idx, dataset in enumerate(DATASETS):
            dataset_data = all_model_data[model][dataset]
            
            # 提取数据
            means = []
            stds = []
            
            for token_idx, token_len in enumerate(TOKEN_LENGTHS):
                noise_data = dataset_data['noise_diff']['white'][f'tk{token_len}']
                means.append(noise_data['mean'])
                stds.append(noise_data['std'])
            
            # 绘制数据点和误差棒
            token_indices = list(range(len(TOKEN_LENGTHS)))
            color = DATASET_GRADIENT_COLORS[dataset][0]
            
            ax.errorbar(token_indices, means, yerr=stds, 
                       marker='o', linewidth=3, markersize=12, 
                       capsize=5, capthick=2, label=dataset.title(), color=color)
            ax.scatter(token_indices, means, s=200, color=color, zorder=5)
            
            # 收集reference值
            ref_value = dataset_data['reference_value']
            reference_texts.append(f"{dataset}: {ref_value:.4f}")
        
        # 设置X轴标注
        ax.set_xticks(range(len(TOKEN_LENGTHS)))
        ax.set_xticklabels([f'{tk}tk' for tk in TOKEN_LENGTHS], fontsize=12, fontweight='bold')
        
        # 设置纵轴范围
        ax.set_ylim(-3, 1)
        
        # 设置标签和格式
        ax.set_xlabel('Token Length Groups', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Difference from Reference', fontsize=12, fontweight='bold')
        
        # 将图例移到右上角
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 在左上角显示reference值
        reference_text = '\n'.join(reference_texts)
        ax.text(0.02, 0.98, reference_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined line plot saved to: {output_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Generate white noise line plots')
    parser.add_argument('--group-by', choices=['dataset', 'model', 'combined'], default='model',
                       help='Group plots by dataset, model, or combined difference plots (default: model)')
    return parser.parse_args()

def main():
    """主函数：根据参数生成white noise折线图"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = '/home/evev/asap-dataset/+Loss_Plot/Phase5_1/line_plot_white_noise'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始生成white noise折线图 (group by {args.group_by})...")
    
    if args.group_by == 'model':
        # 按模型分组：4张图，每张图3个数据集子图
        suffix = '_by_model'
        for model in MODELS:
            print(f"处理 {model} model...")
            
            # 收集数据
            data = collect_data_for_model(model)
            
            # 检查是否有有效数据
            has_data = False
            for dataset in DATASETS:
                if data[dataset]['reference'] is not None:
                    has_data = True
                    break
            
            if not has_data:
                print(f"警告: {model} 没有找到有效数据")
                continue
            
            # 生成折线图
            create_line_plot_by_model(model, data, output_dir, suffix)
            
            print(f"完成 {model}")
        
        print(f"所有图表已保存到: {output_dir}")
        print("总共生成了4张折线图，每张图包含3个数据集子图")
    
    elif args.group_by == 'dataset':
        # 按数据集分组：3张图，每张图4个模型子图
        suffix = '_by_dataset'
        for dataset in DATASETS:
            print(f"处理 {dataset} dataset...")
            
            # 收集数据
            data = collect_data_for_dataset(dataset)
            
            # 检查是否有有效数据
            has_data = False
            for model in MODELS:
                if data[model]['reference'] is not None:
                    has_data = True
                    break
            
            if not has_data:
                print(f"警告: {dataset} 没有找到有效数据")
                continue
            
            # 生成折线图
            create_line_plot_by_dataset(dataset, data, output_dir, suffix)
            
            print(f"完成 {dataset}")
        
        print(f"所有图表已保存到: {output_dir}")
        print("总共生成了3张折线图，每张图包含4个模型子图")
    
    elif args.group_by == 'combined':
        # Combined模式：1张图，包含4个模型子图，每个子图包含3个数据集的差值线
        suffix = '_combined'
        
        # 收集所有模型的数据
        data_all_models = {}
        for model in MODELS:
            print(f"处理 {model} model (combined mode)...")
            
            # 收集差值数据
            data = collect_data_for_model_difference(model)
            
            # 检查是否有有效数据
            has_data = False
            for dataset in DATASETS:
                for token_len in TOKEN_LENGTHS:
                    if data[dataset]['noise_diff']['white'][f'tk{token_len}'] is not None:
                        has_data = True
                        break
                if has_data:
                    break
            
            if not has_data:
                print(f"警告: {model} 没有找到有效数据")
                continue
            
            data_all_models[model] = data
            print(f"完成 {model} 数据收集")
        
        # 生成合并的差值折线图
        if data_all_models:
            create_line_plot_combined_single(data_all_models, output_dir, suffix)
            print(f"所有图表已保存到: {output_dir}")
            print("生成了1张合并的差值折线图，包含4个模型子图，每个子图包含3个数据集的差值线")
            print("纵轴显示相对于reference的差值，范围为-3到1")
            print("asap数据集使用淡黄色渐变，shutter使用淡粉色渐变，unconditional使用淡蓝色渐变")
        else:
            print("警告: 没有找到任何有效数据")
    
    if args.group_by in ['model', 'dataset']:
        print("每个子图只显示white noise的折线，使用渐变色圆点和线条")
        print("Reference用绿色虚线表示")
    elif args.group_by == 'combined':
        print("每个子图显示3个数据集的差值折线，使用不同颜色区分数据集")
        print("Reference值显示在左上角，图例显示在右上角")
        print("纵轴范围固定为-3到1")

if __name__ == '__main__':
    main()