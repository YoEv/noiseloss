import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import glob
from matplotlib.colors import to_rgb

# 基础马卡龙色彩配置
BASE_MACARON_COLORS = {
    'reference': '#7FB069',  # 深一点的马卡龙绿
    'white': '#F5F5DC',      # 浅马卡龙白
    'pink': '#FFB6C1',       # 浅马卡龙粉
    'blue': '#B0E0E6',       # 浅马卡龙蓝
    'brown': '#D2B48C'       # 浅马卡龙棕
}

# 数据集和模型配置
DATASETS = ['asap', 'shutter', 'unconditional']
MODELS = ['small', 'medium', 'mgen-melody', 'large']
TOKEN_LENGTHS = [5, 10, 50, 100, 150, 200]
NOISE_COLORS = ['white', 'pink', 'blue', 'brown']

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

# 生成所有噪音类型的渐变色
GRADIENT_COLORS = {}
for noise_color in NOISE_COLORS:
    GRADIENT_COLORS[noise_color] = generate_gradient_colors(BASE_MACARON_COLORS[noise_color])

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

def collect_data_for_noise_dataset(noise_color, dataset):
    """收集特定噪音类型和数据集的所有数据"""
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
        
        # 收集噪音数据
        for token_len in TOKEN_LENGTHS:
            noise_dir = f'{dataset}_replace_noise_{noise_color}_at5_tk{token_len}_token_loss_{model}'
            noise_path = os.path.join(base_dir, noise_dir)
            data[model][f'tk{token_len}'] = load_csv_files(noise_path)
    
    return data

def create_chart_plot_new(noise_color, dataset, data, output_dir):
    """创建新布局的chart plot（柱状图）- 无error bar"""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 计算均值
    means = []
    colors = []
    positions = []
    labels = []
    
    pos = 1
    group_positions = []
    
    for model_idx, model in enumerate(MODELS):
        group_start = pos
        
        # Reference数据
        ref_data = data[model]['reference']
        if ref_data is not None:
            means.append(np.mean(ref_data))
            colors.append(BASE_MACARON_COLORS['reference'])
            positions.append(pos)
            labels.append('Ref')
            pos += 1
        
        # Token长度数据
        for token_idx, token_len in enumerate(TOKEN_LENGTHS):
            token_data = data[model][f'tk{token_len}']
            if token_data is not None:
                means.append(np.mean(token_data))
                colors.append(GRADIENT_COLORS[noise_color][token_idx])
                positions.append(pos)
                labels.append(f'{token_len}')
                pos += 1
        
        group_positions.append((group_start, pos - 1))
        pos += 2  # 组间间隔
    
    # 创建柱状图（无error bar）
    if means:
        bars = ax.bar(positions, means, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=0.5, width=0.8)
    
    # 设置Y轴从2开始
    y_min = 2
    y_max = max(means) * 1.2 if means else 3
    ax.set_ylim(y_min, y_max)
    
    # 设置X轴标签（只显示模型名）
    model_positions = [(start + end) / 2 for start, end in group_positions]
    ax.set_xticks(model_positions)
    ax.set_xticklabels([model.title() for model in MODELS])
    
    # 添加组间分隔线
    for i in range(len(group_positions) - 1):
        sep_pos = (group_positions[i][1] + group_positions[i+1][0]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    # 创建图例
    legend_elements = []
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=BASE_MACARON_COLORS['reference'], 
                                       alpha=0.7, edgecolor='black', label='Reference'))
    
    for token_idx, token_len in enumerate(TOKEN_LENGTHS):
        legend_elements.append(plt.Rectangle((0,0),1,1, 
                                           facecolor=GRADIENT_COLORS[noise_color][token_idx], 
                                           alpha=0.7, edgecolor='black', 
                                           label=f'tk{token_len}'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    ax.set_ylabel('Mean Per Token Loss')
    ax.set_title(f'Chart Plot: {dataset.title()} Dataset - {noise_color.title()} Noise')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'chartplot_{dataset}_{noise_color}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 创建输出目录
    output_dir = '/home/evev/asap-dataset/+Loss_Plot/Phase5_1/color_noise_comparison_new'
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成新布局的可视化图表...")
    
    # 为每个噪音类型和数据集组合生成图表
    for noise_color in NOISE_COLORS:
        for dataset in DATASETS:
            print(f"处理 {noise_color} noise - {dataset} dataset...")
            
            # 收集数据
            data = collect_data_for_noise_dataset(noise_color, dataset)
            
            # 检查是否有有效数据
            has_data = False
            for model in MODELS:
                if data[model]['reference'] is not None:
                    has_data = True
                    break
            
            if not has_data:
                print(f"警告: {noise_color} - {dataset} 没有找到有效数据")
                continue
            
            # 只生成chart plot（移除了box plot）
            create_chart_plot_new(noise_color, dataset, data, output_dir)
            
            print(f"完成 {noise_color} - {dataset}")
    
    print(f"所有图表已保存到: {output_dir}")
    print("总共生成了12张chart plot图")
    print("每幅图包含4个模型组（包括large），每组7列（1个reference + 6个token长度）")

if __name__ == '__main__':
    main()