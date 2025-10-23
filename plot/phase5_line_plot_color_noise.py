import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import glob
from matplotlib.colors import to_rgb

# 基础马卡龙色彩配置 - 调整为更深的颜色
BASE_MACARON_COLORS = {
    'reference': '#4CAF50',  # 更深的绿色
    'white': '#E8E8E8',      # 更深的灰白色
    'pink': '#F8BBD9',       # 更深的粉色
    'blue': '#B3D9FF',       # 更深的蓝色
    'brown': '#DEB887'       # 更深的棕色
}

# 数据集和模型配置
DATASETS = ['asap', 'shutter', 'unconditional']
MODELS = ['small', 'medium', 'mgen-melody', 'large']
TOKEN_LENGTHS = [5, 10, 50, 100, 150, 200]
NOISE_COLORS = ['white', 'pink', 'blue', 'brown']

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

# 生成所有噪音类型的渐变色
GRADIENT_COLORS = {}
for noise_color in NOISE_COLORS:
    GRADIENT_COLORS[noise_color] = generate_gradient_colors(BASE_MACARON_COLORS[noise_color])

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
        
        # 收集噪音数据
        data[dataset]['noise'] = {}
        for noise_color in NOISE_COLORS:
            data[dataset]['noise'][noise_color] = {}
            for token_len in TOKEN_LENGTHS:
                noise_dir = f'{dataset}_replace_noise_{noise_color}_at5_tk{token_len}_token_loss_{model}'
                noise_path = os.path.join(base_dir, noise_dir)
                data[dataset]['noise'][noise_color][f'tk{token_len}'] = load_csv_files(noise_path)
    
    return data

def create_line_plot(model, data, output_dir):
    """创建折线图，X轴按token分组，每个子图显示一个数据集"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Line Plot: {model.title()} Model - All Datasets', fontsize=16, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(DATASETS):
        ax = axes[dataset_idx]
        
        # 获取reference值
        ref_data = data[dataset]['reference']
        ref_mean = np.mean(ref_data) if ref_data is not None else None
        
        # 为每种噪音类型绘制折线
        for noise_color in NOISE_COLORS:
            x_values = []
            y_values = []
            colors = []
            
            for token_idx, token_len in enumerate(TOKEN_LENGTHS):
                token_data = data[dataset]['noise'][noise_color][f'tk{token_len}']
                if token_data is not None:
                    # X轴按token分组：所有噪音都在同一位置，可以重叠
                    x_pos = token_idx * 5 + 2  # 所有噪音都在每组的中心位置
                    x_values.append(x_pos)
                    y_values.append(np.mean(token_data))
                    colors.append(GRADIENT_COLORS[noise_color][token_idx])
            
            if x_values and y_values:
                # 绘制折线，连接同一噪音类型的不同token长度
                ax.plot(x_values, y_values, color=BASE_MACARON_COLORS[noise_color], 
                       linewidth=3, alpha=0.9, label=f'{noise_color.title()} Noise')
                
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
        for noise_color in NOISE_COLORS:
            for token_len in TOKEN_LENGTHS:
                token_data = data[dataset]['noise'][noise_color][f'tk{token_len}']
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
        token_positions = [i * 5 + 2 for i in range(len(TOKEN_LENGTHS))]  # 所有噪音的共同位置
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
    filename = f'line_plot_{model}_all_datasets.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"保存图片: {filepath}")

def main():
    """主函数：为所有模型生成折线图"""
    # 创建输出目录
    output_dir = '/home/evev/asap-dataset/+Loss_Plot/Phase5_1/line_plot_color_noise'
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成折线图...")
    
    # 为每个模型生成图表
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
        create_line_plot(model, data, output_dir)
        
        print(f"完成 {model}")
    
    print(f"所有图表已保存到: {output_dir}")
    print("总共生成了4张折线图，每张图包含3个数据集子图")
    print("每个子图显示4种噪音类型的折线，使用渐变色圆点和线条")
    print("Reference用绿色虚线表示")

if __name__ == '__main__':
    main()