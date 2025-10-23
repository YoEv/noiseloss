import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 配置参数 - 与phase5_region_analysis_3areas.py保持一致
# 删除重复的NOISE_LENGTH, COMPROMISED_END等，因为现在动态计算

PARAMS = {
    'base_dir': '/home/evev/asap-dataset/+Loss/Phase5_1',
    'output_dir': '/home/evev/asap-dataset/+Loss_Plot/Phase5_1/3regions_boxplot',
    'debug': True
}

# 颜色配置 - 用户指定的颜色
REGION_COLORS = {
    'bumped': '#FFFFE0',      # 浅黄色
    'compromised': '#FFB6C1', # 浅粉色
    'regression': '#ADD8E6'   # 浅蓝色
}

# 数据库配置
DATABASES = ['asap', 'ShutterStock_32k', 'Unconditional']
MODEL_TYPE = 'small'  # 只处理small模型

# 噪声配置参数 - 处理多个token长度
NOISE_CONFIGS = {
    'noise_color': ['white'],
    'noise_length': ['50', '100', '150', '200'],  # 处理四个范围
    'noise_volume': ['ori']
}

# 区域检测参数
BUMPED_START_FIXED = 245  # 固定的bumped起始位置
STABLE_LEN = 5  # 连续稳定点数量
# COMPROMISED_END 现在动态计算：249 + token数量
# NOISE_LENGTH 现在从循环中获取，不再是固定值
REGRESSION_START_OFFSET = 50

def detect_regions_ma(ori_loss, mix_loss, compromised_end=403):
    """检测三个区域：bumped, compromised, regression"""
    # 计算差分曲线
    diff_curve = np.array(mix_loss) - np.array(ori_loss)
    
    # 计算基线（前200个token的均值）
    baseline = np.mean(diff_curve[:200])
    
    # 固定bumped区域起始位置
    bumped_start = BUMPED_START_FIXED
    
    # 寻找bumped区域结束位置（第一个回到基线附近的位置）
    bumped_end = bumped_start
    threshold = 0.1  # 阈值
    
    for i in range(bumped_start + 1, min(len(diff_curve), compromised_end)):
        if abs(diff_curve[i] - baseline) < threshold:
            # 检查是否连续STABLE_LEN个点都在阈值内
            stable_count = 0
            for j in range(i, min(i + STABLE_LEN, len(diff_curve))):
                if abs(diff_curve[j] - baseline) < threshold:
                    stable_count += 1
                else:
                    break
            
            if stable_count >= STABLE_LEN:
                bumped_end = i
                break
    
    # 定义compromised区域
    compromised_start = bumped_end
    # compromised_end 现在作为参数传入
    
    # 定义regression区域
    regression_start = compromised_end + REGRESSION_START_OFFSET
    regression_end = min(regression_start + 100, len(diff_curve))
    
    # 计算各区域的平均差值
    def calculate_avg_diff(start, end):
        if start < len(diff_curve) and end <= len(diff_curve) and start < end:
            return np.mean(diff_curve[start:end])
        return 0
    
    return {
        'bumped': {
            'start': bumped_start,
            'end': bumped_end,
            'detected': bumped_end > bumped_start,
            'avg_diff': calculate_avg_diff(bumped_start, bumped_end)
        },
        'compromised': {
            'start': compromised_start,
            'end': compromised_end,
            'detected': compromised_end > compromised_start,
            'avg_diff': calculate_avg_diff(compromised_start, compromised_end)
        },
        'regression': {
            'start': regression_start,
            'end': regression_end,
            'detected': regression_end > regression_start,
            'avg_diff': calculate_avg_diff(regression_start, regression_end)
        }
    }

def load_single_condition_data(condition_dir):
    """加载单个条件目录的所有CSV数据，返回平均loss序列"""
    per_token_dir = os.path.join(condition_dir, 'per_token')
    
    if not os.path.exists(per_token_dir):
        return None
    
    csv_files = [f for f in os.listdir(per_token_dir) if f.endswith('.csv')]
    if not csv_files:
        return None
    
    all_loss_data = []
    
    for csv_file in csv_files:
        file_path = os.path.join(per_token_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            if 'token_position' in df.columns and 'avg_loss_value' in df.columns:
                df = df.dropna().sort_values('token_position')
                loss_values = df['avg_loss_value'].values
                if len(loss_values) > 0:
                    all_loss_data.append(loss_values)
        except Exception as e:
            if PARAMS['debug']:
                print(f"读取文件错误 {csv_file}: {e}")
            continue
    
    if not all_loss_data:
        return None
    
    # 计算所有文件的平均loss序列
    min_length = min(len(seq) for seq in all_loss_data)
    truncated_data = [seq[:min_length] for seq in all_loss_data]
    avg_loss = np.mean(truncated_data, axis=0)
    
    return avg_loss

def collect_database_data(database_name):
    """收集指定数据库的数据"""
    base_dir = PARAMS['base_dir']
    
    database_data = {}
    
    # 遍历所有噪声配置
    for noise_type in NOISE_CONFIGS['noise_color']:
        for noise_length in NOISE_CONFIGS['noise_length']:
            # 构建目录名 - 使用实际的目录命名格式
            if database_name == 'asap':
                ori_dir_name = f"asap_ori_{MODEL_TYPE}"
                noise_dir_name = f"asap_replace_noise_{noise_type}_at5_tk{noise_length}_token_loss_{MODEL_TYPE}"
            elif database_name == 'ShutterStock_32k':
                ori_dir_name = f"ShutterStock_32k_ori_{MODEL_TYPE}"
                noise_dir_name = f"shutter_replace_noise_{noise_type}_at5_tk{noise_length}_token_loss_{MODEL_TYPE}"
            elif database_name == 'Unconditional':
                ori_dir_name = f"Unconditional_ori_{MODEL_TYPE}"
                noise_dir_name = f"unconditional_replace_noise_{noise_type}_at5_tk{noise_length}_token_loss_{MODEL_TYPE}"
            
            ori_dir = os.path.join(base_dir, ori_dir_name)
            noise_dir = os.path.join(base_dir, noise_dir_name)
            
            if PARAMS['debug']:
                print(f"  检查目录: {ori_dir}")
                print(f"  检查目录: {noise_dir}")
            
            if os.path.exists(ori_dir) and os.path.exists(noise_dir):
                if PARAMS['debug']:
                    print(f"  找到目录，开始处理: {noise_type}_{noise_length}")
                
                # 加载数据
                ori_data = load_single_condition_data(ori_dir)
                noise_data = load_single_condition_data(noise_dir)
                
                if ori_data is not None and noise_data is not None:
                    # 计算compromised_end
                    compromised_end = 249 + int(noise_length)
                    
                    # 确保数据长度一致
                    min_len = min(len(ori_data), len(noise_data))
                    ori_data = ori_data[:min_len]
                    noise_data = noise_data[:min_len]
                    
                    # 检测区域
                    regions = detect_regions_ma(ori_data, noise_data, compromised_end)
                    
                    if regions is not None:
                        # 存储数据
                        key = f"{noise_type}_{noise_length}"
                        if key not in database_data:
                            database_data[key] = {'bumped': [], 'compromised': [], 'regression': []}
                        
                        for region_name in ['bumped', 'compromised', 'regression']:
                            if regions[region_name]['detected']:
                                database_data[key][region_name].append(regions[region_name]['avg_diff'])
            else:
                if PARAMS['debug']:
                    print(f"  目录不存在: {ori_dir} 或 {noise_dir}")
    
    return database_data

def collect_all_data():
    """收集所有数据库的数据"""
    all_data = {}
    
    # 改为显示实际处理的配置
    print(f"开始收集数据，处理token长度: {NOISE_CONFIGS['noise_length']}")
    print(f"处理噪声颜色: {NOISE_CONFIGS['noise_color']}")
    
    for database in DATABASES:
        if PARAMS['debug']:
            print(f"\n正在处理数据库: {database}")
        
        data = collect_database_data(database)
        if data and len(data) > 0:
            all_data[database] = data
            if PARAMS['debug']:
                print(f"  ✅ 成功收集 {database} 的数据")
        else:
            if PARAMS['debug']:
                print(f"  ❌ 未能收集到 {database} 的数据")
    
    return all_data

def create_boxplot(all_data):
    """创建箱线图"""
    if not all_data:
        print("没有可用数据进行绘图")
        return
    
    # 准备数据
    plot_data = []
    
    for database in DATABASES:
        if database in all_data:
            database_data = all_data[database]
            
            # 遍历所有噪声配置
            for config_key, config_data in database_data.items():
                noise_type, noise_length = config_key.split('_')
                
                # 为每个区域添加数据
                for region in ['bumped', 'compromised', 'regression']:
                    if region in config_data and len(config_data[region]) > 0:
                        values = config_data[region]
                        for value in values:
                            plot_data.append({
                                'Database': database,
                                'Region': region,
                                'Loss_Difference': value,
                                'Noise_Type': noise_type,
                                'Noise_Length': noise_length
                            })
    
    if not plot_data:
        print("没有有效的数据点进行绘图")
        return
    
    df = pd.DataFrame(plot_data)
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 创建箱线图
    sns.boxplot(data=df, x='Region', y='Loss_Difference', hue='Database', 
                palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # 设置颜色背景
    ax = plt.gca()
    
    # 添加区域背景色
    regions = ['bumped', 'compromised', 'regression']
    colors = [REGION_COLORS['bumped'], REGION_COLORS['compromised'], REGION_COLORS['regression']]
    
    for i, (region, color) in enumerate(zip(regions, colors)):
        ax.axvspan(i-0.4, i+0.4, alpha=0.3, color=color, zorder=0)
    
    plt.title(f'Loss Difference by Region and Database\n(All noise configurations)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Loss Difference (relative to baseline)', fontsize=12)
    plt.legend(title='Database', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(PARAMS['output_dir'], exist_ok=True)
    output_path = os.path.join(PARAMS['output_dir'], 
                              f"regions_boxplot_all_configs.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n箱线图已保存: {output_path}")
    
    # 打印统计信息
    print("\n=== 数据统计 ===")
    for database in DATABASES:
        if database in all_data:
            print(f"\n{database}:")
            database_data = all_data[database]
            for config_key, config_data in database_data.items():
                print(f"  配置 {config_key}:")
                for region in ['bumped', 'compromised', 'regression']:
                    if region in config_data and len(config_data[region]) > 0:
                        values = config_data[region]
                        print(f"    {region}: {len(values)} 个数据点, 平均值: {np.mean(values):.4f}")

def main():
    """主函数"""
    print("开始收集数据...")
    all_data = collect_all_data()
    
    if not all_data:
        print("没有收集到任何数据")
        return
    
    print(f"\n成功收集到 {len(all_data)} 个数据库的数据")
    
    print("\n开始绘制箱线图...")
    create_boxplot(all_data)
    
    print("\n完成！")

if __name__ == "__main__":
    main()