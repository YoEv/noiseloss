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

# 数据路径
OUTPUT_DIR = "+Loss_Plot/Phase5_1/3regions_violinplot_gradient"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 渐变色配置：每个区域使用不同颜色系，每个数据库使用不同深度
GRADIENT_COLORS = {
    "bumped": {  # Peak区域 - 黄色系
        "asap": "#FFFACD",        # 浅黄
        "shutter": "#FFD700",     # 中黄
        "unconditional": "#DAA520"  # 深黄
    },
    "compromised": {  # Assimilation区域 - 粉色系
        "asap": "#FFE4E1",        # 浅粉
        "shutter": "#FFB6C1",     # 中粉
        "unconditional": "#FF69B4"  # 深粉
    },
    "regression": {  # Recovery区域 - 蓝色系
        "asap": "#E6F3FF",        # 浅蓝
        "shutter": "#87CEEB",     # 中蓝
        "unconditional": "#4169E1"  # 深蓝
    }
}

# 统一分组定义：顺序与配色
GROUP_ORDER = ["asap", "shutter", "unconditional"]
GROUP_META = {
    "asap": {"color": "#FFFACD", "label": "asap"},
    "shutter": {"color": "#F8BBD9", "label": "shutter"},
    "unconditional": {"color": "#B3D9FF", "label": "unconditional"},
}

# 三种模型（small / mgen-melody / medium / large）下的三组数据（asap/shutter/unconditional）
MODELS = {
    "small": {
        "asap": {
            "ori": "+Loss/Phase5_1/asap_ori_small/per_token",
            "noise_pattern": "+Loss/Phase5_1/asap_replace_noise_white_at5_tk{tk}_token_loss_small/per_token",
            "color": "#FFFACD",
            "label": "asap",
        },
        "shutter": {
            "ori": "+Loss/Phase5_1/ShutterStock_32k_ori_small/per_token",
            "noise_pattern": "+Loss/Phase5_1/shutter_replace_noise_white_at5_tk{tk}_token_loss_small/per_token",
            "color": "#FFB6C1",
            "label": "shutter",
        },
        "unconditional": {
            "ori": "+Loss/Phase5_1/Unconditional_ori_small/per_token",
            "noise_pattern": "+Loss/Phase5_1/unconditional_replace_noise_white_at5_tk{tk}_token_loss_small/per_token",
            "color": "#ADD8E6",
            "label": "unconditional",
        },
    },
    "mgen-melody": {
        "asap": {
            "ori": "+Loss/Phase5_1/asap_ori_mgen-melody/per_token",
            "noise_pattern": "+Loss/Phase5_1/asap_replace_noise_white_at5_tk{tk}_token_loss_mgen-melody/per_token",
            "color": "#FFFACD",
            "label": "asap",
        },
        "shutter": {
            "ori": "+Loss/Phase5_1/ShutterStock_32k_ori_mgen-melody/per_token",
            "noise_pattern": "+Loss/Phase5_1/shutter_replace_noise_white_at5_tk{tk}_token_loss_mgen-melody/per_token",
            "color": "#FFB6C1",
            "label": "shutter",
        },
        "unconditional": {
            "ori": "+Loss/Phase5_1/Unconditional_ori_mgen-melody/per_token",
            "noise_pattern": "+Loss/Phase5_1/unconditional_replace_noise_white_at5_tk{tk}_token_loss_mgen-melody/per_token",
            "color": "#ADD8E6",
            "label": "unconditional",
        },
    },
    "medium": {
        "asap": {
            "ori": "+Loss/Phase5_1/asap_ori_medium/per_token",
            "noise_pattern": "+Loss/Phase5_1/asap_replace_noise_white_at5_tk{tk}_token_loss_medium/per_token",
            "color": "#FFFACD",
            "label": "asap",
        },
        "shutter": {
            "ori": "+Loss/Phase5_1/ShutterStock_32k_ori_medium/per_token",
            "noise_pattern": "+Loss/Phase5_1/shutter_replace_noise_white_at5_tk{tk}_token_loss_medium/per_token",
            "color": "#FFB6C1",
            "label": "shutter",
        },
        "unconditional": {
            "ori": "+Loss/Phase5_1/Unconditional_ori_medium/per_token",
            "noise_pattern": "+Loss/Phase5_1/unconditional_replace_noise_white_at5_tk{tk}_token_loss_medium/per_token",
            "color": "#ADD8E6",
            "label": "unconditional",
        },
    },
    "large": {
        "asap": {
            "ori": "+Loss/Phase5_1_Large/Phase5_1/asap_ori_large/per_token",
            "noise_pattern": "+Loss/Phase5_1_Large/Phase5_1/asap_replace_noise_white_at5_tk{tk}_token_loss_large/per_token",
            "color": "#FFFACD",
            "label": "asap",
        },
        "shutter": {
            "ori": "+Loss/Phase5_1_Large/Phase5_1/ShutterStock_32k_ori_large/per_token",
            "noise_pattern": "+Loss/Phase5_1_Large/Phase5_1/shutter_replace_noise_white_at5_tk{tk}_token_loss_large/per_token",
            "color": "#FFB6C1",
            "label": "shutter",
        },
        "unconditional": {
            "ori": "+Loss/Phase5_1_Large/Phase5_1/Unconditional_ori_large/per_token",
            "noise_pattern": "+Loss/Phase5_1_Large/Phase5_1/unconditional_replace_noise_white_at5_tk{tk}_token_loss_large/per_token",
            "color": "#ADD8E6",
            "label": "unconditional",
        },
    },
}
NOISE_TOKENS = [50, 100, 150, 200]

def extract_noise_length_from_path(noise_path):
    match = re.search(r'tk(\d+)', noise_path)
    if match:
        return int(match.group(1))
    return 100

def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df['avg_loss_value'].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def detect_regions_ma(ori_loss, mix_loss, noise_length):
    diff = mix_loss - ori_loss
    baseline = np.mean(diff[:200])
    
    compromised_end = 249 + noise_length + 4
    
    bumped_start = BUMPED_START_FIXED
    
    bumped_end = None
    for i in range(bumped_start + STABLE_LEN, len(diff)):
        if all(diff[j] <= baseline for j in range(i - STABLE_LEN, i)):
            bumped_end = i - STABLE_LEN
            break
    
    if bumped_end is None:
        bumped_end = bumped_start + 50
    
    compromised_start = bumped_end
    compromised_end = min(compromised_end, len(diff) - 1)
    
    regression_start = compromised_end
    regression_end = len(diff) - 1
    
    return {
        'bumped': (bumped_start, bumped_end),
        'compromised': (compromised_start, compromised_end),
        'regression': (regression_start, regression_end),
        'baseline': baseline
    }

def analyze_single_file(ori_file, noise_file, noise_length):
    ori_loss = load_csv_data(ori_file)
    mix_loss = load_csv_data(noise_file)
    
    if ori_loss is None or mix_loss is None:
        return None
    
    min_len = min(len(ori_loss), len(mix_loss))
    ori_loss = ori_loss[:min_len]
    mix_loss = mix_loss[:min_len]
    
    regions = detect_regions_ma(ori_loss, mix_loss, noise_length)
    filename = os.path.basename(ori_file).replace('_tokens_avg.csv', '')
    
    diff = mix_loss - ori_loss
    bumped_start, bumped_end = regions['bumped']
    compromised_start, compromised_end = regions['compromised']
    regression_start, regression_end = regions['regression']
    
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
    }
    
    return result

def collect_group_results(ori_dir, noise_pattern):
    """
    聚合某一组（asap/shutter/unconditional）在 tk50/100/150/200 上，
    统计三个区域的平均损失差值，返回 {'bumped': [...], 'compromised': [...], 'regression': [...]}。
    """
    aggregated = {"bumped": [], "compromised": [], "regression": []}
    ori_files = glob.glob(os.path.join(ori_dir, "*.csv"))
    if not ori_files:
        print(f"[collect] 警告: 找不到 ori 文件于 {ori_dir}")
        return aggregated

    for tk in NOISE_TOKENS:
        noise_path = noise_pattern.format(tk=tk)
        noise_len = tk
        missing = 0
        for ori_file in ori_files:
            filename = os.path.basename(ori_file)
            noise_file = os.path.join(noise_path, filename)
            if not os.path.exists(noise_file):
                missing += 1
                continue
            res = analyze_single_file(ori_file, noise_file, noise_len)
            if res is None:
                continue
            aggregated["bumped"].append(res["bumped_avg_diff"])
            aggregated["compromised"].append(res["compromised_avg_diff"])
            aggregated["regression"].append(res["regression_avg_diff"])
        print(f"[collect] {os.path.basename(ori_dir)} tk{tk}: 缺失噪声文件={missing}, "
              f"bumped={len(aggregated['bumped'])}, compromised={len(aggregated['compromised'])}, regression={len(aggregated['regression'])}")
    return aggregated

def plot_grouped_violinplot_on_axis(ax, all_group_results, group_meta, title):
    """
    在给定的 ax 上绘制分组小提琴图（渐变色版本）：
    横轴三个区域（bumped/compromised/regression），每个区域内紧凑排列三组（asap/shutter/unconditional）。
    每个区域使用不同的颜色系，每个数据库使用不同深度的颜色。
    """
    region_order = ["bumped", "compromised", "regression"]
    region_labels = ["Peak", "Assimilation", "Recovery"]
    groups_in_order = ["asap", "shutter", "unconditional"]

    centers = [0.4, 2, 3.6]
    offsets = {"asap": -0.5, "shutter": 0.0, "unconditional": 0.5}
    widths = 0.45

    for i, region in enumerate(region_order):
        for g in groups_in_order:
            data = [all_group_results[g][region]]
            position = [centers[i] + offsets[g]]
            
            # 使用渐变色配置
            color = GRADIENT_COLORS[region][g]
            
            parts = ax.violinplot(data, positions=position, widths=widths, showmeans=False, showmedians=True, showextrema=True)
            
            # 设置小提琴图的颜色
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # 设置中位数线的样式
            if 'cmedians' in parts:
                parts['cmedians'].set_color('#333333')
                parts['cmedians'].set_linewidth(2)

    ax.set_xticks(centers)
    ax.set_xticklabels(region_labels, fontsize=17)
    ax.grid(False)
    
    # 每个子图都设置y轴刻度为指定值
    ax.set_yticks([-8, -4, 0, 4])
    ax.set_yticklabels([-8, -4, 0, 4], fontsize=28)
    ax.set_ylim(-10, 6)

    return all_group_results

def main():
    print("开始区域分析（white 噪声，tk50/100/150/200，四模型：Small / Medium / Melody / Large，渐变色小提琴图版本）...")
    print(f"输出目录: {OUTPUT_DIR}")

    # 收集所有模型的结果
    all_models_results = {}
    for model_key, model_config in MODELS.items():
        print(f"\n处理模型: {model_key}")
        all_models_results[model_key] = {}
        for group_key, group_config in model_config.items():
            print(f"  处理组: {group_key}")
            all_models_results[model_key][group_key] = collect_group_results(
                group_config["ori"], group_config["noise_pattern"]
            )

    # 计算全局 y 轴范围
    all_values = []
    for model_results in all_models_results.values():
        for group_results in model_results.values():
            for region_data in group_results.values():
                all_values.extend(region_data)
    
    # 创建 1x4 子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 模型顺序调整
    model_order = ["small", "medium", "mgen-melody", "large"]
    titles = {
        "small": "MusicGen Small",
        "medium": "MusicGen Medium",
        "mgen-melody": "MusicGen Melody",
        "large": "MusicGen Large",
    }

    # 保存统计数据到文件
    stats_path = os.path.join(OUTPUT_DIR, "region_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as stats_file:
        stats_file.write("Average Loss Difference Statistics (Gradient Color Violin Version)\n")
        stats_file.write("=============================================================\n")
        
        for model_key in model_order:
            stats_file.write(f"\n=== {titles[model_key]} Model ===\n")
            
            region_order = ["bumped", "compromised", "regression"]
            region_labels = ["Peak", "Assimilation", "Recovery"]
            groups_in_order = ["asap", "shutter", "unconditional"]
            
            for i, region in enumerate(region_order):
                stats_file.write(f"\n{region_labels[i]} Region:\n")
                
                region_means = []
                region_stds = []
                
                for g in groups_in_order:
                    data = all_models_results[model_key][g][region]
                    if len(data) > 0:
                        mean_val = np.mean(data)
                        std_val = np.std(data, ddof=1)
                        stats_file.write(f"  {g}: {mean_val:.2f} ± {std_val:.2f}\n")
                        region_means.append(mean_val)
                        region_stds.append(std_val)
                    else:
                        stats_file.write(f"  {g}: No data\n")
                
                if len(region_means) > 0:
                    avg_mean = np.mean(region_means)
                    avg_std = np.mean(region_stds)
                    stats_file.write(f"  Average across databases: {avg_mean:.2f} ± {avg_std:.2f}\n")

    for ax, model_key in zip(axes, model_order):
        plot_grouped_violinplot_on_axis(ax, all_models_results[model_key], MODELS[model_key],
                                     f"{titles[model_key]} Model")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "no_legend_avg_loss_diff_grouped_violinplot_gradient_4models_1x4.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True, facecolor='none')
    plt.close()
    print(f"总图已保存: {out_path}")
    print(f"统计数据已保存: {stats_path}")

if __name__ == "__main__":
    main()