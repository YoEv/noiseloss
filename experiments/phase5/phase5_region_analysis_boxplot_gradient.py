import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import glob
import re

STABLE_LEN = 5
BUMPED_START_FIXED = 245

OUTPUT_DIR = "+Loss_Plot/Phase5_1/3regions_boxplot_gradient"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRADIENT_COLORS = {
    "bumped": {
        "asap": "#FFFACD",
        "shutter": "#FFD700",
        "unconditional": "#DAA520"
    },
    "compromised": {
        "asap": "#FFE4E1",
        "shutter": "#FFB6C1",
        "unconditional": "#FF69B4"
    },
    "regression": {
        "asap": "#E6F3FF",
        "shutter": "#87CEEB",
        "unconditional": "#4169E1"
    }
}

GROUP_ORDER = ["asap", "shutter", "unconditional"]
GROUP_META = {
    "asap": {"color": "#FFFACD", "label": "asap"},
    "shutter": {"color": "#F8BBD9", "label": "shutter"},
    "unconditional": {"color": "#B3D9FF", "label": "unconditional"},
}

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
    aggregated = {"bumped": [], "compromised": [], "regression": []}
    ori_files = glob.glob(os.path.join(ori_dir, "*.csv"))
    if not ori_files:
        print(f"[collect] Warning: Cannot find ori files in {ori_dir}")
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
        print(f"[collect] {os.path.basename(ori_dir)} tk{tk}: missing noise files={missing}, "
              f"bumped={len(aggregated['bumped'])}, compromised={len(aggregated['compromised'])}, regression={len(aggregated['regression'])}")
    return aggregated

def plot_grouped_boxplot_on_axis(ax, all_group_results, group_meta, title):
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
            
            color = GRADIENT_COLORS[region][g]
            
            bp = ax.boxplot(data, positions=position, widths=widths, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
            for med in bp["medians"]:
                med.set_color("#333333")
                med.set_linewidth(2)

    ax.set_xticks(centers)
    ax.set_xticklabels(region_labels, fontsize=17)
    ax.grid(False)
    
    ax.set_yticks([-8, -4, 0, 4])
    ax.set_yticklabels([-8, -4, 0, 4], fontsize=28)
    ax.set_ylim(-10, 6)

    return all_group_results

def main():
    all_models_results = {}
    for model_key, model_config in MODELS.items():
        print(f"\nProcessing model: {model_key}")
        all_models_results[model_key] = {}
        for group_key, group_config in model_config.items():
            print(f"  Processing group: {group_key}")
            all_models_results[model_key][group_key] = collect_group_results(
                group_config["ori"], group_config["noise_pattern"]
            )

    all_values = []
    for model_results in all_models_results.values():
        for group_results in model_results.values():
            for region_data in group_results.values():
                all_values.extend(region_data)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    model_order = ["small", "medium", "mgen-melody", "large"]
    titles = {
        "small": "MusicGen Small",
        "medium": "MusicGen Medium",
        "mgen-melody": "MusicGen Melody",
        "large": "MusicGen Large",
    }

    stats_path = os.path.join(OUTPUT_DIR, "region_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as stats_file:
        stats_file.write("Average Loss Difference Statistics (Gradient Color Version)\n")
        stats_file.write("========================================================\n")
        
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
        plot_grouped_boxplot_on_axis(ax, all_models_results[model_key], MODELS[model_key],
                                     f"{titles[model_key]} Model")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "no_legend_avg_loss_diff_grouped_boxplot_gradient_4models_1x4.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True, facecolor='none')
    plt.close()
    print(f"Main plot saved: {out_path}")
    print(f"Statistics data saved: {stats_path}")

if __name__ == "__main__":
    main()