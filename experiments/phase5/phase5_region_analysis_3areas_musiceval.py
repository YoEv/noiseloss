import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import glob
import re

STABLE_LEN = 5
BUMPED_START_FIXED_5S = 245
BUMPED_START_FIXED_20S = 999
NOISE_LENGTH = 20

OUTPUT_DIR = "test_out/musiceval_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MUSICEVAL_GROUPS = {
    "original": {
        "path": "+Loss/Phase5_1_MusicEval_small/musiceval_ori_small/per_token",
        "color": "#90EE90",
        "label": "Original",
        "noise_position": None,
    },
    "noise_at5s": {
        "path": "+Loss/Phase5_1_MusicEval_small/musiceval_at5_small/per_token",
        "color": "#FFB6C1",
        "label": "Noise at 5s",
        "noise_position": "5s",
    },
    "noise_at20s": {
        "path": "+Loss/Phase5_1_MusicEval_small/musiceval_at20_small/per_token",
        "color": "#ADD8E6",
        "label": "Noise at 20s",
        "noise_position": "20s",
    },
}

def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df['avg_loss_value'].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def detect_regions_ma(ori_loss, mix_loss, noise_position="5s"):
    diff = mix_loss - ori_loss
    baseline = np.mean(diff[:200])
    
    if noise_position == "20s":
        bumped_start = BUMPED_START_FIXED_20S
        compromised_end = bumped_start + NOISE_LENGTH + 4
    else:
        bumped_start = BUMPED_START_FIXED_5S
        compromised_end = 249 + NOISE_LENGTH + 4
    
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

def analyze_single_file_pair(ori_file, noise_file, noise_type):
    ori_loss = load_csv_data(ori_file)
    mix_loss = load_csv_data(noise_file)
    
    if ori_loss is None or mix_loss is None:
        return None
    
    min_len = min(len(ori_loss), len(mix_loss))
    ori_loss = ori_loss[:min_len]
    mix_loss = mix_loss[:min_len]
    
    noise_position = MUSICEVAL_GROUPS[noise_type]["noise_position"]
    regions = detect_regions_ma(ori_loss, mix_loss, noise_position)
    filename = os.path.basename(ori_file).replace('_tokens_avg.csv', '')
    
    diff = mix_loss - ori_loss
    bumped_start, bumped_end = regions['bumped']
    compromised_start, compromised_end = regions['compromised']
    regression_start, regression_end = regions['regression']
    
    bumped_diff = diff[bumped_start:bumped_end] if bumped_end > bumped_start else []
    bumped_max = np.max(bumped_diff) if len(bumped_diff) > 0 else 0
    bumped_avg = np.mean(bumped_diff) if len(bumped_diff) > 0 else 0
    
    result = {
        'filename': filename,
        'noise_type': noise_type,
        'noise_position': noise_position,
        'data_length': len(ori_loss),
        'regions': regions,
        'bumped_start': bumped_start,
        'bumped_length': bumped_end - bumped_start,
        'compromised_length': compromised_end - compromised_start,
        'regression_length': regression_end - regression_start,
        'bumped_avg_diff': bumped_avg,
        'bumped_max_diff': bumped_max,
        'compromised_avg_diff': np.mean(diff[compromised_start:compromised_end]) if compromised_end > compromised_start else 0,
        'regression_avg_diff': np.mean(diff[regression_start:regression_end]) if regression_end > regression_start else 0,
    }
    
    return result

def analyze_musiceval_dataset():
    print("Starting MusicEval dataset region analysis...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Noise length: {NOISE_LENGTH} tokens")
    print(f"5-second position bumped_start: {BUMPED_START_FIXED_5S}")
    print(f"20-second position bumped_start: {BUMPED_START_FIXED_20S}")
    print("-" * 50)
    
    ori_dir = MUSICEVAL_GROUPS["original"]["path"]
    ori_files = glob.glob(os.path.join(ori_dir, "*.csv"))
    
    if not ori_files:
        print(f"Warning: Cannot find original files in {ori_dir}")
        return
    
    print(f"Found {len(ori_files)} original files")
    
    all_results = []
    bumped_stats = []
    
    for noise_type in ["noise_at5s", "noise_at20s"]:
        noise_dir = MUSICEVAL_GROUPS[noise_type]["path"]
        noise_position = MUSICEVAL_GROUPS[noise_type]["noise_position"]
        print(f"\nAnalyzing {noise_type} (position: {noise_position}, path: {noise_dir})")
        
        missing_count = 0
        processed_count = 0
        
        for ori_file in ori_files:
            filename = os.path.basename(ori_file)
            noise_file = os.path.join(noise_dir, filename)
            
            if not os.path.exists(noise_file):
                missing_count += 1
                continue
            
            result = analyze_single_file_pair(ori_file, noise_file, noise_type)
            if result is None:
                continue
            
            all_results.append(result)
            
            bumped_stats.append({
                'filename': result['filename'],
                'noise_type': noise_type,
                'noise_position': noise_position,
                'bumped_start': result['bumped_start'],
                'bumped_max': result['bumped_max_diff'],
                'bumped_avg': result['bumped_avg_diff']
            })
            
            processed_count += 1
        
        print(f"{noise_type}: processed={processed_count}, missing={missing_count}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(OUTPUT_DIR, "musiceval_region_analysis_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nDetailed analysis results saved: {results_path}")
    
    if bumped_stats:
        bumped_df = pd.DataFrame(bumped_stats)
        bumped_path = os.path.join(OUTPUT_DIR, "bumped_area_statistics.csv")
        bumped_df.to_csv(bumped_path, index=False)
        print(f"Bumped area statistics saved: {bumped_path}")
        
        bumped_txt_path = os.path.join(OUTPUT_DIR, "bumped_area_statistics.txt")
        with open(bumped_txt_path, 'w', encoding='utf-8') as f:
            f.write("MusicEval Dataset Bumped Area Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Noise length: {NOISE_LENGTH} tokens\n")
            f.write(f"5-second position bumped_start: {BUMPED_START_FIXED_5S}\n")
            f.write(f"20-second position bumped_start: {BUMPED_START_FIXED_20S}\n\n")
            
            for _, row in bumped_df.iterrows():
                f.write(f"File: {row['filename']}\n")
                f.write(f"Noise type: {row['noise_type']}\n")
                f.write(f"Noise position: {row['noise_position']}\n")
                f.write(f"Bumped Start: {row['bumped_start']}\n")
                f.write(f"Bumped Area maximum: {row['bumped_max']:.6f}\n")
                f.write(f"Bumped Area average: {row['bumped_avg']:.6f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"Bumped area statistics text file saved: {bumped_txt_path}")
    
    plot_region_analysis(all_results)
    
    print("\nMusicEval dataset analysis completed!")

def plot_region_analysis(all_results):
    if not all_results:
        return
    
    noise_types = ["noise_at5s", "noise_at20s"]
    regions = ["bumped", "compromised", "regression"]
    
    stats_data = {}
    for noise_type in noise_types:
        type_results = [r for r in all_results if r['noise_type'] == noise_type]
        stats_data[noise_type] = {
            'bumped': [r['bumped_avg_diff'] for r in type_results],
            'compromised': [r['compromised_avg_diff'] for r in type_results],
            'regression': [r['regression_avg_diff'] for r in type_results]
        }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    region_labels = ['Bumped', 'Compromised', 'Regression']
    x_pos = np.arange(len(region_labels))
    width = 0.35
    
    noise_at5s_means = [np.mean(stats_data['noise_at5s'][r]) for r in regions]
    noise_at20s_means = [np.mean(stats_data['noise_at20s'][r]) for r in regions]
    
    ax1.bar(x_pos - width/2, noise_at5s_means, width, label='Noise at 5s', 
            color=MUSICEVAL_GROUPS['noise_at5s']['color'], alpha=0.8)
    ax1.bar(x_pos + width/2, noise_at20s_means, width, label='Noise at 20s', 
            color=MUSICEVAL_GROUPS['noise_at20s']['color'], alpha=0.8)
    
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Average Loss Difference')
    ax1.set_title(f'Average Loss Difference by Region (Noise Length: {NOISE_LENGTH} tokens)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(region_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    noise_at5s_stds = [np.std(stats_data['noise_at5s'][r]) for r in regions]
    noise_at20s_stds = [np.std(stats_data['noise_at20s'][r]) for r in regions]
    
    ax2.bar(x_pos - width/2, noise_at5s_stds, width, label='Noise at 5s', 
            color=MUSICEVAL_GROUPS['noise_at5s']['color'], alpha=0.8)
    ax2.bar(x_pos + width/2, noise_at20s_stds, width, label='Noise at 20s', 
            color=MUSICEVAL_GROUPS['noise_at20s']['color'], alpha=0.8)
    
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Standard Deviation of Loss Difference')
    ax2.set_title('Standard Deviation by Region')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(region_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "musiceval_region_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Regional analysis chart saved: {plot_path}")
    
    summary_stats = []
    for noise_type in noise_types:
        for region in regions:
            data = stats_data[noise_type][region]
            summary_stats.append({
                'noise_type': noise_type,
                'noise_position': MUSICEVAL_GROUPS[noise_type]['noise_position'],
                'region': region,
                'count': len(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(OUTPUT_DIR, "musiceval_summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")

def main():
    analyze_musiceval_dataset()

if __name__ == "__main__":
    main()