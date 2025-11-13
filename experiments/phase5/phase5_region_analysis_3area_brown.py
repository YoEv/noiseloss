import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

NOISE_LENGTH = 150
COMPROMISED_END = 403
STABLE_LEN = 5
BUMPED_START_FIXED = 245

ORIGINAL_PATH = "+Loss/Phase5_1/ShutterStock_32k_ori_small/per_token"
NOISE_PATH = "+Loss/Phase5_1/shutter_replace_noise_brown_at5_tk150_token_loss_small/per_token"
BROWN_REFERENCE_PATH = "+Loss/Phase5_1/noise_color_ori_small/per_token/brown_noise_12db_15.0s_tokens_avg.csv"
OUTPUT_DIR = "test_out/brown"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df['avg_loss_value'].values

def detect_regions_ma(ori_loss, mix_loss):
    """
    Three Areas: bumped, compromised, regression
    
    Args:
        ori_loss:   np.array, Original Music per-token loss
        mix_loss:   np.array, Music + Purturb per-token loss
    
    Returns:
        dict: Three areas starting and ending points
    """
    diff = mix_loss - ori_loss
    baseline = np.mean(diff[:200])
    
    bumped_start = BUMPED_START_FIXED
    
    bumped_end = None
    for i in range(bumped_start + STABLE_LEN, len(diff)):
        if all(diff[j] <= baseline for j in range(i - STABLE_LEN, i)):
            bumped_end = i - STABLE_LEN
            break
    
    if bumped_end is None:
        bumped_end = bumped_start + 50
    
    compromised_start = bumped_end
    compromised_end = min(COMPROMISED_END, len(diff) - 1)
    
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
    plot (including brown noise reference)
    
    Args:
        ori_loss: np.array, original music loss
        mix_loss: np.array, music + purturb loss
        brown_ref_loss: np.array, brown noise reference loss
        regions: dict, area region
        filename: str
        output_dir: str
    """
    diff = mix_loss - ori_loss
    
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
    
    ax1.plot(ori_loss, label='Original Loss', alpha=0.7, color='blue', linewidth=1.5)
    ax1.plot(mix_loss, label='Mixed Loss (with brown noise)', alpha=0.7, color='red', linewidth=1.5)
    
    if brown_ref_truncated is not None:
        ax1.plot(brown_ref_truncated, label='Brown Noise Reference', alpha=0.7, color='brown', linewidth=1.5, linestyle='--')
    
    ax1.axhline(y=regions['baseline'] + np.mean(ori_loss[:200]), color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
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
    
    ax2.plot(diff, label='Loss Difference (Mixed - Original)', color='purple', alpha=0.8, linewidth=1.5)
    
    if brown_diff is not None:
        brown_diff_plot = brown_diff[:len(diff)]
        ax2.plot(brown_diff_plot, label='Brown Reference Difference', color='brown', alpha=0.8, linewidth=1.5, linestyle='--')
    
    ax2.axhline(y=regions['baseline'], color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    ax2.axvspan(bumped_start, bumped_end, alpha=0.3, color='orange', label='Bumped')
    ax2.axvspan(compromised_start, compromised_end, alpha=0.3, color='red', label='Compromised')
    ax2.axvspan(regression_start, regression_end, alpha=0.3, color='green', label='Regression')
    
    ax2.set_title('Loss Difference Analysis with Brown Noise Reference', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Loss Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{filename}_brown_regions_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Image saved: {output_path}")

def analyze_single_file_with_brown_ref(ori_file, noise_file, brown_ref_loss, output_dir):
    ori_loss = load_csv_data(ori_file)
    mix_loss = load_csv_data(noise_file)
    
    if ori_loss is None or mix_loss is None:
        return None
    
    min_len = min(len(ori_loss), len(mix_loss))
    if brown_ref_loss is not None:
        min_len = min(min_len, len(brown_ref_loss))
    
    ori_loss = ori_loss[:min_len]
    mix_loss = mix_loss[:min_len]
    
    regions = detect_regions_ma(ori_loss, mix_loss)
    
    filename = os.path.basename(ori_file).replace('_tokens_avg.csv', '')
    
    plot_regions_analysis_with_brown_ref(ori_loss, mix_loss, brown_ref_loss, regions, filename, output_dir)
    
    diff = mix_loss - ori_loss
    bumped_start, bumped_end = regions['bumped']
    compromised_start, compromised_end = regions['compromised']
    regression_start, regression_end = regions['regression']
    
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
        **brown_stats
    }
    
    return result

def plot_summary_statistics_with_brown(results, output_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
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
    
    bumped_diffs = [r['bumped_avg_diff'] for r in results]
    compromised_diffs = [r['compromised_avg_diff'] for r in results]
    regression_diffs = [r['regression_avg_diff'] for r in results]
    
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
    
    ax3.boxplot([bumped_lens, compromised_lens, regression_lens], 
                labels=['Bumped', 'Compromised', 'Regression'])
    ax3.set_title('Region Length Box Plot', fontweight='bold')
    ax3.set_ylabel('Length (tokens)')
    ax3.grid(True, alpha=0.3)
    
    if brown_bumped_diffs:
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
    
    summary_path = os.path.join(output_dir, "brown_regions_summary_statistics.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary saved: {summary_path}")

def main():
    # print("Starting Brown Noise region analysis...")
    # print(f"Noise length: {NOISE_LENGTH}")
    # print(f"Compromised region end position: {COMPROMISED_END}")
    # print(f"Bumped region fixed start position: {BUMPED_START_FIXED}")
    # print(f"Brown Reference path: {BROWN_REFERENCE_PATH}")
    # print(f"Output directory: {OUTPUT_DIR}")
    # print("-" * 50)
    
    # print("Loading Brown Noise Reference...")
    brown_ref_loss = load_csv_data(BROWN_REFERENCE_PATH)
    if brown_ref_loss is None:
        print(f"Warning: Cannot load Brown Noise Reference file {BROWN_REFERENCE_PATH}")
        print("Continuing analysis without reference comparison")
    else:
        print(f"Successfully loaded Brown Noise Reference, data length: {len(brown_ref_loss)}")
    
    ori_files = glob.glob(os.path.join(ORIGINAL_PATH, "*.csv"))
    
    results = []
    successful_analyses = 0
    
    for ori_file in ori_files:
        filename = os.path.basename(ori_file)
        noise_file = os.path.join(NOISE_PATH, filename)
        
        if not os.path.exists(noise_file):
            print(f"Warning: Cannot find corresponding noise file {noise_file}")
            continue
        
        print(f"Analyzing file: {filename}")
        
        result = analyze_single_file_with_brown_ref(ori_file, noise_file, brown_ref_loss, OUTPUT_DIR)
        
        if result:
            results.append(result)
            successful_analyses += 1
            
            regions = result['regions']
            print(f"  Bumped: {regions['bumped'][0]}-{regions['bumped'][1]} (length: {result['bumped_length']})")
            print(f"  Compromised: {regions['compromised'][0]}-{regions['compromised'][1]} (length: {result['compromised_length']})")
            print(f"  Regression: {regions['regression'][0]}-{regions['regression'][1]} (length: {result['regression_length']})")
            print(f"  Average difference - Bumped: {result['bumped_avg_diff']:.4f}, Compromised: {result['compromised_avg_diff']:.4f}, Regression: {result['regression_avg_diff']:.4f}")
            
            if 'brown_bumped_avg_diff' in result:
                print(f"  Brown Ref difference - Bumped: {result['brown_bumped_avg_diff']:.4f}, Compromised: {result['brown_compromised_avg_diff']:.4f}, Regression: {result['brown_regression_avg_diff']:.4f}")
        else:
            print(f"  Analysis failed")
        
        print()
    
    if results:
        print("=" * 50)
        print("Summary statistics:")
        print(f"Total files: {len(ori_files)}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Success rate: {successful_analyses/len(ori_files)*100:.1f}%")
        
        avg_bumped_len = np.mean([r['bumped_length'] for r in results])
        avg_compromised_len = np.mean([r['compromised_length'] for r in results])
        avg_regression_len = np.mean([r['regression_length'] for r in results])
        
        avg_bumped_diff = np.mean([r['bumped_avg_diff'] for r in results])
        avg_compromised_diff = np.mean([r['compromised_avg_diff'] for r in results])
        avg_regression_diff = np.mean([r['regression_avg_diff'] for r in results])
        
        print(f"Average region length - Bumped: {avg_bumped_len:.1f}, Compromised: {avg_compromised_len:.1f}, Regression: {avg_regression_len:.1f}")
        print(f"Average difference - Bumped: {avg_bumped_diff:.4f}, Compromised: {avg_compromised_diff:.4f}, Regression: {avg_regression_diff:.4f}")
        
        if brown_ref_loss is not None and 'brown_bumped_avg_diff' in results[0]:
            avg_brown_bumped = np.mean([r['brown_bumped_avg_diff'] for r in results])
            avg_brown_compromised = np.mean([r['brown_compromised_avg_diff'] for r in results])
            avg_brown_regression = np.mean([r['brown_regression_avg_diff'] for r in results])
            print(f"Brown Ref average difference - Bumped: {avg_brown_bumped:.4f}, Compromised: {avg_brown_compromised:.4f}, Regression: {avg_brown_regression:.4f}")
        
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(OUTPUT_DIR, "brown_regions_analysis_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to: {results_csv_path}")
        
        plot_summary_statistics_with_brown(results, OUTPUT_DIR)
        
    else:
        print("No files successfully analyzed")

if __name__ == "__main__":
    main()