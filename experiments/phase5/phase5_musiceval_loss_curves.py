import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re

OUTPUT_DIR = "test_out/musiceval_ori_small_loss_curves"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = "+Loss/Phase5_1_MusicEval_small/musiceval_ori_small/per_token"

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df['token_position'].values, df['avg_loss_value'].values

def plot_single_loss_curve(token_positions, loss_values, title, output_path):
    plt.figure(figsize=(12, 6))
    
    plt.plot(token_positions, loss_values, 'b-', linewidth=1.5, alpha=0.8, label='Loss')
    
    plt.title(f'Loss Curve - {title}', fontsize=14, fontweight='bold')
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Average Loss Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)
    min_loss = np.min(loss_values)
    max_loss = np.max(loss_values)
    
    stats_text = f'Mean: {mean_loss:.3f}\nStd: {std_loss:.3f}\nMin: {min_loss:.3f}\nMax: {max_loss:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean': mean_loss,
        'std': std_loss,
        'min': min_loss,
        'max': max_loss,
        'total_tokens': len(loss_values)
    }

def extract_file_id(filename):
    match = re.search(r'S010_P(\d+)', filename)
    if match:
        return match.group(1)
    return filename

def create_summary_plot(all_stats, output_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    file_ids = list(all_stats.keys())
    means = [all_stats[fid]['mean'] for fid in file_ids]
    stds = [all_stats[fid]['std'] for fid in file_ids]
    mins = [all_stats[fid]['min'] for fid in file_ids]
    maxs = [all_stats[fid]['max'] for fid in file_ids]
    
    ax1.hist(means, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Distribution of Mean Loss Values')
    ax1.set_xlabel('Mean Loss')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(stds, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Distribution of Loss Standard Deviations')
    ax2.set_xlabel('Standard Deviation')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    ax3.hist(mins, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax3.set_title('Distribution of Minimum Loss Values')
    ax3.set_xlabel('Minimum Loss')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    ax4.hist(maxs, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_title('Distribution of Maximum Loss Values')
    ax4.set_xlabel('Maximum Loss')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_statistics_csv(all_stats, output_path):
    data = []
    for file_id, stats in all_stats.items():
        data.append({
            'file_id': file_id,
            'mean_loss': stats['mean'],
            'std_loss': stats['std'],
            'min_loss': stats['min'],
            'max_loss': stats['max'],
            'total_tokens': stats['total_tokens']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Statistics saved to: {output_path}")

def main():
    print("Starting MusicEval Loss Curve Analysis...")
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    csv_files.sort()
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_stats = {}
    
    for i, csv_file in enumerate(csv_files):
        filename = os.path.basename(csv_file)
        file_id = extract_file_id(filename)
        
        print(f"Processing {i+1}/{len(csv_files)}: {filename}")
        
        token_positions, loss_values = load_csv_data(csv_file)
        
        if token_positions is None or loss_values is None:
            print(f"Skipping {filename} due to loading error")
            continue
        
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}_loss_curve.png")
        stats = plot_single_loss_curve(token_positions, loss_values, file_id, output_path)
        
        all_stats[file_id] = stats
        
        print(f"  Saved: {output_path}")
        print(f"  Stats: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, Tokens={stats['total_tokens']}")
    
    if all_stats:
        summary_plot_path = os.path.join(OUTPUT_DIR, "summary_statistics.png")
        create_summary_plot(all_stats, summary_plot_path)
        print(f"Summary plot saved to: {summary_plot_path}")
        
        stats_csv_path = os.path.join(OUTPUT_DIR, "loss_statistics.csv")
        save_statistics_csv(all_stats, stats_csv_path)
    
    print(f"\nAnalysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"Total files processed: {len(all_stats)}")

if __name__ == "__main__":
    main()