import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from scipy import stats as scipy_stats

def find_matching_csv_files(dir1, dir2):
    """
    Find matching CSV files between two directories based on filename.
    
    Args:
        dir1 (str): Path to first directory (baseline)
        dir2 (str): Path to second directory (comparison)
    
    Returns:
        list: List of tuples (file1_path, file2_path) for matching files
    """
    if not os.path.exists(dir1):
        print(f"Error: Directory {dir1} does not exist.")
        return []
    
    if not os.path.exists(dir2):
        print(f"Error: Directory {dir2} does not exist.")
        return []
    
    # Get all CSV files from both directories
    csv_files1 = glob.glob(os.path.join(dir1, '*.csv'))
    csv_files2 = glob.glob(os.path.join(dir2, '*.csv'))
    
    # Create dictionaries with basename as key
    files1_dict = {os.path.basename(f): f for f in csv_files1}
    files2_dict = {os.path.basename(f): f for f in csv_files2}
    
    # Find matching files
    matching_files = []
    for basename in files1_dict:
        if basename in files2_dict:
            matching_files.append((files1_dict[basename], files2_dict[basename]))
        else:
            print(f"Warning: No matching file found for {basename} in {dir2}")
    
    print(f"Found {len(matching_files)} matching CSV file pairs.")
    return matching_files

def plot_loss_difference_single_file(file1, file2, output_dir, noise_start=250, noise_end=270, plot_start=200, plot_end=300, y_min=None, y_max=None):
    """
    Plots the difference of loss values between two CSV files.
    Each CSV file represents one song with token_position and avg_loss_value columns.
    Difference is calculated as: file2_loss - file1_loss
    
    Args:
        file1 (str): Path to the first CSV file (baseline).
        file2 (str): Path to the second CSV file (comparison).
        output_dir (str): Directory to save the output plot images.
        noise_start (int): Start token index for noise region.
        noise_end (int): End token index for noise region.
        plot_start (int): Start token index for plotting range.
        plot_end (int): End token index for plotting range.
        y_min (float): Minimum y-axis value for fixed scale.
        y_max (float): Maximum y-axis value for fixed scale.
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Check if files have the expected columns
        expected_columns = ['token_position', 'avg_loss_value']
        if not all(col in df1.columns for col in expected_columns):
            print(f"Error: File {file1} missing expected columns. Found: {list(df1.columns)}")
            return
        if not all(col in df2.columns for col in expected_columns):
            print(f"Error: File {file2} missing expected columns. Found: {list(df2.columns)}")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract song name from filename
        song_name = os.path.splitext(os.path.basename(file1))[0]
        song_name = song_name.replace('_tokens_avg', '')  # Remove suffix if present
        
        print(f"Processing: {song_name}")
        
        # Ensure both files have the same length and are sorted by token_position
        df1 = df1.sort_values('token_position').reset_index(drop=True)
        df2 = df2.sort_values('token_position').reset_index(drop=True)
        
        # Check if both files have the same structure
        if len(df1) != len(df2):
            print(f"Warning: Files have different lengths. File1: {len(df1)}, File2: {len(df2)}")
            print("Using the minimum length for comparison.")
            min_len = min(len(df1), len(df2))
            df1 = df1.iloc[:min_len]
            df2 = df2.iloc[:min_len]
        
        # Check if token positions match
        if not df1['token_position'].equals(df2['token_position']):
            print("Warning: Token positions don't match between files. Aligning by position.")
            # Merge on token_position to ensure alignment
            merged = pd.merge(df1, df2, on='token_position', suffixes=('_file1', '_file2'))
            df1 = merged[['token_position', 'avg_loss_value_file1']].rename(columns={'avg_loss_value_file1': 'avg_loss_value'})
            df2 = merged[['token_position', 'avg_loss_value_file2']].rename(columns={'avg_loss_value_file2': 'avg_loss_value'})
        
        # Filter to plot range for visualization
        plot_mask = (df1['token_position'] >= plot_start) & (df1['token_position'] <= plot_end)
        
        if not plot_mask.any():
            print(f"Warning: No data in plot range {plot_start}-{plot_end} for {song_name}")
            return
        
        # Get filtered data for plotting
        plot_df1 = df1[plot_mask].copy()
        plot_df2 = df2[plot_mask].copy()
        token_indices = plot_df1['token_position']
        
        # Calculate the difference for plotting: file2 - file1
        loss_diff = plot_df2['avg_loss_value'] - plot_df1['avg_loss_value']
        
        # Filter to noise region for all statistics
        noise_mask = (df1['token_position'] >= noise_start) & (df1['token_position'] <= noise_end)
        
        if not noise_mask.any():
            print(f"Warning: No data in noise region {noise_start}-{noise_end} for {song_name}")
            return
        
        # Get noise region data for statistics
        noise_df1 = df1[noise_mask].copy()
        noise_df2 = df2[noise_mask].copy()
        noise_loss_diff = noise_df2['avg_loss_value'] - noise_df1['avg_loss_value']
            
        # Calculate statistics for noise region
        noise_stats = {
            'mean': noise_loss_diff.mean(),
            'std': noise_loss_diff.std(),
            'min': noise_loss_diff.min(),
            'max': noise_loss_diff.max(),
            'median': noise_loss_diff.median()
        }
        
        # Print statistics for this song (based on NOISE REGION)
        print(f"\nStatistics for {song_name} (Noise region: {noise_start}-{noise_end}):")
        print(f"  Number of tokens in noise region: {len(noise_loss_diff)}")
        print(f"  File 1 - Mean: {noise_df1['avg_loss_value'].mean():.6f}, Std: {noise_df1['avg_loss_value'].std():.6f}")
        print(f"  File 2 - Mean: {noise_df2['avg_loss_value'].mean():.6f}, Std: {noise_df2['avg_loss_value'].std():.6f}")
        print(f"  Loss difference - Mean: {noise_loss_diff.mean():.6f}, Std: {noise_loss_diff.std():.6f}")
        print(f"  Loss difference - Min: {noise_loss_diff.min():.6f}, Max: {noise_loss_diff.max():.6f}")
        
        # Count positive and negative differences in NOISE REGION
        positive_count = (noise_loss_diff > 0).sum()
        negative_count = (noise_loss_diff < 0).sum()
        zero_count = (noise_loss_diff == 0).sum()
        
        print(f"\nDistribution (Noise region):")
        print(f"  Positive differences: {positive_count} ({positive_count/len(noise_loss_diff)*100:.1f}%)")
        print(f"  Negative differences: {negative_count} ({negative_count/len(noise_loss_diff)*100:.1f}%)")
        print(f"  Zero differences: {zero_count} ({zero_count/len(noise_loss_diff)*100:.1f}%)")
        
        # Statistical significance test for NOISE REGION
        if len(noise_df1['avg_loss_value']) > 1 and len(noise_df2['avg_loss_value']) > 1:
            t_stat, p_value = scipy_stats.ttest_rel(noise_df1['avg_loss_value'], noise_df2['avg_loss_value'])
            print(f"  t-statistic: {t_stat:.6f}, p-value: {p_value:.6f}")
            print(f"  Significant difference (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        # Print noise region statistics summary
        print(f"\nNoise Region Statistics Summary (Tokens {noise_start}-{noise_end}):")
        print(f"  Mean: {noise_stats['mean']:.6f}")
        print(f"  Std: {noise_stats['std']:.6f}")
        print(f"  Min: {noise_stats['min']:.6f}")
        print(f"  Max: {noise_stats['max']:.6f}")
        print(f"  Median: {noise_stats['median']:.6f}")
            
        # Calculate global y-axis limits if not provided
        if y_min is None or y_max is None:
            if y_min is None:
                y_min = loss_diff.min() * 1.1  # Add 10% margin
            if y_max is None:
                y_max = loss_diff.max() * 1.1  # Add 10% margin
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Loss difference bar chart
        colors = ['red' if x > 0 else 'blue' for x in loss_diff]
        ax.bar(token_indices, loss_diff, color=colors, alpha=0.6, width=1.0)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Set fixed y-axis limits
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        # Add noise region annotation
        if noise_start >= plot_start and noise_end <= plot_end:
            # Add vertical lines to mark noise region
            ax.axvline(x=noise_start, color='green', linestyle='--', linewidth=2, alpha=0.8, label=f'Noise Start (Token {noise_start})')
            ax.axvline(x=noise_end, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Noise End (Token {noise_end})')
            
            # Add shaded region for noise
            ax.axvspan(noise_start, noise_end, alpha=0.2, color='yellow', label=f'Noise Region ({noise_start}-{noise_end})')
        
        # Clean song name for title and filename
        clean_song_name = song_name.replace('.wav', '').replace('_', ' ')
        
        ax.set_title(f'Loss Difference: {clean_song_name} (Tokens {plot_start}-{plot_end})\n({os.path.basename(file2)} - {os.path.basename(file1)})', fontsize=14)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Average Loss Difference', fontsize=12)
        ax.set_xlim(plot_start, plot_end)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add text annotation for interpretation
        ax.text(0.02, 0.98, 'Red: File2 > File1\nBlue: File2 < File1', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        safe_song_name = song_name.replace('.wav', '').replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f'loss_diff_{safe_song_name}_tokens_{plot_start}_{plot_end}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Save detailed statistics to file (ALL BASED ON NOISE REGION)
        stats_file = output_file.replace('.png', '_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write(f"LOSS DIFFERENCE ANALYSIS REPORT - {song_name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"File 1 (baseline): {file1}\n")
            f.write(f"File 2 (comparison): {file2}\n")
            f.write(f"Song: {song_name}\n")
            f.write(f"Plot Range: Tokens {plot_start} to {plot_end}\n")
            f.write(f"Noise Region: Token {noise_start} to {noise_end}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
            
            f.write(f"Tokens Compared in Noise Region: {len(noise_loss_diff)}\n\n")
            
            f.write("Original Loss Statistics (Noise Region):\n")
            f.write(f"File 1 - Mean: {noise_df1['avg_loss_value'].mean():.8f}, Std: {noise_df1['avg_loss_value'].std():.8f}\n")
            f.write(f"File 2 - Mean: {noise_df2['avg_loss_value'].mean():.8f}, Std: {noise_df2['avg_loss_value'].std():.8f}\n\n")
            
            f.write("Loss Difference Statistics (File2 - File1, Noise Region):\n")
            f.write(f"Mean: {noise_loss_diff.mean():.8f}\n")
            f.write(f"Std: {noise_loss_diff.std():.8f}\n")
            f.write(f"Min: {noise_loss_diff.min():.8f}\n")
            f.write(f"Max: {noise_loss_diff.max():.8f}\n")
            f.write(f"Median: {noise_loss_diff.median():.8f}\n\n")
            
            f.write(f"Distribution (Noise Region):\n")
            f.write(f"Positive differences: {positive_count} ({positive_count/len(noise_loss_diff)*100:.2f}%)\n")
            f.write(f"Negative differences: {negative_count} ({negative_count/len(noise_loss_diff)*100:.2f}%)\n")
            f.write(f"Zero differences: {zero_count} ({zero_count/len(noise_loss_diff)*100:.2f}%)\n\n")
            
            if len(noise_df1['avg_loss_value']) > 1 and len(noise_df2['avg_loss_value']) > 1:
                f.write(f"Statistical Test (Paired t-test, Noise Region):\n")
                f.write(f"t-statistic: {t_stat:.8f}\n")
                f.write(f"p-value: {p_value:.8f}\n")
                f.write(f"Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}\n")
        
        plt.close()  # Close the figure to free memory
        
        print(f"\nPlot saved to directory: {output_dir}")
        print(f"Plot range: Tokens {plot_start} to {plot_end}")
        print(f"All statistics based on noise region: Tokens {noise_start} to {noise_end}")
        if y_min is not None and y_max is not None:
            print(f"Fixed y-axis range: {y_min:.4f} to {y_max:.4f}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
    except KeyError as e:
        print(f"Error: Column {e} not found. Expected columns: 'token_position', 'avg_loss_value'")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_multiple_file_comparisons(dir1, dir2, output_base_dir, noise_start=250, noise_end=270, plot_start=200, plot_end=300, y_min=None, y_max=None):
    """
    Process multiple CSV file comparisons between two directories.
    
    Args:
        dir1 (str): Path to first directory (baseline - asap_loss_time_ins_ori)
        dir2 (str): Path to second directory (comparison - asap_loss_time_ins_tk5)
        output_base_dir (str): Base directory for output plots
        noise_start (int): Start token index for noise region
        noise_end (int): End token index for noise region
        plot_start (int): Start token index for plotting range
        plot_end (int): End token index for plotting range
        y_min (float): Minimum y-axis value for fixed scale
        y_max (float): Maximum y-axis value for fixed scale
    """
    print(f"Comparing CSV files between:")
    print(f"  Directory 1 (baseline): {dir1}")
    print(f"  Directory 2 (comparison): {dir2}")
    print(f"  Output base directory: {output_base_dir}")
    
    # Find matching CSV files
    matching_files = find_matching_csv_files(dir1, dir2)
    
    if not matching_files:
        print("No matching CSV files found. Exiting.")
        return
    
    # Process each file pair
    for i, (file1, file2) in enumerate(matching_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing file pair {i}/{len(matching_files)}:")
        print(f"  File 1: {file1}")
        print(f"  File 2: {file2}")
        
        # Create specific output directory for this file pair
        file1_basename = os.path.splitext(os.path.basename(file1))[0]
        file2_basename = os.path.splitext(os.path.basename(file2))[0]
        output_dir = os.path.join(output_base_dir, f"{file1_basename}_vs_{file2_basename}")
        
        # Process this file pair
        plot_loss_difference_single_file(file1, file2, output_dir, 
                                        noise_start, noise_end, 
                                        plot_start, plot_end,
                                        y_min, y_max)
    
    print(f"\n{'='*80}")
    print(f"Completed processing {len(matching_files)} file pairs.")
    print(f"All results saved to: {output_base_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare loss differences between CSV files from two directories.")
    parser.add_argument("--dir1", type=str, 
                        default="/home/evev/asap-dataset/asap_loss_time_ins_ori",
                        help="Path to the first directory (baseline - asap_loss_time_ins_ori).")
    parser.add_argument("--dir2", type=str, 
                        default="/home/evev/asap-dataset/asap_loss_time_ins_tk5",
                        help="Path to the second directory (comparison - asap_loss_time_ins_tk5).")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/evev/asap-dataset/Plot/loss_diff_comparison_plots",
                        help="Base directory to save the output plot images.")
    parser.add_argument("--noise_start", type=int, 
                        default=250,
                        help="Start token index for noise region.")
    parser.add_argument("--noise_end", type=int, 
                        default=270,
                        help="End token index for noise region.")
    parser.add_argument("--plot_start", type=int, 
                        default=200,
                        help="Start token index for plotting range.")
    parser.add_argument("--plot_end", type=int, 
                        default=300,
                        help="End token index for plotting range.")
    parser.add_argument("--y_min", type=float, 
                        default=None,
                        help="Minimum y-axis value for fixed scale (auto if not specified).")
    parser.add_argument("--y_max", type=float, 
                        default=None,
                        help="Maximum y-axis value for fixed scale (auto if not specified).")

    args = parser.parse_args()

    process_multiple_file_comparisons(args.dir1, args.dir2, args.output_dir, 
                                    args.noise_start, args.noise_end, 
                                    args.plot_start, args.plot_end,
                                    args.y_min, args.y_max)

if __name__ == "__main__":
    main()