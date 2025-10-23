import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_loss_difference(file1, file2, output_file):
    """
    Plots the difference of loss values between two CSV files with token-based data.
    Difference is calculated as: file2_loss - file1_loss

    Args:
        file1 (str): Path to the first CSV file (baseline).
        file2 (str): Path to the second CSV file (comparison).
        output_file (str): Path to save the output plot image.
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Check if both files have the same structure
        if len(df1) != len(df2):
            print(f"Warning: Files have different lengths. File1: {len(df1)}, File2: {len(df2)}")
            print("Using the minimum length for comparison.")
            min_len = min(len(df1), len(df2))
            df1 = df1.iloc[:min_len]
            df2 = df2.iloc[:min_len]
        
        # Ensure token indices match (or use index if they don't)
        if 'token_idx' in df1.columns and 'token_idx' in df2.columns:
            # Check if token indices are aligned
            if not df1['token_idx'].equals(df2['token_idx']):
                print("Warning: Token indices don't match between files. Using file1's token indices.")
            token_indices = df1['token_idx']
        else:
            print("Warning: 'token_idx' column not found. Using sequential indices.")
            token_indices = range(len(df1))
        
        # Calculate the difference: file2 - file1
        loss_diff = df2['loss'] - df1['loss']
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Loss difference bar chart
        colors = ['red' if x > 0 else 'blue' for x in loss_diff]
        ax.bar(token_indices, loss_diff, color=colors, alpha=0.6, width=1.0)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_title(f'Loss Difference ({os.path.basename(file2)} - {os.path.basename(file1)})')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Loss Difference')
        ax.grid(True, alpha=0.3)
        
        # Add text annotation for interpretation
        ax.text(0.02, 0.98, 'Red: File2 > File1\nBlue: File2 < File1', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Print detailed statistics
        print("\n" + "="*60)
        print("LOSS DIFFERENCE ANALYSIS")
        print("="*60)
        print(f"File 1 (baseline): {os.path.basename(file1)}")
        print(f"File 2 (comparison): {os.path.basename(file2)}")
        print(f"Number of tokens compared: {len(loss_diff)}")
        print("\nOriginal Loss Statistics:")
        print(f"  File 1 - Mean: {df1['loss'].mean():.6f}, Std: {df1['loss'].std():.6f}, Range: [{df1['loss'].min():.6f}, {df1['loss'].max():.6f}]")
        print(f"  File 2 - Mean: {df2['loss'].mean():.6f}, Std: {df2['loss'].std():.6f}, Range: [{df2['loss'].min():.6f}, {df2['loss'].max():.6f}]")
        
        print("\nLoss Difference Statistics (File2 - File1):")
        print(f"  Mean difference: {loss_diff.mean():.6f}")
        print(f"  Std deviation: {loss_diff.std():.6f}")
        print(f"  Min difference: {loss_diff.min():.6f}")
        print(f"  Max difference: {loss_diff.max():.6f}")
        print(f"  Median difference: {loss_diff.median():.6f}")
        
        # Count positive and negative differences
        positive_count = (loss_diff > 0).sum()
        negative_count = (loss_diff < 0).sum()
        zero_count = (loss_diff == 0).sum()
        
        print(f"\nDifference Distribution:")
        print(f"  File2 > File1 (positive): {positive_count} tokens ({positive_count/len(loss_diff)*100:.1f}%)")
        print(f"  File2 < File1 (negative): {negative_count} tokens ({negative_count/len(loss_diff)*100:.1f}%)")
        print(f"  File2 = File1 (zero): {zero_count} tokens ({zero_count/len(loss_diff)*100:.1f}%)")
        
        # Statistical significance (basic)
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_rel(df1['loss'], df2['loss'])
        print(f"\nStatistical Test (Paired t-test):")
        print(f"  t-statistic: {t_stat:.6f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant difference (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        # Save detailed statistics to file
        stats_file = output_file.replace('.png', '_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("LOSS DIFFERENCE ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"File 1 (baseline): {file1}\n")
            f.write(f"File 2 (comparison): {file2}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
            
            f.write(f"Tokens Compared: {len(loss_diff)}\n\n")
            
            f.write("Original Loss Statistics:\n")
            f.write(f"File 1 - Mean: {df1['loss'].mean():.8f}, Std: {df1['loss'].std():.8f}\n")
            f.write(f"File 2 - Mean: {df2['loss'].mean():.8f}, Std: {df2['loss'].std():.8f}\n\n")
            
            f.write("Loss Difference Statistics (File2 - File1):\n")
            f.write(f"Mean: {loss_diff.mean():.8f}\n")
            f.write(f"Std: {loss_diff.std():.8f}\n")
            f.write(f"Min: {loss_diff.min():.8f}\n")
            f.write(f"Max: {loss_diff.max():.8f}\n")
            f.write(f"Median: {loss_diff.median():.8f}\n\n")
            
            f.write(f"Distribution:\n")
            f.write(f"Positive differences: {positive_count} ({positive_count/len(loss_diff)*100:.2f}%)\n")
            f.write(f"Negative differences: {negative_count} ({negative_count/len(loss_diff)*100:.2f}%)\n")
            f.write(f"Zero differences: {zero_count} ({zero_count/len(loss_diff)*100:.2f}%)\n\n")
            
            f.write(f"Statistical Test (Paired t-test):\n")
            f.write(f"t-statistic: {t_stat:.8f}\n")
            f.write(f"p-value: {p_value:.8f}\n")
            f.write(f"Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}\n")
        
        print(f"\nDetailed statistics saved to: {stats_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
    except KeyError as e:
        print(f"Error: Column {e} not found. Expected columns: 'token_idx', 'loss'")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot loss difference between two token-based CSV files.")
    parser.add_argument("--file1", type=str, 
                        default="../Loss/Phase3_2_3/loss_Beethoven_ori_time_ins_rock_music_small.csv",
                        help="Path to the first CSV file (baseline).")
    parser.add_argument("--file2", type=str, 
                        default="../Loss/Phase3_2_3/loss_Beethoven_ori_time_ins_silence_small.csv",
                        help="Path to the second CSV file (comparison).")
    parser.add_argument("--output_file", type=str, 
                        default="../Loss_Plot/token_loss_difference.png",
                        help="Path to save the output plot image.")

    args = parser.parse_args()

    plot_loss_difference(args.file1, args.file2, args.output_file)

if __name__ == "__main__":
    main()