import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss_comparison(file1, file2, output_file):
    """
    Plots a comparison of loss values from two CSV files with token-based data.

    Args:
        file1 (str): Path to the first CSV file (e.g., loss_time_in_small.csv).
        file2 (str): Path to the second CSV file (e.g., loss_time_in_ori_small.csv).
        output_file (str): Path to save the output plot image.
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Set up the plot
        plt.figure(figsize=(12, 7))

        # Plot data from the first file - using token_idx instead of start_percent
        plt.plot(df1['token_idx'], df1['loss'], marker='o', linestyle='-',
                 label=os.path.basename(file1).replace('.csv', ''), markersize=3)

        # Plot data from the second file - using token_idx instead of start_percent
        plt.plot(df2['token_idx'], df2['loss'], marker='x', linestyle='--',
                 label=os.path.basename(file2).replace('.csv', ''), markersize=3)

        # Add titles and labels - updated for token-based plotting
        plt.title('Loss Comparison Across All Tokens')
        plt.xlabel('Token Index')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Print some statistics
        print(f"File 1 ({os.path.basename(file1)}): {len(df1)} tokens, Loss range: {df1['loss'].min():.6f} - {df1['loss'].max():.6f}")
        print(f"File 2 ({os.path.basename(file2)}): {len(df2)} tokens, Loss range: {df2['loss'].min():.6f} - {df2['loss'].max():.6f}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
    except KeyError as e:
        print(f"Error: Column {e} not found. Expected columns: 'filename', 'token_idx', 'loss'")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot loss comparison from two token-based CSV files.")
    parser.add_argument("--file1", type=str, default="../Loss/Phase3_2_3/loss_Beethoven_ori_time_ins_rock_music_small.csv",
                        help="Path to the first CSV file.")
    parser.add_argument("--file2", type=str, default="../Loss/Phase3_2_3/loss_Beethoven_ori_time_ins_silence_small.csv",
                        help="Path to the second CSV file.")
    parser.add_argument("--output_file", type=str, default="../Loss_Plot/token_loss_comparison.png",
                        help="Path to save the output plot image.")

    args = parser.parse_args()

    plot_loss_comparison(args.file1, args.file2, args.output_file)

if __name__ == "__main__":
    main()