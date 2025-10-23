import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from pathlib import Path

def plot_single_file_tokens(csv_file, output_dir):
    """
    Plot token loss curve for a single CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
        output_dir (str): Directory to save the output plot.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Set up the plot
        plt.figure(figsize=(15, 8))
        
        # Plot the token loss curve
        plt.plot(df['token_position'], df['avg_loss_value'], 
                marker='o', linestyle='-', markersize=2, linewidth=1.5)
        
        # Extract filename for title
        filename = os.path.basename(csv_file).replace('_tokens_avg.csv', '')
        
        # Add titles and labels
        plt.title(f'Token Loss Curve - {filename}', fontsize=14, fontweight='bold')
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Average Loss Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add some statistics as text
        max_loss = df['avg_loss_value'].max()
        min_loss = df['avg_loss_value'].min()
        mean_loss = df['avg_loss_value'].mean()
        
        stats_text = f'Max: {max_loss:.4f}\nMin: {min_loss:.4f}\nMean: {mean_loss:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(output_dir, f'{filename}_token_curve.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {output_file}")
        print(f"  Tokens: {len(df)}, Loss range: {min_loss:.6f} - {max_loss:.6f}")
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def plot_all_files_comparison(input_dir, output_dir, group_by='noise_type'):
    """
    Plot comparison of all files, grouped by noise type or dB level.
    
    Args:
        input_dir (str): Directory containing CSV files.
        output_dir (str): Directory to save output plots.
        group_by (str): 'noise_type' or 'db_level' for grouping.
    """
    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_dir, '*_tokens_avg.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    # Define color mapping for dB levels
    db_color_map = {
        '12': '#2ca02c',    # 绿色 for -12db
        '20': '#ff7f0e',    # 橘色 for -20db  
        '30': '#1f77b4'     # 蓝色 for -30db
    }
    
    # Parse filenames to extract noise type and dB level
    file_info = []
    for file in csv_files:
        basename = os.path.basename(file)
        parts = basename.replace('_tokens_avg.csv', '').split('_')
        if len(parts) >= 3:
            noise_type = parts[0]
            db_level = parts[2].replace('db', '')
            file_info.append({
                'file': file,
                'noise_type': noise_type,
                'db_level': db_level,
                'basename': basename.replace('_tokens_avg.csv', '')
            })
    
    if group_by == 'noise_type':
        # Group by noise type
        noise_types = list(set([info['noise_type'] for info in file_info]))
        
        for noise_type in noise_types:
            plt.figure(figsize=(15, 10))
            
            files_of_type = [info for info in file_info if info['noise_type'] == noise_type]
            # Sort by dB level for consistent ordering
            files_of_type.sort(key=lambda x: int(x['db_level']))
            
            for info in files_of_type:
                df = pd.read_csv(info['file'])
                color = db_color_map.get(info['db_level'], '#000000')  # Default to black if not found
                plt.plot(df['token_position'], df['avg_loss_value'], 
                        marker='o', linestyle='-', markersize=1.5, linewidth=1.5,
                        color=color, label=f"-{info['db_level']}dB")
            
            plt.title(f'Token Loss Comparison - {noise_type.title()} Noise', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Token Position', fontsize=12)
            plt.ylabel('Average Loss Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f'{noise_type}_noise_comparison.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comparison plot saved: {output_file}")
    
    elif group_by == 'db_level':
        # Group by dB level
        db_levels = list(set([info['db_level'] for info in file_info]))
        
        # Define color mapping for noise types
        noise_color_map = {
            'white': '#1f77b4',   # 蓝色
            'pink': '#ff7f0e',    # 橘色
            'brown': '#2ca02c',   # 绿色
            'blue': '#d62728'     # 红色
        }
        
        for db_level in db_levels:
            plt.figure(figsize=(15, 10))
            
            files_of_db = [info for info in file_info if info['db_level'] == db_level]
            
            for info in files_of_db:
                df = pd.read_csv(info['file'])
                color = noise_color_map.get(info['noise_type'], '#000000')  # Default to black if not found
                plt.plot(df['token_position'], df['avg_loss_value'], 
                        marker='o', linestyle='-', markersize=1.5, linewidth=1.5,
                        color=color, label=f"{info['noise_type'].title()} Noise")
            
            plt.title(f'Token Loss Comparison - -{db_level}dB Level', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Token Position', fontsize=12)
            plt.ylabel('Average Loss Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f'{db_level}db_comparison.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comparison plot saved: {output_file}")

def plot_all_files_overview(input_dir, output_dir):
    """
    Plot all files in a single overview plot.
    
    Args:
        input_dir (str): Directory containing CSV files.
        output_dir (str): Directory to save output plot.
    """
    csv_files = glob.glob(os.path.join(input_dir, '*_tokens_avg.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    plt.figure(figsize=(20, 12))
    
    colors = plt.cm.tab20(range(len(csv_files)))
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file).replace('_tokens_avg.csv', '')
            
            plt.plot(df['token_position'], df['avg_loss_value'], 
                    color=colors[i], linestyle='-', linewidth=1.5,
                    label=filename, alpha=0.8)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    plt.title('Token Loss Curves - All Noise Types and dB Levels', 
             fontsize=18, fontweight='bold')
    plt.xlabel('Token Position', fontsize=14)
    plt.ylabel('Average Loss Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'all_noise_token_curves_overview.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overview plot saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot token loss curves from noise color generation CSV files.")
    parser.add_argument("--input_dir", "-i", type=str, 
                       default="loss_noise_color_gen",
                       help="Directory containing CSV files (default: loss_noise_color_gen)")
    parser.add_argument("--output_dir", "-o", type=str, 
                       default="Loss_Plot/noise_color_tokens",
                       help="Directory to save output plots (default: Loss_Plot/noise_color_tokens)")
    parser.add_argument("--plot_type", "-t", type=str, 
                       choices=['individual', 'comparison', 'overview', 'all'],
                       default='all',
                       help="Type of plots to generate (default: all)")
    parser.add_argument("--group_by", "-g", type=str,
                       choices=['noise_type', 'db_level'],
                       default='noise_type',
                       help="How to group comparison plots (default: noise_type)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing CSV files from: {args.input_dir}")
    print(f"Saving plots to: {args.output_dir}")
    
    if args.plot_type in ['individual', 'all']:
        print("\nGenerating individual plots...")
        csv_files = glob.glob(os.path.join(args.input_dir, '*_tokens_avg.csv'))
        for csv_file in csv_files:
            plot_single_file_tokens(csv_file, args.output_dir)
    
    if args.plot_type in ['comparison', 'all']:
        print(f"\nGenerating comparison plots (grouped by {args.group_by})...")
        plot_all_files_comparison(args.input_dir, args.output_dir, args.group_by)
    
    if args.plot_type in ['overview', 'all']:
        print("\nGenerating overview plot...")
        plot_all_files_overview(args.input_dir, args.output_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()