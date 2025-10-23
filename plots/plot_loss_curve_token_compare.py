import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import argparse

def load_reference_data(reference_file):
    """
    Load reference data (white noise 30db)
    """
    df = pd.read_csv(reference_file)
    return df['token_position'].values, df['avg_loss_value'].values

def load_asap_data_from_directory(directory_path):
    """
    Load and combine all CSV files from a directory.
    Extract filename from file path and add it as a column.
    """
    all_data = []
    
    for csv_file in os.listdir(directory_path):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(directory_path, csv_file)
            df = pd.read_csv(file_path)
            
            # Extract filename without extension and path
            filename = os.path.splitext(csv_file)[0]
            # Remove the suffix '_ori_cut15s_tokens_avg' to get the base filename
            if filename.endswith('_ori_cut15s_tokens_avg'):
                filename = filename[:-len('_ori_cut15s_tokens_avg')]
            
            # Add filename column
            df['filename'] = filename
            
            # Rename columns to match expected format
            if 'token_position' in df.columns:
                df = df.rename(columns={'token_position': 'token_idx', 'avg_loss_value': 'loss'})
            
            all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No CSV files found in directory: {directory_path}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Group by filename
    grouped = combined_df.groupby('filename')
    
    return grouped

def plot_comparison(reference_tokens, reference_loss, filename, ori_data, compare_data, 
                   compare_label, output_dir):
    """
    Plot comparison between reference, original, and comparison data for one file
    """
    plt.figure(figsize=(15, 10))
    
    # Plot reference data (blue)
    plt.plot(reference_tokens, reference_loss, 
             color='#1f77b4', linestyle='-', linewidth=2, 
             label='White Noise -30dB (Reference)', alpha=0.8)
    
    # Plot original data (green)
    if len(ori_data) > 0:
        plt.plot(ori_data['token_idx'], ori_data['loss'], 
                 color='#2ca02c', linestyle='-', linewidth=2, 
                 label='Original Data', alpha=0.8)
    
    # Plot comparison data (orange)
    if len(compare_data) > 0:
        plt.plot(compare_data['token_idx'], compare_data['loss'], 
                 color='#ff7f0e', linestyle='-', linewidth=2, 
                 label=compare_label, alpha=0.8)
    
    # Formatting
    plt.title(f'Loss Curve Comparison - {os.path.basename(filename)}\n{compare_label}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Token Index', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save plot
    safe_filename = os.path.basename(filename).replace('.wav', '').replace('/', '_')
    compare_suffix = compare_label.replace(' ', '_').replace('(', '').replace(')', '')
    output_file = os.path.join(output_dir, f'{safe_filename}_{compare_suffix}.png')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot loss curve comparisons')
    parser.add_argument('--reference_file', 
                       default='loss_noise_color_gen/white_noise_30db_15.0s_tokens_avg.csv',
                       help='Reference data file (white noise 30db)')
    parser.add_argument('--ori_dir', 
                       default='asap_loss_time_ins_ori',
                       help='Original ASAP data directory')
    parser.add_argument('--output_dir', 
                       default='Loss_Plot/asap_noise_ori_replaceNoise_token_comparison',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Define comparison directories and their labels
    comparison_dirs = {
        'asap_loss_time_ins_tk5': '5 Tokens',
        'asap_loss_time_ins_tk10': '10 Tokens',
        'asap_loss_time_ins_tk20': '20 Tokens',
        'asap_loss_time_ins_tk50': '50 Tokens',
        'asap_loss_time_ins_tk100': '100 Tokens',
        'asap_loss_time_ins_tk200': '200 Tokens'
    }
    
    # Load reference data
    print(f"Loading reference data from {args.reference_file}...")
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file {args.reference_file} not found")
        return
    reference_tokens, reference_loss = load_reference_data(args.reference_file)
    
    # Load original data
    print(f"Loading original data from {args.ori_dir}...")
    if not os.path.exists(args.ori_dir):
        print(f"Error: Original directory {args.ori_dir} not found")
        return
    ori_grouped = load_asap_data_from_directory(args.ori_dir)
    if ori_grouped is None:
        print(f"Error: No data loaded from {args.ori_dir}")
        return
    
    # Get list of all filenames
    all_filenames = list(ori_grouped.groups.keys())
    print(f"Found {len(all_filenames)} files to process")
    
    # Process each comparison directory
    for compare_dir, compare_label in comparison_dirs.items():
        if not os.path.exists(compare_dir):
            print(f"Warning: {compare_dir} not found, skipping...")
            continue
            
        print(f"\nProcessing {compare_label} ({compare_dir})...")
        compare_grouped = load_asap_data_from_directory(compare_dir)
        
        if compare_grouped is None:
            print(f"Warning: No data loaded from {compare_dir}, skipping...")
            continue
        
        # Create plots for each filename
        for filename in all_filenames:
            try:
                # Get original data for this file
                ori_data = ori_grouped.get_group(filename) if filename in ori_grouped.groups else pd.DataFrame()
                
                # Get comparison data for this file
                compare_data = compare_grouped.get_group(filename) if filename in compare_grouped.groups else pd.DataFrame()
                
                # Create comparison plot
                plot_comparison(reference_tokens, reference_loss, filename, 
                              ori_data, compare_data, compare_label, args.output_dir)
                
            except Exception as e:
                print(f"Error processing {filename} for {compare_label}: {e}")
                continue
    
    print(f"\nAll plots saved to {args.output_dir}")
    print(f"Total files processed: {len(all_filenames)}")
    print(f"Total plots generated: {len(all_filenames) * len(comparison_dirs)}")

if __name__ == "__main__":
    main()