import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_statistics_from_file(file_path):
    """
    Extract mean and std from a statistics file.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract mean and std from Loss Difference Statistics section
        mean_match = re.search(r'Loss Difference Statistics.*?Mean: ([\d\.-]+)', content, re.DOTALL)
        std_match = re.search(r'Loss Difference Statistics.*?Std: ([\d\.-]+)', content, re.DOTALL)
        
        if mean_match and std_match:
            mean = float(mean_match.group(1))
            std = float(std_match.group(1))
            return mean, std
        else:
            print(f"Warning: Could not extract statistics from {file_path}")
            return None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def get_song_name_from_filename(filename):
    """
    Extract song name from statistics filename.
    """
    # Remove the statistics suffix and extract song info
    song_name = filename.replace('_statistics.txt', '')
    # Remove the loss_diff_ prefix and _200_750 suffix pattern
    song_name = re.sub(r'^loss_diff_', '', song_name)
    song_name = re.sub(r'_\d+_\d+$', '', song_name)
    # Remove the _short30_preview_cut14s_tokens_avg part
    song_name = song_name.replace('_short30_preview_cut14s_tokens_avg', '')
    return song_name

def collect_data_from_directories():
    """
    Collect mean and std data from all 6 categories.
    """
    base_dir = '/home/evev/asap-dataset/Loss_Plot/Phase4_3'
    categories = ['5', '10', '20', '50', '100', '200']
    
    # Dictionary to store data: {song_name: {category: {'mean': value, 'std': value}}}
    data = {}
    
    for category in categories:
        dir_path = os.path.join(base_dir, f'loss_diff_plot_shutter_tk{category}')
        
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist")
            continue
        
        # Navigate through subdirectories to find statistics files
        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path):
                # Look for statistics files in this subdirectory
                for file in os.listdir(subdir_path):
                    if file.endswith('_statistics.txt'):
                        file_path = os.path.join(subdir_path, file)
                        song_name = get_song_name_from_filename(file)
                        
                        mean, std = extract_statistics_from_file(file_path)
                        
                        if mean is not None and std is not None:
                            if song_name not in data:
                                data[song_name] = {}
                            data[song_name][category] = {'mean': mean, 'std': std}
    
    return data, categories

def create_comparison_plots():
    """
    Create comparison plots for mean and std across all categories.
    """
    data, categories = collect_data_from_directories()
    
    if not data:
        print("No data found. Please check the directory structure.")
        return
    
    # Get list of songs that have data for all categories
    complete_songs = []
    for song_name, song_data in data.items():
        if len(song_data) == len(categories):  # Has data for all categories
            complete_songs.append(song_name)
    
    if not complete_songs:
        print("No songs found with complete data across all categories.")
        return
    
    print(f"Found {len(complete_songs)} songs with complete data across all categories.")
    
    # Sort songs for consistent ordering
    complete_songs.sort()
    
    # Prepare data for plotting
    means_data = []
    stds_data = []
    
    for category in categories:
        category_means = []
        category_stds = []
        
        for song in complete_songs:
            category_means.append(data[song][category]['mean'])
            category_stds.append(data[song][category]['std'])
        
        means_data.append(category_means)
        stds_data.append(category_stds)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Mean Loss Difference
    x = np.arange(len(complete_songs))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (category, means) in enumerate(zip(categories, means_data)):
        ax1.plot(x, means, marker='o', linewidth=2, markersize=6, label=f'{category}', color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Songs', fontsize=12)
    ax1.set_ylabel('Mean Loss Difference', fontsize=12)
    ax1.set_title('Mean Loss Difference Statistics (File2 - File1, Noise Region) Across Token Lengths', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([song[:15] + '...' if len(song) > 15 else song for song in complete_songs], 
                        rotation=45, ha='right', fontsize=10)
    ax1.legend(title='Token Length', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    # Set y-axis maximum to 6 for token_length_comparison
    ax1.set_ylim(top=6)
    
    # Plot 2: Standard Deviation of Loss Difference
    for i, (category, stds) in enumerate(zip(categories, stds_data)):
        ax2.plot(x, stds, marker='s', linewidth=2, markersize=6, label=f'{category}', color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Songs', fontsize=12)
    ax2.set_ylabel('Std Loss Difference', fontsize=12)
    ax2.set_title('Standard Deviation of Loss Difference Statistics (File2 - File1, Noise Region) Across Token Lengths', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([song[:15] + '...' if len(song) > 15 else song for song in complete_songs], 
                        rotation=45, ha='right', fontsize=10)
    ax2.legend(title='Token Length', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    # Set y-axis maximum to 6 for std loss difference
    ax2.set_ylim(top=6)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/home/evev/asap-dataset/Plot/shutter_token_length_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Create a new figure for the mean plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate mean of means for each token length
    token_lengths = [5, 10, 20, 50, 100, 200]  # Numeric values for plotting
    mean_of_means = []
    
    for i, category in enumerate(categories):
        category_means = means_data[i]
        mean_of_category = np.mean(category_means)
        mean_of_means.append(mean_of_category)
    
    # Plot the mean of means
    ax3.plot(token_lengths, mean_of_means, marker='o', linewidth=3, markersize=8, 
             color='#1f77b4', alpha=0.8)
    ax3.fill_between(token_lengths, mean_of_means, alpha=0.3, color='#1f77b4')
    
    ax3.set_xlabel('Noise Token Length', fontsize=12)
    ax3.set_ylabel('Mean of Loss Difference', fontsize=12)
    ax3.set_title('Average Loss Difference Across All Songs by Token Length', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Set x-axis to show all token lengths
    ax3.set_xticks(token_lengths)
    ax3.set_xticklabels([f'{tl}tk' for tl in token_lengths])
    
    # Set y-axis maximum to 2.5
    ax3.set_ylim(bottom=min(0, min(mean_of_means) - 0.1), top=2.5)
    
    plt.tight_layout()
    
    # Save the mean plot
    mean_output_file = '/home/evev/asap-dataset/Plot/shutter_mean_by_token_length.png'
    plt.savefig(mean_output_file, dpi=300, bbox_inches='tight')
    print(f"Mean plot saved to {mean_output_file}")
    
    # Create a new figure for the box plot
    fig4, ax5 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    
    for i, category in enumerate(categories):
        category_means = means_data[i]
        box_data.append(category_means)
        box_labels.append(f'{token_lengths[i]}tk')
    
    # Create box plot
    bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Customize box plot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_xlabel('Noise Token Length', fontsize=12)
    ax5.set_ylabel('Mean Loss Difference', fontsize=12)
    ax5.set_title('Distribution of Mean Loss Difference by Token Length (Box Plot)', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    # Save the box plot
    box_output_file = '/home/evev/asap-dataset/Plot/shutter_boxplot_by_token_length.png'
    plt.savefig(box_output_file, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to {box_output_file}")

    # Create a new figure for the count plot
    fig3, ax4 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate counts for each token length
    token_lengths = [5, 10, 20, 50, 100, 200]
    positive_counts = []
    negative_counts = []
    
    for i, category in enumerate(categories):
        category_means = means_data[i]
        positive_count = sum(1 for mean in category_means if mean > 0)
        negative_count = sum(1 for mean in category_means if mean < 0)
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)
    
    # Create bar plot
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, positive_counts, width, label='Mean Loss Diff > 0', 
                    color='#ff7f0e', alpha=0.8)
    bars2 = ax4.bar(x + width/2, negative_counts, width, label='Mean Loss Diff < 0', 
                    color='#1f77b4', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax4.set_xlabel('Noise Token Length', fontsize=12)
    ax4.set_ylabel('Count of Songs', fontsize=12)
    ax4.set_title('Count of Songs with Positive vs Negative Mean Loss Difference by Token Length', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{tl}tk' for tl in token_lengths])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis maximum to 120
    ax4.set_ylim(0, 120)
    
    plt.tight_layout()
    
    # Save the count plot
    count_output_file = '/home/evev/asap-dataset/Plot/shutter_count_by_token_length.png'
    plt.savefig(count_output_file, dpi=300, bbox_inches='tight')
    print(f"Count plot saved to {count_output_file}")
    
    # Save summary statistics
    summary_file = '/home/evev/asap-dataset/Plot/shutter_token_length_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("SHUTTER TOKEN LENGTH COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of songs analyzed: {len(complete_songs)}\n")
        f.write(f"Token length categories: {', '.join(categories)}\n\n")
        
        f.write("MEAN LOSS DIFFERENCE STATISTICS:\n")
        f.write("-"*40 + "\n")
        for i, category in enumerate(categories):
            category_means = means_data[i]
            f.write(f"{category:>6}: Mean={np.mean(category_means):.6f}, Std={np.std(category_means):.6f}, "
                   f"Min={np.min(category_means):.6f}, Max={np.max(category_means):.6f}\n")
        
        f.write("\nSTANDARD DEVIATION STATISTICS:\n")
        f.write("-"*40 + "\n")
        for i, category in enumerate(categories):
            category_stds = stds_data[i]
            f.write(f"{category:>6}: Mean={np.mean(category_stds):.6f}, Std={np.std(category_stds):.6f}, "
                   f"Min={np.min(category_stds):.6f}, Max={np.max(category_stds):.6f}\n")
        
        f.write("\nSONGS INCLUDED:\n")
        f.write("-"*40 + "\n")
        for i, song in enumerate(complete_songs, 1):
            f.write(f"{i:2d}. {song}\n")
    
    print(f"Summary statistics saved to {summary_file}")
    
    # Create a detailed CSV file
    csv_file = '/home/evev/asap-dataset/Plot/shutter_token_length_data.csv'
    
    # Prepare data for CSV
    csv_data = []
    for song in complete_songs:
        row = {'Song': song}
        for category in categories:
            row[f'{category}_mean'] = data[song][category]['mean']
            row[f'{category}_std'] = data[song][category]['std']
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print(f"Detailed data saved to {csv_file}")
    
    plt.show()

if __name__ == "__main__":
    create_comparison_plots()