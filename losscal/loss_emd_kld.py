import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
import argparse
from tqdm import tqdm
from scipy import stats


def extract_features(audio_path, sr=22050, n_mfcc=20, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract MFCC, mel spectrogram, and STFT features from an audio file.

    Parameters:
        audio_path (str): Path to the audio file
        sr (int): Sampling rate
        n_mfcc (int): Number of MFCC coefficients
        n_mels (int): Number of mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop length for frame-wise analysis

    Returns:
        dict: Dictionary containing extracted features
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # Extract STFT
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft, hop_length=hop_length)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                hop_length=hop_length)

    # Normalize features to ensure they sum to 1 for probability distributions
    stft_norm = stft / np.sum(stft)
    mel_spec_norm = mel_spec / np.sum(mel_spec)
    mfcc_norm = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc) + 1e-10)
    mfcc_norm = mfcc_norm / np.sum(mfcc_norm)

    return {
        'mfcc': mfcc_norm,
        'mel_spec': mel_spec_norm,
        'stft': stft_norm,
        'filename': os.path.basename(audio_path)
    }

def compute_emd(dist1, dist2):
    """
    Compute Earth Mover's Distance (Wasserstein distance) between two distributions.

    For multi-dimensional features, we compute the mean of EMD across all dimensions.

    Parameters:
        dist1 (np.ndarray): First distribution
        dist2 (np.ndarray): Second distribution

    Returns:
        float: EMD score
    """
    # Handle multi-dimensional distributions by computing EMD for each dimension
    if len(dist1.shape) > 1:
        emd_scores = []
        for i in range(dist1.shape[0]):
            # Ensure distributions sum to 1
            p = dist1[i] / np.sum(dist1[i])
            q = dist2[i] / np.sum(dist2[i])
            # Add small epsilon to avoid zero values
            p = p + 1e-10
            q = q + 1e-10
            # Normalize again
            p = p / np.sum(p)
            q = q / np.sum(q)
            emd_scores.append(wasserstein_distance(p, q))
        return np.mean(emd_scores)
    else:
        # For 1D distributions
        return wasserstein_distance(dist1, dist2)


def compute_kld(dist1, dist2):
    """
    Compute Kullback-Leibler divergence between two distributions.

    Args:
        dist1, dist2: numpy arrays representing probability distributions

    Returns:
        float: KLD score
    """
    # Handle multi-dimensional distributions by computing KLD for each dimension
    if len(dist1.shape) > 1:
        kld_scores = []
        for i in range(dist1.shape[0]):
            # Ensure distributions sum to 1
            p = dist1[i] / np.sum(dist1[i])
            q = dist2[i] / np.sum(dist2[i])
            # Add small epsilon to avoid zero values
            p = p + 1e-10
            q = q + 1e-10
            # Normalize again
            p = p / np.sum(p)
            q = q / np.sum(q)

            # Handle different lengths by truncating to the minimum length
            min_length = min(len(p), len(q))
            p = p[:min_length]
            q = q[:min_length]

            # Compute KLD
            kld = np.sum(rel_entr(p, q))
            kld_scores.append(kld)
        return np.mean(kld_scores)
    else:
        # For 1D distributions
        # Ensure distributions sum to 1
        p = dist1 / np.sum(dist1)
        q = dist2 / np.sum(dist2)
        # Add small epsilon to avoid zero values
        p = p + 1e-10
        q = q + 1e-10
        # Normalize again
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Handle different lengths by truncating to the minimum length
        min_length = min(len(p), len(q))
        p = p[:min_length]
        q = q[:min_length]

        # Compute KLD
        return np.sum(rel_entr(p, q))


def get_file_group(filename):
    """
    Determine the group of a file based on its suffix.

    Parameters:
        filename (str): The filename to check

    Returns:
        str: The group name
    """
    # 支持 asap_100_rhythm 数据集的文件命名模式
    if "_ori_cut15s.wav" in filename:
        return "ori"
    elif "_rhythm1_4_step1_r10_cut15s.wav" in filename:
        return "r10"
    elif "_rhythm1_4_step1_r20_cut15s.wav" in filename:
        return "r20"
    elif "_rhythm1_4_step1_r30_cut15s.wav" in filename:
        return "r30"
    elif "_rhythm1_4_step1_r40_cut15s.wav" in filename:
        return "r40"
    # 支持 Shutter_color_30p 数据集的文件命名模式
    elif "_brown_0db_30p.mp3" in filename:
        return "brown_0db"
    elif "_brown_6db_30p.mp3" in filename:
        return "brown_6db"
    elif "_pink_0db_30p.mp3" in filename:
        return "pink_0db"
    elif "_pink_6db_30p.mp3" in filename:
        return "pink_6db"
    else:
        return "unknown"


def compare_audio_files(original_dir, generated_dir, output_dir='results'):
    """
    Compare audio files in original_dir with corresponding files in generated_dir
    using EMD and KLD on MFCC, mel spectrogram, and STFT features.
    Group files by their suffixes and calculate metrics for each group.

    Parameters:
        original_dir (str): Directory containing original audio files
        generated_dir (str): Directory containing generated audio files
        output_dir (str): Directory to save results

    Returns:
        pd.DataFrame: DataFrame containing comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of files in original directory
    # 支持 .wav 和 .mp3 文件格式
    original_files = [f for f in os.listdir(original_dir) if f.endswith(('.wav', '.mp3'))]

    # Initialize results dictionary
    results = {
        'original_file': [],
        'generated_file': [],
        'group': [],
        'mfcc_emd': [],
        'mel_spec_emd': [],
        'stft_emd': [],
        'mfcc_kld': [],
        'mel_spec_kld': [],
        'stft_kld': []
    }

    # Process each file
    for orig_file in tqdm(original_files, desc="Processing files"):
        # Find corresponding file in generated directory
        gen_file = orig_file  # Assuming same filename in both directories

        if gen_file not in os.listdir(generated_dir):
            print(f"Warning: No matching file for {orig_file} in generated directory")
            continue

        # Determine the group based on file suffix
        group = get_file_group(orig_file)
        if group == "unknown":
            print(f"Warning: Unknown group for file {orig_file}")
            continue

        # Extract features
        orig_features = extract_features(os.path.join(original_dir, orig_file))
        gen_features = extract_features(os.path.join(generated_dir, gen_file))

        if orig_features is None or gen_features is None:
            continue

        # Compute EMD and KLD for each feature type
        mfcc_emd = compute_emd(orig_features['mfcc'], gen_features['mfcc'])
        mel_spec_emd = compute_emd(orig_features['mel_spec'], gen_features['mel_spec'])
        stft_emd = compute_emd(orig_features['stft'], gen_features['stft'])

        mfcc_kld = compute_kld(orig_features['mfcc'], gen_features['mfcc'])
        mel_spec_kld = compute_kld(orig_features['mel_spec'], gen_features['mel_spec'])
        stft_kld = compute_kld(orig_features['stft'], gen_features['stft'])

        # Add results to dictionary
        results['original_file'].append(orig_file)
        results['generated_file'].append(gen_file)
        results['group'].append(group)
        results['mfcc_emd'].append(mfcc_emd)
        results['mel_spec_emd'].append(mel_spec_emd)
        results['stft_emd'].append(stft_emd)
        results['mfcc_kld'].append(mfcc_kld)
        results['mel_spec_kld'].append(mel_spec_kld)
        results['stft_kld'].append(stft_kld)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'audio_similarity_results.csv'), index=False)

    # Calculate and print average scores by group
    group_avg_scores = results_df.groupby('group').mean()

    print("\nAverage Scores by Group:")
    print(group_avg_scores[['mfcc_emd', 'mel_spec_emd', 'stft_emd', 'mfcc_kld', 'mel_spec_kld', 'stft_kld']])

    # Perform statistical analysis to check for significant differences between groups
    perform_statistical_analysis(results_df, output_dir)

    # Create visualizations
    create_visualizations(results_df, output_dir)

    return results_df


def perform_statistical_analysis(results_df, output_dir):
    """
    Perform statistical analysis to check for significant differences between groups.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing comparison results
        output_dir (str): Directory to save results
    """
    metrics = ['mfcc_emd', 'mel_spec_emd', 'stft_emd', 'mfcc_kld', 'mel_spec_kld', 'stft_kld']
    groups = results_df['group'].unique()

    # Create a DataFrame to store ANOVA results
    anova_results = pd.DataFrame(columns=['Metric', 'F-statistic', 'p-value', 'Significant'])

    for metric in metrics:
        # Prepare data for ANOVA
        data_by_group = [results_df[results_df['group'] == group][metric].values for group in groups]

        # Perform one-way ANOVA
        f_stat, p_val = stats.f_oneway(*data_by_group)

        # Add results to DataFrame
        anova_results = pd.concat([anova_results, pd.DataFrame({
            'Metric': [metric],
            'F-statistic': [f_stat],
            'p-value': [p_val],
            'Significant': ['Yes' if p_val < 0.05 else 'No']
        })], ignore_index=True)

    # Save ANOVA results to CSV
    anova_results.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False)

    print("\nStatistical Analysis Results (ANOVA):")
    print(anova_results)

    # If significant differences are found, perform post-hoc tests
    for metric in metrics:
        if anova_results[anova_results['Metric'] == metric]['Significant'].values[0] == 'Yes':
            print(f"\nPost-hoc tests for {metric}:")

            # Create a DataFrame for pairwise comparisons
            posthoc_results = pd.DataFrame(columns=['Group 1', 'Group 2', 't-statistic', 'p-value', 'Significant'])

            # Perform pairwise t-tests
            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    data1 = results_df[results_df['group'] == group1][metric].values
                    data2 = results_df[results_df['group'] == group2][metric].values

                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

                    # Add results to DataFrame
                    posthoc_results = pd.concat([posthoc_results, pd.DataFrame({
                        'Group 1': [group1],
                        'Group 2': [group2],
                        't-statistic': [t_stat],
                        'p-value': [p_val],
                        'Significant': ['Yes' if p_val < 0.05 else 'No']
                    })], ignore_index=True)

            # Save post-hoc results to CSV
            posthoc_results.to_csv(os.path.join(output_dir, f'posthoc_{metric}.csv'), index=False)
            print(posthoc_results)


def create_visualizations(results_df, output_dir):
    """
    Create visualizations of the similarity results, grouped by file suffix.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing comparison results
        output_dir (str): Directory to save visualizations
    """
    # Get unique groups
    groups = sorted(results_df['group'].unique())

    # Define colors for each group
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Create boxplots for EMD scores by group
    plt.figure(figsize=(15, 8))

    # Create positions for grouped boxplots
    feature_types = ['MFCC', 'Mel Spectrogram', 'STFT']
    positions = np.arange(len(feature_types))
    width = 0.15

    for i, group in enumerate(groups):
        group_data = results_df[results_df['group'] == group]
        emd_data = [group_data['mfcc_emd'], group_data['mel_spec_emd'], group_data['stft_emd']]

        bp = plt.boxplot(emd_data, positions=positions + (i - 2) * width, widths=width,
                         patch_artist=True, boxprops=dict(facecolor=colors[i]))

        # Add a label for the group in the legend
        plt.plot([], [], color=colors[i], label=group)

    plt.title('Earth Mover\'s Distance (EMD) Scores by Group')
    plt.ylabel('EMD Score (lower is better)')
    plt.xlabel('Feature Type')
    plt.xticks(positions, feature_types)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emd_boxplot_by_group.png'), dpi=300, bbox_inches='tight')

    # Create boxplots for KLD scores by group
    plt.figure(figsize=(15, 8))

    for i, group in enumerate(groups):
        group_data = results_df[results_df['group'] == group]
        kld_data = [group_data['mfcc_kld'], group_data['mel_spec_kld'], group_data['stft_kld']]

        bp = plt.boxplot(kld_data, positions=positions + (i - 2) * width, widths=width,
                         patch_artist=True, boxprops=dict(facecolor=colors[i]))

        # Add a label for the group in the legend
        plt.plot([], [], color=colors[i], label=group)

    plt.title('Kullback-Leibler Divergence (KLD) Scores by Group')
    plt.ylabel('KLD Score (lower is better)')
    plt.xlabel('Feature Type')
    plt.xticks(positions, feature_types)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kld_boxplot_by_group.png'), dpi=300, bbox_inches='tight')

    # Create bar chart for average EMD scores by group and feature type
    plt.figure(figsize=(15, 8))

    # Calculate average scores by group
    group_avg = results_df.groupby('group').mean()

    # Set up bar positions
    bar_width = 0.15
    index = np.arange(len(feature_types))

    # Plot bars for each group
    for i, group in enumerate(groups):
        plt.bar(index + (i - 2) * bar_width,
                [group_avg.loc[group, 'mfcc_emd'],
                 group_avg.loc[group, 'mel_spec_emd'],
                 group_avg.loc[group, 'stft_emd']],
                bar_width, label=group, color=colors[i])

    plt.xlabel('Feature Type')
    plt.ylabel('Average EMD Score')
    plt.title('Average EMD Scores by Group and Feature Type')
    plt.xticks(index, feature_types)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_emd_by_group.png'), dpi=300, bbox_inches='tight')

    # Create bar chart for average KLD scores by group and feature type
    plt.figure(figsize=(15, 8))

    # Plot bars for each group
    for i, group in enumerate(groups):
        plt.bar(index + (i - 2) * bar_width,
                [group_avg.loc[group, 'mfcc_kld'],
                 group_avg.loc[group, 'mel_spec_kld'],
                 group_avg.loc[group, 'stft_kld']],
                bar_width, label=group, color=colors[i])

    plt.xlabel('Feature Type')
    plt.ylabel('Average KLD Score')
    plt.title('Average KLD Scores by Group and Feature Type')
    plt.xticks(index, feature_types)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_kld_by_group.png'), dpi=300, bbox_inches='tight')

    # Create a combined visualization showing both EMD and KLD for all groups
    plt.figure(figsize=(18, 10))

    # Set up positions for the grouped bar chart
    n_groups = len(groups)
    n_metrics = 2  # EMD and KLD
    n_features = len(feature_types)

    # Create a more complex grouped bar chart
    # Outer grouping by feature type, then by metric, then by group
    group_width = 0.8
    metric_width = group_width / n_metrics
    bar_width = metric_width / n_groups

    for f_idx, feature in enumerate(feature_types):
        for m_idx, metric in enumerate(['EMD', 'KLD']):
            for g_idx, group in enumerate(groups):
                x_pos = f_idx + (m_idx - 0.5) * metric_width + (g_idx - n_groups/2 + 0.5) * bar_width

                if metric == 'EMD':
                    if feature == 'MFCC':
                        value = group_avg.loc[group, 'mfcc_emd']
                    elif feature == 'Mel Spectrogram':
                        value = group_avg.loc[group, 'mel_spec_emd']
                    else:  # STFT
                        value = group_avg.loc[group, 'stft_emd']
                else:  # KLD
                    if feature == 'MFCC':
                        value = group_avg.loc[group, 'mfcc_kld']
                    elif feature == 'Mel Spectrogram':
                        value = group_avg.loc[group, 'mel_spec_kld']
                    else:  # STFT
                        value = group_avg.loc[group, 'stft_kld']

                # Plot the bar
                plt.bar(x_pos, value, bar_width, color=colors[g_idx],
                        alpha=0.7 if metric == 'EMD' else 1.0,
                        hatch='/' if metric == 'KLD' else None)

    # Set up the x-axis
    plt.xlabel('Feature Type and Metric')
    plt.ylabel('Score Value')
    plt.title('Comparison of EMD and KLD Scores Across Groups and Features')

    # Create custom x-tick positions and labels
    tick_positions = []
    tick_labels = []

    for f_idx, feature in enumerate(feature_types):
        tick_positions.append(f_idx)
        tick_labels.append(feature)

    plt.xticks(tick_positions, tick_labels)

    # Create a custom legend
    from matplotlib.patches import Patch

    # Legend for groups
    group_patches = [Patch(color=colors[i], label=group) for i, group in enumerate(groups)]

    # Legend for metrics
    metric_patches = [
        Patch(color='gray', alpha=0.7, label='EMD'),
        Patch(color='gray', hatch='/', label='KLD')
    ]

    # Add both legends
    plt.legend(handles=group_patches + metric_patches, loc='upper right')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics_by_group.png'), dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Calculate similarity between audio files using EMD and KLD.')
    parser.add_argument('--original_dir', type=str, default='/home/evev/asap-dataset/Shutter_color_30p',
                        help='Directory containing original audio files')
    parser.add_argument('--generated_dir', type=str, default='/home/evev/asap-dataset/gen_Shutter_color_30p',
                        help='Directory containing generated audio files')
    parser.add_argument('--output_dir', type=str, default='Shutter_color_30p_emd_kld_results',
                        help='Directory to save results')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sampling rate for audio processing')
    parser.add_argument('--n_mfcc', type=int, default=20,
                        help='Number of MFCC coefficients')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel bands')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Hop length for frame-wise analysis')

    args = parser.parse_args()

    print(f"Comparing audio files in:\n  {args.original_dir}\n  {args.generated_dir}")
    print(f"Results will be saved to: {args.output_dir}")

    # Compare audio files
    results_df = compare_audio_files(
        args.original_dir,
        args.generated_dir,
        args.output_dir
    )

    print(f"\nResults saved to {os.path.join(args.output_dir, 'audio_similarity_results.csv')}")
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()