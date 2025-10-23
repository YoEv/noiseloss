import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re

def load_and_process_wav_data(file_path, suffix=None, only_ori=False):
    """Load and process wav data from loss file

    Args:
        file_path: Path to the loss file
        suffix: Specific suffix to filter files (e.g., '_pitch1_1_step1_semi10')
        only_ori: Only load original files with '_ori' in name
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                filename, loss = line.rsplit(':', 1)
                filename = filename.strip()

                # Filter by suffix if provided
                if suffix and suffix not in filename:
                    continue

                # Skip non-ori files if only_ori is True
                if only_ori and '_ori' not in filename:
                    continue

                try:
                    loss = float(loss.strip())
                    if not np.isnan(loss):  # Only add non-NaN values
                        data.append(loss)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    return np.array(data)

def calculate_statistics(data_dict):
    """Calculate statistics for each dataset"""
    stats_data = []
    for dataset_name, values in data_dict.items():
        if len(values) == 0:
            continue
        stats_data.append({
            'Dataset': dataset_name,
            'Mean': np.mean(values),
            'Variance': np.var(values, ddof=1),
            'Std Dev': np.std(values, ddof=1),
            'Count': len(values)
        })
    return pd.DataFrame(stats_data)

def perform_statistical_tests(data_dict, reference_key='Original'):
    """Perform statistical tests between datasets

    Args:
        data_dict: Dictionary of datasets
        reference_key: Key for reference dataset to compare against
    """
    datasets = list(data_dict.keys())
    results = {
        'Comparison': [],
        'Mean Diff': [],
        't-statistic': [],
        'p-value': [],
        'Effect Size': []
    }

    # Check if reference key exists
    if reference_key not in data_dict or len(data_dict[reference_key]) == 0:
        print(f"Warning: Reference key '{reference_key}' not found or empty")
        # Compare all pairs if no reference
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                data1 = data_dict[datasets[i]]
                data2 = data_dict[datasets[j]]

                if len(data1) == 0 or len(data2) == 0:
                    continue

                # Calculate statistics
                mean_diff = np.mean(data1) - np.mean(data2)
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

                # Calculate Cohen's d effect size
                pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
                effect_size = mean_diff / pooled_std if pooled_std != 0 else 0

                results['Comparison'].append(f"{datasets[i]} vs {datasets[j]}")
                results['Mean Diff'].append(mean_diff)
                results['t-statistic'].append(t_stat)
                results['p-value'].append(p_val)
                results['Effect Size'].append(effect_size)
    else:
        # Compare each dataset against reference
        reference_data = data_dict[reference_key]
        for dataset in datasets:
            if dataset == reference_key or len(data_dict[dataset]) == 0:
                continue

            data = data_dict[dataset]

            # Calculate statistics
            mean_diff = np.mean(data) - np.mean(reference_data)
            t_stat, p_val = stats.ttest_ind(data, reference_data, equal_var=False)

            # Calculate Cohen's d effect size
            pooled_std = np.sqrt((np.var(data, ddof=1) + np.var(reference_data, ddof=1)) / 2)
            effect_size = mean_diff / pooled_std if pooled_std != 0 else 0

            results['Comparison'].append(f"{dataset} vs {reference_key}")
            results['Mean Diff'].append(mean_diff)
            results['t-statistic'].append(t_stat)
            results['p-value'].append(p_val)
            results['Effect Size'].append(effect_size)

    return pd.DataFrame(results)

def plot_comparison(data_dict, output_dir, title_prefix=""):
    """Create comparison plots"""
    plt.style.use('seaborn')

    # Filter out empty datasets
    filtered_dict = {k: v for k, v in data_dict.items() if len(v) > 0}

    if len(filtered_dict) <= 1:
        print("Warning: Not enough datasets with data for comparison")
        return

    # 1. Boxplot comparison
    plt.figure(figsize=(12, 8))
    plt.boxplot([data for data in filtered_dict.values()], labels=list(filtered_dict.keys()))
    plt.title(f'{title_prefix}Loss Distribution Comparison')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}loss_distribution_comparison.png'), dpi=300)
    plt.close()

    # 2. Violin plot
    plt.figure(figsize=(12, 8))
    plt.violinplot([data for data in filtered_dict.values()])
    plt.title(f'{title_prefix}Loss Distribution Density')
    plt.ylabel('Loss Value')
    plt.xticks(range(1, len(filtered_dict.keys()) + 1), list(filtered_dict.keys()), rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}loss_density_comparison.png'), dpi=300)
    plt.close()

def load_data_by_category(file_path, category_suffixes, include_original=True):
    """Load data for multiple category suffixes

    Args:
        file_path: Path to the loss file
        category_suffixes: Dictionary of category names and their suffixes
        include_original: Whether to include original files
    """
    data_dict = {}

    # Add original data if requested
    if include_original:
        original_data = load_and_process_wav_data(file_path, only_ori=True)
        if len(original_data) > 0:
            data_dict["Original"] = original_data

    # Add data for each category suffix
    for category_name, suffix in category_suffixes.items():
        category_data = load_and_process_wav_data(file_path, suffix=suffix)
        if len(category_data) > 0:
            data_dict[category_name] = category_data

    return data_dict

if __name__ == "__main__":
    # Set up paths
    output_dir = "Loss_Comparison_WAV"
    os.makedirs(output_dir, exist_ok=True)

    # Define pitch category suffixes
    pitch_semi_suffixes = {
        "Semi 10%": "_pitch1_1_step1_semi10",
        "Semi 30%": "_pitch1_1_step1_semi30",
        "Semi 50%": "_pitch1_1_step1_semi50",
        "Semi 80%": "_pitch1_1_step1_semi80"
    }

    pitch_oct_suffixes = {
        "Octave 10%": "_pitch1_2_step1_oct10",
        "Octave 30%": "_pitch1_2_step1_oct30",
        "Octave 50%": "_pitch1_2_step1_oct50",
        "Octave 80%": "_pitch1_2_step1_oct80"
    }

    pitch_dia_suffixes = {
        "Diatonic 10%": "_pitch1_3_step1_dia10",
        "Diatonic 30%": "_pitch1_3_step1_dia30",
        "Diatonic 50%": "_pitch1_3_step1_dia50",
        "Diatonic 80%": "_pitch1_3_step1_dia80"
    }

    # Define rhythm category suffixes
    rhythm_suffixes = {
        "Rhythm 10%": "_rhythm1_4_step1_r10",
        "Rhythm 20%": "_rhythm1_4_step1_r20",
        "Rhythm 30%": "_rhythm1_4_step1_r30",
        "Rhythm 40%": "_rhythm1_4_step1_r40"
    }

    # Define structure category suffixes
    structure_suffixes = {
        "Tempo 10%": "_tempo1_1_step1_nt10",
        "Tempo 30%": "_tempo1_1_step1_nt30",
        "Tempo 50%": "_tempo1_1_step1_nt50",
        "Tempo 80%": "_tempo1_1_step1_nt80"
    }

    # Define velocity category suffixes
    velocity_suffixes = {
        "Velocity 10%": "_velocity1_1_step1_nt10",
        "Velocity 30%": "_velocity1_1_step1_nt30",
        "Velocity 50%": "_velocity1_1_step1_nt50",
        "Velocity 80%": "_velocity1_1_step1_nt80"
    }

    # Load pitch data
    print("\nProcessing Pitch Semitone Data...")
    pitch_semi_data = load_data_by_category('asap_pitch_mel_results.txt', pitch_semi_suffixes)

    print("\nProcessing Pitch Octave Data...")
    pitch_oct_data = load_data_by_category('asap_pitch_mel_results.txt', pitch_oct_suffixes)

    print("\nProcessing Pitch Diatonic Data...")
    pitch_dia_data = load_data_by_category('asap_pitch_mel_results.txt', pitch_dia_suffixes)

    # Load rhythm data
    print("\nProcessing Rhythm Data...")
    rhythm_data = load_data_by_category('asap_rhythm_mel_results.txt', rhythm_suffixes)

    # Load structure data
    print("\nProcessing Structure Data...")
    structure_data = load_data_by_category('asap_structure_mel_results.txt', structure_suffixes)

    # Load velocity data
    print("\nProcessing Velocity Data...")
    velocity_data = load_data_by_category('asap_velocity_mel_results.txt', velocity_suffixes)

    # Process and plot pitch semitone data
    if pitch_semi_data:
        stats_df = calculate_statistics(pitch_semi_data)
        print("\nPitch Semitone Descriptive Statistics:")
        print(stats_df.to_string(float_format="%.8f"))

        test_results = perform_statistical_tests(pitch_semi_data)
        print("\nPitch Semitone Statistical Test Results:")
        print(test_results.to_string(float_format="%.8f"))

        plot_comparison(pitch_semi_data, output_dir, "Pitch Semitone ")

    # Process and plot pitch octave data
    if pitch_oct_data:
        stats_df = calculate_statistics(pitch_oct_data)
        print("\nPitch Octave Descriptive Statistics:")
        print(stats_df.to_string(float_format="%.8f"))

        test_results = perform_statistical_tests(pitch_oct_data)
        print("\nPitch Octave Statistical Test Results:")
        print(test_results.to_string(float_format="%.8f"))

        plot_comparison(pitch_oct_data, output_dir, "Pitch Octave ")

    # Process and plot pitch diatonic data
    if pitch_dia_data:
        stats_df = calculate_statistics(pitch_dia_data)
        print("\nPitch Diatonic Descriptive Statistics:")
        print(stats_df.to_string(float_format="%.8f"))

        test_results = perform_statistical_tests(pitch_dia_data)
        print("\nPitch Diatonic Statistical Test Results:")
        print(test_results.to_string(float_format="%.8f"))

        plot_comparison(pitch_dia_data, output_dir, "Pitch Diatonic ")

    # Process and plot rhythm data
    if rhythm_data:
        stats_df = calculate_statistics(rhythm_data)
        print("\nRhythm Descriptive Statistics:")
        print(stats_df.to_string(float_format="%.8f"))

        test_results = perform_statistical_tests(rhythm_data)
        print("\nRhythm Statistical Test Results:")
        print(test_results.to_string(float_format="%.8f"))

        plot_comparison(rhythm_data, output_dir, "Rhythm ")

    # Process and plot structure data
    if structure_data:
        stats_df = calculate_statistics(structure_data)
        print("\nStructure Descriptive Statistics:")
        print(stats_df.to_string(float_format="%.8f"))

        test_results = perform_statistical_tests(structure_data)
        print("\nStructure Statistical Test Results:")
        print(test_results.to_string(float_format="%.8f"))

        plot_comparison(structure_data, output_dir, "Structure ")

    # Process and plot velocity data
    if velocity_data:
        stats_df = calculate_statistics(velocity_data)
        print("\nVelocity Descriptive Statistics:")
        print(stats_df.to_string(float_format="%.8f"))

        test_results = perform_statistical_tests(velocity_data)
        print("\nVelocity Statistical Test Results:")
        print(test_results.to_string(float_format="%.8f"))

        plot_comparison(velocity_data, output_dir, "Velocity ")

    print(f"\nAnalysis complete. Plots saved in {output_dir}")