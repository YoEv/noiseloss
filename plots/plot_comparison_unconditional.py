import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def extract_topk_value(filename):
    """Extract topk value from filename"""
    # Extract topk value from filenames like 'loss_cal_unconditional_topk2_samples.txt'
    match = re.search(r'topk(\d+)', filename)
    if match:
        return f"topk{match.group(1)}"
    return "unknown"

def load_and_process_unconditional(file_paths):
    """Load and process unconditional topk data"""
    # Extract topk categories from file paths
    categories = set()
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        topk_category = extract_topk_value(filename)
        categories.add(topk_category)
    
    # Sort categories by topk value
    sorted_categories = sorted(categories, key=lambda x: int(x.replace('topk', '')) if x != 'unknown' else 0)
    
    category_data = {category: [] for category in sorted_categories}
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        topk_category = extract_topk_value(filename)
        
        if topk_category not in category_data:
            continue
            
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    
                    # Parse line format: "filename: loss_value"
                    parts = line.rsplit(':', 1)
                    if len(parts) != 2:
                        continue
                        
                    try:
                        loss = float(parts[1].strip())
                        category_data[topk_category].append(loss)
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Remove empty categories
    return {k: v for k, v in category_data.items() if v}

def calculate_statistics_unconditional(category_data):
    """Calculate statistics for each topk category"""
    stats_data = []
    for category in category_data.keys():
        values = np.array(category_data[category])
        stats_data.append({
            'TopK': category,
            'Mean': np.mean(values),
            'Variance': np.var(values, ddof=1),
            'Std Dev': np.std(values, ddof=1),
            'Count': len(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Median': np.median(values)
        })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('TopK')

def perform_statistical_tests_unconditional(category_data):
    """Perform ANOVA and post-hoc tests for topk comparison"""
    # Prepare data for ANOVA
    groups = []
    group_names = []
    
    for category, values in category_data.items():
        groups.append(values)
        group_names.append(category)
    
    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    # Prepare data for Tukey's HSD test
    all_values = []
    all_groups = []
    
    for category, values in category_data.items():
        all_values.extend(values)
        all_groups.extend([category] * len(values))
    
    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(all_values, all_groups, alpha=0.05)
    
    return {
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'tukey': tukey_results
    }

def create_results_table_unconditional(stats_df, test_results):
    """Create combined results table"""
    # Add ANOVA results to the stats dataframe
    results_df = stats_df.copy()
    
    # Add ANOVA information
    anova_info = f"F-statistic: {test_results['anova']['f_statistic']:.4f}, "
    anova_info += f"p-value: {test_results['anova']['p_value']:.6f}, "
    anova_info += f"Significant: {test_results['anova']['significant']}"
    
    print(f"\nANOVA Results: {anova_info}")
    print("\nTukey's HSD Post-hoc Test Results:")
    print(test_results['tukey'])
    
    return results_df

def plot_comparison_unconditional(stats_df, category_data, output_dir):
    """Create box plot for topk comparison"""
    # Prepare data for plotting
    plot_data = []
    labels = []
    
    # Sort by topk value for better visualization
    sorted_categories = sorted(category_data.keys(), 
                             key=lambda x: int(x.replace('topk', '')) if x != 'unknown' else 0)
    
    for category in sorted_categories:
        plot_data.append(category_data[category])
        labels.append(category)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    box_plot = plt.boxplot(plot_data, labels=labels, patch_artist=True)
    
    # Customize colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the plot
    plt.title('Loss Comparison Across Different TopK Values (Unconditional Generation)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('TopK Value', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, category in enumerate(sorted_categories):
        mean_val = stats_df.loc[category, 'Mean']
        plt.text(i+1, mean_val, f'{mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, 'unconditional_topk_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Box plot saved to: {plot_file}")
    
    # Create a line plot showing the trend
    plt.figure(figsize=(10, 6))
    
    topk_values = [int(cat.replace('topk', '')) for cat in sorted_categories if cat != 'unknown']
    mean_values = [stats_df.loc[cat, 'Mean'] for cat in sorted_categories if cat != 'unknown']
    std_values = [stats_df.loc[cat, 'Std Dev'] for cat in sorted_categories if cat != 'unknown']
    
    plt.errorbar(topk_values, mean_values, yerr=std_values, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    plt.title('Mean Loss vs TopK Value (Unconditional Generation)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('TopK Value', fontsize=12)
    plt.ylabel('Mean Loss Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # Use log scale for better visualization
    
    # Add value labels
    for x, y in zip(topk_values, mean_values):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # Save the trend plot
    trend_file = os.path.join(output_dir, 'unconditional_topk_trend.png')
    plt.savefig(trend_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Trend plot saved to: {trend_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare unconditional generation loss across different TopK values')
    parser.add_argument('-i', '--input', nargs='+', 
                        default=[
                            'loss_cal_unconditional_topk2_samples.txt',
                            'loss_cal_unconditional_topk10_samples.txt', 
                            'loss_cal_unconditional_topk50_samples.txt',
                            'loss_cal_unconditional_topk100_samples.txt',
                            'loss_cal_unconditional_topk150_samples.txt',
                            'loss_cal_unconditional_topk200_samples.txt',
                            'loss_cal_unconditional_topk250_samples.txt',
                            'loss_cal_unconditional_topk500_samples.txt'
                        ],
                        help='Input file paths containing unconditional loss data')
    parser.add_argument('-o', '--output', default="Plot/Unconditional_TopK_Comparison",
                        help='Output directory for charts and results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the data
    print("Loading unconditional topk data...")
    data = load_and_process_unconditional(args.input)
    
    if not data:
        print("No data found in the input files.")
        exit(1)
    
    print(f"Found data for TopK values: {list(data.keys())}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_statistics_unconditional(data)
    
    # Perform statistical tests
    print("\nPerforming ANOVA and post-hoc tests...")
    test_results = perform_statistical_tests_unconditional(data)
    
    # Create results table
    print("\nGenerating statistical results summary...")
    results_df = create_results_table_unconditional(stats_df, test_results)
    
    # Display results
    print("\nStatistical Results Summary:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(results_df.to_string())
    
    # Save results to CSV
    results_file = os.path.join(output_dir, 'unconditional_topk_statistical_results.csv')
    results_df.to_csv(results_file)
    print(f"\nStatistical results saved to: {results_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_comparison_unconditional(stats_df, data, output_dir)
    print(f"\nAnalysis complete. Plots and statistical results saved in {output_dir}")