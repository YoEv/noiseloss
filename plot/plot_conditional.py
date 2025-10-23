import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def extract_dataset_type(filename):
    """Extract dataset type from filename"""
    if 'asap_100' in filename.lower():
        return 'ASAP-100 (Original)'
    elif 'conditional' in filename.lower():
        # Extract topk value if present
        match = re.search(r'tk(\d+)', filename)
        if match:
            return f'Generated Conditional (TopK{match.group(1)})'
        else:
            return 'Generated Conditional'
    else:
        return 'Unknown'

def load_and_process_conditional(file_paths):
    """Load and process conditional comparison data"""
    # Extract dataset categories from file paths
    categories = set()
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        dataset_type = extract_dataset_type(filename)
        categories.add(dataset_type)
    
    category_data = {category: [] for category in categories}
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        dataset_type = extract_dataset_type(filename)
        
        if dataset_type not in category_data:
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
                        category_data[dataset_type].append(loss)
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Remove empty categories
    return {k: v for k, v in category_data.items() if v}

def calculate_statistics_conditional(category_data):
    """Calculate statistics for each dataset category"""
    stats_data = []
    for category in category_data.keys():
        values = np.array(category_data[category])
        stats_data.append({
            'Dataset': category,
            'Mean': np.mean(values),
            'Variance': np.var(values, ddof=1),
            'Std Dev': np.std(values, ddof=1),
            'Count': len(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Median': np.median(values),
            'Q1': np.percentile(values, 25),
            'Q3': np.percentile(values, 75),
            'IQR': np.percentile(values, 75) - np.percentile(values, 25)
        })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('Dataset')

def perform_statistical_tests_conditional(category_data):
    """Perform statistical tests for conditional comparison"""
    categories = list(category_data.keys())
    
    if len(categories) != 2:
        print(f"Warning: Expected 2 categories for comparison, found {len(categories)}")
        return {}
    
    group1_name, group2_name = categories
    group1_data = np.array(category_data[group1_name])
    group2_data = np.array(category_data[group2_name])
    
    # Perform t-test (assuming normal distribution)
    t_stat, t_p_value = ttest_ind(group1_data, group2_data)
    
    # Perform Mann-Whitney U test (non-parametric alternative)
    u_stat, u_p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                         (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                        (len(group1_data) + len(group2_data) - 2))
    cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    return {
        'groups': {
            'group1': group1_name,
            'group2': group2_name
        },
        't_test': {
            't_statistic': t_stat,
            'p_value': t_p_value,
            'significant': t_p_value < 0.05
        },
        'mann_whitney': {
            'u_statistic': u_stat,
            'p_value': u_p_value,
            'significant': u_p_value < 0.05
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'interpretation': effect_interpretation
        }
    }

def create_results_table_conditional(stats_df, test_results):
    """Create combined results table with test results"""
    results_df = stats_df.copy()
    
    if test_results:
        print("\n" + "="*60)
        print("STATISTICAL TEST RESULTS")
        print("="*60)
        
        print(f"\nComparing: {test_results['groups']['group1']} vs {test_results['groups']['group2']}")
        
        print(f"\nT-Test Results:")
        print(f"  t-statistic: {test_results['t_test']['t_statistic']:.4f}")
        print(f"  p-value: {test_results['t_test']['p_value']:.6f}")
        print(f"  Significant: {test_results['t_test']['significant']}")
        
        print(f"\nMann-Whitney U Test Results:")
        print(f"  U-statistic: {test_results['mann_whitney']['u_statistic']:.4f}")
        print(f"  p-value: {test_results['mann_whitney']['p_value']:.6f}")
        print(f"  Significant: {test_results['mann_whitney']['significant']}")
        
        print(f"\nEffect Size (Cohen's d):")
        print(f"  Cohen's d: {test_results['effect_size']['cohens_d']:.4f}")
        print(f"  Interpretation: {test_results['effect_size']['interpretation']}")
        
        print("\n" + "="*60)
    
    return results_df

def plot_comparison_conditional(stats_df, category_data, test_results, output_dir):
    """Create comprehensive plots for conditional comparison"""
    categories = list(category_data.keys())
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Box Plot
    plot_data = [category_data[cat] for cat in categories]
    box_plot = ax1.boxplot(plot_data, labels=categories, patch_artist=True)
    
    # Customize box plot colors
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors[:len(categories)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Loss Distribution Comparison\n(Box Plot)', fontweight='bold')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add mean values as text
    for i, category in enumerate(categories):
        mean_val = stats_df.loc[category, 'Mean']
        ax1.text(i+1, mean_val, f'μ={mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 2. Violin Plot
    violin_parts = ax2.violinplot(plot_data, positions=range(1, len(categories)+1), 
                                 showmeans=True, showmedians=True)
    
    # Customize violin plot colors
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    
    ax2.set_title('Loss Distribution Comparison\n(Violin Plot)', fontweight='bold')
    ax2.set_ylabel('Loss Value')
    ax2.set_xticks(range(1, len(categories)+1))
    ax2.set_xticklabels(categories, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram Comparison
    for i, (category, data) in enumerate(category_data.items()):
        ax3.hist(data, bins=30, alpha=0.6, label=category, 
                color=colors[i % len(colors)], density=True)
    
    ax3.set_title('Loss Distribution Histograms', fontweight='bold')
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical Summary Bar Chart
    metrics = ['Mean', 'Median', 'Std Dev']
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, category in enumerate(categories):
        values = [
            stats_df.loc[category, 'Mean'],
            stats_df.loc[category, 'Median'],
            stats_df.loc[category, 'Std Dev']
        ]
        ax4.bar(x + i*width, values, width, label=category, 
               color=colors[i % len(colors)], alpha=0.7)
    
    ax4.set_title('Statistical Metrics Comparison', fontweight='bold')
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistical test results as text
    if test_results:
        test_text = f"T-test p-value: {test_results['t_test']['p_value']:.4f}\n"
        test_text += f"Mann-Whitney p-value: {test_results['mann_whitney']['p_value']:.4f}\n"
        test_text += f"Cohen's d: {test_results['effect_size']['cohens_d']:.3f} ({test_results['effect_size']['interpretation']})"
        
        fig.text(0.02, 0.02, test_text, fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for test results
    
    # Save the comprehensive plot
    plot_file = os.path.join(output_dir, 'conditional_comparison_comprehensive.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive plot saved to: {plot_file}")
    
    # Create a separate detailed box plot
    plt.figure(figsize=(10, 8))
    
    box_plot = plt.boxplot(plot_data, labels=categories, patch_artist=True, 
                          notch=True, showmeans=True)
    
    # Customize colors
    for patch, color in zip(box_plot['boxes'], colors[:len(categories)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Conditional Generation Loss Comparison\n(Detailed Box Plot)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add detailed statistics
    for i, category in enumerate(categories):
        stats_text = f"n={stats_df.loc[category, 'Count']}\n"
        stats_text += f"μ={stats_df.loc[category, 'Mean']:.3f}\n"
        stats_text += f"σ={stats_df.loc[category, 'Std Dev']:.3f}"
        
        plt.text(i+1, stats_df.loc[category, 'Max'], stats_text, 
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the detailed box plot
    detailed_plot_file = os.path.join(output_dir, 'conditional_comparison_detailed.png')
    plt.savefig(detailed_plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Detailed box plot saved to: {detailed_plot_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare conditional generation loss between original and generated data')
    parser.add_argument('-i', '--input', nargs='+', 
                        default=[
                            'loss_cal_asap_100.txt',
                            'loss_gen_asap_conditional_noise_5tk_topk250.txt'
                        ],
                        help='Input file paths containing conditional loss data')
    parser.add_argument('-o', '--output', default="Plot/Conditional_Noise_Comparison",
                        help='Output directory for charts and results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the data
    print("Loading conditional comparison data...")
    data = load_and_process_conditional(args.input)
    
    if not data:
        print("No data found in the input files.")
        exit(1)
    
    print(f"Found data for datasets: {list(data.keys())}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_statistics_conditional(data)
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    test_results = perform_statistical_tests_conditional(data)
    
    # Create results table
    print("\nGenerating statistical results summary...")
    results_df = create_results_table_conditional(stats_df, test_results)
    
    # Display results
    print("\nStatistical Results Summary:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(results_df.to_string())
    
    # Save results to CSV
    results_file = os.path.join(output_dir, 'conditional_comparison_statistical_results.csv')
    results_df.to_csv(results_file)
    print(f"\nStatistical results saved to: {results_file}")
    
    # Save test results to text file
    if test_results:
        test_results_file = os.path.join(output_dir, 'statistical_test_results.txt')
        with open(test_results_file, 'w') as f:
            f.write("CONDITIONAL GENERATION COMPARISON - STATISTICAL TEST RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Comparison: {test_results['groups']['group1']} vs {test_results['groups']['group2']}\n\n")
            
            f.write("T-Test Results:\n")
            f.write(f"  t-statistic: {test_results['t_test']['t_statistic']:.6f}\n")
            f.write(f"  p-value: {test_results['t_test']['p_value']:.6f}\n")
            f.write(f"  Significant (α=0.05): {test_results['t_test']['significant']}\n\n")
            
            f.write("Mann-Whitney U Test Results:\n")
            f.write(f"  U-statistic: {test_results['mann_whitney']['u_statistic']:.6f}\n")
            f.write(f"  p-value: {test_results['mann_whitney']['p_value']:.6f}\n")
            f.write(f"  Significant (α=0.05): {test_results['mann_whitney']['significant']}\n\n")
            
            f.write("Effect Size:\n")
            f.write(f"  Cohen's d: {test_results['effect_size']['cohens_d']:.6f}\n")
            f.write(f"  Interpretation: {test_results['effect_size']['interpretation']}\n")
        
        print(f"Test results saved to: {test_results_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_comparison_conditional(stats_df, data, test_results, output_dir)
    print(f"\nAnalysis complete. Plots and statistical results saved in {output_dir}")