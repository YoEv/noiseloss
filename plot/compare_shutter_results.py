import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
ori_results_path = "/home/evev/asap-dataset/Shutter_emd_kld_results/audio_similarity_results.csv"
modified_results_path = "/home/evev/asap-dataset/Shutter_30p_emd_kld_results/audio_similarity_results.csv"

# Read CSV files
ori_df = pd.read_csv(ori_results_path)
modified_df = pd.read_csv(modified_results_path)

# Add group column to original data
ori_df['group'] = 'ori'

# Extract metrics for comparison
metrics = ['mfcc_emd', 'mel_spec_emd', 'stft_emd', 'mfcc_kld', 'mel_spec_kld', 'stft_kld']

# Create directory for plots if it doesn't exist
plots_dir = "Shutter_ori_comparison_plots"
os.makedirs(plots_dir, exist_ok=True)

# Get the four groups from modified data
group_names = modified_df['group'].unique()
print(f"Found groups in modified data: {', '.join(group_names)}")

# Create ANOVA results DataFrame
anova_results = pd.DataFrame(columns=['Metric', 'Group', 'F-statistic', 'p-value', 'Significant'])

# For each metric, compare ori with each of the four groups
print("\nPerforming ANOVA tests comparing original data with each group:\n")
print("{:<15} {:<15} {:<15} {:<15} {:<10}".format("Metric", "Group", "F-statistic", "p-value", "Significant"))
print("-" * 70)

for metric in metrics:
    # Get original values
    ori_values = ori_df[metric].values
    
    # Create a figure for box plots of this metric across all groups
    plt.figure(figsize=(12, 8))
    
    # Add original data to the plot data
    plot_data = ori_df[['group', metric]].copy()
    
    # For each group in the modified data
    for group in group_names:
        # Get values for this group
        group_values = modified_df[modified_df['group'] == group][metric].values
        
        # Perform one-way ANOVA test between ori and this group
        f_stat, p_val = stats.f_oneway(ori_values, group_values)
        
        # Determine significance
        significant = "Yes" if p_val < 0.05 else "No"
        
        # Print results
        print("{:<15} {:<15} {:<15.6f} {:<15.6f} {:<10}".format(
            metric, group, f_stat, p_val, significant))
        
        # Add to results DataFrame
        anova_results = pd.concat([anova_results, pd.DataFrame({
            'Metric': [metric],
            'Group': [group],
            'F-statistic': [f_stat],
            'p-value': [p_val],
            'Significant': [significant]
        })], ignore_index=True)
        
        # Add this group's data to the plot data
        group_df = modified_df[modified_df['group'] == group][['group', metric]].copy()
        plot_data = pd.concat([plot_data, group_df])
    
    # Create box plot for this metric across all groups
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='group', y=metric, data=plot_data)
    plt.title(f'Box Plot of {metric} by Group')
    plt.xlabel('Group')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    
    # Add significance annotations if applicable
    for i, group in enumerate(group_names, 1):
        p_val = anova_results[(anova_results['Metric'] == metric) & 
                             (anova_results['Group'] == group)]['p-value'].values[0]
        significant = p_val < 0.05
        
        if significant:
            plt.annotate(f"p={p_val:.6f} *", 
                         xy=(i, plot_data[plot_data['group'] == group][metric].max()), 
                         ha='center', fontsize=10)
        else:
            plt.annotate(f"p={p_val:.6f} (n.s.)", 
                         xy=(i, plot_data[plot_data['group'] == group][metric].max()), 
                         ha='center', fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{metric}_boxplot.png"))
    plt.close()

# Perform post-hoc tests for metrics with significant differences
significant_pairs = anova_results[anova_results['Significant'] == "Yes"]

if not significant_pairs.empty:
    print("\n\nPost-hoc test results for metrics with significant differences:\n")
    
    for _, row in significant_pairs.iterrows():
        metric = row['Metric']
        group = row['Group']
        
        print(f"Metric: {metric}, Group: {group}")
        ori_values = ori_df[metric].values
        group_values = modified_df[modified_df['group'] == group][metric].values
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(ori_values, group_values, equal_var=False)
        print(f"  t-statistic: {t_stat:.6f}, p-value: {p_val:.6f}")
        print(f"  ori group mean: {np.mean(ori_values):.6f}, std dev: {np.std(ori_values):.6f}")
        print(f"  {group} group mean: {np.mean(group_values):.6f}, std dev: {np.std(group_values):.6f}")
        print()
        
        # Create a more detailed box plot for significant metrics
        plt.figure(figsize=(10, 8))
        
        # Create a DataFrame with just these two groups for comparison
        compare_df = pd.concat([
            ori_df[['group', metric]],
            modified_df[modified_df['group'] == group][['group', metric]]
        ])
        
        # Add individual data points with jitter
        ax = sns.boxplot(x='group', y=metric, data=compare_df)
        sns.stripplot(x='group', y=metric, data=compare_df, size=4, color='black', alpha=0.3)
        
        # Add mean markers
        plt.plot([0], [np.mean(ori_values)], marker='o', markersize=10, color='red', label='Mean')
        plt.plot([1], [np.mean(group_values)], marker='o', markersize=10, color='red')
        
        # Add significance bar
        x1, x2 = 0, 1
        y = max(ori_values.max(), group_values.max()) * 1.1
        plt.plot([x1, x2], [y, y], 'k-')
        plt.text((x1+x2)/2, y*1.05, f"p={p_val:.6f} *", ha='center', va='bottom', fontsize=12)
        
        plt.title(f'Detailed Box Plot of {metric}: ori vs {group} (Significant Difference)')
        plt.xlabel('Group')
        plt.ylabel(metric)
        plt.legend()
        
        # Save the detailed plot
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{metric}_{group}_detailed_boxplot.png"))
        plt.close()
else:
    print("\n\nNo metrics with significant differences were found.")

print(f"\nBox plots have been saved to the '{plots_dir}' directory.")