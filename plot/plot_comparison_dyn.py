import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re

def extract_dynamic_category(filename):
    """Extract dynamic range and percentage from filename"""
    # Pattern to match dynamic range and percentage
    pattern = r'_dyn(\d+-\d+)_(\d+)p_rndvol'
    match = re.search(pattern, filename)
    if match:
        dyn_range = match.group(1)
        percentage = int(match.group(2))
        return dyn_range, percentage
    return None, None

def extract_base_filename(filename):
    """Extract base filename (ID and name part)"""
    # Extract the part before _short30_preview_cut14s
    pattern = r'(.+?)_short30_preview_cut14s'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return filename

def load_and_process_data(ori_file, dyn_file):
    """Load and process data from original and dynamic range files"""
    # Initialize data structure
    data = {
        'original': [],
        'dyn1-5': {'5p': [], '10p': [], '30p': [], '50p': [], '80p': []},
        'dyn10-50': {'5p': [], '10p': [], '30p': [], '50p': [], '80p': []},
        'dyn100-300': {'5p': [], '10p': [], '30p': [], '50p': [], '80p': []},
        'dyn300-500': {'5p': [], '10p': [], '30p': [], '50p': [], '80p': []}
    }
    
    # Dictionary to store original filenames and their loss values
    original_files = {}
    
    # Load original data
    with open(ori_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            filename, loss = line.rsplit(':', 1)
            filename = filename.strip()
            try:
                loss = float(loss.strip())
                base_filename = extract_base_filename(filename)
                original_files[base_filename] = loss
                data['original'].append(loss)
            except ValueError:
                continue
    
    # Load dynamic range data
    with open(dyn_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            filename, loss = line.rsplit(':', 1)
            filename = filename.strip()
            try:
                loss = float(loss.strip())
                # Extract base filename (ID and name part)
                base_filename = extract_base_filename(filename)
                
                # Only process if we have the corresponding original file
                if base_filename in original_files:
                    dyn_range, percentage = extract_dynamic_category(filename)
                    if dyn_range and percentage:
                        # Add 'dyn' prefix to match data structure keys
                        dyn_key = f"dyn{dyn_range}"
                        percentage_key = f"{percentage}p"
                        if dyn_key in data and percentage_key in data[dyn_key]:
                            data[dyn_key][percentage_key].append(loss)
            except ValueError:
                continue
    
    # Print debug information
    print(f"Original files loaded: {len(data['original'])}")
    for dyn_range in ['dyn1-5', 'dyn10-50', 'dyn100-300', 'dyn300-500']:
        total_dyn_files = sum(len(data[dyn_range][p]) for p in data[dyn_range])
        print(f"{dyn_range} files loaded: {total_dyn_files}")
        for p in data[dyn_range]:
            if data[dyn_range][p]:
                print(f"  {p}: {len(data[dyn_range][p])} files")
    
    return data

def calculate_statistics(data):
    """Calculate statistics for each category and percentage"""
    stats_data = []
    
    # Add original data statistics
    original_values = np.array(data['original'])
    stats_data.append({
        'Category': 'original',
        'Percentage': 'N/A',
        'Mean': np.mean(original_values),
        'Variance': np.var(original_values, ddof=1),
        'Std Dev': np.std(original_values, ddof=1),
        'Count': len(original_values),
        'Type': 'Original'
    })
    
    # Add statistics for each dynamic range and percentage
    for dyn_range in ['dyn1-5', 'dyn10-50', 'dyn100-300', 'dyn300-500']:
        for percentage, values_list in data[dyn_range].items():
            if values_list:  # Only include if there's data
                values = np.array(values_list)
                stats_data.append({
                    'Category': f"{dyn_range}_{percentage}",
                    'Percentage': percentage,
                    'Mean': np.mean(values),
                    'Variance': np.var(values, ddof=1),
                    'Std Dev': np.std(values, ddof=1),
                    'Count': len(values),
                    'Type': 'Modified'
                })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('Category')

def perform_statistical_tests(data, dyn_range):
    """Perform ANOVA and post-hoc tests comparing original to each percentage in the dynamic range"""
    # Prepare data for ANOVA
    category_data = {'original': data['original']}
    
    # Add data for each percentage in the dynamic range
    for percentage in ['5p', '10p', '30p', '50p', '80p']:
        if percentage in data[dyn_range] and data[dyn_range][percentage]:
            category_data[f"{dyn_range}_{percentage}"] = data[dyn_range][percentage]
    
    # Check if we have at least two categories with data
    categories = list(category_data.keys())
    if len(categories) < 2:
        print(f"Warning: Not enough categories with data for {dyn_range}. Need at least 2 categories for statistical tests.")
        return {'ANOVA': None, 'PostHoc': None}
    
    # Perform one-way ANOVA
    category_values = [category_data[cat] for cat in categories]
    
    try:
        f_stat, p_value = stats.f_oneway(*category_values)
        
        # Post-hoc tests (Tukey's HSD) if ANOVA is significant
        posthoc_results = None
        if p_value < 0.05 and len(categories) > 2:
            try:
                # Try to use Tukey's HSD if available
                from scipy.stats import tukey_hsd
                posthoc_results = tukey_hsd(*category_values)
                
                # Create post-hoc comparison with original
                original_data = category_data['original']
                posthoc_categories = []
                posthoc_pvalues = []
                posthoc_significant = []
                
                for i, cat in enumerate(categories):
                    if cat != 'original':
                        # Find the comparison with original
                        ori_idx = categories.index('original')
                        cat_idx = i
                        
                        if hasattr(posthoc_results, 'pvalue'):
                            pval = posthoc_results.pvalue[ori_idx, cat_idx]
                        else:
                            # Fallback to t-test
                            _, pval = stats.ttest_ind(original_data, category_data[cat])
                        
                        posthoc_categories.append(cat)
                        posthoc_pvalues.append(pval)
                        posthoc_significant.append(pval < 0.05)
                
                posthoc_results = {
                    'category': posthoc_categories,
                    'pvalue': posthoc_pvalues,
                    'significant': posthoc_significant
                }
            except ImportError:
                # Fallback to pairwise t-tests if tukey_hsd is not available
                original_data = category_data['original']
                posthoc_categories = []
                posthoc_pvalues = []
                posthoc_significant = []
                
                for cat in categories:
                    if cat != 'original':
                        _, pval = stats.ttest_ind(original_data, category_data[cat])
                        posthoc_categories.append(cat)
                        posthoc_pvalues.append(pval)
                        posthoc_significant.append(pval < 0.05)
                
                posthoc_results = {
                    'category': posthoc_categories,
                    'pvalue': posthoc_pvalues,
                    'significant': posthoc_significant
                }
        
        return {
            'ANOVA': {
                'F-statistic': f_stat,
                'p-value': p_value,
                'significant': p_value < 0.05
            },
            'PostHoc': posthoc_results
        }
    except Exception as e:
        print(f"Error in statistical tests for {dyn_range}: {str(e)}")
        return {'ANOVA': None, 'PostHoc': None}

def create_combined_results_table(stats_df, test_results):
    """Create a combined results table with statistics and test results"""
    # Start with statistics
    results_df = stats_df.copy()
    
    # Add significance column
    results_df['Significant_vs_Original'] = False
    
    if test_results and test_results['PostHoc']:
        posthoc = test_results['PostHoc']
        for i, category in enumerate(posthoc['category']):
            if category in results_df.index:
                results_df.loc[category, 'Significant_vs_Original'] = posthoc['significant'][i]
    
    # Add ANOVA results as a summary row
    if test_results and test_results['ANOVA']:
        anova_info = f"F={test_results['ANOVA']['F-statistic']:.4f}, p={test_results['ANOVA']['p-value']:.4f}"
    else:
        anova_info = "N/A"
    
    # Reset index to make Category a column
    results_df = results_df.reset_index()
    
    # Add ANOVA summary as the first row
    anova_row = pd.DataFrame({
        'Category': ['ANOVA_Summary'],
        'Percentage': [''],
        'Mean': [anova_info],
        'Variance': [''],
        'Std Dev': [''],
        'Count': [''],
        'Type': ['Statistical Test'],
        'Significant_vs_Original': [test_results['ANOVA']['significant'] if test_results and test_results['ANOVA'] else False]
    })
    
    results_df = pd.concat([anova_row, results_df], ignore_index=True)
    
    return results_df

def create_boxplot(data, dyn_range, output_dir):
    """Create box plot comparing original to each percentage in the dynamic range"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data for box plot
    plot_data = [data['original']]
    labels = ['Original']
    
    for percentage in ['5p', '10p', '30p', '50p', '80p']:
        if percentage in data[dyn_range] and data[dyn_range][percentage]:
            plot_data.append(data[dyn_range][percentage])
            labels.append(f"{dyn_range}_{percentage}")
    
    # Create box plot
    box = plt.boxplot(plot_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Highlight original data
    box['boxes'][0].set_facecolor('red')
    box['boxes'][0].set_alpha(1.0)
    
    plt.title(f'Loss Distribution - {dyn_range} vs Original')
    plt.ylabel('Loss Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{dyn_range}_boxplot.png'), dpi=300)
    plt.close()

def main():
    # Set up file paths
    ori_file = '/home/evev/asap-dataset/loss_Shutter_ori_medium.txt'
    dyn_file = '/home/evev/asap-dataset/loss_Shutter_dyn_rnd_medium.txt'
    output_dir = '/home/evev/asap-dataset/Loss_Cal_Plot/Plot_Shutter_dyn_medium'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    data = load_and_process_data(ori_file, dyn_file)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_df = calculate_statistics(data)
    
    # Save overall statistics
    stats_df.to_csv(os.path.join(output_dir, 'overall_statistics.csv'))
    
    # Process each dynamic range group
    for dyn_range in ['dyn1-5', 'dyn10-50', 'dyn100-300', 'dyn300-500']:
        print(f"Processing {dyn_range}...")
        
        # Perform statistical tests (ANOVA)
        test_results = perform_statistical_tests(data, dyn_range)
        
        # Create combined results table
        combined_results = create_combined_results_table(stats_df.loc[[idx for idx in stats_df.index if idx == 'original' or idx.startswith(dyn_range)]], test_results)
        
        # Save test results
        combined_results.to_csv(os.path.join(output_dir, f'{dyn_range}_test_results.csv'), index=False)
        
        # Create box plot
        create_boxplot(data, dyn_range, output_dir)
    
    print(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()