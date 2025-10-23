import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re

def extract_base_filename(filename):
    """Extract base filename (ID and name part)"""
    # Extract the part before _short30_preview_cut14s
    pattern = r'(.+?)_short30_preview_cut14s'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return filename

def extract_noise_type(filename):
    """Extract noise type from filename"""
    if '_noise_end_white' in filename:
        return 'white'
    elif '_noise_end_brown' in filename:
        return 'brown'
    return None

def load_and_process_data(ori_file, noise_file):
    """Load and process data from original and noise files"""
    # Initialize data structure
    data = {
        'original': [],
        'white_noise': [],
        'brown_noise': []
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
    
    # Load noise data
    with open(noise_file, 'r') as f:
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
                    noise_type = extract_noise_type(filename)
                    if noise_type == 'white':
                        data['white_noise'].append(loss)
                    elif noise_type == 'brown':
                        data['brown_noise'].append(loss)
            except ValueError:
                continue
    
    # Print debug information
    print(f"Original files loaded: {len(data['original'])}")
    print(f"White noise files loaded: {len(data['white_noise'])}")
    print(f"Brown noise files loaded: {len(data['brown_noise'])}")
    
    return data

def calculate_statistics(data):
    """Calculate statistics for each category"""
    stats_data = []
    
    # Add statistics for each category
    for category, values_list in data.items():
        if values_list:  # Only include if there's data
            values = np.array(values_list)
            stats_data.append({
                'Category': category,
                'Mean': np.mean(values),
                'Variance': np.var(values, ddof=1),
                'Std Dev': np.std(values, ddof=1),
                'Count': len(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Median': np.median(values)
            })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('Category')

def perform_anova_test(data):
    """Perform ANOVA test comparing original, white noise, and brown noise"""
    # Prepare data for ANOVA
    category_data = {}
    categories = []
    
    for category, values in data.items():
        if values:  # Only include categories with data
            category_data[category] = values
            categories.append(category)
    
    # Check if we have at least two categories with data
    if len(categories) < 2:
        print("Warning: Not enough categories with data. Need at least 2 categories for statistical tests.")
        return {'ANOVA': None, 'PostHoc': None}
    
    # Perform one-way ANOVA
    category_values = [category_data[cat] for cat in categories]
    
    try:
        f_stat, p_value = stats.f_oneway(*category_values)
        
        # Post-hoc tests (pairwise t-tests) if ANOVA is significant
        posthoc_results = None
        if p_value < 0.05 and len(categories) > 2:
            # Perform pairwise t-tests between original and each noise type
            original_data = category_data.get('original', [])
            posthoc_categories = []
            posthoc_pvalues = []
            posthoc_significant = []
            
            for cat in categories:
                if cat != 'original' and original_data:
                    _, pval = stats.ttest_ind(original_data, category_data[cat])
                    posthoc_categories.append(f'original_vs_{cat}')
                    posthoc_pvalues.append(pval)
                    posthoc_significant.append(pval < 0.05)
            
            # Also compare white vs brown noise if both exist
            if 'white_noise' in category_data and 'brown_noise' in category_data:
                _, pval = stats.ttest_ind(category_data['white_noise'], category_data['brown_noise'])
                posthoc_categories.append('white_vs_brown_noise')
                posthoc_pvalues.append(pval)
                posthoc_significant.append(pval < 0.05)
            
            posthoc_results = {
                'comparison': posthoc_categories,
                'pvalue': posthoc_pvalues,
                'significant': posthoc_significant
            }
        
        return {
            'ANOVA': {
                'F-statistic': f_stat,
                'p-value': p_value,
                'significant': p_value < 0.05,
                'categories': categories
            },
            'PostHoc': posthoc_results
        }
    except Exception as e:
        print(f"Error in statistical tests: {str(e)}")
        return {'ANOVA': None, 'PostHoc': None}

def create_results_table(stats_df, test_results):
    """Create a combined results table with statistics and test results"""
    # Start with statistics
    results_df = stats_df.copy().reset_index()
    
    # Add ANOVA results as summary information
    if test_results and test_results['ANOVA']:
        anova_info = {
            'Category': 'ANOVA_Summary',
            'Mean': f"F={test_results['ANOVA']['F-statistic']:.4f}",
            'Variance': f"p={test_results['ANOVA']['p-value']:.4f}",
            'Std Dev': f"Significant: {test_results['ANOVA']['significant']}",
            'Count': len(test_results['ANOVA']['categories']),
            'Min': '',
            'Max': '',
            'Median': ''
        }
        
        # Insert ANOVA summary at the beginning
        anova_df = pd.DataFrame([anova_info])
        results_df = pd.concat([anova_df, results_df], ignore_index=True)
    
    # Add post-hoc results if available
    if test_results and test_results['PostHoc']:
        posthoc = test_results['PostHoc']
        for i, comparison in enumerate(posthoc['comparison']):
            posthoc_info = {
                'Category': f'PostHoc_{comparison}',
                'Mean': f"p={posthoc['pvalue'][i]:.4f}",
                'Variance': f"Significant: {posthoc['significant'][i]}",
                'Std Dev': '',
                'Count': '',
                'Min': '',
                'Max': '',
                'Median': ''
            }
            posthoc_df = pd.DataFrame([posthoc_info])
            results_df = pd.concat([results_df, posthoc_df], ignore_index=True)
    
    return results_df

def create_boxplot(data, output_dir):
    """Create box plot comparing original, white noise, and brown noise"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plot
    plot_data = []
    labels = []
    colors = ['red', 'lightblue', 'brown']
    
    # Add data in order: original, white noise, brown noise
    for category in ['original', 'white_noise', 'brown_noise']:
        if category in data and data[category]:
            plot_data.append(data[category])
            if category == 'original':
                labels.append('Original')
            elif category == 'white_noise':
                labels.append('White Noise')
            elif category == 'brown_noise':
                labels.append('Brown Noise')
    
    # Create box plot
    box = plt.boxplot(plot_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    for i, (patch, color) in enumerate(zip(box['boxes'], colors[:len(plot_data)])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Loss Distribution - Original vs Noise Types')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'noise_end_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up file paths
    ori_file = '/home/evev/asap-dataset/loss_Shutter_ori_medium.txt'
    noise_file = '/home/evev/asap-dataset/loss_Shutter_noise_end_medium.txt'
    output_dir = '/home/evev/asap-dataset/Loss_Cal_Plot/Plot_Shutter_noise_end_medium'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    data = load_and_process_data(ori_file, noise_file)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_df = calculate_statistics(data)
    
    # Perform ANOVA test
    print("Performing ANOVA test...")
    test_results = perform_anova_test(data)
    
    # Create combined results table
    print("Creating results table...")
    combined_results = create_results_table(stats_df, test_results)
    
    # Save results
    stats_df.to_csv(os.path.join(output_dir, 'noise_end_statistics.csv'))
    combined_results.to_csv(os.path.join(output_dir, 'noise_end_anova_results.csv'), index=False)
    
    # Create box plot
    print("Creating box plot...")
    create_boxplot(data, output_dir)
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Results saved in: {output_dir}")
    
    if test_results and test_results['ANOVA']:
        anova = test_results['ANOVA']
        print(f"\nANOVA Results:")
        print(f"F-statistic: {anova['F-statistic']:.4f}")
        print(f"p-value: {anova['p-value']:.4f}")
        print(f"Significant difference: {anova['significant']}")
        
        if test_results['PostHoc']:
            print(f"\nPost-hoc comparisons:")
            posthoc = test_results['PostHoc']
            for i, comparison in enumerate(posthoc['comparison']):
                print(f"{comparison}: p={posthoc['pvalue'][i]:.4f}, significant={posthoc['significant'][i]}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()