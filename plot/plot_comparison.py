import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re

# Define function to extract all categories
def get_all_categories(file_paths):
    """Extract all unique categories from data files"""
    categories = set()
    original_category = '_ori'  # Define original category name
    categories.add(original_category)  # Ensure original category always exists

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                filename = line.rsplit(':', 1)[0].strip()

                # Skip original files (with _ori suffix)
                if '_ori' in filename:
                    continue

                # 使用正则表达式提取类别
                # 尝试匹配常见的修改模式
                patterns = [
                    r'_(pink|brown)_(\d+)db_(\d+)p',  # 修改为匹配您的文件名格式
                    r'_(low|mid|high)_(\d+)p',            # 频率范围和百分比
                    r'_(fq_dele_random)_(\d+)p',          # 随机频率删除
                    r'(noise)_(\d+)',                     # 噪声和百分比（不需要前导下划线）
                    r'(randomnoise)_(\d+)',               # 随机噪声和百分比（不需要前导下划线）
                    # 添加频率删除模式
                    r'_fq(\d+-\d+)_(\d+)p',              # 频率删除模式 fq40-800_30p
                    #r'_(velocity|pitch|structure|rhythm)', # 其他修改类型
                    # 添加新的pitch子组模式
                    r'_pitch1_1_step1_(semi\d+)',         # semitone pitch change
                    r'_pitch1_2_step1_(oct\d+)',          # octave pitch change
                    r'_pitch1_3_step1_(dia\d+)',          # diatonic pitch change
                    r'_rhythm1_4_step1_(r\d+)',           # rhythm change
                    r'_tempo1_1_step1_(nt\d+)',           # tempo/structure change
                    r'_velocity1_1_step1_(nt\d+)',        # velocity change
                    r'(reordered)_(\d+)p'                 # reordered files with percentage
                ]

                for pattern in patterns:
                    match = re.search(pattern, filename)
                    if match:
                        if pattern == r'_(pink|brown)_(\d+)db_(\d+)p' and len(match.groups()) == 3:
                            # 特殊处理您的文件名格式，组合噪声类型、分贝和百分比
                            category = f"_{match.group(1)}_{match.group(2)}db_{match.group(3)}p"
                        elif pattern == r'_fq(\d+-\d+)_(\d+)p' and len(match.groups()) == 2:
                            # 处理频率删除模式
                            category = f"_fq{match.group(1)}_{match.group(2)}p"
                        elif pattern == r'_pitch1_1_step1_(semi\d+)' and len(match.groups()) == 1:
                            # 处理semitone pitch change
                            category = f"_semi_{match.group(1)}"
                        elif pattern == r'_pitch1_2_step1_(oct\d+)' and len(match.groups()) == 1:
                            # 处理octave pitch change
                            category = f"_oct_{match.group(1)}"
                        elif pattern == r'_pitch1_3_step1_(dia\d+)' and len(match.groups()) == 1:
                            # 处理diatonic pitch change
                            category = f"_dia_{match.group(1)}"
                        elif pattern == r'_rhythm1_4_step1_(r\d+)' and len(match.groups()) == 1:
                            # 处理rhythm change
                            category = f"_r_{match.group(1)}"
                        elif pattern == r'_tempo1_1_step1_(nt\d+)' and len(match.groups()) == 1:
                            # 处理tempo/structure change
                            category = f"_t_{match.group(1)}"
                        elif pattern == r'_velocity1_1_step1_(nt\d+)' and len(match.groups()) == 1:
                            # 处理velocity change
                            category = f"_v_{match.group(1)}"
                        elif pattern == r'(reordered)_(\d+)p' and len(match.groups()) == 2:
                            # 处理reordered files with percentage
                            category = f"_reordered_{match.group(2)}p"
                        elif len(match.groups()) == 2:
                            category = f"_{match.group(1)}_{match.group(2)}"
                        else:
                            category = f"_{match.group(1)}"
                        categories.add(category)
                        break

    # Put original category first, sort other categories alphabetically
    # Check if we're dealing with frequency deletion data (fq_dele)
    fq_dele_pattern = re.compile(r'_fq(\d+-\d+)_(\d+)p$')
    reordered_pattern = re.compile(r'_reordered_(\d+)p$')
    
    # Filter categories that match the pattern
    fq_low_categories = []  # For fq20-1000 range
    fq_high_categories = []  # For fq2000-20000 range
    reordered_categories = []  # For reordered categories
    other_categories = []
    
    for cat in categories:
        if cat != original_category:
            fq_match = fq_dele_pattern.search(cat)
            reordered_match = reordered_pattern.search(cat)
            
            if fq_match:
                # Extract the frequency range and percentage value for sorting
                freq_range = fq_match.group(1)
                percentage = int(fq_match.group(2))
                
                # Group by frequency range
                if freq_range == "20-1000" or freq_range == "40-800":
                    fq_low_categories.append((cat, percentage))
                elif freq_range == "2000-20000":
                    fq_high_categories.append((cat, percentage))
                else:
                    # Any other frequency range goes to other categories
                    other_categories.append(cat)
            elif reordered_match:
                # Extract the percentage value for sorting reordered categories
                percentage = int(reordered_match.group(1))
                reordered_categories.append((cat, percentage))
            else:
                other_categories.append(cat)
    
    # Sort each group by percentage value
    fq_low_categories.sort(key=lambda x: x[1])  # Sort by percentage
    fq_high_categories.sort(key=lambda x: x[1])  # Sort by percentage
    reordered_categories.sort(key=lambda x: x[1])  # Sort reordered by percentage
    
    sorted_fq_low = [cat for cat, _ in fq_low_categories]
    sorted_fq_high = [cat for cat, _ in fq_high_categories]
    sorted_reordered = [cat for cat, _ in reordered_categories]
    
    # Sort other categories alphabetically
    sorted_other = sorted(other_categories)
    
    # Combine all categories with original first, then low frequency group, then high frequency group, then reordered group
    sorted_categories = [original_category] + sorted_fq_low + sorted_fq_high + sorted_reordered + sorted_other
    return sorted_categories

def load_and_process(file_paths):
    """Load and process data for all categories"""
    categories = get_all_categories(file_paths)
    category_data = {category: [] for category in categories}

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    filename, loss = line.rsplit(':', 1)
                    filename = filename.strip()
                    try:
                        loss = float(loss.strip())
                    except ValueError:
                        continue

                    # Check if it's an original file
                    if '_ori' in filename:
                        category_data['_ori'].append(loss)
                        continue
                        
                    # Check if it's a reordered file
                    if filename.startswith('reordered_'):
                        category_data['_reordered'].append(loss)
                        continue

                    # Match appropriate category
                    matched = False
                    for category in categories:
                        if category != '_ori' and category != '_reordered' and category in filename:
                            category_data[category].append(loss)
                            matched = True
                            break

                    # If no category matched, try regex patterns
                    if not matched:
                        patterns = [
                            r'_(pink|brown)_(\d+)db_(\d+)p',
                            r'_(low|mid|high)_(\d+)p',
                            r'_(fq_dele_random)_(\d+)p',
                            r'(noise)_(\d+)',
                            r'(randomnoise)_(\d+)',
                            r'_fq(\d+-\d+)_(\d+)p',
                            r'_pitch1_1_step1_(semi\d+)',
                            r'_pitch1_2_step1_(oct\d+)',
                            r'_pitch1_3_step1_(dia\d+)',
                            r'_rhythm1_4_step1_(r\d+)',
                            r'_tempo1_1_step1_(nt\d+)',
                            r'_velocity1_1_step1_(nt\d+)',
                            r'(reordered)_(\d+)p'
                        ]

                        for pattern in patterns:
                            match = re.search(pattern, filename)
                            if match:
                                if pattern == r'_(pink|brown)_(\d+)db_(\d+)p' and len(match.groups()) == 3:
                                    matched_category = f"_{match.group(1)}_{match.group(2)}db_{match.group(3)}p"
                                elif pattern == r'_fq(\d+-\d+)_(\d+)p' and len(match.groups()) == 2:
                                    matched_category = f"_fq{match.group(1)}_{match.group(2)}p"
                                elif pattern == r'_pitch1_1_step1_(semi\d+)' and len(match.groups()) == 1:
                                    matched_category = f"_semi_{match.group(1)}"
                                elif pattern == r'_pitch1_2_step1_(oct\d+)' and len(match.groups()) == 1:
                                    matched_category = f"_oct_{match.group(1)}"
                                elif pattern == r'_pitch1_3_step1_(dia\d+)' and len(match.groups()) == 1:
                                    matched_category = f"_dia_{match.group(1)}"
                                elif pattern == r'_rhythm1_4_step1_(r\d+)' and len(match.groups()) == 1:
                                    matched_category = f"_r_{match.group(1)}"
                                elif pattern == r'_tempo1_1_step1_(nt\d+)' and len(match.groups()) == 1:
                                    matched_category = f"_t_{match.group(1)}"
                                elif pattern == r'_velocity1_1_step1_(nt\d+)' and len(match.groups()) == 1:
                                    matched_category = f"_v_{match.group(1)}"
                                elif pattern == r'^(reordered)_' and len(match.groups()) == 1:
                                    # 处理reordered files
                                    matched_category = f"_reordered"
                                elif len(match.groups()) == 2:
                                    matched_category = f"_{match.group(1)}_{match.group(2)}"
                                else:
                                    matched_category = f"_{match.group(1)}"

                                if matched_category in categories:
                                    category_data[matched_category].append(loss)
                                    matched = True
                                    break

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    # Remove empty categories
    return {k: v for k, v in category_data.items() if v}

def calculate_statistics(category_data):
    """Calculate statistics for each category"""
    stats_data = []
    for category in category_data.keys():
        values = np.array(category_data[category])
        stats_data.append({
            'Category': category,
            'Mean': np.mean(values),
            'Variance': np.var(values, ddof=1),
            'Std Dev': np.std(values, ddof=1),
            'Count': len(values),
            'Type': 'Original' if category == '_ori' else 'Modified'
        })

    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('Category')

def perform_statistical_tests(category_data):
    """Perform ANOVA and post-hoc tests"""
    # Prepare data for ANOVA
    all_data = []
    all_labels = []
    
    for category, values in category_data.items():
        all_data.extend(values)
        all_labels.extend([category] * len(values))
    
    # Convert to arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # Perform one-way ANOVA
    categories = list(category_data.keys())
    category_values = [category_data[cat] for cat in categories]
    
    try:
        f_stat, p_value = stats.f_oneway(*category_values)
        
        # Post-hoc tests (Tukey's HSD) if ANOVA is significant
        posthoc_results = None
        if p_value < 0.05 and len(categories) > 2:
            from scipy.stats import tukey_hsd
            try:
                posthoc_results = tukey_hsd(*category_values)
                
                # Create post-hoc comparison with original
                if '_ori' in category_data:
                    original_data = category_data['_ori']
                    posthoc_categories = []
                    posthoc_pvalues = []
                    posthoc_significant = []
                    
                    for i, cat in enumerate(categories):
                        if cat != '_ori':
                            # Find the comparison with original
                            ori_idx = categories.index('_ori')
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
                if '_ori' in category_data:
                    original_data = category_data['_ori']
                    posthoc_categories = []
                    posthoc_pvalues = []
                    posthoc_significant = []
                    
                    for cat in categories:
                        if cat != '_ori':
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
        print(f"Error in statistical tests: {str(e)}")
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
        'Mean': [anova_info],
        'Variance': [''],
        'Std Dev': [''],
        'Count': [''],
        'Type': ['Statistical Test'],
        'Significant_vs_Original': [test_results['ANOVA']['significant'] if test_results and test_results['ANOVA'] else False]
    })
    
    results_df = pd.concat([anova_row, results_df], ignore_index=True)
    
    return results_df

def plot_comparison(stats_df, category_data, output_dir):
    """Generate box plots grouped by frequency range"""
    categories = list(category_data.keys())
    
    # Identify frequency range groups
    fq_low_pattern = re.compile(r'_fq(20-1000|40-800)_\d+p$')
    fq_high_pattern = re.compile(r'_fq2000-20000_\d+p$')
    
    # Create category groups
    ori_categories = ['_ori'] if '_ori' in categories else []
    fq_low_categories = [cat for cat in categories if fq_low_pattern.search(cat)]
    fq_high_categories = [cat for cat in categories if fq_high_pattern.search(cat)]
    other_categories = [cat for cat in categories if cat not in ori_categories + fq_low_categories + fq_high_categories]
    
    # Create a combined plot with all categories
    plt.figure(figsize=(15, 8))
    
    # Set up colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    # Box plot for all categories
    data_for_boxplot = [category_data[cat] for cat in categories]
    box = plt.boxplot(data_for_boxplot, labels=categories, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Highlight original data
    if '_ori' in categories:
        ori_idx = categories.index('_ori')
        box['boxes'][ori_idx].set_facecolor('red')
        box['boxes'][ori_idx].set_alpha(1.0)

    plt.title('Loss Distribution by Category')
    plt.ylabel('Loss Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_categories_boxplot.png'), dpi=300)
    
    # Create separate plots for each frequency group if they exist
    if fq_low_categories:
        plt.figure(figsize=(12, 6))
        group_categories = ori_categories + fq_low_categories
        data_for_boxplot = [category_data[cat] for cat in group_categories]
        box = plt.boxplot(data_for_boxplot, labels=group_categories, patch_artist=True)
        
        # Color the boxes
        group_colors = plt.cm.Set3(np.linspace(0, 1, len(group_categories)))
        for patch, color in zip(box['boxes'], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Highlight original data
        if '_ori' in group_categories:
            ori_idx = group_categories.index('_ori')
            box['boxes'][ori_idx].set_facecolor('red')
            box['boxes'][ori_idx].set_alpha(1.0)

        plt.title('Loss Distribution - Low Frequency Range (20-1000 Hz)')
        plt.ylabel('Loss Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'low_frequency_boxplot.png'), dpi=300)
    
    if fq_high_categories:
        plt.figure(figsize=(12, 6))
        group_categories = ori_categories + fq_high_categories
        data_for_boxplot = [category_data[cat] for cat in group_categories]
        box = plt.boxplot(data_for_boxplot, labels=group_categories, patch_artist=True)
        
        # Color the boxes
        group_colors = plt.cm.Set3(np.linspace(0, 1, len(group_categories)))
        for patch, color in zip(box['boxes'], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Highlight original data
        if '_ori' in group_categories:
            ori_idx = group_categories.index('_ori')
            box['boxes'][ori_idx].set_facecolor('red')
            box['boxes'][ori_idx].set_alpha(1.0)

        plt.title('Loss Distribution - High Frequency Range (2000-20000 Hz)')
        plt.ylabel('Loss Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'high_frequency_boxplot.png'), dpi=300)

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process MIDI loss data and generate statistical analysis')
    parser.add_argument('-i', '--input', nargs='+', 
                        default=['loss_Shutter_medium.txt','loss_Shutter_fq_dele_medium.txt'],
                        help='Input file paths containing loss data (default: shutter_loss.txt and loss_Shutter_reordered_token_medium.txt)')
    parser.add_argument('-o', '--output', default="Loss_Cal_Plot/Plot_Shutter_fq_dele_Medium",
                        help='Output directory for charts and results')

    # Parse arguments
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data for all categories
    print("Loading data...")
    data = load_and_process(args.input)

    # 2. Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_statistics(data)

    # 3. Statistical tests (ANOVA)
    print("\nPerforming ANOVA and post-hoc tests...")
    test_results = perform_statistical_tests(data)

    # 4. Create combined results table
    print("\nGenerating statistical results summary...")
    combined_results = create_combined_results_table(stats_df, test_results)

    # 5. Output combined results table
    print("\nStatistical Results Summary:")
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 150)  # Set display width
    pd.set_option('display.float_format', '{:.8f}'.format)  # Set float format
    print(combined_results.to_string(index=False))

    # 6. Save results to CSV file
    results_file = os.path.join(output_dir, 'statistical_results.csv')
    combined_results.to_csv(results_file, index=False)
    print(f"\nStatistical results saved to: {results_file}")

    # 7. Generate only the box plot
    print("\nGenerating box plot...")
    plot_comparison(stats_df, data, output_dir)
    print("\nAnalysis complete. Box plot and statistical results saved in", output_dir)