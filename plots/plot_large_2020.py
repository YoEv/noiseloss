import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from scipy import stats
from collections import defaultdict

def extract_detailed_categories_from_files(file_paths):
    """Extract detailed categories from loss data files with subgroup detection"""
    category_data = defaultdict(list)
    
    # Define detailed regex patterns for subgroup detection
    patterns = {
        # Shutter patterns with detailed subgroup
        'shutter_ori': r'\d+_[^_]+_short30_preview_cut14s_ori\.mp3$',
        'shutter_reordered_phrase_5p': r'reordered_5p_phrase\.mp3$',
        'shutter_reordered_phrase_10p': r'reordered_10p_phrase\.mp3$', 
        'shutter_reordered_phrase_30p': r'reordered_30p_phrase\.mp3$',
        'shutter_reordered_phrase_50p': r'reordered_50p_phrase\.mp3$',
        'shutter_reordered_phrase_80p': r'reordered_80p_phrase\.mp3$',
        'shutter_reordered_bar_5p': r'reordered_5p_bar\.mp3$',
        'shutter_reordered_bar_10p': r'reordered_10p_bar\.mp3$',
        'shutter_reordered_bar_30p': r'reordered_30p_bar\.mp3$',
        'shutter_reordered_bar_50p': r'reordered_50p_bar\.mp3$',
        'shutter_reordered_bar_80p': r'reordered_80p_bar\.mp3$',
        'shutter_reordered_token_5p': r'reordered_5p_token\.mp3$',
        'shutter_reordered_token_10p': r'reordered_10p_token\.mp3$',
        'shutter_reordered_token_30p': r'reordered_30p_token\.mp3$',
        'shutter_reordered_token_50p': r'reordered_50p_token\.mp3$',
        'shutter_reordered_token_80p': r'reordered_80p_token\.mp3$',
        'shutter_fq_low_5p': r'fq20-1000_5p\.mp3$',
        'shutter_fq_low_10p': r'fq20-1000_10p\.mp3$',
        'shutter_fq_low_30p': r'fq20-1000_30p\.mp3$',
        'shutter_fq_low_50p': r'fq20-1000_50p\.mp3$',
        'shutter_fq_low_80p': r'fq20-1000_80p\.mp3$',
        'shutter_fq_high_5p': r'fq2000-20000_5p\.mp3$',
        'shutter_fq_high_10p': r'fq2000-20000_10p\.mp3$',
        'shutter_fq_high_30p': r'fq2000-20000_30p\.mp3$',
        'shutter_fq_high_50p': r'fq2000-20000_50p\.mp3$',
        'shutter_fq_high_80p': r'fq2000-20000_80p\.mp3$',
        
        # ASAP patterns with detailed subgroup
        'asap_ori': r'_midi_score_short\.mid_ori_cut15s\.wav$',
        'asap_rhythm_r10': r'_rhythm1_4_step1_r10_cut15s\.wav$',
        'asap_rhythm_r20': r'_rhythm1_4_step1_r20_cut15s\.wav$',
        'asap_rhythm_r30': r'_rhythm1_4_step1_r30_cut15s\.wav$',
        'asap_rhythm_r40': r'_rhythm1_4_step1_r40_cut15s\.wav$',
        'asap_tempo_nt10': r'_tempo1_1_step1_nt10_cut15s\.wav$',
        'asap_tempo_nt30': r'_tempo1_1_step1_nt30_cut15s\.wav$',
        'asap_tempo_nt50': r'_tempo1_1_step1_nt50_cut15s\.wav$',
        'asap_tempo_nt80': r'_tempo1_1_step1_nt80_cut15s\.wav$',
        'asap_velocity_nt10': r'_velocity1_1_step1_nt10_cut15s\.wav$',
        'asap_velocity_nt30': r'_velocity1_1_step1_nt30_cut15s\.wav$',
        'asap_velocity_nt50': r'_velocity1_1_step1_nt50_cut15s\.wav$',
        'asap_velocity_nt80': r'_velocity1_1_step1_nt80_cut15s\.wav$'
    }
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        filename, loss_str = line.split(':', 1)
                        filename = filename.strip()
                        loss = float(loss_str.strip())
                        
                        # Check each pattern
                        matched = False
                        for category, pattern in patterns.items():
                            if re.search(pattern, filename):
                                category_data[category].append(loss)
                                matched = True
                                break
                        
                        if not matched:
                            print(f"未匹配的文件: {filename}")
                            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    # Remove empty categories
    return {k: v for k, v in category_data.items() if v}

def group_categories_for_analysis(category_data):
    """Group categories for statistical analysis"""
    grouped_data = {
        # Shutter groups with subgroup
        'Shutter_Original': category_data.get('shutter_ori', []),
        'Shutter_Phrase_5p': category_data.get('shutter_reordered_phrase_5p', []),
        'Shutter_Phrase_10p': category_data.get('shutter_reordered_phrase_10p', []),
        'Shutter_Phrase_30p': category_data.get('shutter_reordered_phrase_30p', []),
        'Shutter_Phrase_50p': category_data.get('shutter_reordered_phrase_50p', []),
        'Shutter_Phrase_80p': category_data.get('shutter_reordered_phrase_80p', []),
        'Shutter_Bar_5p': category_data.get('shutter_reordered_bar_5p', []),
        'Shutter_Bar_10p': category_data.get('shutter_reordered_bar_10p', []),
        'Shutter_Bar_30p': category_data.get('shutter_reordered_bar_30p', []),
        'Shutter_Bar_50p': category_data.get('shutter_reordered_bar_50p', []),
        'Shutter_Bar_80p': category_data.get('shutter_reordered_bar_80p', []),
        'Shutter_Token_5p': category_data.get('shutter_reordered_token_5p', []),
        'Shutter_Token_10p': category_data.get('shutter_reordered_token_10p', []),
        'Shutter_Token_30p': category_data.get('shutter_reordered_token_30p', []),
        'Shutter_Token_50p': category_data.get('shutter_reordered_token_50p', []),
        'Shutter_Token_80p': category_data.get('shutter_reordered_token_80p', []),
        'Shutter_FQ_Low_5p': category_data.get('shutter_fq_low_5p', []),
        'Shutter_FQ_Low_10p': category_data.get('shutter_fq_low_10p', []),
        'Shutter_FQ_Low_30p': category_data.get('shutter_fq_low_30p', []),
        'Shutter_FQ_Low_50p': category_data.get('shutter_fq_low_50p', []),
        'Shutter_FQ_Low_80p': category_data.get('shutter_fq_low_80p', []),
        'Shutter_FQ_High_5p': category_data.get('shutter_fq_high_5p', []),
        'Shutter_FQ_High_10p': category_data.get('shutter_fq_high_10p', []),
        'Shutter_FQ_High_30p': category_data.get('shutter_fq_high_30p', []),
        'Shutter_FQ_High_50p': category_data.get('shutter_fq_high_50p', []),
        'Shutter_FQ_High_80p': category_data.get('shutter_fq_high_80p', []),
        
        # ASAP groups with subgroup
        'ASAP_Original': category_data.get('asap_ori', []),
        'ASAP_Rhythm_r10': category_data.get('asap_rhythm_r10', []),
        'ASAP_Rhythm_r20': category_data.get('asap_rhythm_r20', []),
        'ASAP_Rhythm_r30': category_data.get('asap_rhythm_r30', []),
        'ASAP_Rhythm_r40': category_data.get('asap_rhythm_r40', []),
        'ASAP_Tempo_nt10': category_data.get('asap_tempo_nt10', []),
        'ASAP_Tempo_nt30': category_data.get('asap_tempo_nt30', []),
        'ASAP_Tempo_nt50': category_data.get('asap_tempo_nt50', []),
        'ASAP_Tempo_nt80': category_data.get('asap_tempo_nt80', []),
        'ASAP_Velocity_nt10': category_data.get('asap_velocity_nt10', []),
        'ASAP_Velocity_nt30': category_data.get('asap_velocity_nt30', []),
        'ASAP_Velocity_nt50': category_data.get('asap_velocity_nt50', []),
        'ASAP_Velocity_nt80': category_data.get('asap_velocity_nt80', [])
    }
    
    # Remove empty groups
    return {k: v for k, v in grouped_data.items() if v}

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
            'Type': 'Original' if 'Original' in category else 'Modified'
        })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df.set_index('Category')

def perform_subgroup_vs_original_tests(category_data, group_type='shutter'):
    """Perform t-tests comparing each subgroup against its original group"""
    results = []
    
    # Determine original data key and subgroups based on group_type
    if group_type.lower() == 'shutter':
        original_key = 'Shutter_Original'
        subgroups = {
            'Phrase': [k for k in category_data.keys() if k.startswith('Shutter_Phrase_') and k != original_key],
            'Bar': [k for k in category_data.keys() if k.startswith('Shutter_Bar_') and k != original_key],
            'Token': [k for k in category_data.keys() if k.startswith('Shutter_Token_') and k != original_key],
            'FQ_Low': [k for k in category_data.keys() if k.startswith('Shutter_FQ_Low_') and k != original_key],
            'FQ_High': [k for k in category_data.keys() if k.startswith('Shutter_FQ_High_') and k != original_key]
        }
    elif group_type.lower() == 'asap':
        original_key = 'ASAP_Original'
        subgroups = {
            'Rhythm': [k for k in category_data.keys() if k.startswith('ASAP_Rhythm_') and k != original_key],
            'Tempo': [k for k in category_data.keys() if k.startswith('ASAP_Tempo_') and k != original_key],
            'Velocity': [k for k in category_data.keys() if k.startswith('ASAP_Velocity_') and k != original_key]
        }
    else:
        return results
    
    # Check if original data exists
    if original_key not in category_data or not category_data[original_key]:
        print(f"警告: 未找到{group_type}原始数据，无法进行对比测试")
        return results
    
    original_data = category_data[original_key]
    
    # Perform t-tests for each subgroup against original
    for subgroup_type, subgroup_keys in subgroups.items():
        for subgroup_key in subgroup_keys:
            if subgroup_key in category_data and category_data[subgroup_key]:
                subgroup_data = category_data[subgroup_key]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(original_data, subgroup_data)
                
                results.append({
                    'Group': group_type,
                    'Subgroup_Type': subgroup_type,
                    'Subgroup': subgroup_key,
                    'T-statistic': t_stat,
                    'p-value': p_value,
                    'Significant': p_value < 0.05
                })
    
    return results

def perform_anova_within_subgroups(category_data, group_type='shutter'):
    """Perform ANOVA tests within each subgroup type (including original)"""
    results = []
    
    # Determine original data key and subgroups based on group_type
    if group_type.lower() == 'shutter':
        original_key = 'Shutter_Original'
        subgroup_types = {
            'Phrase': 'Shutter_Phrase_',
            'Bar': 'Shutter_Bar_',
            'Token': 'Shutter_Token_',
            'FQ_Low': 'Shutter_FQ_Low_',
            'FQ_High': 'Shutter_FQ_High_'
        }
    elif group_type.lower() == 'asap':
        original_key = 'ASAP_Original'
        subgroup_types = {
            'Rhythm': 'ASAP_Rhythm_',
            'Tempo': 'ASAP_Tempo_',
            'Velocity': 'ASAP_Velocity_'
        }
    else:
        return results
    
    # Check if original data exists
    if original_key not in category_data or not category_data[original_key]:
        print(f"警告: 未找到{group_type}原始数据，无法进行ANOVA测试")
        return results
    
    # Perform ANOVA for each subgroup type
    for subgroup_name, subgroup_prefix in subgroup_types.items():
        # Get all categories for this subgroup type
        subgroup_categories = {k: v for k, v in category_data.items() 
                              if k.startswith(subgroup_prefix) or k == original_key}
        
        if len(subgroup_categories) < 2:
            continue
        
        # Prepare data for ANOVA
        categories = list(subgroup_categories.keys())
        category_values = [subgroup_categories[cat] for cat in categories]
        
        try:
            f_stat, p_value = stats.f_oneway(*category_values)
            
            results.append({
                'Group': group_type,
                'Subgroup_Type': subgroup_name,
                'F-statistic': f_stat,
                'p-value': p_value,
                'Significant': p_value < 0.05,
                'Categories_Tested': len(categories)
            })
            
            # If ANOVA is significant, perform post-hoc tests
            if p_value < 0.05 and len(categories) > 2:
                # Perform pairwise t-tests with original as reference
                for cat in categories:
                    if cat != original_key:
                        _, pval = stats.ttest_ind(subgroup_categories[original_key], 
                                                 subgroup_categories[cat])
                        
                        results.append({
                            'Group': group_type,
                            'Subgroup_Type': f"{subgroup_name}_PostHoc",
                            'Comparison': f"{original_key} vs {cat}",
                            'p-value': pval,
                            'Significant': pval < 0.05
                        })
                        
        except Exception as e:
            print(f"ANOVA测试错误 ({group_type} {subgroup_name}): {str(e)}")
    
    return results

def create_detailed_boxplots(category_data, output_dir):
    """Create detailed box plots for different groups, always including original data"""
    # Separate Shutter and ASAP data
    shutter_data = {k: v for k, v in category_data.items() if k.startswith('Shutter_')}
    asap_data = {k: v for k, v in category_data.items() if k.startswith('ASAP_')}
    
    # Check if original data exists
    shutter_original = shutter_data.get('Shutter_Original', [])
    asap_original = asap_data.get('ASAP_Original', [])
    
    # Helper function to extract percentage for sorting
    def extract_percentage(category_name):
        if 'Original' in category_name:
            return 0  # Original comes first
        # Extract percentage from names like 'Shutter_Phrase_5p', 'Shutter_Bar_10p', etc.
        import re
        match = re.search(r'_(\d+)p$', category_name)
        if match:
            return int(match.group(1))
        return 999  # Put unmatched items at the end
    
    # Create Shutter subgroup plots
    if shutter_data and shutter_original:
        # Group by modification type
        phrase_data = {k: v for k, v in shutter_data.items() if 'Phrase' in k}
        bar_data = {k: v for k, v in shutter_data.items() if 'Bar' in k}
        token_data = {k: v for k, v in shutter_data.items() if 'Token' in k}
        fq_low_data = {k: v for k, v in shutter_data.items() if 'FQ_Low' in k}
        fq_high_data = {k: v for k, v in shutter_data.items() if 'FQ_High' in k}
        
        # Always add original data to each group
        phrase_data['Shutter_Original'] = shutter_original
        bar_data['Shutter_Original'] = shutter_original
        token_data['Shutter_Original'] = shutter_original
        fq_low_data['Shutter_Original'] = shutter_original
        fq_high_data['Shutter_Original'] = shutter_original
        
        # Plot each Shutter subgroup
        for group_name, group_data in [
            ('Phrase', phrase_data), 
            ('Bar', bar_data), 
            ('Token', token_data), 
            ('FQ_Low', fq_low_data),
            ('FQ_High', fq_high_data)
        ]:
            if len(group_data) > 1:  # At least original + one subgroup
                plt.figure(figsize=(12, 8))
                
                # Sort categories: Original first, then by percentage (5p, 10p, 30p, 50p, 80p)
                categories = sorted(list(group_data.keys()), key=extract_percentage)
                data_for_boxplot = [group_data[cat] for cat in categories]
                
                box = plt.boxplot(data_for_boxplot, labels=categories, patch_artist=True)
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Highlight original data
                if 'Shutter_Original' in categories:
                    ori_idx = categories.index('Shutter_Original')
                    box['boxes'][ori_idx].set_facecolor('red')
                    box['boxes'][ori_idx].set_alpha(1.0)
                
                plt.title(f'Shutter {group_name} - Loss Distribution by Percentage (vs Original)')
                plt.ylabel('Loss Value')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shutter_{group_name.lower()}_vs_original_boxplot.png'), dpi=300)
                plt.close()
    
    # Create ASAP subgroup plots
    if asap_data and asap_original:
        # Group by modification type
        rhythm_data = {k: v for k, v in asap_data.items() if 'Rhythm' in k}
        tempo_data = {k: v for k, v in asap_data.items() if 'Tempo' in k}
        velocity_data = {k: v for k, v in asap_data.items() if 'Velocity' in k}
        
        # Always add original data to each group
        rhythm_data['ASAP_Original'] = asap_original
        tempo_data['ASAP_Original'] = asap_original
        velocity_data['ASAP_Original'] = asap_original
        
        # Plot each ASAP subgroup
        for group_name, group_data in [
            ('Rhythm', rhythm_data), 
            ('Tempo', tempo_data), 
            ('Velocity', velocity_data)
        ]:
            if len(group_data) > 1:  # At least original + one subgroup
                plt.figure(figsize=(12, 8))
                
                # Sort categories to ensure original is first
                categories = sorted(list(group_data.keys()), 
                                   key=lambda x: (0 if x == 'ASAP_Original' else 1, x))
                data_for_boxplot = [group_data[cat] for cat in categories]
                
                box = plt.boxplot(data_for_boxplot, labels=categories, patch_artist=True)
                colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
                
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Highlight original data
                if 'ASAP_Original' in categories:
                    ori_idx = categories.index('ASAP_Original')
                    box['boxes'][ori_idx].set_facecolor('blue')
                    box['boxes'][ori_idx].set_alpha(1.0)
                
                plt.title(f'ASAP {group_name} - Loss Distribution by Parameter (vs Original)')
                plt.ylabel('Loss Value')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'asap_{group_name.lower()}_vs_original_boxplot.png'), dpi=300)
                plt.close()

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process large dataset loss data with detailed subgroup analysis vs original')
    parser.add_argument('-i', '--input', nargs='+', 
                        default=['loss_large_match.txt'],
                        help='Input file paths containing loss data')
    parser.add_argument('-o', '--output', default="Loss_Cal_Plot/Plot_Large_2020_Detailed_vs_Original",
                        help='Output directory for charts and results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载数据并提取详细分类...")
    category_data = extract_detailed_categories_from_files(args.input)
    
    print(f"\n检测到的分类数量: {len(category_data)}")
    for category, data in category_data.items():
        print(f"{category}: {len(data)} 个样本")
    
    # Group categories for analysis
    grouped_data = group_categories_for_analysis(category_data)
    
    print("\n计算统计信息...")
    stats_df = calculate_statistics(grouped_data)
    
    # Perform subgroup vs original tests
    print("\n执行 Shutter 子组与原始数据的对比测试...")
    shutter_test_results = perform_subgroup_vs_original_tests(grouped_data, 'shutter')
    
    print("\n执行 ASAP 子组与原始数据的对比测试...")
    asap_test_results = perform_subgroup_vs_original_tests(grouped_data, 'asap')
    
    # Perform ANOVA within each subgroup type
    print("\n执行 Shutter 各子组类型的 ANOVA 测试...")
    shutter_anova_results = perform_anova_within_subgroups(grouped_data, 'shutter')
    
    print("\n执行 ASAP 各子组类型的 ANOVA 测试...")
    asap_anova_results = perform_anova_within_subgroups(grouped_data, 'asap')
    
    # Save detailed results
    print("\n保存详细统计结果...")
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'detailed_statistics.csv')
    stats_df.to_csv(stats_file)
    print(f"统计结果保存至: {stats_file}")
    
    # Save t-test results
    if shutter_test_results or asap_test_results:
        ttest_results_df = pd.DataFrame(shutter_test_results + asap_test_results)
        ttest_file = os.path.join(output_dir, 'subgroup_vs_original_ttest_results.csv')
        ttest_results_df.to_csv(ttest_file, index=False)
        print(f"子组与原始数据对比测试结果保存至: {ttest_file}")
    
    # Save ANOVA results
    if shutter_anova_results or asap_anova_results:
        anova_results_df = pd.DataFrame(shutter_anova_results + asap_anova_results)
        anova_file = os.path.join(output_dir, 'subgroup_anova_results.csv')
        anova_results_df.to_csv(anova_file, index=False)
        print(f"子组ANOVA测试结果保存至: {anova_file}")
    
    # Generate detailed box plots
    print("\n生成详细箱线图(包含原始数据对比)...")
    create_detailed_boxplots(grouped_data, output_dir)
    
    print(f"\n分析完成！所有结果保存在 {output_dir}")
    print("\n生成的文件:")
    print("- detailed_statistics.csv: 详细统计信息")
    print("- subgroup_vs_original_ttest_results.csv: 子组与原始数据的t检验结果")
    print("- subgroup_anova_results.csv: 子组ANOVA测试结果")
    print("- shutter_*_vs_original_boxplot.png: Shutter各子组与原始数据对比的箱线图")
    print("- asap_*_vs_original_boxplot.png: ASAP各子组与原始数据对比的箱线图")

if __name__ == "__main__":
    main()