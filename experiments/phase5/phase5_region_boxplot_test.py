import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PARAMS = {
    'base_dir': '/home/evev/asap-dataset/+Loss/Phase5_1',
    'output_dir': '/home/evev/asap-dataset/+Loss_Plot/Phase5_1/3regions_boxplot',
    'debug': True
}

REGION_COLORS = {
    'bumped': '#FFFFE0',
    'compromised': '#FFB6C1',
    'regression': '#ADD8E6'
}

DATABASES = ['asap', 'ShutterStock_32k', 'Unconditional']
MODEL_TYPE = 'small'

NOISE_CONFIGS = {
    'noise_color': ['white'],
    'noise_length': ['50', '100', '150', '200'],
    'noise_volume': ['ori']
}

BUMPED_START_FIXED = 245
STABLE_LEN = 5
REGRESSION_START_OFFSET = 50

def detect_regions_ma(ori_loss, mix_loss, compromised_end=403):
    """bumped, compromised, regression"""
    diff_curve = np.array(mix_loss) - np.array(ori_loss)
    
    baseline = np.mean(diff_curve[:200])
    
    bumped_start = BUMPED_START_FIXED

    bumped_end = bumped_start
    threshold = 0.1
    
    for i in range(bumped_start + 1, min(len(diff_curve), compromised_end)):
        if abs(diff_curve[i] - baseline) < threshold:
            stable_count = 0
            for j in range(i, min(i + STABLE_LEN, len(diff_curve))):
                if abs(diff_curve[j] - baseline) < threshold:
                    stable_count += 1
                else:
                    break
            
            if stable_count >= STABLE_LEN:
                bumped_end = i
                break
    
    compromised_start = bumped_end
    
    regression_start = compromised_end + REGRESSION_START_OFFSET
    regression_end = min(regression_start + 100, len(diff_curve))
    
    def calculate_avg_diff(start, end):
        if start < len(diff_curve) and end <= len(diff_curve) and start < end:
            return np.mean(diff_curve[start:end])
        return 0
    
    return {
        'bumped': {
            'start': bumped_start,
            'end': bumped_end,
            'detected': bumped_end > bumped_start,
            'avg_diff': calculate_avg_diff(bumped_start, bumped_end)
        },
        'compromised': {
            'start': compromised_start,
            'end': compromised_end,
            'detected': compromised_end > compromised_start,
            'avg_diff': calculate_avg_diff(compromised_start, compromised_end)
        },
        'regression': {
            'start': regression_start,
            'end': regression_end,
            'detected': regression_end > regression_start,
            'avg_diff': calculate_avg_diff(regression_start, regression_end)
        }
    }

def load_single_condition_data(condition_dir):
    per_token_dir = os.path.join(condition_dir, 'per_token')
    
    if not os.path.exists(per_token_dir):
        return None
    
    csv_files = [f for f in os.listdir(per_token_dir) if f.endswith('.csv')]
    if not csv_files:
        return None
    
    all_loss_data = []
    
    for csv_file in csv_files:
        file_path = os.path.join(per_token_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            if 'token_position' in df.columns and 'avg_loss_value' in df.columns:
                df = df.dropna().sort_values('token_position')
                loss_values = df['avg_loss_value'].values
                if len(loss_values) > 0:
                    all_loss_data.append(loss_values)
        except Exception as e:
            if PARAMS['debug']:
                print(f"Error on loading {csv_file}:{e}")
            continue
    
    if not all_loss_data:
        return None
    
    min_length = min(len(seq) for seq in all_loss_data)
    truncated_data = [seq[:min_length] for seq in all_loss_data]
    avg_loss = np.mean(truncated_data, axis=0)
    
    return avg_loss

def collect_database_data(database_name):
    base_dir = PARAMS['base_dir']
    
    database_data = {}
    
    for noise_type in NOISE_CONFIGS['noise_color']:
        for noise_length in NOISE_CONFIGS['noise_length']:
            if database_name == 'asap':
                ori_dir_name = f"asap_ori_{MODEL_TYPE}"
                noise_dir_name = f"asap_replace_noise_{noise_type}_at5_tk{noise_length}_token_loss_{MODEL_TYPE}"
            elif database_name == 'ShutterStock_32k':
                ori_dir_name = f"ShutterStock_32k_ori_{MODEL_TYPE}"
                noise_dir_name = f"shutter_replace_noise_{noise_type}_at5_tk{noise_length}_token_loss_{MODEL_TYPE}"
            elif database_name == 'Unconditional':
                ori_dir_name = f"Unconditional_ori_{MODEL_TYPE}"
                noise_dir_name = f"unconditional_replace_noise_{noise_type}_at5_tk{noise_length}_token_loss_{MODEL_TYPE}"
            
            ori_dir = os.path.join(base_dir, ori_dir_name)
            noise_dir = os.path.join(base_dir, noise_dir_name)
            
            if PARAMS['debug']:
                print(f"  Checking directory: {ori_dir}")
                print(f"  Checking directory: {noise_dir}")
            
            if os.path.exists(ori_dir) and os.path.exists(noise_dir):
                if PARAMS['debug']:
                    print(f"  Found directory, starting processing: {noise_type}_{noise_length}")
                
                ori_data = load_single_condition_data(ori_dir)
                noise_data = load_single_condition_data(noise_dir)
                
                if ori_data is not None and noise_data is not None:
                    compromised_end = 249 + int(noise_length)
                    
                    min_len = min(len(ori_data), len(noise_data))
                    ori_data = ori_data[:min_len]
                    noise_data = noise_data[:min_len]
                    
                    regions = detect_regions_ma(ori_data, noise_data, compromised_end)
                    
                    if regions is not None:
                        key = f"{noise_type}_{noise_length}"
                        if key not in database_data:
                            database_data[key] = {'bumped': [], 'compromised': [], 'regression': []}
                        
                        for region_name in ['bumped', 'compromised', 'regression']:
                            if regions[region_name]['detected']:
                                database_data[key][region_name].append(regions[region_name]['avg_diff'])
            else:
                if PARAMS['debug']:
                    print(f"  Directory does not exist: {ori_dir} or {noise_dir}")
    
    return database_data

def collect_all_data():
    all_data = {}
    
    for database in DATABASES:
        if PARAMS['debug']:
            print(f"\nProcessing: {database}")
        
        data = collect_database_data(database)
        if data and len(data) > 0:
            all_data[database] = data
            if PARAMS['debug']:
                print(f"  Successfully collected data for {database}")
        else:
            if PARAMS['debug']:
                print(f"  Failed to collect data for {database}")
    
    return all_data

def create_boxplot(all_data):
    if not all_data:
        print("No available data for plotting")
        return
    
    plot_data = []
    
    for database in DATABASES:
        if database in all_data:
            database_data = all_data[database]
            
            for config_key, config_data in database_data.items():
                noise_type, noise_length = config_key.split('_')
                
                for region in ['bumped', 'compromised', 'regression']:
                    if region in config_data and len(config_data[region]) > 0:
                        values = config_data[region]
                        for value in values:
                            plot_data.append({
                                'Database': database,
                                'Region': region,
                                'Loss_Difference': value,
                                'Noise_Type': noise_type,
                                'Noise_Length': noise_length
                            })
    
    if not plot_data:
        print("No valid data points for plotting")
        return
    
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(15, 10))
    
    sns.boxplot(data=df, x='Region', y='Loss_Difference', hue='Database', 
                palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax = plt.gca()
    
    regions = ['bumped', 'compromised', 'regression']
    colors = [REGION_COLORS['bumped'], REGION_COLORS['compromised'], REGION_COLORS['regression']]
    
    for i, (region, color) in enumerate(zip(regions, colors)):
        ax.axvspan(i-0.4, i+0.4, alpha=0.3, color=color, zorder=0)
    
    plt.title(f'Loss Difference by Region and Database\n(All noise configurations)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Loss Difference (relative to baseline)', fontsize=12)
    plt.legend(title='Database', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(PARAMS['output_dir'], exist_ok=True)
    output_path = os.path.join(PARAMS['output_dir'], 
                              f"regions_boxplot_all_configs.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nBoxplot saved: {output_path}")
    
    print("\n=== Data Statistics ===")
    for database in DATABASES:
        if database in all_data:
            print(f"\n{database}:")
            database_data = all_data[database]
            for config_key, config_data in database_data.items():
                print(f"  Configuration {config_key}:")
                for region in ['bumped', 'compromised', 'regression']:
                    if region in config_data and len(config_data[region]) > 0:
                        values = config_data[region]
                        print(f"    {region}: {len(values)} data points, average: {np.mean(values):.4f}")

def main():
    print("Starting to collect data...")
    all_data = collect_all_data()
    
    if not all_data:
        print("No data collected")
        return
    
    print(f"\nSuccessfully collected data for {len(all_data)} databases")
    
    print("\nStarting to draw boxplot...")
    create_boxplot(all_data)
    
    print("\nComplete!")

if __name__ == "__main__":
    main()