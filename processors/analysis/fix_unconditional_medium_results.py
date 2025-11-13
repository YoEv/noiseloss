#!/usr/bin/env python3
import os
import re
import glob
from pathlib import Path

def extract_data_from_log(log_file_path):
    results = {}
    current_subfolder = None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if 'per_token/unconditional_topk' in line and '_samples_' in line:
                match = re.search(r'per_token/(unconditional_topk\d+_samples)_', line)
                if match:
                    current_subfolder = match.group(1)
            
            elif current_subfolder and re.match(r'^sample_\d+\.wav: [\d\.]+$', line):
                parts = line.split(': ')
                if len(parts) == 2:
                    filename = parts[0]
                    loss_value = parts[1]
                    key = f"{current_subfolder}/{filename}"
                    results[key] = loss_value
    
    return results

def extract_data_from_ori_results(results_file_path):
    results = {}
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data_lines = [line.strip() for line in lines[1:] if line.strip()]
    
    samples_per_group = 20
    subfolders = ['unconditional_topk100_samples', 'unconditional_topk200_samples', 'unconditional_topk50_samples']
    
    for i, line in enumerate(data_lines):
        if '\t' in line:
            parts = line.split('\t')
        else:
            parts = line.split()
        
        if len(parts) >= 2:
            filename = parts[0]
            loss_value = parts[1]
            
            group_index = i // samples_per_group
            if group_index < len(subfolders):
                subfolder = subfolders[group_index]
                key = f"{subfolder}/{filename}"
                results[key] = loss_value
    
    return results

def write_results_txt(results_dict, output_path):

    sorted_items = sorted(results_dict.items())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for key, value in sorted_items:
            f.write(f"{key}: {value}\n")

def process_unconditional_medium_directories():
    base_dir = "/home/evev/asap-dataset/+Loss/Phase5_1"
    
    pattern = os.path.join(base_dir, "unconditional_*_medium")
    medium_dirs = glob.glob(pattern)
    
    ori_dir = os.path.join(base_dir, "Unconditional_ori_medium")
    if os.path.exists(ori_dir):
        medium_dirs.append(ori_dir)
    
    processed_count = 0
    
    for dir_path in medium_dirs:
        dir_name = os.path.basename(dir_path)
        log_file = os.path.join(dir_path, "results.txt.log")
        results_file = os.path.join(dir_path, "results.txt")
        
        if not os.path.exists(results_file):
            print(f"Warning: {results_file} doesn't exit, skip")
            continue
        
        results_data = {}
        
        if dir_name == "Unconditional_ori_medium":
            results_data = extract_data_from_ori_results(results_file)
            print(f"  从ori results.txt提取了 {len(results_data)} 条记录")
        elif os.path.exists(log_file):
            results_data = extract_data_from_log(log_file)
            print(f"  从.log文件提取了 {len(results_data)} 条记录")
        else:
            print(f"  警告: 既没有.log文件也不是ori目录，跳过")
            continue
        
        if not results_data:
            print(f"  警告: 未提取到数据")
            continue
        
        backup_file = results_file + ".backup"
        if os.path.exists(backup_file):
            import time
            timestamp = int(time.time())
            backup_file = f"{results_file}.backup.{timestamp}"
        
        os.rename(results_file, backup_file)
        print(f"  备份原文件到: {backup_file}")
        
        write_results_txt(results_data, results_file)
        print(f"  成功写入 {len(results_data)} 条记录到 {results_file}")
        
        processed_count += 1
    
    print(f"\n处理完成，共处理了 {processed_count} 个目录")

if __name__ == "__main__":
    process_unconditional_medium_directories()