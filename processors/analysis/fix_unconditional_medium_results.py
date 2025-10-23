#!/usr/bin/env python3
import os
import re
import glob
from pathlib import Path

def extract_data_from_log(log_file_path):
    """从.log文件中提取损失值数据"""
    results = {}
    current_subfolder = None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测子文件夹名称
            if 'per_token/unconditional_topk' in line and '_samples_' in line:
                # 从路径中提取子文件夹名
                match = re.search(r'per_token/(unconditional_topk\d+_samples)_', line)
                if match:
                    current_subfolder = match.group(1)
            
            # 提取样本损失值
            elif current_subfolder and re.match(r'^sample_\d+\.wav: [\d\.]+$', line):
                parts = line.split(': ')
                if len(parts) == 2:
                    filename = parts[0]
                    loss_value = parts[1]
                    key = f"{current_subfolder}/{filename}"
                    results[key] = loss_value
    
    return results

def extract_data_from_ori_results(results_file_path):
    """从ori的results.txt文件中提取数据并添加子文件夹前缀"""
    results = {}
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过表头
    data_lines = [line.strip() for line in lines[1:] if line.strip()]
    
    # 根据数据行数推断子文件夹分组
    # 假设每个子文件夹有20个样本，按顺序分组
    samples_per_group = 20
    subfolders = ['unconditional_topk100_samples', 'unconditional_topk200_samples', 'unconditional_topk50_samples']
    
    for i, line in enumerate(data_lines):
        if '\t' in line:
            # 制表符分隔格式
            parts = line.split('\t')
        else:
            # 空格分隔格式
            parts = line.split()
        
        if len(parts) >= 2:
            filename = parts[0]
            loss_value = parts[1]
            
            # 确定子文件夹
            group_index = i // samples_per_group
            if group_index < len(subfolders):
                subfolder = subfolders[group_index]
                key = f"{subfolder}/{filename}"
                results[key] = loss_value
    
    return results

def write_results_txt(results_dict, output_path):
    """将结果写入results.txt文件"""
    # 按键名排序
    sorted_items = sorted(results_dict.items())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for key, value in sorted_items:
            f.write(f"{key}: {value}\n")

def process_unconditional_medium_directories():
    """批量处理所有unconditional medium目录"""
    base_dir = "/home/evev/asap-dataset/+Loss/Phase5_1"
    
    # 处理噪声目录（有.log文件的）
    pattern = os.path.join(base_dir, "unconditional_*_medium")
    medium_dirs = glob.glob(pattern)
    
    # 处理ori目录
    ori_dir = os.path.join(base_dir, "Unconditional_ori_medium")
    if os.path.exists(ori_dir):
        medium_dirs.append(ori_dir)
    
    processed_count = 0
    
    for dir_path in medium_dirs:
        dir_name = os.path.basename(dir_path)
        log_file = os.path.join(dir_path, "results.txt.log")
        results_file = os.path.join(dir_path, "results.txt")
        
        if not os.path.exists(results_file):
            print(f"警告: {results_file} 不存在，跳过")
            continue
            
        print(f"处理: {dir_path}")
        
        try:
            results_data = {}
            
            if dir_name == "Unconditional_ori_medium":
                # 处理ori目录，从现有results.txt提取数据
                results_data = extract_data_from_ori_results(results_file)
                print(f"  从ori results.txt提取了 {len(results_data)} 条记录")
            elif os.path.exists(log_file):
                # 处理噪声目录，从.log文件提取数据
                results_data = extract_data_from_log(log_file)
                print(f"  从.log文件提取了 {len(results_data)} 条记录")
            else:
                print(f"  警告: 既没有.log文件也不是ori目录，跳过")
                continue
            
            if not results_data:
                print(f"  警告: 未提取到数据")
                continue
            
            # 备份原始results.txt
            backup_file = results_file + ".backup"
            if os.path.exists(backup_file):
                # 如果备份已存在，添加时间戳
                import time
                timestamp = int(time.time())
                backup_file = f"{results_file}.backup.{timestamp}"
            
            os.rename(results_file, backup_file)
            print(f"  备份原文件到: {backup_file}")
            
            # 写入新的results.txt
            write_results_txt(results_data, results_file)
            print(f"  成功写入 {len(results_data)} 条记录到 {results_file}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  错误: 处理 {dir_path} 时出错: {e}")
    
    print(f"\n处理完成，共处理了 {processed_count} 个目录")

if __name__ == "__main__":
    process_unconditional_medium_directories()