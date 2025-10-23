#!/bin/bash

# Phase5_1 批量Loss计算脚本
# 计算四个数据集在三个模型下的per token loss

set -e  # 遇到错误时退出

# 基础路径配置
BASE_DIR="/home/evev/asap-dataset"
LOSS_CAL_DIR="${BASE_DIR}/_lossCal"
OUTPUT_BASE_DIR="${BASE_DIR}/+Loss/Phase5_1"

# 数据集配置
declare -A DATASETS=(
    ["noise_color_ori"]="${BASE_DIR}/noise_color_ori"
    ["asap_ori"]="${BASE_DIR}/asap_ori"
    ["ShutterStock_32k_ori"]="${BASE_DIR}/ShutterStock_32k_ori"
    ["Unconditional_ori"]="${BASE_DIR}/Unconditional_ori"
)

# 模型配置
declare -A MODELS=(
    ["small"]="loss_cal_small.py"
    ["medium"]="loss_cal_medium.py"
    ["mgen-melody"]="loss_cal_mgen-melody.py"
)

# 创建输出目录
mkdir -p "${OUTPUT_BASE_DIR}"

echo "=== Phase5_1 批量Loss计算开始 ==="
echo "输出目录: ${OUTPUT_BASE_DIR}"
echo "处理数据集: ${!DATASETS[@]}"
echo "使用模型: ${!MODELS[@]}"
echo

# 记录开始时间
START_TIME=$(date +%s)

# 遍历所有数据集和模型组合
for dataset_name in "${!DATASETS[@]}"; do
    dataset_path="${DATASETS[$dataset_name]}"
    
    echo "处理数据集: $dataset_name"
    echo "数据集路径: $dataset_path"
    
    # 检查数据集是否存在
    if [ ! -d "$dataset_path" ]; then
        echo "警告: 数据集目录不存在: $dataset_path"
        continue
    fi
    
    for model_name in "${!MODELS[@]}"; do
        model_script="${MODELS[$model_name]}"
        model_script_path="${LOSS_CAL_DIR}/${model_script}"
        
        echo "  使用模型: $model_name ($model_script)"
        
        # 检查模型脚本是否存在
        if [ ! -f "$model_script_path" ]; then
            echo "  错误: 模型脚本不存在: $model_script_path"
            continue
        fi
        
        # 创建输出目录结构
        output_dir="${OUTPUT_BASE_DIR}/${dataset_name}_${model_name}_token_loss_medium"
        per_token_dir="${output_dir}/per_token"
        mkdir -p "$per_token_dir"
        
        # 设置输出文件
        results_file="${output_dir}/results.txt"
        log_file="${output_dir}/results.txt.log"
        
        echo "    输出目录: $output_dir"
        echo "    开始计算..."
        
        # 记录任务开始时间
        task_start=$(date +%s)
        
        # 执行loss计算
        cd "$LOSS_CAL_DIR"
        
        if [ "$model_name" = "mgen-melody" ]; then
            # melody模型使用不同的参数
            python "$model_script" \
                --audio_dir "$dataset_path" \
                --output_file "$results_file" \
                --reduction "none" \
                --token_output_dir "$per_token_dir" \
                > "$log_file" 2>&1
        else
            # small和medium模型使用相同的参数
            python "$model_script" \
                --audio_dir "$dataset_path" \
                --output_file "$results_file" \
                --reduction "none" \
                --token_output_dir "$per_token_dir" \
                > "$log_file" 2>&1
        fi
        
        # 检查执行结果
        if [ $? -eq 0 ]; then
            task_end=$(date +%s)
            task_duration=$((task_end - task_start))
            echo "    ✓ 完成 (耗时: ${task_duration}秒)"
            
            # 统计生成的文件数量
            if [ -d "$per_token_dir" ]; then
                csv_count=$(find "$per_token_dir" -name "*.csv" | wc -l)
                echo "    生成CSV文件: $csv_count 个"
            fi
            
            if [ -f "$results_file" ]; then
                result_lines=$(wc -l < "$results_file")
                echo "    结果文件行数: $result_lines"
            fi
        else
            echo "    ✗ 失败 - 查看日志: $log_file"
        fi
        
        echo
    done
    
    echo "数据集 $dataset_name 处理完成"
    echo "----------------------------------------"
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=== Phase5_1 批量Loss计算完成 ==="
echo "总耗时: ${TOTAL_DURATION}秒 ($(($TOTAL_DURATION / 60))分钟)"
echo "输出目录: ${OUTPUT_BASE_DIR}"

# 生成处理摘要
echo
echo "=== 处理摘要 ==="
for dataset_name in "${!DATASETS[@]}"; do
    echo "数据集: $dataset_name"
    for model_name in "${!MODELS[@]}"; do
        output_dir="${OUTPUT_BASE_DIR}/${dataset_name}_${model_name}_token_loss_medium"
        if [ -d "$output_dir" ]; then
            results_file="${output_dir}/results.txt"
            per_token_dir="${output_dir}/per_token"
            
            if [ -f "$results_file" ] && [ -d "$per_token_dir" ]; then
                result_count=$(wc -l < "$results_file" 2>/dev/null || echo "0")
                csv_count=$(find "$per_token_dir" -name "*.csv" 2>/dev/null | wc -l)
                echo "  $model_name: ✓ ($result_count 结果, $csv_count CSV文件)"
            else
                echo "  $model_name: ✗ (输出文件缺失)"
            fi
        else
            echo "  $model_name: ✗ (目录不存在)"
        fi
    done
done

echo
echo "脚本执行完成！"