#!/bin/bash

# 批量计算Phase5_1数据的per token loss脚本
# 使用方法: ./batch_loss_calculation_phase5.sh

set -e  # 遇到错误时退出

echo "开始批量计算Phase5_1数据的per token loss..."

# 定义基础路径
INPUT_BASE="+Dataset/Phase5_1"
OUTPUT_BASE="+Loss/Phase5_1"

# 定义三个loss计算脚本
LOSS_SCRIPTS=(
    "_lossCal/loss_cal_small.py"
    "_lossCal/loss_cal_medium.py"
    "_lossCal/loss_cal_mgen-melody.py"
)

# 定义对应的模型名称后缀
MODEL_SUFFIXES=(
    "small"
    "medium"
    "mgen-melody"
)

# 创建输出基础目录
mkdir -p "$OUTPUT_BASE"

echo "\n=== 扫描输入目录结构 ==="
if [ ! -d "$INPUT_BASE" ]; then
    echo "错误: 输入目录 $INPUT_BASE 不存在"
    exit 1
fi

# 获取所有子目录
SUBDIRS=()
while IFS= read -r -d '' dir; do
    SUBDIRS+=("$(basename "$dir")")
done < <(find "$INPUT_BASE" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "找到 ${#SUBDIRS[@]} 个子目录需要处理:"
for subdir in "${SUBDIRS[@]}"; do
    echo "  - $subdir"
done

# 处理每个子目录
for subdir in "${SUBDIRS[@]}"; do
    echo "\n=== 处理目录: $subdir ==="
    
    input_dir="$INPUT_BASE/$subdir"
    
    # 检查目录是否存在音频文件
    audio_count=$(find "$input_dir" -name "*.wav" -o -name "*.mp3" | wc -l)
    if [ "$audio_count" -eq 0 ]; then
        echo "警告: 目录 $input_dir 中没有找到音频文件，跳过"
        continue
    fi
    
    echo "找到 $audio_count 个音频文件"
    
    # 使用三个模型分别计算loss
    for i in "${!LOSS_SCRIPTS[@]}"; do
        script="${LOSS_SCRIPTS[$i]}"
        model_suffix="${MODEL_SUFFIXES[$i]}"
        
        echo "\n--- 使用模型: $model_suffix ---"
        
        # 创建输出目录名称: 原目录名 + _token_loss + 模型后缀
        output_dir="$OUTPUT_BASE/${subdir}_token_loss_${model_suffix}"
        output_file="$output_dir/results.txt"
        
        echo "输入目录: $input_dir"
        echo "输出目录: $output_dir"
        
        # 创建输出目录
        mkdir -p "$output_dir"
        
        # 运行loss计算脚本
        echo "开始计算 per token loss..."
        
        if [ "$model_suffix" = "mgen-melody" ]; then
            # mgen-melody脚本参数格式稍有不同
            python "$script" \
                --audio_dir "$input_dir" \
                --output_file "$output_file" \
                --reduction "none" \
                --token_output_dir "$output_dir/per_token"
        else
            # small和medium脚本使用相同的参数格式
            python "$script" \
                --audio_dir "$input_dir" \
                --output_file "$output_file" \
                --reduction "none" \
                --token_output_dir "$output_dir/per_token"
        fi
        
        if [ $? -eq 0 ]; then
            echo "✓ 完成: $output_dir"
        else
            echo "✗ 失败: $output_dir"
        fi
        
        # 清理GPU内存
        echo "清理GPU内存..."
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
        
        # 添加短暂延迟以确保内存完全释放
        sleep 2
    done
done

echo "\n=== 处理完成统计 ==="
echo "输入目录: $INPUT_BASE"
echo "输出目录: $OUTPUT_BASE"
echo "处理的子目录数: ${#SUBDIRS[@]}"
echo "使用的模型数: ${#LOSS_SCRIPTS[@]}"
echo "总计生成的输出目录数: $((${#SUBDIRS[@]} * ${#LOSS_SCRIPTS[@]}))"

echo "\n=== 输出目录结构预览 ==="
if [ -d "$OUTPUT_BASE" ]; then
    echo "$OUTPUT_BASE/"
    find "$OUTPUT_BASE" -maxdepth 1 -type d | sort | while read -r dir; do
        if [ "$dir" != "$OUTPUT_BASE" ]; then
            echo "├── $(basename "$dir")/"
        fi
    done
fi

echo "\n所有处理完成!"