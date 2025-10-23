#!/bin/bash

# 批量噪音替换脚本
# 使用方法: ./batch_replace_noise.sh

set -e  # 遇到错误时退出

echo "开始批量噪音替换处理..."

# 定义基础参数
FRAME_RATE=50
SEED=42

# 定义token长度数组
TOKEN_LENGTHS=(5 10 50 100 150 200)

# 1. 处理 D_asap_silent_sele_ori 目录 (pink, blue, brown, -30dB)
echo "\n=== 处理 D_asap_silent_sele_ori 目录 ==="
INPUT_DIR="D_asap_silent_sele_ori"
NOISE_TYPES=("pink" "blue" "brown")
DB_LEVEL=-30

if [ ! -d "$INPUT_DIR" ]; then
    echo "警告: 目录 $INPUT_DIR 不存在，跳过处理"
else
    for noise_type in "${NOISE_TYPES[@]}"; do
        echo "\n--- 处理噪音类型: $noise_type ---"
        for token_length in "${TOKEN_LENGTHS[@]}"; do
            output_dir="asap_replace_noise_${noise_type}_at5_tk${token_length}"
            echo "处理: $INPUT_DIR -> $output_dir (${token_length}tk, ${noise_type}, ${DB_LEVEL}dB)"
            
            python _processors/replace_noise_at5.py \
                --input "$INPUT_DIR" \
                --output "$output_dir" \
                --tokens $token_length \
                --db $DB_LEVEL \
                --frame_rate $FRAME_RATE \
                --seed $SEED \
                --noise_type $noise_type
            
            echo "完成: $output_dir"
        done
    done
fi

# 2. 处理 D_Unconditional_ori 目录 (pink, blue, brown, -20dB)
echo "\n=== 处理 D_Unconditional_ori 目录 ==="
INPUT_DIR="D_Unconditional_ori"
NOISE_TYPES=("pink" "blue" "brown")
DB_LEVEL=-20

if [ ! -d "$INPUT_DIR" ]; then
    echo "警告: 目录 $INPUT_DIR 不存在，跳过处理"
else
    for noise_type in "${NOISE_TYPES[@]}"; do
        echo "\n--- 处理噪音类型: $noise_type ---"
        for token_length in "${TOKEN_LENGTHS[@]}"; do
            output_dir="unconditional_replace_noise_${noise_type}_at5_tk${token_length}"
            echo "处理: $INPUT_DIR -> $output_dir (${token_length}tk, ${noise_type}, ${DB_LEVEL}dB)"
            
            python _processors/replace_noise_at5.py \
                --input "$INPUT_DIR" \
                --output "$output_dir" \
                --tokens $token_length \
                --db $DB_LEVEL \
                --frame_rate $FRAME_RATE \
                --seed $SEED \
                --noise_type $noise_type
            
            echo "完成: $output_dir"
        done
    done
fi

# 3. 处理 D_ShutterStock_32k 目录 (white, pink, blue, brown, -12dB)
echo "\n=== 处理 D_ShutterStock_32k 目录 ==="
INPUT_DIR="D_ShutterStock_32k"
NOISE_TYPES=("white" "pink" "blue" "brown")
DB_LEVEL=-12

if [ ! -d "$INPUT_DIR" ]; then
    echo "警告: 目录 $INPUT_DIR 不存在，跳过处理"
else
    for noise_type in "${NOISE_TYPES[@]}"; do
        echo "\n--- 处理噪音类型: $noise_type ---"
        for token_length in "${TOKEN_LENGTHS[@]}"; do
            output_dir="shutter_replace_noise_${noise_type}_at5_tk${token_length}"
            echo "处理: $INPUT_DIR -> $output_dir (${token_length}tk, ${noise_type}, ${DB_LEVEL}dB)"
            
            python _processors/replace_noise_at5.py \
                --input "$INPUT_DIR" \
                --output "$output_dir" \
                --tokens $token_length \
                --db $DB_LEVEL \
                --frame_rate $FRAME_RATE \
                --seed $SEED \
                --noise_type $noise_type
            
            echo "完成: $output_dir"
        done
    done
fi

echo "\n=== 所有处理完成! ==="
echo "总共处理的组合数:"
echo "- D_asap_silent_sele_ori: 3种噪音 × 6种token长度 = 18个输出目录"
echo "- D_Unconditional_ori: 3种噪音 × 6种token长度 = 18个输出目录"
echo "- D_ShutterStock_32k: 4种噪音 × 6种token长度 = 24个输出目录"
echo "总计: 60个输出目录"