# 定义两个数组
audio_dirs=(
    "data_100"
    "data_100_color_noise"
    "data_100_fq_deletion_random"
    "data_100_noise"
    "data_100_randomnoise"
)

output_files=(
    "data_100_loss_cal_mert.txt"
    "data_100_color_noise_loss_cal_mert.txt"
    "data_100_fq_deletion_random_loss_cal_mert.txt"
    "data_100_noise_loss_cal_mert.txt"
    "data_100_randomnoise_loss_cal_mert.txt"
)

# 检查数组长度是否相同
if [ ${#audio_dirs[@]} -ne ${#output_files[@]} ]; then
    echo "错误：audio_dirs和output_files数组长度不一致"
    exit 1
fi

# 循环执行
for i in "${!audio_dirs[@]}"; do
    echo "正在处理: ${audio_dirs[$i]} => ${output_files[$i]}"
    python loss_cal_mert_raw.py \
        --audio_dir "${audio_dirs[$i]}" \
        --output_file "${output_files[$i]}" \
        # --loss_type cross_entropy

    # 检查命令是否执行成功
    if [ $? -ne 0 ]; then
        echo "警告: ${audio_dirs[$i]} 处理失败"
    fi
done

echo "所有任务已完成"
