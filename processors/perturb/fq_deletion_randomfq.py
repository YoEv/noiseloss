import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import signal

def delete_frequency_range(audio_data, sr, freq_ranges, deletion_percentage, min_segment_len=1, max_segment_len=5):
    """
    在音频中随机位置删除指定频率范围的内容，每个片段随机选择一个频率范围

    参数:
    - audio_data: 音频数据数组
    - sr: 采样率
    - freq_ranges: 可选的频率范围列表，每个元素格式为 (min_freq, max_freq)
    - deletion_percentage: 删除占总长度的百分比 (0-1)
    - min_segment_len: 最小删除片段长度(秒)
    - max_segment_len: 最大删除片段长度(秒)

    返回:
    - 处理后的音频数据
    """
    # 创建音频数据的副本
    processed_audio = np.copy(audio_data)

    # 检查音频是否为立体声
    is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1

    # 计算音频总长度(秒)
    total_duration = len(audio_data) / sr

    # 计算需要删除频率的总时长(秒)
    deletion_duration = total_duration * deletion_percentage

    # 计算需要删除的片段数量
    remaining_duration = deletion_duration
    deletion_positions = []

    # 生成不重叠的删除位置
    while remaining_duration > 0:
        # 确定当前片段长度(在min_segment_len和max_segment_len之间，且不超过剩余长度)
        segment_len = min(random.uniform(min_segment_len, max_segment_len), remaining_duration)

        # 随机选择起始位置(秒)
        max_start = total_duration - segment_len
        if max_start <= 0:
            break

        start_time = random.uniform(0, max_start)

        # 检查是否与现有删除片段重叠
        overlap = False
        for pos_info in deletion_positions:
            pos = pos_info[0]  # 获取起始时间
            length = pos_info[1]  # 获取片段长度
            if not (start_time + segment_len <= pos or start_time >= pos + length):
                overlap = True
                break

        if not overlap:
            # 为每个片段随机选择一个频率范围
            freq_range = random.choice(freq_ranges)
            deletion_positions.append((start_time, segment_len, freq_range))
            remaining_duration -= segment_len

    # 对每个删除片段进行处理
    for start_time, segment_len, freq_range in deletion_positions:
        # 转换为样本索引
        start_idx = int(start_time * sr)
        end_idx = int((start_time + segment_len) * sr)

        # 提取要处理的片段
        if is_stereo:
            segment = processed_audio[start_idx:end_idx, :]
        else:
            segment = processed_audio[start_idx:end_idx]

        # 对片段进行短时傅里叶变换(STFT)
        if is_stereo:
            # 处理左右声道
            for channel in range(segment.shape[1]):
                # 进行STFT
                stft = np.fft.rfft(segment[:, channel])

                # 计算频率分辨率
                freq_resolution = sr / (2 * (len(stft) - 1))

                # 计算要删除的频率范围对应的索引
                min_idx = int(freq_range[0] / freq_resolution)
                max_idx = int(freq_range[1] / freq_resolution)

                # 确保索引在有效范围内
                min_idx = max(0, min_idx)
                max_idx = min(len(stft) - 1, max_idx)

                # 将指定频率范围的幅度设为0（删除频率内容）
                stft[min_idx:max_idx+1] = 0

                # 进行逆STFT
                segment[:, channel] = np.fft.irfft(stft, len(segment[:, channel]))
        else:
            # 进行STFT
            stft = np.fft.rfft(segment)

            # 计算频率分辨率
            freq_resolution = sr / (2 * (len(stft) - 1))

            # 计算要删除的频率范围对应的索引
            min_idx = int(freq_range[0] / freq_resolution)
            max_idx = int(freq_range[1] / freq_resolution)

            # 确保索引在有效范围内
            min_idx = max(0, min_idx)
            max_idx = min(len(stft) - 1, max_idx)

            # 将指定频率范围的幅度设为0（删除频率内容）
            stft[min_idx:max_idx+1] = 0

            # 进行逆STFT
            segment = np.fft.irfft(stft, len(segment))

        # 将处理后的片段放回原音频
        if is_stereo:
            processed_audio[start_idx:end_idx, :] = segment
        else:
            processed_audio[start_idx:end_idx] = segment

    return processed_audio

def process_audio_file(file_path, output_dir, deletion_percentages):
    """处理单个音频文件，随机选择频率范围并删除不同比例的内容"""
    try:
        # 读取音频文件
        audio_data, sr = sf.read(file_path)

        # 获取文件名和扩展名
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)

        # 定义可能的频率范围
        freq_ranges = [
            (40, 1000),    # 低频范围
            (600, 1500),  # 中频范围
            (2000, 6000)  # 高频范围
        ]

        # 为每个删除比例创建一个版本
        for percentage in deletion_percentages:
            # 删除指定频率范围，每个片段随机选择频率范围
            processed_audio = delete_frequency_range(audio_data, sr, freq_ranges, percentage)

            # 创建后缀
            suffix = f"_fq_dele_random_{int(percentage*100)}p"

            # 创建输出文件名
            output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")

            # 保存处理后的音频
            sf.write(output_file, processed_audio, sr)

        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_base_dir):
    """处理目录中的所有音频文件"""
    # 定义删除比例
    deletion_percentages = [0.1, 0.3, 0.5, 0.8]  # 10%, 30%, 50%, 80%

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在!")
        return

    # 确保输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)

    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # 检查是否找到子文件夹
    if not subdirs:
        print(f"警告: 在 '{input_dir}' 中没有找到子文件夹!")
        print("尝试直接处理输入目录中的音频文件...")

        # 获取输入目录中的所有音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            audio_files.extend(list(Path(input_dir).glob(f"*{ext}")))

        if not audio_files:
            print(f"错误: 在 '{input_dir}' 中没有找到音频文件!")
            return

        print(f"找到 {len(audio_files)} 个音频文件")

        # 处理每个音频文件
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            process_audio_file(str(audio_file), output_base_dir, deletion_percentages)

        return

    print(f"在 '{input_dir}' 中找到 {len(subdirs)} 个子文件夹")

    # 处理每个子文件夹
    for subdir in tqdm(subdirs, desc="处理文件夹"):
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_base_dir, subdir)

        # 创建输出子目录
        os.makedirs(output_subdir, exist_ok=True)

        # 获取所有音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            found_files = list(Path(input_subdir).glob(f"*{ext}"))
            audio_files.extend(found_files)

        # 检查是否找到音频文件
        if not audio_files:
            print(f"警告: 在子文件夹 '{subdir}' 中没有找到音频文件!")
            continue

        print(f"在子文件夹 '{subdir}' 中找到 {len(audio_files)} 个音频文件")

        # 处理每个音频文件
        for audio_file in tqdm(audio_files, desc=f"处理 {subdir} 中的文件", leave=False):
            process_audio_file(str(audio_file), output_subdir, deletion_percentages)

def main():
    parser = argparse.ArgumentParser(description="在音频文件中随机删除指定频率范围的内容")
    parser.add_argument("--input_dir", default="data_100",
                        help="包含音频文件夹的输入目录 (默认: data_100)")
    parser.add_argument("--output_dir", default="data_100_fq_deletion_random",
                        help="输出目录 (默认: data_100_fq_deletion_random)")
    args = parser.parse_args()

    # 获取绝对路径
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"开始处理音频文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 检查必要的库是否已安装
    try:
        import soundfile
        from scipy import signal
        print("所有必要的库已安装")
    except ImportError as e:
        print(f"错误: 未安装必要的库! {str(e)}")
        print("请使用以下命令安装所需库:")
        print("pip install numpy soundfile tqdm scipy")
        return

    # 处理所有音频文件
    process_directory(input_dir, output_dir)

    print("处理完成！")

if __name__ == "__main__":
    main()