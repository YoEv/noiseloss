import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import signal

def delete_frequency_range(audio_data, sr, freq_range, deletion_percentage, min_segment_len=0.01, max_segment_len=0.1):
    """
    在音频中随机位置删除指定频率范围的内容
    """
    # 确保采样率是数字类型
    sr = float(sr)
    
    # 确保频率范围是数字元组
    if isinstance(freq_range, str):
        freq_range = parse_freq_range(freq_range)
    elif isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
        freq_range = (float(freq_range[0]), float(freq_range[1]))
    else:
        raise ValueError(f"Invalid freq_range format: {freq_range}. Expected tuple of two numbers or string 'min-max'.")
    
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
        for pos, length in deletion_positions:
            if not (start_time + segment_len <= pos or start_time >= pos + length):
                overlap = True
                break

        if not overlap:
            deletion_positions.append((start_time, segment_len))
            remaining_duration -= segment_len

    # 对每个删除片段进行处理
    for start_time, segment_len in deletion_positions:
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

def process_audio_file(file_path, output_dir, freq_configs):
    """处理单个音频文件，删除不同频率范围的内容"""
    try:
        # 读取音频文件
        audio_data, sr = sf.read(file_path)
        
        # 确保采样率是整数类型（soundfile.write需要整数）
        sr = int(sr)

        # 获取文件名和扩展名
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)

        # 为每个频率配置创建一个版本
        for freq_range, percentage, suffix in freq_configs:
            # freq_range should already be a proper tuple from process_directory
            processed_audio = delete_frequency_range(audio_data, sr, freq_range, percentage)

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

def parse_freq_range(freq_str):
    """解析频率范围字符串，格式: 'min_freq-max_freq'"""
    try:
        min_freq, max_freq = map(float, freq_str.split('-'))
        return (min_freq, max_freq)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid frequency range format: {freq_str}. Use 'min_freq-max_freq' (e.g., '40-800')")

def process_directory(input_dir, output_base_dir, freq_ranges, deletion_percentages):
    """处理目录中的所有音频文件"""
    # 生成频率配置
    freq_configs = []
    for freq_range in freq_ranges:
        # 确保freq_range是元组格式
        if isinstance(freq_range, str):
            freq_range = parse_freq_range(freq_range)
        elif isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
            freq_range = (float(freq_range[0]), float(freq_range[1]))
        
        for percentage in deletion_percentages:
            # 创建描述性后缀
            freq_desc = f"fq{int(freq_range[0])}-{int(freq_range[1])}"
            perc_desc = f"{int(percentage*100)}p"
            suffix = f"_{freq_desc}_{perc_desc}"
            freq_configs.append((freq_range, percentage, suffix))

    print(f"将生成 {len(freq_configs)} 种配置:")
    for freq_range, percentage, suffix in freq_configs:
        print(f"  - 频率范围: {freq_range[0]}-{freq_range[1]}Hz, 删除比例: {percentage*100}%, 后缀: {suffix}")

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
            process_audio_file(str(audio_file), output_base_dir, freq_configs)

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
            process_audio_file(str(audio_file), output_subdir, freq_configs)

def main():
    parser = argparse.ArgumentParser(description="对音频文件进行频率删除处理")
    parser.add_argument("--input_dir", default="data_100",
                        help="包含音频文件夹的输入目录 (默认: data_100)")
    parser.add_argument("--output_dir", default="data_100_fq_deletion",
                        help="输出目录 (默认: data_100_fq_deletion)")
    parser.add_argument("--freq_ranges", nargs="+", type=parse_freq_range,
                        default=["2000-20000"],
                        help="频率删除范围，格式: 'min_freq-max_freq' (默认: 40-800)")
    parser.add_argument("--deletion_percentages", nargs="+", type=float,
                        default=[0.05, 0.10, 0.30, 0.50, 0.80],
                        help="删除比例列表 (0-1之间的小数，默认: 0.05 0.10 0.30 0.50 0.80)")
    parser.add_argument("--min_segment", type=float, default=0.01,
                        help="最小删除片段长度(秒) (默认: 0.01)")
    parser.add_argument("--max_segment", type=float, default=0.05,
                        help="最大删除片段长度(秒) (默认: 0.1)")
    
    args = parser.parse_args()

    # 验证参数
    if args.min_segment >= args.max_segment:
        print("错误: 最小片段长度必须小于最大片段长度!")
        return
    
    if not all(0 <= p <= 1 for p in args.deletion_percentages):
        print("错误: 删除比例必须在0-1之间!")
        return

    # 获取绝对路径
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"开始处理音频文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"频率范围: {args.freq_ranges}")
    print(f"删除比例: {[f'{p*100}%' for p in args.deletion_percentages]}")
    print(f"片段长度范围: {args.min_segment}s - {args.max_segment}s")

    # 更新删除函数的默认参数
    global delete_frequency_range
    original_func = delete_frequency_range
    def updated_delete_frequency_range(audio_data, sr, freq_range, deletion_percentage, min_segment_len=args.min_segment, max_segment_len=args.max_segment):
        return original_func(audio_data, sr, freq_range, deletion_percentage, min_segment_len, max_segment_len)
    
    # 临时替换函数
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, 'delete_frequency_range', updated_delete_frequency_range)

    # 处理所有音频文件
    process_directory(input_dir, output_dir, args.freq_ranges, args.deletion_percentages)

    print("处理完成！")

if __name__ == "__main__":
    main()