import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import signal

def generate_white_noise(length, db_level=12):
    """
    生成白噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    noise = np.random.normal(0, 0.1, length)
    # 归一化到0dB
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    noise = noise * gain

    return noise

def generate_pink_noise(length, db_level=12):
    """
    生成粉红噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    # 生成白噪音
    white_noise = np.random.normal(0, 0.1, length)

    # 应用粉红滤波器
    # 粉红噪音的功率谱密度与频率成反比 (1/f)
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)

    # 归一化到0dB
    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    pink_noise = pink_noise * gain

    return pink_noise

def generate_brown_noise(length, db_level=12):
    """
    生成布朗噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    # 生成白噪音
    white_noise = np.random.normal(0, 0.1, length)

    # 应用布朗滤波器
    # 布朗噪音的功率谱密度与频率成反比的平方 (1/f^2)
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)

    # 归一化到0dB
    if np.max(np.abs(brown_noise)) > 0:
        brown_noise = brown_noise / np.max(np.abs(brown_noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    brown_noise = brown_noise * gain

    return brown_noise

def generate_blue_noise(length, db_level=12):
    """
    生成蓝噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    # 生成白噪音
    white_noise = np.random.normal(0, 0.1, length)

    # 应用蓝噪音滤波器
    # 蓝噪音的功率谱密度与频率成正比 (f)
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)

    # 归一化到0dB
    if np.max(np.abs(blue_noise)) > 0:
        blue_noise = blue_noise / np.max(np.abs(blue_noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    blue_noise = blue_noise * gain

    return blue_noise

def add_random_noise_to_audio(audio_data, sr, noise_percentage, min_segment_len=1, max_segment_len=5):
    """
    在音频中随机位置添加随机类型的噪音

    参数:
    - audio_data: 音频数据数组
    - sr: 采样率
    - noise_percentage: 噪音占总长度的百分比 (0-1)
    - min_segment_len: 最小噪音片段长度(秒)
    - max_segment_len: 最大噪音片段长度(秒)

    返回:
    - 添加噪音后的音频数据
    """
    # 创建音频数据的副本
    noisy_audio = np.copy(audio_data)

    # 检查音频是否为立体声
    is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1

    # 计算音频总长度(秒)
    total_duration = len(audio_data) / sr

    # 计算需要添加噪音的总时长(秒)
    noise_duration = total_duration * noise_percentage

    # 计算需要添加的噪音片段数量
    remaining_duration = noise_duration
    noise_positions = []

    # 噪音类型列表
    noise_types = ['white', 'pink', 'brown', 'blue']

    # 生成不重叠的噪音位置
    while remaining_duration > 0:
        # 确定当前片段长度(在min_segment_len和max_segment_len之间，且不超过剩余长度)
        segment_len = min(random.uniform(min_segment_len, max_segment_len), remaining_duration)

        # 随机选择起始位置(秒)
        max_start = total_duration - segment_len
        if max_start <= 0:
            break

        start_time = random.uniform(0, max_start)

        # 检查是否与现有噪音片段重叠
        overlap = False
        for pos, length, _ in noise_positions:
            if not (start_time + segment_len <= pos or start_time >= pos + length):
                overlap = True
                break

        if not overlap:
            # 随机选择噪音类型
            noise_type = random.choice(noise_types)
            noise_positions.append((start_time, segment_len, noise_type))
            remaining_duration -= segment_len

    # 添加噪音
    for start_time, segment_len, noise_type in noise_positions:
        # 转换为样本索引
        start_idx = int(start_time * sr)
        end_idx = int((start_time + segment_len) * sr)

        # 生成指定类型的噪音
        if noise_type == 'white':
            noise = generate_white_noise(end_idx - start_idx)
        elif noise_type == 'pink':
            noise = generate_pink_noise(end_idx - start_idx)
        elif noise_type == 'brown':
            noise = generate_brown_noise(end_idx - start_idx)
        elif noise_type == 'blue':
            noise = generate_blue_noise(end_idx - start_idx)

        # 调整噪音的响度（乘以一个系数，控制噪音的相对强度）
        noise = noise * 0.3  # 可以调整这个系数来控制噪音的强度

        # 如果是立体声，将噪音转换为立体声
        if is_stereo:
            noise = np.column_stack((noise, noise))

        # 将噪音添加到音频中
        noisy_audio[start_idx:end_idx] = noisy_audio[start_idx:end_idx] + noise

    # 归一化音频，防止削波
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 0.95:
        noisy_audio = noisy_audio / max_val * 0.95

    return noisy_audio

def process_audio_file(file_path, output_dir, noise_percentages):
    """处理单个音频文件，添加不同百分比的随机噪音"""
    try:
        # 读取音频文件
        audio_data, sr = sf.read(file_path)

        # 获取文件名和扩展名
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)

        # 为每个噪音百分比创建一个版本
        for percentage, suffix in noise_percentages:
            # 添加随机噪音
            noisy_audio = add_random_noise_to_audio(audio_data, sr, percentage)

            # 创建输出文件名
            output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")

            # 保存处理后的音频
            sf.write(output_file, noisy_audio, sr)

        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_base_dir):
    """处理目录中的所有音频文件"""
    # 定义噪音百分比和对应的后缀
    noise_percentages = [
        (0.1, "_randomnoise_10"),
        (0.3, "_randomnoise_30"),
        (0.5, "_randomnoise_50"),
        (0.8, "_randomnoise_80")
    ]

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
            process_audio_file(str(audio_file), output_base_dir, noise_percentages)

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
            process_audio_file(str(audio_file), output_subdir, noise_percentages)

def main():
    parser = argparse.ArgumentParser(description="在音频文件中添加随机类型的噪音")
    parser.add_argument("--input_dir", default="data_100",
                        help="包含音频文件夹的输入目录 (默认: data_100)")
    parser.add_argument("--output_dir", default="data_100_random_noise",
                        help="输出目录 (默认: data_100_random_noise)")
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

    process_directory(input_dir, output_dir)

    print("处理完成！")

if __name__ == "__main__":
    main()