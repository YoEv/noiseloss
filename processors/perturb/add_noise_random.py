import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import signal

def generate_white_noise(length, db_level=12):
    noise = np.random.normal(0, 0.1, length)
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    gain = 10 ** (db_level / 20)
    noise = noise * gain
    return noise

def generate_pink_noise(length, db_level=12):
    white_noise = np.random.normal(0, 0.1, length)
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)
    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
    gain = 10 ** (db_level / 20)
    pink_noise = pink_noise * gain
    return pink_noise

def generate_brown_noise(length, db_level=12):
    white_noise = np.random.normal(0, 0.1, length)
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)
    if np.max(np.abs(brown_noise)) > 0:
        brown_noise = brown_noise / np.max(np.abs(brown_noise))
    gain = 10 ** (db_level / 20)
    brown_noise = brown_noise * gain
    return brown_noise

def generate_blue_noise(length, db_level=12):
    white_noise = np.random.normal(0, 0.1, length)
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)
    if np.max(np.abs(blue_noise)) > 0:
        blue_noise = blue_noise / np.max(np.abs(blue_noise))
    gain = 10 ** (db_level / 20)
    blue_noise = blue_noise * gain
    return blue_noise

def add_random_noise_to_audio(audio_data, sr, noise_percentage, min_segment_len=1, max_segment_len=5):
    noisy_audio = np.copy(audio_data)
    is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1
    total_duration = len(audio_data) / sr
    noise_duration = total_duration * noise_percentage
    remaining_duration = noise_duration
    noise_positions = []
    noise_types = ['white', 'pink', 'brown', 'blue']

    while remaining_duration > 0:
        segment_len = min(random.uniform(min_segment_len, max_segment_len), remaining_duration)
        max_start = total_duration - segment_len
        if max_start <= 0:
            break
        start_time = random.uniform(0, max_start)
        overlap = False
        for pos, length, _ in noise_positions:
            if not (start_time + segment_len <= pos or start_time >= pos + length):
                overlap = True
                break
        if not overlap:
            noise_type = random.choice(noise_types)
            noise_positions.append((start_time, segment_len, noise_type))
            remaining_duration -= segment_len

    for start_time, segment_len, noise_type in noise_positions:
        start_idx = int(start_time * sr)
        end_idx = int((start_time + segment_len) * sr)
        if noise_type == 'white':
            noise = generate_white_noise(end_idx - start_idx)
        elif noise_type == 'pink':
            noise = generate_pink_noise(end_idx - start_idx)
        elif noise_type == 'brown':
            noise = generate_brown_noise(end_idx - start_idx)
        elif noise_type == 'blue':
            noise = generate_blue_noise(end_idx - start_idx)
        noise = noise * 0.3
        if is_stereo:
            noise = np.column_stack((noise, noise))
        noisy_audio[start_idx:end_idx] = noisy_audio[start_idx:end_idx] + noise

    max_val = np.max(np.abs(noisy_audio))
    if max_val > 0.95:
        noisy_audio = noisy_audio / max_val * 0.95

    return noisy_audio

def process_audio_file(file_path, output_dir, noise_percentages):
    try:
        audio_data, sr = sf.read(file_path)
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        for percentage, suffix in noise_percentages:
            noisy_audio = add_random_noise_to_audio(audio_data, sr, percentage)
            output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")
            sf.write(output_file, noisy_audio, sr)
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_base_dir):
    noise_percentages = [
        (0.1, "_randomnoise_10"),
        (0.3, "_randomnoise_30"),
        (0.5, "_randomnoise_50"),
        (0.8, "_randomnoise_80")
    ]

    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在!")
        return

    os.makedirs(output_base_dir, exist_ok=True)
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        print(f"警告: 在 '{input_dir}' 中没有找到子文件夹!")
        print("尝试直接处理输入目录中的音频文件...")

        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            audio_files.extend(list(Path(input_dir).glob(f"*{ext}")))
        if not audio_files:
            print(f"错误: 在 '{input_dir}' 中没有找到音频文件!")
            return
        print(f"找到 {len(audio_files)} 个音频文件")
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            process_audio_file(str(audio_file), output_base_dir, noise_percentages)
        return

    print(f"在 '{input_dir}' 中找到 {len(subdirs)} 个子文件夹")

    for subdir in tqdm(subdirs, desc="处理文件夹"):
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_base_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            found_files = list(Path(input_subdir).glob(f"*{ext}"))
            audio_files.extend(found_files)
        if not audio_files:
            print(f"警告: 在子文件夹 '{subdir}' 中没有找到音频文件!")
            continue
        print(f"在子文件夹 '{subdir}' 中找到 {len(audio_files)} 个音频文件")
        for audio_file in tqdm(audio_files, desc=f"处理 {subdir} 中的文件", leave=False):
            process_audio_file(str(audio_file), output_subdir, noise_percentages)

def main():
    parser = argparse.ArgumentParser(description="在音频文件中添加随机类型的噪音")
    parser.add_argument("--input_dir", default="data_100",
                        help="包含音频文件夹的输入目录 (默认: data_100)")
    parser.add_argument("--output_dir", default="data_100_random_noise",
                        help="输出目录 (默认: data_100_random_noise)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"开始处理音频文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

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