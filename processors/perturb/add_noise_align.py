import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
import librosa
import librosa.display
from scipy import signal
import matplotlib.pyplot as plt

def analyze_audio(audio_data, sr, segment_duration=1.0):
    segment_samples = int(segment_duration * sr)

    num_segments = int(np.ceil(len(audio_data) / segment_samples))

    segments_analysis = []

    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = min((i + 1) * segment_samples, len(audio_data))
        segment = audio_data[start_idx:end_idx]

        if len(segment.shape) > 1 and segment.shape[1] > 1:
            segment = np.mean(segment, axis=1)

        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), 'constant')

        stft = librosa.stft(segment)
        magnitude = np.abs(stft)

        freq_distribution = np.mean(magnitude, axis=1)

        db_spectrum = librosa.amplitude_to_db(magnitude, ref=np.max)
        loudness = np.mean(db_spectrum, axis=1)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        segments_analysis.append({
            'start_time': i * segment_duration,
            'end_time': min((i + 1) * segment_duration, len(audio_data) / sr),
            'freqs': freqs,
            'freq_distribution': freq_distribution,
            'loudness': loudness
        })

    return segments_analysis

def generate_aligned_noise(audio_analysis, sr, duration):
    total_samples = int(duration * sr)

    aligned_noise = np.zeros(total_samples)

    for segment in audio_analysis:
        start_sample = int(segment['start_time'] * sr)
        end_sample = int(segment['end_time'] * sr)
        segment_length = end_sample - start_sample

        white_noise = np.random.normal(0, 0.1, segment_length)

        filtered_noise = white_noise.copy()

        for i in range(len(segment['freqs']) - 1):
            center_freq = segment['freqs'][i]
            loudness = segment['loudness'][i]

            if loudness < -60:
                continue

            if i < len(segment['freqs']) - 1:
                bandwidth = segment['freqs'][i+1] - segment['freqs'][i]
            else:
                bandwidth = segment['freqs'][i] - segment['freqs'][i-1]

            nyquist = sr / 2.0
            low_freq = max(0.001, min(0.999, (center_freq - bandwidth/2) / nyquist))
            high_freq = max(0.001, min(0.999, (center_freq + bandwidth/2) / nyquist))

            if low_freq >= high_freq:
                continue

            b, a = signal.butter(2, [low_freq, high_freq], btype='band')
            band_noise = signal.lfilter(b, a, white_noise)

            gain = 10 ** (loudness / 20)

            band_noise = band_noise * gain * segment['freq_distribution'][i] / np.max(segment['freq_distribution'])

            filtered_noise += band_noise

        if np.max(np.abs(filtered_noise)) > 0:
            filtered_noise = filtered_noise / np.max(np.abs(filtered_noise)) * 0.3

        if start_sample < len(aligned_noise) and end_sample <= len(aligned_noise):
            aligned_noise[start_sample:end_sample] += filtered_noise
    if np.max(np.abs(aligned_noise)) > 0:
        aligned_noise = aligned_noise / np.max(np.abs(aligned_noise)) * 0.5

    return aligned_noise

def add_noise_to_audio(audio_data, sr, noise_percentage, min_segment_len=1, max_segment_len=5):
    noisy_audio = np.copy(audio_data)
    is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1
    total_duration = len(audio_data) / sr
    audio_analysis = analyze_audio(audio_data, sr)
    noise_duration = total_duration * noise_percentage
    remaining_duration = noise_duration
    noise_positions = []

    while remaining_duration > 0:
        segment_len = min(random.uniform(min_segment_len, max_segment_len), remaining_duration)

        max_start = total_duration - segment_len
        if max_start <= 0:
            break

        start_time = random.uniform(0, max_start)

        overlap = False
        for pos, length in noise_positions:
            if not (start_time + segment_len <= pos or start_time >= pos + length):
                overlap = True
                break

        if not overlap:
            noise_positions.append((start_time, segment_len))
            remaining_duration -= segment_len

    for start_time, segment_len in noise_positions:
        # 转换为样本索引
        start_idx = int(start_time * sr)
        end_idx = int((start_time + segment_len) * sr)

        # 生成与原音频对齐的噪音
        aligned_noise = generate_aligned_noise(audio_analysis, sr, segment_len)

        # 如果是立体声，将噪音转换为立体声
        if is_stereo:
            aligned_noise = np.column_stack((aligned_noise, aligned_noise))

        # 将噪音添加到音频中
        segment_length = min(len(aligned_noise), end_idx - start_idx)
        noisy_audio[start_idx:start_idx + segment_length] += aligned_noise[:segment_length]

    # 归一化音频，防止削波
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 0.95:
        noisy_audio = noisy_audio / max_val * 0.95

    return noisy_audio

def process_audio_file(file_path, output_dir, noise_percentages):
    audio_data, sr = sf.read(file_path)

    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)

    for percentage, suffix in noise_percentages:
        noisy_audio = add_noise_to_audio(audio_data, sr, percentage)

        output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")

        sf.write(output_file, noisy_audio, sr)

    return True

def process_directory(input_dir, output_base_dir):

    noise_percentages = [
        (0.1, "_alignednoise_10"),
        (0.3, "_alignednoise_30"),
        (0.5, "_alignednoise_50"),
        (0.8, "_alignednoise_80")
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

def visualize_audio_analysis(audio_path, output_dir=None):
    audio_data, sr = sf.read(audio_path)

    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)

    analysis = analyze_audio(audio_data, sr)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    for i, segment in enumerate(analysis[:5]):  # 只显示前5个段
        plt.plot(segment['freqs'], segment['freq_distribution'], label=f"段 {i+1}")
    plt.title("频率分布")
    plt.xlabel("频率 (Hz)")
    plt.ylabel("幅度")
    plt.legend()
    plt.xscale('log')

    plt.subplot(2, 1, 2)
    for i, segment in enumerate(analysis[:5]):
        plt.plot(segment['freqs'], segment['loudness'], label=f"段 {i+1}")
    plt.title("响度分布")
    plt.xlabel("频率 (Hz)")
    plt.ylabel("响度 (dB)")
    plt.legend()
    plt.xscale('log')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(audio_path)
        name, _ = os.path.splitext(file_name)
        plt.savefig(os.path.join(output_dir, f"{name}_analysis.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="在音频文件中添加与原音频对齐的噪音")
    parser.add_argument("--input_dir", default="data_100",
                        help="包含音频文件夹的输入目录 (默认: data_100)")
    parser.add_argument("--output_dir", default="data_100_aligned_noise",
                        help="输出目录 (默认: data_100_aligned_noise)")
    parser.add_argument("--visualize", action="store_true",
                        help="是否可视化音频分析结果")
    parser.add_argument("--visualize_dir", default="analysis_plots",
                        help="保存可视化结果的目录 (默认: analysis_plots)")
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"开始处理音频文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")


    if args.visualize:
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            if os.path.isdir(input_dir):
                for root, _, files in os.walk(input_dir):
                    for file in files:
                        if file.lower().endswith(ext):
                            audio_files.append(os.path.join(root, file))
                            break
                    if audio_files:
                        break
            if audio_files:
                break

        if audio_files:
            print(f"可视化音频文件: {audio_files[0]}")
            visualize_audio_analysis(audio_files[0], args.visualize_dir)
        else:
            print("未找到音频文件进行可视化")

    process_directory(input_dir, output_dir)

    print("处理完成！")

if __name__ == "__main__":
    main()