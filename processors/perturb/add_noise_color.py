import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import signal

def generate_white_noise(length, db_level=0):

    noise = np.random.normal(0, 0.1, length)
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    gain = 10 ** (db_level / 20)
    noise = noise * gain

    return noise

def generate_pink_noise(length, db_level=0):

    white_noise = np.random.normal(0, 0.1, length)
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)

    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
    gain = 10 ** (db_level / 20)
    pink_noise = pink_noise * gain

    return pink_noise

def generate_brown_noise(length, db_level=0):

    white_noise = np.random.normal(0, 0.1, length)
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)

    if np.max(np.abs(brown_noise)) > 0:
        brown_noise = brown_noise / np.max(np.abs(brown_noise))
    gain = 10 ** (db_level / 20)
    brown_noise = brown_noise * gain

    return brown_noise

def generate_blue_noise(length, db_level=0):

    white_noise = np.random.normal(0, 0.1, length)
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)

    if np.max(np.abs(blue_noise)) > 0:
        blue_noise = blue_noise / np.max(np.abs(blue_noise))
    gain = 10 ** (db_level / 20)
    blue_noise = blue_noise * gain

    return blue_noise

def add_noise_to_audio(audio_data, sr, noise_type, db_level=0, noise_percentage=0.4, min_segment_len=0.01, max_segment_len=0.1):

    noisy_audio = np.copy(audio_data)
    is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1
    total_duration = len(audio_data) / sr
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

    # 添加噪音
    for start_time, segment_len in noise_positions:
        start_idx = int(start_time * sr)
        end_idx = int((start_time + segment_len) * sr)
        if noise_type == 'white':
            noise = generate_white_noise(end_idx - start_idx, db_level)
        elif noise_type == 'pink':
            noise = generate_pink_noise(end_idx - start_idx, db_level)
        elif noise_type == 'brown':
            noise = generate_brown_noise(end_idx - start_idx, db_level)
        elif noise_type == 'blue':
            noise = generate_blue_noise(end_idx - start_idx, db_level)
        else:
            noise = generate_white_noise(end_idx - start_idx, db_level)

        noise = noise * 0.3

        if is_stereo:
            noise = np.column_stack((noise, noise))
        noisy_audio[start_idx:end_idx] = noisy_audio[start_idx:end_idx] + noise

    max_val = np.max(np.abs(noisy_audio))
    if max_val > 0.95:
        noisy_audio = noisy_audio / max_val * 0.95

    return noisy_audio

def process_audio_file(file_path, output_dir, noise_configs):

    try:
        audio_data, sr = sf.read(file_path)
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        for noise_type, db_level, suffix in noise_configs:
            noisy_audio = add_noise_to_audio(audio_data, sr, noise_type, db_level)
            output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")
            sf.write(output_file, noisy_audio, sr)
        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_base_dir):

    noise_configs = [
        ('pink', 0, '_pink_0db_40p'),
        ('brown', 0, '_brown_0db_40p'),
        ('pink', 6, '_pink_6db_40p'),
        ('brown', 6, '_brown_6db_40p'),
    ]

    if not os.path.exists(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist!")
        return

    os.makedirs(output_base_dir, exist_ok=True)
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        print(f"Warning: no subdirectories found in '{input_dir}'!")
        print("Trying to process audio files in the input directory directly...")

        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            audio_files.extend(list(Path(input_dir).glob(f"*{ext}")))
        if not audio_files:
            print(f"Error: no audio files found in '{input_dir}'!")
            return
        print(f"Found {len(audio_files)} audio files")
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            process_audio_file(str(audio_file), output_base_dir, noise_configs)
        return

    print(f"Found {len(subdirs)} subdirectories in '{input_dir}'")

    for subdir in tqdm(subdirs, desc="Processing folders"):
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_base_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            found_files = list(Path(input_subdir).glob(f"*{ext}"))
            audio_files.extend(found_files)
        if not audio_files:
            print(f"Warning: no audio files found in subdirectory '{subdir}'!")
            continue
        print(f"Found {len(audio_files)} audio files in subdirectory '{subdir}'")
        for audio_file in tqdm(audio_files, desc=f"Processing files in {subdir}", leave=False):
            process_audio_file(str(audio_file), output_subdir, noise_configs)

def main():
    parser = argparse.ArgumentParser(description="Add colored noise to audio files at random positions")
    parser.add_argument("--input_dir", default="data_100", help="Input directory containing audio folders")
    parser.add_argument("--output_dir", default="data_100_color_noise", help="Output directory")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("Starting audio processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    try:
        import soundfile
        from scipy import signal
        print("All required libraries are installed")
    except ImportError as e:
        print(f"Error: required libraries not installed! {str(e)}")
        print("Install with: pip install numpy soundfile tqdm scipy")
        return

    process_directory(input_dir, output_dir)

    print("Processing complete!")

if __name__ == "__main__":
    main()