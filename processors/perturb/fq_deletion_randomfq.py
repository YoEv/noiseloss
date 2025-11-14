import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import signal

def delete_frequency_range(audio_data, sr, freq_ranges, deletion_percentage, min_segment_len=1, max_segment_len=5):
    processed_audio = np.copy(audio_data)
    is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1
    total_duration = len(audio_data) / sr
    deletion_duration = total_duration * deletion_percentage
    remaining_duration = deletion_duration
    deletion_positions = []
    while remaining_duration > 0:
        segment_len = min(random.uniform(min_segment_len, max_segment_len), remaining_duration)
        max_start = total_duration - segment_len
        if max_start <= 0:
            break
        start_time = random.uniform(0, max_start)
        overlap = False
        for pos_info in deletion_positions:
            pos = pos_info[0]
            length = pos_info[1]
            if not (start_time + segment_len <= pos or start_time >= pos + length):
                overlap = True
                break
        if not overlap:
            freq_range = random.choice(freq_ranges)
            deletion_positions.append((start_time, segment_len, freq_range))
            remaining_duration -= segment_len
    for start_time, segment_len, freq_range in deletion_positions:
        start_idx = int(start_time * sr)
        end_idx = int((start_time + segment_len) * sr)
        if is_stereo:
            segment = processed_audio[start_idx:end_idx, :]
        else:
            segment = processed_audio[start_idx:end_idx]
        if is_stereo:
            for channel in range(segment.shape[1]):
                stft = np.fft.rfft(segment[:, channel])
                freq_resolution = sr / (2 * (len(stft) - 1))
                min_idx = int(freq_range[0] / freq_resolution)
                max_idx = int(freq_range[1] / freq_resolution)
                min_idx = max(0, min_idx)
                max_idx = min(len(stft) - 1, max_idx)
                stft[min_idx:max_idx+1] = 0
                segment[:, channel] = np.fft.irfft(stft, len(segment[:, channel]))
        else:
            stft = np.fft.rfft(segment)
            freq_resolution = sr / (2 * (len(stft) - 1))
            min_idx = int(freq_range[0] / freq_resolution)
            max_idx = int(freq_range[1] / freq_resolution)
            min_idx = max(0, min_idx)
            max_idx = min(len(stft) - 1, max_idx)
            stft[min_idx:max_idx+1] = 0
            segment = np.fft.irfft(stft, len(segment))
        if is_stereo:
            processed_audio[start_idx:end_idx, :] = segment
        else:
            processed_audio[start_idx:end_idx] = segment

    return processed_audio

def process_audio_file(file_path, output_dir, deletion_percentages):
    try:
        audio_data, sr = sf.read(file_path)
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        freq_ranges = [
            (40, 1000),
            (600, 1500),
            (2000, 6000)
        ]
        for percentage in deletion_percentages:
            processed_audio = delete_frequency_range(audio_data, sr, freq_ranges, percentage)
            suffix = f"_fq_dele_random_{int(percentage*100)}p"
            output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")
            sf.write(output_file, processed_audio, sr)

        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_base_dir):
    deletion_percentages = [0.1, 0.3, 0.5, 0.8]

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
            process_audio_file(str(audio_file), output_base_dir, deletion_percentages)
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
            process_audio_file(str(audio_file), output_subdir, deletion_percentages)

def main():
    parser = argparse.ArgumentParser(description="Randomly delete specified frequency ranges in audio files")
    parser.add_argument("--input_dir", default="data_100", help="Input directory containing audio folders")
    parser.add_argument("--output_dir", default="data_100_fq_deletion_random", help="Output directory")
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