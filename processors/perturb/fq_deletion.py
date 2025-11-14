import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import signal

def delete_frequency_range(audio_data, sr, freq_range, deletion_percentage, min_segment_len=0.01, max_segment_len=0.1):
    sr = float(sr)
    
    if isinstance(freq_range, str):
        freq_range = parse_freq_range(freq_range)
    elif isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
        freq_range = (float(freq_range[0]), float(freq_range[1]))
    else:
        raise ValueError(f"Invalid freq_range format: {freq_range}. Expected tuple of two numbers or string 'min-max'.")
    
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
        for pos, length in deletion_positions:
            if not (start_time + segment_len <= pos or start_time >= pos + length):
                overlap = True
                break
        if not overlap:
            deletion_positions.append((start_time, segment_len))
            remaining_duration -= segment_len

    # 对每个删除片段进行处理
    for start_time, segment_len in deletion_positions:
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

def process_audio_file(file_path, output_dir, freq_configs):
    try:
        audio_data, sr = sf.read(file_path)
        
        sr = int(sr)

        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)

        for freq_range, percentage, suffix in freq_configs:
            processed_audio = delete_frequency_range(audio_data, sr, freq_range, percentage)

            output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")

            sf.write(output_file, processed_audio, sr)

        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def parse_freq_range(freq_str):
    try:
        min_freq, max_freq = map(float, freq_str.split('-'))
        return (min_freq, max_freq)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid frequency range format: {freq_str}. Use 'min_freq-max_freq' (e.g., '40-800')")

def process_directory(input_dir, output_base_dir, freq_ranges, deletion_percentages):
    freq_configs = []
    for freq_range in freq_ranges:
        if isinstance(freq_range, str):
            freq_range = parse_freq_range(freq_range)
        elif isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
            freq_range = (float(freq_range[0]), float(freq_range[1]))
        
        for percentage in deletion_percentages:
            freq_desc = f"fq{int(freq_range[0])}-{int(freq_range[1])}"
            perc_desc = f"{int(percentage*100)}p"
            suffix = f"_{freq_desc}_{perc_desc}"
            freq_configs.append((freq_range, percentage, suffix))

    print(f"Will generate {len(freq_configs)} configurations:")
    for freq_range, percentage, suffix in freq_configs:
        print(f"  - Frequency range: {freq_range[0]}-{freq_range[1]}Hz, deletion: {percentage*100}%, suffix: {suffix}")

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
            process_audio_file(str(audio_file), output_base_dir, freq_configs)
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
            process_audio_file(str(audio_file), output_subdir, freq_configs)

def main():
    parser = argparse.ArgumentParser(description="Delete specified frequency ranges in audio files")
    parser.add_argument("--input_dir", default="data_100", help="Input directory containing audio folders")
    parser.add_argument("--output_dir", default="data_100_fq_deletion", help="Output directory")
    parser.add_argument("--freq_ranges", nargs="+", type=parse_freq_range, default=["2000-20000"], help="Frequency deletion ranges, format: 'min_freq-max_freq'")
    parser.add_argument("--deletion_percentages", nargs="+", type=float, default=[0.05, 0.10, 0.30, 0.50, 0.80], help="Deletion percentages (0-1)")
    parser.add_argument("--min_segment", type=float, default=0.01, help="Minimum segment length (seconds)")
    parser.add_argument("--max_segment", type=float, default=0.05, help="Maximum segment length (seconds)")
    
    args = parser.parse_args()

    if args.min_segment >= args.max_segment:
        print("Error: min segment length must be less than max segment length!")
        return
    
    if not all(0 <= p <= 1 for p in args.deletion_percentages):
        print("Error: deletion percentages must be between 0 and 1!")
        return

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("Starting audio processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frequency ranges: {args.freq_ranges}")
    print(f"Deletion percentages: {[f'{p*100}%' for p in args.deletion_percentages]}")
    print(f"Segment length range: {args.min_segment}s - {args.max_segment}s")

    global delete_frequency_range
    original_func = delete_frequency_range
    def updated_delete_frequency_range(audio_data, sr, freq_range, deletion_percentage, min_segment_len=args.min_segment, max_segment_len=args.max_segment):
        return original_func(audio_data, sr, freq_range, deletion_percentage, min_segment_len, max_segment_len)
    
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, 'delete_frequency_range', updated_delete_frequency_range)

    process_directory(input_dir, output_dir, args.freq_ranges, args.deletion_percentages)

    print("Processing complete!")

if __name__ == "__main__":
    main()