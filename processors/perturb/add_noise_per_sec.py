import numpy as np
import soundfile as sf
import argparse
import os

def generate_white_noise(length, db_level=-20):
    white_noise = np.random.randn(length)
    white_noise = white_noise / np.std(white_noise)
    scaling_factor = 10 ** (db_level / 20)
    return white_noise * scaling_factor

def add_noise_at_time(audio_data, sample_rate, start_time_sec, token_length=5, db_level=-20, frame_rate=50):
    start_sample = int(start_time_sec * sample_rate)
    single_token_duration = 1.0 / frame_rate
    duration_samples = int(token_length * single_token_duration * sample_rate)
    
    if start_sample >= len(audio_data):
        return audio_data
    
    end_sample = min(start_sample + duration_samples, len(audio_data))
    actual_duration = end_sample - start_sample
    
    if len(audio_data.shape) == 1:
        noise = generate_white_noise(actual_duration, db_level)
        audio_data[start_sample:end_sample] += noise
    else:
        for channel in range(audio_data.shape[1]):
            noise = generate_white_noise(actual_duration, db_level)
            audio_data[start_sample:end_sample, channel] += noise
    
    return audio_data

def process_audio_file(input_path, output_path, start_time_sec, token_length=5, db_level=-20, frame_rate=50):
    audio_data, sample_rate = sf.read(input_path)
    modified_audio = add_noise_at_time(audio_data, sample_rate, start_time_sec, token_length, db_level, frame_rate)
    sf.write(output_path, modified_audio, sample_rate)

def process_directory(input_dir, output_dir, start_time_sec, token_length=5, db_level=-20, frame_rate=50):
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.wav') or filename.lower().endswith('.mp3'):
                # Get the relative path from input_dir to maintain folder structure
                rel_path = os.path.relpath(root, input_dir)
                
                input_path = os.path.join(root, filename)
                
                # Create corresponding output directory structure
                if rel_path == '.':
                    output_subdir = output_dir
                else:
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                
                output_path = os.path.join(output_subdir, filename)
                process_audio_file(input_path, output_path, start_time_sec, token_length, db_level, frame_rate)
                print(f"Processed: {input_path} -> {output_path}")

def process_all_time_intervals(input_dir, base_output_dir, token_length=5, db_level=-20, frame_rate=50):
    """Process audio files for all time intervals from 1 to 14 seconds"""
    for sec in range(1, 15):  # 1 to 14 seconds
        output_dir = f"{base_output_dir}_add_noise_at_{sec}sec_tk{token_length}"
        print(f"\nProcessing noise at {sec} second(s) -> {output_dir}")
        print("=" * 60)
        process_directory(input_dir, output_dir, sec, token_length, db_level, frame_rate)
        print(f"Completed processing for {sec} second(s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add noise at different time intervals (1-14 seconds) and store in separate folders")
    parser.add_argument("--input", required=True, help="Input directory (e.g., asap_silent_sele_ori)")
    parser.add_argument("--output_base", required=True, help="Base output directory name (e.g., asap)")
    parser.add_argument("--tokens", type=int, default=20, help="Token length (default: 20)")
    parser.add_argument("--db", type=float, default=-12, help="Noise level in dB (default: -12)")
    parser.add_argument("--frame_rate", type=int, default=50, help="Model frame rate (tokens per second, default: 50)")
    parser.add_argument("--single_time", type=int, help="Process only a single time interval (1-14 seconds)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        exit(1)
    
    if args.single_time:
        if 1 <= args.single_time <= 14:
            output_dir = f"{args.output_base}_add_noise_at_{args.single_time}sec_tk{args.tokens}"
            print(f"Processing noise at {args.single_time} second(s) -> {output_dir}")
            process_directory(args.input, output_dir, args.single_time, args.tokens, args.db, args.frame_rate)
        else:
            print("Error: --single_time must be between 1 and 14")
            exit(1)
    else:
        # Process all time intervals from 1 to 14 seconds
        process_all_time_intervals(args.input, args.output_base, args.tokens, args.db, args.frame_rate)
        print("\nAll processing completed!")
        print(f"Created 14 folders: {args.output_base}_add_noise_at_1sec_tk{args.tokens} to {args.output_base}_add_noise_at_14sec_tk{args.tokens}")