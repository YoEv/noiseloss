import numpy as np
import soundfile as sf
import argparse
import os

def generate_white_noise(length, db_level=-20, seed=42):
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.randn(length)
    white_noise = white_noise / np.std(white_noise)
    scaling_factor = 10 ** (db_level / 20)
    return white_noise * scaling_factor

def add_noise_at_20s(audio_data, sample_rate, token_length=5, db_level=-20, frame_rate=50):
    start_sample = int(20 * sample_rate)  # 改为20秒
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

def process_audio_file(input_path, output_path, token_length=5, db_level=-20, frame_rate=50):
    audio_data, sample_rate = sf.read(input_path)
    modified_audio = add_noise_at_20s(audio_data, sample_rate, token_length, db_level, frame_rate)
    sf.write(output_path, modified_audio, sample_rate)

def process_directory(input_dir, output_dir, token_length=5, db_level=-20, frame_rate=50):
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
                process_audio_file(input_path, output_path, token_length, db_level, frame_rate)
                print(f"Processed: {input_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory")
    parser.add_argument("--tokens", type=int, default=20, help="Token length (default: 20)")
    parser.add_argument("--db", type=float, default=-12, help="Noise level in dB (default: -12)")
    parser.add_argument("--frame_rate", type=int, default=50, help="Model frame rate (tokens per second, default: 50)")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        process_audio_file(args.input, args.output, args.tokens, args.db, args.frame_rate)
    elif os.path.isdir(args.input):
        process_directory(args.input, args.output, args.tokens, args.db, args.frame_rate)