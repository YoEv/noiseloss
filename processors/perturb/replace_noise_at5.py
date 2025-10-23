import numpy as np
import soundfile as sf
import argparse
import os
from scipy import signal

def generate_white_noise(length, db_level=-20, seed=42):
    """Generate white noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.randn(length)
    white_noise = white_noise / np.std(white_noise)
    scaling_factor = 10 ** (db_level / 20)
    return white_noise * scaling_factor

def generate_pink_noise(length, db_level=-20, seed=42):
    """Generate pink noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    # 生成白噪音
    white_noise = np.random.normal(0, 1, length)
    
    # 应用粉红滤波器 (1/f)
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)
    
    # 归一化
    if np.std(pink_noise) > 0:
        pink_noise = pink_noise / np.std(pink_noise)
    
    # 应用dB增益
    scaling_factor = 10 ** (db_level / 20)
    return pink_noise * scaling_factor

def generate_brown_noise(length, db_level=-20, seed=42):
    """Generate brown noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    # 生成白噪音
    white_noise = np.random.normal(0, 1, length)
    
    # 应用布朗滤波器 (1/f^2)
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)
    
    # 归一化
    if np.std(brown_noise) > 0:
        brown_noise = brown_noise / np.std(brown_noise)
    
    # 应用dB增益
    scaling_factor = 10 ** (db_level / 20)
    return brown_noise * scaling_factor

def generate_blue_noise(length, db_level=-20, seed=42):
    """Generate blue noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    # 生成白噪音
    white_noise = np.random.normal(0, 1, length)
    
    # 应用蓝噪音滤波器 (f)
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)
    
    # 归一化
    if np.std(blue_noise) > 0:
        blue_noise = blue_noise / np.std(blue_noise)
    
    # 应用dB增益
    scaling_factor = 10 ** (db_level / 20)
    return blue_noise * scaling_factor

def generate_noise(noise_type, length, db_level=-20, seed=42):
    """Generate noise of specified type"""
    noise_generators = {
        'white': generate_white_noise,
        'pink': generate_pink_noise,
        'brown': generate_brown_noise,
        'blue': generate_blue_noise
    }
    
    if noise_type not in noise_generators:
        raise ValueError(f"Unsupported noise type: {noise_type}. Supported types: {list(noise_generators.keys())}")
    
    return noise_generators[noise_type](length, db_level, seed)

def replace_with_noise_at_5s(audio_data, sample_rate, token_length=5, db_level=-20, frame_rate=50, seed=42, noise_type='white'):
    """Replace the token segment starting at 5 seconds with specified noise type"""
    start_sample = int(5 * sample_rate)
    # Correct token duration calculation: each token = 1/frame_rate seconds
    single_token_duration = 1.0 / frame_rate
    duration_samples = int(token_length * single_token_duration * sample_rate)
    
    if start_sample >= len(audio_data):
        return audio_data
    
    end_sample = min(start_sample + duration_samples, len(audio_data))
    actual_duration = end_sample - start_sample
    
    if len(audio_data.shape) == 1:
        # Mono audio
        noise = generate_noise(noise_type, actual_duration, db_level, seed)
        audio_data[start_sample:end_sample] = noise  # Replace instead of add
    else:
        # Stereo/multi-channel audio
        for channel in range(audio_data.shape[1]):
            # Use different seed for each channel to avoid identical noise
            channel_seed = seed + channel if seed is not None else None
            noise = generate_noise(noise_type, actual_duration, db_level, channel_seed)
            audio_data[start_sample:end_sample, channel] = noise  # Replace instead of add
    
    return audio_data

def process_audio_file(input_path, output_path, token_length=5, db_level=-20, frame_rate=50, seed=42, noise_type='white'):
    """Process a single audio file"""
    audio_data, sample_rate = sf.read(input_path)
    modified_audio = replace_with_noise_at_5s(audio_data, sample_rate, token_length, db_level, frame_rate, seed, noise_type)
    sf.write(output_path, modified_audio, sample_rate)

def process_directory(input_dir, output_dir, token_length=5, db_level=-20, frame_rate=50, seed=42, noise_type='white'):
    """Process all audio files in a directory recursively"""
    os.makedirs(output_dir, exist_ok=True)
    
    file_counter = 0
    
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
                
                # Use different seed for each file to ensure variety while maintaining reproducibility
                file_seed = seed + file_counter if seed is not None else None
                process_audio_file(input_path, output_path, token_length, db_level, frame_rate, file_seed, noise_type)
                print(f"Processed: {input_path} -> {output_path}")
                file_counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace token segments in audio files with specified noise type")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output file or directory")
    parser.add_argument("--tokens", type=int, default=20, help="Token length (default: 20)")
    parser.add_argument("--db", type=float, default=-12, choices=[-30, -20, -12], help="Noise level in dB (choices: -30, -20, -12, default: -12)")
    parser.add_argument("--frame_rate", type=int, default=50, help="Model frame rate (tokens per second, default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible noise generation (default: 42)")
    parser.add_argument("--noise_type", type=str, default='white', choices=['white', 'pink', 'brown', 'blue'], help="Type of noise to generate (choices: white, pink, brown, blue, default: white)")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        process_audio_file(args.input, args.output, args.tokens, args.db, args.frame_rate, args.seed, args.noise_type)
    elif os.path.isdir(args.input):
        process_directory(args.input, args.output, args.tokens, args.db, args.frame_rate, args.seed, args.noise_type)
    else:
        print(f"Error: Input path '{args.input}' is neither a file nor a directory.")