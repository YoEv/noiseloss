import numpy as np
import soundfile as sf
import argparse
import os
from scipy import signal

# noise sample rate 32000

def generate_white_noise(length, db_level=-20, seed=42):
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.randn(length)
    white_noise = white_noise / np.std(white_noise)
    scaling_factor = 10 ** (db_level / 20)
    return white_noise * scaling_factor

def generate_pink_noise(length, db_level=-20, seed=42):
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.normal(0, 1, length)
    
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)
    
    if np.std(pink_noise) > 0:
        pink_noise = pink_noise / np.std(pink_noise)
    
    scaling_factor = 10 ** (db_level / 20)
    return pink_noise * scaling_factor

def generate_brown_noise(length, db_level=-20, seed=42):
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.normal(0, 1, length)
    
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)
    
    if np.std(brown_noise) > 0:
        brown_noise = brown_noise / np.std(brown_noise)
    
    scaling_factor = 10 ** (db_level / 20)
    return brown_noise * scaling_factor

def generate_blue_noise(length, db_level=-20, seed=42):
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.normal(0, 1, length)
    
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)
    
    if np.std(blue_noise) > 0:
        blue_noise = blue_noise / np.std(blue_noise)
    
    scaling_factor = 10 ** (db_level / 20)
    return blue_noise * scaling_factor

def generate_noise_files(output_dir, duration=15, sample_rate=32000, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    noise_generators = {
        'white': generate_white_noise,
        'pink': generate_pink_noise,
        'brown': generate_brown_noise,
        'blue': generate_blue_noise
    }
    
    db_levels = [-12, -20, -30]
    
    length = int(duration * sample_rate)
    
    print(f"Generating {len(noise_generators)} types of noise at {len(db_levels)} dB levels...")
    print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz, Seed: {seed}")
    
    for noise_type, generator_func in noise_generators.items():
        for db_level in db_levels:
            noise_data = generator_func(length, db_level, seed)
            
            filename = f"{noise_type}_noise_{abs(db_level)}db_{duration}s.wav"
            filepath = os.path.join(output_dir, filename)
            
            sf.write(filepath, noise_data, sample_rate)
            print(f"Generated: {filename}")
    
    print(f"\nAll noise files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate 15-second colored noise files at different dB levels")
    parser.add_argument("--output", "-o", required=True, help="Output directory for noise files")
    parser.add_argument("--duration", "-d", type=float, default=15.0, help="Duration in seconds (default: 15)")
    parser.add_argument("--sample_rate", "-sr", type=int, default=32000, help="Sample rate (default: 32000)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    generate_noise_files(
        output_dir=args.output,
        duration=args.duration,
        sample_rate=args.sample_rate,
        seed=args.seed
    )

if __name__ == "__main__":
    main()