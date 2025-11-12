import os
import numpy as np
import soundfile as sf
from scipy import signal

def generate_white_noise(length, db_level=0):
    """
    Generate white noise.

    Parameters:
    - length: number of samples in the noise
    - db_level: desired loudness level (in dB)
    """
    noise = np.random.normal(0, 0.1, length)
    # Normalize to 0 dB
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))

    # Apply dB gain
    gain = 10 ** (db_level / 20)  # Convert dB to linear gain
    noise = noise * gain

    return noise

def generate_pink_noise(length, db_level=0):
    """
    Generate pink noise.

    Parameters:
    - length: number of samples in the noise
    - db_level: desired loudness level (in dB)
    """
    # Generate white noise
    white_noise = np.random.normal(0, 0.1, length)

    # Apply pink noise filter
    # Pink noise has a power spectral density proportional to 1/f
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)

    # Normalize to 0 dB
    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))

    # Apply dB gain
    gain = 10 ** (db_level / 20)  # Convert dB to linear gain
    pink_noise = pink_noise * gain

    return pink_noise

def generate_brown_noise(length, db_level=0):
    """
    Generate brown noise.

    Parameters:
    - length: number of samples in the noise
    - db_level: desired loudness level (in dB)
    """
    # Generate white noise
    white_noise = np.random.normal(0, 0.1, length)

    # Apply brown noise filter
    # Brown noise has a power spectral density proportional to 1/f²
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)

    # Normalize to 0 dB
    if np.max(np.abs(brown_noise)) > 0:
        brown_noise = brown_noise / np.max(np.abs(brown_noise))

    # Apply dB gain
    gain = 10 ** (db_level / 20)  # Convert dB to linear gain
    brown_noise = brown_noise * gain

    return brown_noise

def generate_blue_noise(length, db_level=0):
    """
    Generate blue noise.

    Parameters:
    - length: number of samples in the noise
    - db_level: desired loudness level (in dB)
    """
    # Generate white noise
    white_noise = np.random.normal(0, 0.1, length)

    # Apply blue noise filter
    # Blue noise has a power spectral density proportional to f
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)

    # Normalize to 0 dB
    if np.max(np.abs(blue_noise)) > 0:
        blue_noise = blue_noise / np.max(np.abs(blue_noise))

    # Apply dB gain
    gain = 10 ** (db_level / 20)  # Convert dB to linear gain
    blue_noise = blue_noise * gain

    return blue_noise

def generate_pure_noise_files(duration=14, sample_rate=32000, db_level=6, output_dir="pure_noise_6db"):
    """
    Generate and save four types of pure noise audio files.

    Parameters:
    - duration: audio length in seconds
    - sample_rate: sampling rate
    - db_level: desired loudness level (in dB)
    - output_dir: output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total number of samples
    num_samples = int(duration * sample_rate)
    
    # Define noise types and their corresponding functions
    noise_types = {
        'white': generate_white_noise,
        'pink': generate_pink_noise,
        'brown': generate_brown_noise,
        'blue': generate_blue_noise
    }
    
    print(f"Generating {duration}-second pure noise files at {db_level} dB...")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Output directory: {output_dir}")
    
    # Generate each type of noise
    for noise_name, noise_func in noise_types.items():
        print(f"Generating {noise_name} noise...")
        
        # Generate noise
        noise_data = noise_func(num_samples, db_level)
        
        # Prevent clipping
        max_val = np.max(np.abs(noise_data))
        if max_val > 0.95:
            noise_data = noise_data / max_val * 0.95
        
        # Generate filename
        filename = f"{noise_name}_noise_{duration}s_{db_level}db.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save audio file
        sf.write(filepath, noise_data, sample_rate)
        
        print(f"  Saved: {filepath}")
        print(f"  Samples: {len(noise_data)}")
        print(f"  Max amplitude: {np.max(np.abs(noise_data)):.4f}")
        print()
    
    print("All noise files have been generated successfully!")
    print("\nGenerated files:")
    for noise_name in noise_types.keys():
        filename = f"{noise_name}_noise_{duration}s_{db_level}db.wav"
        print(f"  - {filename}")

def main():
    """
    Main function: generate four types of pure noise files (14s, 6dB)
    """
    # Check required dependencies
    try:
        import soundfile
        from scipy import signal
        print("All required libraries are installed.")
    except ImportError as e:
        print(f"Error: Missing required library! {str(e)}")
        print("Please install them with:")
        print("pip install numpy soundfile scipy")
        return
    
    # Generate noise files
    generate_pure_noise_files(
        duration=14,        # 14 seconds
        sample_rate=32000,  # 32 kHz sampling rate
        db_level=6,         # 6 dB loudness
        output_dir="pure_noise_6db_14s"  # Output directory
    )

if __name__ == "__main__":
    main()
