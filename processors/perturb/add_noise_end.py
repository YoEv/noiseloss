import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal

def generate_white_noise(length):
    """
    Generate white noise
    """
    noise = np.random.normal(0, 0.1, length)
    # Normalize to 0dB
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    return noise

def generate_pink_noise(length):
    """
    Generate pink noise
    """
    # Generate white noise
    white_noise = np.random.normal(0, 0.1, length)
    
    # Apply pink filter
    # Pink noise power spectral density is inversely proportional to frequency (1/f)
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)
    
    # Normalize to 0dB
    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
    
    return pink_noise

def generate_brown_noise(length):
    """
    Generate brown noise
    """
    # Generate white noise
    white_noise = np.random.normal(0, 0.1, length)
    
    # Apply brown filter
    # Brown noise power spectral density is inversely proportional to frequency squared (1/f^2)
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)
    
    # Normalize to 0dB
    if np.max(np.abs(brown_noise)) > 0:
        brown_noise = brown_noise / np.max(np.abs(brown_noise))
    
    return brown_noise

def add_gradual_noise(input_file, output_file, noise_duration=5.0, noise_type='white'):
    """
    Add gradually increasing noise to an audio file.
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to output audio file
        noise_duration (float): Duration in seconds for the noise transition
        noise_type (str): Type of noise ('white', 'pink', 'brown')
    """
    # Load audio file
    audio, sr = librosa.load(input_file, sr=None)
    duration = len(audio) / sr
    
    print(f"Processing {os.path.basename(input_file)} - Duration: {duration:.2f}s")
    
    # Calculate noise transition parameters
    noise_start_time = max(0, duration - noise_duration)
    noise_start_sample = int(noise_start_time * sr)
    
    # Generate noise using proper algorithms
    if noise_type == 'white':
        noise = generate_white_noise(len(audio))
    elif noise_type == 'pink':
        noise = generate_pink_noise(len(audio))
    elif noise_type == 'brown':
        noise = generate_brown_noise(len(audio))
    else:
        noise = generate_white_noise(len(audio))
    
    # Normalize noise to match audio amplitude
    audio_rms = np.sqrt(np.mean(audio**2))
    noise = noise * audio_rms * 0.5  # Scale noise to be reasonable
    
    # Create the mixed audio
    mixed_audio = audio.copy()
    
    # Apply gradual transition from original audio to pure noise
    for i in range(len(audio)):
        time_position = i / sr
        
        if time_position < noise_start_time:
            # Pure original audio
            continue
        else:
            # Calculate transition factor (0 = pure audio, 1 = pure noise)
            transition_progress = (time_position - noise_start_time) / noise_duration
            transition_progress = min(1.0, transition_progress)
            
            # Apply smooth transition using cosine interpolation
            smooth_factor = 0.5 * (1 - np.cos(np.pi * transition_progress))
            
            # Mix audio and noise
            mixed_audio[i] = audio[i] * (1 - smooth_factor) + noise[i] * smooth_factor
    
    # Save the processed audio
    sf.write(output_file, mixed_audio, sr)
    print(f"Saved: {output_file}")

def process_directory(input_dir, output_dir, noise_duration=5.0, noise_type='white'):
    """
    Process all audio files in a directory.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        noise_duration (float): Duration in seconds for the noise transition
        noise_type (str): Type of noise to add
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported audio extensions
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
    
    # Process each audio file
    audio_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    print(f"Noise transition duration: {noise_duration} seconds")
    print(f"Noise type: {noise_type}")
    print("-" * 50)
    
    for audio_file in audio_files:
        # Create output filename
        output_filename = f"{audio_file.stem}_noise_end_{noise_type}{audio_file.suffix}"
        output_file = output_path / output_filename
        
        try:
            add_gradual_noise(str(audio_file), str(output_file), 
                            noise_duration, noise_type)
        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
    
    print("-" * 50)
    print(f"Processing complete. Output files saved to: {output_dir}")

def main():
    """
    Main function to run the noise addition process.
    """
    # Configuration
    input_directory = "ShutterStock_20_cut14"
    output_directory = "ShutterStock_20_cut14_noise_end"
    noise_duration = 10.0  # seconds
    noise_type = 'brown'  # 'white', 'pink', or 'brown'
    
    print("Audio Noise Addition Tool")
    print("=" * 30)
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Noise transition duration: {noise_duration} seconds")
    print(f"Noise type: {noise_type}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
        print("Please make sure the directory exists and try again.")
        return
    
    # Process the directory
    try:
        process_directory(input_directory, output_directory, 
                        noise_duration, noise_type)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()