import os
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import argparse
from pathlib import Path
import gc  # For garbage collection

def change_volume_of_segments(audio_data, sr, segment_range, selection_percentage, min_segment_len=None, max_segment_len=None):
    """
    Change the volume of randomly selected segments in the audio file
    
    Parameters:
    - audio_data: Audio data array
    - sr: Sample rate
    - segment_range: Tuple of (min_segment_len, max_segment_len) in seconds
    - selection_percentage: Percentage of audio to modify (0-1)
    - min_segment_len: Override for minimum segment length (seconds)
    - max_segment_len: Override for maximum segment length (seconds)
    
    Returns:
    - Modified audio data
    """
    # Create a copy of the audio data
    modified_audio = np.copy(audio_data)
    
    # Use provided segment range or override with min/max parameters
    min_segment_len = min_segment_len if min_segment_len is not None else segment_range[0]
    max_segment_len = max_segment_len if max_segment_len is not None else segment_range[1]
    
    # Check if audio is stereo
    is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] > 1
    
    # Calculate total audio duration in seconds
    total_duration = len(audio_data) / sr
    
    # Calculate total duration to modify based on selection percentage
    modification_duration = total_duration * selection_percentage
    
    # Track remaining duration to modify and positions of modifications
    remaining_duration = modification_duration
    
    # Maximum attempts to prevent infinite loops with high percentages
    max_attempts = 1000
    attempts = 0
    
    # Use a more efficient data structure for tracking modified segments
    # Instead of storing all positions, we'll use a boolean array to mark modified samples
    modified_samples = np.zeros(len(audio_data), dtype=bool)
    
    # Generate non-overlapping modification positions
    while remaining_duration > 0 and attempts < max_attempts:
        attempts += 1
        
        # Determine current segment length (between min and max, not exceeding remaining duration)
        segment_len = min(random.uniform(min_segment_len, max_segment_len), remaining_duration)
        
        # Random start position (in seconds)
        max_start = total_duration - segment_len
        if max_start <= 0:
            break
            
        start_time = random.uniform(0, max_start)
        
        # Convert to sample indices
        start_idx = int(start_time * sr)
        end_idx = int((start_time + segment_len) * sr)
        
        # Check if this segment overlaps with any existing modified segments
        if np.any(modified_samples[start_idx:end_idx]):
            continue
        
        # Mark this segment as modified
        modified_samples[start_idx:end_idx] = True
        
        # Randomly choose between low volume (0.02-0.1) or high volume (3.0-6.0)
        if random.choice([True, False]):
            # Low volume range
            volume_factor = random.uniform(0.02, 0.1)
        else:
            # High volume range
            volume_factor = random.uniform(3.0, 6.0)
        
        # Apply volume change
        if is_stereo:
            modified_audio[start_idx:end_idx, :] = modified_audio[start_idx:end_idx, :] * volume_factor
        else:
            modified_audio[start_idx:end_idx] = modified_audio[start_idx:end_idx] * volume_factor
        
        remaining_duration -= segment_len
    
    # Clean up
    del modified_samples
    
    return modified_audio

def process_audio_file(file_path, output_dir, segment_configs):
    """
    Process a single audio file, changing volume for different segment ranges
    """
    try:
        # Read audio file
        audio_data, sr = sf.read(file_path)
        
        # Ensure sample rate is integer type (required by soundfile.write)
        sr = int(sr)
        
        # Get filename and extension
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        
        # Process each segment configuration one at a time
        for segment_range, percentage, suffix in segment_configs:
            # Change volume of segments
            processed_audio = change_volume_of_segments(
                audio_data, sr, segment_range, percentage)
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")
            
            # Save processed audio
            sf.write(output_file, processed_audio, sr)
            
            # Clean up to free memory
            del processed_audio
            gc.collect()
            
        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(input_dir, output_base_dir, segment_ranges, selection_percentages):
    """
    Process all audio files in the directory
    """
    # Generate segment configurations
    segment_configs = []
    for segment_range in segment_ranges:
        for percentage in selection_percentages:
            # Create descriptive suffix
            segment_desc = f"dyn{int(segment_range[0]*100)}-{int(segment_range[1]*100)}"
            perc_desc = f"{int(percentage*100)}p"
            suffix = f"_{segment_desc}_{perc_desc}_rndvol"
            segment_configs.append((segment_range, percentage, suffix))
    
    print(f"Will generate {len(segment_configs)} configurations:")
    for segment_range, percentage, suffix in segment_configs:
        print(f"  - Segment range: {segment_range[0]}-{segment_range[1]}s, Selection: {percentage*100}%, Random volume factor: either 0.02-0.1 or 3.0-6.0, Suffix: {suffix}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Check if subdirectories were found
    if not subdirs:
        print(f"Warning: No subdirectories found in '{input_dir}'!")
        print("Attempting to process audio files directly in the input directory...")
        
        # Get all audio files in the input directory
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            audio_files.extend(list(Path(input_dir).glob(f"*{ext}")))
        
        if not audio_files:
            print(f"Error: No audio files found in '{input_dir}'!")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            process_audio_file(str(audio_file), output_base_dir, segment_configs)
            # Force garbage collection after each file
            gc.collect()
        
        return
    
    print(f"Found {len(subdirs)} subdirectories in '{input_dir}'")
    
    # Process each subdirectory
    for subdir in tqdm(subdirs, desc="Processing directories"):
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_base_dir, subdir)
        
        # Create output subdirectory
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            found_files = list(Path(input_subdir).glob(f"*{ext}"))
            audio_files.extend(found_files)
        
        # Check if audio files were found
        if not audio_files:
            print(f"Warning: No audio files found in subdirectory '{subdir}'!")
            continue
        
        print(f"Found {len(audio_files)} audio files in subdirectory '{subdir}'")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=f"Processing files in {subdir}", leave=False):
            process_audio_file(str(audio_file), output_subdir, segment_configs)
            # Force garbage collection after each file
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Change volume of audio segments")
    parser.add_argument("--input_dir", default="ShutterStock_20_cut14",
                        help="Input directory containing audio files (default: ShutterStock_20_cut14)")
    parser.add_argument("--output_dir", default="Shutter_dyn_rnd",
                        help="Output directory (default: Shutter_dyn_rnd)")
    parser.add_argument("--segment_ranges", nargs="+", type=str,
                        default=["0.01-0.05", "0.1-0.5", "1-3", "3-5"],
                        help="Segment ranges in seconds, format: 'min-max' (default: 0.01-0.05 0.1-0.5 1-3 3-5)")
    parser.add_argument("--selection_percentages", nargs="+", type=float,
                        default=[0.05, 0.10, 0.30, 0.50, 0.80],
                        help="Selection percentages (0-1, default: 0.05 0.10 0.30 0.50 0.80)")
    
    args = parser.parse_args()
    
    # Validate parameters
    if not all(0 <= p <= 1 for p in args.selection_percentages):
        print("Error: Selection percentages must be between 0 and 1!")
        return
    
    # Parse segment ranges
    segment_ranges = []
    for range_str in args.segment_ranges:
        try:
            min_val, max_val = map(float, range_str.split('-'))
            if min_val >= max_val:
                print(f"Error: Invalid segment range '{range_str}': minimum must be less than maximum!")
                return
            segment_ranges.append((min_val, max_val))
        except ValueError:
            print(f"Error: Invalid segment range format '{range_str}'. Use 'min-max' (e.g., '0.01-0.05')")
            return
    
    # Get absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Starting audio processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Segment ranges: {segment_ranges}")
    print(f"Selection percentages: {[f'{p*100}%' for p in args.selection_percentages]}")
    print(f"Volume factor: Random between either 0.02-0.1 or 3.0-6.0 for each segment")
    
    # Process all audio files
    process_directory(input_dir, output_dir, segment_ranges, args.selection_percentages)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()