#!/usr/bin/env python3

import os
import random
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cut music into small segments and reorder them randomly with different percentages.')
    parser.add_argument('--input_dir', type=str, default='/home/evev/asap-dataset/ShutterStock_20_cut14',
                        help='Directory containing input music files')
    parser.add_argument('--output_dir', type=str, default='/home/evev/asap-dataset/ShutterStock_20_reordered_section',
                        help='Directory to save reordered music files')
    parser.add_argument('--min_segment_length', type=float, default=3,
                        help='Minimum length of each segment in seconds')
    parser.add_argument('--max_segment_length', type=float, default=5,
                        help='Maximum length of each segment in seconds')
    parser.add_argument('--shuffle_percentages', nargs='+', type=float, 
                        default=[0.05, 0.10, 0.30, 0.50, 0.80],
                        help='Percentages of music to shuffle (default: 0.05 0.10 0.30 0.50 0.80)')
    parser.add_argument('--file_extension', type=str, default='.mp3',
                        help='File extension of audio files to process')
    return parser.parse_args()

# First, ensure there's only one duration function
# At the top of the file, rename get_duration to get_audio_duration for consistency
def get_audio_duration(file_path):
    """Get the duration of an audio file using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        print(f"Error getting duration: {result.stderr}")
        return 0
    return float(result.stdout.strip())

def cut_audio_into_variable_segments(input_file, temp_dir, min_segment_length, max_segment_length):
    """Cut audio into segments with accurate timing and validation"""
    duration = get_audio_duration(input_file)  # Changed from get_duration to get_audio_duration
    if duration <= 0:
        print(f"Error: Invalid duration {duration} for file {input_file}")
        return []
        
    segments = []
    current_time = 0
    segment_index = 0
    
    # Create a progress bar for segment creation
    pbar = tqdm(total=int(duration), desc="Creating segments", unit="sec")
    
    while current_time < duration:
        # Random segment length between min and max
        segment_length = random.uniform(min_segment_length, max_segment_length)
        end_time = min(current_time + segment_length, duration)
        actual_length = end_time - current_time
        
        # Skip segments that are too short
        if actual_length < 0.001:  # Less than 1ms
            break
            
        output_segment = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp3")
        
        # Use more robust FFmpeg command with re-encoding to ensure valid output
        cmd = [
            'ffmpeg', '-y', '-ss', str(current_time),
            '-i', input_file, '-t', str(actual_length),
            '-acodec', 'mp3', '-ab', '128k',  # Re-encode to ensure valid MP3
            '-ar', '44100',  # Standard sample rate
            '-ac', '2',      # Stereo
            output_segment
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error cutting segment {segment_index}: {result.stderr.decode()}")
            current_time = end_time
            segment_index += 1
            pbar.update(int(actual_length))
            continue
            
        # Validate the created segment
        if os.path.exists(output_segment):
            file_size = os.path.getsize(output_segment)
            if file_size > 1000:  # At least 1KB
                # Double-check with ffprobe that it's a valid audio file
                probe_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', output_segment
                ]
                probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if probe_result.returncode == 0:
                    try:
                        segment_duration = float(probe_result.stdout.strip())
                        if segment_duration > 0:
                            segments.append(output_segment)
                            # Removed print statement about created valid segment
                        else:
                            print(f"Warning: Segment {segment_index} has zero duration, skipping")
                            os.remove(output_segment)
                    except ValueError:
                        print(f"Warning: Could not parse duration for segment {segment_index}, skipping")
                        os.remove(output_segment)
                else:
                    print(f"Warning: Segment {segment_index} failed validation, skipping")
                    os.remove(output_segment)
            else:
                print(f"Warning: Segment {segment_index} is too small ({file_size} bytes), skipping")
                os.remove(output_segment)
        else:
            print(f"Warning: Segment {segment_index} was not created")
            
        current_time = end_time
        segment_index += 1
        pbar.update(int(actual_length))
    
    pbar.close()
    print(f"Successfully created {len(segments)} valid segments out of {segment_index} attempts")
    return segments

def shuffle_segments_by_percentage(segments, shuffle_percentage):
    """Shuffle segments by replacing originals at selected positions"""
    total = len(segments)
    num_shuffle = int(total * shuffle_percentage)
    
    if num_shuffle < 1:
        return segments.copy()

    # Create working copy
    shuffled = segments.copy()
    
    # 1. Select positions to replace
    positions = random.sample(range(total), num_shuffle)
    
    # 2. Extract segments at these positions
    selected = [segments[i] for i in positions]
    
    # 3. Shuffle the selected segments
    random.shuffle(selected)
    
    # 4. Replace original positions with shuffled segments
    for i, pos in enumerate(positions):
        shuffled[pos] = selected[i]
    
    return shuffled

# In join_segments function, add duration validation:
def join_segments(concat_file, output_file):
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file, '-c', 'copy', output_file
    ]
    subprocess.run(cmd, check=True)
    
    # Verify output duration
    orig_duration = get_audio_duration(segments[0])  # Get from first segment
    new_duration = get_audio_duration(output_file)
    
    if abs(new_duration - orig_duration) > 0.5:
        raise ValueError(f"Duration mismatch: {new_duration:.1f}s vs original {orig_duration:.1f}s")

def create_concat_file(segments, concat_file):
    """Create a concat file for ffmpeg with validation"""
    valid_segments = []
    
    # First validate all segments
    for segment in segments:
        if os.path.exists(segment):
            file_size = os.path.getsize(segment)
            if file_size > 1000:  # At least 1KB
                # Quick validation with ffprobe
                probe_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', segment
                ]
                result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    valid_segments.append(segment)
                else:
                    print(f"Warning: Skipping invalid segment: {segment}")
            else:
                print(f"Warning: Skipping small segment ({file_size} bytes): {segment}")
        else:
            print(f"Warning: Segment file does not exist: {segment}")
    
    if not valid_segments:
        print("Error: No valid segments found for concatenation")
        return False
        
    try:
        with open(concat_file, 'w') as f:
            # Each segment should appear exactly once in the concat file
            for segment in valid_segments:
                # Use absolute path and escape single quotes
                abs_path = os.path.abspath(segment).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")
        print(f"Created concat file with {len(valid_segments)} valid segments")
        return True
    except Exception as e:
        print(f"Error creating concat file: {e}")
        return False

def join_segments(concat_file, output_file, original_file):
    """Join audio segments using the concat file"""
    if not os.path.exists(concat_file):
        print(f"Error: Concat file does not exist: {concat_file}")
        return False
        
    # Modified FFmpeg command with additional parameters to prevent silence
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file, 
        '-c:a', 'mp3', '-b:a', '128k',  # Re-encode to ensure consistent output
        '-af', 'apad=whole_dur=0',  # Prevent padding with silence
        output_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error joining segments: {result.stderr.decode()}")
        return False
    
    # Validate output file
    if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
        # Verify output duration against original file
        orig_duration = get_audio_duration(original_file)
        new_duration = get_audio_duration(output_file)
        
        if abs(new_duration - orig_duration) > 0.5:
            print(f"Warning: Duration mismatch: {new_duration:.1f}s vs original {orig_duration:.1f}s")
            return False
            
        print(f"Successfully created output: {output_file}")
        return True
    else:
        print(f"Error: Output file is invalid or too small: {output_file}")
        return False

def process_file(input_file, output_dir, min_segment_length, max_segment_length, shuffle_percentages):
    """Process a single audio file with different shuffle percentages"""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Processing {os.path.basename(input_file)}...")
        
        # Cut into variable-length segments
        segments = cut_audio_into_variable_segments(input_file, temp_dir, min_segment_length, max_segment_length)
        if not segments:
            print(f"Error: No valid segments created for {input_file}")
            return
            
        print(f"Created {len(segments)} valid segments")
        
        # Process each shuffle percentage
        for shuffle_percentage in shuffle_percentages:
            print(f"  Processing {shuffle_percentage*100}% shuffle...")
            
            # Shuffle segments by percentage using the correct logic
            shuffled_segments = shuffle_segments_by_percentage(segments, shuffle_percentage)
            
            # Create output filename
            percentage_str = f"{int(shuffle_percentage*100)}p"
            output_file = os.path.join(output_dir, f"{base_name}_reordered_{percentage_str}.mp3")
            
            # Create concat file and join segments
            concat_file = os.path.join(temp_dir, f"concat_{percentage_str}.txt")
            
            if create_concat_file(shuffled_segments, concat_file):
                if join_segments(concat_file, output_file, input_file):  # Pass original file
                    print(f"    Successfully saved to {output_file}")
                else:
                    print(f"    Failed to create {output_file}")
            else:
                print(f"    Failed to create concat file for {percentage_str}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate arguments
    if args.min_segment_length >= args.max_segment_length:
        print("Error: Minimum segment length must be less than maximum segment length!")
        return
    
    if not all(0 <= p <= 1 for p in args.shuffle_percentages):
        print("Error: Shuffle percentages must be between 0 and 1!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Segment length range: {args.min_segment_length}s - {args.max_segment_length}s")
    print(f"Shuffle percentages: {[f'{p*100}%' for p in args.shuffle_percentages]}")
    
    # Get all audio files in the input directory
    input_files = [f for f in os.listdir(args.input_dir) 
                  if f.endswith(args.file_extension)]
    
    if not input_files:
        print(f"No {args.file_extension} files found in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file with a progress bar
    for input_file in tqdm(input_files, desc="Processing files", unit="file"):
        input_path = os.path.join(args.input_dir, input_file)
        process_file(input_path, args.output_dir, args.min_segment_length, 
                    args.max_segment_length, args.shuffle_percentages)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()