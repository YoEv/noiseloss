#!/usr/bin/env python3

import os
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import math

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Randomly shuffle audio tokens with 5% shuffle ratio.')
    parser.add_argument('--input_dir', type=str, default='/home/evev/asap-dataset/ShutterStock_32k_ori',
                        help='Directory containing input music files')
    parser.add_argument('--output_dir', type=str, default='/home/evev/asap-dataset/shutter_shuffle_random',
                        help='Directory to save shuffled music files')
    parser.add_argument('--shuffle_start_time', type=float, default=0.0,
                        help='Start time for shuffling in seconds (default: 0.0)')
    parser.add_argument('--shuffle_end_time', type=float, default=14.0,
                        help='End time for shuffling in seconds (default: 14.0)')
    parser.add_argument('--token_lengths', nargs='+', type=int, 
                        default=[2, 5, 10, 35, 50, 70],
                        help='Token lengths to shuffle (default:2 5 10 35 50 70)')
    parser.add_argument('--shuffle_percentage', type=float, default=0.2,
                        help='Percentage of shuffle range to actually shuffle (default: 0.2 = 20%)')
    parser.add_argument('--file_extension', type=str, default='.wav',
                        help='File extension of audio files to process')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Sample rate for audio processing (default: 32000)')
    parser.add_argument('--tokens_per_second', type=float, default=50.0,
                        help='Number of tokens per second (default: 50.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    return parser.parse_args()

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

def convert_time_to_token_position(time_seconds, tokens_per_second=50.0):
    """Convert time in seconds to token position"""
    return int(time_seconds * tokens_per_second)

def convert_token_to_time(token_position, tokens_per_second=50.0):
    """Convert token position to time in seconds"""
    return token_position / tokens_per_second

def extract_audio_segment(input_file, start_time, end_time, output_file):
    """Extract a segment of audio using ffmpeg"""
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y', '-ss', str(start_time),
        '-i', input_file, '-t', str(duration),
        '-acodec', 'pcm_s16le', '-ar', '32000',
        output_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0

def get_random_shuffle_positions(shuffle_range_tokens, token_length, shuffle_percentage, tokens_per_second, total_audio_tokens):
    """Get random positions for token shuffling within the shuffle range
    
    Args:
        shuffle_range_tokens: Total tokens available in shuffle range
        token_length: Length of each token block
        shuffle_percentage: Percentage of TOTAL audio to shuffle (e.g., 0.2 for 20%)
        tokens_per_second: Tokens per second
        total_audio_tokens: Total tokens in the entire audio file
    
    Returns:
        List of (start_token, end_token) pairs for shuffling
    """
    # Calculate how many tokens we should shuffle based on TOTAL audio length
    total_shuffle_tokens = int(total_audio_tokens * shuffle_percentage)
    
    # Calculate how many token pairs we can fit
    tokens_per_pair = token_length * 2  # Each pair needs 2 * token_length
    max_pairs = total_shuffle_tokens // tokens_per_pair
    
    if max_pairs < 1:
        return []
    
    # Generate random positions for token pairs (limited to shuffle range)
    positions = []
    available_positions = list(range(0, shuffle_range_tokens - tokens_per_pair + 1, token_length))
    
    # Limit max_pairs to available positions in shuffle range
    if len(available_positions) < max_pairs:
        max_pairs = len(available_positions)
    
    # Randomly select positions
    selected_positions = random.sample(available_positions, max_pairs)
    
    for pos in selected_positions:
        # Each position defines a pair: [pos:pos+token_length] and [pos+token_length:pos+2*token_length]
        positions.append((pos, pos + token_length, pos + token_length, pos + 2 * token_length))
    
    return positions

def shuffle_tokens_randomly(input_file, output_file, shuffle_start_time, shuffle_end_time, 
                           token_length, shuffle_percentage, tokens_per_second=50.0):
    """Randomly shuffle audio tokens within the specified range
    
    Args:
        input_file: Input audio file path
        output_file: Output audio file path  
        shuffle_start_time: Start time for shuffling range (e.g., 5.0)
        shuffle_end_time: End time for shuffling range (e.g., 14.0)
        token_length: Length of each token block to shuffle
        shuffle_percentage: Percentage of TOTAL audio to shuffle (e.g., 0.2)
        tokens_per_second: Number of tokens per second (default: 50.0)
    """
    
    # Get total audio duration and calculate total tokens
    total_duration = get_audio_duration(input_file)
    total_audio_tokens = int(total_duration * tokens_per_second)
    
    # Calculate shuffle range in tokens
    shuffle_range_time = shuffle_end_time - shuffle_start_time
    shuffle_range_tokens = int(shuffle_range_time * tokens_per_second)
    
    # Get random shuffle positions (now based on total audio tokens)
    shuffle_positions = get_random_shuffle_positions(
        shuffle_range_tokens, token_length, shuffle_percentage, tokens_per_second, total_audio_tokens
    )
    
    if not shuffle_positions:
        print(f"    Warning: No shuffle positions available for token length {token_length}")
        print(f"    Total audio tokens: {total_audio_tokens}, Target shuffle tokens: {int(total_audio_tokens * shuffle_percentage)}")
        # Just copy the original file
        shutil.copy2(input_file, output_file)
        return True
    
    print(f"    Shuffling {len(shuffle_positions)} random token pairs of length {token_length}")
    print(f"    Total audio tokens: {total_audio_tokens}, Target shuffle tokens: {int(total_audio_tokens * shuffle_percentage)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        segments = []
        
        # Extract before shuffle region (0 to shuffle_start_time)
        if shuffle_start_time > 0:
            before_file = os.path.join(temp_dir, 'before.wav')
            if extract_audio_segment(input_file, 0, shuffle_start_time, before_file):
                segments.append(before_file)
            else:
                print(f"Error extracting before segment")
                return False
        
        # Process the shuffle range
        current_time = shuffle_start_time
        start_token_offset = convert_time_to_token_position(shuffle_start_time, tokens_per_second)
        
        # Create a mapping of which tokens to swap
        token_swaps = {}
        for pos in shuffle_positions:
            block1_start, block1_end, block2_start, block2_end = pos
            # Map block1 tokens to block2 and vice versa
            for i in range(token_length):
                token_swaps[block1_start + i] = block2_start + i
                token_swaps[block2_start + i] = block1_start + i
        
        # Process each token in the shuffle range
        for token_idx in range(shuffle_range_tokens):
            token_start_time = shuffle_start_time + (token_idx / tokens_per_second)
            token_end_time = token_start_time + (1 / tokens_per_second)
            
            # Check if this token should be swapped
            if token_idx in token_swaps:
                # Use the swapped token's position
                swap_token_idx = token_swaps[token_idx]
                swap_start_time = shuffle_start_time + (swap_token_idx / tokens_per_second)
                swap_end_time = swap_start_time + (1 / tokens_per_second)
                
                # Extract the swapped token
                token_file = os.path.join(temp_dir, f'token_{token_idx:04d}.wav')
                if not extract_audio_segment(input_file, swap_start_time, swap_end_time, token_file):
                    print(f"Error extracting swapped token {token_idx}")
                    return False
            else:
                # Use the original token
                token_file = os.path.join(temp_dir, f'token_{token_idx:04d}.wav')
                if not extract_audio_segment(input_file, token_start_time, token_end_time, token_file):
                    print(f"Error extracting original token {token_idx}")
                    return False
            
            segments.append(token_file)
        
        # Extract after shuffle region (from shuffle_end_time to end)
        total_duration = get_audio_duration(input_file)
        if shuffle_end_time < total_duration:
            after_file = os.path.join(temp_dir, 'after.wav')
            if extract_audio_segment(input_file, shuffle_end_time, total_duration, after_file):
                segments.append(after_file)
            else:
                print(f"Error extracting after segment")
                return False
        
        # Create concat file for joining all segments
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for segment in segments:
                if os.path.exists(segment):
                    f.write(f"file '{os.path.abspath(segment)}'\n")
        
        # Join segments
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file, 
            '-acodec', 'pcm_s16le', '-ar', '32000',
            output_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Error joining segments: {result.stderr.decode()}")
            return False
            
        return True

def process_file(input_file, output_dir, shuffle_start_time, shuffle_end_time, 
                token_lengths, shuffle_percentage, tokens_per_second):
    """Process a single audio file with random token shuffling"""
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print(f"Processing: {base_name}")
    
    # Process each token length
    for token_length in token_lengths:
        print(f"  Processing random shuffle for {token_length} tokens...")
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{base_name}_shuffle_random_tk{token_length}.wav")
        
        # Perform random token shuffle
        if shuffle_tokens_randomly(input_file, output_file, shuffle_start_time, shuffle_end_time,
                                 token_length, shuffle_percentage, tokens_per_second):
            print(f"    Successfully saved to {output_file}")
        else:
            print(f"    Failed to create {output_file}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Validate arguments
    if args.shuffle_start_time >= args.shuffle_end_time:
        print("Error: Shuffle start time must be less than end time!")
        return
    
    if not all(t > 0 for t in args.token_lengths):
        print("Error: Token lengths must be positive integers!")
        return
        
    if not 0 < args.shuffle_percentage <= 1:
        print("Error: Shuffle percentage must be between 0 and 1!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shuffle time range: {args.shuffle_start_time}s - {args.shuffle_end_time}s")
    print(f"Shuffle percentage: {args.shuffle_percentage * 100:.1f}%")
    print(f"Token lengths: {args.token_lengths}")
    print(f"Tokens per second: {args.tokens_per_second}")
    print(f"Sample rate: {args.sample_rate}Hz")
    
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
        process_file(input_path, args.output_dir, args.shuffle_start_time, args.shuffle_end_time,
                    args.token_lengths, args.shuffle_percentage, args.tokens_per_second)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()