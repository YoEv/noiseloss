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
import itertools

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Shuffle audio tokens at specific positions between 5-14 seconds.')
    parser.add_argument('--input_dir', type=str, default='/home/evev/asap-dataset/ShutterStock_32k_ori',
                        help='Directory containing input music files')
    parser.add_argument('--output_dir', type=str, default='/home/evev/asap-dataset/shutter_double_shuffle_order',
                        help='Directory to save shuffled music files')
    parser.add_argument('--shuffle_start_time', type=float, default=5.0,
                        help='Start time for shuffling in seconds (default: 5.0)')
    parser.add_argument('--shuffle_end_time', type=float, default=14.0,
                        help='End time for shuffling in seconds (default: 14.0)')
    parser.add_argument('--token_lengths', nargs='+', type=int, 
                        default=[5, 10, 50, 100, 150, 200],
                        help='Token lengths to shuffle (default: 5 10 50 100 150 200)')
    parser.add_argument('--file_extension', type=str, default='.wav',
                        help='File extension of audio files to process')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Sample rate for audio processing (default: 32000)')
    parser.add_argument('--tokens_per_second', type=float, default=50.0,
                        help='Number of tokens per second (default: 50.0)')
    parser.add_argument('--double_shuffle', action='store_true',
                        help='Enable double shuffle mode (two different token lengths)')
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
    """Convert time in seconds to token position
    
    With 50 tokens per second, each token represents 0.02 seconds
    """
    token_position = int(time_seconds * tokens_per_second)
    return token_position

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

def shuffle_audio_tokens(input_file, output_file, shuffle_start_time, token_length, tokens_per_second=50.0):
    """Shuffle audio tokens by swapping two adjacent blocks
    
    Args:
        input_file: Input audio file path
        output_file: Output audio file path  
        shuffle_start_time: Time in seconds where shuffling starts (e.g., 5.0)
        token_length: Number of tokens to swap (e.g., 5, 10, 50, etc.)
        tokens_per_second: Number of tokens per second (default: 50.0)
    """
    
    # Calculate token positions
    # At 5 seconds, we're at token 250 (5 * 50 = 250)
    start_token = convert_time_to_token_position(shuffle_start_time, tokens_per_second)
    
    # Define the two blocks to swap
    # For token_length=5: swap tokens 250-254 with tokens 255-259
    # For token_length=10: swap tokens 250-259 with tokens 260-269
    block1_start = start_token
    block1_end = start_token + token_length
    block2_start = start_token + token_length  
    block2_end = start_token + 2 * token_length
    
    # Convert token positions back to time
    block1_start_time = convert_token_to_time(block1_start, tokens_per_second)
    block1_end_time = convert_token_to_time(block1_end, tokens_per_second)
    block2_start_time = convert_token_to_time(block2_start, tokens_per_second)
    block2_end_time = convert_token_to_time(block2_end, tokens_per_second)
    
    print(f"    Swapping tokens {block1_start}-{block1_end-1} ({block1_start_time:.3f}s-{block1_end_time:.3f}s) with tokens {block2_start}-{block2_end-1} ({block2_start_time:.3f}s-{block2_end_time:.3f}s)")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract segments
        before_file = os.path.join(temp_dir, 'before.wav')
        block1_file = os.path.join(temp_dir, 'block1.wav')
        block2_file = os.path.join(temp_dir, 'block2.wav')
        after_file = os.path.join(temp_dir, 'after.wav')
        
        # Extract before shuffle region (0 to block1_start_time)
        if block1_start_time > 0:
            if not extract_audio_segment(input_file, 0, block1_start_time, before_file):
                print(f"Error extracting before segment")
                return False
        
        # Extract block 1 (to be swapped with block 2)
        if not extract_audio_segment(input_file, block1_start_time, block1_end_time, block1_file):
            print(f"Error extracting block 1")
            return False
            
        # Extract block 2 (to be swapped with block 1)
        if not extract_audio_segment(input_file, block2_start_time, block2_end_time, block2_file):
            print(f"Error extracting block 2")
            return False
        
        # Extract after shuffle region (from block2_end_time to end)
        total_duration = get_audio_duration(input_file)
        if block2_end_time < total_duration:
            if not extract_audio_segment(input_file, block2_end_time, total_duration, after_file):
                print(f"Error extracting after segment")
                return False
        
        # Create concat file for joining: before + block2 + block1 + after
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            if os.path.exists(before_file):
                f.write(f"file '{os.path.abspath(before_file)}'\n")
            f.write(f"file '{os.path.abspath(block2_file)}'\n")  # Swapped: block2 first
            f.write(f"file '{os.path.abspath(block1_file)}'\n")  # Swapped: block1 second
            if os.path.exists(after_file):
                f.write(f"file '{os.path.abspath(after_file)}'\n")
        
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

def shuffle_double_tokens(input_file, output_file, shuffle_start_time, shuffle_end_time, 
                         token_length1, token_length2, tokens_per_second=50.0):
    """Shuffle audio with two different token lengths in random positions within time range
    
    Args:
        input_file: Input audio file path
        output_file: Output audio file path  
        shuffle_start_time: Start time for shuffling range (e.g., 5.0)
        shuffle_end_time: End time for shuffling range (e.g., 14.0)
        token_length1: First token length to shuffle
        token_length2: Second token length to shuffle
        tokens_per_second: Number of tokens per second (default: 50.0)
    """
    
    # Calculate available time range for shuffling
    available_time = shuffle_end_time - shuffle_start_time
    max_token_length = max(token_length1, token_length2)
    
    # Ensure we have enough space for both token blocks
    required_time = (token_length1 + token_length2) / tokens_per_second
    if available_time < required_time:
        print(f"    Warning: Not enough time range ({available_time:.2f}s) for tokens ({required_time:.2f}s required)")
        return False
    
    # Randomly choose start position within the available range
    max_start_offset = available_time - required_time
    random_offset = random.uniform(0, max_start_offset)
    actual_start_time = shuffle_start_time + random_offset
    
    # Calculate token positions
    start_token = convert_time_to_token_position(actual_start_time, tokens_per_second)
    
    # Define the two blocks to swap (different sizes)
    block1_start = start_token
    block1_end = start_token + token_length1
    block2_start = start_token + token_length1
    block2_end = start_token + token_length1 + token_length2
    
    # Convert token positions back to time
    block1_start_time = convert_token_to_time(block1_start, tokens_per_second)
    block1_end_time = convert_token_to_time(block1_end, tokens_per_second)
    block2_start_time = convert_token_to_time(block2_start, tokens_per_second)
    block2_end_time = convert_token_to_time(block2_end, tokens_per_second)
    
    print(f"    Swapping {token_length1} tokens ({block1_start}-{block1_end-1}, {block1_start_time:.3f}s-{block1_end_time:.3f}s) with {token_length2} tokens ({block2_start}-{block2_end-1}, {block2_start_time:.3f}s-{block2_end_time:.3f}s)")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract segments
        before_file = os.path.join(temp_dir, 'before.wav')
        block1_file = os.path.join(temp_dir, 'block1.wav')
        block2_file = os.path.join(temp_dir, 'block2.wav')
        after_file = os.path.join(temp_dir, 'after.wav')
        
        # Extract before shuffle region (0 to block1_start_time)
        if block1_start_time > 0:
            if not extract_audio_segment(input_file, 0, block1_start_time, before_file):
                print(f"Error extracting before segment")
                return False
        
        # Extract block 1 (to be swapped with block 2)
        if not extract_audio_segment(input_file, block1_start_time, block1_end_time, block1_file):
            print(f"Error extracting block 1")
            return False
            
        # Extract block 2 (to be swapped with block 1)
        if not extract_audio_segment(input_file, block2_start_time, block2_end_time, block2_file):
            print(f"Error extracting block 2")
            return False
        
        # Extract after shuffle region (from block2_end_time to end)
        total_duration = get_audio_duration(input_file)
        if block2_end_time < total_duration:
            if not extract_audio_segment(input_file, block2_end_time, total_duration, after_file):
                print(f"Error extracting after segment")
                return False
        
        # Create concat file for joining: before + block2 + block1 + after
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            if os.path.exists(before_file):
                f.write(f"file '{os.path.abspath(before_file)}'\n")
            f.write(f"file '{os.path.abspath(block2_file)}'\n")  # Swapped: block2 first
            f.write(f"file '{os.path.abspath(block1_file)}'\n")  # Swapped: block1 second
            if os.path.exists(after_file):
                f.write(f"file '{os.path.abspath(after_file)}'\n")
        
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

def shuffle_continuous_token_pairs(input_file, output_file, shuffle_start_time, shuffle_end_time, 
                                 token_length, tokens_per_second=50.0):
    """Shuffle audio by continuously swapping adjacent token pairs of the same length
    
    Args:
        input_file: Input audio file path
        output_file: Output audio file path  
        shuffle_start_time: Start time for shuffling range (e.g., 5.0)
        shuffle_end_time: End time for shuffling range (e.g., 14.0)
        token_length: Length of each token block to swap
        tokens_per_second: Number of tokens per second (default: 50.0)
    """
    
    # Calculate available time range for shuffling
    available_time = shuffle_end_time - shuffle_start_time
    
    # Calculate how many token pairs we can fit in the time range
    pair_duration = (2 * token_length) / tokens_per_second  # Each pair needs 2 * token_length time
    max_pairs = int(available_time / pair_duration)
    
    if max_pairs < 1:
        print(f"    Warning: Not enough time range ({available_time:.2f}s) for even one token pair ({pair_duration:.2f}s required)")
        return False
    
    print(f"    Swapping {max_pairs} pairs of {token_length}-token blocks in range {shuffle_start_time}s-{shuffle_end_time}s")
    
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
        
        # Process each token pair in the shuffle range
        current_time = shuffle_start_time
        
        for pair_idx in range(max_pairs):
            # Calculate time positions for this pair
            block1_start_time = current_time
            block1_end_time = current_time + (token_length / tokens_per_second)
            block2_start_time = block1_end_time
            block2_end_time = block2_start_time + (token_length / tokens_per_second)
            
            # Calculate token positions for logging
            start_token = convert_time_to_token_position(block1_start_time, tokens_per_second)
            block1_tokens = f"{start_token}-{start_token + token_length - 1}"
            block2_tokens = f"{start_token + token_length}-{start_token + 2 * token_length - 1}"
            
            print(f"      Pair {pair_idx + 1}: Swapping tokens {block1_tokens} with {block2_tokens}")
            
            # Extract the two blocks
            block1_file = os.path.join(temp_dir, f'pair{pair_idx}_block1.wav')
            block2_file = os.path.join(temp_dir, f'pair{pair_idx}_block2.wav')
            
            if not extract_audio_segment(input_file, block1_start_time, block1_end_time, block1_file):
                print(f"Error extracting block 1 of pair {pair_idx + 1}")
                return False
                
            if not extract_audio_segment(input_file, block2_start_time, block2_end_time, block2_file):
                print(f"Error extracting block 2 of pair {pair_idx + 1}")
                return False
            
            # Add swapped blocks to segments (block2 first, then block1)
            segments.extend([block2_file, block1_file])
            
            # Move to next pair
            current_time = block2_end_time
        
        # Extract any remaining audio in the shuffle range
        if current_time < shuffle_end_time:
            remaining_file = os.path.join(temp_dir, 'remaining_shuffle.wav')
            if extract_audio_segment(input_file, current_time, shuffle_end_time, remaining_file):
                segments.append(remaining_file)
        
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
                token_lengths, tokens_per_second, double_shuffle=False):
    """Process a single audio file with token shuffling"""
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print(f"Processing: {base_name}")
    
    if double_shuffle:
        # Process each token length with continuous pair swapping
        for token_length in token_lengths:
            print(f"  Processing continuous pair shuffle for {token_length} tokens...")
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{base_name}_shuffle_pairs_tk{token_length}.wav")
            
            # Perform continuous token pair shuffle
            if shuffle_continuous_token_pairs(input_file, output_file, shuffle_start_time, shuffle_end_time,
                                             token_length, tokens_per_second):
                print(f"    Successfully saved to {output_file}")
            else:
                print(f"    Failed to create {output_file}")
    else:
        # Process each token length (original behavior)
        for token_length in token_lengths:
            print(f"  Processing {token_length} token shuffle...")
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{base_name}_shuffle_tk{token_length}.wav")
            
            # Perform token shuffle
            if shuffle_audio_tokens(input_file, output_file, shuffle_start_time, token_length, tokens_per_second):
                print(f"    Successfully saved to {output_file}")
            else:
                print(f"    Failed to create {output_file}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate arguments
    if args.shuffle_start_time >= args.shuffle_end_time:
        print("Error: Shuffle start time must be less than end time!")
        return
    
    if not all(t > 0 for t in args.token_lengths):
        print("Error: Token lengths must be positive integers!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shuffle time range: {args.shuffle_start_time}s - {args.shuffle_end_time}s")
    print(f"Token lengths: {args.token_lengths}")
    print(f"Double shuffle mode: {args.double_shuffle}")
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
                    args.token_lengths, args.tokens_per_second, args.double_shuffle)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()