import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def cut_audio_file(input_file, output_dir, cut_length):
    """
    Cut audio file to specified length using FFmpeg and save to output directory

    Args:
        input_file: Path to the input audio file
        output_dir: Directory to save the output audio file
        cut_length: Length to cut the audio file to (in seconds)
    """
    try:
        input_path = Path(input_file)
        output_filename = f"{input_path.stem}_cut{cut_length}s{input_path.suffix}"
        output_path = os.path.join(output_dir, output_filename)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use FFmpeg to cut the audio file
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_file,
            "-t", str(cut_length),
            "-c:a", "copy",  # Copy audio codec to avoid re-encoding
            "-y",  # Overwrite output file if it exists
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"✓ Cut: {input_path.name} -> {output_filename} ({cut_length}s)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Cutting failed for {input_path.name}: {e.stderr.decode().strip()}")
        return False
    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {str(e)}")
        return False

def batch_cut(input_dir, output_dir, cut_length, threads=4):
    """
    Batch cut all supported audio files in the input directory

    Args:
        input_dir: Directory containing audio files to cut
        output_dir: Directory to save cut audio files
        cut_length: Length to cut the audio files to (in seconds)
        threads: Number of parallel processing threads
    """
    # Supported audio formats (FFmpeg supports many more)
    supported_formats = ('.wav', '.m4a', '.mp3', '.flac', '.ogg', '.aac', '.mp4', '.wma')

    # Find all audio files
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"Error: No supported audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files to cut...")

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(
            lambda f: cut_audio_file(f, output_dir, cut_length),
            audio_files
        ))

    success_count = sum(results)
    print(f"\nCutting complete: {success_count}/{len(audio_files)} successful")
    print(f"Output directory: {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cut audio files to specified length using FFmpeg')
    parser.add_argument('--input', required=True, help='Input directory containing audio files')
    parser.add_argument('--output', required=True, help='Output directory for cut audio files')
    parser.add_argument('--length', type=int, required=True, choices=[9, 14, 15],
                        help='Length to cut audio files to (in seconds)')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel processing threads')

    args = parser.parse_args()

    # Check if ffmpeg is installed
    if not os.path.exists(os.path.join('/usr', 'bin', 'ffmpeg')):
        print("Error: FFmpeg is required but not found")
        print("Install FFmpeg with: sudo apt install ffmpeg")
        return

    # Run the batch cutting
    batch_cut(args.input, args.output, args.length, args.threads)

if __name__ == "__main__":
    main()