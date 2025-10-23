import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def convert_audio_to_wav(input_file, output_dir):
    """
    Convert audio file to WAV format with 32kHz, mono, 16-bit using FFmpeg

    Args:
        input_file: Path to the input audio file
        output_dir: Directory to save the output WAV file
    """
    try:
        input_path = Path(input_file)
        output_filename = f"{input_path.stem}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use FFmpeg to convert the file
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_file,
            "-ar", "32000",  # 32kHz sample rate
            "-ac", "1",      # mono (1 channel)
            "-sample_fmt", "s16",  # 16-bit depth
            "-y",            # Overwrite output file if it exists
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"✓ Converted: {input_path.name} -> {output_filename}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed for {input_path.name}: {e.stderr.decode().strip()}")
        return False
    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {str(e)}")
        return False

def batch_convert(input_dir, output_dir, threads=4):
    """
    Batch convert all supported audio files in the input directory

    Args:
        input_dir: Directory containing audio files to convert
        output_dir: Directory to save converted WAV files
        threads: Number of parallel conversion threads
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

    print(f"Found {len(audio_files)} audio files to convert...")

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(
            lambda f: convert_audio_to_wav(f, output_dir),
            audio_files
        ))

    success_count = sum(results)
    print(f"\nConversion complete: {success_count}/{len(audio_files)} successful")
    print(f"Output directory: {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert audio files to WAV format (32kHz, mono, 16-bit) using FFmpeg')
    parser.add_argument('--input', required=True, help='Input directory containing audio files')
    parser.add_argument('--output', required=True, help='Output directory for converted WAV files')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel conversion threads')

    args = parser.parse_args()

    # Check if ffmpeg is installed
    if not os.path.exists(os.path.join('/usr', 'bin', 'ffmpeg')):
        print("Error: FFmpeg is required but not found")
        print("Install FFmpeg with: sudo apt install ffmpeg")
        return

    # Run the batch conversion
    batch_convert(args.input, args.output, args.threads)

if __name__ == "__main__":
    main()