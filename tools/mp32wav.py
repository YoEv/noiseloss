
import os
import sys
import argparse
from pathlib import Path
from pydub import AudioSegment

def convert_mp3_to_wav(input_file, output_file, sample_rate=44100):
    audio = AudioSegment.from_mp3(input_file)
    
    audio = audio.set_frame_rate(sample_rate)
    
    audio.export(output_file, format="wav")
    print(f"✓ Converted: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    return True

def batch_convert(input_dir, output_dir_44k, output_dir_32k):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    Path(output_dir_44k).mkdir(parents=True, exist_ok=True)
    Path(output_dir_32k).mkdir(parents=True, exist_ok=True)
    
    mp3_files = list(input_path.glob("*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {input_dir}")
        return
    
    print(f"Found {len(mp3_files)} MP3 files")
    print("Starting conversion...")
    
    success_count = 0
    total_count = len(mp3_files)
    
    for mp3_file in mp3_files:
        base_name = mp3_file.stem
        
        output_44k = Path(output_dir_44k) / f"{base_name}.wav"
        success_44k = convert_mp3_to_wav(str(mp3_file), str(output_44k), 44100)
        
        output_32k = Path(output_dir_32k) / f"{base_name}.wav"
        success_32k = convert_mp3_to_wav(str(mp3_file), str(output_32k), 32000)
        
        if success_44k and success_32k:
            success_count += 1
    
    print(f"\nConversion complete! {success_count}/{total_count} files converted successfully")
    print(f"44.1kHz WAV files saved to: {output_dir_44k}")
    print(f"32kHz WAV files saved to: {output_dir_32k}")

def main():
    parser = argparse.ArgumentParser(description="Batch convert MP3 to WAV")
    parser.add_argument("input_dir", help="Input directory containing MP3 files")
    parser.add_argument("--output_44k", default="output_44k", help="44.1kHz output directory (default: output_44k)")
    parser.add_argument("--output_32k", default="output_32k", help="32kHz output directory (default: output_32k)")
    
    args = parser.parse_args()
    
    print("MP3 to WAV Converter")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"44.1kHz output directory: {args.output_44k}")
    print(f"32kHz output directory: {args.output_32k}")
    print("=" * 50)
    
    batch_convert(args.input_dir, args.output_44k, args.output_32k)

if __name__ == "__main__":
    main()