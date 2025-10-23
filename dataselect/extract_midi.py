import os
import shutil
import argparse

def extract_midi_files(source_dir, target_dir):
    """
    Extract all MIDI files from source_dir and its subdirectories
    and copy them to target_dir with unique names.

    Args:
        source_dir: Source directory containing MIDI files in subdirectories
        target_dir: Target directory where all MIDI files will be copied
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Counter for duplicate filenames
    file_counter = {}

    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mid'):
                # Get the full path of the source file
                source_file = os.path.join(root, file)

                # Handle duplicate filenames by adding a counter
                if file in file_counter:
                    file_counter[file] += 1
                    base_name, ext = os.path.splitext(file)
                    target_file = os.path.join(target_dir, f"{base_name}_{file_counter[file]}{ext}")
                else:
                    file_counter[file] = 0
                    target_file = os.path.join(target_dir, file)

                # Copy the file
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} -> {target_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract MIDI files from subdirectories')
    parser.add_argument('--source', default='./velocity_mod_short/velocity_processed',
                        help='Source directory containing MIDI files in subdirectories')
    parser.add_argument('--target', default='./velocity_short_extracted_midi_files',
                        help='Target directory where all MIDI files will be copied')

    args = parser.parse_args()

    print(f"Extracting MIDI files from: {args.source}")
    print(f"Copying to: {args.target}")

    extract_midi_files(args.source, args.target)

    # Count the number of extracted files
    midi_count = len([f for f in os.listdir(args.target) if f.endswith('.mid')])
    print(f"\nExtraction complete. {midi_count} MIDI files extracted to {args.target}")

if __name__ == "__main__":
    main()