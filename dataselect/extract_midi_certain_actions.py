import os
import shutil
import argparse

def extract_midi_files(source_dir, target_dir):
    """
    Extract specific MIDI files from source_dir and its subdirectories
    and copy them to target_dir with unique names.

    Args:
        source_dir: Source directory containing MIDI files in subdirectories
        target_dir: Target directory where specific MIDI files will be copied
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Define the specific suffixes we want to extract
    target_suffixes = [
        "_score_short.mid_ori.mid_ori.mid",
        "short_velocity1_1_step1_nt80_pitch1_1_step1_semi80.mid",
        "short_velocity1_1_step1_nt80_pitch1_2_step1_oct80.mid",
        "short_velocity1_1_step1_nt80_pitch1_3_step1_dia80.mid"
    ]

    # Counter for duplicate filenames
    file_counter = {}
    extracted_files = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file ends with any of our target suffixes
            if any(file.endswith(suffix) for suffix in target_suffixes):
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
                extracted_files += 1

    return extracted_files

def main():
    parser = argparse.ArgumentParser(description='Extract specific MIDI files from subdirectories')
    parser.add_argument('--source', default='./velocity_pitch_mod_short/pitch_processed',
                        help='Source directory containing MIDI files in subdirectories')
    parser.add_argument('--target', default='./velocity_pitch_short_extracted_specific_midi_files',
                        help='Target directory where specific MIDI files will be copied')

    args = parser.parse_args()

    print(f"Extracting specific MIDI files from: {args.source}")
    print(f"Copying to: {args.target}")

    # Extract the files and get the count
    midi_count = extract_midi_files(args.source, args.target)

    print(f"\nExtraction complete. {midi_count} MIDI files extracted to {args.target}")
    print(f"Files extracted have the following suffixes:")
    print("  - _score_short.mid_ori.mid_ori.mid")
    print("  - short_velocity1_1_step1_nt80_pitch1_1_step1_semi80.mid")
    print("  - short_velocity1_1_step1_nt80_pitch1_2_step1_oct80.mid")
    print("  - short_velocity1_1_step1_nt80_pitch1_3_step1_dia80.mid")
    # print("  - _score.mid_ori")
    # print("  - velocity1_1_step1_nt80_pitch1_1_step1_semi80")
    # print("  - velocity1_1_step1_nt80_pitch1_2_step1_oct80")
    # print("  - velocity1_1_step1_nt80_pitch1_3_step1_dia80")

if __name__ == "__main__":
    main()