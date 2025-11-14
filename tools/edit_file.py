import os
import glob
import re
import argparse

def process_file(input_file, output_file):
    """Process a single file to extract MIDI filename and overall_mse"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            match = re.search(
                r'.*_processed/.*?/(.*?\.mid):.*?overall_mse: (\d+\.\d+)',
                line.strip()
            )
            if match:
                midi_name = match.group(1)
                overall_mse = match.group(2)
                f_out.write(f"{midi_name}: {overall_mse}\n")

def main():
    parser = argparse.ArgumentParser(description='Extract MIDI filenames and MSE values')
    parser.add_argument('--input', required=True,
                       help='Input file or directory path (supports glob like *.txt)')
    parser.add_argument('--output_dir', default='simplified_results',
                       help='Output directory path (default: ./simplified_results)')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        input_files = glob.glob(args.input)

    if not input_files:
        print(f"Error: No input files found {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing {len(input_files)} files...")

    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, f"loss_cal_{filename}")
        process_file(input_file, output_file)
        print(f"Processed: {input_file} -> {output_file}")

    print(f"\nDone! Results saved in: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()