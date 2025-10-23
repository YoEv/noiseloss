import torch
import os
import argparse
import sys
from tqdm import tqdm
import miditoolkit
from transformers import AutoTokenizer, AutoModel
import numpy as np
import glob

# Add the path to the directory containing midiprocessor
sys.path.append("/home/evev/asap-dataset/muzic/musecoco/2-attribute2music_model")
from midiprocessor.midi_encoding import MidiEncoder

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    """Load MuseCoco model from local files"""
    try:
        # Add the parent directory to path to ensure imports work correctly
        muzic_path = "/home/evev/asap-dataset/muzic"
        sys.path.insert(0, muzic_path)
        # Also add the muzic directory itself to the path
        sys.path.insert(0, os.path.dirname(muzic_path))
        
        # Import MuseCoco modules directly without the muzic prefix
        from musecoco.model import MuseCocoModel
        from musecoco.tokenizer import MusicTokenizer

        # Load model from checkpoint
        model_path = os.path.join(muzic_path, "musecoco/2-attribute2music_model/checkpoints")
        model = MuseCocoModel.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        # Load tokenizer
        tokenizer_path = os.path.join(muzic_path, "musecoco/tokenizer")
        tokenizer = MusicTokenizer.from_pretrained(tokenizer_path)

        print(f"MuseCoco model loaded successfully from local files on {device}")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model from local files: {str(e)}")
        print("Please make sure you have cloned the Microsoft Muzic repository and installed all dependencies.")
        raise Exception("Could not load MuseCoco model. Please check the repository path and ensure it exists.")

def midi_to_tokens(midi_path, tokenizer):
    """Convert MIDI file to tokens using MuseCoco tokenizer"""
    try:
        # Import necessary modules from muzic
        sys.path.append("/home/evev/asap-dataset/muzic/musecoco/2-attribute2music_model")
        # Use the MidiEncoder from the model directory as requested

        # Initialize the MIDI encoder with REMIGEN encoding method
        encoder = MidiEncoder(encoding_method='REMIGEN')

        # Encode the MIDI file to token strings
        token_str_list = encoder.encode_file(midi_path)

        # Convert token strings to model input format
        tokens = tokenizer.encode(token_str_list, return_tensors="pt")

        return tokens

    except Exception as e:
        print(f"Error processing MIDI file {os.path.basename(midi_path)}: {str(e)}")
        return None

def process_single_midi(midi_path, model, tokenizer, device):
    """Process a single MIDI file and calculate loss"""
    try:
        # Convert MIDI to tokens
        tokens = midi_to_tokens(midi_path, tokenizer)
        if tokens is None:
            return None

        tokens = tokens.to(device)

        # Calculate loss using the model
        with torch.no_grad():
            # Forward pass through the model
            outputs = model(input_ids=tokens, return_dict=True)

            # Get the loss from the model output
            # For MuseCoco, we'll use the model's built-in loss calculation
            loss = outputs.loss

            # If the model doesn't return loss directly, calculate it manually
            if loss is None:
                logits = outputs.logits
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = tokens[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )

        return loss.item()

    except Exception as e:
        print(f"Error processing {os.path.basename(midi_path)}: {str(e)}")
        return None

def process_midi_directory(midi_dir, model, tokenizer, device, output_file=None):
    """Process all MIDI files in a directory"""
    # Find all MIDI files in directory and subdirectories
    midi_files = []
    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))

    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_dir} or its subdirectories")

    print(f"\nProcessing {len(midi_files)} MIDI files")
    results = []
    error_files = []

    # Open output file if specified
    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"Results will be saved to: {output_file}")

    for midi_path in tqdm(midi_files, desc="Processing"):
        # Use relative path for output
        rel_path = os.path.relpath(midi_path, midi_dir)
        loss = process_single_midi(midi_path, model, tokenizer, device)

        if loss is not None:
            results.append((rel_path, loss))
            # Write to file immediately if output file is specified
            if out_file:
                out_file.write(f"{rel_path}: {loss:.8f}\n")
                out_file.flush()  # Ensure data is written immediately
        else:
            error_files.append(rel_path)

    # Close output file if opened
    if out_file:
        out_file.close()

    # Write error files list to file
    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Failed to process {len(error_files)} files\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\nFailed to process {len(error_files)} files, saved to: {error_file_path}")

    return results, error_files

def get_missing_files(midi_dir, results_file):
    """Get list of files that haven't been processed yet"""
    # Get all MIDI files in directory and subdirectories
    all_midi_files = []
    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                rel_path = os.path.relpath(os.path.join(root, file), midi_dir)
                all_midi_files.append(rel_path)

    all_midi_files = set(all_midi_files)

    # Get processed files from results file
    processed_files = set()
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename = line.split(':', 1)[0].strip()
                    processed_files.add(filename)
    except FileNotFoundError:
        print(f"Results file {results_file} does not exist, will process all files")

    # Return files that are in all_midi_files but not in processed_files
    missing_files = all_midi_files - processed_files
    return list(missing_files)

def process_missing_files(midi_dir, results_file):
    """Process files that haven't been processed yet (using CPU)"""
    missing_files = get_missing_files(midi_dir, results_file)

    if not missing_files:
        print("No missing files to process")
        return

    print(f"Found {len(missing_files)} unprocessed files, will process using CPU")

    # Print all missing filenames before processing
    print("\nList of unprocessed files:")
    for filename in missing_files:
        print(f"{filename}")
    print("\nStarting to process these files...\n")

    # Load model on CPU
    device = "cpu"
    model, tokenizer = load_model(device)

    # Process each missing file
    results = []
    error_files = []

    # Open results file in append mode
    with open(results_file, 'a') as f:
        for filename in tqdm(missing_files, desc="Processing"):
            # Get full path
            full_path = os.path.join(midi_dir, filename)

            # Process file
            loss = process_single_midi(full_path, model, tokenizer, device)

            if loss is not None:
                results.append((filename, loss))
                # Write to file immediately
                f.write(f"{filename}: {loss:.8f}\n")
                f.flush()  # Ensure data is written immediately
            else:
                error_files.append(filename)

    # Write error files list to file
    if error_files:
        error_file_path = results_file + ".cpu_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Failed to process {len(error_files)} files\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\nFailed to process {len(error_files)} files, saved to: {error_file_path}")

    return results, error_files

def main():
    parser = argparse.ArgumentParser(description="Calculate loss values for MIDI files using MuseCoco")
    parser.add_argument("--midi_dir", type=str, default="/home/evev/asap-dataset/muzic/musecoco/LMD_100_pitch",
                        help="Directory containing MIDI files (will search recursively)")
    parser.add_argument("--output_file", type=str,
                        help="Output file path for results")
    parser.add_argument("--process_missing", action="store_true",
                        help="Process missing files (using CPU)")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU processing for all files")

    args = parser.parse_args()

    # Redirect stdout to file if output_file is specified and not processing missing files
    original_stdout = None
    if args.output_file and not args.process_missing:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        if args.process_missing and args.output_file:
            # Process missing files on CPU
            process_missing_files(args.midi_dir, args.output_file)
        else:
            # Normal processing
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")

            model, tokenizer = load_model(device)

            assert os.path.isdir(args.midi_dir), f"Directory does not exist: {args.midi_dir}"

            results, error_files = process_midi_directory(
                args.midi_dir, model, tokenizer, device, args.output_file
            )

            if not args.output_file:
                print("\nProcessing results:")
                for filename, loss in results:
                    print(f"{filename}: {loss:.8f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if 'device' in locals() and device == 'cuda':
            torch.cuda.empty_cache()
        print("GPU memory cleared")

        # Restore stdout if redirected
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()