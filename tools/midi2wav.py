import os
import subprocess
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# SOUNDFONT_PATH = "soundfonts/GenralUser-GS/GeneralUser-GS.sf2"
# # SOUNDFONT_PATH = "soundfonts/YDP-GrandPiano-SF2-20160804/YDP-GrandPiano-20160804.sf2"
# # soundfonts/SalamanderGrandPianoV3_48khz24bit/SalamanderGrandPianoV3.sfz
# # soundfonts/YDP-GrandPiano-SF2-20160804/YDP-GrandPiano-20160804.sf2
# MIDI_INPUT_DIR = "LMD_100"
# # MIDI_INPUT_DIR = "velocity_pitch_mod_extract"
# # MIDI_INPUT_DIR = "structure_mod/structure_processed"
# # MIDI_INPUT_DIR = "velocity_mod_test3/velocity_processed"
# #MIDI_INPUT_DIR = "mod_all/pitch_processed"
# #MIDI_INPUT_DIR = "selected_rhythm_files"
# #"final_modified_output_error_fixed"
# WAV_OUTPUT_DIR = "LMD_100_wav"
# SAMPLE_RATE = "32000"
# CHANNELS = "1"
# THREADS = 4
# Default values
DEFAULT_SOUNDFONT_PATH = "soundfonts/GeneralUser-GS/GeneralUser-GS.sf2"
DEFAULT_MIDI_INPUT_DIR = "Dataset/ASAP_100"
DEFAULT_WAV_OUTPUT_DIR = "ASAP_100_wav"
DEFAULT_SAMPLE_RATE = "32000"
DEFAULT_CHANNELS = "1"
DEFAULT_THREADS = 4
def convert_midi_to_wav(midi_path, output_dir, soundfont_path, sample_rate, channels):
    midi_name = Path(midi_path).stem
    wav_path = os.path.join(output_dir, f"{midi_name}.wav")

    temp_wav = os.path.join(output_dir, f"temp_{midi_name}.wav")
    fluidsynth_cmd = [
        "fluidsynth",
        "-ni",
        "-F", temp_wav,
        "-r", "44100",
        "-g", "0.9",
        "-q",
        soundfont_path,
        midi_path
    ]

    subprocess.run(fluidsynth_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    sox_cmd = [
        "sox",
        temp_wav,
        "-b", "16",
        "-c", channels,
        "-r", sample_rate,
        wav_path,
        "gain", "-3"
    ]

    subprocess.run(sox_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    os.remove(temp_wav)

    return True


def batch_convert(midi_input_dir, wav_output_dir, soundfont_path, sample_rate, channels, threads):
    os.makedirs(wav_output_dir, exist_ok=True)

    midi_files = []
    for root, _, files in os.walk(midi_input_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))

    if not midi_files:
        print(f"Error: No MIDI files found in {midi_input_dir}")
        return

    print(f"Found {len(midi_files)} MIDI files to convert...")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(
            lambda f: convert_midi_to_wav(f, wav_output_dir, soundfont_path, sample_rate, channels),
            midi_files
        ))

    success = sum(results)
    print(f"\nConversion complete: success {success}/{len(midi_files)}")
    print(f"Output directory: {wav_output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert MIDI files to WAV format using FluidSynth and SoX')

    parser.add_argument('--soundfont', type=str, default=DEFAULT_SOUNDFONT_PATH,
                        help=f'Path to the soundfont file (default: {DEFAULT_SOUNDFONT_PATH})')

    parser.add_argument('--input', type=str, default=DEFAULT_MIDI_INPUT_DIR,
                        help=f'Input directory containing MIDI files (default: {DEFAULT_MIDI_INPUT_DIR})')

    parser.add_argument('--output', type=str, default=DEFAULT_WAV_OUTPUT_DIR,
                        help=f'Output directory for WAV files (default: {DEFAULT_WAV_OUTPUT_DIR})')

    parser.add_argument('--sample-rate', type=str, default=DEFAULT_SAMPLE_RATE,
                        help=f'Sample rate for output WAV files (default: {DEFAULT_SAMPLE_RATE})')

    parser.add_argument('--channels', type=str, default=DEFAULT_CHANNELS,
                        help=f'Number of channels for output WAV files (default: {DEFAULT_CHANNELS})')

    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS,
                        help=f'Number of parallel conversion threads (default: {DEFAULT_THREADS})')

    args = parser.parse_args()

    required_cmds = ["fluidsynth", "sox"]
    for cmd in required_cmds:
        if not shutil.which(cmd):
            print(f"Error: Required command not found: {cmd}")
            print("Ubuntu/Debian install:")
            print(f"sudo apt install {'fluidsynth' if cmd == 'fluidsynth' else 'sox'}")
            exit(1)

    if not os.path.isfile(args.soundfont):
        print(f"Error: Soundfont file not found: {args.soundfont}")
        exit(1)

    if not os.path.isdir(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        exit(1)

    print(f"Using soundfont: {args.soundfont}")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Channels: {args.channels}")
    print(f"Threads: {args.threads}")

    batch_convert(args.input, args.output, args.soundfont, args.sample_rate, args.channels, args.threads)

if __name__ == "__main__":
    main()