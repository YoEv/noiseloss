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
    try:
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

        #print(f"✓ 转换成功: {Path(midi_path).name} -> {Path(wav_path).name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ 转换失败 {Path(midi_path).name}: {e.stderr.decode().strip()}")
        return False
    except Exception as e:
        print(f"✗ 处理 {Path(midi_path).name} 时出错: {str(e)}")
        return False

def batch_convert(midi_input_dir, wav_output_dir, soundfont_path, sample_rate, channels, threads):
    """批量转换MIDI文件"""
    os.makedirs(wav_output_dir, exist_ok=True)

    midi_files = []
    for root, _, files in os.walk(midi_input_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))

    if not midi_files:
        print(f"错误: 在 {midi_input_dir} 中未找到MIDI文件")
        return

    print(f"找到 {len(midi_files)} 个MIDI文件待转换...")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(
            lambda f: convert_midi_to_wav(f, wav_output_dir, soundfont_path, sample_rate, channels),
            midi_files
        ))

    success = sum(results)
    print(f"\n转换完成: 成功 {success}/{len(midi_files)}")
    print(f"输出目录: {wav_output_dir}")

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

    # 检查依赖
    required_cmds = ["fluidsynth", "sox"]
    for cmd in required_cmds:
        if not shutil.which(cmd):
            print(f"错误: 需要安装 {cmd}")
            print("Ubuntu/Debian安装命令:")
            print(f"sudo apt install {'fluidsynth' if cmd == 'fluidsynth' else 'sox'}")
            exit(1)

    # 检查soundfont文件是否存在
    if not os.path.isfile(args.soundfont):
        print(f"错误: 找不到soundfont文件: {args.soundfont}")
        exit(1)

    # 检查输入目录是否存在
    if not os.path.isdir(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        exit(1)

    print(f"使用soundfont: {args.soundfont}")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"采样率: {args.sample_rate}")
    print(f"声道数: {args.channels}")
    print(f"线程数: {args.threads}")

    batch_convert(args.input, args.output, args.soundfont, args.sample_rate, args.channels, args.threads)

if __name__ == "__main__":
    main()