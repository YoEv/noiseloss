import torch
import torchaudio
import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel, AutoConfig

def audio_read(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)

    if not hasattr(config, 'conv_pos_batch_norm'):
        config.conv_pos_batch_norm = False

    model = AutoModel.from_pretrained(
        "m-a-p/MERT-v0-public",
        config=config,
        trust_remote_code=True
    ).to(device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v0-public",
        trust_remote_code=True
    )

    print(f"MERT-v0-public模型加载成功")

    return model, feature_extractor

def process_single_audio(audio_path, model, feature_extractor, device):
    try:
        audio, sr = audio_read(audio_path)

        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
            sr = 16000

        audio = audio.to(device)

        audio_np = audio.squeeze().cpu().numpy()

        inputs = feature_extractor(
            audio_np,
            sampling_rate=sr,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # 计算重建损失（输入和输出嵌入之间的MSE）
            hidden_states = outputs.hidden_states
            input_embeddings = hidden_states[0] #保留输入嵌入
            output_embeddings = hidden_states[-1]

            loss = torch.nn.functional.mse_loss(
                input_embeddings,
                output_embeddings,
                reduction='mean'
            )

        return loss.item()

    except Exception as e:
        print(f"处理 {os.path.basename(audio_path)} 时出错: {str(e)}")
        return None

def process_audio_directory(audio_dir, model, feature_extractor, device, output_file=None):
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"目录 {audio_dir} 中没有WAV/MP3文件")

    print(f"\n开始处理 {len(audio_files)} 个音频")
    results = []
    error_files = []  # 用于跟踪处理失败的文件

    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"结果将保存到: {output_file}")

    for filename in tqdm(audio_files, desc="处理进度"):
        audio_path = os.path.join(audio_dir, filename)
        loss = process_single_audio(audio_path, model, feature_extractor, device)
        if loss is not None:
            results.append((filename, loss))
            # Write to file immediately if output file is specified
            if out_file:
                out_file.write(f"{filename}: {loss:.8f}\n")
                out_file.flush()  # Ensure data is written immediately
        else:
            error_files.append(filename)

    # Close output file if opened
    if out_file:
        out_file.close()

    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"处理失败的文件数量: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n处理失败的 {len(error_files)} 个文件已保存到: {error_file_path}")

    return results, error_files

def get_missing_files(audio_dir, results_file):
    # Get all audio files in directory
    all_audio_files = set(f for f in os.listdir(audio_dir)
                         if f.lower().endswith(('.wav', '.mp3')))

    # Get processed files from results file
    processed_files = set()
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename = line.split(':', 1)[0].strip()
                    processed_files.add(filename)
    except FileNotFoundError:
        print(f"结果文件 {results_file} 不存在，将处理所有文件")

    # Return files that are in all_audio_files but not in processed_files
    missing_files = all_audio_files - processed_files
    return list(missing_files)

def process_missing_files(audio_dir, results_file):
    missing_files = get_missing_files(audio_dir, results_file)

    if not missing_files:
        print("没有缺失的文件需要处理")
        return

    print(f"发现 {len(missing_files)} 个未处理的文件，将使用CPU处理")

    # Print all missing filenames before processing
    print("\n未处理的文件列表:")
    for filename in missing_files:
        print(f"{filename}")
    print("\n开始处理这些文件...\n")

    # Load model on CPU
    device = "cpu"
    print(f"Using device: {device}")
    model, feature_extractor = load_model(device)

    # Open results file in append mode
    with open(results_file, 'a') as out_file:
        for filename in tqdm(missing_files, desc="CPU处理进度"):
            audio_path = os.path.join(audio_dir, filename)
            loss = process_single_audio(audio_path, model, feature_extractor, device)
            if loss is not None:
                out_file.write(f"{filename}: {loss:.4f}\n")
                out_file.flush()  # Ensure data is written immediately

def main():
    parser = argparse.ArgumentParser(description="计算音频文件的loss值 (MERT-v0-public)")
    parser.add_argument("--audio_dir", type=str, default="pitch_out_wav",
                        help="音频文件目录")
    parser.add_argument("--output_file", type=str,
                        help="输出结果文件路径")
    parser.add_argument("--process_missing", action="store_true",
                        help="处理缺失的文件（使用CPU）")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU处理所有文件")

    args = parser.parse_args()

    # Redirect stdout to file if output_file is specified and not processing missing files
    original_stdout = None
    if args.output_file and not args.process_missing:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        if args.process_missing and args.output_file:
            # Process missing files on CPU
            process_missing_files(args.audio_dir, args.output_file)
        else:
            # Normal processing
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")

            model, feature_extractor = load_model(device)

            assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"

            results, error_files = process_audio_directory(
                args.audio_dir, model, feature_extractor, device, args.output_file
            )

            if not args.output_file:
                print("\n处理结果:")
                for filename, loss in results:
                    print(f"{filename}: {loss:.4f}")

    except Exception as e:
        print(f"错误发生: {str(e)}")
    finally:
        if 'device' in locals() and device == 'cuda':
            torch.cuda.empty_cache()
        print("显存已清理")

        # Restore stdout if redirected
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()