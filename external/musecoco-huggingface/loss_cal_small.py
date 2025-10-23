import torch
import torchaudio
import os
import argparse
import sys
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import functional as F

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model = model.to(device)
    model.decoder = model.decoder.to(dtype=torch.float32)
    model.audio_encoder = model.audio_encoder.to(dtype=torch.float32)
    compression_ratio = model.config.audio_encoder.frame_rate / model.config.audio_encoder.sampling_rate
    
    print(f"压缩比率: {compression_ratio:.4f} (每音频秒生成{model.config.audio_encoder.frame_rate}个tokens)")

    return model, processor

def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ):

    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    ce = torch.zeros([], device=targets.device)
    ce_per_codebook = []
    for k in range(K):
        logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
        targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]
        q_ce = F.cross_entropy(ce_logits, ce_targets)
        ce += q_ce
        ce_per_codebook.append(q_ce.detach())
    # 计算各码本平均交叉熵
    ce = ce / K
    return ce, ce_per_codebook

def process_single_audio(audio_path, model, processor, device, loss_type="cross_entropy"):
    try:
        audio, sr = audio_read(audio_path)

        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to(device)
        original_length = audio.shape[-1]

        # Calculate frame rate from model config
        frame_rate = model.config.audio_encoder.frame_rate
        valid_token_length = int((original_length / sr) * frame_rate) - 1

        with torch.no_grad():
            # 添加调试打印
            print(f"开始编码音频: {audio.shape}")
            encoded = model.audio_encoder.encode(audio.unsqueeze(0))  # 添加batch维度
            print(f"编码结果类型: {type(encoded)}")
            print(f"编码结果结构: {encoded}")
            
            # 使用 .audio_codes 属性
            tokens = encoded.audio_codes.long()
            
            print(f"tokens 形状: {tokens.shape}")

        # 处理4维张量 [B, C, K, T]，其中C是新增的维度
        B, C, K, T = tokens.shape
        input_tokens = tokens[:, :, :, :-1].contiguous()
        target_tokens = tokens[:, :, :, 1:].contiguous()

        # 创建掩码
        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                mask[b, c, :, :valid_token_length] = True

        # Calculate Loss without conditioning
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                # 调整decoder的输入形状
                input_tokens_reshaped = input_tokens.view(B, C * K, -1)
                model.decoder = model.decoder.to(dtype=torch.float32)
                outputs = model.decoder(input_tokens_reshaped)
                
                # Access the logits attribute of the CausalLMOutputWithCrossAttentions object
                output_logits = outputs.logits

            if loss_type == "cross_entropy":
                # 调整输出形状以匹配目标
                logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
                ce, ce_per_codebook = _compute_cross_entropy(
                    logits.view(B * C, K, -1, output_logits.size(-1)), 
                    target_tokens.view(B * C, K, -1), 
                    mask.view(B * C, K, -1))
                return ce.item()
            else:  # MSE loss
                # 调整MSE损失计算
                sample_logits = output_logits.view(B, C, K, -1, output_logits.size(-1))[0, 0, :, :valid_token_length, :].reshape(-1, output_logits.size(-1))
                sample_target = target_tokens[0, 0, :, :valid_token_length].reshape(-1)
                # 计算MSE损失 - 将目标转换为one-hot向量
                target_one_hot = torch.zeros_like(sample_logits)
                target_one_hot.scatter_(1, sample_target.unsqueeze(1), 1.0)
                loss = torch.nn.functional.mse_loss(
                    torch.softmax(sample_logits, dim=1),
                    target_one_hot,
                    reduction='mean'
                )
                return loss.item()

    except Exception as e:
        print(f"处理 {os.path.basename(audio_path)} 时出错: {str(e)}")
        return None

def process_audio_directory(audio_dir, model, processor, device, output_file=None, loss_type="cross_entropy"):
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"目录 {audio_dir} 中没有WAV/MP3文件")

    print(f"\n开始处理 {len(audio_files)} 个音频 (损失类型: {loss_type})")
    results = []
    error_files = [] 

    # Open output file if specified
    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"结果将保存到: {output_file}")

    for filename in tqdm(audio_files, desc="处理进度"):
        audio_path = os.path.join(audio_dir, filename)
        loss = process_single_audio(audio_path, model, processor, device, loss_type)
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

def process_missing_files(audio_dir, results_file, loss_type="cross_entropy"):
    """处理缺失的文件（使用CPU）"""
    missing_files = get_missing_files(audio_dir, results_file)

    if not missing_files:
        print("没有缺失的文件需要处理")
        return

    print(f"发现 {len(missing_files)} 个未处理的文件，将使用CPU处理 (损失类型: {loss_type})")

    # Print all missing filenames before processing
    print("\n未处理的文件列表:")
    for filename in missing_files:
        print(f"{filename}")
    print("\n开始处理这些文件...\n")

    # Load model on CPU
    device = "cpu"
    print(f"Using device: {device}")
    model, processor = load_model(device)

    # Open results file in append mode
    with open(results_file, 'a') as out_file:
        for filename in tqdm(missing_files, desc="CPU处理进度"):
            audio_path = os.path.join(audio_dir, filename)
            loss = process_single_audio(audio_path, model, processor, device, loss_type)
            if loss is not None:
                out_file.write(f"{filename}: {loss:.4f}\n")
                out_file.flush()  # Ensure data is written immediately

def main():
    parser = argparse.ArgumentParser(description="计算音频文件的loss值 (使用musicgen-small模型)")
    parser.add_argument("--audio_dir", type=str, default="pitch_out_wav",
                        help="音频文件目录")
    parser.add_argument("--output_file", type=str, default="results_small.txt",
                        help="输出结果文件路径")
    parser.add_argument("--process_missing", action="store_true",
                        help="处理缺失的文件（使用CPU）")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU处理所有文件")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "mse"], default="cross_entropy",
                        help="损失函数类型: cross_entropy或mse")

    args = parser.parse_args()

    # Redirect stdout to file if output_file is specified and not processing missing files
    original_stdout = None
    if args.output_file and not args.process_missing:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        if args.process_missing and args.output_file:
            # Process missing files on CPU
            process_missing_files(args.audio_dir, args.output_file, args.loss_type)
        else:
            # Normal processing
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")

            model, processor = load_model(device)

            assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"

            results, error_files = process_audio_directory(
                args.audio_dir, model, processor, device, args.output_file, args.loss_type
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