import torch
import torchaudio
import os
import argparse
import sys
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import functional as F
from datasets import load_dataset
import tempfile
import requests
from urllib.parse import urlparse

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

def download_audio_from_url(url, temp_dir):
    """从URL下载音频文件到临时目录"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 从URL获取文件扩展名
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            filename = "audio.wav"  # 默认扩展名
        
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_path
    except Exception as e:
        print(f"下载音频失败: {str(e)}")
        return None

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

def process_hf_dataset(dataset_name, model, processor, device, output_file=None, loss_type="cross_entropy", audio_column="audio", split="train"):
    """处理Hugging Face数据集中的音频文件"""
    try:
        # 加载数据集
        print(f"正在加载数据集: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        print(f"数据集加载完成，共 {len(dataset)} 个样本")
        
        # 检查音频列是否存在
        if audio_column not in dataset.column_names:
            available_columns = dataset.column_names
            print(f"错误: 数据集中没有找到音频列 '{audio_column}'")
            print(f"可用的列: {available_columns}")
            return [], []
        
        print(f"\n开始处理 {len(dataset)} 个音频样本 (损失类型: {loss_type})")
        results = []
        error_files = []
        
        # 创建临时目录用于存储下载的音频文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 打开输出文件
            out_file = None
            if output_file:
                out_file = open(output_file, 'w')
                print(f"结果将保存到: {output_file}")
            
            for idx, sample in enumerate(tqdm(dataset, desc="处理进度")):
                try:
                    # 获取音频数据
                    audio_data = sample[audio_column]
                    
                    # 处理不同的音频数据格式
                    if isinstance(audio_data, dict):
                        # 如果音频数据是字典格式（包含array和sampling_rate）
                        if 'array' in audio_data and 'sampling_rate' in audio_data:
                            # 直接使用音频数组数据
                            audio_array = torch.tensor(audio_data['array'], dtype=torch.float32)
                            if audio_array.dim() == 1:
                                audio_array = audio_array.unsqueeze(0)
                            
                            # 保存为临时文件
                            temp_audio_path = os.path.join(temp_dir, f"sample_{idx}.wav")
                            torchaudio.save(temp_audio_path, audio_array, audio_data['sampling_rate'])
                            
                        elif 'path' in audio_data:
                            # 如果包含文件路径
                            temp_audio_path = audio_data['path']
                        else:
                            print(f"样本 {idx}: 不支持的音频数据格式")
                            error_files.append(f"sample_{idx}")
                            continue
                    elif isinstance(audio_data, str):
                        # 如果是URL或文件路径
                        if audio_data.startswith(('http://', 'https://')):
                            # 下载音频文件
                            temp_audio_path = download_audio_from_url(audio_data, temp_dir)
                            if temp_audio_path is None:
                                error_files.append(f"sample_{idx}")
                                continue
                        else:
                            # 本地文件路径
                            temp_audio_path = audio_data
                    else:
                        print(f"样本 {idx}: 未知的音频数据类型: {type(audio_data)}")
                        error_files.append(f"sample_{idx}")
                        continue
                    
                    # 处理音频文件
                    loss = process_single_audio(temp_audio_path, model, processor, device, loss_type)
                    
                    if loss is not None:
                        sample_name = f"sample_{idx}"
                        results.append((sample_name, loss))
                        # 立即写入文件
                        if out_file:
                            out_file.write(f"{sample_name}: {loss:.8f}\n")
                            out_file.flush()
                    else:
                        error_files.append(f"sample_{idx}")
                        
                except Exception as e:
                    print(f"处理样本 {idx} 时出错: {str(e)}")
                    error_files.append(f"sample_{idx}")
            
            # 关闭输出文件
            if out_file:
                out_file.close()
        
        # 保存错误文件列表
        if error_files and device == "cuda":
            error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
            with open(error_file_path, 'w') as f:
                f.write(f"处理失败的样本数量: {len(error_files)}\n\n")
                for sample_name in error_files:
                    f.write(f"{sample_name}\n")
            print(f"\n处理失败的 {len(error_files)} 个样本已保存到: {error_file_path}")
        
        return results, error_files
        
    except Exception as e:
        print(f"处理数据集时出错: {str(e)}")
        return [], []

def main():
    parser = argparse.ArgumentParser(description="计算Hugging Face数据集中音频文件的loss值 (使用musicgen-small模型)")
    parser.add_argument("--dataset_name", type=str, default="Dev4IC/garbage-dataset", 
                        help="Hugging Face数据集名称")
    parser.add_argument("--audio_column", type=str, default="audio",
                        help="数据集中音频列的名称")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分割 (train/test/validation)")
    parser.add_argument("--output_file", type=str, default="results_hf_small.txt",
                        help="输出结果文件路径")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU处理所有文件")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "mse"], default="cross_entropy",
                        help="损失函数类型: cross_entropy或mse")

    args = parser.parse_args()

    # 重定向stdout到日志文件
    original_stdout = None
    if args.output_file:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        device = setup_device(args.force_cpu)
        print(f"Using device: {device}")

        model, processor = load_model(device)

        results, error_files = process_hf_dataset(
            args.dataset_name, model, processor, device, 
            args.output_file, args.loss_type, args.audio_column, args.split
        )

        if not args.output_file:
            print("\n处理结果:")
            for sample_name, loss in results:
                print(f"{sample_name}: {loss:.4f}")

        print(f"\n处理完成! 成功处理 {len(results)} 个样本，失败 {len(error_files)} 个样本")

    except Exception as e:
        print(f"错误发生: {str(e)}")
    finally:
        if 'device' in locals() and device == 'cuda':
            torch.cuda.empty_cache()
        print("显存已清理")

        # 恢复stdout
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()