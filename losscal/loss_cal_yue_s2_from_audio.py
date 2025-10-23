import os
import csv
import argparse
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchaudio
import librosa
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append('/home/evev/asap-dataset')
sys.path.append('/home/evev/asap-dataset/external/YuEGP/inference')
sys.path.append('/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer')

from codecmanipulator import CodecManipulator
from models.soundstream_hubert_new import SoundStream

class XCodecAudioTokenizer:
    def __init__(self, stage=2, device="cuda", verbose=True):
        self.stage = stage
        self.device = device
        self.verbose = verbose
        
        # 根据stage设置codebook数量
        if stage == 1:
            self.n_quantizer = 1
        elif stage == 2:
            self.n_quantizer = 8
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        # 初始化CodecManipulator with required parameters
        self.codec_manipulator = CodecManipulator(
            codec_type="xcodec", 
            quantizer_begin=0, 
            n_quantizer=self.n_quantizer
        )
        
        # Load XCodec model
        self.model = self._load_xcodec_model()
        
    def _load_xcodec_model(self):
        """Load XCodec SoundStream model"""
        try:
            model = SoundStream(
                n_filters=32,
                D=128,
                ratios=[8, 5, 4, 2],
                sample_rate=16000,   # 与 hop=320 -> 50 fps 对齐
                bins=1024,
                normalize=True,
                causal=True,
            )
            model_path = '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'
            if os.path.exists(model_path):
                # 关键：PyTorch 2.6+ 需显式禁用 weights_only 安全模式
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                except TypeError:
                    checkpoint = torch.load(model_path, map_location='cpu')
                state = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
                model.load_state_dict(state, strict=False)
                if self.verbose:
                    print(f"Loaded checkpoint from {model_path}")
            model = model.to(self.device)
            model.eval()
            if self.verbose:
                print(f"Loaded XCodec model on {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load XCodec model: {e}")
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            if self.device == "cuda":
                # Use torchaudio for GPU processing
                audio, sr = torchaudio.load(audio_path)
                # Convert to mono
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                # Resample to 16kHz（与 hop=320 -> 50fps 对齐）
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, sr, 16000)
            else:
                # Use librosa for CPU processing
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                audio = torch.tensor(audio).unsqueeze(0)
            audio = audio.to(self.device)
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            raise
    
    def encode_audio_to_tokens(self, audio_path: str) -> np.ndarray:
        """Encode audio to tokens using XCodec"""
        try:
            audio = self.load_audio(audio_path)
            if self.model is None:
                raise RuntimeError("XCodec model is not initialized.")

            with torch.no_grad():
                if hasattr(self.model, 'encode'):
                    # 关键：按照期望码本数传入带宽（每个码本 0.5 kbps）
                    target_bw_kbps = 0.5 * self.n_quantizer
                    encoded = self.model.encode(audio.unsqueeze(0), target_bw=target_bw_kbps)
                else:
                    encoded = self.model(audio.unsqueeze(0))

                if self.verbose:
                    print(f"Model.encode output type: {type(encoded)}")

                # 统一取出 codes
                if isinstance(encoded, torch.Tensor):
                    tokens = encoded
                elif isinstance(encoded, dict) and ('codes' in encoded or 'audio_codes' in encoded):
                    tokens = encoded.get('codes', encoded.get('audio_codes'))
                elif hasattr(encoded, 'codes'):
                    tokens = encoded.codes
                elif hasattr(encoded, 'audio_codes'):
                    tokens = encoded.audio_codes
                elif isinstance(encoded, (tuple, list)) and len(encoded) > 0:
                    tokens = encoded[0]
                else:
                    raise RuntimeError("Cannot extract codebook tokens from model output.")

                if self.verbose and isinstance(tokens, torch.Tensor):
                    try:
                        print(f"Raw tokens tensor shape: {tuple(tokens.shape)}")
                    except Exception:
                        pass

                tokens_np = tokens.detach().cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)

                # 形状规范化到 (K, T)
                ncb_candidates = (self.n_quantizer, 12)
                if tokens_np.ndim == 3:
                    # 支持三种排列：
                    # 1) (K, B, T): 码本在第0维（本项目RVQ encode返回就是这种）
                    # 2) (B, K, T): 码本在第1维
                    # 3) (B, T, K): 码本在第2维
                    if tokens_np.shape[0] in ncb_candidates:
                        # (K, B, T) -> (K, T)
                        # 取第一个batch
                        tokens_np = tokens_np[:, 0, :]
                    elif tokens_np.shape[1] in ncb_candidates:
                        # (B, K, T) -> (K, T)
                        tokens_np = tokens_np[0]
                    elif tokens_np.shape[2] in ncb_candidates:
                        # (B, T, K) -> (K, T)
                        tokens_np = np.transpose(tokens_np[0], (1, 0))
                    else:
                        # 回退：按 (B, K, T) 处理
                        tokens_np = tokens_np[0]
                elif tokens_np.ndim == 4:
                    # 可能是 (B, C, K, T) 或 (B, C, T, K)
                    if tokens_np.shape[2] in ncb_candidates:
                        tokens_np = tokens_np[0, 0]            # (K, T)
                    elif tokens_np.shape[3] in ncb_candidates:
                        tokens_np = np.transpose(tokens_np[0, 0], (1, 0))  # (T, K) -> (K, T)
                    else:
                        tokens_np = tokens_np[0, 0]
                elif tokens_np.ndim == 2:
                    # (K, T) 或 (T, K)
                    if tokens_np.shape[0] not in ncb_candidates and tokens_np.shape[1] in ncb_candidates:
                        tokens_np = tokens_np.T

                # 只取 stage 对应的码本数
                selected_tokens = tokens_np[:self.n_quantizer]

                if self.verbose:
                    print(f"Encoded audio {os.path.basename(audio_path)}: shape {selected_tokens.shape}")
                return selected_tokens
        except Exception as e:
            print(f"Error encoding audio {audio_path}: {e}")
            raise
    
    def tokens_to_ids(self, tokens: np.ndarray) -> List[int]:
        """Convert tokens to token IDs using CodecManipulator"""
        try:
            # Use CodecManipulator to convert tokens to IDs
            token_ids = self.codec_manipulator.npy2ids(tokens)
            return token_ids.tolist() if isinstance(token_ids, np.ndarray) else token_ids
        except Exception as e:
            print(f"Error converting tokens to IDs: {e}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate loss using YuE-s2 model from audio files")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model_name_or_path", type=str, default="m-a-p/YuE-s2-1B-general", help="Model name or path")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2], help="XCodec stage (1 or 2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Model dtype")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--token", type=str, help="Hugging Face access token")
    
    # 新增内存优化参数
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for processing")
    parser.add_argument("--batch_interval", type=int, default=5, help="Clear cache every N files")
    parser.add_argument("--low_memory", action="store_true", help="Enable low memory mode")
    
    # 新增per-token loss保存参数
    parser.add_argument("--save_per_token", action="store_true", help="Save per-token loss as CSV files")
    parser.add_argument("--token_output_dir", type=str, help="Directory to save per-token CSV files (default: output_dir/tokens)")
    
    return parser.parse_args()

def pick_dtype(arg: str) -> torch.dtype:
    if arg == "float16":
        return torch.float16
    elif arg == "bfloat16":
        return torch.bfloat16
    elif arg == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {arg}")

def load_model(model_name_or_path: str, dtype: torch.dtype, device: str, trust_remote_code: bool, token: str = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        trust_remote_code=trust_remote_code,
        token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype, 
        trust_remote_code=trust_remote_code,
        token=token
    ).to(device)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def compute_per_token_nll(model: AutoModelForCausalLM, token_ids: List[int], device: str) -> torch.Tensor:
    if len(token_ids) < 2:
        return torch.tensor([0.0], device=device)
    
    input_ids = torch.tensor(token_ids[:-1], device=device).unsqueeze(0)
    target_ids = torch.tensor(token_ids[1:], device=device).unsqueeze(0)
    
    outputs = model(input_ids)
    logits = outputs.logits
    
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = target_ids[..., :].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss

def save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path=None, n_quantizer: int = None):
    """
    Save per-token loss as CSV file in the same format as loss_cal_small.py
    - 自动检测展平的一维序列 + 已知 n_quantizer：按帧均值输出
    """
    if relative_path:
        rel_dir = os.path.dirname(relative_path)
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        if rel_dir:
            filename = f"{rel_dir.replace(os.sep, '_')}_{base_filename}"
        else:
            filename = base_filename
    else:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    avg_csv_file = os.path.join(token_output_dir, f"{filename}_tokens_avg.csv")
    os.makedirs(token_output_dir, exist_ok=True)
    
    try:
        # Convert to numpy array
        if isinstance(ce_per_token, torch.Tensor):
            ce_np = ce_per_token.cpu().numpy()
        else:
            ce_np = np.array(ce_per_token)
        
        # 如果是一维展平序列，且提供了 n_quantizer，则按帧聚合
        if ce_np.ndim == 1 and n_quantizer and n_quantizer > 1 and ce_np.shape[0] >= n_quantizer:
            total = ce_np.shape[0]
            remainder = total % n_quantizer
            use_len = total - remainder
            if remainder != 0:
                # 丢弃尾部无法凑满一帧的若干个 token，保证 (T, n_quantizer) 对齐
                ce_np = ce_np[:use_len]
            per_frame = ce_np.reshape(-1, n_quantizer).mean(axis=1)
            values_to_save = per_frame
        else:
            # 视为 Already 逐帧或无法聚合，则原样保存
            values_to_save = ce_np

        # Calculate average per token position (逐帧平均后的曲线)
        avg_records = []
        for t, loss_val in enumerate(values_to_save):
            avg_records.append({
                'token_position': t,
                'avg_loss_value': float(loss_val)
            })
        
        # Save average data
        if avg_records:
            avg_df = pd.DataFrame(avg_records)
            avg_df.to_csv(avg_csv_file, index=False)
            print(f"Saved {len(avg_records)} average records to {avg_csv_file}")
            overall_mean = avg_df['avg_loss_value'].mean()
            print(f"Overall average loss: {overall_mean:.6f}")
            return overall_mean
        else:
            print(f"No data to save: {filename}")
            return None
    except Exception as e:
        print(f"Error saving CSV {filename}: {str(e)}")
        return None

def process_single_audio(audio_path, tokenizer_model, yue_model, device, token_output_dir=None, relative_path=None):
    try:
        # 清理GPU缓存
        if device != "cpu":
            torch.cuda.empty_cache()
        
        # Encode audio to tokens using XCodec
        tokens = tokenizer_model.encode_audio_to_tokens(audio_path)
        
        # Convert tokens to token IDs
        token_ids = tokenizer_model.tokens_to_ids(tokens)
        
        # 分批处理长序列
        max_seq_length = 512  # 限制序列长度（建议设为 n_quantizer 的倍数，默认 512 已满足 512%8==0）
        if len(token_ids) > max_seq_length:
            # 分块处理（步长=chunk_size - n_quantizer，块间重叠 n_quantizer 个 token，用于边界对齐）
            chunk_size = max_seq_length
            step = chunk_size - tokenizer_model.n_quantizer
            overlap = tokenizer_model.n_quantizer  # 用于去重的长度基准
            all_losses = []
            
            # 注意：这里用 len(token_ids) 而不是 len(token_ids) - 1，循环内按长度判断
            for i in range(0, len(token_ids), step):
                chunk_ids = token_ids[i:i + chunk_size]
                if len(chunk_ids) < 2:
                    continue
                chunk_nll = compute_per_token_nll(yue_model, chunk_ids, device)
                
                # 丢弃与上一块重复的预测损失：重叠 n_quantizer 个 token -> 重复 (n_quantizer - 1) 个 NLL
                if i > 0:
                    trim = overlap - 1
                    if chunk_nll.size(0) > trim:
                        chunk_nll = chunk_nll[trim:]
                    else:
                        chunk_nll = chunk_nll.new_zeros(0)
                
                all_losses.extend(chunk_nll.cpu().numpy().tolist())
                # 清理中间结果
                del chunk_nll
                if device != "cpu":
                    torch.cuda.empty_cache()
            
            if all_losses:
                nll = torch.tensor(all_losses, device=device)
                avg_loss = nll.mean().item()
            else:
                avg_loss = 0.0
                nll = torch.tensor([], device=device)
        else:
            # 正常处理短序列
            nll = compute_per_token_nll(yue_model, token_ids, device)
            avg_loss = nll.mean().item()
        
        # 保存per-token数据到CSV（如果指定了token_output_dir）
        if token_output_dir and len(nll) > 0:
            saved_avg = save_tokens_as_csv(nll, audio_path, token_output_dir, relative_path, n_quantizer=tokenizer_model.n_quantizer)
            if saved_avg is not None:
                avg_loss = saved_avg  # 使用CSV保存函数返回的平均值
        
        # 清理变量
        del tokens, token_ids, nll
        if device != "cpu":
            torch.cuda.empty_cache()
        
        return avg_loss
    except Exception as e:
        print(f"Error processing audio {os.path.basename(audio_path)}: {str(e)}")
        if device != "cpu":
            torch.cuda.empty_cache()
        return None

def process_audio_directory(audio_dir, tokenizer_model, yue_model, device, output_file=None, token_output_dir=None, batch_interval=5):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                rel_path = os.path.relpath(os.path.join(root, f), audio_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    print(f"Processing {len(audio_files)} audio files")
    if token_output_dir:
        print(f"Per-token CSV files will be saved to: {token_output_dir}")
    
    results = []
    error_files = []

    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"Results will be saved to: {output_file}")

    for i, rel_filename in enumerate(tqdm(audio_files, desc="Processing")):
        audio_path = os.path.join(audio_dir, rel_filename)
        loss = process_single_audio(audio_path, tokenizer_model, yue_model, device, token_output_dir, rel_filename)
        
        if loss is not None:
            results.append((rel_filename, loss))
            if out_file:
                out_file.write(f"{rel_filename}: {loss:.8f}\n")
                out_file.flush()
        else:
            error_files.append(rel_filename)
        
        # 定期清理GPU缓存
        if (i + 1) % batch_interval == 0 and device != "cpu":
            torch.cuda.empty_cache()
            print(f"\nCleared GPU cache after processing {i + 1} files")

    if out_file:
        out_file.close()

    if error_files:
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Failed files count: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n{len(error_files)} failed files saved to: {error_file_path}")

    return results, error_files

def main():
    args = parse_args()
    
    if args.force_cpu:
        args.device = "cpu"
    
    dtype = pick_dtype(args.dtype)
    
    print(f"Using device: {args.device}")
    print(f"XCodec stage: {args.stage}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # 设置token输出目录
    token_output_dir = None
    if args.save_per_token:
        token_output_dir = args.token_output_dir if args.token_output_dir else os.path.join(args.output_dir, "tokens")
        print(f"Per-token CSV files will be saved to: {token_output_dir}")
    
    # Initialize XCodec tokenizer
    tokenizer_model = XCodecAudioTokenizer(stage=args.stage, device=args.device, verbose=True)
    
    # Load YuE model
    print("Loading YuE model...")
    tokenizer, yue_model = load_model(
        args.model_name_or_path, 
        dtype, 
        args.device, 
        args.trust_remote_code,
        args.token
    )
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    if token_output_dir:
        os.makedirs(token_output_dir, exist_ok=True)
    
    # Process audio directory
    output_file = os.path.join(args.output_dir, "results.txt")
    results, error_files = process_audio_directory(
        args.audio_dir, tokenizer_model, yue_model, args.device, 
        output_file, token_output_dir, args.batch_interval
    )
    
    print(f"\nProcessed {len(results)} files successfully, {len(error_files)} failed")
    
    if results:
        avg_loss = sum(loss for _, loss in results) / len(results)
        print(f"Average loss across all files: {avg_loss:.6f}")
    
    # Clean up GPU memory
    if args.device == 'cuda':
        torch.cuda.empty_cache()
        print("GPU memory cleared")

if __name__ == "__main__":
    main()