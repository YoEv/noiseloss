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
            
            with torch.no_grad():
                # Encode audio to tokens
                codes = self.model.encode(audio.unsqueeze(0), target_bw=6.0)
                
                # codes shape: [n_quantizer, batch, time]
                # We need [n_quantizer, time] for single audio
                if codes.dim() == 3:
                    codes = codes.squeeze(1)  # Remove batch dimension
                
                # Convert to numpy
                tokens = codes.cpu().numpy()
                
                if self.verbose:
                    print(f"Encoded audio to tokens: {tokens.shape}")
                
                return tokens
        except Exception as e:
            print(f"Error encoding audio {audio_path}: {e}")
            raise
    
    def tokens_to_ids(self, tokens: np.ndarray) -> List[int]:
        """Convert tokens to token IDs using CodecManipulator"""
        try:
            # Use CodecManipulator to convert tokens to IDs
            token_ids = self.codec_manipulator.npy2ids(tokens)
            return token_ids
        except Exception as e:
            print(f"Error converting tokens to IDs: {e}")
            raise

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
    将per-token loss保存为CSV文件，按照loss_cal_small.py的格式
    """
    if relative_path:
        # 使用相对路径信息，将路径分隔符替换为下划线避免文件系统问题
        rel_dir = os.path.dirname(relative_path)
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        if rel_dir:
            # 将路径分隔符替换为下划线，创建唯一的文件名
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
            print(f"保存了 {len(avg_records)} 条平均值记录到 {avg_csv_file}")
            overall_mean = avg_df['avg_loss_value'].mean()
            print(f"总体平均loss: {overall_mean:.6f}")
            return overall_mean
        else:
            print(f"没有数据可保存: {filename}")
            return None
    except Exception as e:
        print(f"保存CSV时出错 {filename}: {str(e)}")
        return None

def process_single_audio(audio_path, tokenizer_model, yue_model, device, token_output_dir=None, relative_path=None):
    """
    处理单个音频文件，计算per-token loss
    """
    try:
        # Encode audio to tokens
        tokens = tokenizer_model.encode_audio_to_tokens(audio_path)
        
        # Convert tokens to token IDs
        token_ids = tokenizer_model.tokens_to_ids(tokens)
        
        if len(token_ids) < 2:
            print(f"Token sequence too short for {os.path.basename(audio_path)}")
            return None
        
        # Compute per-token NLL
        per_token_loss = compute_per_token_nll(yue_model, token_ids, device)
        
        # Save per-token loss if requested
        if token_output_dir:
            avg_loss = save_tokens_as_csv(
                per_token_loss, audio_path, token_output_dir, 
                relative_path, tokenizer_model.n_quantizer
            )
            return avg_loss if avg_loss is not None else per_token_loss.mean().item()
        else:
            return per_token_loss.mean().item()
            
    except Exception as e:
        print(f"处理音频时出错 {os.path.basename(audio_path)}: {str(e)}")
        return None

def process_audio_directory(audio_dir, tokenizer_model, yue_model, device, output_file=None, token_output_dir=None, batch_interval=5):
    """
    处理音频目录，计算所有音频文件的loss
    """
    # Find all audio files
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
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

def process_dataset_batch():
    """
    批量处理指定的数据集目录
    """
    # 定义要处理的数据集目录
    dataset_dirs = [
        "+Dataset/Phase5_1",
        "ShutterStock_32k_ori", 
        "asap_ori",
        "Unconditional_ori",
        "noise_color_ori"
    ]
    
    # 输出基础目录
    output_base = "+Loss/Phase5_1_yueGP"
    
    # 模型参数
    model_name_or_path = "m-a-p/YuE-s2-1B-general"
    stage = 2
    device = "cuda"
    dtype = pick_dtype("float16")
    
    print("开始批量计算YuE-s2 per token loss...")
    print(f"使用模型: {model_name_or_path}")
    print(f"XCodec stage: {stage}")
    print(f"设备: {device}")
    
    # 初始化模型（只初始化一次）
    print("\n=== 初始化模型 ===")
    tokenizer_model = XCodecAudioTokenizer(stage=stage, device=device, verbose=True)
    print("Loading YuE model...")
    tokenizer, yue_model = load_model(model_name_or_path, dtype, device, True)
    
    # 处理每个数据集目录
    for dataset_dir in dataset_dirs:
        print(f"\n=== 处理数据集: {dataset_dir} ===")
        
        if not os.path.exists(dataset_dir):
            print(f"警告: 目录 {dataset_dir} 不存在，跳过")
            continue
        
        # 获取所有子目录
        subdirs = []
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        
        if not subdirs:
            print(f"警告: 目录 {dataset_dir} 中没有子目录，跳过")
            continue
            
        print(f"找到 {len(subdirs)} 个子目录: {subdirs}")
        
        # 处理每个子目录
        for subdir in subdirs:
            print(f"\n--- 处理子目录: {subdir} ---")
            
            input_dir = os.path.join(dataset_dir, subdir)
            
            # 检查是否有音频文件
            audio_count = 0
            for root, dirs, files in os.walk(input_dir):
                for f in files:
                    if f.lower().endswith(('.wav', '.mp3')):
                        audio_count += 1
            
            if audio_count == 0:
                print(f"警告: 目录 {input_dir} 中没有找到音频文件，跳过")
                continue
            
            print(f"找到 {audio_count} 个音频文件")
            
            # 创建输出目录
            dataset_name = os.path.basename(dataset_dir)
            output_dir = os.path.join(output_base, f"{dataset_name}_{subdir}_token_loss_yue_s2")
            output_file = os.path.join(output_dir, "results.txt")
            token_output_dir = os.path.join(output_dir, "per_token")
            
            print(f"输入目录: {input_dir}")
            print(f"输出目录: {output_dir}")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(token_output_dir, exist_ok=True)
            
            try:
                # 处理音频文件
                results, error_files = process_audio_directory(
                    input_dir, tokenizer_model, yue_model, device, 
                    output_file, token_output_dir, batch_interval=5
                )
                
                if results:
                    avg_loss = sum(loss for _, loss in results) / len(results)
                    print(f"✓ 完成: {output_dir}")
                    print(f"  处理成功: {len(results)} 个文件")
                    print(f"  处理失败: {len(error_files)} 个文件")
                    print(f"  平均loss: {avg_loss:.6f}")
                else:
                    print(f"✗ 失败: {output_dir} - 没有成功处理的文件")
                    
            except Exception as e:
                print(f"✗ 失败: {output_dir} - {str(e)}")
            
            # 清理GPU内存
            if device == 'cuda':
                torch.cuda.empty_cache()
                print("GPU内存已清理")
    
    print("\n=== 所有处理完成 ===")
    print(f"输出目录: {output_base}")
    
    # 最终清理
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("最终GPU内存清理完成")

def main():
    """
    主函数 - 直接运行批量处理
    """
    try:
        process_dataset_batch()
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"\n处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()