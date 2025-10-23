#!/usr/bin/env python3
"""
AudioCraft Audio Tokenizer
使用现有的 audiocraft 环境对音频进行 tokenization
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path

# 添加 audiocraft 路径
sys.path.insert(0, '/home/evev/asap-dataset/audiocraft')

from audiocraft.models import CompressionModel

class AudioCraftTokenizer:
    def __init__(self, model_name='facebook/encodec_24khz', device='cuda'):
        """
        初始化 AudioCraft tokenizer
        
        Args:
            model_name: 模型名称，默认使用 encodec_24khz
            device: 设备，默认使用 cuda
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载压缩模型"""
        print(f"Loading AudioCraft model: {self.model_name}")
        try:
            self.model = CompressionModel.get_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_audio(self, audio_path, target_sr=24000):
        """加载音频文件"""
        try:
            # 使用 torchaudio 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样到目标采样率
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            return waveform.to(self.device)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def tokenize_audio(self, audio_path):
        """对音频进行 tokenization"""
        # 加载音频
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        # 添加 batch 维度
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, T]
        
        try:
            with torch.no_grad():
                # 使用模型编码音频
                encoded_frames = self.model.encode(waveform)
                
                # 提取 codes (tokens)
                if hasattr(encoded_frames, 'codes'):
                    codes = encoded_frames.codes
                else:
                    codes = encoded_frames[0]  # 有些版本返回 tuple
                
                # 转换为 numpy 数组
                if isinstance(codes, torch.Tensor):
                    codes_np = codes.cpu().numpy()
                else:
                    codes_np = codes
                
                return codes_np
        except Exception as e:
            print(f"Error tokenizing audio {audio_path}: {e}")
            return None
    
    def process_directory(self, input_dir, output_dir, audio_extensions=None):
        """处理整个目录的音频文件"""
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        # 遍历所有音频文件
        for audio_file in input_path.rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                print(f"Processing: {audio_file}")
                
                # 生成输出文件路径
                relative_path = audio_file.relative_to(input_path)
                output_file = output_path / relative_path.with_suffix('.npy')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 检查是否已存在
                if output_file.exists():
                    print(f"  Skipping (already exists): {output_file}")
                    continue
                
                # 进行 tokenization
                tokens = self.tokenize_audio(str(audio_file))
                
                if tokens is not None:
                    # 保存为 .npy 文件
                    np.save(output_file, tokens)
                    print(f"  Saved: {output_file}")
                    processed_count += 1
                else:
                    print(f"  Failed: {audio_file}")
                    failed_count += 1
        
        print(f"\nProcessing complete:")
        print(f"  Processed: {processed_count} files")
        print(f"  Failed: {failed_count} files")
        
        return processed_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='AudioCraft Audio Tokenizer')
    parser.add_argument('input_dir', help='Input directory containing audio files')
    parser.add_argument('output_dir', help='Output directory for .npy token files')
    parser.add_argument('--model', default='facebook/encodec_24khz', 
                       help='AudioCraft model name (default: facebook/encodec_24khz)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.wav', '.mp3', '.flac', '.m4a'],
                       help='Audio file extensions to process')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    try:
        # 创建 tokenizer
        tokenizer = AudioCraftTokenizer(model_name=args.model, device=args.device)
        
        # 处理目录
        processed, failed = tokenizer.process_directory(
            args.input_dir, args.output_dir, args.extensions
        )
        
        if failed > 0:
            return 1
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())