#!/usr/bin/env python3
"""
XCodec Audio Tokenizer
基于YuEGP的XCodec实现的音频tokenizer
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path

# 添加YuEGP路径
sys.path.insert(0, '/home/evev/asap-dataset/external/YuEGP/inference')
sys.path.insert(0, '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer')
sys.path.insert(0, '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/descriptaudiocodec')

from codecmanipulator import CodecManipulator
from models.soundstream_hubert_new import SoundStream
import yaml

class XCodecTokenizer:
    def __init__(self, stage=2, device='cuda'):
        """
        初始化XCodec tokenizer
        
        Args:
            stage: 1 (1个codebook) 或 2 (8个codebook)
            device: 设备，默认使用cuda
        """
        self.device = device
        self.stage = stage
        
        # 根据stage设置codebook数量
        if stage == 1:
            self.n_quantizer = 1
        elif stage == 2:
            self.n_quantizer = 8
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        # 初始化CodecManipulator
        self.codec_manipulator = CodecManipulator(
            codec_type="xcodec", 
            quantizer_begin=0, 
            n_quantizer=self.n_quantizer
        )
        
        # 加载XCodec模型
        self._load_model()
    
    def _load_model(self):
        """加载XCodec模型"""
        print(f"Loading XCodec model (stage {self.stage})")
        try:
            # 加载配置
            config_path = '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/final_ckpt/config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 创建模型
            self.model = SoundStream(**config['model'])
            
            # 加载权重
            ckpt_path = '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            
            self.model.to(self.device)
            self.model.eval()
            print(f"XCodec model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading XCodec model: {e}")
            raise
    
    def load_audio(self, audio_path, target_sr=16000):
        """加载音频文件"""
        try:
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
        """对音频进行tokenization"""
        # 加载音频
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        # 添加batch维度
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, T]
        
        try:
            with torch.no_grad():
                # 使用XCodec编码
                encoded = self.model.encode(waveform)
                
                # 提取codes
                if hasattr(encoded, 'codes'):
                    codes = encoded.codes
                elif isinstance(encoded, tuple):
                    codes = encoded[0]
                else:
                    codes = encoded
                
                # 只取前n_quantizer个codebook
                if codes.shape[1] > self.n_quantizer:
                    codes = codes[:, :self.n_quantizer, :]
                
                # 转换为numpy
                codes_np = codes.squeeze(0).cpu().numpy()  # [n_quantizer, T]
                
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
                
                # 进行tokenization
                tokens = self.tokenize_audio(str(audio_file))
                
                if tokens is not None:
                    # 保存为.npy文件
                    np.save(output_file, tokens)
                    print(f"  Saved: {output_file} (shape: {tokens.shape})")
                    processed_count += 1
                else:
                    print(f"  Failed: {audio_file}")
                    failed_count += 1
        
        print(f"\nProcessing complete:")
        print(f"  Processed: {processed_count} files")
        print(f"  Failed: {failed_count} files")
        
        return processed_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='XCodec Audio Tokenizer')
    parser.add_argument('input_dir', help='Input directory containing audio files')
    parser.add_argument('output_dir', help='Output directory for .npy token files')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2],
                       help='XCodec stage: 1 (1 codebook) or 2 (8 codebooks)')
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
        # 创建tokenizer
        tokenizer = XCodecTokenizer(stage=args.stage, device=args.device)
        
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