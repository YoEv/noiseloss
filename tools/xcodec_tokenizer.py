
import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, '/home/evev/asap-dataset/external/YuEGP/inference')
sys.path.insert(0, '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer')
sys.path.insert(0, '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/descriptaudiocodec')

from codecmanipulator import CodecManipulator
from models.soundstream_hubert_new import SoundStream
import yaml

class XCodecTokenizer:
    def __init__(self, stage=2, device='cuda'):
        self.device = device
        self.stage = stage
        
        if stage == 1:
            self.n_quantizer = 1
        elif stage == 2:
            self.n_quantizer = 8
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        self.codec_manipulator = CodecManipulator(
            codec_type="xcodec", 
            quantizer_begin=0, 
            n_quantizer=self.n_quantizer
        )
        
        self._load_model()
    
    def _load_model(self):
        print(f"Loading XCodec model (stage {self.stage})")
        config_path = '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/final_ckpt/config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model = SoundStream(**config['model'])
        
        ckpt_path = '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        
        self.model.to(self.device)
        self.model.eval()
        print(f"XCodec model loaded successfully on {self.device}")

    def load_audio(self, audio_path, target_sr=16000):
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        return waveform.to(self.device)
 
    def tokenize_audio(self, audio_path):
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, T]
        
        with torch.no_grad():
            encoded = self.model.encode(waveform)
            
            if hasattr(encoded, 'codes'):
                codes = encoded.codes
            elif isinstance(encoded, tuple):
                codes = encoded[0]
            else:
                codes = encoded
            
            if codes.shape[1] > self.n_quantizer:
                codes = codes[:, :self.n_quantizer, :]
            
            codes_np = codes.squeeze(0).cpu().numpy()  # [n_quantizer, T]
            
            return codes_np

    
    def process_directory(self, input_dir, output_dir, audio_extensions=None):
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
                
                relative_path = audio_file.relative_to(input_path)
                output_file = output_path / relative_path.with_suffix('.npy')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                if output_file.exists():
                    print(f"  Skipping (already exists): {output_file}")
                    continue
                
                tokens = self.tokenize_audio(str(audio_file))
                
                if tokens is not None:
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
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    tokenizer = XCodecTokenizer(stage=args.stage, device=args.device)
    
    processed, failed = tokenizer.process_directory(
        args.input_dir, args.output_dir, args.extensions
    )
    
    if failed > 0:
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())