#!/usr/bin/env python3
"""
AudioCraft Audio Tokenizer
Tokenize audio using a preinstalled AudioCraft environment
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path

# Add AudioCraft path
sys.path.insert(0, '/home/evev/asap-dataset/audiocraft')

from audiocraft.models import CompressionModel

class AudioCraftTokenizer:
    def __init__(self, model_name='facebook/encodec_24khz', device='cuda'):
        """
        Initialize AudioCraft tokenizer.
        
        Args:
            model_name: Model name, default encodec_24khz
            device: Device to use, default cuda
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load compression model"""
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
        """Load audio file"""
        try:
            # Load with torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to target sample rate
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            return waveform.to(self.device)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def tokenize_audio(self, audio_path):
        """Tokenize audio"""
        # Load audio
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        # Add batch dimension
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, T]
        
        try:
            with torch.no_grad():
                # Encode with model
                encoded_frames = self.model.encode(waveform)
                
                # Extract codes (tokens)
                if hasattr(encoded_frames, 'codes'):
                    codes = encoded_frames.codes
                else:
                    codes = encoded_frames[0]
                
                # Convert to numpy array
                if isinstance(codes, torch.Tensor):
                    codes_np = codes.cpu().numpy()
                else:
                    codes_np = codes
                
                return codes_np
        except Exception as e:
            print(f"Error tokenizing audio {audio_path}: {e}")
            return None
    
    def process_directory(self, input_dir, output_dir, audio_extensions=None):
        """Process all audio files in a directory"""
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        # Iterate audio files
        for audio_file in input_path.rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                print(f"Processing: {audio_file}")
                
                # Build output file path
                relative_path = audio_file.relative_to(input_path)
                output_file = output_path / relative_path.with_suffix('.npy')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if exists
                if output_file.exists():
                    print(f"  Skipping (already exists): {output_file}")
                    continue
                
                # Run tokenization
                tokens = self.tokenize_audio(str(audio_file))
                
                if tokens is not None:
                    # Save as .npy file
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
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    try:
        # Create tokenizer
        tokenizer = AudioCraftTokenizer(model_name=args.model, device=args.device)
        
        # Process directory
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