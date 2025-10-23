import os
import torch
import torchaudio
import argparse
import sys
import librosa 
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import functional as F
import numpy as np
import pandas as pd

def setup_device(force_cpu=False):
    """Setup and return the appropriate device (CUDA/CPU)"""
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def load_model(device):
    """Load the MusicGen medium model"""
    model_name = "facebook/musicgen-medium"
    print(f"Loading model: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    return processor, model

def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reduction="mean"
    ):
    """Compute cross entropy loss with mask"""
    # logits: (B, K, T, vocab_size)
    # targets: (B, K, T)
    # mask: (B, K, T)
    
    B, K, T, vocab_size = logits.shape
    
    # Flatten for easier computation
    logits_flat = logits.view(-1, vocab_size)  # (B*K*T, vocab_size)
    targets_flat = targets.view(-1)  # (B*K*T,)
    mask_flat = mask.view(-1)  # (B*K*T,)
    
    # Compute cross entropy for all positions
    ce_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*K*T,)
    
    # Apply mask
    ce_flat = ce_flat * mask_flat.float()
    
    if reduction == "none":
        # Reshape back to (B, K, T)
        ce_per_token = ce_flat.view(B, K, T)
        return ce_per_token
    elif reduction == "mean":
        # Only average over valid (masked) positions
        valid_count = mask_flat.sum()
        if valid_count > 0:
            return ce_flat.sum() / valid_count
        else:
            return torch.tensor(0.0, device=logits.device)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path=None):
    """Save per-token losses to CSV file"""
    try:
        ce_np = ce_per_token.detach().cpu().numpy()
        avg_records = []

        if ce_np.ndim == 3:  # (B, K, T)
            B, K, T = ce_np.shape
            for b in range(B):
                for t in range(T):
                    token_avg = float(ce_np[b, :, t].mean())
                    avg_records.append({'token_position': t, 'avg_loss_value': token_avg})
        elif ce_np.ndim == 2:  # (K, T)
            K, T = ce_np.shape
            for t in range(T):
                token_avg = float(ce_np[:, t].mean())
                avg_records.append({'token_position': t, 'avg_loss_value': token_avg})

        if avg_records:
            # Create output directory
            os.makedirs(token_output_dir, exist_ok=True)
            
            # Get filename without extension
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Create CSV path
            if relative_path and relative_path != '.':
                csv_subdir = os.path.join(token_output_dir, relative_path)
                os.makedirs(csv_subdir, exist_ok=True)
                csv_path = os.path.join(csv_subdir, f"{base_name}_per_token.csv")
            else:
                csv_path = os.path.join(token_output_dir, f"{base_name}_per_token.csv")
            
            avg_df = pd.DataFrame(avg_records)
            avg_df.to_csv(csv_path, index=False)
            print(f"Saved {len(avg_records)} records to {csv_path}")
            overall_mean = avg_df['avg_loss_value'].mean()
            print(f"Overall average loss: {overall_mean:.6f}")
            return overall_mean
        else:
            print(f"No data to save: {audio_path}")
            return None
    except Exception as e:
        print(f"Error saving CSV for {audio_path}: {str(e)}")
        return None

def process_single_audio(audio_path, model, processor, device, reduction="mean", token_output_dir=None, relative_path=None):
    """Process a single audio file"""
    try:
        # Read audio
        if device.type == "cuda":
            audio_data, sample_rate = audio_read(audio_path)
            audio_data = audio_data.mean(dim=0, keepdim=True)
            if sample_rate != 32000:
                audio_data = torchaudio.functional.resample(audio_data, sample_rate, 32000)
            audio_data = audio_data.to(device)
            valid_token_length = min(audio_data.shape[-1] // 320, 1500)
        else:
            audio_data, sample_rate = librosa.load(audio_path, sr=32000, mono=True)
            audio_data = torch.tensor(audio_data).unsqueeze(0).to(device)
            valid_token_length = min(audio_data.shape[-1] // 320, 1500)
        
        with torch.no_grad():
            # Encode audio using the correct method
            encoded = model.audio_encoder.encode(audio_data.unsqueeze(0))
            tokens = encoded.audio_codes.long()
            
        B, C, K, T = tokens.shape
        input_tokens = tokens[:, :, :, :-1].contiguous()
        target_tokens = tokens[:, :, :, 1:].contiguous()
        
        # Create mask
        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                mask[b, c, :, :valid_token_length] = True
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                input_tokens_reshaped = input_tokens.view(B, C * K, -1)
                model.decoder = model.decoder.to(dtype=torch.float32)
                outputs = model.decoder(input_tokens_reshaped)
                output_logits = outputs.logits
            
            logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
            
            if reduction == "none":
                # Compute per-token loss
                ce_per_token = _compute_cross_entropy(
                    logits.view(B * C, K, -1, output_logits.size(-1)), 
                    target_tokens.view(B * C, K, -1), 
                    mask.view(B * C, K, -1),
                    reduction=reduction)
                
                if token_output_dir:
                    # Save as CSV and return average loss
                    avg_loss = save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path)
                    return avg_loss if avg_loss is not None else ce_per_token.mean().item()
                else:
                    return ce_per_token.mean().item()
            else:
                # Compute average loss
                ce_loss = _compute_cross_entropy(
                    logits.view(B * C, K, -1, output_logits.size(-1)), 
                    target_tokens.view(B * C, K, -1), 
                    mask.view(B * C, K, -1),
                    reduction=reduction)
                return ce_loss.item()
                
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def process_audio_directory(audio_dir, model, processor, device, output_file=None, reduction="mean", token_output_dir=None):
    """Process all audio files in the directory with specific suffixes"""
    # Find all audio files with the specified suffixes
    target_suffixes = ['_tk100.wav', '_tk150.wav', '_tk200.wav']
    audio_files = []
    
    for file in os.listdir(audio_dir):
        if any(file.endswith(suffix) for suffix in target_suffixes):
            audio_files.append(os.path.join(audio_dir, file))
    
    audio_files.sort()
    
    print(f"Found {len(audio_files)} audio files with target suffixes")
    
    results = []
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        relative_path = os.path.relpath(os.path.dirname(audio_path), audio_dir)
        if relative_path == '.':
            relative_path = None
            
        avg_loss = process_single_audio(audio_path, model, processor, device, reduction, token_output_dir, relative_path)
        
        if avg_loss is not None:
            filename = os.path.basename(audio_path)
            results.append((filename, avg_loss))
            print(f"Processed: {filename} - Avg Loss: {avg_loss:.6f}")
    
    # Save summary results
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Shuffle Random Token Loss Analysis Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total files processed: {len(results)}\n")
            f.write(f"Target suffixes: {', '.join(target_suffixes)}\n")
            f.write("\n")
            
            if results:
                avg_losses = [loss for _, loss in results]
                f.write(f"Overall average loss: {np.mean(avg_losses):.6f}\n")
                f.write(f"Loss std deviation: {np.std(avg_losses):.6f}\n")
                f.write(f"Min loss: {np.min(avg_losses):.6f}\n")
                f.write(f"Max loss: {np.max(avg_losses):.6f}\n")
                f.write("\n")
                
                f.write("Per-file results:\n")
                f.write("-" * 30 + "\n")
                for filename, loss in results:
                    f.write(f"{filename}: {loss:.6f}\n")
        
        print(f"\nSummary saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compute per-token loss for shuffle random audio files')
    parser.add_argument('--input_dir', type=str, default='shutter_shuffle_random',
                       help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='shutter_shuffle_random_results_medium',
                       help='Output directory for results')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--reduction', type=str, default='none', choices=['mean', 'none'],
                       help='Loss reduction method')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device(args.force_cpu)
    
    # Load model
    processor, model = load_model(device)
    
    # Set output file path
    output_file = os.path.join(args.output_dir, "result_shuffle.txt")
    
    # Process audio files
    results = process_audio_directory(
        args.input_dir, model, processor, device, 
        output_file=output_file, 
        reduction=args.reduction, 
        token_output_dir=args.output_dir if args.reduction == 'none' else None
    )
    
    print(f"\nProcessing complete! Processed {len(results)} files.")
    print(f"Results saved to: {output_file}")
    if args.reduction == 'none':
        print(f"Individual token results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()