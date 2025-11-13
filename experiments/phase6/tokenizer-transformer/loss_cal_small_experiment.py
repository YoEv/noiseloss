import torch
import torchaudio
import os
import argparse
import sys
import librosa 
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
# from torch.nn import functional as F

import pandas as pd
import torch.nn.functional as F





def setup_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model = model.to(device)
    model.eval()
    

    return model, processor


def log_shape(name, x):
    return


def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reduction="mean"
    ):
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    if reduction == "none":
        ce_per_token = torch.zeros_like(targets, dtype=torch.float32)
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))
            targets_k = targets[:, k, ...].contiguous().view(-1)
            mask_k = mask[:, k, ...].contiguous().view(-1)
            ce_flat = F.cross_entropy(logits_k, targets_k, reduction="none")
            ce_masked = ce_flat * mask_k.float()
            ce_per_token[:, k, :] = ce_masked.view(B, T)
        return ce_per_token
    else:
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))
            targets_k = targets[:, k, ...].contiguous().view(-1)
            mask_k = mask[:, k, ...].contiguous().view(-1)
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        ce = ce / K
        return ce, ce_per_codebook

def save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path=None):
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
    
    ce_np = ce_per_token.cpu().numpy()
    
    avg_records = []
    
    if ce_np.ndim == 3:
        B, K, T = ce_np.shape
        for b in range(B):
            for t in range(T):
                token_avg = float(ce_np[b, :, t].mean())
                avg_records.append({'token_position': t, 'avg_loss_value': token_avg})
    elif ce_np.ndim == 2:
        K, T = ce_np.shape
        for t in range(T):
            token_avg = float(ce_np[:, t].mean())
            avg_records.append({'token_position': t, 'avg_loss_value': token_avg})
    
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

def process_single_audio(audio_path, model, processor, device, reduction="mean", token_output_dir=None, relative_path=None, hook_layers=2):
    if device == "cuda":
        audio, sr = audio_read(audio_path)
        audio = audio.mean(dim=0, keepdim=True)
        if sr != 32000:
            audio = torchaudio.functional.resample(audio, sr, 32000)
        audio = audio.to(device)
        valid_token_length = min(audio.shape[-1] // 320, 1500)
    else:
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        audio = torch.tensor(audio).unsqueeze(0).to(device)
        valid_token_length = min(audio.shape[-1] // 320, 1500)

    print(f"[audio] sr={sr}, valid_token_length={valid_token_length}")
    log_shape("audio(1,C=T)", audio)

    with torch.no_grad():
        encoded = model.audio_encoder.encode(audio.unsqueeze(0)) 
        # HuggingFace MusicGen audio_encoder returns an object with audio_codes
        tokens = encoded.audio_codes.long()
        print(f"[encode] encoded_type={type(encoded).__name__}")

    # Encodec HF implementation usually returns (B, C, K, T)
    log_shape("tokens(B,C,K,T)", tokens)
    B, C, K, T = tokens.shape
    print(f"[tokens dims] B={B}, C={C}, K={K}, T={T}")

    input_tokens = tokens[:, :, :, :-1].contiguous()
    target_tokens = tokens[:, :, :, 1:].contiguous()
    log_shape("input_tokens(B,C,K,T-1)", input_tokens)
    log_shape("target_tokens(B,C,K,T-1)", target_tokens)

    mask = torch.zeros_like(target_tokens, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            mask[b, c, :, :valid_token_length] = True
    log_shape("mask(B,C,K,T-1)", mask)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', enabled=False):
            input_tokens_reshaped = input_tokens.view(B, C * K, -1)
            log_shape("decoder_input(B, C*K, T-1)", input_tokens_reshaped)
            model.decoder = model.decoder.to(dtype=torch.float32)

            outputs = model.decoder(input_tokens_reshaped)
            output_logits = outputs.logits

            log_shape("output_logits(B, C*K, T-1, card)", output_logits)

        # reshape back to (B, C, K, T-1, card)
        logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
        log_shape("logits(B,C,K,T-1,card)", logits)

        if reduction == "none":
            ce_per_token = _compute_cross_entropy(
                logits.view(B * C, K, -1, output_logits.size(-1)), 
                target_tokens.view(B * C, K, -1), 
                mask.view(B * C, K, -1),
                reduction=reduction)
            # ce_per_token shape (B*C, K, T-1)
            log_shape("ce_per_token(B*C,K,T-1)", ce_per_token)
            
            if token_output_dir:
                avg_loss = save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path)
                print(f"[avg_loss] {avg_loss}")
                return avg_loss if avg_loss is not None else ce_per_token.mean().item()
            else:
                return ce_per_token.mean().item()
        else:
            ce, ce_per_codebook = _compute_cross_entropy(
                logits.view(B * C, K, -1, output_logits.size(-1)), 
                target_tokens.view(B * C, K, -1), 
                mask.view(B * C, K, -1),
                reduction=reduction)
            print(f"[loss(mean)] ce={float(ce)}, codebooks={K}")
            return ce.item()


def process_audio_directory(audio_dir, model, processor, device, output_file=None, reduction="mean", token_output_dir=None):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), audio_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in {audio_dir}")

    print(f"Processing {len(audio_files)} audio files")
    results = []
    error_files = [] 

    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"Results will be saved to: {output_file}")

    for rel_filename in tqdm(audio_files, desc="Processing"):
        audio_path = os.path.join(audio_dir, rel_filename)
        loss = process_single_audio(audio_path, model, processor, device, reduction, token_output_dir, rel_filename)
        if loss is not None:
            results.append((rel_filename, loss))
            if out_file:
                out_file.write(f"{rel_filename}: {loss:.8f}\n")
                out_file.flush()
        else:
            error_files.append(rel_filename)

    if out_file:
        out_file.close()

    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Failed files count: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n{len(error_files)} failed files saved to: {error_file_path}")

    return results, error_files

def main():
    parser = argparse.ArgumentParser(description="Compute loss values for audio files using musicgen-small")
    parser.add_argument("--audio_dir", type=str, default="pitch_out_wav", help="Audio directory")
    parser.add_argument("--audio_path", type=str, default=None, help="Single audio file path (preferred over audio_dir)")
    parser.add_argument("--output_file", type=str, default="results_small.txt", help="Output results file path")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU for all files")
    parser.add_argument("--reduction", type=str, choices=["mean", "none"], default="mean", help="Loss reduction type: mean or none (save per-token losses)")
    parser.add_argument("--token_output_dir", type=str, default=None, help="Directory to save per-token losses when reduction=none")
    parser.add_argument("--hook_layers", type=int, default=2, help="Number of Transformer layers to register hooks (default 2)")
    
    args = parser.parse_args()

    if args.reduction == "none" and not args.token_output_dir:
        args.token_output_dir = "experiments/phase6/tokenizer-transformer"
        print(f"Using default token output directory: {args.token_output_dir}")

    original_stdout = None
    if args.output_file:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    device = setup_device(args.force_cpu)
    print(f"Using device: {device}")

    model, processor = load_model(device)

    if args.audio_path:
        assert os.path.isfile(args.audio_path), f"File not found: {args.audio_path}"
        print(f"Processing single audio: {args.audio_path}")
        loss = process_single_audio(
            args.audio_path, model, processor, device, args.reduction, args.token_output_dir, relative_path=None, hook_layers=args.hook_layers
        )
        if loss is None:
            print(f"\nResult: {os.path.basename(args.audio_path)}: Failed (loss=None)")
        else:
            print(f"\nResult: {os.path.basename(args.audio_path)}: {loss:.4f}")
        if 'device' in locals() and device == 'cuda':
            torch.cuda.empty_cache()
        print("GPU memory cleared")
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        return

    assert os.path.isdir(args.audio_dir), f"Directory not found: {args.audio_dir}"
    results, error_files = process_audio_directory(
        args.audio_dir, model, processor, device, args.output_file, args.reduction, args.token_output_dir
    )

    if not args.output_file:
        print("\nResults:")
        for filename, loss in results:
            print(f"{filename}: {loss:.4f}")

    if 'device' in locals() and device == 'cuda':
        torch.cuda.empty_cache()
    print("GPU memory cleared")

    if original_stdout:
        sys.stdout.close()
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()
