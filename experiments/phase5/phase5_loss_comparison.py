import torch
import torchaudio
import os
import argparse
import json
import pandas as pd
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import CrossEntropyLoss
import librosa
import numpy as np

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

def nll_segment_teacher_forcing(model, tokens, start, L, codebooks='auto', mask_invalid=None):
    """
    Compute per-token NLL for a specified segment
    
    Args:
        model: MusicGen model
        tokens: LongTensor [B, T, C] or [B, C, T] depending on implementation (C=codebooks)
        start: start position
        L: segment length
        codebooks: codebook configuration
        mask_invalid: mask for invalid positions
    
    Returns:
        per-token NLL (mean over codebooks, masked), scalar
    """
    # Assume tokens shape is [B, C, K, T] (per MusicGen format)
    if len(tokens.shape) == 4:
        B, C, K, T = tokens.shape
        # Rearrange to [B, T, C*K] for computation
        tokens_reshaped = tokens.permute(0, 3, 1, 2).contiguous().view(B, T, C*K)
    else:
        B, T, C = tokens.shape
        K = 1
    
    ce = CrossEntropyLoss(reduction='none')
    losses = []
    
    # Compute CE per codebook and average
    for c in range(C):
        for k in range(K):
            # Take sequence for codebook c, layer k: [B, T]
            if len(tokens.shape) == 4:
                seq_ck = tokens[:, c, k, :]  # [B, T]
            else:
                seq_ck = tokens[:, :, c]  # [B, T]
            
            # Teacher forcing: target at t is seq_ck[:, t]
            # Build input and target
            input_seq = seq_ck[:, :-1]  # [B, T-1]
            target_seq = seq_ck[:, 1:]  # [B, T-1]
            
            logits_ck = None  # Initialize to avoid undefined reference
            # Model forward
            with torch.no_grad():
                # Adjust to actual model API as needed
                # Assume model takes token input and returns logits
                if len(tokens.shape) == 4:
                    # Use full multi-codebook input
                    input_tokens = tokens[:, :, :, :-1]  # [B, C, K, T-1]
                    input_reshaped = input_tokens.view(B, C*K, -1)  # [B, C*K, T-1]
                    outputs = model.decoder(input_reshaped)
                    logits = getattr(outputs, "logits", outputs)  # expect 3D

                    # Normalize to [B, time, vocab]
                    if logits.dim() != 3:
                        raise ValueError(f"Unsupported logits dim {logits.dim()} for tokens shape {tuple(tokens.shape)}")
                    ck_total = C * K
                    expected_t = T - 1
                    expected_full = expected_t * ck_total
                    b, d1, d2 = logits.shape  # candidates [B, time, vocab] or [B, vocab, time]

                    def is_time(n: int) -> bool:
                        return (n == expected_t) or (n == expected_full) or (n % ck_total == 0 and (n // ck_total) == expected_t)

                    if is_time(d1):
                        pass  # already [B, time, vocab]
                    elif is_time(d2):
                        logits = logits.transpose(1, 2)  # [B, time, vocab]
                        b, d1, d2 = logits.shape
                    else:
                        raise ValueError(f"Cannot identify time dimension from logits shape {tuple(logits.shape)}; expected_t={expected_t}, ck_total={ck_total}")

                    seq_len = d1
                    vocab_sz = d2

                    # Align logits to current (c,k) codebook time series
                    if seq_len == expected_t:
                        logits_ck = logits  # [B, T-1, vocab]
                    elif seq_len == expected_full:
                        offset = c * K + k
                        time_index = torch.arange(expected_t, device=logits.device) * ck_total + offset  # [T-1]
                        logits_ck = logits.index_select(dim=1, index=time_index)  # [B, T-1, vocab]
                    else:
                        if seq_len % ck_total == 0 and (seq_len // ck_total) == expected_t:
                            logits_reshaped = logits.view(B, expected_t, ck_total, vocab_sz)  # [B, T-1, C*K, vocab]
                            logits_ck = logits_reshaped[:, :, c * K + k, :]  # [B, T-1, vocab]
                        else:
                            raise ValueError(f"Unsupported logits shape {tuple(logits.shape)} for tokens shape {tuple(tokens.shape)}")
                else:
                    # Simplified: normalize to [B, T-1, vocab]
                    outputs = model.decoder(input_seq.unsqueeze(1))
                    logits_raw = getattr(outputs, "logits", outputs)  # Possibly [B, T-1, V] or [B, V, T-1] or [B, T-1]
                    if logits_raw.dim() == 3:
                        if logits_raw.size(1) != target_seq.size(1) and logits_raw.size(2) == target_seq.size(1):
                            logits_ck = logits_raw.transpose(1, 2)  # [B, T-1, V]
                        else:
                            logits_ck = logits_raw  # [B, T-1, V]
                    elif logits_raw.dim() == 2:
                        logits_ck = logits_raw.unsqueeze(1)  # [B, 1, V]
                    else:
                        raise ValueError(f"Unexpected logits_raw shape {tuple(logits_raw.shape)}")

            # Evaluate NLL only within window [start, start+L); skip if logits_ck invalid or window invalid
            if logits_ck is None:
                continue

            sl = start
            sr = min(start + L, target_seq.size(1))

            # Final alignment: if logits time dim is integer multiple of target (typically C*K), select by (c,k) offset
            if logits_ck.size(1) != target_seq.size(1):
                ck_total = C * K
                if logits_ck.size(1) % target_seq.size(1) == 0 and (logits_ck.size(1) // target_seq.size(1)) == ck_total:
                    expected_t = target_seq.size(1)
                    offset = c * K + k
                    # On global time axis, extract (c,k) subsequence with stride=ck_total
                    time_index = torch.arange(expected_t, device=logits_ck.device) * ck_total + offset  # [T-1]
                    logits_ck = logits_ck.index_select(dim=1, index=time_index)  # [B, T-1, vocab]
                else:
                    raise ValueError(f"Final align failed: logits_ck.time={logits_ck.size(1)} vs target.time={target_seq.size(1)}, C*K={ck_total}")

            # Clip sr to logits_ck time length to avoid out-of-bounds
            max_time = logits_ck.size(1)
            sr = min(sr, max_time)

            if sr <= sl:
                continue

            logits_win = logits_ck[:, sl:sr, :]  # [B, L?, vocab_size]
            target_win = target_seq[:, sl:sr]    # [B, L]

            # Window-level time alignment to ensure logits_win and target_win have equal length
            if logits_win.size(1) != target_win.size(1):
                lw = logits_win.size(1)
                tw = target_win.size(1)
                ck_total = C * K
                if tw > 0 and lw % tw == 0:
                    ratio = lw // tw
                    offset = (c * K + k) % max(ratio, 1)  # current (c,k) offset
                    idx = torch.arange(tw, device=logits_win.device) * ratio + offset
                    idx = idx.clamp_max(lw - 1)
                    logits_win = logits_win.index_select(dim=1, index=idx)  # [B, tw, vocab]
                    # keep target_win as [B, tw]
                else:
                    # Fallback: clip both to the same minimum length
                    min_len = min(lw, tw)
                    logits_win = logits_win[:, :min_len, :]
                    target_win = target_win[:, :min_len]

            V = logits_win.size(-1)
            Lw = logits_win.size(1)

            # Compute cross entropy (using actual window length Lw)
            loss_tok = ce(logits_win.reshape(-1, V), target_win.reshape(-1))
            loss_tok = loss_tok.view(B, Lw)  # [B, Lw]
            
            if mask_invalid is not None:
                # mask_invalid: [B, T, C] (True means valid) or [B, C, K, T]
                if len(mask_invalid.shape) == 4:
                    m = mask_invalid[:, c, k, 1:][:, sl:sl+Lw]  # align to target
                else:
                    m = mask_invalid[:, 1:, c][:, sl:sl+Lw]     # align to target
                loss_tok = loss_tok.masked_fill(~m, 0.0)
                denom = m.float().sum().clamp_min(1.0)
                loss_c = loss_tok.sum() / denom
            else:
                loss_c = loss_tok.mean()
            
            losses.append(loss_c)
    
    # Average over codebooks (or custom weighting)
    if losses:
        loss = torch.stack(losses).mean()
        return loss.item()
    else:
        return 0.0

@torch.no_grad()
def generate_hole(model, prefix_tokens, L, temperature=0.7, codebooks='auto'):
    """
    Generate tokens for a missing segment
    
    Args:
        model: MusicGen model
        prefix_tokens: [B, t0, C] prefix tokens
        L: generation length
        temperature: generation temperature
        codebooks: codebook configuration
    
    Returns:
        gen_tokens: [B, L, C] generated tokens
    """
    if len(prefix_tokens.shape) == 4:
        B, C, K, t0 = prefix_tokens.shape
        # Rearrange to a format suitable for generation
        cur = prefix_tokens.clone()
    else:
        B, t0, C = prefix_tokens.shape
        K = 1
        cur = prefix_tokens.clone()
    
    gen = []
    
    for step in range(L):
        # Forward pass to get next-step distribution
        if len(cur.shape) == 4:
            input_reshaped = cur.view(B, C*K, -1)
            outputs = model.decoder(input_reshaped)
            logits = outputs.logits  # [B, T, vocab_size]
            
            # Take logits at the last step
            next_logits = logits[:, -1, :]  # [B, vocab_size]
        else:
            outputs = model.decoder(cur)
            next_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        
        # Sample next token
        if temperature == 0.0:
            next_tokens = torch.argmax(next_logits, dim=-1)  # [B]
        else:
            probs = torch.softmax(next_logits / max(1e-6, temperature), dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        
        # 添加到序列
        if len(cur.shape) == 4:
            # Need to expand to multi-codebook format
            next_tokens_expanded = next_tokens.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]
            next_tokens_expanded = next_tokens_expanded.expand(B, C, K, 1)
            cur = torch.cat([cur, next_tokens_expanded], dim=-1)
            gen.append(next_tokens_expanded.squeeze(-1))  # [B, C, K]
        else:
            next_tokens_expanded = next_tokens.unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
            next_tokens_expanded = next_tokens_expanded.expand(B, 1, C)
            cur = torch.cat([cur, next_tokens_expanded], dim=1)
            gen.append(next_tokens_expanded.squeeze(1))  # [B, C]
    
    if len(prefix_tokens.shape) == 4:
        gen_tokens = torch.stack(gen, dim=-1)  # [B, C, K, L]
    else:
        gen_tokens = torch.stack(gen, dim=1)   # [B, L, C]
    
    return gen_tokens

def eval_gap_fill(model, processor, ori_audio_path, filled_audio_path, device,
                 t0_sec=5.0, L_tokens=50, temperature=0.0, mask_invalid=None):
    """
    Compute Δ = gen_loss_with_blank - ori_loss
    
    Args:
        model: MusicGen model
        processor: audio processor
        ori_audio_path: path to original audio
        filled_audio_path: path to filled audio
        device: device
        t0_sec: start position (seconds)
        L_tokens: segment length (tokens)
        temperature: generation temperature
        mask_invalid: invalid position mask
    
    Returns:
        dict containing ori_nll, gen_nll, delta
    """
    try:
        # Load and encode original audio
        if device == "cuda":
            ori_audio, sr = audio_read(ori_audio_path)
            ori_audio = ori_audio.mean(dim=0, keepdim=True)
            if sr != 32000:
                ori_audio = torchaudio.functional.resample(ori_audio, sr, 32000)
            ori_audio = ori_audio.to(device)
        else:
            ori_audio, sr = librosa.load(ori_audio_path, sr=32000, mono=True)
            ori_audio = torch.tensor(ori_audio).unsqueeze(0).to(device)
        
        # Encode to tokens
        with torch.no_grad():
            ori_encoded = model.audio_encoder.encode(ori_audio.unsqueeze(0))
            ori_tokens = ori_encoded.audio_codes.long()  # [B, C, K, T]
        
        B, C, K, T = ori_tokens.shape
        
        # Calculate token position
        tokens_per_sec = 32000 // 320  # 约100 tokens/sec
        t0 = int(t0_sec * tokens_per_sec)
        
        assert t0 + L_tokens <= T, f"Segment exceeds audio length: {t0 + L_tokens} > {T}"
        
        # 1) NLL on original segment (teacher forcing)
        nll_ori = nll_segment_teacher_forcing(model, ori_tokens, t0, L_tokens, mask_invalid=mask_invalid)
        
        # 2) Load filled audio
        if device == "cuda":
            filled_audio, sr2 = audio_read(filled_audio_path)
            filled_audio = filled_audio.mean(dim=0, keepdim=True)
            if sr2 != 32000:
                filled_audio = torchaudio.functional.resample(filled_audio, sr2, 32000)
            filled_audio = filled_audio.to(device)
        else:
            filled_audio, sr2 = librosa.load(filled_audio_path, sr=32000, mono=True)
            filled_audio = torch.tensor(filled_audio).unsqueeze(0).to(device)
        
        # Encode filled audio
        with torch.no_grad():
            filled_encoded = model.audio_encoder.encode(filled_audio.unsqueeze(0))
            filled_tokens = filled_encoded.audio_codes.long()  # [B, C, K, T']
        
        # 3) Teacher-forcing NLL on generated segment within filled audio
        nll_gen_ctx = nll_segment_teacher_forcing(model, filled_tokens, t0, L_tokens, mask_invalid=mask_invalid)
        
        delta = nll_gen_ctx - nll_ori
        
        return {
            "ori_nll": float(nll_ori),
            "gen_nll": float(nll_gen_ctx),
            "delta": float(delta),
            "ori_audio_path": ori_audio_path,
            "filled_audio_path": filled_audio_path,
            "t0_sec": t0_sec,
            "L_tokens": L_tokens
        }
        
    except Exception as e:
        print(f"Error during gap filling evaluation: {str(e)}")
        return None

def process_directory_comparison(ori_dir, filled_dir, output_file, model, processor, device,
                               t0_sec=5.0, L_tokens=50, temperature=0.0):
    """
    Batch-compare loss of original vs filled audio
    """
    # Find all audio files
    audio_files = []
    for root, dirs, files in os.walk(ori_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), ori_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        print(f"No audio files found in {ori_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    results = []
    for rel_path in audio_files:
        ori_path = os.path.join(ori_dir, rel_path)
        filled_path = os.path.join(filled_dir, rel_path)
        
        if not os.path.exists(filled_path):
            print(f"Warning: filled audio not found {filled_path}")
            continue
        
        print(f"\nCompare: {rel_path}")
        
        result = eval_gap_fill(
            model, processor, ori_path, filled_path, device,
            t0_sec, L_tokens, temperature
        )
        
        if result:
            result['file_path'] = rel_path
            results.append(result)
            print(f"  Original NLL: {result['ori_nll']:.6f}")
            print(f"  Generated NLL: {result['gen_nll']:.6f}")
            print(f"  Delta Δ: {result['delta']:.6f}")
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为JSON
    with open(output_file.replace('.txt', '.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as text
    with open(output_file, 'w') as f:
        f.write(f"Gap Filling Loss Comparison Results\n")
        f.write(f"Params: t0={t0_sec}s, L={L_tokens}tokens, temperature={temperature}\n\n")
        
        for result in results:
            f.write(f"File: {result['file_path']}\n")
            f.write(f"  Original NLL: {result['ori_nll']:.8f}\n")
            f.write(f"  Generated NLL: {result['gen_nll']:.8f}\n")
            f.write(f"  Delta: {result['delta']:.8f}\n\n")
        
        # Summary stats
        if results:
            deltas = [r['delta'] for r in results]
            f.write(f"\nSummary:\n")
            f.write(f"  Total files: {len(results)}\n")
            f.write(f"  Mean Δ: {np.mean(deltas):.8f}\n")
            f.write(f"  Std Dev: {np.std(deltas):.8f}\n")
            f.write(f"  Min Δ: {np.min(deltas):.8f}\n")
            f.write(f"  Max Δ: {np.max(deltas):.8f}\n")
            f.write(f"  Count Δ<0: {sum(1 for d in deltas if d < 0)}\n")
            f.write(f"  Count Δ>0: {sum(1 for d in deltas if d > 0)}\n")
    
    print(f"\nComparison saved to: {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare loss between original audio and gap-filled audio")
    parser.add_argument("--ori_dir", type=str, default="MusicEval_Small_ori",
                       help="Directory of original audio")
    parser.add_argument("--filled_dir", type=str, default="MusicEval_Small_filled",
                       help="Directory of filled audio")
    parser.add_argument("--output_file", type=str, default="gap_fill_loss_comparison.txt",
                       help="Output results file")
    parser.add_argument("--t0_sec", type=float, default=5.0,
                       help="Start position for evaluation (seconds)")
    parser.add_argument("--L_tokens", type=int, default=50,
                       help="Segment length (tokens)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature (for generation mode, typically 0 for deterministic evaluation)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU")
    
    args = parser.parse_args()
    
    # Device
    device = setup_device(args.force_cpu)
    print(f"Device: {device}")
    
    # Load model
    print("Loading MusicGen model...")
    model, processor = load_model(device)
    
    # Compare
    results = process_directory_comparison(
        args.ori_dir, args.filled_dir, args.output_file,
        model, processor, device,
        args.t0_sec, args.L_tokens, args.temperature
    )
    
    print(f"\nBatch comparison complete: processed {len(results)} files")
    
    if results:
        deltas = [r['delta'] for r in results]
        print(f"\nSummary:")
        print(f"  Mean Δ: {np.mean(deltas):.6f}")
        print(f"  Std Dev: {np.std(deltas):.6f}")
        print(f"  Count Δ<0: {sum(1 for d in deltas if d < 0)} ({100*sum(1 for d in deltas if d < 0)/len(deltas):.1f}%)")
        print(f"  Count Δ>0: {sum(1 for d in deltas if d > 0)} ({100*sum(1 for d in deltas if d > 0)/len(deltas):.1f}%)")

if __name__ == "__main__":
    main()