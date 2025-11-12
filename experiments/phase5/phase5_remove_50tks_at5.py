import torch
import torchaudio
import os
import argparse
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read, audio_write
import librosa

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

def remove_tokens_at_position(audio_path, output_path, model, processor, device,
                             remove_start_sec=5.0, remove_length_tokens=50):
    """
    Remove a token segment of specified length at a given time and save the result
    
    Args:
        audio_path: input audio path
        output_path: output audio path
        model: MusicGen model
        processor: audio processor
        device: device
        remove_start_sec: start time to remove (seconds)
        remove_length_tokens: number of tokens to remove
    """
    try:
        # Load audio
        if device == "cuda":
            audio, sr = audio_read(audio_path)
            audio = audio.mean(dim=0, keepdim=True)
            if sr != 32000:
                audio = torchaudio.functional.resample(audio, sr, 32000)
            audio = audio.to(device)
        else:
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
        
        # 编码为tokens
        with torch.no_grad():
            encoded = model.audio_encoder.encode(audio.unsqueeze(0))
            
            # Check encoded result
            if encoded is None:
                print(f"Error: audio encoding failed {audio_path}")
                return None
                
            # Get tokens according to MusicGen actual output format
            tokens = None  # initialize tokens variable
            if hasattr(encoded, 'audio_codes'):
                tokens = encoded.audio_codes.long()  # [B, C, K, T]
                print(f"Using audio_codes attribute to get tokens")
            elif hasattr(encoded, 'codes'):
                tokens = encoded.codes.long()  # 可能的替代属性名
                print(f"Using codes attribute to get tokens")
            elif isinstance(encoded, torch.Tensor):
                tokens = encoded.long()  # 直接返回tensor的情况
                print(f"Using tensor as tokens directly")
            else:
                print(f"Error: cannot extract tokens from encoded result {audio_path}")
                print(f"Encoded result type: {type(encoded)}")
                if hasattr(encoded, '__dict__'):
                    print(f"Available attributes: {list(encoded.__dict__.keys())}")
                return None

            # Robustly extract scales (could be Tensor/list/dict/tuple)
            scales = None
            if hasattr(encoded, 'audio_scales'):
                scales = encoded.audio_scales
                print(f"Using audio_scales attribute for scales, type: {type(scales)}")
            elif hasattr(encoded, 'scales'):
                scales = encoded.scales
                print(f"Using scales attribute for scales, type: {type(scales)}")
            elif isinstance(encoded, (list, tuple)) and len(encoded) >= 2:
                maybe_scales = encoded[1]
                scales = maybe_scales
                print("Extracted scales from returned tuple")
            else:
                print("Warning: scales not found in encoded result; will try decoding without scales (may fail)")

            # Move to the same device (supports Tensor/list/tuple/dict)
            def _to_device(obj, device):
                if torch.is_tensor(obj):
                    return obj.to(device)
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_to_device(x, device) for x in obj)
                if isinstance(obj, dict):
                    return {k: _to_device(v, device) for k, v in obj.items()}
                return obj

            tokens = tokens.to(device)
            scales = _to_device(scales, device)

        # Check tokens
        if tokens is None:
            print(f"Error: tokens is None {audio_path}")
            return None
            
        print(f"Successfully got tokens, shape: {tokens.shape}")
        B, C, K, T = tokens.shape
        
        # Compute removal position (token index)
        # MusicGen compression ratio is ~320:1 (32000Hz -> 100Hz)
        tokens_per_sec = 32000 // 320  # ~100 tokens/sec
        remove_start_token = int(remove_start_sec * tokens_per_sec)
        remove_end_token = remove_start_token + remove_length_tokens
        
        print(f"Original audio length: {T} tokens ({T/tokens_per_sec:.2f}s)")
        print(f"Removal: token {remove_start_token} to {remove_end_token} (start at {remove_start_sec}s, remove {remove_length_tokens} tokens)")
        
        # Bounds check
        if remove_end_token > T:
            print(f"Warning: removal end exceeds audio length; clipping to end")
            remove_end_token = T
            remove_length_tokens = remove_end_token - remove_start_token
        
        if remove_start_token >= T:
            print(f"Error: removal start exceeds audio length")
            return None
        
        # Create removed tokens
        # Before + after (skip middle)
        tokens_before = tokens[:, :, :, :remove_start_token]
        tokens_after = tokens[:, :, :, remove_end_token:]
        
        # Concatenate removed tokens
        tokens_removed = torch.cat([tokens_before, tokens_after], dim=-1)

        # Also slice scales consistently (supports Tensor/list/tuple/dict)
        def _remove_slice(obj):
            if torch.is_tensor(obj):
                return torch.cat([obj[..., :remove_start_token], obj[..., remove_end_token:]], dim=-1)
            if isinstance(obj, (list, tuple)):
                return type(obj)(_remove_slice(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _remove_slice(v) for k, v in obj.items()}
            return None

        scales_removed = _remove_slice(scales) if scales is not None else None
        if scales is None:
            print("Warning: missing scales; trying to decode without scales (may fail)")
        
        print(f"Audio length after removal: {tokens_removed.shape[-1]} tokens ({tokens_removed.shape[-1]/tokens_per_sec:.2f}s)")
        
        # 解码为音频
        with torch.no_grad():
            # Prefer call with scales; if incompatible, fallback
            try:
                if scales_removed is not None:
                    decoded_out = model.audio_encoder.decode(tokens_removed, scales_removed)
                else:
                    decoded_out = model.audio_encoder.decode(tokens_removed)
            except TypeError:
                print("Decoder requires scales; retry with placeholder scales=1")
                dummy_scales = torch.ones(
                    (tokens_removed.shape[0], tokens_removed.shape[1], 1, tokens_removed.shape[-1]),
                    dtype=torch.float32, device=tokens_removed.device
                )
                decoded_out = model.audio_encoder.decode(tokens_removed, dummy_scales)

            # Extract audio Tensor from various return types (supports EncodecDecoderOutput.audio_values)
            print(f"Decode return type: {type(decoded_out)}")
            wave = None
            if isinstance(decoded_out, torch.Tensor):
                wave = decoded_out
            elif hasattr(decoded_out, 'audio_values'):
                wave = decoded_out.audio_values
                print("Extracted audio from EncodecDecoderOutput.audio_values")
            elif hasattr(decoded_out, 'audio'):
                wave = decoded_out.audio
            elif hasattr(decoded_out, 'wav'):
                wave = decoded_out.wav
            elif isinstance(decoded_out, (list, tuple)) and len(decoded_out) > 0:
                first = decoded_out[0]
                if hasattr(first, 'audio_values'):
                    wave = first.audio_values
                else:
                    wave = first
            elif isinstance(decoded_out, dict):
                if 'audio_values' in decoded_out:
                    wave = decoded_out['audio_values']
                elif 'audio' in decoded_out:
                    wave = decoded_out['audio']

            if isinstance(wave, (list, tuple)) and len(wave) > 0:
                wave = wave[0]

            if wave is None or not torch.is_tensor(wave):
                attrs = dir(decoded_out) if hasattr(decoded_out, '__class__') else 'N/A'
                print(f"Error: unable to extract audio from decode result, type: {type(decoded_out)}, attrs: {attrs}")
                return None

            print(f"Decode success, audio tensor shape: {wave.shape}")
        
        # Save audio
        decoded_audio = wave.squeeze(0).cpu()
        try:
            decoded_audio = decoded_audio.squeeze(0).cpu()
        except AttributeError as e:
            print(f"Error: post-decode audio handling failed {audio_path}: {e}")
            return None
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as wav
        audio_write(output_path.replace('.wav', ''), decoded_audio, 32000, format='wav')
        
        print(f"Removed audio saved to: {output_path}")
        
        # 返回删除信息
        return {
            'original_length_tokens': T,
            'removed_start_token': remove_start_token,
            'removed_end_token': remove_end_token,
            'removed_length_tokens': remove_length_tokens,
            'final_length_tokens': tokens_removed.shape[-1],
            'removed_start_sec': remove_start_sec,
            'removed_length_sec': remove_length_tokens / tokens_per_sec
        }
        
    except Exception as e:
        print(f"Error processing audio {audio_path}: {str(e)}")
        return None

def process_directory(input_dir, output_dir, model, processor, device,
                     remove_start_sec=5.0, remove_length_tokens=50):
    """
    Batch process audio files in a directory
    """
    # Find all audio filesaudio_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), input_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    results = []
    for rel_path in audio_files:
        input_path = os.path.join(input_dir, rel_path)
        output_path = os.path.join(output_dir, rel_path)
        
        print(f"\nProcess: {rel_path}")
        result = remove_tokens_at_position(
            input_path, output_path, model, processor, device,
            remove_start_sec, remove_length_tokens
        )
        
        if result:
            result['file_path'] = rel_path
            results.append(result)
    
    # Save results
    results_file = os.path.join(output_dir, 'removal_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Removal params: start={remove_start_sec}s, length={remove_length_tokens}tokens\n\n")
        for result in results:
            f.write(f"File: {result['file_path']}\n")
            f.write(f"  Original length: {result['original_length_tokens']} tokens\n")
            f.write(f"  Removal range: {result['removed_start_token']}-{result['removed_end_token']} tokens\n")
            f.write(f"  Removal time: start {result['removed_start_sec']:.2f}s, length {result['removed_length_sec']:.2f}s\n")
            f.write(f"  Final length: {result['final_length_tokens']} tokens\n\n")
    
    print(f"\nResults saved to: {results_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Remove a token segment at a specified position")
    parser.add_argument("--input_dir", type=str, default="MusicEval_Small_ori", 
                       help="Input audio directory")
    parser.add_argument("--output_dir", type=str, default="MusicEval_Small_removed_50tks", 
                       help="Output audio directory")
    parser.add_argument("--remove_start_sec", type=float, default=5.0, 
                       help="Start time to remove (seconds)")
    parser.add_argument("--remove_length_tokens", type=int, default=50, 
                       help="Number of tokens to remove")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU")
    
    args = parser.parse_args()
    
    # Device
    device = setup_device(args.force_cpu)
    print(f"Device: {device}")
    
    # Load model
    print("Loading MusicGen model...")
    model, processor = load_model(device)
    
    # Process
    if os.path.isfile(args.input_dir):
        # Single file
        output_path = args.output_dir if args.output_dir.endswith('.wav') else args.output_dir + '.wav'
        result = remove_tokens_at_position(
            args.input_dir, output_path, model, processor, device,
            args.remove_start_sec, args.remove_length_tokens
        )
        if result:
            print(f"\nDone: {result}")
    else:
        # Directory
        results = process_directory(
            args.input_dir, args.output_dir, model, processor, device,
            args.remove_start_sec, args.remove_length_tokens
        )
        print(f"\nBatch processing complete: processed {len(results)} files")

if __name__ == "__main__":
    main()