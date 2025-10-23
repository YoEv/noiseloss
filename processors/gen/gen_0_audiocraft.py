import os
import torch
import torchaudio
import argparse
import sys
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.lm = model.lm.to(device=device, dtype=torch.float32)
    model.compression_model = model.compression_model.to(device=device, dtype=torch.float32)

    compression_ratio = model.compression_model.frame_rate / model.sample_rate
    print(f"Compression ratio: {compression_ratio:.4f} (generating {model.compression_model.frame_rate} tokens per audio second)")

    return model

def decode_from_embedding(input_path, output_path, model, device, use_only_first_codebook=True):
    try:
        # Read input audio
        audio, sr = audio_read(input_path)

        # Convert to mono and move to device
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.to(device)

        # Get encodec embeddings
        with torch.no_grad():
            encoded_frames = model.compression_model.encode(audio.unsqueeze(0))

        # Process encoded frames
        valid_frames = [f for f in encoded_frames if isinstance(f, torch.Tensor)]
        if not valid_frames:
            raise ValueError("No valid tensor frames found")

        # Use only first codebook but format for 4 codebook expectation
        codebook_data = valid_frames[0].to(device=device, dtype=torch.long)
        
        # Handle different tensor shapes properly
        if codebook_data.dim() == 2:  # [B, T]
            # Add codebook dimension and repeat for 4 codebooks
            codebook_data = codebook_data.unsqueeze(1).expand(-1, 4, -1)  # [B, 4, T]
        elif codebook_data.dim() == 3:  # [B, K, T] where K might be 1
            # If we have [B, 1, T], expand to [B, 4, T]
            if codebook_data.shape[1] == 1:
                codebook_data = codebook_data.expand(-1, 4, -1)
            # If we already have [B, 4, T], keep as is
        elif codebook_data.dim() == 4:  # [B, K, T, D]
            # This is the problematic case - reshape to [B, K, T*D] first
            B, K, T, D = codebook_data.shape
            codebook_data = codebook_data.reshape(B, K, T * D)
            # Then ensure we have 4 codebooks
            if K == 1:
                codebook_data = codebook_data.expand(-1, 4, -1)

        # Decode without autocast to avoid float16 issues
        with torch.no_grad():
            decoded_audio = model.compression_model.decode(codebook_data)

        # Ensure the output is float32 for torchaudio.save compatibility
        decoded_audio = decoded_audio.to(dtype=torch.float32)

        # Explicit memory cleanup
        del codebook_data, valid_frames
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Save output
        torchaudio.save(output_path, decoded_audio.cpu()[0], model.sample_rate)
        return True

    except Exception as e:
        if 'codebook_data' in locals(): del codebook_data
        torch.cuda.empty_cache()
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def process_audio_directory(input_dir, output_dir, model, device, use_only_first_codebook=True):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in directory {input_dir}")

    print(f"\nProcessing {len(audio_files)} audio files")
    success_count = 0
    error_files = []

    for filename in tqdm(audio_files, desc="Processing progress"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        success = decode_from_embedding(input_path, output_path, model, device, use_only_first_codebook)
        if success:
            success_count += 1
        else:
            error_files.append(filename)

    print(f"\nSuccessfully processed {success_count} audio files")
    if error_files:
        print(f"Failed to process {len(error_files)} files:")
        for filename in error_files:
            print(f"  - {filename}")

def main():
    parser = argparse.ArgumentParser(description="Decode audio using only the first codebook of MusicGen model")
    parser.add_argument("--input_dir", type=str, default="data_100",
                        help="Input audio files directory")
    parser.add_argument("--output_dir", type=str, default="gen_0_audiocraft",
                        help="Output audio files directory")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    parser.add_argument("--use_all_codebooks", action="store_true",
                        help="Use all codebooks instead of just the first one")

    args = parser.parse_args()

    # Setup device
    device = setup_device(args.force_cpu)
    print(f"Using device: {device}")

    # Load model
    model = load_model(device)

    # Process audio directory
    process_audio_directory(
        args.input_dir, 
        args.output_dir, 
        model, 
        device, 
        use_only_first_codebook=not args.use_all_codebooks
    )

    # Clean up GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        print("GPU memory cleared")

if __name__ == "__main__":
    main()