import torch
import torchaudio
import os
import argparse
import sys
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes


def setup_device(force_cpu=False):
    """Set up device (CPU or GPU)."""
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device):
    """Load the pretrained MusicGen-melody model."""
    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
    model.lm = model.lm.to(dtype=torch.float32)
    model.compression_model = model.compression_model.to(dtype=torch.float32)

    compression_ratio = model.compression_model.frame_rate / model.sample_rate
    print(f"Compression ratio: {compression_ratio:.4f} "
          f"({model.compression_model.frame_rate} tokens generated per second of audio)")

    return model


def generate_audio(input_path, output_path, model, device):
    """Generate an output audio file using MusicGen based on input audio."""
    # Read input audio
    audio, sr = audio_read(input_path)

    # Convert to mono if necessary
    if audio.dim() > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    audio = audio.to(device)
    original_length = audio.shape[-1]

    # Build conditioning attributes
    condition = ConditioningAttributes(
        text={'description': ""},
        wav={'self_wav': (audio.unsqueeze(0),
                          torch.tensor([original_length], device=device),
                          [sr], [], [])}
    )

    # Generate audio
    with torch.no_grad():
        generated_audio = model.generate_with_chroma(
            descriptions=[""],
            melody_wavs=audio.unsqueeze(0),
            melody_sample_rate=sr,
            progress=True
        )

    # Save generated audio
    torchaudio.save(
        output_path,
        generated_audio.cpu()[0],
        model.sample_rate
    )

    print(f"Successfully generated: {output_path}")
    return True

def process_audio_directory(input_dir, output_dir, model, device):
    """Process all audio files in a directory using MusicGen."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all audio files (WAV/MP3)
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in directory {input_dir}")

    print(f"\nStarting to process {len(audio_files)} audio files...")
    success_count = 0
    error_files = []

    for filename in tqdm(audio_files, desc="Generation progress"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        success = generate_audio(input_path, output_path, model, device)
        if success:
            success_count += 1
        else:
            error_files.append(filename)

    print(f"\nSuccessfully generated {success_count} audio files.")
    if error_files:
        print(f"{len(error_files)} files failed to process:")
        for filename in error_files:
            print(f"  - {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate audio files using the MusicGen-melody model.")
    parser.add_argument("--input_dir", type=str, default="data_100",
                        help="Directory containing input audio files.")
    parser.add_argument("--output_dir", type=str, default="gen_100",
                        help="Directory to save generated audio files.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU usage even if GPU is available.")

    args = parser.parse_args()

    # Set up device
    device = setup_device(args.force_cpu)
    print(f"Using device: {device}")

    # Load model
    model = load_model(device)

    # Process directory
    process_audio_directory(args.input_dir, args.output_dir, model, device)

    # Clean GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        print("Cleared GPU memory.")


if __name__ == "__main__":
    main()
