import torch
import torchaudio
import os
import argparse
import sys
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.modules.conditioners import ConditioningAttributes

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    # Load the style model instead of melody model
    model = MusicGen.get_pretrained('facebook/musicgen-style', device=device)
    model.lm = model.lm.to(dtype=torch.float32)
    model.compression_model = model.compression_model.to(dtype=torch.float32)

    # Set generation parameters
    model.set_generation_params(
        duration=15.0,  # generate 10 seconds
        use_sampling=True,
        top_k=250,
        cfg_coef=3.0,  # Classifier Free Guidance coefficient
        cfg_coef_beta=None,  # double CFG is only useful for text-and-style conditioning
    )

    # Set style conditioner parameters
    model.set_style_conditioner_params(
        eval_q=1,  # integer between 1 and 6
                  # eval_q is the level of quantization that passes
                  # through the conditioner. When low, the models adheres less to the
                  # audio conditioning
        excerpt_length=4.0,  # the length in seconds that is taken by the model in the provided excerpt
    )

    compression_ratio = model.compression_model.frame_rate / model.sample_rate
    print(f"Compression ratio: {compression_ratio:.4f} (generating {model.compression_model.frame_rate} tokens per audio second)")

    return model

def generate_audio(input_path, output_path, model, device):
    try:
        # Read input audio
        audio, sr = audio_read(input_path)

        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to(device)

        # Check audio length and pad if necessary
        min_length_samples = int(4.0 * sr)  # Minimum 4.0 seconds (slightly longer than excerpt_length)
        current_length = audio.shape[-1]

        if current_length < min_length_samples:
            print(f"Audio file {os.path.basename(input_path)} is too short ({current_length/sr:.2f}s), padding to {min_length_samples/sr:.2f}s")
            # Calculate how many repeats we need
            repeats = int(min_length_samples // current_length) + 1
            # Create a new tensor with the right size
            padded_audio = torch.zeros((1, min_length_samples), device=device, dtype=audio.dtype)
            # Fill it with repeated copies of the original
            for i in range(repeats):
                start_idx = i * current_length
                if start_idx >= min_length_samples:
                    break
                end_idx = min(start_idx + current_length, min_length_samples)
                copy_length = end_idx - start_idx
                padded_audio[:, start_idx:end_idx] = audio[:, :copy_length]

            audio = padded_audio

        # Generate audio with style set to "piano music"
        with torch.no_grad():
            generated_audio = model.generate_with_chroma(
                descriptions=["piano music"],  # Set style to piano music
                melody_wavs=audio.unsqueeze(0),
                melody_sample_rate=sr,
                progress=True
            )

        # Save generated audio using audio_write for proper normalization
        audio_write(
            output_path.replace('.wav', ''),  # Remove .wav extension as audio_write adds it
            generated_audio.cpu()[0],
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )

        print(f"Successfully generated audio: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def process_audio_directory(input_dir, output_dir, model, device):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in directory {input_dir}")

    print(f"\nProcessing {len(audio_files)} audio files with piano music style")
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

    print(f"\nSuccessfully generated {success_count} audio files")
    if error_files:
        print(f"Failed to process {len(error_files)} files:")
        for filename in error_files:
            print(f"  - {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate piano music using MusicGen-Style model")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input audio files directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output audio files directory")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")

    args = parser.parse_args()

    # Setup device
    device = setup_device(args.force_cpu)
    print(f"Using device: {device}")

    # Load model
    model = load_model(device)

    # Process audio directory
    process_audio_directory(args.input_dir, args.output_dir, model, device)

    # Clean up GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        print("GPU memory cleared")

if __name__ == "__main__":
    main()