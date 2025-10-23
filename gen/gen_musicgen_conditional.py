import torch
import argparse
import os
from pathlib import Path
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write, audio_read

def generate_conditional_music(input_audio_path, output_path, model_name='facebook/musicgen-small', 
                              context_duration=5.0, generation_duration=10.0, top_k=2, top_p=0.0, 
                              temperature=1.0, use_sampling=True, num_samples=1, create_folder=False):
    """
    Generate conditional music using MusicGen with audio conditioning.
    
    Args:
        input_audio_path: Path to input audio file for conditioning
        output_path: Path for output audio file
        model_name: MusicGen model to use
        context_duration: Duration of context audio in seconds (default: 5.0)
        generation_duration: Duration of generated audio in seconds (default: 10.0)
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        temperature: Sampling temperature
        use_sampling: Whether to use sampling or greedy decoding
        num_samples: Number of samples to generate (default: 1)
        create_folder: Whether to create a folder for organizing samples
    """
    print(f"Loading MusicGen model: {model_name}")
    
    # Load the model
    model = MusicGen.get_pretrained(model_name)
    
    # Set generation parameters - IMPORTANT: duration should be total output duration
    total_duration = context_duration + generation_duration  # 5 + 10 = 15 seconds
    model.set_generation_params(
        duration=total_duration,  # Changed from generation_duration to total_duration
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        use_sampling=use_sampling
    )
    
    # Load and prepare conditioning audio
    print(f"Loading conditioning audio from: {input_audio_path}")
    conditioning_audio, sr = audio_read(input_audio_path)
    
    # Extract first context_duration seconds
    context_samples = int(context_duration * sr)
    if conditioning_audio.shape[-1] < context_samples:
        print(f"Warning: Input audio is shorter than {context_duration}s. Using full audio as context.")
        context_audio = conditioning_audio
    else:
        context_audio = conditioning_audio[..., :context_samples]
    
    print(f"Context audio shape: {context_audio.shape}, duration: {context_audio.shape[-1] / sr:.2f}s")
    
    # Resample if necessary
    if sr != model.sample_rate:
        print(f"Resampling from {sr}Hz to {model.sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(sr, model.sample_rate)
        context_audio = resampler(context_audio)
        # Also resample the original context for concatenation
        context_for_concat = context_audio.clone()
    else:
        context_for_concat = context_audio.clone()
    
    # Ensure correct shape for model (add batch dimension if needed)
    if context_audio.dim() == 2:
        context_audio = context_audio.unsqueeze(0)  # Add batch dimension
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} conditional sample(s)...")
    print(f"Parameters: context={context_duration}s, generation={generation_duration}s, total={total_duration}s, temperature={temperature}")
    
    for i in range(num_samples):
        print(f"\nGenerating sample {i+1}/{num_samples}...")
        
        # Generate conditional music using the context audio
        with torch.no_grad():
            # Use the context audio as conditioning
            generated_audio = model.generate_continuation(
                prompt=context_audio,
                prompt_sample_rate=model.sample_rate,
                descriptions=None  # No text conditioning
            )
        
        # Remove batch dimension and convert to CPU
        generated_audio = generated_audio.squeeze(0).cpu()
        
        # The generated_audio already includes the context + generated content
        full_audio = generated_audio
        
        # Determine output file path
        if num_samples == 1:
            output_file = Path(output_path)
        else:
            output_file = output_dir / f"{Path(output_path).stem}_sample_{i+1:03d}{Path(output_path).suffix}"
        
        print(f"Saving full audio (context + generated) to: {output_file}")
        print(f"Output duration: {full_audio.shape[-1] / model.sample_rate:.2f}s")
        
        # Save the complete audio (context + generated)
        audio_write(str(output_file.with_suffix('')), full_audio, model.sample_rate, strategy="loudness")
    
    print(f"\nGeneration complete! {num_samples} conditional sample(s) saved.")
    if num_samples == 1:
        print(f"Sample saved as: {output_file}.wav")
    else:
        print(f"All samples saved in: {output_dir}")

def batch_conditional_generation(input_audio_path, output_dir, num_batches=10, samples_per_batch=5, **generation_kwargs):
    """
    Generate multiple batches of conditional samples for analysis.
    
    Args:
        input_audio_path: Path to input audio file for conditioning
        output_dir: Directory to save generated audio files
        num_batches: Number of batches to generate
        samples_per_batch: Number of samples per batch
        **generation_kwargs: Additional arguments for generate_conditional_music
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_batches} batches with {samples_per_batch} samples each")
    print(f"Total samples: {num_batches * samples_per_batch}")
    
    for batch in range(num_batches):
        print(f"\n=== BATCH {batch+1}/{num_batches} ===")
        batch_output = output_path / f"batch_{batch+1:02d}_conditional"
        
        generate_conditional_music(
            input_audio_path,
            str(batch_output),
            num_samples=samples_per_batch,
            create_folder=True,
            **generation_kwargs
        )

def main():
    parser = argparse.ArgumentParser(
        description='Generate conditional music with MusicGen using first 5s as context for remaining 10s'
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Input audio file path or directory for conditioning')
    parser.add_argument('--output', '-o', default='conditional_musicgen',
                        help='Output directory path')
    parser.add_argument('--create-folder', action='store_true',
                        help='Create a folder to organize multiple samples')
    
    # Model arguments
    parser.add_argument('--model', default='facebook/musicgen-small',
                        choices=['facebook/musicgen-small', 'facebook/musicgen-medium', 
                                'facebook/musicgen-large'],
                        help='MusicGen model to use (default: facebook/musicgen-small)')
    
    # Generation parameters
    parser.add_argument('--context-duration', type=float, default=5.1,
                        help='Duration of context audio in seconds (default: 5.0)')
    parser.add_argument('--generation-duration', type=float, default=9.9,
                        help='Duration of generated audio in seconds (default: 10.0)')
    parser.add_argument('--top-k', type=int, default=250,
                        help='Top-k sampling parameter (default: 2)')
    parser.add_argument('--top-p', type=float, default=0.0,
                        help='Top-p sampling parameter (default: 0.0)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--no-sampling', action='store_true',
                        help='Use greedy decoding instead of sampling')
    
    # Batch generation
    parser.add_argument('--num-samples', type=int, default=1,  # Changed from default=5
                        help='Number of samples to generate (default: 1)')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Generate multiple batches for analysis')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='Number of batches to generate in batch mode (default: 5)')
    parser.add_argument('--samples-per-batch', type=int, default=3,
                        help='Number of samples per batch (default: 3)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input is a directory or file
    input_path = Path(args.input)
    
    # Prepare generation parameters
    generation_params = {
        'model_name': args.model,
        'context_duration': args.context_duration,
        'generation_duration': args.generation_duration,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'use_sampling': not args.no_sampling
    }
    
    if input_path.is_dir():
        # Process all audio files in directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'**/*{ext}'))  # recursive search
        
        if not audio_files:
            print(f"No audio files found in directory: {input_path}")
            return
        
        print(f"Found {len(audio_files)} audio files to process")
        
        for i, audio_file in enumerate(audio_files):
            print(f"\n=== Processing file {i+1}/{len(audio_files)}: {audio_file.name} ===")
            
            # Save directly to output directory with descriptive filename
            output_filename = f"{audio_file.stem}_conditional"
            
            try:
                if args.batch_mode:
                    batch_conditional_generation(
                        str(audio_file),
                        str(output_dir),
                        args.num_batches,
                        args.samples_per_batch,
                        **generation_params
                    )
                else:
                    generate_conditional_music(
                        str(audio_file),
                        str(output_dir / output_filename),
                        num_samples=args.num_samples,
                        create_folder=False,  # No subfolders
                        **generation_params
                    )
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
                
    elif input_path.is_file():
        # Process single file
        output_filename = f"{input_path.stem}_conditional"
        
        if args.batch_mode:
            batch_conditional_generation(
                str(input_path),
                str(output_dir),
                args.num_batches,
                args.samples_per_batch,
                **generation_params
            )
        else:
            generate_conditional_music(
                str(input_path),
                str(output_dir / output_filename),
                num_samples=args.num_samples,
                create_folder=False,  # No subfolders
                **generation_params
            )
    else:
        print(f"Error: Input path '{args.input}' does not exist.")
        return

if __name__ == "__main__":
    main()