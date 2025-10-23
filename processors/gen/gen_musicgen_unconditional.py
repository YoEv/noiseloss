import torch
import argparse
import os
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_unconditional_music(output_path, model_name='facebook/musicgen-small', 
                                 duration=15.0, top_k=2, top_p=0.0, temperature=1.0, 
                                 use_sampling=True, num_samples=1, create_folder=False):
    """
    Generate unconditional music using MusicGen model with no input conditioning.
    
    Args:
        output_path: Path to save generated audio
        model_name: MusicGen model to use
        duration: Duration of generated audio in seconds
        top_k: Top-k sampling parameter (2 for very constrained sampling)
        top_p: Top-p sampling parameter
        temperature: Sampling temperature
        use_sampling: Whether to use sampling or greedy decoding
        num_samples: Number of samples to generate
        create_folder: Whether to create a folder for organizing samples
    """
    print(f"Loading MusicGen model: {model_name}")
    
    # Load the model
    model = MusicGen.get_pretrained(model_name)
    
    # Set generation parameters (removed cfg_coeff as it's not supported)
    model.set_generation_params(
        duration=duration,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        use_sampling=use_sampling
    )
    
    # Create output directory if requested
    if create_folder and num_samples > 1:
        output_dir = Path(f"{output_path}_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    else:
        output_dir = Path(output_path).parent
    
    print(f"Generating {num_samples} unconditional samples with top_k={top_k}...")
    print(f"Parameters: duration={duration}s, temperature={temperature}")
    
    for i in range(num_samples):
        print(f"\nGenerating sample {i+1}/{num_samples}...")
        
        # Generate unconditional music (no text, no melody conditioning)
        with torch.no_grad():
            generated_audio = model.generate_unconditional(num_samples=1)
        
        # Determine output file path
        if create_folder and num_samples > 1:
            output_file = output_dir / f"sample_{i+1:03d}"
        else:
            output_file = f"{output_path}_sample_{i+1:03d}"
        
        print(f"Saving to: {output_file}.wav")
        
        # Remove batch dimension and convert to CPU
        generated_audio = generated_audio.squeeze(0).cpu()
        
        # Save using audiocraft's audio_write function
        audio_write(str(output_file), generated_audio, model.sample_rate, strategy="loudness")
    
    print(f"\nGeneration complete! {num_samples} samples saved.")
    if create_folder and num_samples > 1:
        print(f"All samples saved in: {output_dir}")

def batch_unconditional_generation(output_dir, num_batches=10, samples_per_batch=5, **generation_kwargs):
    """
    Generate multiple batches of unconditional samples for analysis.
    
    Args:
        output_dir: Directory to save generated audio files
        num_batches: Number of batches to generate
        samples_per_batch: Number of samples per batch
        **generation_kwargs: Additional arguments for generate_unconditional_music
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_batches} batches with {samples_per_batch} samples each")
    print(f"Total samples: {num_batches * samples_per_batch}")
    
    for batch in range(num_batches):
        print(f"\n=== BATCH {batch+1}/{num_batches} ===")
        batch_output = output_path / f"batch_{batch+1:02d}_unconditional"
        
        generate_unconditional_music(
            str(batch_output),
            num_samples=samples_per_batch,
            create_folder=True,
            **generation_kwargs
        )

def main():
    parser = argparse.ArgumentParser(
        description='Generate unconditional music with MusicGen to explore model sampling preferences'
    )
    
    # Output arguments
    parser.add_argument('--output', '-o', default='unconditional_musicgen',
                        help='Output file prefix or directory')
    parser.add_argument('--create-folder', action='store_true',
                        help='Create a folder to organize multiple samples')
    
    # Model arguments
    parser.add_argument('--model', default='facebook/musicgen-small',
                        choices=['facebook/musicgen-small', 'facebook/musicgen-medium', 
                                'facebook/musicgen-large'],
                        help='MusicGen model to use (default: facebook/musicgen-small)')
    
    # Generation parameters
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Duration of generated audio in seconds (default: 15.0)')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Top-k sampling parameter (default: 2 for constrained sampling)')
    parser.add_argument('--top-p', type=float, default=0.0,
                        help='Top-p sampling parameter (default: 0.0)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--no-sampling', action='store_true',
                        help='Use greedy decoding instead of sampling')
    
    # Batch generation
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples to generate (default: 20)')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Generate multiple batches for analysis')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='Number of batches to generate in batch mode (default: 10)')
    parser.add_argument('--samples-per-batch', type=int, default=5,
                        help='Number of samples per batch (default: 5)')
    
    args = parser.parse_args()
    
    # Prepare generation parameters
    generation_params = {
        'model_name': args.model,
        'duration': args.duration,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'use_sampling': not args.no_sampling
    }
    
    if args.batch_mode:
        # Batch generation for analysis
        batch_unconditional_generation(
            args.output, 
            args.num_batches, 
            args.samples_per_batch, 
            **generation_params
        )
    else:
        # Single generation
        generate_unconditional_music(
            args.output, 
            num_samples=args.num_samples,
            create_folder=args.create_folder,
            **generation_params
        )

if __name__ == "__main__":
    main()