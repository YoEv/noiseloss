"""
NoiseLoss Gradient Experiment - Phase 6
"""

import torch
import torchaudio
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import functional as F

sys.path.append('/home/evev/noiseloss')

class GradientAnalyzer:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.activations = {}
        self.hooks = []
        
    def load_model(self):
        print("Loading MusicGen-Small model...")
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 启用梯度计算但不更新参数
        for p in self.model.parameters():
            p.requires_grad = True
            
        print(f"Model loaded on {self.device}")
        return self.model, self.processor
    
    def register_activation_hooks(self):
        self.activations = {i + 1: [] for i in range(24)}  # Layers 1-24
        
        def create_hook(layer_id):
            def hook(module, input, output):
                # Extract main tensor from tuple if necessary
                activation = output[0] if isinstance(output, tuple) else output
                
                self.activations[layer_id].append(activation)
                if hasattr(activation, 'requires_grad') and activation.requires_grad:
                    activation.retain_grad()
            return hook
        
        # Register hooks for all decoder layers
        layers = self.model.decoder.model.decoder.layers
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(create_hook(i + 1))
            self.hooks.append(hook)
        
        print(f"Registered hooks for {len(layers)} layers")
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def load_audio(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        wav, sr = audio_read(audio_path)
        print(f"Loaded audio: {audio_path}")
        print(f"Shape: {wav.shape}, Sample rate: {sr}")
        return wav, sr
    
    def tokenize_audio(self, wav, sr):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        inputs = self.processor(
            audio=wav.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
                
        return inputs
    
    def compute_per_token_loss(self, inputs):
        # Encode audio to discrete tokens
        with torch.no_grad():
            audio_codes = self.model.audio_encoder.encode(
                inputs["input_values"].to(self.device)
            )
            
            decoder_input_ids = audio_codes[0].transpose(1, 2)  # [B, T, K] -> [B, K, T]
            
            # Check sequence length
            seq_len = decoder_input_ids.shape[3]
            if seq_len <= 1:
                return None
                
            # Prepare input-target pairs
            decoder_inputs = decoder_input_ids[:, :, :, :-1]
            targets = decoder_input_ids[:, :, :, 1:]
        
        # Forward pass (需要梯度计算)
        outputs = self.model.decoder(
            input_ids=decoder_inputs,
            return_dict=True
        )
        
        # Calculate cross-entropy loss per codebook
        logits = outputs.logits
        
        B, K, _, T = targets.shape
        
        ce_per_token = torch.zeros((B, K, T), dtype=torch.float32, device=self.device)
        
        for k in range(K):
            logits_k = logits[k, :, :].contiguous()  # [T, vocab_size]
            targets_k = targets[:, k, 0, :].contiguous().view(-1)  # [T]
            
            ce_flat = F.cross_entropy(logits_k, targets_k, reduction="none")
            ce_per_token[:, k, :] = ce_flat.view(B, T)
        
        # Average over codebooks
        avg_loss_per_token = ce_per_token.mean(dim=1)  # [B, T]
        avg_loss_per_token.requires_grad_(True)
        
        # Clean up intermediate variables
        del audio_codes, decoder_input_ids, decoder_inputs, targets, logits, ce_per_token
        
        return avg_loss_per_token
    
    def compute_activation_gradients(self, loss_per_token):
        """Compute activation gradients for each token and layer"""
        batch_size, seq_len = loss_per_token.shape
        num_layers = 24
        g_h = torch.zeros(num_layers, seq_len)
        
        print("Computing activation gradients...")
        for t in tqdm(range(seq_len)):
            # Clear CUDA cache before each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model.zero_grad(set_to_none=True)
            
            # Use retain_graph=False for the last token, True for others
            retain_graph = (t < seq_len - 1)
            loss_per_token[0, t].backward(retain_graph=retain_graph)
            
            # Collect gradients from each layer
            for layer_id in range(1, num_layers + 1):
                if layer_id not in self.activations or not self.activations[layer_id]:
                    continue
                    
                activation = self.activations[layer_id][-1]
                if activation.grad is not None:
                    grad_norm = activation.grad[0, t, :].norm().item()
                    g_h[layer_id - 1, t] = grad_norm
            
            # Clear gradients after collecting them
            for layer_id in range(1, num_layers + 1):
                if layer_id in self.activations and self.activations[layer_id]:
                    activation = self.activations[layer_id][-1]
                    if activation.grad is not None:
                        activation.grad = None
        
        # Final CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return g_h
    
    def visualize_gradients(self, g_h, save_path=None, skip_tokens=0, colormap='coolwarm'):
        """
        Visualize gradient heatmap
        
        Args:
            g_h: Gradient tensor of shape [num_layers, seq_len]
            save_path: Path to save the heatmap image
            skip_tokens: Number of initial tokens to skip (default: 0)
                        Useful to remove outliers in early tokens
            colormap: Colormap for heatmap (default: 'coolwarm')
                     Options: 'coolwarm' (blue-purple-red), 'plasma' (purple-pink-yellow),
                             'viridis' (purple-blue-green-yellow), 'RdYlBu_r' (red-yellow-blue),
                             'spectral' (red-orange-yellow-green-blue)
        """
        plt.figure(figsize=(12, 8))
        
        # Skip initial tokens if specified
        if skip_tokens > 0 and g_h.shape[1] > skip_tokens:
            g_h_plot = g_h[:, skip_tokens:]
            token_offset = skip_tokens
            print(f"  Skipping first {skip_tokens} tokens for visualization")
        else:
            g_h_plot = g_h
            token_offset = 0
        
        # Use climate-style colormap without white transition
        # Popular options:
        # - 'coolwarm': blue -> purple -> red (smooth transition)
        # - 'plasma': purple -> pink -> yellow (vibrant)
        # - 'viridis': purple -> blue -> green -> yellow (perceptually uniform)
        # - 'RdYlBu_r': red -> yellow -> blue (reversed)
        # - 'spectral': red -> orange -> yellow -> green -> blue
        sns.heatmap(
            g_h_plot.numpy(),
            cmap=colormap,
            cbar_kws={'label': 'Gradient Norm'},
            center=None,  # Don't center the colormap
            robust=True   # Use robust quantiles for better color distribution
        )
        
        title = 'Layer×Time Activation Gradient Heatmap'
        if skip_tokens > 0:
            title += f' (Tokens {skip_tokens}+)'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Layer', fontsize=12)
        
        # Adjust x-axis labels to show actual token positions
        if skip_tokens > 0:
            ax = plt.gca()
            # Get current x-tick positions and labels
            xticks = ax.get_xticks()
            # Adjust labels to show actual token positions
            new_labels = [str(int(tick + token_offset)) if tick + token_offset < g_h.shape[1] else '' 
                         for tick in xticks]
            ax.set_xticklabels(new_labels)
        
        # Improve layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to save memory
            print(f"  Heatmap saved to: {save_path}")
        else:
            plt.show()
        
    def save_results(self, g_h, output_dir, csv_filename=None):
        os.makedirs(output_dir, exist_ok=True)
        
        # Use custom filename if provided, otherwise use default
        if csv_filename is None:
            csv_filename = "gradient_statistics.csv"
        
        # Extract base name without extension for gradient file
        base_name = os.path.splitext(csv_filename)[0]
        gradient_filename = f"{base_name.replace('_gradient_statistics', '')}_activation_gradients.csv"
        
        gradient_file = os.path.join(output_dir, gradient_filename)
        
        df = pd.DataFrame(g_h.numpy())
        df.index.name = 'layer'
        df.columns.name = 'token_position'
        df.to_csv(gradient_file)
        
        print(f"  Gradient data saved to: {gradient_file}")
        
        stats_file = os.path.join(output_dir, csv_filename)
        stats = {
            'layer': range(1, 25),  # 24 layers (1-24)
            'mean_gradient': g_h.mean(dim=1).numpy(),
            'max_gradient': g_h.max(dim=1)[0].numpy(),
            'std_gradient': g_h.std(dim=1).numpy()
        }
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(stats_file, index=False)
        
        print(f"  Statistics saved to: {stats_file}")
        
        return gradient_file, stats_file

def process_single_audio(analyzer, audio_path, output_dir, audio_type="unknown", skip_tokens=0, colormap='coolwarm'):
    """
    Process a single audio file and save results
    
    Args:
        analyzer: GradientAnalyzer instance
        audio_path: Path to audio file
        output_dir: Output directory for results
        audio_type: Type description for logging
        skip_tokens: Number of initial tokens to skip in visualization (default: 0)
        colormap: Colormap for heatmap visualization (default: 'coolwarm')
    """
    print(f"\nProcessing {audio_type}: {os.path.basename(audio_path)}")
    
    try:
        # Process audio
        wav, sr = analyzer.load_audio(audio_path)
        inputs = analyzer.tokenize_audio(wav, sr)
        
        # Compute losses and gradients
        result = analyzer.compute_per_token_loss(inputs)
        if result is None:
            print(f"  Audio too short for analysis: {audio_path}")
            return False
        
        per_token_loss = result
        print(f"  Loss shape: {per_token_loss.shape}, Mean: {per_token_loss.mean().item():.4f}")
        
        g_h = analyzer.compute_activation_gradients(per_token_loss)
        print(f"  Gradients shape: {g_h.shape}, Max: {g_h.max().item():.6f}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results with specific filename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Save CSV (always save full data)
        csv_path = os.path.join(output_dir, f"{base_name}_gradient_statistics.csv")
        analyzer.save_results(g_h, output_dir, csv_filename=f"{base_name}_gradient_statistics.csv")
        
        # Save heatmap (with optional token skipping for better visualization)
        heatmap_path = os.path.join(output_dir, f"{base_name}_gradient_heatmap.png")
        analyzer.visualize_gradients(g_h, save_path=heatmap_path, skip_tokens=skip_tokens, colormap=colormap)
        
        print(f"  Results saved: {csv_path}, {heatmap_path}")
        
        # Clear variables and CUDA cache
        del wav, inputs, per_token_loss, g_h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  CUDA OOM Error for {audio_path}: {e}")
        # Clear CUDA cache and try to recover
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    except Exception as e:
        print(f"  Error processing {audio_path}: {e}")
        # Clear CUDA cache on any error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

def main():
    # Configuration
    base_data_dir = "/home/evev/noiseloss/datasets"
    base_output_dir = "/home/evev/noiseloss/experiments/phase6/results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Visualization configuration
    skip_tokens = 5  # Skip first 5 tokens to avoid outliers in visualization
    colormap = 'coolwarm'  # Climate-style colormap: blue -> purple -> red
    # Other options: 'plasma', 'viridis', 'RdYlBu_r', 'spectral'
    
    # Target file configuration
    target_file = "Beethoven_010_midi_score_short.mid_ori_cut15s.wav"
    subdir = "1_very_slow_under60"
    
    # Define file paths
    original_file = os.path.join(base_data_dir, "ori/asap_ori", subdir, target_file)
    
    # Noise perturbation files with different token counts
    noise_files = [
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_brown_at5_tk5", subdir, target_file),
            "name": "tk5"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_brown_at5_tk10", subdir, target_file),
            "name": "tk10"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_brown_at5_tk50", subdir, target_file),
            "name": "tk50"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_brown_at5_tk100", subdir, target_file),
            "name": "tk100"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_brown_at5_tk150", subdir, target_file),
            "name": "tk150"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_brown_at5_tk200", subdir, target_file),    
            "name": "tk200"
        }
    ]
    
    print(f"Using device: {device}")
    print(f"Target file: {target_file}")
    print(f"Skipping first {skip_tokens} tokens in visualization to avoid outliers")
    print(f"Using colormap: {colormap} (climate-style without white transition)")
    
    # Check if files exist
    all_files = [original_file] + [nf["path"] for nf in noise_files]
    missing_files = [f for f in all_files if not os.path.exists(f)]
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  {f}")
        return
    
    # Initialize analyzer and load model
    analyzer = GradientAnalyzer(device=device)
    analyzer.load_model()
    analyzer.register_activation_hooks()
    
    # Create output directories
    ori_output_dir = os.path.join(base_output_dir, "ori")
    
    # Process original file
    print("="*60)
    print("PROCESSING ORIGINAL FILE")
    print("="*60)
    success_count = 0
    total_count = 0
    
    if process_single_audio(analyzer, original_file, ori_output_dir, "Original", skip_tokens=skip_tokens, colormap=colormap):
        success_count += 1
    total_count += 1
    
    # Process noise perturbation files
    print("\n" + "="*60)
    print("PROCESSING NOISE PERTURBATION FILES")
    print("="*60)
    
    for noise_file in noise_files:
        # Create specific output directory for each noise condition
        noise_output_dir = os.path.join(base_output_dir, "perturb_brown", noise_file["name"])
        if process_single_audio(analyzer, noise_file["path"], noise_output_dir, f"Noise-{noise_file['name']}", skip_tokens=skip_tokens, colormap=colormap):
            success_count += 1
        total_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Successfully processed: {success_count}/{total_count} files")
    print(f"Original results saved in: {ori_output_dir}")
    print(f"Noise perturbation results saved in: {os.path.join(base_output_dir, 'perturb_brown')}")
    print("Subfolders created for each noise condition:")
    for noise_file in noise_files:
        print(f"  - {noise_file['name']}: {os.path.join(base_output_dir, 'perturb_brown', noise_file['name'])}")
    
    # Clean up
    analyzer.clear_hooks()
    del analyzer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

if __name__ == "__main__":
    main()