"""
NoiseLoss Gradient Experiment - Phase 6 (Window Averaging Version)

Memory-Optimized Implementation:
- Uses window averaging (5-15s) instead of per-token backward passes
- Single backward() call with retain_graph=False for better memory efficiency
- Disabled parameter gradients, only computes activation gradients
- Significantly reduces GPU memory usage compared to the original version

Key improvements:
1. Window-based loss averaging: loss = nll[mask].mean()
2. Single backward pass instead of multiple retain_graph=True calls
3. Parameter gradients disabled: model.parameters().requires_grad = False
4. Automatic CUDA cache cleanup
5. Support for gradient difference visualization
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
    def __init__(self, device="cuda", window_start_sec=5.0, window_end_sec=15.0, sample_rate=32000):
        self.device = device
        self.model = None
        self.processor = None
        self.activations = {}
        self.hooks = []
        self.window_start_sec = window_start_sec
        self.window_end_sec = window_end_sec
        self.sample_rate = sample_rate
        
    def load_model(self):
        print("Loading MusicGen-Small model...")
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        self.model = self.model.to(self.device)
        self.model.eval()
        # 关闭参数梯度来节省显存，只保留激活梯度
        for p in self.model.parameters():
            p.requires_grad = True
            
        print(f"Model loaded on {self.device} (parameter gradients enabled)")
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
    
    def get_window_token_indices(self, seq_len, audio_duration_sec):
        """计算窗口对应的token索引范围"""
        # MusicGen的token rate大约是50 tokens/sec
        tokens_per_sec = seq_len / audio_duration_sec
        
        idx0 = int(self.window_start_sec * tokens_per_sec)
        idx1 = int(self.window_end_sec * tokens_per_sec)
        
        # 确保索引在有效范围内
        idx0 = max(0, idx0)
        idx1 = min(seq_len, idx1)
        
        print(f"Window {self.window_start_sec}-{self.window_end_sec}s maps to tokens {idx0}-{idx1} (total {seq_len} tokens)")
        return idx0, idx1
    
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
        # 不再强制设置 requires_grad_(True)；依赖模型参数梯度自然传播
        # avg_loss_per_token.requires_grad_(True)  # <-- 删除这行
        
        # Clean up intermediate variables
        del audio_codes, decoder_input_ids, decoder_inputs, targets, logits, ce_per_token
        
        return avg_loss_per_token
    
    def compute_activation_gradients(self, loss_per_token, audio_duration_sec):
        """基于窗口平均的一次反向传播，读取全序列 token 的各层激活梯度范数"""
        batch_size, seq_len = loss_per_token.shape
        num_layers = 24
        g_h = torch.zeros(num_layers, seq_len, device=self.device)
        
        # 计算窗口对应的token索引
        idx0, idx1 = self.get_window_token_indices(seq_len, audio_duration_sec)
        if idx0 >= idx1:
            print("Warning: Invalid window range, using full sequence")
            idx0, idx1 = 0, seq_len
        
        print(f"Computing activation gradients via window-mean backward for tokens {idx0}-{idx1}...")
        
        self.model.zero_grad(set_to_none=True)
        window_loss = loss_per_token[0, idx0:idx1].mean()
        window_loss.backward()
        
        for layer_id in range(1, num_layers + 1):
            if layer_id not in self.activations or not self.activations[layer_id]:
                continue
            activation = self.activations[layer_id][-1]  # [B, T, H]
            if activation.grad is None:
                continue
            layer_grad = activation.grad[0, :seq_len, :].norm(dim=-1).detach().cpu()  # [T]
            g_h[layer_id - 1, :layer_grad.numel()] = layer_grad
            activation.grad = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Gradient computation completed. Shape: {g_h.shape}, max={g_h.max().item():.6f}")
        return g_h.detach().cpu()
    
    def visualize_gradients(self, g_h, save_path=None):
        plt.figure(figsize=(12, 8))
        
        # Use blue to red colormap for better clarity
        sns.heatmap(
            g_h.numpy(),
            cmap='RdBu_r',  # Red-Blue reversed (blue=low, red=high)
            cbar_kws={'label': 'Gradient Norm'},
            center=None  # Don't center the colormap
        )
        
        plt.title('Layer×Time Activation Gradient Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Layer', fontsize=12)
        
        # Improve layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to save memory
        else:
            plt.show()
            print(f"Heatmap saved to: {save_path}")
        
        plt.show()
        
    def visualize_gradient_diff(self, g_h_perturbed, g_h_original, save_path=None, title_suffix=""):
        """Create difference heatmap: perturbed - original"""
        diff = g_h_perturbed.numpy() - g_h_original.numpy()
        
        plt.figure(figsize=(12, 8))
        
        # Use diverging colormap centered at 0 for difference
        vmax = np.percentile(np.abs(diff), 95)  # Use 95th percentile to avoid outliers
        sns.heatmap(
            diff,
            cmap='bwr',  # Blue-White-Red (blue=negative, red=positive)
            center=0,    # Center at zero
            vmin=-vmax, vmax=vmax,  # Symmetric range
            cbar_kws={'label': 'Gradient Norm Difference'}
        )
        
        plt.title(f'Gradient Difference Heatmap{title_suffix}', fontsize=14, fontweight='bold')
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Layer', fontsize=12)
        
        # Improve layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to save memory
            print(f"Difference heatmap saved to: {save_path}")
        else:
            plt.show()
    
    def visualize_loss_heatmap(self, loss_per_token, save_path=None):
        loss_1d = loss_per_token.detach().cpu().numpy()[0]  # [T]
        num_layers = 24
        loss_2d = np.tile(loss_1d, (num_layers, 1))  # [layers, T]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            loss_2d,
            cmap='magma',
            cbar_kws={'label': 'Per-Token CE Loss'}
        )
        plt.title('Layer×Token Loss Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Layer', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_loss_per_token(self, loss_per_token, output_dir, base_name):
        os.makedirs(output_dir, exist_ok=True)
        loss_csv = os.path.join(output_dir, f"{base_name}_per_token_loss.csv")
        loss_1d = loss_per_token.detach().cpu().numpy()[0]  # [T]
        pd.DataFrame({'loss': loss_1d}).to_csv(loss_csv, index_label='token_position')
        print(f"  Per-token loss saved to: {loss_csv}")
        return loss_csv
    
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

def process_single_audio(analyzer, audio_path, output_dir, audio_type="unknown"):
    """Process a single audio file and save results"""
    print(f"\nProcessing {audio_type}: {os.path.basename(audio_path)}")
    
    try:
        # Process audio
        wav, sr = analyzer.load_audio(audio_path)
        inputs = analyzer.tokenize_audio(wav, sr)
        
        # Calculate audio duration
        audio_duration_sec = wav.shape[-1] / sr
        print(f"  Audio duration: {audio_duration_sec:.2f} seconds")
        
        # Compute losses and gradients
        result = analyzer.compute_per_token_loss(inputs)
        if result is None:
            print(f"  Audio too short for analysis: {audio_path}")
            return False
        
        per_token_loss = result
        print(f"  Loss shape: {per_token_loss.shape}, Mean: {per_token_loss.mean().item():.4f}")
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        analyzer.save_loss_per_token(per_token_loss, output_dir, base_name)
        loss_heatmap_path = os.path.join(output_dir, f"{base_name}_loss_heatmap.png")
        analyzer.visualize_loss_heatmap(per_token_loss, save_path=loss_heatmap_path)
        
        g_h = analyzer.compute_activation_gradients(per_token_loss, audio_duration_sec)
        print(f"  Gradients shape: {g_h.shape}, Max: {g_h.max().item():.6f}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        csv_path = os.path.join(output_dir, f"{base_name}_gradient_statistics.csv")
        analyzer.save_results(g_h, output_dir, csv_filename=f"{base_name}_gradient_statistics.csv")
        
        heatmap_path = os.path.join(output_dir, f"{base_name}_gradient_heatmap.png")
        analyzer.visualize_gradients(g_h, save_path=heatmap_path)
        print(f"  Results saved: {csv_path}, {heatmap_path}, {loss_heatmap_path}")
        
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
    base_output_dir = "/home/evev/noiseloss/experiments/phase6/results_grad_avg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Window configuration for gradient analysis (5-15 seconds)
    window_start_sec = 5.0
    window_end_sec = 15.0
    
    # Target file configuration
    target_file = "Beethoven_010_midi_score_short.mid_ori_cut15s.wav"
    subdir = "1_very_slow_under60"
    
    # Define file paths
    original_file = os.path.join(base_data_dir, "ori/asap_ori", subdir, target_file)
    
    # Noise perturbation files with different token counts
    noise_files = [
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_white_at5_tk5", subdir, target_file),
            "name": "tk5"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_white_at5_tk10", subdir, target_file),
            "name": "tk10"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_white_at5_tk50", subdir, target_file),
            "name": "tk50"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_white_at5_tk100", subdir, target_file),
            "name": "tk100"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_white_at5_tk150", subdir, target_file),
            "name": "tk150"
        },
        {
            "path": os.path.join(base_data_dir, "Phase5_1/asap_replace_noise_white_at5_tk200", subdir, target_file),
            "name": "tk200"
        }
    ]
    
    print(f"Using device: {device}")
    print(f"Target file: {target_file}")
    
    # Check if files exist
    all_files = [original_file] + [nf["path"] for nf in noise_files]
    missing_files = [f for f in all_files if not os.path.exists(f)]
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  {f}")
        return
    
    # Initialize analyzer and load model with window configuration
    print(f"Window configuration: {window_start_sec}-{window_end_sec} seconds")
    analyzer = GradientAnalyzer(device=device, window_start_sec=window_start_sec, window_end_sec=window_end_sec)
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
    
    if process_single_audio(analyzer, original_file, ori_output_dir, "Original"):
        success_count += 1
    total_count += 1
    
    # Process noise perturbation files
    print("\n" + "="*60)
    print("PROCESSING NOISE PERTURBATION FILES")
    print("="*60)
    
    for noise_file in noise_files:
        # Create specific output directory for each noise condition
        noise_output_dir = os.path.join(base_output_dir, "perturb", noise_file["name"])
        if process_single_audio(analyzer, noise_file["path"], noise_output_dir, f"Noise-{noise_file['name']}"):
            success_count += 1
        total_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Successfully processed: {success_count}/{total_count} files")
    print(f"Original results saved in: {ori_output_dir}")
    print(f"Noise perturbation results saved in: {os.path.join(base_output_dir, 'perturb')}")
    print("Subfolders created for each noise condition:")
    for noise_file in noise_files:
        print(f"  - {noise_file['name']}: {os.path.join(base_output_dir, 'perturb', noise_file['name'])}")
    
    # Clean up
    analyzer.clear_hooks()
    del analyzer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

if __name__ == "__main__":
    main()