import torch
import torchaudio
import os
import argparse
import sys
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import numpy as np
import torch.nn.functional as F

try:
    from nnAudio import features as nnAudioFeatures
    NNAUDIO_INSTALLED = True
except ImportError:
    NNAUDIO_INSTALLED = False
    print("WARNING: nnAudio not installed. CQT feature extraction will not be available.")

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_mert_model(device, model_name="m-a-p/MERT-v0-public"):
    # Load MERT model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_auth_token=True)
    model = model.to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True, use_auth_token=True)
    return model, processor

class CQTEncoder(torch.nn.Module):
    """Dedicated CQT encoder for MERT loss calculation"""
    def __init__(self, input_dim, hidden_dim=768):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

def create_cqt_extractor(sample_rate=24000, hop_length=480, n_bins=336):
    """Create a CQT feature extractor using nnAudio"""
    if not NNAUDIO_INSTALLED:
        raise ImportError("nnAudio is required for CQT feature extraction. Install with: pip install nnAudio")

    return nnAudioFeatures.cqt.CQT(
        sr=sample_rate,
        hop_length=hop_length,  # sample_rate//50 = 480
        fmin=32.7,  # Lowest note frequency (C1)
        fmax=None,  # Automatically determined
        n_bins=n_bins,  # Number of frequency bins
        bins_per_octave=n_bins//7,  # Bins per octave
        filter_scale=1,
        norm=1,
        window='hann',
        center=True,
        pad_mode='constant',
        trainable=False,
        output_format='Magnitude',
        verbose=False
    )

def apply_time_masking(hidden_states, mask_prob=0.1, mask_length=10):
    """Apply time masking to hidden states"""
    batch_size, seq_length, hidden_dim = hidden_states.shape

    # Create mask indices
    mask = torch.zeros((batch_size, seq_length), device=hidden_states.device)

    # Determine number of masks
    num_masks = max(1, int(mask_prob * seq_length / mask_length))

    # Apply random masks
    for i in range(batch_size):
        for _ in range(num_masks):
            start = torch.randint(0, seq_length - mask_length + 1, (1,)).item()
            mask[i, start:start+mask_length] = 1

    # Expand mask to match hidden dimension
    mask = mask.unsqueeze(-1).expand(-1, -1, hidden_dim)

    # Create masked embedding (learnable parameter)
    masked_embed = torch.randn_like(hidden_states) * 0.1

    # Apply mask
    masked_hidden = torch.where(mask == 1, masked_embed, hidden_states)

    return masked_hidden, mask

def compute_mert_loss(model, processor, audio, device, temperature=0.1, n_negatives=100,
                     mask_prob=0.1, mask_length=10, cqt_bins=336):
    """
    Compute MERT loss with actual CQT features, masking and dedicated CQT encoder

    Args:
        model: MERT model
        processor: MERT feature extractor
        audio: Audio numpy array (1D)
        device: Device to run computation on
        temperature: Temperature for scaling logits (default: 0.1)
        n_negatives: Number of negative samples (default: 100)
        mask_prob: Probability of masking time steps (default: 0.1)
        mask_length: Length of each mask span (default: 10)
        cqt_bins: Number of CQT bins (default: 336)

    Returns:
        Total loss value
    """
    # Process audio - the processor expects numpy arrays
    inputs = processor(audio, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)

    # Convert audio to tensor for CQT extraction
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Add batch dimension
    else:
        audio_tensor = audio

    # Extract CQT features directly
    cqt_extractor = create_cqt_extractor(sample_rate=processor.sampling_rate, n_bins=cqt_bins)
    cqt_extractor = cqt_extractor.to(device)

    # Get CQT features
    cqt_features = cqt_extractor(audio_tensor.to(device))
    cqt_features = cqt_features.transpose(1, 2)  # [batch, time, freq]

    # Create dedicated CQT encoder
    cqt_encoder = CQTEncoder(input_dim=cqt_bins).to(device)

    # Forward pass through model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden states from all layers
    hidden_states = outputs.hidden_states

    # Get the last hidden state (representation)
    last_hidden = hidden_states[-1]  # [batch_size, sequence_length, hidden_size]
    batch_size, seq_length, hidden_dim = last_hidden.shape

    # Resize CQT features to match hidden state sequence length
    if cqt_features.shape[1] != seq_length:
        cqt_features = F.interpolate(
            cqt_features.transpose(1, 2),
            size=seq_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

    # Apply masking to hidden states
    masked_hidden, mask = apply_time_masking(
        last_hidden,
        mask_prob=mask_prob,
        mask_length=mask_length
    )

    # Process CQT features through dedicated encoder
    encoded_cqt = cqt_encoder(cqt_features)

    # 1. Contrastive loss with masked prediction
    # Normalize the representations for cosine similarity
    normalized_hidden = F.normalize(masked_hidden, dim=-1)
    normalized_target = F.normalize(last_hidden, dim=-1)

    # Reshape for easier processing
    normalized_hidden_flat = normalized_hidden.view(-1, hidden_dim)  # [batch_size*seq_length, hidden_dim]
    normalized_target_flat = normalized_target.view(-1, hidden_dim)  # [batch_size*seq_length, hidden_dim]

    # Compute positive similarities (diagonal elements)
    positive_similarities = torch.sum(normalized_hidden_flat * normalized_target_flat, dim=-1)

    # Compute negative similarities (off-diagonal elements)
    # For each anchor, sample n_negatives random negatives
    negative_indices = []
    for i in range(batch_size * seq_length):
        # Sample random indices different from i
        neg_idx = torch.randint(
            0, batch_size * seq_length, (n_negatives,), device=device
        )
        # Ensure i is not in neg_idx
        neg_idx = torch.where(neg_idx == i, (i + 1) % (batch_size * seq_length), neg_idx)
        negative_indices.append(neg_idx)

    negative_indices = torch.stack(negative_indices)  # [batch_size*seq_length, n_negatives]

    # Gather negative samples
    negative_samples = normalized_target_flat[negative_indices]  # [batch_size*seq_length, n_negatives, hidden_dim]

    # Compute negative similarities
    negative_similarities = torch.bmm(
        normalized_hidden_flat.unsqueeze(1),  # [batch_size*seq_length, 1, hidden_dim]
        negative_samples.transpose(1, 2)  # [batch_size*seq_length, hidden_dim, n_negatives]
    ).squeeze(1)  # [batch_size*seq_length, n_negatives]

    # Concatenate positive and negative similarities
    logits = torch.cat([
        positive_similarities.unsqueeze(1),  # [batch_size*seq_length, 1]
        negative_similarities  # [batch_size*seq_length, n_negatives]
    ], dim=1)  # [batch_size*seq_length, 1+n_negatives]

    # Scale by temperature
    logits = logits / temperature

    # Create labels (positive sample is at index 0)
    labels = torch.zeros(batch_size * seq_length, dtype=torch.long, device=device)

    # Compute NCE loss
    nce_loss = F.cross_entropy(logits, labels)

    # 2. CQT reconstruction loss
    # Predict CQT features from hidden states
    # For simplicity, we'll use MSE between encoded CQT and masked hidden states
    # Only compute loss for non-masked positions
    non_mask = 1.0 - mask.float()
    lcqt_loss = F.mse_loss(
        masked_hidden * non_mask,
        encoded_cqt * non_mask,
        reduction='sum'
    ) / (non_mask.sum() + 1e-6)

    # Apply loss weights from the MERT config
    # loss_weights: [10, 1] - first component (LH) has weight 10, second component (LCQT) has weight 1
    lh_weight = 10.0
    lcqt_weight = 1.0

    # Total loss
    total_loss = (lh_weight * nce_loss + lcqt_weight * lcqt_loss) / (lh_weight + lcqt_weight)

    return total_loss.item()

def process_single_audio(audio_path, model, processor, device, **kwargs):
    try:
        # Load and preprocess audio
        audio, sr = torchaudio.load(audio_path)
        if sr != processor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, processor.sampling_rate)
            audio = resampler(audio)

        # Debug print to see the shape before processing
        print(f"Original audio shape: {audio.shape}")

        # Ensure audio is in the correct format for processor
        # The processor expects a 1D array, not a 2D tensor with channels
        if audio.dim() == 2:  # [channels, time]
            # If stereo, convert to mono by averaging channels
            if audio.size(0) > 1:
                audio = torch.mean(audio, dim=0)
            else:
                audio = audio.squeeze(0)  # Remove channel dimension if mono

        # Now audio should be 1D [time]
        if audio.dim() != 1:
            raise ValueError(f"Expected 1D audio tensor, got shape {audio.shape}")

        # Convert to numpy array for processor
        audio_np = audio.numpy()

        # Compute loss with numpy array instead of tensor
        loss = compute_mert_loss(model, processor, audio_np, device, **kwargs)
        return loss

    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {str(e)}")
        return None

def process_audio_directory(audio_dir, model, processor, device, output_file=None, **kwargs):
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in directory {audio_dir}")

    print(f"\nProcessing {len(audio_files)} audio files")
    results = []
    error_files = []

    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"Results will be saved to: {output_file}")

    for filename in tqdm(audio_files, desc="Processing"):
        audio_path = os.path.join(audio_dir, filename)
        loss = process_single_audio(audio_path, model, processor, device, **kwargs)
        if loss is not None:
            results.append((filename, loss))
            if out_file:
                out_file.write(f"{filename}: {loss:.8f}\n")
                out_file.flush()
        else:
            error_files.append(filename)

    if out_file:
        out_file.close()

    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Failed to process {len(error_files)} files\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\nFailed files saved to: {error_file_path}")

    return results, error_files

def main():
    parser = argparse.ArgumentParser(description="Calculate MERT loss for audio files")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_file", type=str,
                        help="Output file to save results")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU instead of GPU")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for scaling logits (default: 0.1)")
    parser.add_argument("--n_negatives", type=int, default=100,
                        help="Number of negative samples (default: 100)")
    parser.add_argument("--mask_prob", type=float, default=0.1,
                        help="Probability of masking time steps (default: 0.1)")
    parser.add_argument("--mask_length", type=int, default=10,
                        help="Length of each mask span (default: 10)")
    parser.add_argument("--cqt_bins", type=int, default=336,
                        help="Number of CQT bins (default: 336)")
    parser.add_argument("--model_name", type=str, default="m-a-p/MERT-v0-public",
                        help="MERT model name or path (default: m-a-p/MERT-v0-public)")

    args = parser.parse_args()

    device = setup_device(args.force_cpu)
    print(f"Using device: {device}")

    model, processor = load_mert_model(device, args.model_name)

    assert os.path.isdir(args.audio_dir), f"Directory does not exist: {args.audio_dir}"

    # Pass all relevant parameters to the processing function
    results, error_files = process_audio_directory(
        args.audio_dir,
        model,
        processor,
        device,
        output_file=args.output_file,
        temperature=args.temperature,
        n_negatives=args.n_negatives,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
        cqt_bins=args.cqt_bins
    )

    if not args.output_file:
        print("\nProcessing results:")
        for filename, loss in results:
            print(f"{filename}: {loss:.4f}")

    if device == "cuda":
        torch.cuda.empty_cache()
        print("GPU memory cleared")

if __name__ == "__main__":
    main()