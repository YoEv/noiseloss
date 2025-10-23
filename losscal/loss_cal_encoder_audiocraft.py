import torch
import torchaudio
import os
import argparse
import sys
import typing as tp
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes
from torch.nn import functional as F
from torch import nn

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
    model.lm = model.lm.to(dtype=torch.float32)
    model.compression_model = model.compression_model.to(dtype=torch.float32)

    compression_ratio = model.compression_model.frame_rate / model.sample_rate
    print(f"Compression ratio: {compression_ratio:.4f} (generating {model.compression_model.frame_rate} tokens per audio second)")

    return model

def compute_l2_norm(embeddings, audio):
    """Calculate L2 norm between encodec embeddings and original audio with normalization"""
    # Flatten embeddings and audio for comparison
    embeddings_flat = embeddings.view(embeddings.size(0), -1)
    audio_flat = audio.view(audio.size(0), -1)

    # Normalize lengths if needed
    min_length = min(embeddings_flat.size(1), audio_flat.size(1))
    embeddings_flat = embeddings_flat[:, :min_length]
    audio_flat = audio_flat[:, :min_length]

    # Normalize the tensors
    embeddings_norm = F.normalize(embeddings_flat, p=2, dim=1)
    audio_norm = F.normalize(audio_flat, p=2, dim=1)

    # Calculate L2 norm on normalized tensors
    l2_norm = torch.norm(embeddings_norm - audio_norm, p=2, dim=1).mean()
    return l2_norm.item()

def compute_mse_loss(embeddings, audio):
    """Calculate MSE loss between encodec embeddings and original audio with normalization"""
    # Reshape for comparison
    embeddings_flat = embeddings.view(embeddings.size(0), -1)
    audio_flat = audio.view(audio.size(0), -1)

    # Normalize lengths if needed
    min_length = min(embeddings_flat.size(1), audio_flat.size(1))
    embeddings_flat = embeddings_flat[:, :min_length]
    audio_flat = audio_flat[:, :min_length]

    # Normalize the tensors
    embeddings_mean = embeddings_flat.mean(dim=1, keepdim=True)
    embeddings_std = embeddings_flat.std(dim=1, keepdim=True) + 1e-5
    embeddings_norm = (embeddings_flat - embeddings_mean) / embeddings_std

    audio_mean = audio_flat.mean(dim=1, keepdim=True)
    audio_std = audio_flat.std(dim=1, keepdim=True) + 1e-5
    audio_norm = (audio_flat - audio_mean) / audio_std

    # Calculate MSE loss on normalized tensors
    mse_loss = F.mse_loss(embeddings_norm, audio_norm)
    return mse_loss.item()

def _stft(x, n_fft, hop_length, win_length, window, normalized):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        n_fft (int): Size of FFT.
        hop_length (int): Hop length.
        win_length (int): Window length.
        window (Tensor): Window function tensor.
        normalized (bool): Whether to normalize the STFT or not.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, n_fft // 2 + 1).
    """
    # Make sure window is on the same device as x
    window = window.to(x.device)

    x_stft = torch.stft(x, n_fft, hop_length, win_length, window,
                        return_complex=True, normalized=normalized)
    real = x_stft.real
    imag = x_stft.imag

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module.

    Args:
        epsilon (float): Epsilon for numerical stability.
    """
    def __init__(self, epsilon=torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.epsilon)


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module.

    Args:
        epsilon (float): Epsilon for numerical stability.
    """
    def __init__(self, epsilon=torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag + self.epsilon), torch.log(x_mag + self.epsilon))


class STFTLosses(nn.Module):
    """STFT losses.

    Args:
        n_fft (int): Size of FFT.
        hop_length (int): Hop length.
        win_length (int): Window length.
        window (str): Window function type.
        normalized (bool): Whether to use normalized STFT or not.
        epsilon (float): Epsilon for numerical stability.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 120, win_length: int = 600,
                 window: str = "hann_window", normalized: bool = False,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.window_name = window
        self.epsilon = epsilon
        self.spectral_convergenge_loss = SpectralConvergenceLoss(epsilon)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(epsilon)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Predicted signal (B, T).
            y (torch.Tensor): Groundtruth signal (B, T).
        Returns:
            torch.Tensor: Spectral convergence loss value.
            torch.Tensor: Log STFT magnitude loss value.
        """
        # Create window on the same device as input tensors
        window = getattr(torch, self.window_name)(self.win_length).to(x.device)

        # Ensure x and y have the same length
        min_length = min(x.size(-1), y.size(-1))
        x = x[..., :min_length]
        y = y[..., :min_length]

        # Normalize the input signals
        x_mean, x_std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True) + 1e-5
        y_mean, y_std = y.mean(dim=-1, keepdim=True), y.std(dim=-1, keepdim=True) + 1e-5

        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        x_mag = _stft(x_norm, self.n_fft, self.hop_length,
                      self.win_length, window, self.normalized)
        y_mag = _stft(y_norm, self.n_fft, self.hop_length,
                      self.win_length, window, self.normalized)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class STFTLoss(nn.Module):
    """Single Resolution STFT loss.

    Args:
        n_fft (int): Nb of FFT.
        hop_length (int): Hop length.
        win_length (int): Window length.
        window (str): Window function type.
        normalized (bool): Whether to use normalized STFT or not.
        epsilon (float): Epsilon for numerical stability.
        factor_sc (float): Coefficient for the spectral loss.
        factor_mag (float): Coefficient for the magnitude loss.
    """
    def __init__(self, n_fft=1024, hop_length=120, win_length=600,
                 window="hann_window", normalized=False,
                 factor_sc=0.1, factor_mag=0.1,
                 epsilon=torch.finfo(torch.float32).eps):
        super().__init__()
        self.loss = STFTLosses(n_fft, hop_length, win_length, window, normalized, epsilon)
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Single resolution STFT loss.
        """
        sc_loss, mag_loss = self.loss(x, y)
        # Average the losses instead of summing them
        return (self.factor_sc * sc_loss + self.factor_mag * mag_loss) / 2.0


def compute_stft_loss(embeddings, audio, n_fft=1024, hop_length=120, win_length=600):
    """Calculate STFT loss between encodec embeddings and original audio using MusicGen's approach"""
    # Convert embeddings to float if needed
    if not torch.is_floating_point(embeddings):
        embeddings = embeddings.float()

    # Convert audio to float if needed
    if not torch.is_floating_point(audio):
        audio = audio.float()

    # Reshape embeddings to match audio dimensions if needed
    embeddings_flat = embeddings.view(embeddings.size(0), -1)
    audio_flat = audio.view(audio.size(0), -1)

    # Resample embeddings to match audio length if they differ significantly
    if embeddings_flat.size(1) != audio_flat.size(1):
        # Use interpolation to match lengths
        embeddings_resampled = F.interpolate(
            embeddings_flat.unsqueeze(1),  # Add channel dim for interpolate
            size=audio_flat.size(1),
            mode='linear',
            align_corners=False
        ).squeeze(1)  # Remove channel dim after interpolate
    else:
        embeddings_resampled = embeddings_flat

    # Normalize the tensors
    embeddings_mean = embeddings_resampled.mean(dim=1, keepdim=True)
    embeddings_std = embeddings_resampled.std(dim=1, keepdim=True) + 1e-5
    embeddings_norm = (embeddings_resampled - embeddings_mean) / embeddings_std

    audio_mean = audio_flat.mean(dim=1, keepdim=True)
    audio_std = audio_flat.std(dim=1, keepdim=True) + 1e-5
    audio_norm = (audio_flat - audio_mean) / audio_std

    # Create STFT loss module
    stft_loss = STFTLoss(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Calculate loss using the normalized resampled embeddings
    loss = stft_loss(embeddings_norm.squeeze(0), audio_norm.squeeze(0))

    return loss.item()

def compute_mel_loss(embeddings, audio, sample_rate=16000, n_mels=80):
    """Calculate Mel-spectrogram loss between encodec embeddings and original audio with normalization"""
    # Convert embeddings to audio domain if needed
    if embeddings.dim() > 2:
        embeddings = embeddings.view(embeddings.size(0), -1)
    if audio.dim() > 2:
        audio = audio.view(audio.size(0), -1)

    # Ensure same length
    min_length = min(embeddings.size(1), audio.size(1))
    embeddings = embeddings[:, :min_length]
    audio = audio[:, :min_length]

    # Normalize the tensors
    embeddings_mean = embeddings.mean(dim=1, keepdim=True)
    embeddings_std = embeddings.std(dim=1, keepdim=True) + 1e-5
    embeddings_norm = (embeddings - embeddings_mean) / embeddings_std

    audio_mean = audio.mean(dim=1, keepdim=True)
    audio_std = audio.std(dim=1, keepdim=True) + 1e-5
    audio_norm = (audio - audio_mean) / audio_std

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels
    ).to(embeddings.device)

    # Compute mel spectrograms
    mel_embeddings = mel_transform(embeddings_norm)
    mel_audio = mel_transform(audio_norm)

    # Calculate L1 loss on log mel spectrograms
    log_mel_embeddings = torch.log(mel_embeddings + 1e-5)
    log_mel_audio = torch.log(mel_audio + 1e-5)
    mel_loss = F.l1_loss(log_mel_embeddings, log_mel_audio)

    return mel_loss.item()

def process_single_audio(audio_path, model, device, loss_type="l2_norm"):
    try:
        # Load audio
        audio, sr = audio_read(audio_path)

        # Convert stereo to mono if needed
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to(device)
        original_length = audio.shape[-1]

        # Get encodec embeddings
        with torch.no_grad():
            # Get the raw embeddings from the encoder part of the compression model
            encoded_frames = model.compression_model.encode(audio.unsqueeze(0))
            # Extract the first embedding [0] as requested
            embeddings = encoded_frames[0]

            # Convert to float if needed for all loss types
            if not torch.is_floating_point(embeddings):
                embeddings = embeddings.float()

        # Calculate the specified loss
        if loss_type == "l2_norm":
            return compute_l2_norm(embeddings, audio.unsqueeze(0))
        elif loss_type == "mse":
            return compute_mse_loss(embeddings, audio.unsqueeze(0))
        elif loss_type == "stft":
            return compute_stft_loss(embeddings, audio.unsqueeze(0))
        elif loss_type == "mel":
            return compute_mel_loss(embeddings, audio.unsqueeze(0), sr)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {str(e)}")
        return None

def process_audio_directory(audio_dir, model, device, output_file=None, loss_type="l2_norm"):
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in directory {audio_dir}")

    print(f"\nProcessing {len(audio_files)} audio files (Loss type: {loss_type})")
    results = []
    error_files = []  # Track files that failed processing

    # Open output file if specified
    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"Results will be saved to: {output_file}")

    for filename in tqdm(audio_files, desc="Processing progress"):
        audio_path = os.path.join(audio_dir, filename)
        loss = process_single_audio(audio_path, model, device, loss_type)
        if loss is not None:
            results.append((filename, loss))
            # Write to file immediately if output file is specified
            if out_file:
                out_file.write(f"{filename}: {loss:.8f}\n")
                out_file.flush()  # Ensure data is written immediately
        else:
            error_files.append(filename)

    # Close output file if opened
    if out_file:
        out_file.close()

    # Write error files list to file
    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Number of files that failed processing: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n{len(error_files)} failed files saved to: {error_file_path}")

    return results, error_files

def get_missing_files(audio_dir, results_file):
    """Get list of files that haven't been processed yet"""
    # Get all audio files in directory
    all_audio_files = set(f for f in os.listdir(audio_dir)
                         if f.lower().endswith(('.wav', '.mp3')))

    # Get processed files from results file
    processed_files = set()
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename = line.split(':', 1)[0].strip()
                    processed_files.add(filename)
    except FileNotFoundError:
        print(f"Results file {results_file} does not exist, will process all files")

    # Return files that are in all_audio_files but not in processed_files
    missing_files = all_audio_files - processed_files
    return list(missing_files)

def process_missing_files(audio_dir, results_file, loss_type="l2_norm"):
    """Process missing files (using CPU)"""
    missing_files = get_missing_files(audio_dir, results_file)

    if not missing_files:
        print("No missing files to process")
        return

    print(f"Found {len(missing_files)} unprocessed files, will process using CPU (Loss type: {loss_type})")

    # Print all missing filenames before processing
    print("\nList of unprocessed files:")
    for filename in missing_files:
        print(f"{filename}")
    print("\nStarting to process these files...\n")

    # Load model on CPU
    device = "cpu"
    print(f"Using device: {device}")
    model = load_model(device)

    # Open results file in append mode
    with open(results_file, 'a') as out_file:
        for filename in tqdm(missing_files, desc="CPU processing progress"):
            audio_path = os.path.join(audio_dir, filename)
            loss = process_single_audio(audio_path, model, device, loss_type)
            if loss is not None:
                out_file.write(f"{filename}: {loss:.8f}\n")
                out_file.flush()  # Ensure data is written immediately

def main():
    parser = argparse.ArgumentParser(description="Calculate loss between encodec embeddings and original audio")
    parser.add_argument("--audio_dir", type=str, default="pitch_out_wav",
                        help="Directory containing audio files")
    parser.add_argument("--output_file", type=str, default="encoder_results.txt",
                        help="Output file path for results")
    parser.add_argument("--process_missing", action="store_true",
                        help="Process missing files (using CPU)")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU processing for all files")
    parser.add_argument("--loss_type", type=str,
                        choices=["l2_norm", "mse", "stft", "mel"],
                        default="l2_norm",
                        help="Loss function type: l2_norm, mse, stft, or mel")

    args = parser.parse_args()

    # Redirect stdout to file if output_file is specified and not processing missing files
    original_stdout = None
    if args.output_file and not args.process_missing:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    try:
        if args.process_missing and args.output_file:
            # Process missing files on CPU
            process_missing_files(args.audio_dir, args.output_file, args.loss_type)
        else:
            # Normal processing
            device = setup_device(args.force_cpu)
            print(f"Using device: {device}")

            model = load_model(device)

            assert os.path.isdir(args.audio_dir), f"Directory does not exist: {args.audio_dir}"

            results, error_files = process_audio_directory(
                args.audio_dir, model, device, args.output_file, args.loss_type
            )

            if not args.output_file:
                print("\nProcessing results:")
                for filename, loss in results:
                    print(f"{filename}: {loss:.8f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if 'device' in locals() and device == 'cuda':
            torch.cuda.empty_cache()
        print("GPU memory cleared")

        # Restore stdout if redirected
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()