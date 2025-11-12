import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModelForCausalLM
import torchaudio
from tqdm import tqdm

# Add paths for YuEGP dependencies
sys.path.append('/home/evev/asap-dataset')
sys.path.append('/home/evev/asap-dataset/external/YuEGP/inference')
sys.path.append('/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer')

from codecmanipulator import CodecManipulator
from models.soundstream_hubert_new import SoundStream
from transformers import LogitsProcessor, LogitsProcessorList
from mmtokenizer import _MMSentencePieceTokenizer


class YuEGPGenerator:
    def __init__(self, device="cuda", dtype="float16", verbose=True):
        self.device = device
        self.verbose = verbose
        self.dtype = self._get_torch_dtype(dtype)
        
        # Initialize codec manipulator for stage2 (8 quantizers)
        self.codec_manipulator = CodecManipulator(
            codec_type="xcodec", 
            quantizer_begin=0, 
            n_quantizer=8
        )
        
        # pre-compute xcodec ranges for stage2 (K=8)
        self.cb0_start = self.codec_manipulator.global_offset + self.codec_manipulator.quantizer_begin * self.codec_manipulator.codebook_size
        self.cb1_start = self.cb0_start + self.codec_manipulator.codebook_size
        self.cb8_end  = self.cb0_start + self.codec_manipulator.codebook_size * self.codec_manipulator.n_quantizer - 1

        # Load XCodec model for audio reconstruction
        self.xcodec_model = self._load_xcodec_model()
        
        # Load YuE-s2 model (HF model only)
        self.model = self._load_yue_model()

        # mm tokenizer for special tokens ([SOA]/[EOA]/<stage_1>/<stage_2>)
        self.mmtokenizer = self._load_mm_tokenizer()

        # Align allowed token range with actual model vocab to avoid out-of-range ids
        self.model_vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)
        self.allowed_start = int(self.cb1_start)
        self.allowed_end = int(min(self.cb8_end, self.model_vocab_size - 1)) if self.model_vocab_size > 0 else int(self.cb8_end)

        if self.verbose:
            print(f"Loaded YuE-s2 model: m-a-p/YuE-s2-1B-general")
            print(f"Model vocab size: {self.model_vocab_size}")
            print(f"MM tokenizer vocab size: {self.mmtokenizer.vocab_size}")
            print(f"Codebook ranges (stage2 K=8): cb0=[{self.cb0_start}, {self.cb1_start-1}], allowed-gen=[{self.allowed_start}, {self.allowed_end}]")

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch dtype"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        return dtype_map.get(dtype_str, torch.float16)

    def _load_xcodec_model(self):
        """Load XCodec SoundStream model for audio reconstruction"""
        model = SoundStream(
            n_filters=32,
            D=128,
            ratios=[8, 5, 4, 2],
            sample_rate=16000,
            bins=1024,
            normalize=True,
            causal=True,
        )
        model_path = '/home/evev/asap-dataset/external/YuEGP/inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location='cpu')
            state = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
            model.load_state_dict(state, strict=False)
            if self.verbose:
                print(f"Loaded XCodec checkpoint from {model_path}")
        model = model.to(self.device)
        model.eval()
        return model

    def _load_yue_model(self):
        """Load YuE-s2-1B-general model (HF)"""
        model_name = "m-a-p/YuE-s2-1B-general"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=self.dtype, 
            trust_remote_code=True
        ).to(self.device)
        model.eval()
        return model

    def _load_mm_tokenizer(self):
        """Load mm tokenizer for special/control tokens"""
        tok_path = "/home/evev/asap-dataset/external/YuEGP/inference/mm_tokenizer_v0.2_hf/tokenizer.model"
        tokenizer = _MMSentencePieceTokenizer(tok_path)
        return tokenizer

    class _BlockTokenAllowRangeProcessor(LogitsProcessor):
        def __init__(self, start_id, end_id):
            self.start_id = int(start_id)
            self.end_id = int(end_id)

        def __call__(self, input_ids, scores):
            vocab = scores.size(-1)
            s = max(self.start_id, 0)
            e = min(self.end_id, vocab - 1)
            if s > 0:
                scores[:, :s] = -float("inf")
            if e < vocab - 1:
                scores[:, e+1:] = -float("inf")
            return scores

    def generate_stage2_tokens_unconditional(self, num_frames: int = 750, cb0_seed: str = "mid", top_k: int = 0, temperature: float = 1.0) -> List[int]:
        """
        Unconditional Stage-2 generation via teacher forcing with sampling:
        - Build prompt: [SOA] <stage_1> + cb0_seq + <stage_2>
        - For each frame t: append cb0[t], then generate 7 tokens (codebook 1..7) with per-step range gating
        Returns flattened ids of length num_frames*8, starting with cb0.
        """
        assert num_frames > 0, "num_frames must be positive"
        
        if self.verbose:
            print(f"Starting generation of {num_frames} frames ({num_frames * 7} tokens)...")

        # Build cb0 (values in [0, 1023]), then offset to token ids in [cb0_start, cb0_start+1023]
        if cb0_seed == "mid":
            cb0_values = np.full((num_frames,), 512, dtype=np.int32)
        elif cb0_seed == "zero":
            cb0_values = np.zeros((num_frames,), dtype=np.int32)
        elif cb0_seed == "random":
            cb0_values = np.random.randint(0, self.codec_manipulator.codebook_size, size=(num_frames,), dtype=np.int32)
        else:
            cb0_values = np.full((num_frames,), 512, dtype=np.int32)

        cb0_ids = (self.cb0_start + cb0_values).astype(np.int32)

        # Construct initial prompt: [SOA] <stage_1> + flattened cb0 (all frames) + <stage_2>
        prompt_ids = np.concatenate([
            np.array([self.mmtokenizer.soa, self.mmtokenizer.stage_1], dtype=np.int32),
            cb0_ids,
            np.array([self.mmtokenizer.stage_2], dtype=np.int32),
        ], axis=0)

        prompt_ids = torch.as_tensor(prompt_ids, device=self.device).unsqueeze(0)  # (1, L)
        len_prompt = prompt_ids.shape[-1]

        attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        safe_top_k = int(top_k) if isinstance(top_k, (int, np.integer)) and top_k > 0 else 0
        
        total_steps = num_frames * 7
        pbar = tqdm(total=total_steps, desc="Generating tokens", leave=False)

        # Teacher-forcing loop: for each frame, append cb0[t], then generate 7 tokens (cb1..cb7)
        with torch.no_grad():
            for t in range(num_frames):
                cb0_t = torch.as_tensor([[int(cb0_ids[t])]], device=self.device)
                prompt_ids = torch.cat([prompt_ids, cb0_t], dim=1)
                attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

                for step in range(1, 8):
                    start_id = int(self.cb0_start + step * self.codec_manipulator.codebook_size)
                    end_id = int(start_id + self.codec_manipulator.codebook_size - 1)
                    if self.model_vocab_size > 0:
                        end_id = min(end_id, self.model_vocab_size - 1)

                    allow_processor = LogitsProcessorList([
                        self._BlockTokenAllowRangeProcessor(start_id, end_id)
                    ])

                    out = self.model.generate(
                        input_ids=prompt_ids,
                        attention_mask=attention_mask,
                        min_new_tokens=1,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=float(temperature),
                        top_k=safe_top_k,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        eos_token_id=self.mmtokenizer.eoa,
                        pad_token_id=self.mmtokenizer.eoa,
                        logits_processor=allow_processor,
                        use_cache=True,
                    )

                    if out.shape[1] - prompt_ids.shape[1] != 1:
                        pbar.close()
                        print(f"[Warn] Unexpected new token count at frame {t}, step {step}: {out.shape[1]-prompt_ids.shape[1]}")
                        return []

                    prompt_ids = out
                    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
                    
                    pbar.update(1)
                    
                    if (t * 7 + step) % 100 == 0:
                        torch.cuda.empty_cache()
        
        pbar.close()

        # Return tokens after initial prompt (contains [cb0, 7]*num_frames)
        flat_out = prompt_ids[0, len_prompt:].detach().cpu().tolist()
        if self.verbose:
            print(f"Stage2 generated tokens: {len(flat_out)} (expected {num_frames*8})")
            if flat_out:
                print(f"Token range: {min(flat_out)} - {max(flat_out)}")
        return flat_out

    def tokens_to_audio(self, token_ids: List[int]) -> Optional[torch.Tensor]:
        """Convert token IDs back to audio using XCodec"""
        try:
            if not token_ids:
                print("Empty token_ids list")
                return None
            
            if self.verbose:
                print(f"Processing {len(token_ids)} tokens")
                print(f"Token range: {min(token_ids)} - {max(token_ids)}")
                unique_tokens = len(set(token_ids))
                print(f"Unique tokens: {unique_tokens}/{len(token_ids)} ({unique_tokens/len(token_ids)*100:.1f}% diversity)")
                
                if len(token_ids) >= 8:
                    for cb in range(8):
                        cb_tokens = token_ids[cb::8]
                        cb_unique = len(set(cb_tokens))
                        print(f"  Codebook {cb}: {cb_unique}/{len(cb_tokens)} unique tokens")
            
            tokens_np = self.codec_manipulator.ids2npy(np.array(token_ids))
            if self.verbose:
                print(f"CodecManipulator conversion successful: {tokens_np.shape}")  # (8, T)
                print(f"Tokens numpy shape: {tokens_np.shape}, dtype: {tokens_np.dtype}")
                print(f"Tokens numpy range: [{tokens_np.min()}, {tokens_np.max()}]")
            
            tokens_tensor = torch.tensor(tokens_np, device=self.device, dtype=torch.long).unsqueeze(1)  # (8, T) -> (8, 1, T)
            if self.verbose:
                print(f"Tokens tensor shape before decode: {tokens_tensor.shape}")
            
            with torch.no_grad():
                if hasattr(self.xcodec_model, 'decode'):
                    audio = self.xcodec_model.decode(tokens_tensor)
                else:
                    audio = self.xcodec_model.decoder(tokens_tensor)
            if isinstance(audio, (tuple, list)):
                audio = audio[0]
            audio = audio.squeeze().cpu()
            if audio.dim() == 0 or audio.numel() == 0:
                print("Generated audio is empty")
                return None
            if self.verbose:
                duration = len(audio) / 16000.0
                print(f"Generated audio duration: {duration:.2f} seconds")
                print(f"Audio tensor shape: {audio.shape}, range: [{audio.min():.4f}, {audio.max():.4f}]")
            return audio
        except Exception as e:
            print(f"Error converting tokens to audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_audio(self, top_k: int = 50, target_length: int = 750, temperature: float = 1.0, cb0_seed: str = "mid") -> Optional[torch.Tensor]:
        """
        Generate audio end-to-end using Stage-2 teacher forcing with sampling.
        target_length: number of frames (50 fps). 750 ≈ 15s
        """
        if self.verbose:
            print(f"Generating audio (stage2 TF) with target_frames={target_length}, cb0_seed={cb0_seed}, top_k={top_k}, T={temperature}")
        token_ids = self.generate_stage2_tokens_unconditional(
            num_frames=target_length,
            cb0_seed=cb0_seed,
            top_k=top_k,
            temperature=temperature
        )
        if not token_ids:
            return None
        audio = self.tokens_to_audio(token_ids)
        return audio

def batch_unconditional_generation(output_dir: str, top_k_values: List[int], 
                                  num_samples_per_k: int = 20, target_length: int = 750, 
                                  temperature: float = 1.0, device: str = "cuda", 
                                  dtype: str = "float16"):
    """Generate multiple audio samples for different top_k values
    
    Args:
        target_length: frame（50fps），750≈15s
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = YuEGPGenerator(device=device, dtype=dtype, verbose=True)
    
    total_samples = len(top_k_values) * num_samples_per_k
    pbar = tqdm(total=total_samples, desc="Generating audio samples")
    
    for top_k in top_k_values:
        # Create subdirectory for this top_k value
        k_dir = os.path.join(output_dir, f"top_k_{top_k}")
        os.makedirs(k_dir, exist_ok=True)
        
        print(f"\nGenerating {num_samples_per_k} samples with top_k={top_k}")
        
        success_count = 0
        for i in range(num_samples_per_k):
            audio = generator.generate_audio(
                top_k=top_k, 
                target_length=target_length,
                temperature=temperature,
                cb0_seed="mid",
            )
            
            if audio is not None and len(audio) > 16000:
                sr = 16000
                out_path = os.path.join(k_dir, f"sample_{i+1}.wav")
                torchaudio.save(out_path, audio.unsqueeze(0), sample_rate=sr)
                print(f"Saved: {out_path}")
                success_count += 1
            else:
                print(f"Failed to generate sample {i+1} for top_k={top_k} (audio too short or None)")

            pbar.update(1)
        
        print(f"Completed top_k={top_k}: {success_count}/{num_samples_per_k} successful")

    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Generate unconditional music using YuEGP stage2")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated audio")
    parser.add_argument("--top_k_values", type=int, nargs="+", default=[10, 50, 100, 150, 200, 250, 500],
                       help="List of top_k values to use for generation")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per top_k value")
    parser.add_argument("--target_length", type=int, default=750, help="Target number of frames (50 fps). 750 ≈ 15s")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                       help="Model dtype")
    
    args = parser.parse_args()
    
    print(f"YuEGP Unconditional Generation")
    print(f"Output directory: {args.output_dir}")
    print(f"Top-k values: {args.top_k_values}")
    print(f"Samples per top-k: {args.num_samples}")
    print(f"Target frames: {args.target_length} (≈ {args.target_length/50:.1f}s)")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    
    # Generate audio samples
    batch_unconditional_generation(
        output_dir=args.output_dir,
        top_k_values=args.top_k_values,
        num_samples_per_k=args.num_samples,
        target_length=args.target_length,
        temperature=args.temperature,
        device=args.device,
        dtype=args.dtype
    )
    
    # Clean up GPU memory (guarded)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Optionally free cached memory; keep commented to avoid crashes on some drivers
            # torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Warn] CUDA cleanup skipped due to error: {e}")

if __name__ == "__main__":
    main()
