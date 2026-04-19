import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import MusicgenForConditionalGeneration
from transformers.models.musicgen.modeling_musicgen import MusicgenSinusoidalPositionalEmbedding

if not hasattr(MusicgenSinusoidalPositionalEmbedding, "_offset_patched"):
    _orig_init = MusicgenSinusoidalPositionalEmbedding.__init__
    def _patched_init(self, num_positions, embedding_dim):
        self.offset = 2
        _orig_init(self, num_positions, embedding_dim)
    MusicgenSinusoidalPositionalEmbedding.__init__ = _patched_init
    MusicgenSinusoidalPositionalEmbedding._offset_patched = True

from phase7_release.lib.repro.data_paths import get_exp11_split_paths

NUM_CODEBOOKS = 4

def _uid(token_loss_path: str) -> str:
    return hashlib.md5(token_loss_path.encode("utf-8")).hexdigest()[:16]


MAX_AUDIO_SECONDS = 30  # cap to prevent OOM on long audio (seq_len=1500 @ 50tok/s)


def _load_audio_mono_32k(audio_path: str, device: torch.device) -> torch.Tensor:
    audio, sr = sf.read(audio_path)
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T
    wav = torch.from_numpy(audio).float()
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    wav = wav.mean(dim=0, keepdim=True)
    max_samples = int(MAX_AUDIO_SECONDS * 32000)
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]
    return wav.to(device)


def _entropy_from_logits_multi_codebook(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), dim=-1)
    log_p = torch.log(p.clamp_min(1e-12))
    h_kt = -(p * log_p).sum(dim=-1)
    return h_kt


def _fix_len_kt(h_kt: np.ndarray, fixed_t: int) -> np.ndarray:
    h_kt = h_kt.astype(np.float32, copy=False)
    if h_kt.ndim == 1:
        h_kt = h_kt[None, :]
    if h_kt.shape[0] > NUM_CODEBOOKS:
        h_kt = h_kt[:NUM_CODEBOOKS, :]
    elif h_kt.shape[0] < NUM_CODEBOOKS:
        pad_k = np.zeros((NUM_CODEBOOKS - h_kt.shape[0], h_kt.shape[1]), dtype=np.float32)
        h_kt = np.concatenate([h_kt, pad_k], axis=0)
    if fixed_t <= 0:
        return h_kt
    if h_kt.shape[1] >= fixed_t:
        return h_kt[:, :fixed_t].copy()
    out = np.full((h_kt.shape[0], fixed_t), np.nan, dtype=np.float32)
    out[:, : h_kt.shape[1]] = h_kt
    return out


def _extract_entropy_curve(model: MusicgenForConditionalGeneration, audio_path: str, device: torch.device) -> np.ndarray:
    wav = _load_audio_mono_32k(audio_path, device=device)
    with torch.no_grad():
        enc = model.audio_encoder.encode(wav.unsqueeze(0))
        codes = enc.audio_codes.long()
        bsz, c, k, t = codes.shape
        inp = codes[:, :, :, :-1].contiguous().view(bsz, c * k, t - 1)
        out = model.decoder(inp, output_hidden_states=False)
        logits = out.logits
        if logits.dim() == 4:
            logits = logits.squeeze(0)
        ent_kt = _entropy_from_logits_multi_codebook(logits)
        return ent_kt.detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--splits", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--musicgen-model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--fixed-time-steps", type=int, default=1500)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="phase7_release/outputs/features/entropy/musiceval/clean")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split_csv = get_exp11_split_paths(args.config, splits=args.splits)[args.split]
    df = pd.read_csv(split_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    curve_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(curve_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, f"entropy_manifest_{args.split}.csv")

    def _maybe_build_manifest_from_existing() -> bool:
        rows = []
        for _, row in df.iterrows():
            token_loss_path = str(row["token_loss_path"])
            audio_path = str(row["audio_path"])
            uid = _uid(token_loss_path)
            npy_path = os.path.abspath(os.path.join(curve_dir, f"{uid}_entropy.npy"))
            if not os.path.isfile(npy_path):
                return False
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim != 2:
                return False
            rows.append(
                {
                    "audio_path": audio_path,
                    "token_loss_path": token_loss_path,
                    "entropy_curve_path": npy_path,
                    "mean_entropy": float(np.nanmean(arr)),
                    "time_steps": int(arr.shape[1]),
                }
            )
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"[skip-existing] {args.split}: rebuilt missing manifest from existing curves -> {manifest_path}")
        return True

    existing_curve_count = len([p for p in os.listdir(curve_dir) if p.endswith("_entropy.npy")]) if os.path.isdir(curve_dir) else 0
    print(
        f"[paths] split={args.split} out_dir={args.out_dir} curve_dir={curve_dir} manifest={manifest_path}"
    )
    print(
        f"[progress] split={args.split} need={len(df)} have={existing_curve_count} missing={max(0, len(df)-existing_curve_count)}"
    )
    if args.skip_existing and os.path.isfile(manifest_path):
        try:
            mdf = pd.read_csv(manifest_path)
            if len(mdf) == len(df) and "entropy_curve_path" in mdf.columns:
                paths = mdf["entropy_curve_path"].dropna().astype(str).tolist()
                if len(paths) == len(df) and all(os.path.isfile(p) for p in paths):
                    print(
                        f"[skip-existing] {args.split}: found complete entropy manifest "
                        f"{manifest_path} (rows={len(mdf)})"
                    )
                    return
        except Exception:
            pass
    if args.skip_existing and _maybe_build_manifest_from_existing():
        return
    if args.skip_existing:
        print(
            f"[resume-info] {args.split}: existing curves incomplete "
            f"({existing_curve_count}/{len(df)}); continue extraction."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicgenForConditionalGeneration.from_pretrained(args.musicgen_model).to(device)
    model.eval()

    records = []
    reused = 0
    recomputed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"entropy-{args.splits}-{args.split}"):
        token_loss_path = str(row["token_loss_path"])
        audio_path = str(row["audio_path"])
        uid = _uid(token_loss_path)
        out_npy = os.path.abspath(os.path.join(curve_dir, f"{uid}_entropy.npy"))
        if args.skip_existing and os.path.isfile(out_npy):
            try:
                entropy = np.load(out_npy, mmap_mode="r")
                if entropy.ndim == 2:
                    reused += 1
                else:
                    entropy = _extract_entropy_curve(model, audio_path, device=device)  # [K, T]
                    entropy = _fix_len_kt(entropy, args.fixed_time_steps)  # [4, T]
                    np.save(out_npy, entropy)
                    recomputed += 1
            except Exception:
                entropy = _extract_entropy_curve(model, audio_path, device=device)  # [K, T]
                entropy = _fix_len_kt(entropy, args.fixed_time_steps)  # [4, T]
                np.save(out_npy, entropy)
                recomputed += 1
        else:
            entropy = _extract_entropy_curve(model, audio_path, device=device)  # [K, T]
            entropy = _fix_len_kt(entropy, args.fixed_time_steps)  # [4, T]
            np.save(out_npy, entropy)
            recomputed += 1
        records.append(
            {
                "audio_path": audio_path,
                "token_loss_path": token_loss_path,
                "entropy_curve_path": out_npy,
                "mean_entropy": float(np.nanmean(entropy)),
                "time_steps": int(entropy.shape[1]),
            }
        )

    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path} ({len(records)} rows)")
    print(f"[resume-stats] {args.split}: reused={reused}, recomputed={recomputed}")


if __name__ == "__main__":
    main()
