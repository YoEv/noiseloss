import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from tqdm import tqdm


def _load_audio_mono_32k(path: str) -> torch.Tensor:
    wav_np, sr = sf.read(path)
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]
    else:
        wav_np = wav_np.T
    wav = torch.from_numpy(wav_np).float()
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    wav = wav.mean(dim=0, keepdim=True)
    return wav


def _extract_one(model, sae, audio_path: str, device: str) -> np.ndarray:
    wav = _load_audio_mono_32k(audio_path).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(wav, names_filter=[sae.cfg.hook_name])
        sae_in = cache[sae.cfg.hook_name]  # expected [B, T, D]
        feat = sae.encode(sae_in)          # expected [B, T, H]
        feat = feat.squeeze(0).detach().cpu().float().numpy()
    return feat


def _save_chunk(chunk_path: str, chunk_meta_path: str, features: List[np.ndarray], lengths: List[int], hidden_dim: int):
    if not features:
        return
    arr = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    # Save chunk features as torch tensor for compatibility with existing merge pattern.
    torch.save({"features": torch.from_numpy(arr), "lengths": lengths}, chunk_path)
    torch.save({"lengths": lengths, "hidden_dim": hidden_dim}, chunk_meta_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="clean", choices=["clean", "noisy"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-dir", required=True, help="musicdiscovery SAE checkpoint dir containing cfg.json + weights")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--out-prefix", default="sae_features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    musicdiscovery_root = os.path.join(project_root, "external", "musicdiscovery")
    if not os.path.isdir(musicdiscovery_root):
        raise FileNotFoundError(f"musicdiscovery not found: {musicdiscovery_root}")
    sys.path.insert(0, musicdiscovery_root)

    from sae_components.musicgen_hooked import HookedMusicGen
    from sae_lens import SAE

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    split_map = cfg.get("data", {}).get("splits", {}).get(args.splits, {})
    if args.split not in split_map:
        raise ValueError(f"Missing split path for {args.split} in config data.splits.clean")
    split_csv = split_map[args.split]
    if not os.path.isabs(split_csv):
        split_csv = os.path.join(project_root, split_csv)
    if not os.path.isfile(split_csv):
        raise FileNotFoundError(f"split csv not found: {split_csv}")
    df = pd.read_csv(split_csv)
    if "audio_path" not in df.columns:
        raise ValueError(f"{split_csv} must contain audio_path")
    audio_list = df["audio_path"].dropna().astype(str).tolist()

    out_dir = args.output_dir or cfg.get("sae", {}).get("output_dir", "phase7_release/features/sae")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(project_root, out_dir)
    tmp_dir = os.path.join(out_dir, "temp_chunks_musicdiscovery")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    out_npy = os.path.join(out_dir, f"{args.out_prefix}_{args.split}.npy")
    out_meta = os.path.join(out_dir, f"{args.out_prefix}_{args.split}_meta.pt")

    # Resume-friendly: if split artifact already complete, skip expensive extraction.
    if args.skip_existing and os.path.isfile(out_npy) and os.path.isfile(out_meta):
        try:
            meta_obj = torch.load(out_meta, map_location="cpu")
            lengths = meta_obj.get("lengths", [])
            if isinstance(lengths, list) and len(lengths) == len(audio_list):
                print(
                    f"[skip-existing] {args.split}: found complete artifacts "
                    f"{out_npy} and {out_meta} (samples={len(lengths)})"
                )
                return
        except Exception:
            pass

    model = HookedMusicGen(model_name="facebook/musicgen-small", device=args.device)
    sae = SAE.load_from_pretrained(args.checkpoint_dir, device=args.device)
    hidden_dim = int(sae.cfg.d_sae)

    chunk_paths = []
    chunk_meta_paths = []
    lengths_all = []
    frame_total = 0

    n = len(audio_list)
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        s = ci * args.chunk_size
        e = min((ci + 1) * args.chunk_size, n)
        batch_audio = audio_list[s:e]
        chunk_feats = []
        chunk_lengths = []
        for p in tqdm(batch_audio, desc=f"{args.split} chunk {ci+1}/{n_chunks}", leave=False):
            try:
                feat = _extract_one(model, sae, p, args.device)
                if feat.ndim != 2:
                    feat = np.zeros((1, hidden_dim), dtype=np.float32)
            except Exception:
                feat = np.zeros((1, hidden_dim), dtype=np.float32)
            chunk_feats.append(feat)
            chunk_lengths.append(int(feat.shape[0]))

        cpath = os.path.join(tmp_dir, f"{args.out_prefix}_{args.split}_chunk_{ci}.pt")
        mpath = os.path.join(tmp_dir, f"{args.out_prefix}_{args.split}_chunk_{ci}_meta.pt")
        _save_chunk(cpath, mpath, chunk_feats, chunk_lengths, hidden_dim)
        chunk_paths.append(cpath)
        chunk_meta_paths.append(mpath)
        lengths_all.extend(chunk_lengths)
        frame_total += int(sum(chunk_lengths))

    arr = np.memmap(out_npy, dtype="float32", mode="w+", shape=(frame_total, hidden_dim))
    offset = 0
    for cpath in chunk_paths:
        obj = torch.load(cpath, map_location="cpu")
        feat = obj["features"].numpy()
        arr[offset : offset + feat.shape[0]] = feat
        offset += feat.shape[0]
    arr.flush()
    del arr

    torch.save({"lengths": lengths_all, "hidden_dim": hidden_dim, "backend": "musicdiscovery"}, out_meta)
    print(f"Wrote {out_npy} shape=({frame_total}, {hidden_dim})")
    print(f"Wrote {out_meta} samples={len(lengths_all)}")


if __name__ == "__main__":
    main()
