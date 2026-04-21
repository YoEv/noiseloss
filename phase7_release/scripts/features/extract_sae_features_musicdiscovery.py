import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from tqdm import tqdm


MUSICGEN_TOKEN_RATE = 50  # tokens per second at 32 kHz


def _load_audio_mono_32k(path: str) -> Tuple[torch.Tensor, float]:
    """Returns (waveform [samples], duration_seconds)."""
    wav_np, sr = sf.read(path)
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]
    else:
        wav_np = wav_np.T
    dur = wav_np.shape[-1] / sr
    wav = torch.from_numpy(wav_np).float()
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    return wav.mean(dim=0), dur  # [samples], float


def _extract_batch(
    model, sae, audio_paths: List[str], device: str, max_seq_len: int, hidden_dim: int
) -> List[np.ndarray]:
    """Extract SAE features for a batch. Returns list of (T, H) float32 arrays, T <= max_seq_len."""
    wavs, durs = [], []
    for p in audio_paths:
        try:
            w, d = _load_audio_mono_32k(p)
        except Exception:
            w = torch.zeros(32000, dtype=torch.float32)
            d = 1.0
        wavs.append(w)
        durs.append(d)

    max_samples = max(w.shape[0] for w in wavs)
    padded = torch.zeros(len(wavs), max_samples)
    for j, w in enumerate(wavs):
        padded[j, :w.shape[0]] = w
    padded = padded.to(device)

    feats = []
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(padded, names_filter=[sae.cfg.hook_name])
            sae_in = cache[sae.cfg.hook_name]   # [B, T_max, D]
            encoded = sae.encode(sae_in)          # [B, T_max, H]
        for j, dur in enumerate(durs):
            t = max(1, int(dur * MUSICGEN_TOKEN_RATE))
            feat = encoded[j, :t].detach().cpu().float().numpy()
            if max_seq_len > 0 and feat.shape[0] > max_seq_len:
                feat = feat[:max_seq_len]
            if feat.ndim != 2 or feat.shape[1] != hidden_dim:
                feat = np.zeros((1, hidden_dim), dtype=np.float32)
            feats.append(feat)
    except Exception:
        feats = [np.zeros((1, hidden_dim), dtype=np.float32) for _ in audio_paths]
    return feats


def _shard_path(shard_dir: str, idx: int) -> str:
    return os.path.join(shard_dir, f"{idx:06d}.npy")


def _partial_meta_path(out_dir: str, split: str, rank: int) -> str:
    return os.path.join(out_dir, f"_partial_meta_{split}_rank{rank}.pt")


def _write_final_meta(out_dir: str, split: str, world_size: int, out_prefix: str,
                      hidden_dim: int, n_total: int) -> None:
    """Merge per-rank partial metas into the final _meta.pt."""
    all_lengths: Dict[int, int] = {}
    for r in range(world_size):
        p = _partial_meta_path(out_dir, split, r)
        part = torch.load(p, map_location="cpu")
        all_lengths.update(part["lengths_by_idx"])

    lengths = [int(all_lengths[i]) for i in range(n_total)]
    shard_dir = os.path.join(out_dir, split)
    out_meta = os.path.join(out_dir, f"{out_prefix}_{split}_meta.pt")
    torch.save({
        "mode": "sharded",
        "shard_dir": shard_dir,
        "lengths": lengths,
        "hidden_dim": hidden_dim,
        "backend": "musicdiscovery",
    }, out_meta)
    for r in range(world_size):
        p = _partial_meta_path(out_dir, split, r)
        if os.path.isfile(p):
            os.remove(p)
    print(f"[meta] wrote {out_meta} samples={n_total}")


# ---------------------------------------------------------------------------
# Legacy monolithic path (kept for backward compatibility with existing data)
# ---------------------------------------------------------------------------

def _run_monolithic(args, audio_list: List[str], out_dir: str) -> None:
    """Original single flat-memmap extraction path."""
    from sae_components.musicgen_hooked import HookedMusicGen
    from sae_lens import SAE

    out_npy = os.path.join(out_dir, f"{args.out_prefix}_{args.split}.npy")
    out_meta = os.path.join(out_dir, f"{args.out_prefix}_{args.split}_meta.pt")

    if args.skip_existing and os.path.isfile(out_npy) and os.path.isfile(out_meta):
        try:
            meta_obj = torch.load(out_meta, map_location="cpu")
            lengths = meta_obj.get("lengths", [])
            if isinstance(lengths, list) and len(lengths) == len(audio_list):
                print(f"[skip-existing] {args.split}: complete monolithic artifacts found")
                return
        except Exception:
            pass

    model = HookedMusicGen(model_name="facebook/musicgen-small", device=args.device)
    sae = SAE.load_from_pretrained(args.checkpoint_dir, device=args.device)
    hidden_dim = int(sae.cfg.d_sae)

    def _extract_one(path):
        wav, dur = _load_audio_mono_32k(path)
        wav = wav.unsqueeze(0).to(args.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(wav, names_filter=[sae.cfg.hook_name])
            sae_in = cache[sae.cfg.hook_name]
            feat = sae.encode(sae_in).squeeze(0).detach().cpu().float().numpy()
        if args.max_seq_len > 0:
            feat = feat[:args.max_seq_len]
        return feat

    # Pre-scan durations for memmap allocation
    total_est = 0
    for p in tqdm(audio_list, desc="pre-scan", leave=False):
        try:
            info = sf.info(p)
            t = int(info.frames / info.samplerate * MUSICGEN_TOKEN_RATE)
        except Exception:
            t = 180 * MUSICGEN_TOKEN_RATE
        if args.max_seq_len > 0:
            t = min(t, args.max_seq_len)
        total_est += max(1, t)
    total_est = int(total_est * 1.1)

    arr = np.memmap(out_npy, dtype="float32", mode="w+", shape=(total_est, hidden_dim))
    lengths_all, frame_offset = [], 0
    for p in tqdm(audio_list, desc=args.split):
        try:
            feat = _extract_one(p)
            if feat.ndim != 2:
                feat = np.zeros((1, hidden_dim), dtype=np.float32)
        except Exception:
            feat = np.zeros((1, hidden_dim), dtype=np.float32)
        t = feat.shape[0]
        if frame_offset + t > total_est:
            total_est = int((frame_offset + t) * 1.05)
            arr.flush()
            del arr
            arr = np.memmap(out_npy, dtype="float32", mode="r+", shape=(total_est, hidden_dim))
        arr[frame_offset:frame_offset + t] = feat
        lengths_all.append(t)
        frame_offset += t

    arr.flush()
    del arr
    frame_total = frame_offset
    if frame_total < total_est:
        tmp = out_npy + ".trim"
        src = np.memmap(out_npy, dtype="float32", mode="r", shape=(total_est, hidden_dim))
        np.save(tmp, src[:frame_total])
        del src
        os.replace(tmp, out_npy)

    torch.save({"lengths": lengths_all, "hidden_dim": hidden_dim, "backend": "musicdiscovery"}, out_meta)
    print(f"Wrote {out_npy} shape=({frame_total}, {hidden_dim})")


# ---------------------------------------------------------------------------
# Sharded path — one .npy per audio, batched inference, multi-GPU support
# ---------------------------------------------------------------------------

def _run_sharded(args, audio_list: List[str], out_dir: str) -> None:
    from sae_components.musicgen_hooked import HookedMusicGen
    from sae_lens import SAE

    shard_dir = os.path.join(out_dir, args.split)
    os.makedirs(shard_dir, exist_ok=True)

    out_meta = os.path.join(out_dir, f"{args.out_prefix}_{args.split}_meta.pt")
    n_total = len(audio_list)

    # Skip if fully complete (rank 0 only — after merge)
    if args.rank == 0 and args.skip_existing and os.path.isfile(out_meta):
        try:
            meta_obj = torch.load(out_meta, map_location="cpu")
            if (meta_obj.get("mode") == "sharded"
                    and isinstance(meta_obj.get("lengths"), list)
                    and len(meta_obj["lengths"]) == n_total):
                print(f"[skip-existing] {args.split}: complete sharded meta found")
                return
        except Exception:
            pass

    model = HookedMusicGen(model_name="facebook/musicgen-small", device=args.device)
    sae = SAE.load_from_pretrained(args.checkpoint_dir, device=args.device)
    hidden_dim = int(sae.cfg.d_sae)

    # Length-sorted bucketing to reduce intra-batch padding waste
    # Use file size as proxy for duration (fast, no audio decode needed)
    sizes = []
    for i, p in enumerate(audio_list):
        try:
            sizes.append((os.path.getsize(p), i))
        except OSError:
            sizes.append((0, i))
    sorted_indices = [i for _, i in sorted(sizes)]

    # Interleaved rank assignment across sorted order
    my_indices = sorted_indices[args.rank::args.world_size]

    # Resume: skip shards already written
    undone = [i for i in my_indices if not os.path.isfile(_shard_path(shard_dir, i))]
    print(f"[rank {args.rank}] {args.split}: {len(my_indices)} assigned, "
          f"{len(my_indices) - len(undone)} already done, {len(undone)} to extract")

    lengths_by_idx: Dict[int, int] = {}
    # Load already-done lengths for this rank's indices
    for i in my_indices:
        sp = _shard_path(shard_dir, i)
        if os.path.isfile(sp):
            try:
                arr = np.load(sp, mmap_mode="r")
                lengths_by_idx[i] = arr.shape[0]
            except Exception:
                lengths_by_idx[i] = 1

    # Extract in batches
    batch_size = max(1, args.batch_size)
    for batch_start in tqdm(range(0, len(undone), batch_size),
                            desc=f"rank{args.rank}/{args.split}", unit="batch"):
        batch_idx = undone[batch_start:batch_start + batch_size]
        batch_paths = [audio_list[i] for i in batch_idx]
        feats = _extract_batch(model, sae, batch_paths, args.device, args.max_seq_len, hidden_dim)
        for i, feat in zip(batch_idx, feats):
            np.save(_shard_path(shard_dir, i), feat)
            lengths_by_idx[i] = feat.shape[0]

    # Write partial meta for this rank
    partial_path = _partial_meta_path(out_dir, args.split, args.rank)
    torch.save({"lengths_by_idx": lengths_by_idx, "rank": args.rank}, partial_path)
    print(f"[rank {args.rank}] wrote partial meta {partial_path}")

    # Rank 0 waits for all other ranks then merges
    if args.rank == 0 and args.world_size > 1:
        print(f"[rank 0] waiting for {args.world_size - 1} other ranks...")
        deadline = time.time() + 7200  # 2 hour timeout
        while time.time() < deadline:
            missing = [r for r in range(1, args.world_size)
                       if not os.path.isfile(_partial_meta_path(out_dir, args.split, r))]
            if not missing:
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"Timed out waiting for ranks: {missing}")

    if args.rank == 0:
        _write_final_meta(out_dir, args.split, args.world_size, args.out_prefix, hidden_dim, n_total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="clean", choices=["clean", "noisy"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--out-prefix", default="sae_features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    # Disk/throughput options
    parser.add_argument("--mode", default="sharded", choices=["sharded", "monolithic"],
                        help="sharded: one .npy per audio (default); monolithic: legacy flat memmap.")
    parser.add_argument("--max-seq-len", type=int, default=1500,
                        help="Cap tokens per file at extraction. 0=no cap. Default 1500 matches training seq_len.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Audio files per GPU forward pass (sharded mode only).")
    parser.add_argument("--rank", type=int, default=0,
                        help="GPU rank for data-parallel extraction (0-indexed).")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Total number of parallel extraction processes.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = args.project_root
    splits_label = args.splits

    # Resolve dataset CSV
    data_cfg = cfg.get("data", cfg)
    csv_path = None

    # First: Try to get path from runtime config's data.splits
    splits_cfg = data_cfg.get("splits", {})
    if splits_label in splits_cfg and args.split in splits_cfg[splits_label]:
        csv_path = splits_cfg[splits_label][args.split]
        if os.path.isfile(csv_path):
            print(f"[{args.split}] Using CSV from runtime config: {csv_path}")

    if csv_path is None:
        manifests_dir = os.path.join(project_root, data_cfg.get("manifests_dir", "phase7_release/data/manifests"))
        dataset_name = data_cfg.get("dataset_name", "musiceval")
        csv_name = f"{dataset_name}_{splits_label}_{args.split}.csv"
        candidates = [
            os.path.join(manifests_dir, csv_name),
            os.path.join(project_root, "phase7_release", "data", "full_splits", csv_name),
            os.path.join(project_root, "phase7_release", "data", "musiceval_splits", csv_name),
        ]
        for c in candidates:
            if os.path.isfile(c):
                csv_path = c
                break
        if csv_path is None:
            # Fallback: use state runtime_splits subdir (run_tag matching config key)
            runtime_splits_dir = os.path.join(project_root, "phase7_release", "outputs",
                                               "run_state", "runtime_splits")
            rt_candidate = os.path.join(runtime_splits_dir, run_tag, f"{args.split}.csv")
            if os.path.isfile(rt_candidate):
                csv_path = rt_candidate

    if csv_path is None:
        raise FileNotFoundError(f"Cannot find CSV for {splits_label} {args.split}")

    df = pd.read_csv(csv_path)
    audio_col = "audio_path"
    audio_list = [
        p if os.path.isabs(p) else os.path.join(project_root, p)
        for p in df[audio_col].tolist()
    ]
    print(f"[{args.split}] {len(audio_list)} audio files from {csv_path}")

    # Resolve output dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        sae_cfg = cfg.get("sae", {})
        out_dir = os.path.join(project_root, sae_cfg.get("output_dir", "phase7_release/outputs/full/features/sae"),
                               dataset_name, splits_label)
    os.makedirs(out_dir, exist_ok=True)

    # Add musicdiscovery to sys.path
    md_path = os.path.join(project_root, "external", "musicdiscovery")
    if md_path not in sys.path:
        sys.path.insert(0, md_path)

    if args.mode == "sharded":
        _run_sharded(args, audio_list, out_dir)
    else:
        _run_monolithic(args, audio_list, out_dir)


if __name__ == "__main__":
    main()
