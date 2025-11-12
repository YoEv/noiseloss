import torch
import torchaudio
import os
import argparse
import sys
import librosa 
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
# from torch.nn import functional as F
import numpy as np
import pandas as pd
import torch.nn.functional as F
import inspect
import transformers.models.musicgen.modeling_musicgen as mg

codebook_size = 2048          # 以实际 config 为准
num_codebooks = 4             # 以实际 config 为准

def setup_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model = model.to(device)
    model.eval()
    
    # 打印模型结构，确认内部字段名
    print("[model]", type(model))
    print("[model] 直接子模块:", [n for n, _ in model.named_children()])
    # 尝试查看 transformer 本体
    if hasattr(model, "model"):
        print("[model.model]", type(model.model))
        print("[model.model] 子模块:", [n for n, _ in model.model.named_children()])
    else:
        print("[model] 没有 .model 属性，继续找 transformer 层...")
    # 直接看 decoder
    dec = model.decoder
    print("[model.decoder]", type(dec))
    print("[model.decoder] 子模块:", [n for n, _ in dec.named_children()])
    # 再看 decoder.model
    dec_model = dec.model
    print("[model.decoder.model]", type(dec_model))
    print("[model.decoder.model] 子模块:", [n for n, _ in dec_model.named_children()])
    # 再往下探一层：model.decoder.model.decoder
    dec_dec = dec_model.decoder
    print("[model.decoder.model.decoder]", type(dec_dec))
    print("[model.decoder.model.decoder] 子模块:", [n for n, _ in dec_dec.named_children()])
    
    compression_ratio = model.config.audio_encoder.frame_rate / model.config.audio_encoder.sampling_rate
    print(f"压缩比率: {compression_ratio:.4f} (每音频秒生成{model.config.audio_encoder.frame_rate}个tokens)")
    # 打印一些模型配置的关键信息（如可用）
    try:
        print(f"音频采样率: {model.config.audio_encoder.sampling_rate}, token帧率: {model.config.audio_encoder.frame_rate}")
    except Exception:
        pass
    return model, processor

# 顶部：新增 hook 辅助函数和注册工具
def log_shape(name, x):
    try:
        shape = tuple(x.shape)
        dtype = getattr(x, "dtype", None)
        device = getattr(x, "device", None)
        print(f"[形状] {name}: shape={shape}, dtype={dtype}, device={device}")
    except Exception as e:
        print(f"[形状] {name}: 非Tensor或无shape ({type(x)}), err={e}")

def hook(name):
    def _hook(mod, inp, out):
        x = out if isinstance(out, torch.Tensor) else out[0]
        print(f"[{name}] {tuple(x.shape)} {x.dtype} {x.device}")
    return _hook

def register_decoder_hooks(model, hook_layers=2):
    transformer = model.decoder.model.decoder
    cfg = getattr(transformer, "config", None)
    num_codebooks = getattr(cfg, "num_codebooks", 4)
    print(f"[hook] 使用的 codebooks 数: {num_codebooks}")
    print(f"[{transformer.__class__.__name__}] 子模块: {list(transformer._modules.keys())}")

    # 关键：包装 decoder.forward，打印是否使用 input_ids 或 inputs_embeds
    orig_forward = transformer.forward
    def forward_wrapper(*args, **kwargs):
        iid = kwargs.get("input_ids", kwargs.get("decoder_input_ids", None))
        emb = kwargs.get("inputs_embeds", None)
        iid_shape = None if iid is None else tuple(iid.shape)
        emb_shape = None if emb is None else tuple(emb.shape)
        print(f"[decoder.forward] input_ids={iid_shape} inputs_embeds={emb_shape}")
        return orig_forward(*args, **kwargs)
    transformer.forward = forward_wrapper

    handles = []

    # embed_tokens：打印 (mod, inp, out) 并展示展平 (t, cb) 映射
    def embed_tok_hook(mod, inp, out):
        tokens = inp[0]  # [batch, seq_len]
        print(f"[tok_embed] mod={mod.__class__.__name__}, inp_shape={tuple(tokens.shape)}, out_shape={tuple(out.shape)}")
        transformer._last_tok_embed = out  # [batch, seq_len, embed_dim]
        seq_len = tokens.shape[1]
        K = num_codebooks
        N = min(24, seq_len)
        mapping = [f"{i}:(t={i // K},cb={i % K})" for i in range(N)]
        print("[embed] " + ", ".join(mapping))
    handles.append(transformer.embed_tokens.register_forward_hook(embed_tok_hook))

    # embed_positions：打印 (mod, inp, out)，缓存 pos_embed（如果存在）
    if hasattr(transformer, "embed_positions"):
        def embed_pos_hook(mod, inp, out):
            first_inp = inp[0] if len(inp) > 0 else None
            shape_inp = tuple(first_inp.shape) if isinstance(first_inp, torch.Tensor) else "N/A"
            print(f"[pos_embed] mod={mod.__class__.__name__}, inp_shape={shape_inp}, out_shape={tuple(out.shape)}")
            transformer._last_pos_embed = out  # 可能是 [seq_len, embed_dim] 或 [bsz, seq_len, embed_dim]
        handles.append(transformer.embed_positions.register_forward_hook(embed_pos_hook))
    else:
        print("[pos_embed] embed_positions 不存在，跳过挂载")

    # 第一层的 forward_pre_hook：打印进入第 0 层的 hidden_states（即 tok+pos 结果）
    def first_layer_pre_hook(mod, inp):
        hidden_states = inp[0]
        print(f"[layer0_in] mod={mod.__class__.__name__}, hidden_states_shape={tuple(hidden_states.shape)}")
        if hasattr(transformer, "_last_tok_embed") and hasattr(transformer, "_last_pos_embed"):
            tok = transformer._last_tok_embed
            pos = transformer._last_pos_embed
            summed = tok + (pos if pos.dim() == 3 else pos.unsqueeze(0))
            print(f"[embed_sum] tok_shape={tuple(tok.shape)}, pos_shape={tuple(pos.shape)}, sum_shape={tuple(summed.shape)}")
            print(summed[0, :2, :8])
    handles.append(transformer.layers[0].register_forward_pre_hook(first_layer_pre_hook))

    # self_attn：真实掩码 + 并行延迟可视化
    for layer_idx in range(min(hook_layers, len(transformer.layers))):
        layer = transformer.layers[layer_idx]
        def make_attn_hook(idx):
            def attn_hook(module, input, output):
                mask2d = getattr(module, "_last_mask", None)
                shape4d = getattr(module, "_last_mask_4d_shape", None)
                if shape4d is not None:
                    print(f"[Layer{idx}] attention_mask(4D) shape={shape4d} -> (bsz, heads, tgt_len, src_len)")
                if isinstance(mask2d, torch.Tensor):
                    print(f"[Layer{idx}] attn_mask(2D view) shape={tuple(mask2d.shape)} -> (tgt_len, src_len)")
                    print(mask2d[:10, :10])
                    K = num_codebooks
                    seq_len = mask2d.shape[0]
                    demo_steps = min(3, (seq_len // K))
                    for t in range(demo_steps):
                        base = t * K
                        for cb in range(K):
                            row = base + cb
                            if row >= seq_len:
                                break
                            same_time_allowed = list(range(base, base + cb + 1))
                            print(f"[Layer{idx}] row={row} (t={t}, cb={cb}) 可关注同一时刻的列: {same_time_allowed}")
                return
            return attn_hook
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(layer_idx)))

    return handles
# 注意力底层日志补丁：打印真实 attn_mask / is_causal（同时拦截 F 与 transformers.musicgen 路径）
_sdpa_orig = None
_mha_orig = None
_mg_sdpa_orig = None
_mg_mha_orig = None

def sdpa_with_logging(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    try:
        print(f"[SDPA] q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)}")
    except Exception:
        pass
    if isinstance(attn_mask, torch.Tensor):
        try:
            print(f"[SDPA] attn_mask shape={tuple(attn_mask.shape)} dtype={attn_mask.dtype} device={attn_mask.device}")
            sample = attn_mask[0, 0, :10, :10] if attn_mask.dim() == 4 else attn_mask[:10, :10]
            print(sample)
        except Exception as e:
            print(f"[SDPA] 打印attn_mask失败: {e}")
    else:
        print(f"[SDPA] attn_mask=None is_causal={is_causal}")
        if is_causal:
            T = query.shape[-2]
            ref = torch.triu(torch.full((T, T), float('-inf'), device=query.device), diagonal=1)
            print("[SDPA] 参考因果mask[:10,:10]:")
            print(ref[:10, :10])
    return _sdpa_orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

def mha_forward_with_logging(*args, **kwargs):
    attn_mask = kwargs.get("attn_mask", None)
    is_causal = kwargs.get("is_causal", None)
    # 某些版本以位置参数传入，做兜底
    if attn_mask is None and len(args) >= 17:
        attn_mask = args[16]
    if is_causal is None and len(args) >= 25:
        is_causal = args[24]
    if isinstance(attn_mask, torch.Tensor):
        try:
            print(f"[MHA] attn_mask shape={tuple(attn_mask.shape)} dtype={attn_mask.dtype} device={attn_mask.device}")
            sample = attn_mask[0, 0, :10, :10] if attn_mask.dim() == 4 else attn_mask[:10, :10]
            print(sample)
        except Exception as e:
            print(f"[MHA] 打印attn_mask失败: {e}")
    else:
        print(f"[MHA] attn_mask=None is_causal={is_causal}")
    return _mha_orig(*args, **kwargs)

def enable_attention_logging():
    global _sdpa_orig, _mha_orig, _mg_sdpa_orig, _mg_mha_orig
    # 1) 拦截 torch.nn.functional 路径
    if _sdpa_orig is None and hasattr(F, "scaled_dot_product_attention"):
        _sdpa_orig = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = sdpa_with_logging
    if _mha_orig is None and hasattr(F, "multi_head_attention_forward"):
        _mha_orig = F.multi_head_attention_forward
        F.multi_head_attention_forward = mha_forward_with_logging
    # 2) 拦截 transformers.musicgen 内的可能别名
    if _mg_sdpa_orig is None and hasattr(mg, "scaled_dot_product_attention"):
        _mg_sdpa_orig = mg.scaled_dot_product_attention
        mg.scaled_dot_product_attention = sdpa_with_logging
    if _mg_mha_orig is None and hasattr(mg, "multi_head_attention_forward"):
        _mg_mha_orig = mg.multi_head_attention_forward
        mg.multi_head_attention_forward = mha_forward_with_logging

def disable_attention_logging():
    global _sdpa_orig, _mha_orig, _mg_sdpa_orig, _mg_mha_orig
    if _sdpa_orig is not None:
        F.scaled_dot_product_attention = _sdpa_orig
        _sdpa_orig = None
    if _mha_orig is not None:
        F.multi_head_attention_forward = _mha_orig
        _mha_orig = None
    if _mg_sdpa_orig is not None:
        mg.scaled_dot_product_attention = _mg_sdpa_orig
        _mg_sdpa_orig = None
    if _mg_mha_orig is not None:
        mg.multi_head_attention_forward = _mg_mha_orig
        _mg_mha_orig = None

def force_math_sdpa():
    # 强制禁用 flash/mem-efficient，走 math 路径，便于日志捕获
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SDP] 已切换到 math 实现（禁用 flash/mem-efficient）")
    except Exception as e:
        print("[SDP] 切换SDP内核失败:", e)

# 额外：包装 self_attn.forward，尝试打印传入的 attn_mask / attention_mask / key_padding_mask / is_causal
class ForwardWrapperHandle:
    def __init__(self, module, orig_forward):
        self.module = module
        self.orig_forward = orig_forward
    def remove(self):
        self.module.forward = self.orig_forward

def wrap_attn_forward_with_logging(module, tag):
    orig = module.forward

    def wrapper(*args, **kwargs):
        # 直接读取显式传入的掩码
        mask_val = kwargs.get("attention_mask", None)
        if mask_val is None:
            mask_val = kwargs.get("attn_mask", None)

        # 缓存 4D 形状与 2D 视图（不做异常捕获/兜底）
        if isinstance(mask_val, torch.Tensor):
            if mask_val.dim() == 4:
                module._last_mask_4d_shape = tuple(mask_val.shape)  # (bsz, heads, tgt_len, src_len)
                module._last_mask = mask_val[0, 0]                  # 2D (tgt_len, src_len)
            elif mask_val.dim() == 2:
                module._last_mask_4d_shape = None
                module._last_mask = mask_val
            else:
                module._last_mask_4d_shape = None
                module._last_mask = None
        else:
            module._last_mask_4d_shape = None
            module._last_mask = None

        module._last_is_causal = kwargs.get("is_causal", None)
        return orig(*args, **kwargs)

    module.forward = wrapper
    return ForwardWrapperHandle(module, orig)

def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reduction="mean"
    ):
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    print(f"[CE入口] logits={tuple(logits.shape)}, targets={tuple(targets.shape)}, mask={tuple(mask.shape)}, B={B}, K={K}, T={T}, card={logits.size(-1)}")
    
    if reduction == "none":
        # 返回每个codebook每个时间步的loss，形状为(B, K, T)
        ce_per_token = torch.zeros_like(targets, dtype=torch.float32)
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            # 仅打印第一个codebook的局部形状，避免日志爆炸
            if k == 0:
                print(f"[CE(k=0)] logits_k={tuple(logits_k.shape)}, targets_k={tuple(targets_k.shape)}, mask_k={tuple(mask_k.shape)}")
            ce_flat = F.cross_entropy(logits_k, targets_k, reduction="none")
            ce_masked = ce_flat * mask_k.float()
            ce_per_token[:, k, :] = ce_masked.view(B, T)
        print(f"[CE结果] ce_per_token={tuple(ce_per_token.shape)} (B,K,T)")
        return ce_per_token
    else:
        # 原有的平均计算逻辑
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            if k == 0:
                print(f"[CE(k=0)] logits_k={tuple(logits_k.shape)}, targets_k={tuple(targets_k.shape)}, mask_k={tuple(mask_k.shape)}")
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        ce = ce / K
        print(f"[CE均值] ce={float(ce.item())}, codebooks={K}")
        return ce, ce_per_codebook

def save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path=None):
    """
    将per-token loss保存为CSV文件，只保存平均值
    """
    if relative_path:
        # 使用相对路径信息，将路径分隔符替换为下划线避免文件系统问题
        rel_dir = os.path.dirname(relative_path)
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        if rel_dir:
            # 将路径分隔符替换为下划线，创建唯一的文件名
            filename = f"{rel_dir.replace(os.sep, '_')}_{base_filename}"
        else:
            filename = base_filename
    else:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    avg_csv_file = os.path.join(token_output_dir, f"{filename}_tokens_avg.csv")
    os.makedirs(token_output_dir, exist_ok=True)
    
    # 转换为numpy数组
    ce_np = ce_per_token.cpu().numpy()
    
    # 计算每个token位置的平均值
    avg_records = []
    
    if ce_np.ndim == 3:  # Shape: (B, K, T)
        B, K, T = ce_np.shape
        for b in range(B):
            for t in range(T):
                # 计算该token位置上所有codebook的平均值
                token_avg = float(ce_np[b, :, t].mean())
                avg_records.append({
                    'token_position': t,
                    'avg_loss_value': token_avg
                })
    elif ce_np.ndim == 2:  # Shape: (K, T)
        K, T = ce_np.shape
        for t in range(T):
            # 计算该token位置上所有codebook的平均值
            token_avg = float(ce_np[:, t].mean())
            avg_records.append({
                'token_position': t,
                'avg_loss_value': token_avg
            })
    
    # 保存平均值数据
    if avg_records:
        avg_df = pd.DataFrame(avg_records)
        avg_df.to_csv(avg_csv_file, index=False)
        print(f"保存了 {len(avg_records)} 条平均值记录到 {avg_csv_file}")
        
        # 计算并返回总体平均loss
        overall_mean = avg_df['avg_loss_value'].mean()
        print(f"总体平均loss: {overall_mean:.6f}")
        
        return overall_mean
    else:
        print(f"没有数据可保存: {filename}")
        return None

def process_single_audio(audio_path, model, processor, device, reduction="mean", token_output_dir=None, relative_path=None, hook_layers=2):
    if device == "cuda":
        audio, sr = audio_read(audio_path)
        audio = audio.mean(dim=0, keepdim=True)
        if sr != 32000:
            audio = torchaudio.functional.resample(audio, sr, 32000)
        audio = audio.to(device)
        valid_token_length = min(audio.shape[-1] // 320, 1500)
    else:
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        audio = torch.tensor(audio).unsqueeze(0).to(device)
        valid_token_length = min(audio.shape[-1] // 320, 1500)

    print(f"[音频] sr={sr}, valid_token_length={valid_token_length}")
    log_shape("audio(1,C=T)", audio)

    with torch.no_grad():
        encoded = model.audio_encoder.encode(audio.unsqueeze(0)) 
        # HuggingFace MusicGen 的 audio_encoder 返回包含 audio_codes 的对象
        tokens = encoded.audio_codes.long()
        print(f"[编码] encoded类型={type(encoded).__name__}")

    # Encodec的HF实现通常返回 (B, C, K, T)
    log_shape("tokens(B,C,K,T)", tokens)
    B, C, K, T = tokens.shape
    print(f"[tokens维度] B={B}, C={C}, K={K}, T={T}")

    input_tokens = tokens[:, :, :, :-1].contiguous()
    target_tokens = tokens[:, :, :, 1:].contiguous()
    log_shape("input_tokens(B,C,K,T-1)", input_tokens)
    log_shape("target_tokens(B,C,K,T-1)", target_tokens)

    mask = torch.zeros_like(target_tokens, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            mask[b, c, :, :valid_token_length] = True
    log_shape("mask(B,C,K,T-1)", mask)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', enabled=False):
            input_tokens_reshaped = input_tokens.view(B, C * K, -1)
            log_shape("decoder_input(B, C*K, T-1)", input_tokens_reshaped)
            model.decoder = model.decoder.to(dtype=torch.float32)

            # 注册 forward hooks（统一使用 hook_layers）
            handles = register_decoder_hooks(model, hook_layers=hook_layers)

            # 包装 self_attn.forward，尝试直接打印传入的 mask 参数
            transformer = model.decoder.model.decoder
            for layer_idx in range(min(hook_layers, len(transformer.layers))):
                attn = transformer.layers[layer_idx].self_attn
                handles.append(wrap_attn_forward_with_logging(attn, tag=f"Layer{layer_idx}.self_attn"))

            # 强制使用 math SDPA，并启用底层注意力日志补丁（捕获真实 attn_mask / is_causal）
            force_math_sdpa()
            enable_attention_logging()
            try:
                outputs = model.decoder(input_tokens_reshaped)
                output_logits = outputs.logits
            finally:
                # 关闭补丁并清理 hooks
                disable_attention_logging()
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass

            log_shape("output_logits(B, C*K, T-1, card)", output_logits)

        # 还原到 (B, C, K, T-1, card)
        logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
        log_shape("logits(B,C,K,T-1,card)", logits)

        if reduction == "none":
            ce_per_token = _compute_cross_entropy(
                logits.view(B * C, K, -1, output_logits.size(-1)), 
                target_tokens.view(B * C, K, -1), 
                mask.view(B * C, K, -1),
                reduction=reduction)
            # ce_per_token 形状 (B*C, K, T-1)
            log_shape("ce_per_token(B*C,K,T-1)", ce_per_token)
            
            if token_output_dir:
                avg_loss = save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path)
                print(f"[平均loss] {avg_loss}")
                return avg_loss if avg_loss is not None else ce_per_token.mean().item()
            else:
                return ce_per_token.mean().item()
        else:
            ce, ce_per_codebook = _compute_cross_entropy(
                logits.view(B * C, K, -1, output_logits.size(-1)), 
                target_tokens.view(B * C, K, -1), 
                mask.view(B * C, K, -1),
                reduction=reduction)
            print(f"[loss(mean)] ce={float(ce)}, codebooks={K}")
            return ce.item()


def process_audio_directory(audio_dir, model, processor, device, output_file=None, reduction="mean", token_output_dir=None):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), audio_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in {audio_dir}")

    print(f"Processing {len(audio_files)} audio files")
    results = []
    error_files = [] 

    out_file = None
    if output_file:
        out_file = open(output_file, 'w')
        print(f"Results will be saved to: {output_file}")

    for rel_filename in tqdm(audio_files, desc="Processing"):
        audio_path = os.path.join(audio_dir, rel_filename)
        loss = process_single_audio(audio_path, model, processor, device, reduction, token_output_dir, rel_filename)
        if loss is not None:
            results.append((rel_filename, loss))
            if out_file:
                out_file.write(f"{rel_filename}: {loss:.8f}\n")
                out_file.flush()
        else:
            error_files.append(rel_filename)

    if out_file:
        out_file.close()

    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Failed files count: {len(error_files)}\n\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n{len(error_files)} failed files saved to: {error_file_path}")

    return results, error_files

def main():
    parser = argparse.ArgumentParser(description="使用musicgen-small模型计算音频文件的loss值")
    parser.add_argument("--audio_dir", type=str, default="pitch_out_wav", help="音频文件目录")
    parser.add_argument("--audio_path", type=str, default=None, help="单个音频文件路径（优先于audio_dir）")
    parser.add_argument("--output_file", type=str, default="results_small.txt", help="输出结果文件路径")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU处理所有文件")
    parser.add_argument("--reduction", type=str, choices=["mean", "none"], default="mean", help="Loss reduction类型: mean或none (保存per-token losses)")
    parser.add_argument("--token_output_dir", type=str, default=None, help="当reduction=none时保存per-token losses的目录")
    parser.add_argument("--hook_layers", type=int, default=2, help="注册的Transformer层数（默认2层，避免日志过多）")
    
    args = parser.parse_args()

    if args.reduction == "none" and not args.token_output_dir:
        args.token_output_dir = "experiments/phase6/tokenizer-transformer"
        print(f"使用默认token输出目录: {args.token_output_dir}")

    original_stdout = None
    if args.output_file:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")

    model, processor = load_model(device)

    # 单文件模式：优先执行
    if args.audio_path:
        assert os.path.isfile(args.audio_path), f"文件不存在: {args.audio_path}"
        print(f"处理单个音频: {args.audio_path}")
        loss = process_single_audio(
            args.audio_path, model, processor, device, args.reduction, args.token_output_dir, relative_path=None, hook_layers=args.hook_layers
        )
        if loss is None:
            print(f"\n处理结果: {os.path.basename(args.audio_path)}: 失败（loss=None）")
        else:
            print(f"\n处理结果: {os.path.basename(args.audio_path)}: {loss:.4f}")
        if 'device' in locals() and device == 'cuda':
            torch.cuda.empty_cache()
        print("GPU内存已清理")
        if original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        return

    assert os.path.isdir(args.audio_dir), f"目录不存在: {args.audio_dir}"
    results, error_files = process_audio_directory(
        args.audio_dir, model, processor, device, args.output_file, args.reduction, args.token_output_dir
    )

    if not args.output_file:
        print("\n处理结果:")
        for filename, loss in results:
            print(f"{filename}: {loss:.4f}")

    if 'device' in locals() and device == 'cuda':
        torch.cuda.empty_cache()
    print("GPU内存已清理")

    if original_stdout:
        sys.stdout.close()
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()
