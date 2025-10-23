import torch
import torchaudio
import os
import argparse
import json
import pandas as pd
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import CrossEntropyLoss
import librosa
import numpy as np

def setup_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model = model.to(device)
    model.eval()
    return model, processor

def nll_segment_teacher_forcing(model, tokens, start, L, codebooks='auto', mask_invalid=None):
    """
    计算指定片段的per-token NLL
    
    Args:
        model: MusicGen模型
        tokens: LongTensor [B, T, C] 或 [B, C, T] 取决于实现（C=codebooks）
        start: 开始位置
        L: 片段长度
        codebooks: codebook配置
        mask_invalid: 无效位置的mask
    
    Returns:
        per-token NLL (mean over codebooks, masked), scalar
    """
    # 假设tokens形状为 [B, C, K, T]（根据MusicGen的实际格式）
    if len(tokens.shape) == 4:
        B, C, K, T = tokens.shape
        # 重新排列为 [B, T, C*K] 用于计算
        tokens_reshaped = tokens.permute(0, 3, 1, 2).contiguous().view(B, T, C*K)
    else:
        B, T, C = tokens.shape
        K = 1
    
    ce = CrossEntropyLoss(reduction='none')
    losses = []
    
    # 对每个codebook单独计算CE，再平均
    for c in range(C):
        for k in range(K):
            # 取第c个码本第k层的序列 [B, T]
            if len(tokens.shape) == 4:
                seq_ck = tokens[:, c, k, :]  # [B, T]
            else:
                seq_ck = tokens[:, :, c]  # [B, T]
            
            # teacher-forcing：预测t的目标是seq_ck[:, t]
            # 构造输入和目标
            input_seq = seq_ck[:, :-1]  # [B, T-1]
            target_seq = seq_ck[:, 1:]  # [B, T-1]
            
            logits_ck = None  # 新增：初始化，避免后续未定义引用
            # 模型前向传播
            with torch.no_grad():
                # 这里需要根据实际模型API调整
                # 假设模型接受token输入并返回logits
                if len(tokens.shape) == 4:
                    # 使用完整的多codebook输入
                    input_tokens = tokens[:, :, :, :-1]  # [B, C, K, T-1]
                    input_reshaped = input_tokens.view(B, C*K, -1)  # [B, C*K, T-1]
                    outputs = model.decoder(input_reshaped)
                    logits = getattr(outputs, "logits", outputs)  # 期望3维

                    # 统一标准到 [B, time, vocab]
                    if logits.dim() != 3:
                        raise ValueError(f"Unsupported logits dim {logits.dim()} for tokens shape {tuple(tokens.shape)}")
                    ck_total = C * K
                    expected_t = T - 1
                    expected_full = expected_t * ck_total
                    b, d1, d2 = logits.shape  # 候选 [B, time, vocab] 或 [B, vocab, time]

                    def is_time(n: int) -> bool:
                        return (n == expected_t) or (n == expected_full) or (n % ck_total == 0 and (n // ck_total) == expected_t)

                    if is_time(d1):
                        pass  # 已经是 [B, time, vocab]
                    elif is_time(d2):
                        logits = logits.transpose(1, 2)  # [B, time, vocab]
                        b, d1, d2 = logits.shape
                    else:
                        raise ValueError(f"Cannot identify time dimension from logits shape {tuple(logits.shape)}; expected_t={expected_t}, ck_total={ck_total}")

                    seq_len = d1
                    vocab_sz = d2

                    # 将logits对齐到当前(c,k)码本的时间序列
                    if seq_len == expected_t:
                        logits_ck = logits  # [B, T-1, vocab]
                    elif seq_len == expected_full:
                        offset = c * K + k
                        time_index = torch.arange(expected_t, device=logits.device) * ck_total + offset  # [T-1]
                        logits_ck = logits.index_select(dim=1, index=time_index)  # [B, T-1, vocab]
                    else:
                        if seq_len % ck_total == 0 and (seq_len // ck_total) == expected_t:
                            logits_reshaped = logits.view(B, expected_t, ck_total, vocab_sz)  # [B, T-1, C*K, vocab]
                            logits_ck = logits_reshaped[:, :, c * K + k, :]  # [B, T-1, vocab]
                        else:
                            raise ValueError(f"Unsupported logits shape {tuple(logits.shape)} for tokens shape {tuple(tokens.shape)}")
                else:
                    # 简化版本：标准化为 [B, T-1, vocab]
                    outputs = model.decoder(input_seq.unsqueeze(1))
                    logits_raw = getattr(outputs, "logits", outputs)  # 可能是 [B, T-1, V] 或 [B, V, T-1] 或 [B, T-1]
                    if logits_raw.dim() == 3:
                        if logits_raw.size(1) != target_seq.size(1) and logits_raw.size(2) == target_seq.size(1):
                            logits_ck = logits_raw.transpose(1, 2)  # [B, T-1, V]
                        else:
                            logits_ck = logits_raw  # [B, T-1, V]
                    elif logits_raw.dim() == 2:
                        logits_ck = logits_raw.unsqueeze(1)  # [B, 1, V]
                    else:
                        raise ValueError(f"Unexpected logits_raw shape {tuple(logits_raw.shape)}")

            # 只评估窗口[start, start+L)内的NLL；若logits_ck无效或窗口无效则跳过
            if logits_ck is None:
                continue

            sl = start
            sr = min(start + L, target_seq.size(1))

            # 最后一里路对齐：若 logits 的时间维是 target 的整倍数（通常是 C*K 倍），按 (c,k) 偏移抽取
            if logits_ck.size(1) != target_seq.size(1):
                ck_total = C * K
                if logits_ck.size(1) % target_seq.size(1) == 0 and (logits_ck.size(1) // target_seq.size(1)) == ck_total:
                    expected_t = target_seq.size(1)
                    offset = c * K + k
                    # 在全局时间轴上用 stride=ck_total 抽取 (c,k) 子序列
                    time_index = torch.arange(expected_t, device=logits_ck.device) * ck_total + offset  # [T-1]
                    logits_ck = logits_ck.index_select(dim=1, index=time_index)  # [B, T-1, vocab]
                else:
                    raise ValueError(f"Final align failed: logits_ck.time={logits_ck.size(1)} vs target.time={target_seq.size(1)}, C*K={ck_total}")

            # 将 sr 再裁剪到 logits_ck 的时间长度，避免越界
            max_time = logits_ck.size(1)
            sr = min(sr, max_time)

            if sr <= sl:
                continue

            logits_win = logits_ck[:, sl:sr, :]  # [B, L?, vocab_size]
            target_win = target_seq[:, sl:sr]    # [B, L]

            # 新增：窗口级时间维对齐，确保 logits_win 与 target_win 的长度一致
            if logits_win.size(1) != target_win.size(1):
                lw = logits_win.size(1)
                tw = target_win.size(1)
                ck_total = C * K
                if tw > 0 and lw % tw == 0:
                    ratio = lw // tw
                    offset = (c * K + k) % max(ratio, 1)  # 当前 (c,k) 的偏移
                    idx = torch.arange(tw, device=logits_win.device) * ratio + offset
                    idx = idx.clamp_max(lw - 1)
                    logits_win = logits_win.index_select(dim=1, index=idx)  # [B, tw, vocab]
                    # target_win 保持 [B, tw]
                else:
                    # 兜底：两者裁剪到相同最小长度
                    min_len = min(lw, tw)
                    logits_win = logits_win[:, :min_len, :]
                    target_win = target_win[:, :min_len]

            V = logits_win.size(-1)
            Lw = logits_win.size(1)

            # 计算交叉熵（按照真实窗口长度 Lw）
            loss_tok = ce(logits_win.reshape(-1, V), target_win.reshape(-1))
            loss_tok = loss_tok.view(B, Lw)  # [B, Lw]
            
            if mask_invalid is not None:
                # mask_invalid: [B, T, C]，True表示有效 或 [B, C, K, T]
                if len(mask_invalid.shape) == 4:
                    m = mask_invalid[:, c, k, 1:][:, sl:sl+Lw]  # 对齐target
                else:
                    m = mask_invalid[:, 1:, c][:, sl:sl+Lw]     # 对齐target
                loss_tok = loss_tok.masked_fill(~m, 0.0)
                denom = m.float().sum().clamp_min(1.0)
                loss_c = loss_tok.sum() / denom
            else:
                loss_c = loss_tok.mean()
            
            losses.append(loss_c)
    
    # 多码本平均（或自定义加权）
    if losses:
        loss = torch.stack(losses).mean()
        return loss.item()
    else:
        return 0.0

@torch.no_grad()
def generate_hole(model, prefix_tokens, L, temperature=0.7, codebooks='auto'):
    """
    生成缺失片段的tokens
    
    Args:
        model: MusicGen模型
        prefix_tokens: [B, t0, C] 前缀tokens
        L: 生成长度
        temperature: 生成温度
        codebooks: codebook配置
    
    Returns:
        gen_tokens: [B, L, C] 生成的tokens
    """
    if len(prefix_tokens.shape) == 4:
        B, C, K, t0 = prefix_tokens.shape
        # 重新排列为适合生成的格式
        cur = prefix_tokens.clone()
    else:
        B, t0, C = prefix_tokens.shape
        K = 1
        cur = prefix_tokens.clone()
    
    gen = []
    
    for step in range(L):
        # 模型前向传播获取下一步分布
        if len(cur.shape) == 4:
            input_reshaped = cur.view(B, C*K, -1)
            outputs = model.decoder(input_reshaped)
            logits = outputs.logits  # [B, T, vocab_size]
            
            # 取最后一步的logits
            next_logits = logits[:, -1, :]  # [B, vocab_size]
        else:
            outputs = model.decoder(cur)
            next_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        
        # 采样下一个token
        if temperature == 0.0:
            next_tokens = torch.argmax(next_logits, dim=-1)  # [B]
        else:
            probs = torch.softmax(next_logits / max(1e-6, temperature), dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        
        # 添加到序列
        if len(cur.shape) == 4:
            # 需要扩展到多codebook格式
            next_tokens_expanded = next_tokens.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]
            next_tokens_expanded = next_tokens_expanded.expand(B, C, K, 1)
            cur = torch.cat([cur, next_tokens_expanded], dim=-1)
            gen.append(next_tokens_expanded.squeeze(-1))  # [B, C, K]
        else:
            next_tokens_expanded = next_tokens.unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
            next_tokens_expanded = next_tokens_expanded.expand(B, 1, C)
            cur = torch.cat([cur, next_tokens_expanded], dim=1)
            gen.append(next_tokens_expanded.squeeze(1))  # [B, C]
    
    if len(prefix_tokens.shape) == 4:
        gen_tokens = torch.stack(gen, dim=-1)  # [B, C, K, L]
    else:
        gen_tokens = torch.stack(gen, dim=1)   # [B, L, C]
    
    return gen_tokens

def eval_gap_fill(model, processor, ori_audio_path, filled_audio_path, device,
                 t0_sec=5.0, L_tokens=50, temperature=0.0, mask_invalid=None):
    """
    计算Δ = gen_loss_with_blank - ori_loss
    
    Args:
        model: MusicGen模型
        processor: 音频处理器
        ori_audio_path: 原始音频路径
        filled_audio_path: 填充后音频路径
        device: 设备
        t0_sec: 开始位置（秒）
        L_tokens: 片段长度（tokens）
        temperature: 生成温度
        mask_invalid: 无效位置mask
    
    Returns:
        包含ori_nll, gen_nll, delta的字典
    """
    try:
        # 加载并编码原始音频
        if device == "cuda":
            ori_audio, sr = audio_read(ori_audio_path)
            ori_audio = ori_audio.mean(dim=0, keepdim=True)
            if sr != 32000:
                ori_audio = torchaudio.functional.resample(ori_audio, sr, 32000)
            ori_audio = ori_audio.to(device)
        else:
            ori_audio, sr = librosa.load(ori_audio_path, sr=32000, mono=True)
            ori_audio = torch.tensor(ori_audio).unsqueeze(0).to(device)
        
        # 编码为tokens
        with torch.no_grad():
            ori_encoded = model.audio_encoder.encode(ori_audio.unsqueeze(0))
            ori_tokens = ori_encoded.audio_codes.long()  # [B, C, K, T]
        
        B, C, K, T = ori_tokens.shape
        
        # 计算token位置
        tokens_per_sec = 32000 // 320  # 约100 tokens/sec
        t0 = int(t0_sec * tokens_per_sec)
        
        assert t0 + L_tokens <= T, f"片段超出音频长度: {t0 + L_tokens} > {T}"
        
        # 1) 原片段NLL（teacher-forcing）
        nll_ori = nll_segment_teacher_forcing(model, ori_tokens, t0, L_tokens, mask_invalid=mask_invalid)
        
        # 2) 加载填充后的音频
        if device == "cuda":
            filled_audio, sr2 = audio_read(filled_audio_path)
            filled_audio = filled_audio.mean(dim=0, keepdim=True)
            if sr2 != 32000:
                filled_audio = torchaudio.functional.resample(filled_audio, sr2, 32000)
            filled_audio = filled_audio.to(device)
        else:
            filled_audio, sr2 = librosa.load(filled_audio_path, sr=32000, mono=True)
            filled_audio = torch.tensor(filled_audio).unsqueeze(0).to(device)
        
        # 编码填充后的音频
        with torch.no_grad():
            filled_encoded = model.audio_encoder.encode(filled_audio.unsqueeze(0))
            filled_tokens = filled_encoded.audio_codes.long()  # [B, C, K, T']
        
        # 3) 对填充后音频中的生成片段做teacher-forcing NLL
        nll_gen_ctx = nll_segment_teacher_forcing(model, filled_tokens, t0, L_tokens, mask_invalid=mask_invalid)
        
        delta = nll_gen_ctx - nll_ori
        
        return {
            "ori_nll": float(nll_ori),
            "gen_nll": float(nll_gen_ctx),
            "delta": float(delta),
            "ori_audio_path": ori_audio_path,
            "filled_audio_path": filled_audio_path,
            "t0_sec": t0_sec,
            "L_tokens": L_tokens
        }
        
    except Exception as e:
        print(f"评估gap filling时出错: {str(e)}")
        return None

def process_directory_comparison(ori_dir, filled_dir, output_file, model, processor, device,
                               t0_sec=5.0, L_tokens=50, temperature=0.0):
    """
    批量比较原始音频和填充音频的loss
    """
    # 找到所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(ori_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), ori_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        print(f"在 {ori_dir} 中未找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    results = []
    for rel_path in audio_files:
        ori_path = os.path.join(ori_dir, rel_path)
        filled_path = os.path.join(filled_dir, rel_path)
        
        if not os.path.exists(filled_path):
            print(f"警告: 填充音频不存在 {filled_path}")
            continue
        
        print(f"\n比较: {rel_path}")
        
        result = eval_gap_fill(
            model, processor, ori_path, filled_path, device,
            t0_sec, L_tokens, temperature
        )
        
        if result:
            result['file_path'] = rel_path
            results.append(result)
            print(f"  原始NLL: {result['ori_nll']:.6f}")
            print(f"  生成NLL: {result['gen_nll']:.6f}")
            print(f"  差异Δ: {result['delta']:.6f}")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为JSON
    with open(output_file.replace('.txt', '.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存为文本格式
    with open(output_file, 'w') as f:
        f.write(f"Gap Filling Loss Comparison Results\n")
        f.write(f"参数: t0={t0_sec}秒, L={L_tokens}tokens, temperature={temperature}\n\n")
        
        for result in results:
            f.write(f"文件: {result['file_path']}\n")
            f.write(f"  原始NLL: {result['ori_nll']:.8f}\n")
            f.write(f"  生成NLL: {result['gen_nll']:.8f}\n")
            f.write(f"  差异Δ: {result['delta']:.8f}\n\n")
        
        # 统计摘要
        if results:
            deltas = [r['delta'] for r in results]
            f.write(f"\n统计摘要:\n")
            f.write(f"  总文件数: {len(results)}\n")
            f.write(f"  平均Δ: {np.mean(deltas):.8f}\n")
            f.write(f"  标准差: {np.std(deltas):.8f}\n")
            f.write(f"  最小Δ: {np.min(deltas):.8f}\n")
            f.write(f"  最大Δ: {np.max(deltas):.8f}\n")
            f.write(f"  Δ<0的文件数: {sum(1 for d in deltas if d < 0)}\n")
            f.write(f"  Δ>0的文件数: {sum(1 for d in deltas if d > 0)}\n")
    
    print(f"\n比较结果已保存到: {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="比较原始音频和生成填充音频的loss差异")
    parser.add_argument("--ori_dir", type=str, default="MusicEval_Small_ori",
                       help="原始音频目录")
    parser.add_argument("--filled_dir", type=str, default="MusicEval_Small_filled",
                       help="填充后音频目录")
    parser.add_argument("--output_file", type=str, default="gap_fill_loss_comparison.txt",
                       help="输出结果文件")
    parser.add_argument("--t0_sec", type=float, default=5.0,
                       help="评估片段开始位置（秒）")
    parser.add_argument("--L_tokens", type=int, default=50,
                       help="评估片段长度（tokens）")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="生成温度（用于生成模式，通常设为0进行确定性评估）")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载MusicGen模型...")
    model, processor = load_model(device)
    
    # 进行比较
    results = process_directory_comparison(
        args.ori_dir, args.filled_dir, args.output_file,
        model, processor, device,
        args.t0_sec, args.L_tokens, args.temperature
    )
    
    print(f"\n批量比较完成，共处理 {len(results)} 个文件")
    
    if results:
        deltas = [r['delta'] for r in results]
        print(f"\n结果摘要:")
        print(f"  平均Δ: {np.mean(deltas):.6f}")
        print(f"  标准差: {np.std(deltas):.6f}")
        print(f"  Δ<0的文件数: {sum(1 for d in deltas if d < 0)} ({100*sum(1 for d in deltas if d < 0)/len(deltas):.1f}%)")
        print(f"  Δ>0的文件数: {sum(1 for d in deltas if d > 0)} ({100*sum(1 for d in deltas if d > 0)/len(deltas):.1f}%)")

if __name__ == "__main__":
    main()