import torch
import torchaudio
import os
import argparse
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read, audio_write
import librosa

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

def remove_tokens_at_position(audio_path, output_path, model, processor, device, 
                             remove_start_sec=5.0, remove_length_tokens=50):
    """
    在指定时间位置删除指定长度的token，并保存删除后的音频
    
    Args:
        audio_path: 输入音频路径
        output_path: 输出音频路径
        model: MusicGen模型
        processor: 音频处理器
        device: 设备
        remove_start_sec: 开始删除的时间位置（秒）
        remove_length_tokens: 删除的token数量
    """
    try:
        # 加载音频
        if device == "cuda":
            audio, sr = audio_read(audio_path)
            audio = audio.mean(dim=0, keepdim=True)
            if sr != 32000:
                audio = torchaudio.functional.resample(audio, sr, 32000)
            audio = audio.to(device)
        else:
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
        
        # 编码为tokens
        with torch.no_grad():
            encoded = model.audio_encoder.encode(audio.unsqueeze(0))
            
            # 检查编码结果
            if encoded is None:
                print(f"错误: 音频编码失败 {audio_path}")
                return None
                
            # 根据MusicGen的实际输出格式获取tokens
            tokens = None  # 初始化tokens变量
            if hasattr(encoded, 'audio_codes'):
                tokens = encoded.audio_codes.long()  # [B, C, K, T]
                print(f"使用audio_codes属性获取tokens")
            elif hasattr(encoded, 'codes'):
                tokens = encoded.codes.long()  # 可能的替代属性名
                print(f"使用codes属性获取tokens")
            elif isinstance(encoded, torch.Tensor):
                tokens = encoded.long()  # 直接返回tensor的情况
                print(f"直接使用tensor作为tokens")
            else:
                print(f"错误: 无法从编码结果中提取tokens {audio_path}")
                print(f"编码结果类型: {type(encoded)}")
                if hasattr(encoded, '__dict__'):
                    print(f"可用属性: {list(encoded.__dict__.keys())}")
                return None

            # 新增：稳健提取 scales（可能是 Tensor/列表/字典/tuple）
            scales = None
            if hasattr(encoded, 'audio_scales'):
                scales = encoded.audio_scales
                print(f"使用audio_scales属性获取scales，类型: {type(scales)}")
            elif hasattr(encoded, 'scales'):
                scales = encoded.scales
                print(f"使用scales属性获取scales，类型: {type(scales)}")
            elif isinstance(encoded, (list, tuple)) and len(encoded) >= 2:
                maybe_scales = encoded[1]
                scales = maybe_scales
                print("从返回tuple中获取scales")
            else:
                print("警告: 未能从编码结果中找到scales，将尝试无scales解码（可能失败）")

            # 统一迁移到同一设备（支持 Tensor/list/tuple/dict）
            def _to_device(obj, device):
                if torch.is_tensor(obj):
                    return obj.to(device)
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_to_device(x, device) for x in obj)
                if isinstance(obj, dict):
                    return {k: _to_device(v, device) for k, v in obj.items()}
                return obj

            tokens = tokens.to(device)
            scales = _to_device(scales, device)

        # 检查tokens是否成功获取
        if tokens is None:
            print(f"错误: tokens为None {audio_path}")
            return None
            
        print(f"成功获取tokens，形状: {tokens.shape}")
        B, C, K, T = tokens.shape
        
        # 计算删除位置（token索引）
        # MusicGen的压缩比率约为320:1（32000Hz -> 100Hz）
        tokens_per_sec = 32000 // 320  # 约100 tokens/sec
        remove_start_token = int(remove_start_sec * tokens_per_sec)
        remove_end_token = remove_start_token + remove_length_tokens
        
        print(f"原始音频长度: {T} tokens ({T/tokens_per_sec:.2f}秒)")
        print(f"删除位置: token {remove_start_token} 到 {remove_end_token} ({remove_start_sec}秒开始，删除{remove_length_tokens}个tokens)")
        
        # 检查边界
        if remove_end_token > T:
            print(f"警告: 删除结束位置超出音频长度，调整为音频末尾")
            remove_end_token = T
            remove_length_tokens = remove_end_token - remove_start_token
        
        if remove_start_token >= T:
            print(f"错误: 删除开始位置超出音频长度")
            return None
        
        # 创建删除后的tokens
        # 前半部分 + 后半部分（跳过中间部分）
        tokens_before = tokens[:, :, :, :remove_start_token]
        tokens_after = tokens[:, :, :, remove_end_token:]
        
        # 拼接删除后的tokens
        tokens_removed = torch.cat([tokens_before, tokens_after], dim=-1)

        # 新增：对 scales 做同样的裁剪（支持 Tensor/list/tuple/dict）
        def _remove_slice(obj):
            if torch.is_tensor(obj):
                return torch.cat([obj[..., :remove_start_token], obj[..., remove_end_token:]], dim=-1)
            if isinstance(obj, (list, tuple)):
                return type(obj)(_remove_slice(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _remove_slice(v) for k, v in obj.items()}
            return None

        scales_removed = _remove_slice(scales) if scales is not None else None
        if scales is None:
            print("警告: 缺少scales，尝试不带scales解码（可能失败）")
        
        print(f"删除后音频长度: {tokens_removed.shape[-1]} tokens ({tokens_removed.shape[-1]/tokens_per_sec:.2f}秒)")
        
        # 解码为音频
        with torch.no_grad():
            # 优先带 scales 调用；如果接口不兼容则回退
            try:
                if scales_removed is not None:
                    decoded_out = model.audio_encoder.decode(tokens_removed, scales_removed)
                else:
                    decoded_out = model.audio_encoder.decode(tokens_removed)
            except TypeError:
                print("解码接口需要scales，使用占位scales=1重试")
                dummy_scales = torch.ones(
                    (tokens_removed.shape[0], tokens_removed.shape[1], 1, tokens_removed.shape[-1]),
                    dtype=torch.float32, device=tokens_removed.device
                )
                decoded_out = model.audio_encoder.decode(tokens_removed, dummy_scales)

            # 新增：统一从多种返回类型中提取音频 Tensor（特别支持 EncodecDecoderOutput.audio_values）
            print(f"解码返回类型: {type(decoded_out)}")
            wave = None
            if isinstance(decoded_out, torch.Tensor):
                wave = decoded_out
            elif hasattr(decoded_out, 'audio_values'):
                wave = decoded_out.audio_values
                print("从 EncodecDecoderOutput.audio_values 提取音频")
            elif hasattr(decoded_out, 'audio'):
                wave = decoded_out.audio
            elif hasattr(decoded_out, 'wav'):
                wave = decoded_out.wav
            elif isinstance(decoded_out, (list, tuple)) and len(decoded_out) > 0:
                first = decoded_out[0]
                if hasattr(first, 'audio_values'):
                    wave = first.audio_values
                else:
                    wave = first
            elif isinstance(decoded_out, dict):
                if 'audio_values' in decoded_out:
                    wave = decoded_out['audio_values']
                elif 'audio' in decoded_out:
                    wave = decoded_out['audio']

            if isinstance(wave, (list, tuple)) and len(wave) > 0:
                wave = wave[0]

            if wave is None or not torch.is_tensor(wave):
                attrs = dir(decoded_out) if hasattr(decoded_out, '__class__') else 'N/A'
                print(f"错误: 无法从解码结果中提取音频，返回类型: {type(decoded_out)}，可用属性: {attrs}")
                return None

            print(f"解码成功，音频tensor形状: {wave.shape}")
        
        # 保存音频
        decoded_audio = wave.squeeze(0).cpu()
        try:
            decoded_audio = decoded_audio.squeeze(0).cpu()
        except AttributeError as e:
            print(f"错误: 解码音频处理失败 {audio_path}: {e}")
            return None
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为wav文件
        audio_write(output_path.replace('.wav', ''), decoded_audio, 32000, format='wav')
        
        print(f"删除后的音频已保存到: {output_path}")
        
        # 返回删除信息
        return {
            'original_length_tokens': T,
            'removed_start_token': remove_start_token,
            'removed_end_token': remove_end_token,
            'removed_length_tokens': remove_length_tokens,
            'final_length_tokens': tokens_removed.shape[-1],
            'removed_start_sec': remove_start_sec,
            'removed_length_sec': remove_length_tokens / tokens_per_sec
        }
        
    except Exception as e:
        print(f"处理音频时出错 {audio_path}: {str(e)}")
        return None

def process_directory(input_dir, output_dir, model, processor, device, 
                     remove_start_sec=5.0, remove_length_tokens=50):
    """
    批量处理目录中的音频文件
    """
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), input_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        print(f"在 {input_dir} 中未找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    results = []
    for rel_path in audio_files:
        input_path = os.path.join(input_dir, rel_path)
        output_path = os.path.join(output_dir, rel_path)
        
        print(f"\n处理: {rel_path}")
        result = remove_tokens_at_position(
            input_path, output_path, model, processor, device,
            remove_start_sec, remove_length_tokens
        )
        
        if result:
            result['file_path'] = rel_path
            results.append(result)
    
    # 保存处理结果
    results_file = os.path.join(output_dir, 'removal_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"删除参数: 开始时间={remove_start_sec}秒, 删除长度={remove_length_tokens}tokens\n\n")
        for result in results:
            f.write(f"文件: {result['file_path']}\n")
            f.write(f"  原始长度: {result['original_length_tokens']} tokens\n")
            f.write(f"  删除位置: {result['removed_start_token']}-{result['removed_end_token']} tokens\n")
            f.write(f"  删除时间: {result['removed_start_sec']:.2f}秒开始, 删除{result['removed_length_sec']:.2f}秒\n")
            f.write(f"  最终长度: {result['final_length_tokens']} tokens\n\n")
    
    print(f"\n处理结果已保存到: {results_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="在指定位置删除音频中的token片段")
    parser.add_argument("--input_dir", type=str, default="MusicEval_Small_ori", 
                       help="输入音频目录")
    parser.add_argument("--output_dir", type=str, default="MusicEval_Small_removed_50tks", 
                       help="输出音频目录")
    parser.add_argument("--remove_start_sec", type=float, default=5.0, 
                       help="开始删除的时间位置（秒）")
    parser.add_argument("--remove_length_tokens", type=int, default=50, 
                       help="删除的token数量")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载MusicGen模型...")
    model, processor = load_model(device)
    
    # 处理音频
    if os.path.isfile(args.input_dir):
        # 单个文件
        output_path = args.output_dir if args.output_dir.endswith('.wav') else args.output_dir + '.wav'
        result = remove_tokens_at_position(
            args.input_dir, output_path, model, processor, device,
            args.remove_start_sec, args.remove_length_tokens
        )
        if result:
            print(f"\n处理完成: {result}")
    else:
        # 目录
        results = process_directory(
            args.input_dir, args.output_dir, model, processor, device,
            args.remove_start_sec, args.remove_length_tokens
        )
        print(f"\n批量处理完成，共处理 {len(results)} 个文件")

if __name__ == "__main__":
    main()