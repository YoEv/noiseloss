import torch
import torchaudio
import os
import argparse
import sys
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes

def setup_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(device):
    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
    model.lm = model.lm.to(dtype=torch.float32)
    model.compression_model = model.compression_model.to(dtype=torch.float32)

    compression_ratio = model.compression_model.frame_rate / model.sample_rate
    print(f"压缩比率: {compression_ratio:.4f} (每音频秒生成{model.compression_model.frame_rate}个tokens)")

    return model

def generate_audio(input_path, output_path, model, device):
    try:
        # 读取输入音频
        audio, sr = audio_read(input_path)

        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to(device)
        original_length = audio.shape[-1]

        # 创建条件属性
        condition = ConditioningAttributes(
            text={'description': ""},
            wav={'self_wav': (audio.unsqueeze(0),
                 torch.tensor([original_length], device=device),
                 [sr], [], [])}
        )

        # 生成音频
        with torch.no_grad():
            generated_audio = model.generate_with_chroma(
                descriptions=[""],
                melody_wavs=audio.unsqueeze(0),
                melody_sample_rate=sr,
                progress=True
            )

        # 保存生成的音频
        torchaudio.save(
            output_path,
            generated_audio.cpu()[0],
            model.sample_rate
        )
        
        print(f"成功生成音频: {output_path}")
        return True

    except Exception as e:
        print(f"处理 {os.path.basename(input_path)} 时出错: {str(e)}")
        return False

def process_audio_directory(input_dir, output_dir, model, device):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise ValueError(f"目录 {input_dir} 中没有WAV/MP3文件")

    print(f"\n开始处理 {len(audio_files)} 个音频文件")
    success_count = 0
    error_files = []

    for filename in tqdm(audio_files, desc="生成进度"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        success = generate_audio(input_path, output_path, model, device)
        if success:
            success_count += 1
        else:
            error_files.append(filename)

    print(f"\n成功生成 {success_count} 个音频文件")
    if error_files:
        print(f"处理失败的 {len(error_files)} 个文件:")
        for filename in error_files:
            print(f"  - {filename}")

def main():
    parser = argparse.ArgumentParser(description="使用 MusicGen-melody 模型生成音频文件")
    parser.add_argument("--input_dir", type=str, default="data_100",
                        help="输入音频文件目录")
    parser.add_argument("--output_dir", type=str, default="gen_100",
                        help="输出音频文件目录")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用CPU，即使GPU可用")

    args = parser.parse_args()

    # 设置设备
    device = setup_device(args.force_cpu)
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(device)

    # 处理音频目录
    process_audio_directory(args.input_dir, args.output_dir, model, device)

    # 清理GPU内存
    if device == "cuda":
        torch.cuda.empty_cache()
        print("已清理GPU内存")

if __name__ == "__main__":
    main()