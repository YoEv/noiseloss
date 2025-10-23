import numpy as np
import soundfile as sf
import argparse
import os
from scipy import signal
from huggingface_hub import snapshot_download
import shutil
from pathlib import Path

def generate_white_noise(length, db_level=-20, seed=42):
    """Generate white noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    white_noise = np.random.randn(length)
    white_noise = white_noise / np.std(white_noise)
    scaling_factor = 10 ** (db_level / 20)
    return white_noise * scaling_factor

def generate_pink_noise(length, db_level=-20, seed=42):
    """Generate pink noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    # 生成白噪音
    white_noise = np.random.normal(0, 1, length)
    
    # 应用粉红滤波器 (1/f)
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)
    
    # 归一化
    if np.std(pink_noise) > 0:
        pink_noise = pink_noise / np.std(pink_noise)
    
    # 应用dB增益
    scaling_factor = 10 ** (db_level / 20)
    return pink_noise * scaling_factor

def generate_brown_noise(length, db_level=-20, seed=42):
    """Generate brown noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    # 生成白噪音
    white_noise = np.random.normal(0, 1, length)
    
    # 应用布朗滤波器 (1/f^2)
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)
    
    # 归一化
    if np.std(brown_noise) > 0:
        brown_noise = brown_noise / np.std(brown_noise)
    
    # 应用dB增益
    scaling_factor = 10 ** (db_level / 20)
    return brown_noise * scaling_factor

def generate_blue_noise(length, db_level=-20, seed=42):
    """Generate blue noise with fixed random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    # 生成白噪音
    white_noise = np.random.normal(0, 1, length)
    
    # 应用蓝噪音滤波器 (f)
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)
    
    # 归一化
    if np.std(blue_noise) > 0:
        blue_noise = blue_noise / np.std(blue_noise)
    
    # 应用dB增益
    scaling_factor = 10 ** (db_level / 20)
    return blue_noise * scaling_factor

def generate_noise(noise_type, length, db_level=-20, seed=42):
    """Generate noise of specified type"""
    noise_generators = {
        'white': generate_white_noise,
        'pink': generate_pink_noise,
        'brown': generate_brown_noise,
        'blue': generate_blue_noise
    }
    
    if noise_type not in noise_generators:
        raise ValueError(f"Unsupported noise type: {noise_type}. Supported types: {list(noise_generators.keys())}")
    
    return noise_generators[noise_type](length, db_level, seed)

def replace_with_noise_at_5s(audio_data, sample_rate, token_length=5, db_level=-20, frame_rate=50, seed=42, noise_type='white'):
    """Replace the token segment starting at 5 seconds with specified noise type"""
    start_sample = int(5 * sample_rate)
    # Correct token duration calculation: each token = 1/frame_rate seconds
    single_token_duration = 1.0 / frame_rate
    duration_samples = int(token_length * single_token_duration * sample_rate)
    
    if start_sample >= len(audio_data):
        return audio_data
    
    end_sample = min(start_sample + duration_samples, len(audio_data))
    actual_duration = end_sample - start_sample
    
    if len(audio_data.shape) == 1:
        # Mono audio
        noise = generate_noise(noise_type, actual_duration, db_level, seed)
        audio_data[start_sample:end_sample] = noise
    else:
        # Stereo or multi-channel audio
        for channel in range(audio_data.shape[1]):
            noise = generate_noise(noise_type, actual_duration, db_level, seed + channel)
            audio_data[start_sample:end_sample, channel] = noise
    
    return audio_data

def process_audio_file(input_path, output_path, token_length=5, db_level=-20, frame_rate=50, seed=42, noise_type='white'):
    """Process a single audio file"""
    try:
        audio_data, sample_rate = sf.read(input_path)
        modified_audio = replace_with_noise_at_5s(audio_data, sample_rate, token_length, db_level, frame_rate, seed, noise_type)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, modified_audio, sample_rate)
        print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_directory(input_dir, output_dir, token_length=5, db_level=-20, frame_rate=50, seed=42, noise_type='white'):
    """Process all audio files in a directory"""
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                process_audio_file(input_path, output_path, token_length, db_level, frame_rate, seed, noise_type)

def download_dataset_from_hf(repo_id="Dev4IC/phase5_1", local_dir="temp_hf_data"):
    """Download dataset from Hugging Face"""
    print(f"Downloading dataset from {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset"
        )
        print(f"Dataset downloaded to {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def download_original_datasets(output_dir="dataset_ori"):
    """Download the 5 original datasets from Hugging Face to dataset_ori folder"""
    print("\n=== 开始下载原始数据集到 dataset_ori 文件夹 ===")
    
    # 定义5个原始数据集
    original_datasets = [
        "ShutterStock_32k_ori",
        "ShutterStock_44k_ori", 
        "Unconditional_ori",
        "asap_ori",
        "noise_color_ori"
    ]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    repo_id = "Dev4IC/phase5_1"
    
    for dataset_name in original_datasets:
        print(f"\n--- 下载 {dataset_name} ---")
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        
        try:
            # 下载特定的数据集子目录
            snapshot_download(
                repo_id=repo_id,
                local_dir=dataset_output_dir,
                repo_type="dataset",
                allow_patterns=[f"{dataset_name}/*"]
            )
            
            # 移动文件到正确位置（因为下载会创建嵌套目录）
            nested_dir = os.path.join(dataset_output_dir, dataset_name)
            if os.path.exists(nested_dir):
                # 移动所有文件到上级目录
                for item in os.listdir(nested_dir):
                    src = os.path.join(nested_dir, item)
                    dst = os.path.join(dataset_output_dir, item)
                    if os.path.isdir(src):
                        shutil.move(src, dst)
                    else:
                        shutil.move(src, dst)
                # 删除空的嵌套目录
                os.rmdir(nested_dir)
            
            print(f"✓ {dataset_name} 下载完成: {dataset_output_dir}")
            
        except Exception as e:
            print(f"✗ 下载 {dataset_name} 失败: {e}")
    
    print(f"\n=== 原始数据集下载完成，保存在: {output_dir} ===")

def main():
    """Main processing function"""
    print("开始服务器端批量噪音替换处理...")
    
    # 首先下载原始数据集到 dataset_ori 文件夹
    download_original_datasets()
    
    # 定义基础参数
    FRAME_RATE = 50
    SEED = 42
    
    # 定义token长度数组 (6种)
    TOKEN_LENGTHS = [5, 10, 50, 100, 150, 200]
    
    # 定义4种噪音类型
    NOISE_TYPES = ['white', 'pink', 'blue', 'brown']
    
    # 创建输出根目录
    output_root = "+Dataset/Phase5_1"
    os.makedirs(output_root, exist_ok=True)
    
    # 下载数据集
    temp_dir = "temp_hf_data"
    downloaded_path = download_dataset_from_hf(local_dir=temp_dir)
    
    if not downloaded_path:
        print("数据集下载失败，退出处理")
        return
    
    # 定义数据集配置
    datasets_config = [
        {
            "name": "asap_ori",
            "input_dir": os.path.join(temp_dir, "asap_ori"),
            "output_prefix": "asap_replace_noise",
            "db_level": -30
        },
        {
            "name": "Unconditional_ori", 
            "input_dir": os.path.join(temp_dir, "Unconditional_ori"),
            "output_prefix": "unconditional_replace_noise",
            "db_level": -20
        },
        {
            "name": "ShutterStock_32k_ori",
            "input_dir": os.path.join(temp_dir, "ShutterStock_32k_ori"), 
            "output_prefix": "shutter_replace_noise",
            "db_level": -12
        }
    ]
    
    total_combinations = 0
    
    # 处理每个数据集
    for dataset in datasets_config:
        print(f"\n=== 处理 {dataset['name']} 数据集 ===")
        input_dir = dataset['input_dir']
        
        if not os.path.exists(input_dir):
            print(f"警告: 目录 {input_dir} 不存在，跳过处理")
            continue
            
        # 处理每种噪音类型
        for noise_type in NOISE_TYPES:
            print(f"\n--- 处理噪音类型: {noise_type} ---")
            
            # 处理每种token长度
            for token_length in TOKEN_LENGTHS:
                output_dir = os.path.join(
                    output_root,
                    f"{dataset['output_prefix']}_{noise_type}_at5_tk{token_length}"
                )
                
                print(f"处理: {input_dir} -> {output_dir} ({token_length}tk, {noise_type}, {dataset['db_level']}dB)")
                
                process_directory(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    token_length=token_length,
                    db_level=dataset['db_level'],
                    frame_rate=FRAME_RATE,
                    seed=SEED,
                    noise_type=noise_type
                )
                
                print(f"完成: {output_dir}")
                total_combinations += 1
    
    # 清理临时下载目录
    if os.path.exists(temp_dir):
        print(f"\n清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)
    
    print("\n=== 所有处理完成! ===")
    print(f"总共处理的组合数: {total_combinations}")
    print("- asap_ori: 4种噪音 × 6种token长度 = 24个输出目录")
    print("- Unconditional_ori: 4种噪音 × 6种token长度 = 24个输出目录")
    print("- ShutterStock_32k_ori: 4种噪音 × 6种token长度 = 24个输出目录")
    print("总计: 72个输出目录")
    print(f"所有结果保存在: {output_root}")
    print(f"原始数据集保存在: dataset_ori")

if __name__ == "__main__":
    # 添加命令行参数支持（可选）
    parser = argparse.ArgumentParser(description="Server-side batch noise replacement processing")
    parser.add_argument("--repo_id", type=str, default="Dev4IC/phase5_1", help="Hugging Face dataset repository ID")
    parser.add_argument("--output_root", type=str, default="+Dataset/Phase5_1", help="Output root directory")
    parser.add_argument("--dataset_ori_dir", type=str, default="dataset_ori", help="Directory to save original datasets")
    
    args = parser.parse_args()
    
    # 可以通过命令行参数自定义，但默认使用main()中的设置
    main()