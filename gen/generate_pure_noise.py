import os
import numpy as np
import soundfile as sf
from scipy import signal

def generate_white_noise(length, db_level=0):
    """
    生成白噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    noise = np.random.normal(0, 0.1, length)
    # 归一化到0dB
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    noise = noise * gain

    return noise

def generate_pink_noise(length, db_level=0):
    """
    生成粉红噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    # 生成白噪音
    white_noise = np.random.normal(0, 0.1, length)

    # 应用粉红滤波器
    # 粉红噪音的功率谱密度与频率成反比 (1/f)
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)

    # 归一化到0dB
    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    pink_noise = pink_noise * gain

    return pink_noise

def generate_brown_noise(length, db_level=0):
    """
    生成布朗噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    # 生成白噪音
    white_noise = np.random.normal(0, 0.1, length)

    # 应用布朗滤波器
    # 布朗噪音的功率谱密度与频率成反比的平方 (1/f^2)
    b, a = signal.butter(2, 0.1, btype='lowpass')
    brown_noise = signal.lfilter(b, a, white_noise)

    # 归一化到0dB
    if np.max(np.abs(brown_noise)) > 0:
        brown_noise = brown_noise / np.max(np.abs(brown_noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    brown_noise = brown_noise * gain

    return brown_noise

def generate_blue_noise(length, db_level=0):
    """
    生成蓝噪音

    参数:
    - length: 噪音长度
    - db_level: 噪音响度级别(dB)
    """
    # 生成白噪音
    white_noise = np.random.normal(0, 0.1, length)

    # 应用蓝噪音滤波器
    # 蓝噪音的功率谱密度与频率成正比 (f)
    b, a = signal.butter(1, 0.5, btype='highpass')
    blue_noise = signal.lfilter(b, a, white_noise)

    # 归一化到0dB
    if np.max(np.abs(blue_noise)) > 0:
        blue_noise = blue_noise / np.max(np.abs(blue_noise))

    # 应用dB增益
    gain = 10 ** (db_level / 20)  # 将dB转换为线性增益
    blue_noise = blue_noise * gain

    return blue_noise

def generate_pure_noise_files(duration=14, sample_rate=32000, db_level=6, output_dir="pure_noise_6db"):
    """
    生成四种类型的纯噪音音频文件
    
    参数:
    - duration: 音频长度(秒)
    - sample_rate: 采样率
    - db_level: 噪音响度级别(dB)
    - output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算样本数量
    num_samples = int(duration * sample_rate)
    
    # 定义噪音类型和对应的生成函数
    noise_types = {
        'white': generate_white_noise,
        'pink': generate_pink_noise,
        'brown': generate_brown_noise,
        'blue': generate_blue_noise
    }
    
    print(f"开始生成 {duration} 秒长度、{db_level}dB 响度的纯噪音文件...")
    print(f"采样率: {sample_rate} Hz")
    print(f"输出目录: {output_dir}")
    
    # 生成每种类型的噪音
    for noise_name, noise_func in noise_types.items():
        print(f"正在生成 {noise_name} 噪音...")
        
        # 生成噪音
        noise_data = noise_func(num_samples, db_level)
        
        # 确保音频不会削波
        max_val = np.max(np.abs(noise_data))
        if max_val > 0.95:
            noise_data = noise_data / max_val * 0.95
        
        # 生成文件名
        filename = f"{noise_name}_noise_{duration}s_{db_level}db.wav"
        filepath = os.path.join(output_dir, filename)
        
        # 保存音频文件
        sf.write(filepath, noise_data, sample_rate)
        
        print(f"  已保存: {filepath}")
        print(f"  文件大小: {len(noise_data)} 样本")
        print(f"  最大幅度: {np.max(np.abs(noise_data)):.4f}")
        print()
    
    print("所有噪音文件生成完成！")
    print(f"\n生成的文件列表:")
    for noise_name in noise_types.keys():
        filename = f"{noise_name}_noise_{duration}s_{db_level}db.wav"
        print(f"  - {filename}")

def main():
    """
    主函数：生成14秒、6dB的四种纯噪音文件
    """
    # 检查必要的库是否已安装
    try:
        import soundfile
        from scipy import signal
        print("所有必要的库已安装")
    except ImportError as e:
        print(f"错误: 未安装必要的库! {str(e)}")
        print("请使用以下命令安装所需库:")
        print("pip install numpy soundfile scipy")
        return
    
    # 生成纯噪音文件
    generate_pure_noise_files(
        duration=14,        # 14秒
        sample_rate=32000,  # 32kHz采样率
        db_level=6,         # 6dB响度
        output_dir="pure_noise_6db_14s"  # 输出目录
    )

if __name__ == "__main__":
    main()