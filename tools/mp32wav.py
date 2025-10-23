#!/usr/bin/env python3
"""
MP3 to WAV converter
将MP3文件批量转换为WAV格式，支持44.1kHz和32kHz采样率
"""

import os
import sys
import argparse
from pathlib import Path
try:
    from pydub import AudioSegment
except ImportError:
    print("错误：需要安装pydub库")
    print("请运行：pip install pydub")
    sys.exit(1)

def convert_mp3_to_wav(input_file, output_file, sample_rate=44100):
    """
    将MP3文件转换为WAV格式
    
    Args:
        input_file (str): 输入MP3文件路径
        output_file (str): 输出WAV文件路径
        sample_rate (int): 采样率，默认44100Hz
    """
    try:
        # 加载MP3文件
        audio = AudioSegment.from_mp3(input_file)
        
        # 设置采样率
        audio = audio.set_frame_rate(sample_rate)
        
        # 导出为WAV格式
        audio.export(output_file, format="wav")
        print(f"✓ 转换完成: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        return True
        
    except Exception as e:
        print(f"✗ 转换失败: {os.path.basename(input_file)} - {str(e)}")
        return False

def batch_convert(input_dir, output_dir_44k, output_dir_32k):
    """
    批量转换目录下的所有MP3文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir_44k (str): 44.1kHz输出目录
        output_dir_32k (str): 32kHz输出目录
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误：输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    Path(output_dir_44k).mkdir(parents=True, exist_ok=True)
    Path(output_dir_32k).mkdir(parents=True, exist_ok=True)
    
    # 查找所有MP3文件
    mp3_files = list(input_path.glob("*.mp3"))
    
    if not mp3_files:
        print(f"在目录 {input_dir} 中未找到MP3文件")
        return
    
    print(f"找到 {len(mp3_files)} 个MP3文件")
    print("开始转换...")
    
    success_count = 0
    total_count = len(mp3_files)
    
    for mp3_file in mp3_files:
        base_name = mp3_file.stem
        
        # 44.1kHz转换
        output_44k = Path(output_dir_44k) / f"{base_name}.wav"
        success_44k = convert_mp3_to_wav(str(mp3_file), str(output_44k), 44100)
        
        # 32kHz转换
        output_32k = Path(output_dir_32k) / f"{base_name}.wav"
        success_32k = convert_mp3_to_wav(str(mp3_file), str(output_32k), 32000)
        
        if success_44k and success_32k:
            success_count += 1
    
    print(f"\n转换完成！成功转换 {success_count}/{total_count} 个文件")
    print(f"44.1kHz WAV文件保存在: {output_dir_44k}")
    print(f"32kHz WAV文件保存在: {output_dir_32k}")

def main():
    parser = argparse.ArgumentParser(description="MP3转WAV批量转换工具")
    parser.add_argument("input_dir", help="输入MP3文件目录")
    parser.add_argument("--output_44k", default="output_44k", help="44.1kHz输出目录（默认：output_44k）")
    parser.add_argument("--output_32k", default="output_32k", help="32kHz输出目录（默认：output_32k）")
    
    args = parser.parse_args()
    
    print("MP3转WAV转换工具")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    print(f"44.1kHz输出目录: {args.output_44k}")
    print(f"32kHz输出目录: {args.output_32k}")
    print("=" * 50)
    
    batch_convert(args.input_dir, args.output_44k, args.output_32k)

if __name__ == "__main__":
    main()