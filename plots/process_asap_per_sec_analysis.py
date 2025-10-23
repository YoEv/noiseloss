#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合脚本：处理ASAP数据集的per-second噪声分析
包含三个主要步骤：
1. 计算per-token losses
2. 生成loss差异图片
3. 合并图片
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

class ASAPPerSecAnalyzer:
    def __init__(self, base_dir="/home/evev/asap-dataset"):
        self.base_dir = Path(base_dir)
        self.token_lengths = [5, 10, 20]
        self.seconds = list(range(1, 15))  # 1-14秒
        # 移除旧的noise_end_mapping，改为动态计算
        # self.noise_end_mapping = {5: 254, 10: 259, 20: 269}
        
    def step1_calculate_losses(self, force_cpu=False):
        """
        步骤1：计算42个文件夹的per-token losses
        """
        print("=== 步骤1：计算per-token losses ===")
        
        # 生成所有需要处理的文件夹名称
        folders_to_process = []
        for sec in self.seconds:
            for tk in self.token_lengths:
                folder_name = f"asap_add_noise_at_{sec}sec_tk{tk}"
                folders_to_process.append((folder_name, sec, tk))
        
        print(f"总共需要处理 {len(folders_to_process)} 个文件夹")
        
        for folder_name, sec, tk in tqdm(folders_to_process, desc="处理文件夹"):
            input_dir = self.base_dir / folder_name
            output_dir = f"asap_loss_time_ins_{sec}sec_tk{tk}"
            
            if not input_dir.exists():
                print(f"警告：文件夹 {input_dir} 不存在，跳过")
                continue
                
            print(f"处理: {folder_name} -> {output_dir}")
            
            # 构建loss_cal_small.py的命令
            cmd = [
                sys.executable, "Loss_Cal/loss_cal_small.py",
                "--audio_dir", str(input_dir),
                "--reduction", "none",
                "--token_output_dir", output_dir
            ]
            
            if force_cpu:
                cmd.append("--force_cpu")
                
            try:
                result = subprocess.run(cmd, cwd=self.base_dir, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                      universal_newlines=True, check=True)
                print(f"✓ 完成: {folder_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ 错误处理 {folder_name}: {e}")
                print(f"错误输出: {e.stderr}")
                
    def step2_generate_plots(self):
        """
        步骤2：按token长度分组生成loss差异图片
        """
        print("\n=== 步骤2：生成loss差异图片 ===")
        
        # 创建输出目录
        output_base = self.base_dir / "Loss_Plot" / "Phase4_3" / "asap_per_sec"
        output_base.mkdir(parents=True, exist_ok=True)
        
        # 原始数据目录（基准）
        ori_dir = "asap_loss_time_ins_ori"
        
        for tk in tqdm(self.token_lengths, desc="处理token长度组"):
            print(f"\n处理 tk{tk} 组...")
            
            for sec in tqdm(self.seconds, desc=f"tk{tk}的秒数", leave=False):
                comparison_dir = f"asap_loss_time_ins_{sec}sec_tk{tk}"
                output_dir = output_base / f"tk{tk}_vs_{sec}sec"
                
                # 检查输入目录是否存在
                if not (self.base_dir / comparison_dir).exists():
                    print(f"警告：目录 {comparison_dir} 不存在，跳过")
                    continue
                
                # 根据用户提供的对应关系计算noise_start和noise_end
                # tk5_vs_1sec: noise_start=50, noise_end=54
                # tk5_vs_2sec: noise_start=100, noise_end=104
                # 规律：noise_start = sec * 50, noise_end = noise_start + tk - 1
                noise_start = sec * 50
                noise_end = noise_start + tk - 1
                
                # 构建plot_loss_diff_time_in_separate.py的命令
                cmd = [
                    sys.executable, "Plot/plot_loss_diff_time_in_separate.py",
                    "--dir1", ori_dir,
                    "--dir2", comparison_dir,
                    "--output_dir", str(output_dir),
                    "--noise_start", str(noise_start),
                    "--noise_end", str(noise_end),
                    "--plot_start", "0",
                    "--plot_end", "750",
                    "--y_min", "-8",
                    "--y_max", "12"
                ]
                
                try:
                    result = subprocess.run(cmd, cwd=self.base_dir,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                          universal_newlines=True, check=True)
                    print(f"✓ 完成: tk{tk} vs {sec}sec (noise_start={noise_start}, noise_end={noise_end})")
                except subprocess.CalledProcessError as e:
                    print(f"✗ 错误生成图片 tk{tk} vs {sec}sec: {e}")
                    print(f"错误输出: {e.stderr}")
                    
    def step3_merge_images(self):
        """
        步骤3：合并每组的14张图片（直接叠加，覆盖区域颜色加深）
        """
        print("\n=== 步骤3：合并图片 ===")
        
        input_base = self.base_dir / "Loss_Plot" / "Phase4_3" / "asap_per_sec"
        output_base = self.base_dir / "Loss_Plot" / "Phase4_3" / "asap_per_sec_sum"
        output_base.mkdir(parents=True, exist_ok=True)
        
        for tk in tqdm(self.token_lengths, desc="合并token长度组"):
            print(f"\n合并 tk{tk} 组的图片...")
            
            # 收集所有图片文件（根据实际路径结构）
            image_files = []
            for sec in self.seconds:
                plot_dir = input_base / f"tk{tk}_vs_{sec}sec"
                if plot_dir.exists():
                    # 查找所有子目录中的PNG文件
                    for subdir in plot_dir.iterdir():
                        if subdir.is_dir():
                            png_files = list(subdir.glob("*.png"))
                            if png_files:
                                image_files.extend([(sec, f) for f in png_files])
                                
            if not image_files:
                print(f"警告：tk{tk} 组没有找到图片文件")
                continue
                
            # 按歌曲名称分组（从文件名中提取歌曲标识）
            songs_dict = {}
            for sec, img_path in image_files:
                # 从文件名中提取歌曲标识（去掉loss_diff_前缀和后缀）
                filename = img_path.stem
                if filename.startswith('loss_diff_'):
                    # 提取歌曲名称部分
                    song_part = filename[10:]  # 去掉'loss_diff_'前缀
                    # 去掉最后的时间范围部分（如_0_750）
                    if '_0_750' in song_part:
                        song_name = song_part.replace('_0_750', '')
                    else:
                        song_name = song_part
                else:
                    song_name = filename
                    
                if song_name not in songs_dict:
                    songs_dict[song_name] = {}
                songs_dict[song_name][sec] = img_path
                
            # 为每首歌曲创建合并图片
            for song_name, sec_images in tqdm(songs_dict.items(), 
                                            desc=f"tk{tk}的歌曲", leave=False):
                self._merge_song_images_overlay(song_name, sec_images, tk, output_base)
                
    def _merge_song_images_overlay(self, song_name, sec_images, tk, output_base):
        """
        使用叠加方式合并单首歌曲的14张图片（覆盖区域颜色加深）
        """
        # 按秒数排序
        sorted_secs = sorted(sec_images.keys())
        
        if len(sorted_secs) < len(self.seconds):
            print(f"警告：{song_name} (tk{tk}) 只有 {len(sorted_secs)} 张图片，期望 {len(self.seconds)} 张")
            
        if not sorted_secs:
            return
            
        # 读取第一张图片获取尺寸
        first_img_path = sec_images[sorted_secs[0]]
        first_img = Image.open(first_img_path)
        # 确保转换为RGB模式（去除透明度通道）
        if first_img.mode != 'RGB':
            first_img = first_img.convert('RGB')
        img_width, img_height = first_img.size
        
        # 创建基础图片（白色背景）
        merged_img = Image.new('RGB', (img_width, img_height), 'white')
        merged_array = np.array(merged_img, dtype=np.float32)
        
        # 叠加所有图片
        for sec in sorted_secs:
            img_path = sec_images[sec]
            img = Image.open(img_path)
            
            # 确保转换为RGB模式（统一颜色通道数）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整图片大小（如果需要）
            if img.size != (img_width, img_height):
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                
            # 转换为numpy数组
            img_array = np.array(img, dtype=np.float32)
            
            # 计算叠加权重（非白色区域进行叠加）
            # 白色像素值为[255, 255, 255]，非白色区域参与叠加
            white_mask = np.all(img_array >= 250, axis=2)  # 接近白色的区域
            
            # 对非白色区域进行加权叠加
            overlay_mask = ~white_mask
            if np.any(overlay_mask):
                # 使用加法混合，但限制最大值避免过度曝光
                alpha = 0.7  # 叠加强度
                merged_array[overlay_mask] = np.minimum(
                    merged_array[overlay_mask] * (1 - alpha) + img_array[overlay_mask] * alpha,
                    255.0
                )
        
        # 转换回PIL图像并保存
        final_img = Image.fromarray(np.uint8(merged_array))
        output_path = output_base / f"{song_name}_tk{tk}_overlay.png"
        final_img.save(output_path, dpi=(300, 300))
        print(f"✓ 保存叠加图片: {output_path}")
        
    def run_all_steps(self, force_cpu=False, skip_step1=False, skip_step2=False, skip_step3=False):
        """
        运行所有步骤
        """
        print("开始ASAP per-second噪声分析流程")
        print(f"基础目录: {self.base_dir}")
        print(f"Token长度: {self.token_lengths}")
        print(f"时间范围: {self.seconds[0]}-{self.seconds[-1]}秒")
        
        try:
            if not skip_step1:
                self.step1_calculate_losses(force_cpu)
            else:
                print("跳过步骤1")
                
            if not skip_step2:
                self.step2_generate_plots()
            else:
                print("跳过步骤2")
                
            if not skip_step3:
                self.step3_merge_images()
            else:
                print("跳过步骤3")
                
            print("\n🎉 所有步骤完成！")
            
        except KeyboardInterrupt:
            print("\n用户中断操作")
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="ASAP数据集per-second噪声分析综合脚本")
    parser.add_argument("--base_dir", type=str, default="/home/evev/asap-dataset",
                       help="基础目录路径")
    parser.add_argument("--force_cpu", action="store_true",
                       help="强制使用CPU进行loss计算")
    parser.add_argument("--skip_step1", action="store_true",
                       help="跳过步骤1（loss计算）")
    parser.add_argument("--skip_step2", action="store_true",
                       help="跳过步骤2（图片生成）")
    parser.add_argument("--skip_step3", action="store_true",
                       help="跳过步骤3（图片合并）")
    
    args = parser.parse_args()
    
    analyzer = ASAPPerSecAnalyzer(args.base_dir)
    analyzer.run_all_steps(
        force_cpu=args.force_cpu,
        skip_step1=args.skip_step1,
        skip_step2=args.skip_step2,
        skip_step3=args.skip_step3
    )

if __name__ == "__main__":
    main()