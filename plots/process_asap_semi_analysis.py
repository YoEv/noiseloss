#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASAP Semi-tone Analysis Pipeline
处理ASAP数据集的半音修改分析流程
"""

import os
import shutil
import subprocess
import glob
from pathlib import Path
import mido
import numpy as np
from datetime import datetime

class ASAPSemiAnalyzer:
    def __init__(self):
        self.base_dir = "/home/evev/asap-dataset"
        self.source_audio_dir = os.path.join(self.base_dir, "asap_silent_sele_ori")
        self.source_midi_dir = os.path.join(self.base_dir, "Dataset/Phase2_Stage2_asap_lmd/ASAP_100")
        self.output_midi_ori = os.path.join(self.base_dir, "asap_silent_sele_midi_ori")
        self.output_midi_semi = os.path.join(self.base_dir, "asap_silent_sele_midi_semi")
        self.output_audio_ori = os.path.join(self.base_dir, "asap_silent_sele_wav_ori")
        self.output_audio_semi = os.path.join(self.base_dir, "asap_silent_sele_midi_semi")
        self.output_loss_semi = os.path.join(self.base_dir, "asap_loss_time_ins_semi")
        self.output_loss_ori_new = os.path.join(self.base_dir, "asap_loss_time_ins_ori_new")
        self.output_loss_ori = os.path.join(self.base_dir, "asap_loss_time_ins_ori")
        self.output_plot_dir = os.path.join(self.base_dir, "loss_diff_plot_asap_semi")
        self.soundfont_path = os.path.join(self.base_dir, "soundfonts/GeneralUser-GS/GeneralUser-GS.sf2")
        self.log_file = os.path.join(self.base_dir, "midi_modification_log.txt")
        
        # 创建输出目录
        for dir_path in [self.output_midi_ori, self.output_midi_semi, self.output_audio_ori, 
                        self.output_audio_semi, self.output_loss_semi, self.output_loss_ori_new, 
                        self.output_plot_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def step1_extract_midi_files(self):
        """步骤1: 提取对应的MIDI文件"""
        print("步骤1: 提取MIDI文件...")
        
        # 获取所有音频文件的基础名称
        audio_files = []
        for root, dirs, files in os.walk(self.source_audio_dir):
            for file in files:
                if file.endswith('_ori_cut15s.wav'):
                    # 正确提取基础文件名
                    # 从 "Beethoven_010_midi_score_short.mid_ori_cut15s.wav" 
                    # 提取 "Beethoven_010_midi_score_short.mid"
                    base_name = file.replace('_ori_cut15s.wav', '')
                    audio_files.append(base_name)
        
        print(f"找到 {len(audio_files)} 个音频文件需要匹配MIDI")
        
        # 复制对应的MIDI文件
        copied_count = 0
        for midi_name in audio_files:
            source_path = os.path.join(self.source_midi_dir, midi_name)
            if os.path.exists(source_path):
                dest_path = os.path.join(self.output_midi_ori, midi_name)
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                print(f"复制: {midi_name}")
            else:
                print(f"警告: 未找到MIDI文件 {midi_name}")
        
        print(f"成功复制 {copied_count} 个MIDI文件到 {self.output_midi_ori}")
        return copied_count > 0
    
    def step2_trim_midi_files(self):
        """步骤2: 截取MIDI文件为15秒"""
        print("步骤2: 截取MIDI文件为15秒...")
        
        midi_files = glob.glob(os.path.join(self.output_midi_ori, "*.mid"))
        
        for midi_file in midi_files:
            try:
                mid = mido.MidiFile(midi_file)
                
                # 计算15秒对应的ticks
                tempo = 500000  # 默认tempo (120 BPM)
                ticks_per_beat = mid.ticks_per_beat
                
                # 寻找tempo变化
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == 'set_tempo':
                            tempo = msg.tempo
                            break
                
                # 计算15秒对应的ticks
                seconds_per_tick = (tempo / 1000000) / ticks_per_beat
                target_ticks = int(15 / seconds_per_tick)
                
                # 创建新的MIDI文件
                new_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
                
                for track in mid.tracks:
                    new_track = mido.MidiTrack()
                    current_time = 0
                    
                    for msg in track:
                        current_time += msg.time
                        if current_time <= target_ticks:
                            new_track.append(msg)
                        else:
                            # 添加最后一个消息但调整时间
                            if msg.type in ['note_off', 'end_of_track']:
                                msg.time = target_ticks - (current_time - msg.time)
                                new_track.append(msg)
                            break
                    
                    new_mid.tracks.append(new_track)
                
                # 保存截取后的文件
                output_path = os.path.join(self.output_midi_ori, os.path.basename(midi_file))
                new_mid.save(output_path)
                print(f"截取完成: {os.path.basename(midi_file)}")
                
            except Exception as e:
                print(f"处理 {midi_file} 时出错: {e}")
        
        print("MIDI文件截取完成")
        return True
    
    def step3_synthesize_original_audio(self):
        """步骤3: 合成原始MIDI文件为WAV"""
        print("步骤3: 合成原始MIDI文件为WAV...")
        
        midi_files = glob.glob(os.path.join(self.output_midi_ori, "*.mid"))
        
        for midi_file in midi_files:
            try:
                base_name = os.path.splitext(os.path.basename(midi_file))[0]
                output_wav = os.path.join(self.output_audio_ori, f"{base_name}.wav")
                
                # 使用fluidsynth合成音频
                cmd = [
                    "fluidsynth",
                    "-ni",
                    self.soundfont_path,
                    midi_file,
                    "-F", output_wav,
                    "-r", "48000"
                ]
                
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                
                if result.returncode == 0:
                    print(f"原始音频合成完成: {base_name}.wav")
                else:
                    print(f"原始音频合成失败: {base_name} - {result.stderr}")
                    
            except Exception as e:
                print(f"处理 {midi_file} 时出错: {e}")
        
        print("原始音频合成完成")
        return True

    def step4_calculate_original_losses(self):
        """步骤4: 计算原始音频的per token loss"""
        print("步骤4: 计算原始音频的per token loss...")
        
        try:
            cmd = [
                "python", "Loss_Cal/loss_cal_small.py",
                "--audio_dir", self.output_audio_ori,
                "--token_output_dir", self.output_loss_ori_new,
                "--reduction", "none"
            ]
            
            result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("原始音频loss计算完成")
                return True
            else:
                print(f"原始音频loss计算失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"计算原始音频loss时出错: {e}")
            return False

    def _note_to_name(self, note_number):
        """将MIDI音符号转换为音符名称"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = note_number // 12 - 1
        note = notes[note_number % 12]
        return f"{note}{octave}"

    def step5_modify_midi_files(self):
        """步骤5: 修改MIDI音符"""
        print("步骤5: 修改MIDI音符...")
        
        midi_files = glob.glob(os.path.join(self.output_midi_ori, "*.mid"))
        modification_log = []
        
        for midi_file in midi_files:
            try:
                # 复制到semi目录
                base_name = os.path.basename(midi_file)
                output_file = os.path.join(self.output_midi_semi, base_name)
                shutil.copy2(midi_file, output_file)
                
                # 加载MIDI文件
                mid = mido.MidiFile(output_file)
                
                # 寻找第5秒左右的音符进行修改
                target_time_seconds = 5.0  # 5秒
                modified = False
                
                for track in mid.tracks:
                    current_time_ticks = 0
                    current_time_seconds = 0
                    
                    for msg in track:
                        current_time_ticks += msg.time
                        # 将ticks转换为秒数
                        current_time_seconds = mido.tick2second(current_time_ticks, mid.ticks_per_beat, 500000)  # 默认tempo
                        
                        # 在第5秒左右寻找note_on消息
                        if (msg.type == 'note_on' and msg.velocity > 0 and 
                            abs(current_time_seconds - target_time_seconds) < 2.0):  # 2秒容差
                            
                            # 检查音符是否在C3-C5范围内 (MIDI note 48-84)
                            if 48 <= msg.note <= 84 and not modified:
                                original_note = msg.note
                                msg.note = min(127, msg.note + 1)  # 提高一个半音
                                
                                log_entry = {
                                    'file': base_name,
                                    'time': current_time_seconds,
                                    'original_note': original_note,
                                    'modified_note': msg.note,
                                    'note_name_original': self._note_to_name(original_note),
                                    'note_name_modified': self._note_to_name(msg.note)
                                }
                                modification_log.append(log_entry)
                                modified = True
                                print(f"修改 {base_name}: {log_entry['note_name_original']} -> {log_entry['note_name_modified']} (时间: {current_time_seconds:.2f}s)")
                                break
                    
                    if modified:
                        break
                
                # 保存修改后的MIDI文件
                mid.save(output_file)
                
                if not modified:
                    print(f"警告: 在 {base_name} 的第5秒左右未找到合适的音符进行修改")
                    
            except Exception as e:
                print(f"处理 {midi_file} 时出错: {e}")
        
        # 保存修改日志
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"MIDI修改日志 - {datetime.now()}\n")
            f.write("="*50 + "\n")
            for entry in modification_log:
                f.write(f"文件: {entry['file']}\n")
                f.write(f"时间: {entry['time']:.2f}秒\n")
                f.write(f"原始音符: {entry['note_name_original']} (MIDI {entry['original_note']})\n")
                f.write(f"修改音符: {entry['note_name_modified']} (MIDI {entry['modified_note']})\n")
                f.write("-"*30 + "\n")
        
        print(f"MIDI修改完成，日志保存在: {self.log_file}")
        return True

    def step6_synthesize_audio(self):
        """步骤6: 合成修改后的音频文件"""
        print("步骤6: 合成修改后的音频文件...")
        
        midi_files = glob.glob(os.path.join(self.output_midi_semi, "*.mid"))
        
        for midi_file in midi_files:
            try:
                base_name = os.path.splitext(os.path.basename(midi_file))[0]
                output_wav = os.path.join(self.output_audio_semi, f"{base_name}.wav")
                
                # 使用fluidsynth合成音频
                cmd = [
                    "fluidsynth",
                    "-ni",
                    self.soundfont_path,
                    midi_file,
                    "-F", output_wav,
                    "-r", "48000"
                ]
                
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                
                if result.returncode == 0:
                    print(f"修改后音频合成完成: {base_name}.wav")
                else:
                    print(f"修改后音频合成失败: {base_name} - {result.stderr}")
                    
            except Exception as e:
                print(f"处理 {midi_file} 时出错: {e}")
        
        print("修改后音频合成完成")
        return True

    def step7_calculate_losses(self):
        """步骤7: 计算修改后音频的per token loss"""
        print("步骤7: 计算修改后音频的per token loss...")
        
        try:
            cmd = [
                "python", "Loss_Cal/loss_cal_small.py",
                "--audio_dir", self.output_audio_semi,
                "--token_output_dir", self.output_loss_semi,
                "--reduction", "none"
            ]
            
            result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("修改后音频loss计算完成")
                # 重命名文件以匹配原始格式
                self._rename_loss_files_to_match_original()
                return True
            else:
                print(f"修改后音频loss计算失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"计算修改后音频loss时出错: {e}")
            return False

    def _rename_loss_files_to_match_original(self):
        """重命名损失文件以匹配原始文件格式"""
        print("重命名损失文件以匹配原始格式...")
        
        # 创建原始文件名到速度分类的映射
        tempo_mapping = {}
        for root, dirs, files in os.walk(self.source_audio_dir):
            for file in files:
                if file.endswith('_ori_cut15s.wav'):
                    # 提取速度分类前缀
                    dir_name = os.path.basename(root)
                    if dir_name.startswith(('1_', '2_', '3_', '4_', '5_')):
                        # 从文件名提取基础MIDI名
                        base_name = file.replace('_ori_cut15s.wav', '')
                        tempo_mapping[base_name] = dir_name
        
        # 重命名生成的CSV文件
        csv_files = glob.glob(os.path.join(self.output_loss_semi, "*_tokens_avg.csv"))
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            # 从 "Chopin_030_midi_score_short_tokens_avg.csv" 提取 "Chopin_030_midi_score_short.mid"
            base_name = filename.replace('_tokens_avg.csv', '.mid')
            
            if base_name in tempo_mapping:
                tempo_prefix = tempo_mapping[base_name]
                new_filename = f"{tempo_prefix}_{base_name}_ori_cut15s_tokens_avg.csv"
                new_path = os.path.join(self.output_loss_semi, new_filename)
                
                try:
                    os.rename(csv_file, new_path)
                    print(f"重命名: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"重命名失败 {filename}: {e}")
            else:
                print(f"警告: 未找到 {base_name} 的速度分类映射")
    
    def step8_plot_comparison(self):
        """步骤8: 绘制对比图"""
        print("步骤8: 绘制对比图...")
        
        try:
            cmd = [
                "python",
                os.path.join(self.base_dir, "Plot/plot_loss_diff_time_in_separate.py"),
                "--dir1", self.output_loss_ori,
                "--dir2", self.output_loss_semi,
                "--output_dir", self.output_plot_dir,
                "--noise_start", "250",
                "--noise_end", "250",
                "--plot_start", "0",
                "--plot_end", "750",
                "--y_min", "-8",
                "--y_max", "12"
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            if result.returncode == 0:
                print("对比图绘制完成")
                print(result.stdout)
            else:
                print(f"绘制失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"绘制对比图时出错: {e}")
            return False
        
        return True
    
    def step9_plot_overall_diff(self):
        """步骤9: 绘制整体差异图"""
        print("步骤9: 绘制整体差异图...")
        
        try:
            cmd = [
                "python",
                os.path.join(self.base_dir, "Plot/plot_asap_loss_diff_comparison.py"),
                "--input_dir", self.output_plot_dir
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            if result.returncode == 0:
                print("整体差异图绘制完成")
                print(result.stdout)
            else:
                print(f"绘制失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"绘制整体差异图时出错: {e}")
            return False
        
        return True
    
    def run_all_steps(self, skip_steps=None):
        """运行所有步骤"""
        if skip_steps is None:
            skip_steps = []
        
        steps = [
            (1, "提取MIDI文件", self.step1_extract_midi_files),
            (2, "截取MIDI文件", self.step2_trim_midi_files),
            (3, "合成原始音频", self.step3_synthesize_original_audio),
            (4, "计算原始音频loss", self.step4_calculate_original_losses),
            (5, "修改MIDI音符", self.step5_modify_midi_files),
            (6, "合成修改后音频", self.step6_synthesize_audio),
            (7, "计算修改后音频loss", self.step7_calculate_losses),
            (8, "绘制对比图", self.step8_plot_comparison),
            (9, "绘制整体差异图", self.step9_plot_overall_diff)
        ]
        
        print("开始ASAP半音分析流程...")
        print("="*50)
        
        for step_num, step_name, step_func in steps:
            if step_num in skip_steps:
                print(f"跳过步骤{step_num}: {step_name}")
                continue
            
            print(f"\n执行步骤{step_num}: {step_name}")
            try:
                success = step_func()
                if not success:
                    print(f"步骤{step_num}失败，停止执行")
                    return False
            except Exception as e:
                print(f"步骤{step_num}出错: {e}")
                return False
        
        print("\n所有步骤完成！")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASAP半音分析流程')
    parser.add_argument('--skip-steps', nargs='*', type=int, default=[], 
                       help='跳过的步骤号 (1-7)')
    parser.add_argument('--step', type=int, choices=range(1, 8),
                       help='只执行指定步骤')
    
    args = parser.parse_args()
    
    analyzer = ASAPSemiAnalyzer()
    
    if args.step:
        # 只执行指定步骤
        steps = {
            1: analyzer.step1_extract_midi_files,
            2: analyzer.step2_trim_midi_files,
            3: analyzer.step3_modify_midi_files,
            4: analyzer.step4_synthesize_audio,
            5: analyzer.step5_calculate_losses,
            6: analyzer.step6_plot_comparison,
            7: analyzer.step7_plot_overall_diff
        }
        
        print(f"执行步骤{args.step}...")
        success = steps[args.step]()
        if success:
            print(f"步骤{args.step}完成")
        else:
            print(f"步骤{args.step}失败")
    else:
        # 执行所有步骤
        analyzer.run_all_steps(skip_steps=args.skip_steps)

if __name__ == "__main__":
    main()