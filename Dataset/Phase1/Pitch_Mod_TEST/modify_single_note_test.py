import os
import random
import copy
from pretty_midi import PrettyMIDI, note_number_to_name
from tqdm import tqdm

class MidiModifier:
    def __init__(self):
        self.log = []
    
    def _deep_clone_midi(self, midi):
        """彻底深克隆MIDI对象（解决pretty_midi的克隆问题）"""
        # 创建新对象并复制基本属性
        new_midi = PrettyMIDI()
        new_midi.time_signature_changes = copy.deepcopy(midi.time_signature_changes)
        new_midi.key_signature_changes = copy.deepcopy(midi.key_signature_changes)
        new_midi.lyrics = copy.deepcopy(midi.lyrics)
        
        # 深克隆乐器轨道
        for instr in midi.instruments:
            new_instr = copy.deepcopy(instr)
            new_midi.instruments.append(new_instr)
        
        return new_midi

    def _get_editable_notes(self, midi):
        """获取可独立修改的音符引用"""
        # 必须重新绑定到克隆后的乐器轨道
        return [note for instr in midi.instruments for note in instr.notes]

    def modify_single_note(self, midi_path, output_path):
        """单音修改（完整修复版）"""
        try:
            # 加载原始MIDI
            orig_midi = PrettyMIDI(midi_path)
            if not orig_midi.instruments:
                return False
            
            # 创建完全独立的克隆
            mod_midi = self._deep_clone_midi(orig_midi)
            mod_notes = self._get_editable_notes(mod_midi)
            
            # 随机选择一个可修改的音符
            if not mod_notes:
                return False
            target_note = random.choice(mod_notes)
            original_pitch = target_note.pitch
            
            # 生成合规修改
            new_pitch = original_pitch + random.choice([-12, 12])
            target_note.pitch = new_pitch
            
            # 记录日志
            self.log.append({
                'file': os.path.basename(output_path),
                'original': f"{note_number_to_name(original_pitch)} ({original_pitch})",
                'modified': f"{note_number_to_name(new_pitch)} ({new_pitch})",
                'time': target_note.start
            })
            
            # 保存修改
            mod_midi.write(output_path)
            return True
            
        except Exception as e:
            print(f"处理错误 {midi_path}: {str(e)}")
            return False

# 使用示例
if __name__ == "__main__":
    modifier = MidiModifier()
    
    input_dir = "./selected_midis_cropped"
    output_dir = "./verified_modified"
    os.makedirs(output_dir, exist_ok=True)
    
    midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
    
    success = 0
    for fname in tqdm(midi_files, desc="Processing"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"mod_{fname}")
        
        if modifier.modify_single_note(in_path, out_path):
            success += 1
    
    print(f"\n成功修改文件: {success}/{len(midi_files)}")
    
    # 打印详细修改记录
    print("\n=== 修改详情验证 ===")
    for entry in modifier.log:
        print(f"文件: {entry['file']}")
        print(f"时间: {entry['time']:.2f}s")
        print(f"原音高: {entry['original']}")
        print(f"新音高: {entry['modified']}\n")