import os
import random
import copy
import json
from pretty_midi import PrettyMIDI
from datetime import datetime

class PitchProcessor:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.safe_pitch_range = (24, 103)  # 安全音高范围
        self.max_attempts = 100  # 最大尝试次数
        self.err_log = []
        self.success_log = []
        self.output_dir = None

    def _load_config(self, path):
        from ruamel.yaml import YAML
        with open(path, 'r', encoding='utf-8') as f:
            return YAML().load(f)

    def _set_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_save_path(self, midi, suffix):
        basename = os.path.basename(getattr(midi, 'filename', 'untitled.mid'))
        name, ext = os.path.splitext(basename)
        new_name = f"{name}{suffix}{ext}"
        return os.path.join(self.output_dir, new_name)

    def _clamp_pitch(self, pitch):
        return max(self.safe_pitch_range[0], min(self.safe_pitch_range[1], pitch))

    # Removed _find_main_track method

    def _get_modifiable_notes(self, midi, percentage, selection_type="random"):
        # Process all tracks instead of just the main track
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend([n for n in instrument.notes
                        if self.safe_pitch_range[0] <= n.pitch <= self.safe_pitch_range[1]])

        mod_count = int(len(all_notes) * percentage / 100)
        if mod_count <= 0:
            return []

        if selection_type == "random":
            return random.sample(all_notes, mod_count)
        else:  # sequential或其他选择类型
            return all_notes[:mod_count]

    def _validate_pitch_change(self, original_pitch, new_pitch, key_signature):
        diatonic_scale = [0,2,4,5,7,9,11]  # 自然音阶
        key = key_signature.key_number % 12 if key_signature else 0
        scale_degrees = [(key + offset) % 12 for offset in diatonic_scale]
        return (new_pitch % 12) in scale_degrees

    def process(self, midi, output_base_dir, output_dir):
        self._set_output_dir(output_dir)
        if not hasattr(midi, 'filename'):
            midi.filename = "untitled.mid"

        for action in self.config['actions']:
            self._process_action(midi, action)

        #self._save_logs(output_base_dir)
        return midi

    def _process_action(self, midi, action):
        """处理单个action（包含多个steps）"""
        action_name = action['name']

        # 修改点2：处理steps层级
        for step in action['steps']:
            self._process_step(midi, action_name, step)

    def _process_step(self, midi, action_name, step):
        """处理单个step（包含多个modifications）"""
        operation = step['operation']
        selection_type = step['selection_type']
        parameters = step['parameters']
        shifts = parameters['shifts']

        # 修改点3：处理modifications层级
        for mod_config in parameters['modifications']:
            mod_midi = copy.deepcopy(midi)
            percentage = mod_config['percentage']
            suffix = mod_config['save_suffix']

            try:
                save_path = self._get_save_path(mod_midi, suffix)
                notes_to_modify = self._get_modifiable_notes(
                    mod_midi, percentage, selection_type)

                if not notes_to_modify:
                    raise ValueError(f"No modifiable notes found for {percentage}%")

                key_sig = mod_midi.key_signature_changes[0] if mod_midi.key_signature_changes else None
                modified_count = 0

                for note in notes_to_modify:
                    original_pitch = note.pitch

                    for _ in range(self.max_attempts):
                        shift = random.choice(shifts)
                        new_pitch = self._clamp_pitch(original_pitch + shift)

                        if self._validate_pitch_change(original_pitch, new_pitch, key_sig):
                            note.pitch = new_pitch
                            modified_count += 1
                            break

                mod_midi.write(save_path)

                result = {
                    'action': action_name,
                    'operation': operation,
                    'selection_type': selection_type,
                    'percentage': percentage,
                    'shift_range': shifts,
                    'requested_mods': len(notes_to_modify),
                    'actual_mods': modified_count,
                    'output_path': save_path,
                    'status': 'success'
                }
                self.success_log.append(result)

            except Exception as e:
                result = {
                    'action': action_name,
                    'operation': operation,
                    'percentage': percentage,
                    'error': str(e),
                    'output_path': save_path,
                    'status': 'failed'
                }
                self.err_log.append(result)

    # def _save_logs(self, output_dir):
    #     """保存日志文件"""
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     log_data = {
    #         'timestamp': timestamp,
    #         'success': self.success_log,
    #         'errors': self.err_log
    #     }

    #     log_path = os.path.join(output_dir, f'pitch_log_{timestamp}.json')
    #     with open(log_path, 'w', encoding='utf-8') as f:
    #         json.dump(log_data, f, indent=2, ensure_ascii=False)