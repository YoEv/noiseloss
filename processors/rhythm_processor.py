# processors/rhythm_processor.py
import os
import random
import copy
import json
import numpy as np
from pretty_midi import PrettyMIDI
from datetime import datetime

class RhythmProcessor:
    def __init__(self, config_path):
        """
        Args:
            config_path: 配置文件路径
            output_dir: 处理器专用输出目录 (由ExperimentRunner创建并传入)
        """
        self.config = self._load_config(config_path)
        self.min_note = 0.25
        self.min_note_conversion = 0.25
        self.min_note_density = 0.0125
        self.min_note_deletion = 0.0125
        self.output_dir = None
        self.err_log = []
        self.success_log = []

    def _load_config(self, path):
        from ruamel.yaml import YAML
        with open(path, 'r', encoding='utf-8') as f:
            return YAML().load(f)

    def _log_success(self, message):
        self.success_log.append(message)

    def _log_err(self, message):
        self.err_log.append(message)

    def _set_output_dir(self, output_dir):
        if output_dir is None:
            raise ValueError("output_dir must be provided")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_save_path(self, midi, suffix):
        basename = os.path.basename(getattr(midi, 'filename', 'untitled.mid'))
        name, ext = os.path.splitext(basename)
        new_name = f"{name}{suffix}{ext}"
        return os.path.join(self.output_dir, new_name)

    def _find_notes_by_duration(self, midi, durations):
        notes = []
        for instr in midi.instruments:
            for n in instr.notes:
                if any(abs((n.end - n.start) - d) < self.min_note for d in durations):
                    if not hasattr(n, 'instrument'):
                        n.instrument = instr
                    notes.append(n)
        return notes

    def _modify_duration(self, notes, source_dur, target_dur):
        modified = 0
        for note in notes:
            if abs((note.end - note.start) - source_dur) <= self.min_note:
                note.end = note.start + target_dur
                modified += 1
                # 确保修改后的音符保留instrument信息
                if not hasattr(note, 'instrument'):
                    note.instrument = None
        return notes

    def process(self, midi, output_base_dir, output_dir):
        self._set_output_dir(output_dir)
        if not hasattr(midi, 'filename'):
            midi.filename = "untitled.mid"

        for action in self.config['actions']:
            for step in action['steps']:
                self._process_step(midi, step)

        path = os.path.join(output_base_dir, 'rhythm_error_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.err_log, f, ensure_ascii=False, indent=2)

        path = os.path.join(output_base_dir, 'rhythm_success_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.success_log, f, ensure_ascii=False, indent=2)

        return midi

    def _process_step(self, midi, step):
        op = step['operation']
        sel_type = step['selection_type']
        params = step['parameters']
        if op == 'duration_change':
            self._process_duration_change(midi, sel_type, params)
        elif op == 'rhythm_conversion':
            self._process_rhythm_conversion(midi, sel_type, params)
        elif op == 'density_adjust':
            self._process_density_adjust(midi, sel_type, params)
        elif op == 'note_deletion':
            self._process_note_deletion(midi, sel_type, params)
        else:
            raise ValueError(f"未知操作类型: {op}")

    def _process_duration_change(self, midi, sel_type, step_config):
        source_dur = step_config['source_durations'] #实际上检测到的音符长度为二分音符即可
        target_dur = step_config['target_duration']

        results = []

        for nc_config in step_config['note_counts']:
            mod_midi = copy.deepcopy(midi)
            count = nc_config['count']
            all_notes = self._find_notes_by_duration(mod_midi, source_dur)
            suffix = nc_config['save_suffix']

            try:
                save_path = self._get_save_path(midi, suffix)
                if count > len(all_notes):
                    print(f"需要修改{count}个音，但只有{len(all_notes)}个符合要求的音符")
                    raise ValueError(f"需要修改{count}个音，但只有{len(all_notes)}个符合要求的音符")

                if (sel_type == 'random') :
                    selected = random.sample(all_notes, count)
                else:
                    selected = []

                modified = self._modify_duration(selected, source_dur, target_dur)

                mod_midi.write(save_path)

                success_result = {
                    'requested_count': count,
                    'actual_modified': str(modified),
                    'output_path': save_path,
                    'status': 'success'
                }
                results.append(success_result)
                self._log_success(success_result)

            except Exception as e:
                this_result = {
                    'file': save_path,
                    'requested_count': count,
                    'error': str(e),
                    'status': 'failed'
                }
                results.append(this_result)
                self._log_err(this_result)

        return results

    def _process_rhythm_conversion(self, midi, sel_type, step_config):
        conversion_map = dict(zip(step_config['source_patterns'], step_config['target_patterns']))

        all_candidates = []
        for instr in midi.instruments:
            for note in instr.notes:
                original_dur = note.end - note.start
                closest_source = min(step_config['source_patterns'],
                                key=lambda x: abs(x - original_dur))
                if abs(closest_source - original_dur) < self.min_note_conversion:
                    all_candidates.append((instr, note, closest_source))

        results = []

        for nc_config in step_config['note_counts']:
            mod_midi = copy.deepcopy(midi)
            count = nc_config['count']
            suffix = nc_config['save_suffix']

            try:
                save_path = self._get_save_path(midi, suffix)
                # calculate possible note count
                if count > len(all_candidates):
                    raise ValueError(
                        f"需要转换{count}个音，但只有{len(all_candidates)}个符合要求的音符")

                # select notes
                if sel_type == 'random':
                    selected = random.sample(all_candidates, count)
                else:  # continuous
                    selected = all_candidates[:count]

                # rhythm conversion
                modified = 0
                for instr, note, src_dur in selected:
                    target_dur = conversion_map[src_dur]
                    note.end = note.start + target_dur
                    modified += 1

                #print("=================Step 6.4====================")
                mod_midi.write(save_path)

                success_result = {
                    'conversion_type': step_config['conversion_type'],
                    'requested_count': count,
                    'actual_modified': modified,
                    'output_path': save_path,
                    'status': 'success'
                }
                results.append(success_result)
                self._log_success(success_result)

            except Exception as e:
                this_result = {
                    'file': save_path,
                    'conversion_type': step_config['conversion_type'],
                    'requested_count': count,
                    'error': str(e),
                    'status': 'failed'
                }
                results.append(this_result)
                self._log_err(this_result)

        return results

    def _process_density_adjust(self, midi, sel_type, step_config):
        total_time = midi.get_end_time()
        results = []

        for ratio_config in step_config['section_ratios']:
            mod_midi = copy.deepcopy(midi)
            section_ratio = ratio_config['ratio']
            suffix = ratio_config['save_suffix']
            section_length = total_time * section_ratio

            try:
                save_path = self._get_save_path(midi, suffix)
                if sel_type == 'random':
                    start_time = random.uniform(0, total_time - section_length)
                else:  # continuous
                    max_start = total_time - section_length
                    start_time = random.uniform(0, max_start) if max_start > 0 else 0

                end_time = start_time + section_length

                # all instruments!!! have to
                modified_notes = 0
                for instr in mod_midi.instruments:
                    # 获取区间内的音符（带边界容错）
                    notes_in_section = [
                        n for n in instr.notes
                        if start_time - self.min_note_density <= n.start <= end_time + self.min_note_density
                    ]

                    if step_config['density_change'] == "halve":
                        for note in notes_in_section:
                            original_duration = note.end - note.start
                            note.end = note.start + (original_duration * 0.5)
                            modified_notes += 1

                    elif step_config['density_change'] == "double":
                        new_notes = []
                        for note in notes_in_section:
                            original_duration = note.end - note.start

                            # divide two notes to count time
                            first_note = copy.deepcopy(note)
                            first_note.end = first_note.start + (original_duration * 0.5)

                            second_note = copy.deepcopy(note)
                            second_note.start = first_note.end
                            second_note.end = note.end

                            # note replace with density adjusted
                            note.start = first_note.start
                            note.end = first_note.end
                            new_notes.append(second_note)

                            modified_notes += 2

                        instr.notes.extend(new_notes)
                        instr.notes.sort(key=lambda x: x.start)

                mod_midi.write(save_path)

                success_result = {
                    'mode': sel_type,
                    'section_ratio': section_ratio,
                    'time_window': (round(start_time, 2), round(end_time, 2)),
                    'modified_notes': str(modified_notes),
                    'output_path': save_path,
                    'status': 'success'
                }
                results.append(success_result)
                self._log_success(success_result)

            except Exception as e:
                this_result = {
                    'file': save_path,
                    'section_ratio': section_ratio,
                    'error': str(e),
                    'status': 'failed'
                }
                results.append(this_result)
                self._log_err(this_result)

        return results

    def _process_note_deletion(self, midi, sel_type, step_config):
        results = []

        for ratio_config in step_config['deletion_ratios']:
            mod_midi = copy.deepcopy(midi)
            ratio = ratio_config['ratio']
            suffix = ratio_config['save_suffix']

            try:
                save_path = self._get_save_path(midi, suffix)
                all_notes = sorted(
                    [n for instr in mod_midi.instruments for n in instr.notes],
                    key=lambda x: x.start
                )

                for note in all_notes:
                    for instr in mod_midi.instruments:
                        if note in instr.notes:
                            note.instrument = instr
                            break

                total_notes = len(all_notes)

                if sel_type == 'random':
                    delete_count = int(total_notes * ratio)
                    to_delete = random.sample(all_notes, delete_count)

                else:  # patterned
                    eighth_note = step_config.get('pattern_interval', 0.125)
                    total_time = mod_midi.get_end_time()
                    section_length = total_time * ratio

                    start_time = random.uniform(0, total_time - section_length)
                    end_time = start_time + section_length

                    pattern_beats = np.arange(
                        start_time,
                        end_time,
                        eighth_note
                    )

                    to_delete = [
                        n for n in all_notes
                        if any(abs(n.start - beat) < self.min_note_deletion for beat in pattern_beats)
                    ]

                # note deletion
                for note in reversed(to_delete):
                    note.instrument.notes.remove(note)

                mod_midi.write(save_path)

                success_result = {
                    'deletion_ratio': ratio,
                    'deletion_mode': step_config['deletion_mode'],
                    'deleted_count': len(to_delete),
                    'remaining_count': total_notes - len(to_delete),
                    'output_path': save_path,
                    'status': 'success'
                }
                results.append(success_result)
                self._log_success(success_result)

            except Exception as e:
                this_result = {
                    'file': save_path,
                    'deletion_ratio': ratio,
                    'error': str(e),
                    'status': 'failed'
                }
                results.append(this_result)
                self._log_err(this_result)

        return results
