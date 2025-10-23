import os
import random
import copy
import json
from pretty_midi import PrettyMIDI
from datetime import datetime

class VelocityProcessor:
    def __init__(self, config_path):
        """
        Args:
            config_path: Configuration file path
            output_dir: Processor-specific output directory (created and passed by ExperimentRunner)
        """
        self.config = self._load_config(config_path)
        self.output_dir = None
        self.err_log = []
        self.success_log = []
        self.min_velocity_diff = 20  # Minimum velocity difference to consider a change

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

    def _get_all_notes(self, midi):
        """Get all notes from all instruments in the MIDI file"""
        all_notes = []
        for instr in midi.instruments:
            for note in instr.notes:
                if not hasattr(note, 'instrument'):
                    note.instrument = instr
                all_notes.append(note)
        return all_notes

    def _modify_velocity(self, notes, target_velocity=None, source_velocities=None):
        """Modify the velocity of selected notes"""
        modified = 0

        for note in notes:
            # If source_velocities is specified, check if the note's velocity matches any in the list
            if source_velocities and len(source_velocities) > 0:
                if not any(abs(note.velocity - v) < self.min_velocity_diff for v in source_velocities):
                    continue

            # If target_velocity is specified, set the note's velocity to that value
            if target_velocity is not None:
                note.velocity = min(127, max(1, target_velocity))  # Ensure velocity is within MIDI range (1-127)
            else:
                # If no target velocity is specified, generate a random velocity
                note.velocity = random.randint(30, 120)  # Random velocity between 30 and 120

            modified += 1

        return modified

    def process(self, midi, output_base_dir, output_dir):
        """Process the MIDI file according to the configuration"""
        self._set_output_dir(output_dir)
        if not hasattr(midi, 'filename'):
            midi.filename = "untitled.mid"

        for action in self.config['actions']:
            for step in action['steps']:
                self._process_step(midi, step)

        # Save logs
        path = os.path.join(output_base_dir, 'velocity_error_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.err_log, f, ensure_ascii=False, indent=2)

        path = os.path.join(output_base_dir, 'velocity_success_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.success_log, f, ensure_ascii=False, indent=2)

        return midi

    def _process_step(self, midi, step):
        """Process a single step from the configuration"""
        op = step['operation']
        sel_type = step['selection_type']
        params = step['parameters']

        if op == 'velocity_change':
            self._process_velocity_change(midi, sel_type, params)
        else:
            raise ValueError(f"Unknown operation type: {op}")

    def _process_velocity_change(self, midi, sel_type, step_config):
        """Process velocity change operation"""
        source_velocities = step_config.get('source_velocities', [])
        target_velocity = step_config.get('target_velocity')

        results = []

        for mod_config in step_config['modifications']:
            mod_midi = copy.deepcopy(midi)
            percentage = mod_config['percentage']
            suffix = mod_config['save_suffix']

            try:
                save_path = self._get_save_path(midi, suffix)
                all_notes = self._get_all_notes(mod_midi)

                if not all_notes:
                    raise ValueError("No notes found in the MIDI file")

                # Calculate how many notes to modify based on the percentage
                mod_count = int(len(all_notes) * percentage / 100)

                if mod_count <= 0:
                    raise ValueError(f"No notes to modify with percentage {percentage}%")

                # Select notes to modify
                if sel_type == 'random':
                    selected_notes = random.sample(all_notes, mod_count)
                else:  # sequential or other selection types
                    selected_notes = all_notes[:mod_count]

                # Modify the velocity of selected notes
                modified_count = self._modify_velocity(
                    selected_notes,
                    target_velocity=target_velocity,
                    source_velocities=source_velocities
                )

                # Save the modified MIDI file
                mod_midi.write(save_path)

                success_result = {
                    'percentage': percentage,
                    'requested_mods': mod_count,
                    'actual_modified': modified_count,
                    'output_path': save_path,
                    'status': 'success'
                }
                results.append(success_result)
                self._log_success(success_result)

            except Exception as e:
                error_result = {
                    'percentage': percentage,
                    'error': str(e),
                    'output_path': save_path if 'save_path' in locals() else None,
                    'status': 'failed'
                }
                results.append(error_result)
                self._log_err(error_result)

        return results