import os
import random
import copy
import json
from pretty_midi import PrettyMIDI
from datetime import datetime

class StructureProcessor:
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

    def process(self, midi, output_base_dir, output_dir):
        """Process the MIDI file according to the configuration"""
        self._set_output_dir(output_dir)
        if not hasattr(midi, 'filename'):
            midi.filename = "untitled.mid"

        for action in self.config['actions']:
            for step in action['steps']:
                self._process_step(midi, step)

        # Save logs
        path = os.path.join(output_base_dir, 'structure_error_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.err_log, f, ensure_ascii=False, indent=2)

        path = os.path.join(output_base_dir, 'structure_success_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.success_log, f, ensure_ascii=False, indent=2)

        return midi

    def _process_step(self, midi, step):
        """Process a single step from the configuration"""
        op = step['operation']
        sel_type = step['selection_type']
        params = step['parameters']

        if op == 'tempo_change':
            self._process_tempo_change(midi, sel_type, params)
        else:
            raise ValueError(f"Unknown operation type: {op}")

    def _process_tempo_change(self, midi, sel_type, step_config):
        """Process tempo change operation with distributed segments"""
        # Get configuration parameters
        tempo_change_len_range = step_config.get('tempo_change_len_range', [2, 4])
        tempo_change_range = step_config.get('tempo_change_range', [0.1, 8.0])

        results = []

        for mod_config in step_config['modifications']:
            try:
                percentage = mod_config['percentage']
                suffix = mod_config['save_suffix']
                save_path = self._get_save_path(midi, suffix)

                # Create a deep copy of the original MIDI
                mod_midi = copy.deepcopy(midi)

                # Calculate the total length of the MIDI file
                total_length = mod_midi.get_end_time()

                # Calculate the total amount of time to modify based on percentage
                total_modify_time = total_length * (percentage / 100)

                # Generate segments to modify
                segments = self._generate_segments_to_modify(
                    total_length,
                    total_modify_time,
                    tempo_change_len_range,
                    sel_type
                )

                # Create a new MIDI file with modified tempo segments
                new_midi = self._create_modified_midi(mod_midi, segments, tempo_change_range)

                # Save the modified MIDI file
                new_midi.write(save_path)

                # Log success
                segment_info = [f"[{round(start, 2)}, {round(end, 2)}, {round(factor, 2)}]" for start, end, factor in segments]

                success_result = {
                    'percentage': percentage,
                    'segments': ", ".join(segment_info),
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

    def _generate_segments_to_modify(self, total_length, total_modify_time, tempo_change_len_range, sel_type):
        """
        Generate segments to modify with tempo changes.

        Args:
            total_length: Total length of the MIDI file in seconds
            total_modify_time: Total time to modify in seconds
            tempo_change_len_range: Range of segment lengths as percentage [min, max]
            sel_type: Selection type (random or sequential)

        Returns:
            List of tuples (start_time, end_time, tempo_factor)
        """
        segments = []
        remaining_time = total_modify_time

        # Convert percentage range to actual time range
        min_segment_percent = tempo_change_len_range[0] / 100
        max_segment_percent = tempo_change_len_range[1] / 100

        min_segment_length = total_length * min_segment_percent
        max_segment_length = total_length * max_segment_percent

        # Create a list of potential start times (dividing the piece into small chunks)
        potential_starts = []
        chunk_size = min(min_segment_length / 2, 1.0)  # Use smaller chunks for more granularity

        for start in range(int(total_length / chunk_size)):
            potential_starts.append(start * chunk_size)

        # Shuffle if random selection
        if sel_type == "random":
            random.shuffle(potential_starts)

        # Generate segments until we've covered the required modification time
        used_ranges = []  # Keep track of used time ranges

        while remaining_time > 0 and potential_starts:
            # Get next start time
            start_time = potential_starts.pop(0)

            # Check if this start time overlaps with any existing segment
            if any(start <= start_time < end for start, end in used_ranges):
                continue

            # Determine segment length (between min and max, but not exceeding remaining time)
            max_possible = min(max_segment_length, remaining_time)
            if max_possible < min_segment_length:
                segment_length = max_possible
            else:
                segment_length = random.uniform(min_segment_length, max_possible)

            # Ensure segment doesn't go beyond the end of the file
            if start_time + segment_length > total_length:
                segment_length = total_length - start_time

            # If segment is too small, skip it
            if segment_length < min_segment_length / 2:
                continue

            # Add segment to the list
            end_time = start_time + segment_length
            tempo_factor = random.uniform(0.1, 8.0)  # Random tempo change factor
            segments.append((start_time, end_time, tempo_factor))

            # Mark this range as used
            used_ranges.append((start_time, end_time))

            # Update remaining time
            remaining_time -= segment_length

        # Sort segments by start time
        segments.sort(key=lambda x: x[0])

        return segments

    def _create_modified_midi(self, original_midi, segments, tempo_change_range):
        """
        Create a new MIDI file with modified tempo segments.

        Args:
            original_midi: Original MIDI object
            segments: List of (start_time, end_time, tempo_factor) tuples
            tempo_change_range: Range of tempo change factors [min, max]

        Returns:
            New MIDI object with modified tempo
        """
        # Create a new MIDI object
        new_midi = PrettyMIDI(resolution=original_midi.resolution)
        if hasattr(original_midi, 'filename'):
            new_midi.filename = original_midi.filename

        # Copy instruments from original MIDI
        for instrument in original_midi.instruments:
            new_instrument = copy.deepcopy(instrument)
            new_instrument.notes = []  # Clear notes, we'll add them with adjusted timings
            new_midi.instruments.append(new_instrument)

        # Initialize time mapping (original time -> new time)
        time_mapping = self._create_time_mapping(original_midi.get_end_time(), segments)

        # Process each instrument
        for i, instrument in enumerate(original_midi.instruments):
            for note in instrument.notes:
                # Map note start and end times to new times
                new_start = self._map_time(note.start, time_mapping)
                new_end = self._map_time(note.end, time_mapping)

                # Create a new note with adjusted timings
                new_note = copy.deepcopy(note)
                new_note.start = new_start
                new_note.end = new_end

                # Add note to the corresponding instrument in the new MIDI
                new_midi.instruments[i].notes.append(new_note)

            # Process control changes
            for control in instrument.control_changes:
                new_control = copy.deepcopy(control)
                new_control.time = self._map_time(control.time, time_mapping)
                new_midi.instruments[i].control_changes.append(new_control)

            # Process pitch bends
            for bend in instrument.pitch_bends:
                new_bend = copy.deepcopy(bend)
                new_bend.time = self._map_time(bend.time, time_mapping)
                new_midi.instruments[i].pitch_bends.append(new_bend)

        # Update tempo changes
        self._update_tempo_changes(original_midi, new_midi, time_mapping, segments)

        return new_midi

    def _create_time_mapping(self, total_length, segments):
        """
        Create a mapping from original time to new time based on tempo changes.

        Args:
            total_length: Total length of the original MIDI in seconds
            segments: List of (start_time, end_time, tempo_factor) tuples

        Returns:
            List of (original_time, new_time) pairs defining the time mapping
        """
        mapping = [(0, 0)]  # Start with (0, 0)
        current_new_time = 0
        last_original_time = 0

        for start, end, tempo_factor in segments:
            # Add mapping point for segment start (if not already added)
            if start > last_original_time:
                # Time between last point and segment start is unchanged
                unchanged_duration = start - last_original_time
                current_new_time += unchanged_duration
                mapping.append((start, current_new_time))

            # Calculate new duration for the modified segment
            original_duration = end - start
            new_duration = original_duration / tempo_factor

            # Add mapping point for segment end
            current_new_time += new_duration
            mapping.append((end, current_new_time))

            last_original_time = end

        # Add final mapping point if needed
        if last_original_time < total_length:
            unchanged_duration = total_length - last_original_time
            current_new_time += unchanged_duration
            mapping.append((total_length, current_new_time))

        return mapping

    def _map_time(self, original_time, time_mapping):
        """
        Map an original time to a new time based on the time mapping.

        Args:
            original_time: Original time in seconds
            time_mapping: List of (original_time, new_time) pairs

        Returns:
            New time in seconds
        """
        # Find the surrounding mapping points
        for i in range(len(time_mapping) - 1):
            if time_mapping[i][0] <= original_time <= time_mapping[i+1][0]:
                # Linear interpolation between mapping points
                orig_start, new_start = time_mapping[i]
                orig_end, new_end = time_mapping[i+1]

                # Calculate the proportion of the way through the segment
                if orig_end == orig_start:  # Avoid division by zero
                    proportion = 0
                else:
                    proportion = (original_time - orig_start) / (orig_end - orig_start)

                # Calculate the new time
                return new_start + proportion * (new_end - new_start)

        # If we get here, the time is outside the mapping range
        if original_time < time_mapping[0][0]:
            return original_time  # Before first mapping point
        else:
            return time_mapping[-1][1] + (original_time - time_mapping[-1][0])  # After last mapping point

    def _update_tempo_changes(self, original_midi, new_midi, time_mapping, segments):
        """
        Update tempo changes in the new MIDI file.

        Args:
            original_midi: Original MIDI object
            new_midi: New MIDI object
            time_mapping: Time mapping from original to new times
            segments: List of (start_time, end_time, tempo_factor) tuples
        """
        # Get original tempo changes or create a default one
        original_tempo_changes = []
        if hasattr(original_midi, 'tempo_changes') and original_midi.tempo_changes:
            original_tempo_changes = original_midi.tempo_changes
        else:
            # Estimate the current tempo if possible, otherwise use a default
            try:
                current_tempo = original_midi.estimate_tempo()
            except:
                current_tempo = 120.0  # Default tempo (120 BPM)

            original_tempo_changes = [(0.0, current_tempo)]

        # Create a list to hold new tempo changes
        new_tempo_changes = []

        # Process each original tempo change
        for time, tempo in original_tempo_changes:
            # Map the time to the new timeline
            new_time = self._map_time(time, time_mapping)

            # Find if this tempo change is within a modified segment
            in_modified_segment = False
            for start, end, tempo_factor in segments:
                if start <= time < end:
                    # This tempo change is within a modified segment
                    in_modified_segment = True
                    # Adjust the tempo based on the tempo factor
                    new_tempo = tempo * tempo_factor
                    break

            if not in_modified_segment:
                new_tempo = tempo

            # Add the tempo change to the new list
            new_tempo_changes.append((new_time, new_tempo))

        # Add tempo changes at segment boundaries if needed
        for start, end, tempo_factor in segments:
            # Find the tempo at the start of the segment
            start_tempo = None
            for time, tempo in sorted(original_tempo_changes):
                if time <= start:
                    start_tempo = tempo
                else:
                    break

            if start_tempo is not None:
                # Add tempo changes at segment boundaries
                new_start_time = self._map_time(start, time_mapping)
                new_end_time = self._map_time(end, time_mapping)

                # Add tempo change at segment start
                new_tempo_changes.append((new_start_time, start_tempo * tempo_factor))

                # Add tempo change at segment end to restore original tempo progression
                # Find the tempo that would be at the end time in the original
                end_tempo = start_tempo
                for time, tempo in sorted(original_tempo_changes):
                    if time <= end:
                        end_tempo = tempo
                    else:
                        break

                new_tempo_changes.append((new_end_time, end_tempo))

        # Sort tempo changes by time and remove duplicates
        new_tempo_changes.sort(key=lambda x: x[0])

        # Remove tempo changes that are too close together (within 10ms)
        filtered_tempo_changes = []
        last_time = -0.1  # Initialize with a negative value

        for time, tempo in new_tempo_changes:
            if time - last_time >= 0.01:  # 10ms threshold
                filtered_tempo_changes.append((time, tempo))
                last_time = time

        # Set the tempo changes in the new MIDI
        new_midi.tempo_changes = filtered_tempo_changes