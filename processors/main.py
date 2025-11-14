import os
import json
import argparse
from datetime import datetime
from pretty_midi import PrettyMIDI
from engine.pipeline import ProcessingPipeline
from engine.parser import ConfigParser
class ExperimentRunner:
    def __init__(self, config_dir='configs'):
        self.pipeline = ProcessingPipeline(config_dir)
        self.parser = ConfigParser(config_dir)
        self._validate_all_configs()
        self.experiment_log = []
        self.error_log = []

    def _validate_all_configs(self):
        for param in self.parser.get_available_parameters():
            self.parser.parse_parameter(param)


    def _log_error(self, error_data):
        error_data['timestamp'] = datetime.now().isoformat()
        self.error_log.append(error_data)

    def _process_single_file(self, input_path, output_base_dir, output_dir, param=None):
        midi = PrettyMIDI(input_path)
        midi.filename = os.path.basename(input_path)
        os.makedirs(output_dir, exist_ok=True)

        orig_path = os.path.join(output_dir, f"{midi.filename}_ori.mid")
        midi.write(orig_path)

        all_results = {}
        params_to_process = [param] if param else self.parser.get_available_parameters()

        for param_name in params_to_process:
            param_results = self.pipeline.process_parameter(
                midi_data=midi,
                param_name=param_name,
                output_base_dir=output_base_dir,
                output_dir=output_dir,
            )
            all_results[param_name] = param_results

        log_entry = {
            'input_file': input_path,
            'output_dir': output_dir,
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'status': 'success'
        }
        self.experiment_log.append(log_entry)
        return all_results


    def run_experiment(self, input_path, output_base_dir, selected_params=None):
        os.makedirs(output_base_dir, exist_ok=True)
        logs_dir = os.path.join(output_base_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        if selected_params == "all" or selected_params is None:
            params_to_process = self.parser.get_available_parameters()
        else:
            params_to_process = selected_params if isinstance(selected_params, list) else [selected_params]
            available_params = self.parser.get_available_parameters()
            for param in params_to_process:
                if param not in available_params:
                    raise ValueError(f"Parameter '{param}' is not available. Available parameters: {available_params}")

        for param in params_to_process:
            param_config = self.parser.parse_parameter(param)
            action_dir = param_config.get('output_dir', f'{param}_processed')
            os.makedirs(os.path.join(output_base_dir, action_dir), exist_ok=True)

        midi_files = []
        if os.path.isdir(input_path):
            midi_files = [f for f in os.listdir(input_path) if f.endswith('.mid')]
            counter = 0
            midi_files_limited = []
            for fname in midi_files:
                midi_files_limited.append(fname)
                counter += 1
                #if counter >= 4:
                #    break
            midi_files = midi_files_limited
        else:
            midi_files = [os.path.basename(input_path)]
            input_path = os.path.dirname(input_path) or '.'

        for param in params_to_process:
            param_config = self.parser.parse_parameter(param)
            action_dir = param_config.get('output_dir', f'{param}_processed')

            for fname in midi_files:
                input_file_path = os.path.join(input_path, fname)
                base_name = os.path.splitext(fname)[0]

                file_output_dir = os.path.join(
                    output_base_dir,
                    action_dir,
                    base_name
                )

                self._process_single_file(
                    input_file_path,
                    output_base_dir,
                    file_output_dir,
                    param=param
                )

        self._save_logs(output_base_dir)
        return self._generate_report(output_base_dir)

    def _save_logs(self, output_base_dir):
        logs_dir = os.path.join(output_base_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        param_logs = {}
        for log in self.experiment_log + self.error_log:
            for param in log.get('results', {}):
                if param not in param_logs:
                    param_logs[param] = {'success': [], 'error': []}
                if log['status'] == 'success':
                    param_logs[param]['success'].append(log)
                else:
                    param_logs[param]['error'].append(log)

        for param, logs in param_logs.items():
            success_path = os.path.join(logs_dir, f'{param}_success_log.json')
            failed_path = os.path.join(logs_dir, f'{param}_error_log.json')

            with open(success_path, 'w') as f:
                json.dump(logs['success'], f, indent=2)

            with open(failed_path, 'w') as f:
                json.dump(logs['error'], f, indent=2)

    def _generate_report(self, output_base_dir):
        success_count = sum(
            1 for log in self.experiment_log
            if log.get('status') == 'success'
        )
        total_files = len(self.experiment_log)

        report = {
            'metadata': {
                'total_files_processed': total_files,
                'success_rate': f"{success_count}/{total_files}",
                'success_percentage': round(success_count/total_files*100, 2) if total_files > 0 else 0
            },
        }

        return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MIDI processing experiment')
    parser.add_argument('--config-dir', default='./configs', help='Directory containing configuration files')
    parser.add_argument('--input-path', default='./selected_midis_short', help='Input path for MIDI files')
    parser.add_argument('--output-dir', default='./pitch_mod_midi_test_1_structure', help='Base directory for output files')
    parser.add_argument('--params', nargs='+', default='all',
                        help='Parameters to process (e.g., rhythm pitch). Use "all" for all parameters')

    args = parser.parse_args()

    runner = ExperimentRunner(config_dir=args.config_dir)

    selected_params = args.params
    if isinstance(selected_params, list) and len(selected_params) == 1 and selected_params[0] == 'all':
        selected_params = 'all'

    final_report = runner.run_experiment(
        input_path=args.input_path,
        output_base_dir=args.output_dir,
        selected_params=selected_params
    )

    print(f"Experiment completed. Success rate: {final_report['metadata']['success_percentage']}%")