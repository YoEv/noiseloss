# engine/pipeline.py
import importlib
from pathlib import Path

class ProcessingPipeline:
    def __init__(self, config_base='configs'):
        self.config_base = Path(config_base)
        self.processors = {}
        self.logs = []
        self.config_path = []

    def _load_processor(self, param_name, config_path,):
        if param_name in self.processors:
            return self.processors[param_name]
        module_name = f"{param_name}_processor"
        module = importlib.import_module(f"processors.{module_name}")
        processor_class = getattr(module, f"{param_name.capitalize()}Processor")
        processor = processor_class(str(config_path))
        self.processors[param_name] = processor
        #print(module_name, processor)

        return processor


    def process_parameter(self, midi_data, param_name, output_base_dir, output_dir):
        """Process a single parameter

        Args:
            midi_data: MIDI data to process
            param_name: Parameter name (e.g., 'rhythm', 'pitch')
            output_base_dir: Base output directory
            output_dir: Directory for this specific MIDI file
        """
        # processor_path = self._get_processor_path(param_name)
        config_path = self.config_base / f"{param_name}_actions.yaml"
        Processor = self._load_processor(param_name, config_path)
        # 直接调用处理器的process（由处理器自行处理action/steps）
        Processor.process(midi_data, output_base_dir, output_dir)

        return {
            'status': 'success',
            'output_dir': output_dir,
        }

        # return processor.process(midi_data, output_base_dir, output_dir)