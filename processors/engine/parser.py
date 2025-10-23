# engine/parser.py
import os
from ruamel.yaml import YAML
from pathlib import Path

class ConfigParser:
    def __init__(self, config_dir='configs'):
        """
        初始化配置解析器
        :param config_dir: 配置文件目录路径
        """
        self.yaml = YAML(typ='safe')
        self.yaml.preserve_quotes = True
        self.config_dir = Path(config_dir).resolve()

        self._available_params = self._discover_parameters()

    def _discover_parameters(self):
        return {
            f.stem.replace('_actions', ''): f
            for f in self.config_dir.glob('*_actions.yaml')
            if f.is_file()
        }

    def get_available_parameters(self):
        """
        获取所有可用的参数类型
        :return: list[str] 参数名列表（如['rhythm', 'pitch']）
        """
        return list(self._available_params.keys())

    def parse_parameter(self, param_name):
        """
        加载指定参数的配置文件
        :param param_name: 参数名（如'rhythm'）
        :return: 解析后的配置字典
        :raises: FileNotFoundError 当配置文件不存在时
        """
        if param_name not in self._available_params:
            raise FileNotFoundError(
                f"参数'{param_name}'的配置文件不存在，可用参数: {self.get_available_parameters()}"
            )

        with open(self._available_params[param_name], 'r', encoding='utf-8') as f:
            config = self.yaml.load(f)
            self.validate_config(config)
            return config

    def validate_config(self, config):
        """
        验证配置结构完整性
        :param config: 待验证的配置字典
        :raises: ValueError 当配置不合法时
        """
        if not isinstance(config, dict):
            raise ValueError("配置文件必须是字典格式")

        # 检查必需字段 - 移除description要求
        required_sections = {'actions'}
        if not required_sections.issubset(config.keys()):
            raise ValueError(f"配置缺少必需字段，需要包含: {required_sections}")

        # 验证action步骤
        for i, action in enumerate(config['actions'], 1):
            if 'steps' not in action:
                raise ValueError(f"Action {i} 缺少'steps'字段")

            for j, step in enumerate(action['steps'], 1):
                if 'operation' not in step:
                    raise ValueError(f"Action {i} 的 Step {j} 缺少'operation'字段")

                # 修改为检查参数存在性
                if 'parameters' not in step:
                    raise ValueError(f"Action {i} 的 Step {j} 缺少'parameters'字段")

                # 检查parameters中是否有note_counts和save_suffix
                if 'note_counts' in step['parameters']:
                    for note in step['parameters']['note_counts']:
                        if 'save_suffix' not in note:
                            raise ValueError(f"Action {i} 的 Step {j} 的 note_counts 缺少'save_suffix'字段")

        return True

    def reload_configs(self):
        self._available_params = self._discover_parameters()