import os
import yaml


class Config:
    # Responsible for storing the configuration of the main application
    def __init__(self):
        # Use parent directory as root
        self._root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._config_path = os.path.join(self._root, 'config.yaml')
        self._config = self.load_config()

        # Profiling
        self._profiling_active = self._config['profiling']['active']
        self._profiling_browser = self._config['profiling']['browser']

        # Execution:
        self._execute_module = self._config['execute']['module']
        self._time_limit = self._config['execute']['time_limit']

        # NCI parameters
        self._nci_conf = self._config['modules']

    def load_config(self) -> dict:
        with open(self._config_path) as f:
            # use safe_load instead load
            config = yaml.safe_load(f)
        return config

    @property
    def profiling_active(self) -> bool:
        return self._profiling_active

    @property
    def profiling_browser(self) -> bool:
        return self._profiling_browser

    @property
    def nci_conf(self) -> dict:
        return self._nci_conf

    @property
    def execute(self) -> str:
        return self._execute_module

    @property
    def time_limit(self) -> int:
        return self._time_limit
