import os
import yaml


class Config:
    # Responsible for storing the configuration of the main application
    def __init__(self):
        # Use parent directory as root
        self._root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._config_path = os.path.join(self._root, 'config.yaml')
        self._config = self.load_config()

        # Build paths
        self._data_path = os.path.join(self._root, self._config['paths']['data'])
        self._data_input_path = os.path.join(self._data_path, self._config['paths']['input'])
        self._data_output_path = os.path.join(self._data_path, self._config['paths']['output'])
        self._data_inter_path = os.path.join(self._data_path, self._config['paths']['inter'])

        # Profiling
        self._profiling_active = self._config['profiling']['active']
        self._profiling_browser = self._config['profiling']['browser']

        # NCI parameters
        self._nci_conf = self._config['nci']

    def load_config(self) -> dict:
        with open(self._config_path) as f:
            # use safe_load instead load
            config = yaml.safe_load(f)
        return config

    @property
    def data_input_path(self) -> str:
        return self._data_input_path

    @property
    def data_output_path(self) -> str:
        return self._data_output_path

    @property
    def data_inter_path(self) -> str:
        return self._data_inter_path

    @property
    def profiling_active(self) -> bool:
        return self._profiling_active

    @property
    def profiling_browser(self) -> bool:
        return self._profiling_browser

    @property
    def nci_conf(self) -> dict:
        return self._nci_conf
