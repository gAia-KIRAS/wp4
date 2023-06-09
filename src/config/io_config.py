import os
import yaml


class IOConfig:
    # Responsible for storing the configuration of the IO class
    def __init__(self):
        self._root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._config_path = os.path.join(self._root, 'io_config.yaml')
        self._config = self.load_config()

    def load_config(self) -> dict:
        with open(self._config_path) as f:
            # use safe_load instead load
            config = yaml.safe_load(f)
        return config

    @property
    def server_name(self) -> str:
        return self._config['server']['name']

    @property
    def username(self) -> str:
        return self._config['server']['username']

    @property
    def password(self) -> str:
        return self._config['server']['password']

    @property
    def base_input_dir(self) -> str:
        return self._config['paths']['base_input_dir']

