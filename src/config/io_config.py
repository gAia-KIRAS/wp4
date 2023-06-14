import os
import yaml


class IOConfig:
    """
    Responsible for storing the configuration of the IO manager.

    Attributes:
        _root: string with the root directory of the project
        _config_path: string with the path to the configuration file
        _config: dictionary with the configuration
    """
    def __init__(self):
        self._root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._config_path = os.path.join(self._root, 'io_config.yaml')
        self._config = self.load_config()

        # Build paths
        self._base_local_dir = os.path.join(self._root, self._config['paths']['base_local_dir'])

    def load_config(self) -> dict:
        """
        Loads the configuration file.

        Returns:
            config (dict): dictionary with the configuration
        """
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
    def base_server_dir(self) -> str:
        return self._config['paths']['base_server_dir']

    @property
    def base_local_dir(self) -> str:
        return self._config['paths']['base_local_dir']

    @property
    def available_tiles(self) -> list:
        return self._config['metadata']['tiles']

    @property
    def available_years(self) -> list:
        return self._config['metadata']['years']

    @property
    def available_products(self) -> list:
        return self._config['metadata']['products']
