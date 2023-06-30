import os
import yaml
from typing import Dict


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
        self._temp_dir = os.path.join(self._base_local_dir, self._config['paths']['temp_dir'])

        self._aoi_path = {
            "shp": os.path.join(self._base_local_dir, self._config['files']['aoi_shp']),
            "gpkg": os.path.join(self._base_local_dir, self._config['files']['aoi_gpkg'])
        }
        self._inventory_path = os.path.join(self._base_local_dir, self._config['files']['inventory'])
        self._records_path = os.path.join(self._base_local_dir, self._config['files']['records'])
        self._all_images_path = os.path.join(self._base_local_dir, self._config['files']['all_images'])

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
    def temp_dir(self) -> str:
        return self._temp_dir

    @property
    def base_local_dir(self) -> str:
        return self._base_local_dir

    @property
    def available_tiles(self) -> list:
        return self._config['metadata']['tiles']

    @property
    def available_years(self) -> list:
        return self._config['metadata']['years']

    @property
    def available_products(self) -> list:
        return self._config['metadata']['products']

    @property
    def aoi_path(self) -> Dict[str, str]:
        return self._aoi_path

    @property
    def inventory_path(self) -> str:
        return self._inventory_path

    @property
    def records_path(self) -> str:
        return self._records_path

    @property
    def all_images_path(self) -> str:
        return self._all_images_path
