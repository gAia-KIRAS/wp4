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

        # Build paths for AOI
        self._aoi_extensions = ['shp', 'gpkg', 'shx', 'prj', 'dbf', 'cpg']
        self._aoi_rel_path = {
            aoi_ext: self._config['files']["aoi_" + aoi_ext] for aoi_ext in self._aoi_extensions
        }
        self._aoi_path = {
            aoi_ext: os.path.join(self._base_local_dir, self._aoi_rel_path[aoi_ext]) for aoi_ext in self._aoi_extensions
        }

        # Build paths for inventory
        self._inv_extensions = ['shp', 'gpkg']
        self._inventory_rel_path = {
            aoi_ext: self._config['files']["inventory_" + aoi_ext] for aoi_ext in self._inv_extensions
        }
        self._inventory_path = {
            aoi_ext: os.path.join(self._base_local_dir, self._inventory_rel_path[aoi_ext])
            for aoi_ext in self._inv_extensions
        }

        self._inventory_poly_rel_path = {
            aoi_ext: self._config['files']["inventory_poly_" + aoi_ext] for aoi_ext in self._inv_extensions
        }
        self._inventory_poly_path = {
            aoi_ext: os.path.join(self._base_local_dir, self._inventory_poly_rel_path[aoi_ext])
            for aoi_ext in self._inv_extensions
        }

        self._records_path = os.path.join(self._base_local_dir, self._config['files']['records'])
        self._records_cd_path = os.path.join(self._base_local_dir, self._config['files']['records_cd'])
        self._results_cd_path = os.path.join(self._base_local_dir, self._config['files']['results_cd'])
        self._records_path_aux = self._records_path
        self._all_images_path = {
            image_type: os.path.join(self._base_local_dir, image_type_path)
            for image_type, image_type_path in self._config['files']['all_images'].items()
        }

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
    def server_repo_root(self) -> str:
        return self._config['paths']['server_repo_root']

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
    def aoi_rel_path(self) -> Dict[str, str]:
        return self._aoi_rel_path

    @property
    def inventory_path(self) -> Dict[str, str]:
        return self._inventory_path

    @property
    def inventory_rel_path(self) -> Dict[str, str]:
        return self._inventory_rel_path

    @property
    def inventory_poly_path(self) -> Dict[str, str]:
        return self._inventory_poly_path

    @property
    def inventory_poly_rel_path(self) -> Dict[str, str]:
        return self._inventory_poly_rel_path

    @property
    def records_path(self) -> str:
        return self._records_path

    @property
    def records_cd_path(self) -> str:
        return self._records_cd_path

    @property
    def results_cd_path(self) -> str:
        return self._results_cd_path

    @property
    def records_path_aux(self) -> str:
        return self._records_path_aux

    @property
    def all_images_path(self) -> dict:
        return self._all_images_path

    @property
    def config_path(self) -> str:
        return self._config_path

    @property
    def server_python_executable(self) -> str:
        return self._config['paths']['server_python_executable']

    def modify_paths_for_server(self):
        self._base_local_dir = f'{self._config["paths"]["base_server_dir"]}/wp4'
        self._aoi_path = {
            "shp": os.path.join(self._base_local_dir, self._aoi_rel_path['shp']),
            "gpkg": os.path.join(self._base_local_dir, self._aoi_rel_path['gpkg'])
        }
        self._inv_extensions = ['shp', 'gpkg']
        self._inventory_rel_path = {
            aoi_ext: self._config['files']["inventory_" + aoi_ext] for aoi_ext in self._inv_extensions
        }
        self._inventory_path = {
            aoi_ext: os.path.join(self._base_local_dir, self._inventory_rel_path[aoi_ext])
            for aoi_ext in self._inv_extensions
        }
        self._records_path = os.path.join(self._base_local_dir, self._config['files']['records'])
        self._records_cd_path = os.path.join(self._base_local_dir, self._config['files']['records_cd'])
        self._results_cd_path = os.path.join(self._base_local_dir, self._config['files']['results_cd'])

        self._all_images_path = {
            image_type: os.path.join(self._base_local_dir, image_type_path)
            for image_type, image_type_path in self._config['files']['all_images'].items()
        }
